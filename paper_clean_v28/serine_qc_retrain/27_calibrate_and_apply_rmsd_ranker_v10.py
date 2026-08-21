#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibrate the frozen V10 RMSD-priority ranker and score a V9 candidate pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from rmsd_ranker_v10 import (
    FEATURE_NAMES,
    MODEL_PROTOCOL,
    PRIMARY_LABEL,
    SECONDARY_LABEL,
    cross_validation_summary,
    feature_matrix,
    historical_site_support,
    predict_logistic,
    sequence_features,
    train_logistic,
)


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DEVELOPMENT = REPO_ROOT / "v10_inputs" / "six_non3av_t05_joint_rmsd_476.csv"
EXPECTED_DEVELOPMENT_SHA256 = "d754c905e00d03c18ce0610b740c9bd6da09ee0a9e9d5d7ce953dc73d86aad05"
EXPECTED_TARGETS = ("1SFI", "3P8F", "3WNE", "3ZGC", "4K1E", "4KEL")
EXPECTED_RELEASE_TARGETS = (
    "1SFI", "3AV9", "3AVA", "3AVB", "3AVF", "3AVG", "3AVH", "3AVI",
    "3AVJ", "3AVK", "3AVM", "3AVN", "3P8F", "3WNE", "3ZGC", "4K1E",
    "4KEL",
)
MIN_RMSD_PRIORITY_POOL_PER_TARGET = 400
EXPECTED_ROWS = 476
EXPECTED_LT3 = 16
EXPECTED_LT5 = 101
RANKER_PROTOCOL = "rmsd_priority_ranker_v10_six_target_loto_v1"
KNOWN_OUTPUTS = (
    "rmsd_ranker_models_v10.json",
    "rmsd_ranker_oof_predictions_476.csv",
    "candidates_rmsd_priority_scored.csv",
    "rmsd_ranker_v10_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    observed: set[str] = set()
    for row in rows:
        for field in row:
            if field not in observed:
                observed.add(field)
                fields.append(field)
    return fields


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = union_fields(rows)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    existing = [out_dir / name for name in KNOWN_OUTPUTS if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "V10 RMSD-ranker output exists; use a new directory or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def validate_development(rows: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    targets = [str(row.get("target_name", "")).upper() for row in rows]
    audit_ids = [str(row.get("audit_row_id", "")) for row in rows]
    checks: Dict[str, bool] = {
        "development_row_count_is_frozen_476": len(rows) == EXPECTED_ROWS,
        "development_targets_are_exact_six_non3av": set(targets) == set(EXPECTED_TARGETS),
        "development_audit_row_ids_are_unique_and_nonempty": (
            all(audit_ids) and len(audit_ids) == len(set(audit_ids))
        ),
        "development_temperature_is_exactly_0.5": all(
            float(row.get("temperature", "nan")) == 0.5 for row in rows
        ),
        "development_candidates_all_contain_methyl_annotation": all(
            any(token.islower() for token in str(row.get("design_seq", "")))
            for row in rows
        ),
        "development_lt3_positive_count_is_frozen_16": sum(
            int(str(row.get(SECONDARY_LABEL, "-1")))
            for row in rows
        )
        == EXPECTED_LT3,
        "development_lt5_positive_count_is_frozen_101": sum(
            int(str(row.get(PRIMARY_LABEL, "-1"))) for row in rows
        )
        == EXPECTED_LT5,
        "development_rmsd_rows_pass_original_alignment_audits": all(
            str(row.get("LT3_Audit", "")) == "PASS"
            and str(row.get("LT5_Audit", "")) == "PASS"
            and int(str(row.get("complete_final_chain_ca_pairing_gate", "0"))) == 1
            and int(str(row.get("complete_positional_peptide_ca_coverage", "0"))) == 1
            for row in rows
        ),
    }
    # Feature recomputation is a hard data-integrity check.
    try:
        features = feature_matrix(rows)
        checks["development_features_recompute_and_are_finite"] = bool(
            features.shape == (EXPECTED_ROWS, len(FEATURE_NAMES))
            and np.all(np.isfinite(features))
        )
    except (TypeError, ValueError):
        checks["development_features_recompute_and_are_finite"] = False
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-csv", default=str(DEFAULT_DEVELOPMENT))
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-candidate-sha256", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    development_path = Path(args.development_csv).resolve()
    candidate_path = Path(args.candidate_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    prepare_output(out_dir, args.overwrite)
    development_rows = read_csv(development_path)
    candidate_rows = read_csv(candidate_path)
    development_hash = sha256_file(development_path)
    candidate_hash = sha256_file(candidate_path)

    input_checks = validate_development(development_rows)
    input_checks["development_csv_sha256_is_frozen"] = (
        development_hash == EXPECTED_DEVELOPMENT_SHA256
    )
    input_checks["candidate_pool_is_nonempty"] = bool(candidate_rows)
    input_checks["candidate_pool_sha256_matches_explicit_contract"] = (
        not args.expected_candidate_sha256
        or candidate_hash.lower() == args.expected_candidate_sha256.lower()
    )
    candidate_keys = [
        (
            str(row.get("target_name", "")).upper(),
            str(row.get("candidate_id", "")),
            str(row.get("design_natural_seq", "")).upper(),
        )
        for row in candidate_rows
    ]
    candidate_counts = {
        target: sum(key[0] == target for key in candidate_keys)
        for target in EXPECTED_RELEASE_TARGETS
    }
    input_checks["candidate_keys_are_unique_and_nonempty"] = (
        all(all(key) for key in candidate_keys)
        and len(candidate_keys) == len(set(candidate_keys))
    )
    input_checks["candidate_pool_has_exactly_the_17_release_targets"] = (
        {key[0] for key in candidate_keys} == set(EXPECTED_RELEASE_TARGETS)
    )
    input_checks["every_target_has_at_least_400_exact_base_eligible_rows"] = all(
        candidate_counts[target] >= MIN_RMSD_PRIORITY_POOL_PER_TARGET
        for target in EXPECTED_RELEASE_TARGETS
    )
    input_checks["candidate_rows_all_contain_strict_methyl_annotation"] = all(
        any(token.islower() for token in str(row.get("design_seq", "")))
        for row in candidate_rows
    )
    try:
        candidate_features = feature_matrix(candidate_rows)
        input_checks["candidate_features_recompute_and_are_finite"] = bool(
            candidate_features.shape == (len(candidate_rows), len(FEATURE_NAMES))
            and np.all(np.isfinite(candidate_features))
        )
    except (TypeError, ValueError):
        candidate_features = np.empty((0, len(FEATURE_NAMES)))
        input_checks["candidate_features_recompute_and_are_finite"] = False
    if not all(input_checks.values()):
        manifest = {
            "quality_gate": "FAIL",
            "protocol": RANKER_PROTOCOL,
            "quality_checks": input_checks,
            "release_status": "BLOCKED_RMSD_RANKER_INPUT_CONTRACT_FAILED",
            "inputs": {
                "development_csv": {"path": str(development_path), "sha256": development_hash},
                "candidate_csv": {"path": str(candidate_path), "sha256": candidate_hash},
            },
        }
        atomic_write_json(out_dir / "rmsd_ranker_v10_manifest.json", manifest)
        failed = [name for name, passed in input_checks.items() if not passed]
        raise RuntimeError("V10 RMSD-ranker input gate failed: " + ", ".join(failed))

    lt5_summary, lt5_oof, lt5_targets = cross_validation_summary(
        development_rows, PRIMARY_LABEL
    )
    lt3_summary, lt3_oof, lt3_targets = cross_validation_summary(
        development_rows, SECONDARY_LABEL
    )
    cv_checks = {
        "primary_lt5_oof_auc_at_least_0.55": lt5_summary["pooled_oof_auc"] >= 0.55,
        "primary_lt5_top_quartile_absolute_enrichment_at_least_0.02": (
            lt5_summary["absolute_enrichment"] >= 0.02
        ),
        "primary_lt5_top_quartile_relative_enrichment_at_least_1.10": (
            lt5_summary["relative_enrichment"] >= 1.10
        ),
        "primary_lt5_every_row_has_exactly_one_target_heldout_prediction": (
            len(lt5_oof) == EXPECTED_ROWS and np.all(np.isfinite(lt5_oof))
        ),
        # <3 A has only 16 positives; completeness is enforced but performance
        # is descriptive and cannot authorize or veto the ranker.
        "secondary_lt3_every_row_has_exactly_one_target_heldout_prediction": (
            len(lt3_oof) == EXPECTED_ROWS and np.all(np.isfinite(lt3_oof))
        ),
    }
    development_features = feature_matrix(development_rows)
    labels_lt5 = np.asarray([int(str(row[PRIMARY_LABEL])) for row in development_rows])
    labels_lt3 = np.asarray([int(str(row[SECONDARY_LABEL])) for row in development_rows])
    model_lt5 = train_logistic(development_features, labels_lt5)
    model_lt3 = train_logistic(development_features, labels_lt3)
    candidate_lt5 = predict_logistic(model_lt5, candidate_features)
    candidate_lt3 = predict_logistic(model_lt3, candidate_features)
    score_checks = {
        "candidate_lt5_scores_are_finite_probabilistic_priorities": bool(
            len(candidate_lt5) == len(candidate_rows)
            and np.all(np.isfinite(candidate_lt5))
            and np.all((0.0 <= candidate_lt5) & (candidate_lt5 <= 1.0))
        ),
        "candidate_lt3_scores_are_finite_probabilistic_priorities": bool(
            len(candidate_lt3) == len(candidate_rows)
            and np.all(np.isfinite(candidate_lt3))
            and np.all((0.0 <= candidate_lt3) & (candidate_lt3 <= 1.0))
        ),
    }

    rank_by_target: Dict[int, int] = {}
    target_counts: Dict[str, int] = {}
    for target in sorted({key[0] for key in candidate_keys}):
        indices = [index for index, key in enumerate(candidate_keys) if key[0] == target]
        ordered = sorted(
            indices,
            key=lambda index: (
                -float(candidate_lt5[index]),
                str(candidate_rows[index].get("candidate_id", "")),
            ),
        )
        target_counts[target] = len(indices)
        for rank, index in enumerate(ordered, start=1):
            rank_by_target[index] = rank

    scored_rows: List[Dict[str, Any]] = []
    for index, source in enumerate(candidate_rows):
        features = sequence_features(source)
        row = dict(source)
        row.update(
            {
                "rmsd_priority_protocol": RANKER_PROTOCOL,
                "rmsd_priority_primary_endpoint": "joint_global_and_cyclic_lt5A",
                "rmsd_priority_score_joint_lt5": f"{float(candidate_lt5[index]):.12g}",
                "rmsd_priority_score_joint_lt3_descriptive": f"{float(candidate_lt3[index]):.12g}",
                "rmsd_priority_rank_within_target": rank_by_target[index],
                "rmsd_priority_feature_vector": json.dumps(
                    [round(float(value), 12) for value in features],
                    separators=(",", ":"),
                ),
                "rmsd_priority_warning": (
                    "ranking_score_not_observed_structure_and_not_a_guarantee_of_lt5A"
                ),
            }
        )
        scored_rows.append(row)

    oof_rows: List[Dict[str, Any]] = []
    for index, source in enumerate(development_rows):
        oof_rows.append(
            {
                "audit_row_id": source["audit_row_id"],
                "target_name": source["target_name"],
                "design_seq": source["design_seq"],
                "design_natural_seq": source["design_natural_seq"],
                "joint_lt5": source["joint_lt5"],
                "joint_lt3": source["joint_lt3"],
                "oof_score_joint_lt5": f"{float(lt5_oof[index]):.12g}",
                "oof_score_joint_lt3_descriptive": f"{float(lt3_oof[index]):.12g}",
                "heldout_unit": source["target_name"],
            }
        )

    models_path = out_dir / "rmsd_ranker_models_v10.json"
    oof_path = out_dir / "rmsd_ranker_oof_predictions_476.csv"
    scored_path = out_dir / "candidates_rmsd_priority_scored.csv"
    models_payload = {
        "protocol": RANKER_PROTOCOL,
        "model_protocol": MODEL_PROTOCOL,
        "feature_names": list(FEATURE_NAMES),
        "primary_endpoint": PRIMARY_LABEL,
        "secondary_descriptive_endpoint": SECONDARY_LABEL,
        "joint_lt5_model": model_lt5,
        "joint_lt3_descriptive_model": model_lt3,
        "development_cv": {
            PRIMARY_LABEL: lt5_summary,
            SECONDARY_LABEL: lt3_summary,
        },
        "historical_site_support": historical_site_support(development_rows),
    }
    atomic_write_json(models_path, models_payload)
    atomic_write_csv(oof_path, oof_rows)
    atomic_write_csv(scored_path, scored_rows)

    reopen_checks = {
        "reopened_oof_has_exactly_476_rows": len(read_csv(oof_path)) == EXPECTED_ROWS,
        "reopened_scored_candidate_count_matches_input": (
            len(read_csv(scored_path)) == len(candidate_rows)
        ),
        "ranker_artifacts_are_nonempty": all(
            path.is_file() and path.stat().st_size > 0
            for path in (models_path, oof_path, scored_path)
        ),
    }
    quality_checks = {**input_checks, **cv_checks, **score_checks, **reopen_checks}
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    manifest = {
        "quality_gate": quality_gate,
        "release_status": (
            "AUTHORIZED_FOR_PRESTRUCTURE_PRIORITY_SELECTION"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_USE_RMSD_PRIORITY_SCORES"
        ),
        "protocol": RANKER_PROTOCOL,
        "scientific_scope": (
            "low_capacity_prestructure_reranking; not observed RMSD and not proof "
            "of improvement on the 17-target prospective batch; joint_lt3 is "
            "reported descriptively and never participates in candidate ordering"
        ),
        "quality_checks": quality_checks,
        "development_cv": {
            PRIMARY_LABEL: lt5_summary,
            SECONDARY_LABEL: lt3_summary,
        },
        "target_cv_diagnostics": {
            PRIMARY_LABEL: lt5_targets,
            SECONDARY_LABEL: lt3_targets,
        },
        "historical_site_support": models_payload["historical_site_support"],
        "candidate_rows": len(candidate_rows),
        "candidate_counts_by_target": target_counts,
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "dependencies": {
            "rmsd_ranker_module": {
                "path": str(SCRIPT_PATH.with_name("rmsd_ranker_v10.py")),
                "sha256": sha256_file(SCRIPT_PATH.with_name("rmsd_ranker_v10.py")),
            }
        },
        "inputs": {
            "development_csv": {"path": str(development_path), "sha256": development_hash},
            "candidate_csv": {"path": str(candidate_path), "sha256": candidate_hash},
        },
        "artifacts": {
            "models": {"path": str(models_path), "sha256": sha256_file(models_path)},
            "oof_predictions": {"path": str(oof_path), "sha256": sha256_file(oof_path)},
            "scored_candidates": {"path": str(scored_path), "sha256": sha256_file(scored_path)},
        },
    }
    manifest_path = out_dir / "rmsd_ranker_v10_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print("===== V10 RMSD PRIORITY RANKER =====", flush=True)
    print(f"Development rows: {len(development_rows)}", flush=True)
    print(f"Candidate rows: {len(candidate_rows)}", flush=True)
    print(
        "LO-target <5A AUC/top-quartile: "
        f"{lt5_summary['pooled_oof_auc']:.4f} / {lt5_summary['top_fraction_rate']:.4f}",
        flush=True,
    )
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError("V10 RMSD-ranker gate failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
