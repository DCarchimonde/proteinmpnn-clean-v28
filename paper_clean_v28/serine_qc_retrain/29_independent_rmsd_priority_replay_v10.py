#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independently replay V10 RMSD-priority scores and package the 17 x 100 batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from rmsd_ranker_v10 import FEATURE_NAMES, predict_logistic, sequence_features


RANKER_PROTOCOL = "rmsd_priority_ranker_v10_six_target_loto_v1"
SELECTOR_PROTOCOL = "independent_v9_cyclic_stability_17x100_release_audit_v1"
V9_REPLAY_PROTOCOL = "batch1_full_model_replay_v9_17x100_release_gate_v1"
V10_PROTOCOL = "independent_rmsd_priority_replay_v10_17x100_v1"
SELECTION_OVERLAY = "rmsd_priority_first_with_evidence_aware_position_gate_v10"
EXPECTED_TARGETS = (
    "1SFI", "3AV9", "3AVA", "3AVB", "3AVF", "3AVG", "3AVH", "3AVI",
    "3AVJ", "3AVK", "3AVM", "3AVN", "3P8F", "3WNE", "3ZGC", "4K1E",
    "4KEL",
)
EXPECTED_ROWS = 1700
QUOTA = 100
SCORE_ATOL = 1.0e-10
FINAL_DETAIL = "1700_详细审计.csv"
FINAL_CONCISE = "1700_给尚哥_极简.csv"
FINAL_FASTA = "1700_给尚哥_结构输入.fasta"
RISK_REPLAY = "v10_rmsd_priority_replay.csv"
FINAL_MANIFEST = "v10_1700_final_manifest.json"
KNOWN_OUTPUTS = (FINAL_DETAIL, FINAL_CONCISE, FINAL_FASTA, RISK_REPLAY, FINAL_MANIFEST)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                result.append(field)
    return result


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=union_fields(rows), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def nested_hash(payload: Mapping[str, Any], section: str, label: str) -> str:
    section_value = payload.get(section, {})
    if not isinstance(section_value, Mapping):
        return ""
    record = section_value.get(label, {})
    if not isinstance(record, Mapping):
        return ""
    return str(record.get("sha256", ""))


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    existing = [out_dir / name for name in KNOWN_OUTPUTS if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "V10 final output exists; use a new directory or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-final-dir", required=True)
    parser.add_argument("--selector-manifest", required=True)
    parser.add_argument("--ranker-manifest", required=True)
    parser.add_argument("--ranker-models", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.v9_final_dir).resolve()
    selector_path = Path(args.selector_manifest).resolve()
    ranker_manifest_path = Path(args.ranker_manifest).resolve()
    models_path = Path(args.ranker_models).resolve()
    out_dir = Path(args.out_dir).resolve()
    prepare_output(out_dir, args.overwrite)
    if out_dir == source_dir:
        raise ValueError("V10 final output must be separate from V9 replay output")

    detail_path = source_dir / FINAL_DETAIL
    concise_path = source_dir / FINAL_CONCISE
    fasta_path = source_dir / FINAL_FASTA
    v9_manifest_path = source_dir / "v9_1700_independent_replay_manifest.json"
    selector = read_json(selector_path)
    ranker = read_json(ranker_manifest_path)
    models = read_json(models_path)
    v9_replay = read_json(v9_manifest_path)
    detailed = read_csv(detail_path)
    counts = Counter(str(row.get("target_name", "")).upper() for row in detailed)

    ranker_checks = ranker.get("quality_checks", {})
    selector_checks = selector.get("quality_checks", {})
    v9_checks = v9_replay.get("quality_checks", {})
    checks: Dict[str, bool] = {
        "ranker_manifest_is_hash_bound_authorized_pass": (
            ranker.get("quality_gate") == "PASS"
            and ranker.get("protocol") == RANKER_PROTOCOL
            and isinstance(ranker_checks, Mapping)
            and bool(ranker_checks)
            and all(value is True for value in ranker_checks.values())
            and nested_hash(ranker, "artifacts", "models") == sha256_file(models_path)
        ),
        "selector_is_v10_overlay_authorized_pass": (
            selector.get("quality_gate") == "PASS"
            and selector.get("protocol") == SELECTOR_PROTOCOL
            and selector.get("selection_overlay") == SELECTION_OVERLAY
            and isinstance(selector_checks, Mapping)
            and bool(selector_checks)
            and all(value is True for value in selector_checks.values())
        ),
        "selector_is_bound_to_the_exact_ranker_manifest_and_scored_pool": (
            nested_hash(selector, "inputs", "rmsd_priority_manifest")
            == sha256_file(ranker_manifest_path)
            and nested_hash(selector, "inputs", "rmsd_priority_csv")
            == nested_hash(ranker, "artifacts", "scored_candidates")
            and bool(nested_hash(ranker, "artifacts", "scored_candidates"))
        ),
        "selector_release_views_are_the_exact_batch1_replay_sources": (
            nested_hash(selector, "release_artifacts", "detailed_audit")
            == sha256_file(detail_path)
            and nested_hash(selector, "release_artifacts", "shangge_concise")
            == sha256_file(concise_path)
            and nested_hash(selector, "release_artifacts", "shangge_fasta")
            == sha256_file(fasta_path)
        ),
        "v9_batch1_model_replay_is_authorized_pass": (
            v9_replay.get("quality_gate") == "PASS"
            and v9_replay.get("protocol") == V9_REPLAY_PROTOCOL
            and isinstance(v9_checks, Mapping)
            and bool(v9_checks)
            and all(value is True for value in v9_checks.values())
        ),
        "v9_batch1_replay_is_bound_to_the_exact_selector_manifest": (
            nested_hash(v9_replay, "inputs", "selector_manifest")
            == sha256_file(selector_path)
        ),
        "v9_replay_detail_hash_matches_named_artifact": (
            nested_hash(v9_replay, "release_artifacts", "detailed")
            == sha256_file(detail_path)
        ),
        "v9_replay_concise_hash_matches_named_artifact": (
            nested_hash(v9_replay, "release_artifacts", "concise")
            == sha256_file(concise_path)
        ),
        "v9_replay_fasta_hash_matches_named_artifact": (
            nested_hash(v9_replay, "release_artifacts", "fasta")
            == sha256_file(fasta_path)
        ),
        "final_detailed_has_exact_17_by_100": (
            len(detailed) == EXPECTED_ROWS
            and set(counts) == set(EXPECTED_TARGETS)
            and all(counts[target] == QUOTA for target in EXPECTED_TARGETS)
        ),
        "every_final_sequence_contains_predicted_methylation": all(
            any(token.islower() for token in str(row.get("design_seq", "")))
            for row in detailed
        ),
        "final_ids_and_candidate_ids_are_unique": (
            len({row.get("final_release_id", "") for row in detailed}) == EXPECTED_ROWS
            and len({row.get("candidate_id", "") for row in detailed}) == EXPECTED_ROWS
        ),
    }

    model_lt5 = models.get("joint_lt5_model", {})
    model_lt3 = models.get("joint_lt3_descriptive_model", {})
    replay_rows: List[Dict[str, Any]] = []
    maximum_delta = 0.0
    failures = 0
    for source in detailed:
        errors: List[str] = []
        try:
            features = sequence_features(source)
            replay_lt5 = float(predict_logistic(model_lt5, features)[0])
            replay_lt3 = float(predict_logistic(model_lt3, features)[0])
            persisted_lt5 = float(source["rmsd_priority_score_joint_lt5"])
            persisted_lt3 = float(
                source["rmsd_priority_score_joint_lt3_descriptive"]
            )
            feature_persisted = np.asarray(
                json.loads(source["rmsd_priority_feature_vector"]), dtype=float
            )
            if feature_persisted.shape != (len(FEATURE_NAMES),) or not np.allclose(
                feature_persisted, features, rtol=0.0, atol=1e-11
            ):
                errors.append("feature_vector_mismatch")
            delta_lt5 = abs(replay_lt5 - persisted_lt5)
            delta_lt3 = abs(replay_lt3 - persisted_lt3)
            maximum_delta = max(maximum_delta, delta_lt5, delta_lt3)
            if delta_lt5 > SCORE_ATOL:
                errors.append("joint_lt5_priority_score_mismatch")
            if delta_lt3 > SCORE_ATOL:
                errors.append("joint_lt3_priority_score_mismatch")
            if source.get("rmsd_priority_protocol") != RANKER_PROTOCOL:
                errors.append("ranker_protocol_mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            replay_lt5 = replay_lt3 = math.nan
            delta_lt5 = delta_lt3 = math.inf
            errors.append(f"replay_exception:{type(exc).__name__}")
        if errors:
            failures += 1
        replay_rows.append(
            {
                "final_release_id": source.get("final_release_id", ""),
                "candidate_id": source.get("candidate_id", ""),
                "target_name": source.get("target_name", ""),
                "design_seq": source.get("design_seq", ""),
                "persisted_joint_lt5_priority_score": source.get(
                    "rmsd_priority_score_joint_lt5", ""
                ),
                "replayed_joint_lt5_priority_score": replay_lt5,
                "joint_lt5_priority_score_abs_delta": delta_lt5,
                "persisted_joint_lt3_priority_score": source.get(
                    "rmsd_priority_score_joint_lt3_descriptive", ""
                ),
                "replayed_joint_lt3_priority_score": replay_lt3,
                "joint_lt3_priority_score_abs_delta": delta_lt3,
                "row_rmsd_priority_replay_status": "PASS" if not errors else "FAIL",
                "problems": ";".join(errors),
            }
        )
    checks["all_1700_rmsd_priority_scores_recompute_independently"] = failures == 0
    checks["maximum_rmsd_priority_score_delta_within_tolerance"] = (
        maximum_delta <= SCORE_ATOL
    )
    quality_gate = "PASS" if all(checks.values()) else "FAIL"

    if quality_gate == "PASS":
        atomic_copy(detail_path, out_dir / FINAL_DETAIL)
        atomic_copy(concise_path, out_dir / FINAL_CONCISE)
        atomic_copy(fasta_path, out_dir / FINAL_FASTA)
        atomic_write_csv(out_dir / RISK_REPLAY, replay_rows)
        checks["final_copies_are_byte_identical_to_batch1_replay_views"] = all(
            sha256_file(source) == sha256_file(out_dir / source.name)
            for source in (detail_path, concise_path, fasta_path)
        )
        checks["reopened_rmsd_priority_replay_has_1700_pass_rows"] = (
            len(read_csv(out_dir / RISK_REPLAY)) == EXPECTED_ROWS
            and all(
                row.get("row_rmsd_priority_replay_status") == "PASS"
                for row in read_csv(out_dir / RISK_REPLAY)
            )
        )
        quality_gate = "PASS" if all(checks.values()) else "FAIL"
    if quality_gate != "PASS":
        for name in (FINAL_DETAIL, FINAL_CONCISE, FINAL_FASTA, RISK_REPLAY):
            path = out_dir / name
            if path.exists():
                path.unlink()

    report = {
        "quality_gate": quality_gate,
        "release_status": (
            "READY_FOR_USER_REVIEW_THEN_STRUCTURE_PREDICTION"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_SEND_TO_SHANGGE"
        ),
        "protocol": V10_PROTOCOL,
        "scientific_scope": (
            "All 1700 rows passed methyl/base model replay and RMSD-priority "
            "score replay. RMSD improvement is not proven until structures are returned."
        ),
        "quality_checks": checks,
        "rows": len(detailed),
        "target_counts": dict(sorted(counts.items())),
        "rmsd_priority_replay_failures": failures,
        "maximum_rmsd_priority_score_absolute_delta": maximum_delta,
        "program": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "dependencies": {
            "rmsd_ranker_module": {
                "path": str(Path(__file__).with_name("rmsd_ranker_v10.py").resolve()),
                "sha256": sha256_file(
                    Path(__file__).with_name("rmsd_ranker_v10.py").resolve()
                ),
            }
        },
        "inputs": {
            "v9_replay_manifest": {
                "path": str(v9_manifest_path),
                "sha256": sha256_file(v9_manifest_path),
            },
            "selector_manifest": {
                "path": str(selector_path),
                "sha256": sha256_file(selector_path),
            },
            "ranker_manifest": {
                "path": str(ranker_manifest_path),
                "sha256": sha256_file(ranker_manifest_path),
            },
            "ranker_models": {"path": str(models_path), "sha256": sha256_file(models_path)},
        },
        "release_artifacts": (
            {
                "detailed": {
                    "path": str(out_dir / FINAL_DETAIL),
                    "sha256": sha256_file(out_dir / FINAL_DETAIL),
                },
                "concise": {
                    "path": str(out_dir / FINAL_CONCISE),
                    "sha256": sha256_file(out_dir / FINAL_CONCISE),
                },
                "fasta": {
                    "path": str(out_dir / FINAL_FASTA),
                    "sha256": sha256_file(out_dir / FINAL_FASTA),
                },
                "rmsd_priority_replay": {
                    "path": str(out_dir / RISK_REPLAY),
                    "sha256": sha256_file(out_dir / RISK_REPLAY),
                },
            }
            if quality_gate == "PASS"
            else {}
        ),
    }
    atomic_write_json(out_dir / FINAL_MANIFEST, report)
    print("===== V10 INDEPENDENT RMSD PRIORITY REPLAY =====", flush=True)
    print(f"Rows: {len(detailed)}; failures: {failures}", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("V10 final handoff is blocked: " + ", ".join(failed))


if __name__ == "__main__":
    main()
