#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-audit a completed V11 pool after the serialized-mean gate repair.

The 42,500 expensive model draws are immutable evidence.  This program never
rescores a model and never changes ``all_candidates.csv``.  It independently
rebuilds the unique/eligible views with the corrected bounded numerical check,
backs up the superseded views, and writes the generation manifest last.

Scientific shortfalls remain explicit in the rebuilt manifest.  A quota-only
FAIL is a valid handoff to the guided deficit sampler; it is not reported as a
technical crash and it never authorizes the final 1,700-row release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
GENERATOR_PATH = (
    REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
)
DEFAULT_ROOT = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "cyclic_native_v11_1700_monomer"
)
DEFAULT_RUN = DEFAULT_ROOT / "generation"
DEFAULT_PLAN = SCRIPT_PATH.with_name(
    "target_plan_v11_cyclic_native_rmsd_priority_1700.json"
)
DEFAULT_MODEL = DEFAULT_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
DEFAULT_AUDIT = (
    DEFAULT_ROOT / "representation_audit" / "cyclic_representation_audit.json"
)
DEFAULT_HISTORICAL = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "all_designs.csv"
)
DEFAULT_PRIOR = REPO_ROOT / "v9_inputs" / "methylated_new_candidates.csv"
DEFAULT_POSITION_POLICY = (
    REPO_ROOT / "v10_inputs" / "evidence_aware_position_concentration_policy.json"
)

REAUDIT_PROTOCOL = "v11_serialized_probability_gate_reaudit_v1"
EXPECTED_PROTOCOL_PREFIX = "temperature_0.5_cyclic_native_relative_positions_v11_"
EXPECTED_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_cyclic_native_relative_positions_v11"
)
KNOWN_PRE_REAUDIT_FAILURES = {
    "every_target_meets_pre_structure_candidate_quota",
    "no_single_position_exceeds_80_percent_of_sites",
    "no_single_residue_exceeds_80_percent_of_sites",
    "no_target_has_single_residue_above_80_percent_when_n_ge_30",
    "no_target_has_unsupported_single_position_above_80_percent_when_n_ge_30",
}
BACKUP_NAMES = (
    "generation_manifest.json",
    "generation_summary_by_target.csv",
    "unique_candidates.csv",
    "methylated_new_candidates.csv",
)


def load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "v11_gate_reaudit_generator", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generator: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def artifact_record(path: Path) -> Dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def manifest_artifacts_match(manifest: Mapping[str, Any]) -> bool:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        return False
    for record in artifacts.values():
        if not isinstance(record, Mapping):
            return False
        path = Path(str(record.get("path", "")))
        if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
            return False
    return True


def source_contract_manifest(run_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    current_path = run_dir / "generation_manifest.json"
    current = read_json(current_path)
    gate = current.get("serialized_gate_reaudit")
    if (
        isinstance(gate, Mapping)
        and gate.get("protocol") == REAUDIT_PROTOCOL
        and manifest_artifacts_match(current)
        and str(dict(current.get("program") or {}).get("sha256", ""))
        == sha256_file(GENERATOR_PATH)
    ):
        print("V11 serialized gate re-audit: already current and hash-valid", flush=True)
        raise SystemExit(0)

    backup_path = run_dir / "pre_serialized_gate_reaudit_backup" / current_path.name
    if backup_path.is_file():
        return backup_path, read_json(backup_path)
    return current_path, current


def validate_source(
    *,
    manifest: Mapping[str, Any],
    source_manifest_path: Path,
    run_dir: Path,
    all_path: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    validated: Mapping[str, Any],
    plan_path: Path,
    model_path: Path,
    audit_path: Path,
) -> Dict[str, Any]:
    failures: List[str] = []
    model_hash = sha256_file(model_path)
    audit_hash = sha256_file(audit_path)
    source_checks = dict(manifest.get("quality_checks") or {})
    source_false = sorted(name for name, passed in source_checks.items() if not passed)
    artifacts = dict(manifest.get("artifacts") or {})
    all_record = artifacts.get("all_candidates", {})
    pinned_audit = dict(manifest.get("cyclic_representation_heldout_audit") or {})
    if not str(plan.get("protocol", "")).startswith(EXPECTED_PROTOCOL_PREFIX):
        failures.append("current plan is not V11 cyclic-native")
    if manifest.get("protocol") != plan.get("protocol"):
        failures.append("source manifest/plan protocol mismatch")
    if float(manifest.get("temperature", -1.0)) != 0.5:
        failures.append("source temperature is not 0.5")
    if float(manifest.get("methyl_threshold", -1.0)) != 0.6:
        failures.append("source threshold is not 0.6")
    if manifest.get("model_expert_qc_protocol") != EXPECTED_EXPERT_PROTOCOL:
        failures.append("source expert protocol mismatch")
    if manifest.get("model_sha256") != model_hash:
        failures.append("source/model SHA-256 mismatch")
    if not (
        isinstance(all_record, Mapping)
        and Path(str(all_record.get("path", ""))).resolve() == all_path.resolve()
        and str(all_record.get("sha256", "")) == sha256_file(all_path)
    ):
        failures.append("all_candidates is not the exact source artifact")
    if int(manifest.get("raw_candidates_generated", -1)) != len(raw_rows):
        failures.append("source raw-row count mismatch")
    if len(raw_rows) != int(validated["expected_raw_candidates"]):
        failures.append("source does not contain the complete frozen draw budget")
    if set(source_false) - KNOWN_PRE_REAUDIT_FAILURES:
        failures.append("source has an unrelated scientific failure")
    if not (
        pinned_audit.get("quality_gate") == "PASS"
        and pinned_audit.get("model_sha256") == model_hash
        and pinned_audit.get("plan_sha256") == sha256_file(plan_path)
        and pinned_audit.get("sha256") == audit_hash
    ):
        failures.append("source does not pin the exact passed V11 audit")

    target_counts = Counter(str(row.get("target_name", "")).upper() for row in raw_rows)
    expected_counts = {
        str(item["target_name"]).upper(): int(item["sequences_per_seed"])
        * len(validated["seeds"])
        for item in validated["targets"]
    }
    if dict(target_counts) != expected_counts:
        failures.append("source target/draw accounting mismatch")
    candidate_ids = [str(row.get("candidate_id", "")) for row in raw_rows]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        failures.append("source candidate IDs are empty or duplicated")
    invalid_rows = []
    for row in raw_rows:
        try:
            valid = (
                int(row.get("length_match", 0)) == 1
                and int(row.get("valid_token_gate", 0)) == 1
                and str(row.get("annotation_mode", ""))
                == "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
                and int(row.get("annotation_representation_ensemble_size", -1))
                == int(row.get("design_length", -2))
                and float(row.get("methyl_threshold", "nan")) == 0.6
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalid_rows.append(str(row.get("candidate_id", "<missing>")))
            if len(invalid_rows) == 10:
                break
    if invalid_rows:
        failures.append("source contains malformed rows: " + ", ".join(invalid_rows))
    if failures:
        raise RuntimeError(
            "V11 serialized gate re-audit refused its source:\n- "
            + "\n- ".join(failures)
        )
    return {
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_false_checks": source_false,
        "source_all_candidates_sha256": sha256_file(all_path),
        "source_rows": len(raw_rows),
        "source_target_counts": dict(sorted(target_counts.items())),
    }


def backup_superseded_views(run_dir: Path) -> Path:
    backup = run_dir / "pre_serialized_gate_reaudit_backup"
    backup.mkdir(parents=True, exist_ok=True)
    for name in BACKUP_NAMES:
        source = run_dir / name
        destination = backup / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    return backup


def run(args: argparse.Namespace) -> None:
    generator = load_generator()
    run_dir = Path(args.run_dir).resolve()
    plan_path = Path(args.plan).resolve()
    model_path = Path(args.model).resolve()
    audit_path = Path(args.representation_audit).resolve()
    historical_path = Path(args.historical_csv).resolve()
    prior_path = Path(args.prior_csv).resolve()
    position_path = Path(args.position_concentration_policy).resolve()
    paths = {
        "all": run_dir / "all_candidates.csv",
        "unique": run_dir / "unique_candidates.csv",
        "eligible": run_dir / "methylated_new_candidates.csv",
        "summary": run_dir / "generation_summary_by_target.csv",
        "target_manifest": run_dir / "target_manifest.csv",
        "manifest": run_dir / "generation_manifest.json",
    }
    for required in (
        plan_path,
        model_path,
        audit_path,
        historical_path,
        prior_path,
        position_path,
        *paths.values(),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_manifest_path, source_manifest = source_contract_manifest(run_dir)
    plan = generator.read_json(plan_path)
    validated = generator.validate_plan(plan)
    raw_rows = generator.read_csv(paths["all"])
    source_validation = validate_source(
        manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        run_dir=run_dir,
        all_path=paths["all"],
        raw_rows=raw_rows,
        plan=plan,
        validated=validated,
        plan_path=plan_path,
        model_path=model_path,
        audit_path=audit_path,
    )
    old_exact, old_natural = generator.old_design_keys(historical_path)
    _prior_rows, prior_exact, prior_natural = generator.validate_prior_handoff(
        prior_path
    )
    unique_rows = generator.aggregate_unique_candidates(
        raw_rows, old_exact, old_natural, prior_exact, prior_natural
    )
    eligible_rows = [
        row for row in unique_rows
        if int(row["eligible_for_new_permeability_screen"])
    ]
    for row in unique_rows:
        row["permeability_id"] = ""

    position_policy = generator.read_json(position_path)
    annotation_audit = generator.audit_annotation_stability(
        raw_rows, eligible_rows, position_policy
    )
    plan_by_target = {
        str(item["target_name"]).upper(): item for item in validated["targets"]
    }
    raw_counts = Counter(str(row["target_name"]).upper() for row in raw_rows)
    unique_counts = Counter(str(row["target_name"]).upper() for row in unique_rows)
    eligible_counts = Counter(str(row["target_name"]).upper() for row in eligible_rows)
    summary_rows: List[Dict[str, Any]] = []
    for target in validated["target_names"]:
        target_unique = [
            row for row in unique_rows if str(row["target_name"]).upper() == target
        ]
        pool_quota = int(plan_by_target[target]["structure_quota"])
        final_quota = int(plan["final_release_quota_per_target"])
        summary_rows.append(
            {
                "target_name": target,
                "current_problem": plan_by_target[target]["current_problem"],
                "raw_generated": raw_counts[target],
                "unique_generated": unique_counts[target],
                "unique_methylated": sum(
                    int(row["passes_methylation_hard_gate"])
                    for row in target_unique
                ),
                "historical_4115_hits": sum(
                    int(row["seen_in_historical_4115"]) for row in target_unique
                ),
                "prior_1333_hits": sum(
                    int(row["seen_in_prior_1333"]) for row in target_unique
                ),
                "new_methylated_for_permeability": eligible_counts[target],
                "planned_preselection_candidate_quota": pool_quota,
                "planned_final_structure_handoff_quota": final_quota,
                "enough_candidates_before_permeability": int(
                    eligible_counts[target] >= pool_quota
                ),
            }
        )
    targets_below = [
        target for target in validated["target_names"]
        if eligible_counts[target] < int(plan_by_target[target]["structure_quota"])
    ]
    quality_checks = {
        **dict(annotation_audit["quality_checks"]),
        "every_target_meets_pre_structure_candidate_quota": not targets_below,
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"

    backup = backup_superseded_views(run_dir)
    immutable_source_manifest = backup / "generation_manifest.json"
    if (
        not immutable_source_manifest.is_file()
        or sha256_file(immutable_source_manifest)
        != source_validation["source_manifest_sha256"]
    ):
        raise RuntimeError("V11 gate re-audit source-manifest backup mismatch")
    source_validation["source_manifest_path"] = str(immutable_source_manifest)
    unique_fields = union_fields(unique_rows)
    generator.atomic_write_csv(paths["unique"], unique_rows, unique_fields)
    generator.atomic_write_csv(paths["eligible"], eligible_rows, unique_fields)
    generator.atomic_write_csv(
        paths["summary"], summary_rows, list(summary_rows[0])
    )

    manifest = dict(source_manifest)
    manifest.update(
        {
            "quality_gate": quality_gate,
            "quality_checks": quality_checks,
            "raw_candidates_expected": int(validated["expected_raw_candidates"]),
            "raw_candidates_generated": len(raw_rows),
            "unique_candidates": len(unique_rows),
            "new_methylated_candidates_for_permeability": len(eligible_rows),
            "all_candidates_csv_sha256": sha256_file(paths["all"]),
            "unique_candidates_csv_sha256": sha256_file(paths["unique"]),
            "methylated_new_candidates_csv_sha256": sha256_file(paths["eligible"]),
            "targets_below_pre_permeability_quota": targets_below,
            "planned_preselection_candidate_pool": int(
                validated["planned_preselection_candidate_pool"]
            ),
            "planned_structure_handoff": int(validated["planned_structure_handoff"]),
            "annotation_stability_audit": annotation_audit,
            "serialized_gate_reaudit": {
                "protocol": REAUDIT_PROTOCOL,
                "program": artifact_record(SCRIPT_PATH),
                "generator": artifact_record(GENERATOR_PATH),
                "immutable_all_candidates": artifact_record(paths["all"]),
                "source_validation": source_validation,
                "backup_dir": str(backup),
                "serialized_probability_recompute_atol": (
                    generator.SERIALIZED_PROBABILITY_RECOMPUTE_ATOL
                ),
                "release_threshold_unchanged": 0.6,
                "release_rule_unchanged": (
                    "round(representation_min,8)>0.6_and_zero_start_disagreement"
                ),
                "superseded_eligible_rows": int(
                    source_manifest.get(
                        "new_methylated_candidates_for_permeability", -1
                    )
                ),
                "corrected_eligible_rows": len(eligible_rows),
                "all_candidates_rows_rewritten": 0,
            },
            "program": artifact_record(GENERATOR_PATH),
        }
    )
    inputs = dict(manifest.get("inputs") or {})
    inputs.update(
        {
            "plan": artifact_record(plan_path),
            "model": artifact_record(model_path),
            "representation_audit": artifact_record(audit_path),
            "historical_csv": artifact_record(historical_path),
            "prior_csv": artifact_record(prior_path),
            "position_concentration_policy": artifact_record(position_path),
        }
    )
    manifest["inputs"] = inputs
    manifest["artifacts"] = {
        "all_candidates": artifact_record(paths["all"]),
        "unique_candidates": artifact_record(paths["unique"]),
        "methylated_new_candidates": artifact_record(paths["eligible"]),
        "generation_summary_by_target": artifact_record(paths["summary"]),
        "target_manifest": artifact_record(paths["target_manifest"]),
    }
    generator.atomic_write_json(paths["manifest"], manifest)

    # Reopen the exact bytes written before reporting completion.
    reopened = read_json(paths["manifest"])
    if not manifest_artifacts_match(reopened):
        raise RuntimeError("V11 gate re-audit artifact reopen check failed")
    if sha256_file(paths["all"]) != source_validation["source_all_candidates_sha256"]:
        raise RuntimeError("V11 gate re-audit unexpectedly changed all_candidates.csv")
    print("===== V11 SERIALIZED GATE RE-AUDIT COMPLETE =====", flush=True)
    print(f"Immutable raw rows reused: {len(raw_rows):,}", flush=True)
    print(f"Corrected robust eligible rows: {len(eligible_rows):,}", flush=True)
    print(f"Targets below internal preselection pool: {targets_below}", flush=True)
    print(f"Final structure handoff remains: {validated['planned_structure_handoff']}", flush=True)
    print(f"Scientific state: {quality_gate}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--representation-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--historical-csv", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--prior-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument(
        "--position-concentration-policy", default=str(DEFAULT_POSITION_POLICY)
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
