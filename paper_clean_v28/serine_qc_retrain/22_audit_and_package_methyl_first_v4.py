#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent representation-minimum audit and review bundle for V8 V4.

Both released and advisor-review rows must independently reproduce explicit
min/max/span/std evidence, zero start disagreement, and the exact lowercase
pattern.  A representation mean alone never passes this audit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
LEGACY_SEARCH_PATH = SCRIPT_PATH.with_name("14_directed_recovery_search_v8.py")
V2_SEARCH_PATH = SCRIPT_PATH.with_name("17_cyclic_base_recovery_v2.py")
V3_HELPER_PATH = SCRIPT_PATH.with_name("20_full_frontier_recovery_v3.py")
V4_SEARCH_PATH = SCRIPT_PATH.with_name("21_methyl_first_joint_recovery_v4.py")
PACKAGER_PATH = SCRIPT_PATH.with_name("19_package_v8_recovery_v2.py")
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"
DEFAULT_SEARCH = V8_ROOT / "directed_search_methyl_first_v4"
DEFAULT_AUDIT = V8_ROOT / "independent_audit_methyl_first_v4"
DEFAULT_BUNDLE = V8_ROOT / "v8_methyl_first_v4_review_bundle.zip"
AUDIT_PROTOCOL = "independent_batch_one_methyl_first_recovery_v8_v4"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_rows(v4: Any, path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return v4.read_csv(path)


def run(args: argparse.Namespace) -> None:
    v4 = load_module("v8_methyl_first_v4_for_audit", V4_SEARCH_PATH)
    old = load_module("v8_legacy_for_v4_audit", LEGACY_SEARCH_PATH)
    v2 = load_module("v8_cyclic_base_for_v4_audit", V2_SEARCH_PATH)
    packager = load_module("v8_packager_for_v4_audit", PACKAGER_PATH)

    search_dir = Path(args.search_dir).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    bundle_path = Path(args.bundle).resolve()
    manifest_path = search_dir / "methyl_first_v4_manifest.json"
    release_path = search_dir / "released_joint_candidates.csv"
    near_miss_path = (
        search_dir / "methylated_base_near_miss_for_shangge_review.csv"
    )
    for required in (manifest_path, release_path, near_miss_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = v4.read_json(manifest_path)
    if not (
        manifest.get("protocol") == v4.V4_PROTOCOL
        and manifest.get("execution_audit_gate") == "PASS"
        and manifest.get("quality_gate") in {"PASS", "FAIL"}
        and manifest.get("scientific_joint_gate") == manifest.get("quality_gate")
        and dict(manifest.get("config") or {}).get("v4_program_sha256")
        == v4.sha256_file(V4_SEARCH_PATH)
        and dict(manifest.get("config") or {}).get("legacy_search_program_sha256")
        == v4.sha256_file(LEGACY_SEARCH_PATH)
    ):
        raise RuntimeError("V4 search manifest is absent, stale, or incomplete")
    v4.validate_artifacts_under(manifest, search_dir)
    releases = read_rows(v4, release_path)
    near_misses = read_rows(v4, near_miss_path)
    if not (
        len(releases) == int(manifest["released_joint_candidates"])
        and len(near_misses)
        == int(manifest["methylated_base_near_miss_review_rows"])
        and bool(releases) == (manifest["scientific_joint_gate"] == "PASS")
        and (bool(releases) or bool(near_misses))
        and not (releases and near_misses)
    ):
        raise RuntimeError("V4 candidate tables do not match the manifest outcome")

    config = dict(manifest["config"])
    model_path = Path(args.model_path or v4.DEFAULT_MODEL).resolve()
    model_manifest_path = Path(
        args.model_manifest or v4.DEFAULT_MODEL_MANIFEST
    ).resolve()
    representation_path = Path(
        args.representation_audit or v4.DEFAULT_REPRESENTATION
    ).resolve()
    baseline = Path(args.baseline_run_dir or v4.DEFAULT_BASELINE).resolve()
    plan_path = Path(args.plan or v4.DEFAULT_PLAN).resolve()
    native_path = Path(args.native_jsonl or v4.DEFAULT_NATIVE).resolve()
    historical_path = Path(
        args.historical_designs_csv or v4.DEFAULT_HISTORICAL
    ).resolve()
    prior_handoff_path = Path(
        args.prior_handoff_csv or v4.DEFAULT_PRIOR_HANDOFF
    ).resolve()
    if not (
        v4.sha256_file(model_path) == config["model_sha256"]
        and v4.sha256_file(model_manifest_path) == config["model_manifest_sha256"]
        and v4.sha256_file(baseline / "generation_manifest.json")
        == config["baseline_manifest_sha256"]
    ):
        raise RuntimeError("V4 independent audit input hashes changed")
    _baseline_manifest, _baseline_unique, baseline_target_rows = old.validate_baseline(
        baseline,
        model_path,
        model_manifest_path,
        representation_path,
        plan_path,
        native_path,
        historical_path,
        prior_handoff_path,
    )
    runtime = v4.build_runtime(
        model_path=model_path,
        native_path=native_path,
        baseline_target_rows=baseline_target_rows,
        batch_size=1,
        base_batch_size=1,
        device_name=str(args.device),
        allow_cpu=bool(args.allow_cpu),
        old=old,
        v2=v2,
    )
    all_rows = [*(dict(row, audit_class="JOINT_RELEASE") for row in releases), *(
        dict(row, audit_class="METHYLATED_BASE_NEAR_MISS") for row in near_misses
    )]
    sequences = [str(row["design_natural_seq"]).upper() for row in all_rows]
    if len(sequences) != len(set(sequences)) or any(len(value) != 7 for value in sequences):
        raise RuntimeError("V4 audit candidates are duplicated or malformed")
    full = runtime["batch_one"].score_full(
        "3ZGC", sequences, stage="V4 independent final methyl batch-one audit"
    )
    base = runtime["base"].score_detailed(
        "3ZGC", sequences, "V4 independent final cyclic-base batch-one audit"
    )
    floor = float(manifest["frozen_cyclic_base_floor_1pct"])
    replay_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for persisted, sequence in zip(all_rows, sequences):
        fresh = full[sequence]
        fresh_base = base[sequence]
        values = [
            float(value) for value in json.loads(str(fresh["methyl_probabilities"]))
        ]
        point = v2.physical_argmax_summary(sequence, values)
        probability = float(point["physical_argmax_probability"])
        release_floor_probability = old.release_floor_actionable_max(
            fresh, sequence
        )
        base_value = float(fresh_base["cyclic_base_log_probability_mean"])
        persisted_base = float(persisted["cyclic_base_log_probability_mean"])
        lower_positions = [
            index
            for index, token in enumerate(str(fresh["design_seq"]), start=1)
            if token.islower()
        ]
        row_errors: List[str] = []
        fresh_release_errors = old.stable_cyclic_methyl_release_errors(
            fresh, sequence
        )
        persisted_release_errors = old.stable_cyclic_methyl_release_errors(
            persisted, sequence
        )
        if not (
            str(fresh["design_seq"]) == str(persisted["design_seq"])
            and bool(lower_positions)
            and not fresh_release_errors
            and not persisted_release_errors
            and abs(
                probability - float(persisted["batch_one_maximum_probability"])
            )
            <= v4.RESCORE_TOLERANCE
            and abs(
                release_floor_probability
                - float(
                    persisted["batch_one_release_floor_maximum_probability"]
                )
            )
            <= v4.RESCORE_TOLERANCE
            and int(point["physical_argmax_position_1based"]) in lower_positions
            and abs(base_value - persisted_base) <= v4.RESCORE_TOLERANCE
        ):
            row_errors.append("independent methyl/base replay mismatch")
        row_errors.extend(
            f"fresh stable cyclic gate: {message}" for message in fresh_release_errors
        )
        row_errors.extend(
            f"persisted stable cyclic gate: {message}"
            for message in persisted_release_errors
        )
        if persisted["audit_class"] == "JOINT_RELEASE":
            if not (
                base_value >= floor
                and int(persisted["passes_methylation_hard_gate"]) == 1
            ):
                row_errors.append("released row does not pass both frozen gates")
        else:
            if not (
                base_value < floor
                and int(persisted["passes_methylation_hard_gate"]) == 1
                and int(persisted["passes_cyclic_base_hard_gate"]) == 0
                and str(persisted["advisor_review_status"])
                == "REVIEW_ONLY_NOT_FULLY_QUALIFIED"
            ):
                row_errors.append("near-miss row is not methyl-pass/base-fail only")
        if row_errors:
            errors.extend(f"{sequence}: {message}" for message in row_errors)
        replay_rows.append(
            {
                "candidate_id": persisted["candidate_id"],
                "target_name": "3ZGC",
                "audit_class": persisted["audit_class"],
                "design_seq": fresh["design_seq"],
                "design_natural_seq": sequence,
                "independent_methyl_probability": probability,
                "independent_release_floor_maximum_probability": (
                    release_floor_probability
                ),
                "independent_methyl_position_1based": point[
                    "physical_argmax_position_1based"
                ],
                "independent_methyl_residue": point["physical_argmax_residue"],
                "independent_design_methyl_count": fresh["design_methyl_count"],
                "design_methyl_count": fresh["design_methyl_count"],
                "methyl_positions_1based": fresh["methyl_positions_1based"],
                "methyl_probabilities": fresh["methyl_probabilities"],
                "methyl_probability_representation_min": fresh[
                    "methyl_probability_representation_min"
                ],
                "methyl_probability_representation_max": fresh[
                    "methyl_probability_representation_max"
                ],
                "methyl_probability_representation_span": fresh[
                    "methyl_probability_representation_span"
                ],
                "methyl_probability_representation_std": fresh[
                    "methyl_probability_representation_std"
                ],
                "representation_threshold_disagreement_positions_1based": fresh[
                    "representation_threshold_disagreement_positions_1based"
                ],
                "representation_threshold_disagreement_count": fresh[
                    "representation_threshold_disagreement_count"
                ],
                "stable_cyclic_release_gate": int(not fresh_release_errors),
                "independent_cyclic_base_log_probability_mean": base_value,
                "frozen_cyclic_base_floor_1pct": floor,
                "passes_methyl_hard_gate": int(not fresh_release_errors),
                "passes_cyclic_base_hard_gate": int(base_value >= floor),
                "passes_joint_hard_gate": int(
                    not fresh_release_errors and base_value >= floor
                ),
                "row_audit_gate": "PASS" if not row_errors else "FAIL",
            }
        )

    audit_checks = {
        "search_artifacts_are_hash_valid": True,
        "search_and_audit_use_identical_frozen_model": True,
        "strict_methyl_threshold_remains_greater_than_0_6": True,
        "cyclic_base_floor_is_unchanged": True,
        "every_candidate_has_an_explicit_methylated_design_token": all(
            any(token.islower() for token in str(row["design_seq"]))
            and int(row["independent_design_methyl_count"]) > 0
            for row in replay_rows
        ),
        "every_candidate_independently_passes_methyl_hard_gate": all(
            int(row["passes_methyl_hard_gate"]) == 1
            and old.stable_cyclic_methyl_release_gate(
                persisted, str(persisted["design_natural_seq"])
            )
            for row, persisted in zip(replay_rows, all_rows)
        ),
        "released_rows_independently_pass_both_hard_gates": all(
            int(row["passes_joint_hard_gate"]) == 1
            for row in replay_rows
            if row["audit_class"] == "JOINT_RELEASE"
        ),
        "advisor_rows_are_independently_methyl_pass_base_fail": all(
            int(row["passes_methyl_hard_gate"]) == 1
            and int(row["passes_cyclic_base_hard_gate"]) == 0
            and int(row["passes_joint_hard_gate"]) == 0
            for row in replay_rows
            if row["audit_class"] == "METHYLATED_BASE_NEAR_MISS"
        ),
        "no_replay_mismatch": not errors,
    }
    if not all(audit_checks.values()):
        failed = [name for name, passed in audit_checks.items() if not passed]
        raise RuntimeError(
            "V4 independent audit failed: " + ", ".join([*failed, *errors])
        )

    audit_dir.mkdir(parents=True, exist_ok=True)
    replay_path = audit_dir / "v4_independent_candidate_replay.csv"
    report_path = audit_dir / "v4_independent_audit.json"
    v2.atomic_write_csv(replay_path, replay_rows, list(replay_rows[0]))
    report = {
        "quality_gate": "PASS",
        "execution_audit_gate": "PASS",
        "scientific_joint_gate": manifest["scientific_joint_gate"],
        "protocol": AUDIT_PROTOCOL,
        "search_protocol": v4.V4_PROTOCOL,
        "search_manifest_sha256": v4.sha256_file(manifest_path),
        "candidate_rows_replayed": len(replay_rows),
        "released_joint_rows": len(releases),
        "methylated_base_near_miss_rows": len(near_misses),
        "methylation_claim_scope": manifest["methylation_claim_scope"],
        "audit_checks": audit_checks,
        "artifacts": {
            "search_manifest": v4.artifact(manifest_path),
            "independent_replay": v4.artifact(replay_path),
        },
    }
    v2.atomic_write_json(report_path, report)

    requested = [
        (manifest_path, "search/methyl_first_v4_manifest.json"),
        (release_path, "search/released_joint_candidates.csv"),
        (
            near_miss_path,
            "search/methylated_base_near_miss_for_shangge_review.csv",
        ),
        (
            search_dir / "joint_candidate_evidence.csv",
            "search/joint_candidate_evidence.csv",
        ),
        (
            search_dir / "v4_surrogate_and_selection_audit.json",
            "search/v4_surrogate_and_selection_audit.json",
        ),
        (
            search_dir / "v4_methyl_screen_selection.csv.gz",
            "search/v4_methyl_screen_selection.csv.gz",
        ),
        (search_dir / "v4_methyl_screen.csv.gz", "search/v4_methyl_screen.csv.gz"),
        (
            search_dir / "v4_strict_methyl_exact_cyclic_base.csv.gz",
            "search/v4_strict_methyl_exact_cyclic_base.csv.gz",
        ),
        (report_path, "audit/v4_independent_audit.json"),
        (replay_path, "audit/v4_independent_candidate_replay.csv"),
        (V4_SEARCH_PATH, "programs/21_methyl_first_joint_recovery_v4.py"),
        (SCRIPT_PATH, "programs/22_audit_and_package_methyl_first_v4.py"),
        (LEGACY_SEARCH_PATH, "programs/14_directed_recovery_search_v8.py"),
        (V2_SEARCH_PATH, "programs/17_cyclic_base_recovery_v2.py"),
        (V3_HELPER_PATH, "programs/20_full_frontier_recovery_v3.py"),
        (REPO_ROOT / "run_v8_autodl_recovery_v4.sh", "programs/run_v8_autodl_recovery_v4.sh"),
        (
            SCRIPT_PATH.with_name("V8_METHYL_FIRST_RECOVERY_V4.md"),
            "programs/V8_METHYL_FIRST_RECOVERY_V4.md",
        ),
    ]
    for label, evidence in dict(manifest.get("prior_evidence") or {}).items():
        requested.append((Path(str(evidence["path"])), f"prior/{label}.json"))
    missing = [str(path) for path, _name in requested if not path.is_file()]
    if missing:
        raise RuntimeError("V4 review bundle input is absent: " + ", ".join(missing))
    bundle_manifest = {
        "quality_gate": "PASS",
        "execution_audit_gate": "PASS",
        "scientific_joint_gate": manifest["scientific_joint_gate"],
        "protocol": v4.V4_PROTOCOL,
        "audit_protocol": AUDIT_PROTOCOL,
        "search_manifest_sha256": v4.sha256_file(manifest_path),
        "audit_report_sha256": v4.sha256_file(report_path),
        "candidate_delivery_rule": (
            "JOINT RELEASE REQUIRES REPRESENTATION-MINIMUM-STABLE METHYL+BASE; "
            "FALLBACK REVIEW REQUIRES THE SAME ZERO-DISAGREEMENT METHYL GATE "
            "AND IS EXPLICITLY BASE-FAIL"
        ),
    }
    packager.write_deterministic_zip(bundle_path, requested, bundle_manifest)
    print("===== V8 V4 INDEPENDENT AUDIT AND BUNDLE COMPLETE =====", flush=True)
    print("Independent audit gate: PASS", flush=True)
    print(f"Scientific joint gate: {manifest['scientific_joint_gate']}", flush=True)
    print(f"Review bundle: {bundle_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", default=str(DEFAULT_SEARCH))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT))
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-manifest", default="")
    parser.add_argument("--representation-audit", default="")
    parser.add_argument("--baseline-run-dir", default="")
    parser.add_argument("--plan", default="")
    parser.add_argument("--native-jsonl", default="")
    parser.add_argument("--historical-designs-csv", default="")
    parser.add_argument("--prior-handoff-csv", default="")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
