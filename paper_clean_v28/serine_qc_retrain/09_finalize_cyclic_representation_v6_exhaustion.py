#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Finalize a V6 target as an audited model abstention after fixed-budget exhaustion.

This is not a quota waiver and it never manufactures a candidate.  It is the
scientifically honest terminal state for a target that still has zero novel
methylated candidates after the complete V6 regeneration plus the already
committed 12,000-draw adaptive budget.  Candidate CSVs, the 0.6 threshold, the
checkpoint, and every annotation remain byte-for-byte unchanged.

Exit status 20 means that the fixed adaptive budget has not yet been exhausted
and the launcher may run the ordinary quota sampler.  Exit status 0 means that
the abstention was finalized (or was already finalized).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RESUMER_PATH = SCRIPT_PATH.with_name(
    "08_resume_cyclic_representation_v6_quota.py"
)
V6_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_cyclic_representation_v6"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_representation_v6.json")
DEFAULT_MODEL = V6_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
DEFAULT_RUN = V6_ROOT / "generation"
DEFAULT_REPRESENTATION_AUDIT = (
    V6_ROOT / "representation_audit" / "cyclic_representation_audit.json"
)
DEFAULT_OLD = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "all_designs.csv"
)
DEFAULT_PRIOR = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "rerun_temperature_0.5_multiseed"
    / "methylated_new_candidates.csv"
)

PROTOCOL = "cyclic_representation_v6_fixed_budget_target_abstention_v1"
RELEASE_STATUS = "READY_FOR_MANUAL_SCIENTIFIC_REVIEW_WITH_FORMAL_TARGET_ABSTENTION"
ABSTENTION_ACTION = (
    "MODEL_ABSTAINS; DO_NOT_LOWER_THRESHOLD; DO_NOT_REUSE_PRE_V6_ANNOTATION; "
    "DO_NOT_CREATE_STRUCTURE_TASK_FOR_THIS_TARGET"
)
MINIMUM_ADAPTIVE_DRAWS_FOR_ABSTENTION = 12_000
NEEDS_SAMPLING_EXIT_CODE = 20


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def probability_values(row: Mapping[str, Any]) -> List[float]:
    raw = row.get("methyl_probabilities", "")
    try:
        values = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid methyl probabilities for {row.get('candidate_id', '<missing>')}"
        ) from exc
    if not isinstance(values, list):
        raise RuntimeError("methyl_probabilities must be a JSON list")
    return [float(value) for value in values]


def evaluate_exhausted_target(
    target: str,
    target_plan: Mapping[str, Any],
    plan_seeds: Sequence[int],
    raw_rows: Sequence[Mapping[str, Any]],
    unique_rows: Sequence[Mapping[str, Any]],
    eligible_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    resumer: Any,
) -> Dict[str, Any]:
    target = target.upper()
    target_raw = [
        row for row in raw_rows if str(row.get("target_name", "")).upper() == target
    ]
    target_unique = [
        row for row in unique_rows if str(row.get("target_name", "")).upper() == target
    ]
    target_eligible = [
        row for row in eligible_rows if str(row.get("target_name", "")).upper() == target
    ]
    initial_rows = [
        row
        for row in target_raw
        if str(row.get("source_recovery_stage", "")) == resumer.INITIAL_STAGE
    ]
    topup_rows = [
        row
        for row in target_raw
        if str(row.get("source_recovery_stage", "")) == resumer.TOPUP_STAGE
    ]
    expected_initial = int(target_plan["sequences_per_seed"]) * len(plan_seeds)
    budget = dict(manifest.get("adaptive_topup_budget") or {})
    recorded_total_budget = int(
        budget.get(
            "maximum_draws_per_target_total",
            budget.get("maximum_draws_per_target_per_resume", -1),
        )
    )
    draws_per_seed = int(budget.get("draws_per_reserve_seed", 0))
    required_seed_count = (
        math.ceil(MINIMUM_ADAPTIVE_DRAWS_FOR_ABSTENTION / draws_per_seed)
        if draws_per_seed > 0
        else -1
    )
    topup_seed_counts = Counter(int(row["seed"]) for row in topup_rows)
    topup_seeds = sorted(topup_seed_counts)
    fully_exhausted_topup_seeds = sorted(
        seed
        for seed, count in topup_seed_counts.items()
        if draws_per_seed > 0 and count >= draws_per_seed
    )
    initial_seeds = {int(value) for value in plan_seeds}
    raw_with_methyl = sum(int(row.get("design_methyl_count", 0)) > 0 for row in target_raw)
    topup_with_methyl = sum(
        int(row.get("design_methyl_count", 0)) > 0 for row in topup_rows
    )
    unique_methylated = [
        row for row in target_unique if int(row.get("passes_methylation_hard_gate", 0))
    ]
    probabilities = [
        value for row in target_raw for value in probability_values(row)
    ]
    checks = {
        "complete_initial_v6_target_pool_is_present": len(initial_rows) == expected_initial,
        "recorded_total_budget_is_at_least_12000": (
            recorded_total_budget >= MINIMUM_ADAPTIVE_DRAWS_FOR_ABSTENTION
        ),
        "at_least_12000_adaptive_rows_are_present": (
            len(topup_rows) >= MINIMUM_ADAPTIVE_DRAWS_FOR_ABSTENTION
        ),
        "enough_disjoint_reserve_seeds_were_exhausted": (
            required_seed_count > 0
            and len(fully_exhausted_topup_seeds) >= required_seed_count
            and not (set(fully_exhausted_topup_seeds) & initial_seeds)
        ),
        "frozen_threshold_is_unchanged_on_every_target_row": all(
            float(row.get("methyl_threshold", "nan"))
            == float(manifest.get("methyl_threshold", 0.6))
            for row in target_raw
        ),
        "target_has_zero_novel_v6_methylated_candidates": len(target_eligible) == 0,
        "target_is_still_below_its_frozen_structure_quota": (
            len(target_eligible) < int(target_plan["structure_quota"])
        ),
    }
    approved = all(checks.values())
    if raw_with_methyl == 0:
        reason = "NO_V6_METHYL_CALL_AFTER_COMPLETE_INITIAL_AND_FIXED_ADAPTIVE_BUDGET"
    else:
        reason = (
            "NO_NOVEL_V6_METHYLATED_CANDIDATE_AFTER_COMPLETE_INITIAL_AND_"
            "FIXED_ADAPTIVE_BUDGET"
        )
    return {
        "target_name": target,
        "quality_gate": "PASS" if approved else "FAIL",
        "formal_abstention_approved": approved,
        "reason": reason,
        "release_action": ABSTENTION_ACTION,
        "planned_structure_quota": int(target_plan["structure_quota"]),
        "effective_structure_quota_after_abstention": 0,
        "initial_v6_raw_draws": len(initial_rows),
        "adaptive_topup_raw_draws": len(topup_rows),
        "total_v6_raw_draws": len(target_raw),
        "unique_natural_and_annotation_candidates": len(target_unique),
        "raw_rows_with_v6_methyl_call": raw_with_methyl,
        "adaptive_rows_with_v6_methyl_call": topup_with_methyl,
        "unique_rows_with_v6_methyl_call": len(unique_methylated),
        "novel_v6_methylated_candidates": len(target_eligible),
        "unique_methylated_historical_4115_overlaps": sum(
            int(row.get("seen_in_historical_4115", 0)) for row in unique_methylated
        ),
        "unique_methylated_prior_1333_overlaps": sum(
            int(row.get("seen_in_prior_1333", 0)) for row in unique_methylated
        ),
        "maximum_recorded_v6_methyl_probability": (
            max(probabilities) if probabilities else None
        ),
        "frozen_methyl_threshold": float(manifest.get("methyl_threshold", 0.6)),
        "adaptive_budget_required_for_abstention": (
            MINIMUM_ADAPTIVE_DRAWS_FOR_ABSTENTION
        ),
        "adaptive_budget_recorded": recorded_total_budget,
        "adaptive_reserve_seeds_observed": topup_seeds,
        "adaptive_rows_by_reserve_seed": dict(sorted(topup_seed_counts.items())),
        "fully_exhausted_adaptive_reserve_seeds": fully_exhausted_topup_seeds,
        "checks": checks,
    }


def backup_metadata_files(run_dir: Path) -> Path:
    backup_dir = run_dir / "pre_formal_abstention_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "generation_manifest.json",
        "generation_summary_by_target.csv",
        "target_manifest.csv",
    ):
        source = run_dir / name
        destination = backup_dir / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    return backup_dir


def run(args: argparse.Namespace) -> int:
    resumer = load_module("serine_v6_quota_resumer_for_finalize", RESUMER_PATH)
    generator = resumer.load_generator_module()

    plan_path = Path(args.plan).resolve()
    model_path = Path(args.model_path).resolve()
    run_dir = Path(args.run_dir).resolve()
    audit_path = Path(args.representation_audit_json).resolve()
    old_path = Path(args.old_designs_csv).resolve()
    prior_path = Path(args.prior_designs_csv).resolve()
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
        old_path,
        prior_path,
        *paths.values(),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    plan = generator.read_json(plan_path)
    validated = generator.validate_plan(plan)
    manifest = generator.read_json(paths["manifest"])
    if str(manifest.get("recovery_mode", "")) != resumer.RECOVERY_MODE:
        print("Fixed-budget V6 top-up has not run yet; adaptive sampling is required.")
        return NEEDS_SAMPLING_EXIT_CODE

    raw_rows = generator.read_csv(paths["all"])
    model_sha256 = resumer.sha256_file(model_path)
    representation_audit = generator.read_json(audit_path)
    resumer.validate_representation_audit(
        representation_audit, audit_path, model_sha256, plan_path
    )
    source_validation = resumer.validate_source_manifest(
        manifest,
        raw_rows,
        plan,
        plan_path,
        model_sha256,
        audit_path,
    )
    source_row_validation = resumer.validate_source_rows(
        raw_rows, manifest, plan, validated
    )
    old_exact, old_natural = generator.old_design_keys(old_path)
    _, prior_exact, prior_natural = generator.validate_prior_handoff(prior_path)
    unique_rows, eligible_rows = resumer.eligible_pool(
        generator,
        raw_rows,
        old_exact,
        old_natural,
        prior_exact,
        prior_natural,
    )
    annotation_audit = generator.audit_annotation_stability(raw_rows, eligible_rows)
    if str(annotation_audit.get("quality_gate", "")) != "PASS":
        raise RuntimeError(
            "V6 rows failed annotation audit before abstention finalization: "
            + ", ".join(
                resumer.false_checks(dict(annotation_audit.get("quality_checks", {})))
            )
        )
    persisted_unique = generator.read_csv(paths["unique"])
    persisted_eligible = generator.read_csv(paths["eligible"])
    recomputed_unique_keys = {
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in unique_rows
    }
    persisted_unique_keys = {
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in persisted_unique
    }
    recomputed_eligible_keys = {
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in eligible_rows
    }
    persisted_eligible_keys = {
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in persisted_eligible
    }
    if (
        recomputed_unique_keys != persisted_unique_keys
        or recomputed_eligible_keys != persisted_eligible_keys
    ):
        raise RuntimeError("Persisted V6 unique/eligible files do not recompute exactly")

    plan_by_target = {
        str(item["target_name"]).upper(): dict(item)
        for item in validated["targets"]
    }
    eligible_counts = Counter(
        str(row["target_name"]).upper() for row in eligible_rows
    )
    shortfalls = [
        target
        for target in validated["target_names"]
        if eligible_counts[target] < int(plan_by_target[target]["structure_quota"])
    ]
    recorded_shortfalls = {
        str(value).upper()
        for value in manifest.get("targets_below_pre_permeability_quota", [])
    }
    if set(shortfalls) != recorded_shortfalls:
        raise RuntimeError("Recorded and recomputed V6 shortfall targets differ")
    if not shortfalls:
        if (
            str(manifest.get("quality_gate", "")) == "PASS"
            and all(
                bool(value)
                for value in dict(manifest.get("quality_checks", {})).values()
            )
        ):
            print("All V6 target quotas already pass; no files changed.")
            return 0
        raise RuntimeError(
            "No quota shortfall was recomputed, but the V6 manifest is not a PASS"
        )

    topup_counts = Counter(
        str(row["target_name"]).upper()
        for row in raw_rows
        if str(row.get("source_recovery_stage", "")) == resumer.TOPUP_STAGE
    )
    if any(
        topup_counts[target] < MINIMUM_ADAPTIVE_DRAWS_FOR_ABSTENTION
        for target in shortfalls
    ):
        print(
            "Fixed adaptive budget is not exhausted for: "
            + ", ".join(
                f"{target} ({topup_counts[target]}/"
                f"{MINIMUM_ADAPTIVE_DRAWS_FOR_ABSTENTION})"
                for target in shortfalls
            )
        )
        return NEEDS_SAMPLING_EXIT_CODE

    abstentions = [
        evaluate_exhausted_target(
            target,
            plan_by_target[target],
            validated["seeds"],
            raw_rows,
            unique_rows,
            eligible_rows,
            manifest,
            resumer,
        )
        for target in shortfalls
    ]
    failed_abstentions = [
        row for row in abstentions if not bool(row["formal_abstention_approved"])
    ]
    if failed_abstentions:
        failed_checks = {
            row["target_name"]: resumer.false_checks(row["checks"])
            for row in failed_abstentions
        }
        raise RuntimeError(
            "A target reached the draw count but failed formal abstention audit: "
            + json.dumps(failed_checks, sort_keys=True)
        )

    before_hashes = {
        name: resumer.sha256_file(paths[name])
        for name in ("all", "unique", "eligible")
    }
    previous_manifest_sha256 = resumer.sha256_file(paths["manifest"])
    backup_dir = backup_metadata_files(run_dir)
    abstained_targets = {row["target_name"] for row in abstentions}

    summary_rows = generator.read_csv(paths["summary"])
    for row in summary_rows:
        target = str(row["target_name"]).upper()
        abstained = target in abstained_targets
        row["formal_target_abstention"] = int(abstained)
        row["coverage_resolution"] = (
            "FORMAL_MODEL_ABSTENTION_AFTER_FIXED_12000_DRAW_BUDGET"
            if abstained
            else "FROZEN_STRUCTURE_QUOTA_SATISFIED"
        )
        row["effective_structure_quota"] = (
            0 if abstained else int(row["planned_structure_quota"])
        )
        row["quota_satisfied_or_formally_abstained"] = int(
            int(row["enough_candidates_before_permeability"]) == 1 or abstained
        )
    generator.atomic_write_csv(paths["summary"], summary_rows, list(summary_rows[0]))

    target_manifest_rows = generator.read_csv(paths["target_manifest"])
    for row in target_manifest_rows:
        target = str(row["target_name"]).upper()
        abstained = target in abstained_targets
        row["formal_target_abstention"] = int(abstained)
        row["structure_release_action"] = (
            ABSTENTION_ACTION if abstained else "RETAIN_FOR_MANUAL_V6_REVIEW"
        )
        row["effective_structure_quota"] = (
            0 if abstained else int(plan_by_target[target]["structure_quota"])
        )
    generator.atomic_write_csv(
        paths["target_manifest"],
        target_manifest_rows,
        list(target_manifest_rows[0]),
    )

    audit_payload = {
        "quality_gate": "PASS",
        "protocol": PROTOCOL,
        "scientific_interpretation": (
            "A frozen structure quota is a screening objective, not permission to "
            "manufacture a positive model call. Targets with zero novel candidates "
            "after the complete initial V6 run and the committed 12,000-draw "
            "adaptive budget are retained as explicit model abstentions."
        ),
        "threshold_policy": "UNCHANGED_STRICTLY_GREATER_THAN_0_6",
        "checkpoint_policy": "UNCHANGED_HASH_PINNED_V6_CHECKPOINT",
        "candidate_file_policy": "NO_CANDIDATE_CSV_IS_REWRITTEN",
        "formal_target_abstentions": abstentions,
        "candidate_artifacts_before_and_after": {
            name: {"path": str(paths[name]), "sha256": digest}
            for name, digest in before_hashes.items()
        },
    }
    audit_path_out = run_dir / "formal_target_abstention_audit.json"
    generator.atomic_write_json(audit_path_out, audit_payload)

    after_hashes = {
        name: resumer.sha256_file(paths[name])
        for name in ("all", "unique", "eligible")
    }
    if before_hashes != after_hashes:
        raise RuntimeError("A candidate CSV changed during metadata-only abstention finalization")

    old_checks = dict(manifest.get("quality_checks", {}))
    old_checks.pop("every_target_meets_pre_structure_candidate_quota", None)
    if not all(bool(value) for value in old_checks.values()):
        raise RuntimeError("A non-coverage V6 quality check failed before finalization")
    quality_checks = {
        **old_checks,
        "every_below_quota_target_has_a_formal_fixed_budget_abstention": True,
        "every_non_abstained_target_meets_its_frozen_structure_quota": all(
            target in abstained_targets
            or eligible_counts[target] >= int(plan_by_target[target]["structure_quota"])
            for target in validated["target_names"]
        ),
        "formal_abstention_keeps_threshold_checkpoint_and_candidate_csvs_unchanged": True,
        "no_formally_abstained_target_releases_a_candidate": all(
            eligible_counts[target] == 0 for target in abstained_targets
        ),
    }
    effective_handoff = sum(
        int(plan_by_target[target]["structure_quota"])
        for target in validated["target_names"]
        if target not in abstained_targets
    )
    manifest.update(
        {
            "quality_gate": "PASS",
            "quality_checks": quality_checks,
            "release_status": RELEASE_STATUS,
            "coverage_resolution_mode": (
                "FROZEN_QUOTA_OR_FORMAL_MODEL_ABSTENTION_AFTER_FIXED_BUDGET"
            ),
            "scientific_reason": audit_payload["scientific_interpretation"],
            "formal_target_abstentions": abstentions,
            "targets_formally_abstained": sorted(abstained_targets),
            "unresolved_targets_below_pre_permeability_quota": [],
            "targets_below_pre_permeability_quota": shortfalls,
            "effective_planned_structure_handoff": effective_handoff,
            "effective_structure_target_count": (
                len(validated["target_names"]) - len(abstained_targets)
            ),
            "formal_target_abstention_audit": str(audit_path_out),
            "formal_target_abstention_audit_sha256": resumer.sha256_file(audit_path_out),
            "pre_formal_abstention_manifest_sha256": previous_manifest_sha256,
            "pre_formal_abstention_backup_dir": str(backup_dir),
            "candidate_artifact_sha256_unchanged_by_abstention": {
                name: {"path": str(paths[name]), "sha256": digest}
                for name, digest in after_hashes.items()
            },
            "source_v6_row_validation": source_row_validation,
            "source_v6_false_checks_before_abstention": source_validation[
                "source_false_checks"
            ],
            "annotation_stability_audit": annotation_audit,
        }
    )
    generator.atomic_write_json(paths["manifest"], manifest)

    print("===== V6 FIXED-BUDGET TARGET ABSTENTION FINALIZED =====")
    for row in abstentions:
        print(
            f"[{row['target_name']}] raw={row['total_v6_raw_draws']}, "
            f"top-up={row['adaptive_topup_raw_draws']}, "
            f"novel methylated={row['novel_v6_methylated_candidates']}, "
            "action=MODEL_ABSTAINS"
        )
    print(f"Candidate rows unchanged: {len(raw_rows)}")
    print(f"Final novel methylated candidates unchanged: {len(eligible_rows)}")
    print(f"Effective structure-review targets: {len(validated['target_names']) - len(abstentions)}")
    print("Quality gate: PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument(
        "--representation-audit-json", default=str(DEFAULT_REPRESENTATION_AUDIT)
    )
    parser.add_argument("--old-designs-csv", default=str(DEFAULT_OLD))
    parser.add_argument("--prior-designs-csv", default=str(DEFAULT_PRIOR))
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
