#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independently re-audit an order-balanced generation before any handoff.

This script deliberately recomputes checks from the CSV rows instead of
trusting the generation manifest.  It performs three separate passes:

1. file/count/row and threshold integrity;
2. annotation determinism, decoder-step anomalies, and provenance-backed
   interpretation of concentrated absolute positions;
3. prior-pool novelty, target coverage, and workflow-release constraints.

A PASS means the result is ready for manual scientific review, not that it has
already been released to the structure collaborator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_clean_v28.serine_qc_retrain.structural_support import (  # noqa: E402
    audit_dominant_position_structural_support,
    read_jsonl as read_structural_jsonl,
)


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_structural_support_v5"
    / "generation"
)
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_structure_failures.json")
DEFAULT_PRIOR = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "rerun_temperature_0.5_multiseed"
    / "methylated_new_candidates.csv"
)
DEFAULT_TRAIN = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_order_balanced_v3"
    / "data"
    / "train_serine_provenance_corrected.jsonl"
)
DEFAULT_TEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_order_balanced_v3"
    / "data"
    / "test_serine_provenance_corrected.jsonl"
)
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
EXPECTED_PRIOR_ROWS = 1_333
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
METHYLATABLE_AA = set(NATURAL_AA) - {"P"}
VALID_TOKENS = set(NATURAL_AA + NATURAL_AA.lower()) - {"p"}
ANNOTATION_MODE = "peptide_only_cyclic_order_ensemble_known_natural_sequence"
ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_order_balanced_v3"
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_list(value: object, field: str, row_id: str) -> List[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{row_id}: invalid JSON in {field}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{row_id}: {field} is not a JSON list")
    return parsed


def naturalize(sequence: str) -> str:
    return sequence.upper()


def methyl_positions(sequence: str) -> List[int]:
    return [index for index, token in enumerate(sequence, start=1) if token.islower()]


def exact_and_natural_keys(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]]]:
    exact: set[Tuple[str, str]] = set()
    natural: set[Tuple[str, str]] = set()
    for row in rows:
        target = str(row.get("target_name", "")).upper()
        sequence = str(row.get("design_seq", ""))
        natural_sequence = str(row.get("design_natural_seq", "")).upper()
        if not natural_sequence:
            natural_sequence = naturalize(sequence)
        exact.add((target, sequence))
        natural.add((target, natural_sequence))
    return exact, natural


def concentration_row(
    target: str,
    positions: Counter[int],
    residues: Counter[str],
    steps: Counter[int],
) -> Dict[str, Any]:
    total = int(sum(positions.values()))
    maximum_position_share = max(positions.values()) / total if total else 0.0
    maximum_residue_share = max(residues.values()) / total if total else 0.0
    maximum_step_share = max(steps.values()) / total if total else 0.0
    applies = total >= (100 if target == "ALL" else 30)
    return {
        "target_name": target,
        "methyl_sites": total,
        "site_position_counts": json.dumps(dict(sorted(positions.items()))),
        "site_residue_counts": json.dumps(dict(sorted(residues.items()))),
        "sampling_step_counts": json.dumps(dict(sorted(steps.items()))),
        "maximum_single_position_share": maximum_position_share,
        "maximum_single_residue_share": maximum_residue_share,
        "maximum_single_sampling_step_share": maximum_step_share,
        "concentration_gate_applies": int(applies),
        "position_gate_pass": int(not applies or maximum_position_share <= 0.80),
        "residue_gate_pass": int(not applies or maximum_residue_share <= 0.80),
        "sampling_step_gate_pass": int(not applies or maximum_step_share <= 0.80),
    }


def audit(
    run_dir: Path,
    plan_path: Path,
    prior_path: Path,
    out_dir: Path,
    train_path: Path | None = None,
    test_path: Path | None = None,
    native_path: Path | None = None,
) -> Dict[str, Any]:
    paths = {
        "all": run_dir / "all_candidates.csv",
        "unique": run_dir / "unique_candidates.csv",
        "eligible": run_dir / "methylated_new_candidates.csv",
        "target_manifest": run_dir / "target_manifest.csv",
        "target_summary": run_dir / "generation_summary_by_target.csv",
        "generation_manifest": run_dir / "generation_manifest.json",
    }
    for path in [plan_path, prior_path, *paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    plan = read_json(plan_path)
    manifest = read_json(paths["generation_manifest"])
    raw_rows = read_csv(paths["all"])
    unique_rows = read_csv(paths["unique"])
    eligible_rows = read_csv(paths["eligible"])
    target_manifest_rows = read_csv(paths["target_manifest"])
    target_summary_rows = read_csv(paths["target_summary"])
    prior_rows = read_csv(prior_path)
    target_plan = {
        str(row["target_name"]).upper(): dict(row) for row in plan["targets"]
    }
    target_names = set(target_plan)
    frozen_targets = {str(value).upper() for value in plan["frozen_targets"]}
    plan_raw = sum(
        int(row["sequences_per_seed"]) * len(plan["seeds"])
        for row in target_plan.values()
    )
    is_adaptive_v5 = (
        str(manifest.get("protocol", ""))
        == "temperature_0.5_structural_support_adaptive_quota_recovery_v5"
    )
    expected_raw = (
        int(manifest.get("raw_candidates_expected", -1))
        if is_adaptive_v5
        else plan_raw
    )
    recovery_stage_counts = Counter(
        str(row.get("source_recovery_stage", "")) for row in raw_rows
    )
    threshold = float(plan["methyl_threshold"])

    row_errors: List[str] = []
    raw_keys: Counter[Tuple[str, str]] = Counter()
    raw_ids: Counter[str] = Counter()
    repeated: MutableMapping[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    position_all: Counter[int] = Counter()
    residue_all: Counter[str] = Counter()
    step_all: Counter[int] = Counter()
    position_by_target: MutableMapping[str, Counter[int]] = defaultdict(Counter)
    residue_by_target: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    step_by_target: MutableMapping[str, Counter[int]] = defaultdict(Counter)
    order_std_maxima: List[float] = []

    for row_number, row in enumerate(raw_rows, start=2):
        row_id = str(row.get("candidate_id", f"CSV row {row_number}"))
        target = str(row.get("target_name", "")).upper()
        sequence = str(row.get("design_seq", ""))
        natural_sequence = str(row.get("design_natural_seq", ""))
        raw_ids[row_id] += 1
        raw_keys[(target, sequence)] += 1
        repeated[(target, natural_sequence)].append(row)
        try:
            probabilities = [
                float(value)
                for value in parse_list(row.get("methyl_probabilities"), "methyl_probabilities", row_id)
            ]
            order_std = [
                float(value)
                for value in parse_list(
                    row.get("methyl_probability_order_std"),
                    "methyl_probability_order_std",
                    row_id,
                )
            ]
            order = [
                int(value)
                for value in parse_list(
                    row.get("decoding_order_absolute"),
                    "decoding_order_absolute",
                    row_id,
                )
            ]
            observed_positions = [
                int(value)
                for value in parse_list(
                    row.get("methyl_positions_1based"),
                    "methyl_positions_1based",
                    row_id,
                )
            ]
        except (TypeError, ValueError) as exc:
            row_errors.append(str(exc))
            continue

        if target not in target_names:
            row_errors.append(f"{row_id}: unexpected target {target}")
        if not sequence or not set(sequence) <= VALID_TOKENS:
            row_errors.append(f"{row_id}: invalid design token")
        if natural_sequence != naturalize(sequence):
            row_errors.append(f"{row_id}: naturalized sequence mismatch")
        if int(row.get("design_length", -1)) != len(sequence):
            row_errors.append(f"{row_id}: design length mismatch")
        if int(row.get("native_length", -1)) != len(sequence):
            row_errors.append(f"{row_id}: native/design length mismatch")
        expected_positions = methyl_positions(sequence)
        if observed_positions != expected_positions:
            row_errors.append(f"{row_id}: methyl position mismatch")
        if int(row.get("design_methyl_count", -1)) != len(expected_positions):
            row_errors.append(f"{row_id}: methyl count mismatch")
        if len(probabilities) != len(sequence) or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in probabilities
        ):
            row_errors.append(f"{row_id}: invalid probability vector")
        if len(order_std) != len(sequence) or any(
            not math.isfinite(value) or value < 0.0 for value in order_std
        ):
            row_errors.append(f"{row_id}: invalid order-std vector")
        if len(order) != len(sequence) or len(set(order)) != len(order):
            row_errors.append(f"{row_id}: invalid external decoding order")
        if str(row.get("annotation_mode", "")) != ANNOTATION_MODE:
            row_errors.append(f"{row_id}: wrong annotation mode")
        if str(row.get("annotation_context_policy", "")) != ANNOTATION_CONTEXT:
            row_errors.append(f"{row_id}: wrong annotation context")
        if int(row.get("annotation_visible_receptor_chains", -1)) != 0:
            row_errors.append(f"{row_id}: receptor chain leaked into annotation context")
        if int(row.get("annotation_order_ensemble_size", -1)) != len(sequence):
            row_errors.append(f"{row_id}: wrong annotation ensemble size")
        if len(probabilities) == len(sequence):
            for index, (token, probability) in enumerate(
                zip(sequence, probabilities), start=1
            ):
                should_be_lower = (
                    token.upper() in METHYLATABLE_AA and probability > threshold
                )
                if token.islower() != should_be_lower:
                    row_errors.append(
                        f"{row_id}: threshold/annotation mismatch at position {index}"
                    )
        if order_std:
            recorded_max = float(row.get("methyl_probability_order_std_max", "nan"))
            if not math.isfinite(recorded_max) or abs(recorded_max - max(order_std)) > 1e-6:
                row_errors.append(f"{row_id}: order-std maximum mismatch")
            order_std_maxima.append(max(order_std))

        sorted_order = sorted(order)
        for position in expected_positions:
            residue = sequence[position - 1].upper()
            position_all[position] += 1
            residue_all[residue] += 1
            position_by_target[target][position] += 1
            residue_by_target[target][residue] += 1
            if len(sorted_order) == len(sequence):
                absolute_position = sorted_order[position - 1]
                step = order.index(absolute_position) + 1
                step_all[step] += 1
                step_by_target[target][step] += 1

    # Release-distribution gates are recomputed on the final eligible unique
    # table, not on repeated raw draws. Raw rows above remain the source for
    # path integrity and repeated-sequence determinism.
    position_all = Counter()
    residue_all = Counter()
    step_all = Counter()
    position_by_target = defaultdict(Counter)
    residue_by_target = defaultdict(Counter)
    step_by_target = defaultdict(Counter)
    order_std_maxima = []
    for row in eligible_rows:
        target = str(row["target_name"]).upper()
        sequence = str(row["design_seq"])
        order = [int(value) for value in json.loads(str(row["decoding_order_absolute"]))]
        order_std = [
            float(value)
            for value in json.loads(str(row["methyl_probability_order_std"]))
        ]
        if order_std:
            order_std_maxima.append(max(order_std))
        sorted_order = sorted(order)
        for position in methyl_positions(sequence):
            residue = sequence[position - 1].upper()
            position_all[position] += 1
            residue_all[residue] += 1
            position_by_target[target][position] += 1
            residue_by_target[target][residue] += 1
            absolute_position = sorted_order[position - 1]
            step = order.index(absolute_position) + 1
            step_all[step] += 1
            step_by_target[target][step] += 1

    repeated_groups = [rows for rows in repeated.values() if len(rows) > 1]
    inconsistent_annotation_groups = 0
    probability_disagreement_groups = 0
    for rows in repeated_groups:
        if len({str(row["design_seq"]) for row in rows}) != 1:
            inconsistent_annotation_groups += 1
        # V4 scores each unique target/natural sequence once and copies that
        # serialized payload to every occurrence.  Require exact persistence;
        # do not reinterpret last-place CUDA batch noise as biology.
        reference_probability_json = str(rows[0]["methyl_probabilities"])
        if any(
            str(row["methyl_probabilities"]) != reference_probability_json
            for row in rows[1:]
        ):
            probability_disagreement_groups += 1

    unique_key_counts = Counter(
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in unique_rows
    )
    aggregation_errors = []
    if any(count != 1 for count in unique_key_counts.values()):
        aggregation_errors.append("unique_candidates.csv contains duplicate keys")
    if set(unique_key_counts) != set(raw_keys):
        aggregation_errors.append("unique/raw target+design key sets differ")
    for row in unique_rows:
        key = (str(row["target_name"]).upper(), str(row["design_seq"]))
        if int(row.get("occurrence_count", -1)) != raw_keys[key]:
            aggregation_errors.append(f"occurrence count mismatch for {key}")

    eligible_keys = {
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in eligible_rows
    }
    expected_eligible_keys = {
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in unique_rows
        if int(row.get("eligible_for_new_permeability_screen", 0)) == 1
    }
    eligible_row_errors = []
    if eligible_keys != expected_eligible_keys:
        eligible_row_errors.append("eligible file does not match unique eligibility flags")
    for row in eligible_rows:
        if int(row.get("design_methyl_count", 0)) <= 0:
            eligible_row_errors.append(
                f"non-methyl row in eligible file: {row.get('candidate_id')}"
            )

    pass_1_checks = {
        "order_balanced_checkpoint_protocol": (
            manifest.get("model_expert_qc_protocol") == EXPERT_PROTOCOL
        ),
        "peptide_only_train_matched_annotation_protocol": (
            manifest.get("annotation_mode") == ANNOTATION_MODE
            and manifest.get("annotation_context_policy") == ANNOTATION_CONTEXT
            and int(manifest.get("annotation_visible_receptor_chains", -1)) == 0
            and bool(manifest.get("train_deployment_context_match"))
        ),
        "raw_count_matches_plan": len(raw_rows) == expected_raw,
        "adaptive_v5_source_plus_topup_accounting": (
            not is_adaptive_v5
            or (
                int(manifest.get("source_v4_raw_candidates_retained", -1))
                + int(manifest.get("adaptive_topup_raw_candidates", -1))
                == len(raw_rows)
                and int(manifest.get("source_v4_raw_candidates_retained", -1))
                == plan_raw
                and recovery_stage_counts["V4_RESCORED_V3_POOL"]
                == int(manifest.get("source_v4_raw_candidates_retained", -1))
                and recovery_stage_counts["V5_ADAPTIVE_QUOTA_TOPUP"]
                == int(manifest.get("adaptive_topup_raw_candidates", -1))
            )
        ),
        "generation_manifest_counts_match_files": (
            int(manifest.get("raw_candidates_generated", -1)) == len(raw_rows)
            and int(manifest.get("unique_candidates", -1)) == len(unique_rows)
            and int(
                manifest.get("new_methylated_candidates_for_permeability", -1)
            )
            == len(eligible_rows)
        ),
        "raw_candidate_ids_are_unique": all(count == 1 for count in raw_ids.values()),
        "all_raw_rows_pass_independent_semantic_validation": not row_errors,
        "unique_aggregation_recomputes_exactly": not aggregation_errors,
        "eligible_subset_recomputes_exactly": not eligible_row_errors,
        "target_manifest_matches_plan": (
            {str(row["target_name"]).upper() for row in target_manifest_rows}
            == target_names
        ),
    }

    concentration_rows = [
        concentration_row("ALL", position_all, residue_all, step_all)
    ]
    for target in sorted(target_names):
        concentration_rows.append(
            concentration_row(
                target,
                position_by_target[target],
                residue_by_target[target],
                step_by_target[target],
            )
        )
    reported_annotation_audit = dict(
        manifest.get("annotation_stability_audit", {})
    )
    reported_positions = {
        int(key): int(value)
        for key, value in dict(
            reported_annotation_audit.get("eligible_site_position_counts", {})
        ).items()
    }
    reported_residues = {
        str(key): int(value)
        for key, value in dict(
            reported_annotation_audit.get("eligible_site_residue_counts", {})
        ).items()
    }
    concentrated_target_rows = [
        row
        for row in concentration_rows
        if str(row["target_name"]) != "ALL"
        and bool(row["concentration_gate_applies"])
        and not bool(row["position_gate_pass"])
    ]
    global_position_concentration = any(
        str(row["target_name"]) == "ALL"
        and bool(row["concentration_gate_applies"])
        and not bool(row["position_gate_pass"])
        for row in concentration_rows
    )
    if concentrated_target_rows:
        missing_support_inputs = [
            str(path)
            for path in (train_path, test_path, native_path)
            if path is None or not path.is_file()
        ]
        if missing_support_inputs:
            structural_support = {
                "quality_gate": "FAIL",
                "method": "forward_cyclic_ca_distance_matrix_heldout_provenance_support_v1",
                "reason": "missing structural-support inputs",
                "missing_inputs": missing_support_inputs,
                "concentrated_target_count": len(concentrated_target_rows),
            }
        else:
            structural_support = audit_dominant_position_structural_support(
                eligible_rows=eligible_rows,
                native_rows=read_structural_jsonl(native_path),
                target_manifest_rows=target_manifest_rows,
                train_records=read_structural_jsonl(train_path),
                test_records=read_structural_jsonl(test_path),
            )
    elif global_position_concentration:
        structural_support = {
            "quality_gate": "FAIL",
            "method": "forward_cyclic_ca_distance_matrix_heldout_provenance_support_v1",
            "reason": (
                "global position concentration was present without a concentrated "
                "target that could be validated structurally"
            ),
            "concentrated_target_count": 0,
        }
    else:
        structural_support = {
            "quality_gate": "PASS",
            "method": "forward_cyclic_ca_distance_matrix_heldout_provenance_support_v1",
            "reason": "no target exceeded the structural-support trigger",
            "concentrated_target_count": 0,
        }
    atomic_write_json(out_dir / "structural_position_support.json", structural_support)

    pass_2_checks = {
        "repeated_natural_sequences_have_one_annotation": (
            inconsistent_annotation_groups == 0
        ),
        "repeated_natural_sequences_have_matching_probabilities": (
            probability_disagreement_groups == 0
        ),
        "no_global_or_target_sampling_step_concentration_above_80_percent": all(
            bool(row["sampling_step_gate_pass"]) for row in concentration_rows
        ),
        "no_global_residue_concentration_above_80_percent": all(
            bool(row["residue_gate_pass"])
            for row in concentration_rows
            if str(row["target_name"]) == "ALL"
        ),
        "dominant_position_concentration_has_heldout_provenance_support": (
            str(structural_support.get("quality_gate", "")) == "PASS"
        ),
        "absolute_position_and_residue_concentration_are_reported_not_filtered": True,
        "generation_manifest_annotation_audit_recomputes": (
            reported_positions == dict(position_all)
            and reported_residues == dict(residue_all)
            and int(
                reported_annotation_audit.get(
                    "raw_inconsistent_annotation_groups", -1
                )
            )
            == inconsistent_annotation_groups
            and int(
                reported_annotation_audit.get(
                    "raw_probability_disagreement_groups", -1
                )
            )
            == probability_disagreement_groups
        ),
    }

    prior_exact, prior_natural = exact_and_natural_keys(prior_rows)
    eligible_exact, eligible_natural = exact_and_natural_keys(eligible_rows)
    historical_path = Path(str(manifest.get("historical_design_csv", "")))
    historical_available = historical_path.is_file()
    if historical_available:
        historical_exact, historical_natural = exact_and_natural_keys(
            read_csv(historical_path)
        )
    else:
        historical_exact, historical_natural = set(), set()
    candidates_by_target = Counter(
        str(row["target_name"]).upper() for row in eligible_rows
    )
    quota_shortfalls = [
        target
        for target, config in target_plan.items()
        if candidates_by_target[target] < int(config["structure_quota"])
    ]
    summary_index = {
        str(row["target_name"]).upper(): row for row in target_summary_rows
    }
    summary_mismatches = []
    for target in sorted(target_names):
        if target not in summary_index:
            summary_mismatches.append(f"missing summary for {target}")
            continue
        if int(summary_index[target]["new_methylated_for_permeability"]) != candidates_by_target[target]:
            summary_mismatches.append(f"candidate summary mismatch for {target}")

    deferred = manifest.get("permeability_status") == "DEFERRED_UNTIL_STRUCTURE_RETURNS"
    permeability_files_absent = not any(
        (run_dir / name).exists()
        for name in ("permeability_input.csv", "permeability_input_manifest.csv")
    )
    pass_3_checks = {
        "prior_handoff_has_exactly_1333_rows": len(prior_rows) == EXPECTED_PRIOR_ROWS,
        "eligible_exact_sequences_are_new_vs_prior_1333": not (
            eligible_exact & prior_exact
        ),
        "eligible_natural_sequences_are_new_vs_prior_1333": not (
            eligible_natural & prior_natural
        ),
        "historical_4115_source_is_available_for_independent_check": historical_available,
        "eligible_exact_sequences_are_new_vs_historical_4115": (
            historical_available and not (eligible_exact & historical_exact)
        ),
        "eligible_natural_sequences_are_new_vs_historical_4115": (
            historical_available and not (eligible_natural & historical_natural)
        ),
        "only_planned_failed_targets_are_present": (
            {str(row["target_name"]).upper() for row in eligible_rows} <= target_names
            and not ({str(row["target_name"]).upper() for row in eligible_rows} & frozen_targets)
        ),
        "every_target_meets_structure_quota": not quota_shortfalls,
        "target_summaries_recompute": not summary_mismatches,
        "permeability_is_still_deferred": deferred and permeability_files_absent,
    }

    pass_1 = "PASS" if all(pass_1_checks.values()) else "FAIL"
    pass_2 = "PASS" if all(pass_2_checks.values()) else "FAIL"
    pass_3 = "PASS" if all(pass_3_checks.values()) else "FAIL"
    quality_gate = "PASS" if pass_1 == pass_2 == pass_3 == "PASS" else "FAIL"
    report = {
        "quality_gate": quality_gate,
        "release_status": (
            "READY_FOR_MANUAL_SCIENTIFIC_REVIEW"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_SEND_TO_SHANGGE"
        ),
        "protocol": "independent_three_pass_peptide_only_generation_audit_v2",
        "run_dir": str(run_dir),
        "plan": str(plan_path),
        "prior_handoff": str(prior_path),
        "pass_1_integrity": {
            "quality_gate": pass_1,
            "checks": pass_1_checks,
            "raw_rows": len(raw_rows),
            "expected_raw_rows": expected_raw,
            "unique_rows": len(unique_rows),
            "eligible_rows": len(eligible_rows),
            "row_error_count": len(row_errors),
            "row_error_examples": row_errors[:25],
            "aggregation_errors": aggregation_errors[:25],
            "eligible_row_errors": eligible_row_errors[:25],
        },
        "pass_2_result_annotation": {
            "quality_gate": pass_2,
            "checks": pass_2_checks,
            "repeated_target_natural_sequence_groups": len(repeated_groups),
            "inconsistent_annotation_groups": inconsistent_annotation_groups,
            "probability_disagreement_groups": probability_disagreement_groups,
            "maximum_candidate_order_probability_std": (
                max(order_std_maxima) if order_std_maxima else 0.0
            ),
            "mean_candidate_order_probability_std_max": (
                sum(order_std_maxima) / len(order_std_maxima)
                if order_std_maxima
                else 0.0
            ),
            "concentration": concentration_rows,
            "absolute_concentration_policy": (
                "position and target-local residue shares are diagnostics; global "
                "residue collapse and sampling-step concentration are hard gates, "
                "and dominant absolute positions require held-out provenance-backed "
                "structural support"
            ),
            "structural_position_support": structural_support,
        },
        "pass_3_novelty_coverage_workflow": {
            "quality_gate": pass_3,
            "checks": pass_3_checks,
            "quota_shortfalls": quota_shortfalls,
            "summary_mismatches": summary_mismatches,
            "prior_exact_overlaps": len(eligible_exact & prior_exact),
            "prior_natural_overlaps": len(eligible_natural & prior_natural),
            "historical_exact_overlaps": len(eligible_exact & historical_exact),
            "historical_natural_overlaps": len(eligible_natural & historical_natural),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    atomic_write_json(out_dir / "three_pass_generation_audit.json", report)
    atomic_write_csv(
        out_dir / "three_pass_concentration_by_target.csv",
        concentration_rows,
        list(concentration_rows[0]),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--prior-handoff-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN))
    parser.add_argument("--test-jsonl", default=str(DEFAULT_TEST))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    plan_path = Path(args.plan).resolve()
    prior_path = Path(args.prior_handoff_csv).resolve()
    train_path = Path(args.train_jsonl).resolve()
    test_path = Path(args.test_jsonl).resolve()
    native_path = Path(args.native_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "triple_audit"
    report = audit(
        run_dir,
        plan_path,
        prior_path,
        out_dir,
        train_path=train_path,
        test_path=test_path,
        native_path=native_path,
    )
    print("===== INDEPENDENT THREE-PASS GENERATION AUDIT =====", flush=True)
    print(f"Quality gate: {report['quality_gate']}", flush=True)
    print(f"Release status: {report['release_status']}", flush=True)
    print(f"Report: {out_dir / 'three_pass_generation_audit.json'}", flush=True)
    if report["quality_gate"] != "PASS":
        failed = []
        for key in (
            "pass_1_integrity",
            "pass_2_result_annotation",
            "pass_3_novelty_coverage_workflow",
        ):
            if report[key]["quality_gate"] != "PASS":
                failed.append(key)
        raise RuntimeError("Three-pass audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
