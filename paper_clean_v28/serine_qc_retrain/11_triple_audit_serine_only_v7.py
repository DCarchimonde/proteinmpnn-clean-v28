#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent three-pass audit for the Ser-only V7 reannotation.

This auditor does not accept formal abstention and does not use nearest-backbone
label similarity as biological ground truth.  It independently reconstructs
threshold annotations and aggregation, checks cyclic/decoder diagnostics and
sampling-step balance, and verifies novelty plus all-17 target coverage.  The
3AV family concentration is additionally interpreted against an explicit
cyclic distance-matrix alignment so a conserved physical site is not confused
with a fixed tensor column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
V7_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_serine_only_cyclic_v7"
DEFAULT_RUN = V7_ROOT / "generation"
DEFAULT_OUT = V7_ROOT / "triple_audit"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_representation_v6.json")
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_HISTORICAL = (
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

V7_GENERATION_PROTOCOL = (
    "temperature_0.5_serine_only_cyclic_v7_reannotation_of_preserved_v6_pool"
)
V7_EXPERT_PROTOCOL = (
    "canonical_clean_v28_serine_only_corrected_labels_"
    "cyclic_representation_augmented_v7"
)
V7_REPRESENTATION_AUDIT_PROTOCOL = (
    "cyclic_representation_equivariance_heldout_gate_v2_serine_only"
)
V7_REPRESENTATION_AUTHORIZATION = (
    "SERINE_ONLY_REPAIR_VALIDATED_FOR_ISOLATED_V7_REANNOTATION"
)
ANNOTATION_MODE = (
    "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
)
ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
EXPECTED_RAW_ROWS = 31_500
EXPECTED_TARGETS = 17
EXPECTED_V6_ALL_SHA256 = (
    "1ab4791c09a1b2428b1a84894d13bb8c4049ba580df05bebd93c263a2e4e634c"
)
EXPECTED_V6_MANIFEST_SHA256 = (
    "067a22a2175c97cf483e64967168eefc676389e302c9acc79a66c70e8290711f"
)
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
METHYLATABLE_AA = set(NATURAL_AA) - {"P"}
VALID_TOKENS = set(NATURAL_AA + NATURAL_AA.lower()) - {"p"}
AV_FAMILY = {
    "3AV9",
    "3AVA",
    "3AVB",
    "3AVF",
    "3AVG",
    "3AVH",
    "3AVI",
    "3AVJ",
    "3AVK",
    "3AVM",
    "3AVN",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


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
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_list(value: object, field: str, row_id: str) -> List[Any]:
    try:
        result = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{row_id}: invalid JSON in {field}") from exc
    if not isinstance(result, list):
        raise ValueError(f"{row_id}: {field} is not a JSON list")
    return result


def methyl_positions(sequence: str) -> List[int]:
    return [index for index, token in enumerate(sequence, start=1) if token.islower()]


def exact_and_natural_keys(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]]]:
    exact: set[Tuple[str, str]] = set()
    natural: set[Tuple[str, str]] = set()
    for row in rows:
        target = str(row.get("target_name", "")).upper()
        design = str(row.get("design_seq", ""))
        natural_sequence = str(row.get("design_natural_seq", "")).upper()
        if not natural_sequence:
            natural_sequence = design.upper()
        exact.add((target, design))
        natural.add((target, natural_sequence))
    return exact, natural


def distance_matrix(coordinates: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [
            math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
            for right in coordinates
        ]
        for left in coordinates
    ]


def cyclic_matrix_rmse(
    reference: Sequence[Sequence[float]], query: Sequence[Sequence[float]], shift: int
) -> float:
    length = len(reference)
    squared = [
        (
            float(reference[i][j])
            - float(query[(i + shift) % length][(j + shift) % length])
        )
        ** 2
        for i in range(length)
        for j in range(length)
    ]
    return math.sqrt(sum(squared) / len(squared))


def av_family_alignment_audit(
    native_rows: Sequence[Mapping[str, Any]],
    selected_chains: Mapping[str, str],
    dominant_position_by_target: Mapping[str, int],
) -> Dict[str, Any]:
    native_index = {
        str(row.get("name", "")).upper(): row for row in native_rows
    }
    reference_target = "3AV9"
    reference_chain = selected_chains[reference_target]
    reference_coordinates = native_index[reference_target][
        f"CA_chain_{reference_chain}"
    ]
    reference_matrix = distance_matrix(reference_coordinates)
    reference_position = dominant_position_by_target.get(reference_target)
    rows: List[Dict[str, Any]] = []
    for target in sorted(AV_FAMILY):
        chain = selected_chains[target]
        coordinates = native_index[target][f"CA_chain_{chain}"]
        if len(coordinates) != len(reference_coordinates):
            rows.append(
                {
                    "target_name": target,
                    "peptide_length": len(coordinates),
                    "best_cyclic_shift": None,
                    "distance_matrix_rmse": None,
                    "dominant_position_1based": dominant_position_by_target.get(target),
                    "maps_to_reference_dominant_position": False,
                    "quality_gate": "FAIL",
                    "reason": "length differs from 3AV9 homolog reference",
                }
            )
            continue
        query_matrix = distance_matrix(coordinates)
        candidates = [
            (cyclic_matrix_rmse(reference_matrix, query_matrix, shift), shift)
            for shift in range(len(coordinates))
        ]
        best_rmse, best_shift = min(candidates)
        target_position = dominant_position_by_target.get(target)
        mapped_position = (
            ((target_position - 1 - best_shift) % len(coordinates)) + 1
            if target_position is not None
            else None
        )
        mapped = (
            reference_position is not None
            and mapped_position is not None
            and mapped_position == reference_position
        )
        row_pass = best_shift == 0 and mapped
        rows.append(
            {
                "target_name": target,
                "peptide_length": len(coordinates),
                "best_cyclic_shift": best_shift,
                "distance_matrix_rmse": best_rmse,
                "dominant_position_1based": target_position,
                "mapped_reference_position_1based": mapped_position,
                "reference_dominant_position_1based": reference_position,
                "maps_to_reference_dominant_position": mapped,
                "quality_gate": "PASS" if row_pass else "FAIL",
                "reason": (
                    "same physical homolog position under best cyclic alignment"
                    if row_pass
                    else "dominant site does not map to the 3AV9 physical homolog site"
                ),
            }
        )
    return {
        "quality_gate": (
            "PASS" if rows and all(row["quality_gate"] == "PASS" for row in rows) else "FAIL"
        ),
        "method": "all_ca_pair_distance_matrix_cyclic_alignment_to_3AV9",
        "interpretation": (
            "A shift-zero conserved dominant site across the homologous 3AV family "
            "is a physical family site, not an arbitrary seventh tensor column."
        ),
        "reference_target": reference_target,
        "targets": rows,
    }


def audit(
    run_dir: Path,
    plan_path: Path,
    native_path: Path,
    historical_path: Path,
    prior_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    paths = {
        "all": run_dir / "all_candidates.csv",
        "unique": run_dir / "unique_candidates.csv",
        "eligible": run_dir / "methylated_new_candidates.csv",
        "target_manifest": run_dir / "target_manifest.csv",
        "target_summary": run_dir / "generation_summary_by_target.csv",
        "generation_manifest": run_dir / "generation_manifest.json",
    }
    for required in (
        plan_path,
        native_path,
        historical_path,
        prior_path,
        *paths.values(),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    plan = read_json(plan_path)
    manifest = read_json(paths["generation_manifest"])
    raw_rows = read_csv(paths["all"])
    unique_rows = read_csv(paths["unique"])
    eligible_rows = read_csv(paths["eligible"])
    target_manifest_rows = read_csv(paths["target_manifest"])
    target_summary_rows = read_csv(paths["target_summary"])
    target_names = {
        str(row["target_name"]).upper() for row in plan["targets"]
    }
    threshold = float(plan["methyl_threshold"])

    row_errors: List[str] = []
    raw_ids: Counter[str] = Counter()
    raw_design_keys: Counter[Tuple[str, str]] = Counter()
    repeated: MutableMapping[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in raw_rows:
        row_id = str(row.get("candidate_id", ""))
        target = str(row.get("target_name", "")).upper()
        sequence = str(row.get("design_seq", ""))
        natural = str(row.get("design_natural_seq", "")).upper()
        raw_ids[row_id] += 1
        raw_design_keys[(target, sequence)] += 1
        repeated[(target, natural)].append(row)
        try:
            positions = [
                int(value)
                for value in parse_list(
                    row.get("methyl_positions_1based"),
                    "methyl_positions_1based",
                    row_id,
                )
            ]
            probabilities = [
                float(value)
                for value in parse_list(
                    row.get("methyl_probabilities"), "methyl_probabilities", row_id
                )
            ]
            order_std = [
                float(value)
                for value in parse_list(
                    row.get("methyl_probability_order_std"),
                    "methyl_probability_order_std",
                    row_id,
                )
            ]
            rep_min = [
                float(value)
                for value in parse_list(
                    row.get("methyl_probability_representation_min"),
                    "methyl_probability_representation_min",
                    row_id,
                )
            ]
            rep_max = [
                float(value)
                for value in parse_list(
                    row.get("methyl_probability_representation_max"),
                    "methyl_probability_representation_max",
                    row_id,
                )
            ]
            rep_span = [
                float(value)
                for value in parse_list(
                    row.get("methyl_probability_representation_span"),
                    "methyl_probability_representation_span",
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
        except ValueError as exc:
            row_errors.append(str(exc))
            continue
        expected_positions = methyl_positions(sequence)
        if not row_id or target not in target_names:
            row_errors.append(f"{row_id}: missing ID or unexpected target")
        if not sequence or not set(sequence) <= VALID_TOKENS:
            row_errors.append(f"{row_id}: invalid output token")
        if natural != sequence.upper() or not set(natural) <= set(NATURAL_AA):
            row_errors.append(f"{row_id}: natural sequence mismatch")
        if positions != expected_positions:
            row_errors.append(f"{row_id}: methyl position mismatch")
        if int(row.get("design_methyl_count", -1)) != len(expected_positions):
            row_errors.append(f"{row_id}: methyl count mismatch")
        vectors = (probabilities, order_std, rep_min, rep_max, rep_span)
        if any(len(values) != len(sequence) for values in vectors):
            row_errors.append(f"{row_id}: annotation vector length mismatch")
            continue
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for values in (probabilities, rep_min, rep_max)
            for value in values
        ) or any(
            not math.isfinite(value) or value < 0.0
            for values in (order_std, rep_span)
            for value in values
        ):
            row_errors.append(f"{row_id}: invalid annotation diagnostic")
        if any(
            minimum > mean + 1e-7
            or mean > maximum + 1e-7
            or abs((maximum - minimum) - span) > 2e-6
            for mean, minimum, maximum, span in zip(
                probabilities, rep_min, rep_max, rep_span
            )
        ):
            row_errors.append(f"{row_id}: representation mean/min/max/span conflict")
        for position, (token, probability) in enumerate(
            zip(sequence, probabilities), start=1
        ):
            should_be_lower = token.upper() in METHYLATABLE_AA and probability > threshold
            if token.islower() != should_be_lower:
                row_errors.append(
                    f"{row_id}: strict threshold mismatch at position {position}"
                )
        if (
            str(row.get("annotation_mode", "")) != ANNOTATION_MODE
            or str(row.get("annotation_context_policy", "")) != ANNOTATION_CONTEXT
            or int(row.get("annotation_visible_receptor_chains", -1)) != 0
            or int(row.get("annotation_representation_ensemble_size", -1))
            != len(sequence)
            or int(row.get("annotation_decoder_order_ensemble_size", -1))
            != len(sequence)
        ):
            row_errors.append(f"{row_id}: annotation context/ensemble mismatch")
        if len(order) != len(sequence) or len(set(order)) != len(order):
            row_errors.append(f"{row_id}: invalid preserved base-sampling order")
        if str(row.get("sampling_path_methyl_probabilities", "")):
            row_errors.append(f"{row_id}: stale V6 expert path probabilities were reused")

    unique_keys = Counter(
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in unique_rows
    )
    aggregation_errors: List[str] = []
    if any(count != 1 for count in unique_keys.values()):
        aggregation_errors.append("unique_candidates contains duplicate target+design keys")
    if set(unique_keys) != set(raw_design_keys):
        aggregation_errors.append("unique/raw target+design key sets differ")
    for row in unique_rows:
        key = (str(row["target_name"]).upper(), str(row["design_seq"]))
        if int(row.get("occurrence_count", -1)) != raw_design_keys[key]:
            aggregation_errors.append(f"occurrence count mismatch: {key}")

    eligible_keys = {
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in eligible_rows
    }
    expected_eligible_keys = {
        (str(row["target_name"]).upper(), str(row["design_seq"]))
        for row in unique_rows
        if int(row.get("eligible_for_new_permeability_screen", 0)) == 1
    }
    eligible_errors: List[str] = []
    if eligible_keys != expected_eligible_keys:
        eligible_errors.append("eligible CSV differs from unique eligibility flags")
    if any(int(row.get("design_methyl_count", 0)) <= 0 for row in eligible_rows):
        eligible_errors.append("eligible CSV contains a non-methyl row")

    candidate_artifacts = dict(manifest.get("candidate_artifacts") or {})
    artifact_hashes_match = all(
        str(dict(candidate_artifacts.get(name) or {}).get("sha256", ""))
        == sha256_file(paths[name])
        for name in ("all", "unique", "eligible", "target_manifest", "target_summary")
    )
    pinned_representation = dict(
        manifest.get("cyclic_representation_heldout_audit") or {}
    )
    pass_1_checks = {
        "generation_manifest_is_serine_only_v7_pass": (
            manifest.get("quality_gate") == "PASS"
            and manifest.get("protocol") == V7_GENERATION_PROTOCOL
            and manifest.get("model_expert_qc_protocol") == V7_EXPERT_PROTOCOL
            and manifest.get("expert_scope") == "serine-only"
        ),
        "heldout_representation_audit_is_pinned": (
            pinned_representation.get("quality_gate") == "PASS"
            and pinned_representation.get("protocol")
            == V7_REPRESENTATION_AUDIT_PROTOCOL
            and pinned_representation.get("release_authorization")
            == V7_REPRESENTATION_AUTHORIZATION
            and pinned_representation.get("model_sha256")
            == manifest.get("model_sha256")
        ),
        "source_v6_artifact_hashes_are_exactly_pinned": (
            manifest.get("source_v6_all_candidates_sha256")
            == EXPECTED_V6_ALL_SHA256
            and manifest.get("source_v6_generation_manifest_sha256")
            == EXPECTED_V6_MANIFEST_SHA256
        ),
        "raw_count_target_set_and_candidate_ids_are_exact": (
            len(raw_rows) == EXPECTED_RAW_ROWS
            and {str(row["target_name"]).upper() for row in raw_rows} == target_names
            and len(target_names) == EXPECTED_TARGETS
            and all(raw_ids)
            and all(count == 1 for count in raw_ids.values())
        ),
        "candidate_artifact_hashes_match_manifest": artifact_hashes_match,
        "all_raw_rows_pass_independent_threshold_and_context_validation": not row_errors,
        "unique_aggregation_recomputes": not aggregation_errors,
        "eligible_subset_recomputes": not eligible_errors,
        "manifest_counts_match_files": (
            int(manifest.get("raw_candidates_generated", -1)) == len(raw_rows)
            and int(manifest.get("unique_candidates", -1)) == len(unique_rows)
            and int(manifest.get("new_methylated_candidates_for_structure_review", -1))
            == len(eligible_rows)
        ),
    }

    repeated_groups = [values for values in repeated.values() if len(values) > 1]
    inconsistent_repeated = [
        values
        for values in repeated_groups
        if len({str(row["design_seq"]) for row in values}) != 1
        or len({str(row["methyl_probabilities"]) for row in values}) != 1
    ]
    positions_all: Counter[int] = Counter()
    residues_all: Counter[str] = Counter()
    steps_all: Counter[int] = Counter()
    positions_by_target: MutableMapping[str, Counter[int]] = defaultdict(Counter)
    residues_by_target: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    steps_by_target: MutableMapping[str, Counter[int]] = defaultdict(Counter)
    for row in eligible_rows:
        target = str(row["target_name"]).upper()
        sequence = str(row["design_seq"])
        order = [int(value) for value in json.loads(str(row["decoding_order_absolute"]))]
        sorted_order = sorted(order)
        for position in methyl_positions(sequence):
            residue = sequence[position - 1].upper()
            absolute_position = sorted_order[position - 1]
            step = order.index(absolute_position) + 1
            positions_all[position] += 1
            residues_all[residue] += 1
            steps_all[step] += 1
            positions_by_target[target][position] += 1
            residues_by_target[target][residue] += 1
            steps_by_target[target][step] += 1

    concentration_rows: List[Dict[str, Any]] = []
    for target in ["ALL", *sorted(target_names)]:
        position_counts = positions_all if target == "ALL" else positions_by_target[target]
        residue_counts = residues_all if target == "ALL" else residues_by_target[target]
        step_counts = steps_all if target == "ALL" else steps_by_target[target]
        total = sum(position_counts.values())
        concentration_rows.append(
            {
                "target_name": target,
                "methyl_sites": total,
                "site_position_counts": json.dumps(dict(sorted(position_counts.items()))),
                "site_residue_counts": json.dumps(dict(sorted(residue_counts.items()))),
                "sampling_step_counts": json.dumps(dict(sorted(step_counts.items()))),
                "dominant_position_1based": (
                    max(position_counts, key=position_counts.get) if position_counts else ""
                ),
                "maximum_single_position_share": (
                    max(position_counts.values()) / total if total else 0.0
                ),
                "maximum_single_residue_share": (
                    max(residue_counts.values()) / total if total else 0.0
                ),
                "maximum_single_sampling_step_share": (
                    max(step_counts.values()) / total if total else 0.0
                ),
            }
        )
    dominant_by_target = {
        str(row["target_name"]): int(row["dominant_position_1based"])
        for row in concentration_rows
        if str(row["target_name"]) in AV_FAMILY
        and str(row["dominant_position_1based"])
    }
    selected_chains = {
        str(row["target_name"]).upper(): str(row["selected_chain"])
        for row in target_manifest_rows
    }
    universal_av_dominant = (
        set(dominant_by_target) == AV_FAMILY
        and len(set(dominant_by_target.values())) == 1
    )
    if universal_av_dominant:
        av_alignment = av_family_alignment_audit(
            read_jsonl(native_path), selected_chains, dominant_by_target
        )
    else:
        av_alignment = {
            "quality_gate": "PASS",
            "method": "all_ca_pair_distance_matrix_cyclic_alignment_to_3AV9",
            "reason": (
                "The 3AV targets do not share one universal dominant absolute "
                "position in V7, so the former all-position-7 signature is absent."
            ),
            "dominant_position_by_target": dict(sorted(dominant_by_target.items())),
            "targets": [],
        }
    global_row = next(row for row in concentration_rows if row["target_name"] == "ALL")
    pass_2_checks = {
        "repeated_target_natural_sequences_have_exactly_one_annotation_payload": (
            not inconsistent_repeated
        ),
        "no_global_position_collapse_above_80_percent": (
            float(global_row["maximum_single_position_share"]) <= 0.80
        ),
        "no_global_residue_collapse_above_80_percent": (
            float(global_row["maximum_single_residue_share"]) <= 0.80
        ),
        "no_global_or_target_sampling_step_collapse_above_80_percent": all(
            float(row["maximum_single_sampling_step_share"]) <= 0.80
            for row in concentration_rows
            if int(row["methyl_sites"]) >= 30
        ),
        "former_3av_position_7_pattern_is_absent_or_maps_to_one_physical_homolog_site": (
            av_alignment["quality_gate"] == "PASS"
        ),
        "representation_audit_proves_mapping_back_before_averaging": (
            pinned_representation.get("quality_gate") == "PASS"
            and pinned_representation.get("protocol")
            == V7_REPRESENTATION_AUDIT_PROTOCOL
        ),
    }

    historical_exact, historical_natural = exact_and_natural_keys(
        read_csv(historical_path)
    )
    prior_exact, prior_natural = exact_and_natural_keys(read_csv(prior_path))
    eligible_exact, eligible_natural = exact_and_natural_keys(eligible_rows)
    candidates_by_target = Counter(
        str(row["target_name"]).upper() for row in eligible_rows
    )
    uncovered_targets = sorted(
        target for target in target_names if candidates_by_target[target] < 1
    )
    summary_index = {
        str(row["target_name"]).upper(): row for row in target_summary_rows
    }
    target_manifest_index = {
        str(row["target_name"]).upper(): row for row in target_manifest_rows
    }
    summaries_recompute = all(
        target in summary_index
        and int(summary_index[target]["novel_methylated_candidates"])
        == candidates_by_target[target]
        and int(summary_index[target]["has_signature_candidate"])
        == int(candidates_by_target[target] > 0)
        for target in target_names
    )
    target_manifests_recompute = all(
        target in target_manifest_index
        and int(target_manifest_index[target]["novel_methylated_candidates"])
        == candidates_by_target[target]
        and int(target_manifest_index[target]["formal_abstention"]) == 0
        for target in target_names
    )
    v7_parent = run_dir.parent
    prohibited_handoff_paths = [
        v7_parent / "handoff",
        v7_parent / "serine_qc_serine_only_cyclic_v7_shangge_handoff.zip",
    ]
    pass_3_checks = {
        "eligible_candidates_are_novel_against_historical_and_prior_pools": (
            not (eligible_exact & historical_exact)
            and not (eligible_natural & historical_natural)
            and not (eligible_exact & prior_exact)
            and not (eligible_natural & prior_natural)
        ),
        "all_17_targets_have_at_least_one_signature_candidate": not uncovered_targets,
        "formal_target_abstention_is_absent": (
            list(manifest.get("targets_formally_abstained", [])) == []
            and not (run_dir / "formal_target_abstention_audit.json").exists()
        ),
        "target_summaries_recompute": summaries_recompute,
        "target_manifests_recompute": target_manifests_recompute,
        "structure_handoff_is_not_created_before_manual_review": not any(
            path.exists() for path in prohibited_handoff_paths
        ),
        "permeability_remains_deferred": (
            manifest.get("permeability_status")
            == "DEFERRED_UNTIL_RETURNED_STRUCTURES_PASS_GATE"
            and int(manifest.get("permeability_input_rows", -1)) == 0
        ),
    }

    pass_1 = "PASS" if all(pass_1_checks.values()) else "FAIL"
    pass_2 = "PASS" if all(pass_2_checks.values()) else "FAIL"
    pass_3 = "PASS" if all(pass_3_checks.values()) else "FAIL"
    quality_gate = "PASS" if pass_1 == pass_2 == pass_3 == "PASS" else "FAIL"
    report = {
        "quality_gate": quality_gate,
        "release_status": (
            "READY_FOR_MANUAL_SCIENTIFIC_REVIEW_NO_STRUCTURE_HANDOFF"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_SEND_TO_SHANGGE"
        ),
        "protocol": "independent_three_pass_serine_only_v7_audit_v1",
        "scientific_conclusions": {
            "serine_provenance_scope": (
                "Only Ser was retrained; all non-Ser experts and shared tensors are "
                "required by the model manifest to remain parent-identical."
            ),
            "seventh_position_question": (
                "The ensemble maps each cyclic serialization back to physical "
                "residues. Global position collapse is gated, and any conserved 3AV "
                "site is independently mapped across homologous backbone geometry."
            ),
            "structural_similarity_policy": (
                "Nearest held-out CA-distance label is diagnostic, not methylation "
                "ground truth; actual returned candidate structures remain the final "
                "global/cyclic RMSD gate."
            ),
            "target_abstention_policy": "DISALLOWED; every target must have a candidate",
        },
        "pass_1_integrity": {
            "quality_gate": pass_1,
            "checks": pass_1_checks,
            "raw_rows": len(raw_rows),
            "unique_rows": len(unique_rows),
            "eligible_rows": len(eligible_rows),
            "row_error_count": len(row_errors),
            "row_error_examples": row_errors[:25],
            "aggregation_errors": aggregation_errors[:25],
            "eligible_errors": eligible_errors[:25],
        },
        "pass_2_result_annotation_and_position": {
            "quality_gate": pass_2,
            "checks": pass_2_checks,
            "repeated_target_natural_sequence_groups": len(repeated_groups),
            "inconsistent_repeated_groups": len(inconsistent_repeated),
            "concentration": concentration_rows,
            "av_family_physical_alignment": av_alignment,
        },
        "pass_3_novelty_coverage_workflow": {
            "quality_gate": pass_3,
            "checks": pass_3_checks,
            "uncovered_targets": uncovered_targets,
            "candidate_count_by_target": dict(sorted(candidates_by_target.items())),
            "historical_exact_overlaps": len(eligible_exact & historical_exact),
            "historical_natural_overlaps": len(eligible_natural & historical_natural),
            "prior_exact_overlaps": len(eligible_exact & prior_exact),
            "prior_natural_overlaps": len(eligible_natural & prior_natural),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    atomic_write_json(out_dir / "three_pass_generation_audit.json", report)
    atomic_write_json(out_dir / "av_family_physical_position_support.json", av_alignment)
    atomic_write_csv(
        out_dir / "three_pass_concentration_by_target.csv",
        concentration_rows,
        list(concentration_rows[0]),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--historical-designs-csv", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--prior-handoff-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    report = audit(
        Path(args.run_dir).resolve(),
        Path(args.plan).resolve(),
        Path(args.native_jsonl).resolve(),
        Path(args.historical_designs_csv).resolve(),
        Path(args.prior_handoff_csv).resolve(),
        out_dir,
    )
    print("===== INDEPENDENT SERINE-ONLY V7 THREE-PASS AUDIT =====", flush=True)
    print(f"Quality gate: {report['quality_gate']}", flush=True)
    print(f"Release status: {report['release_status']}", flush=True)
    print(f"Report: {out_dir / 'three_pass_generation_audit.json'}", flush=True)
    if report["quality_gate"] != "PASS":
        failed = [
            name
            for name in (
                "pass_1_integrity",
                "pass_2_result_annotation_and_position",
                "pass_3_novelty_coverage_workflow",
            )
            if report[name]["quality_gate"] != "PASS"
        ]
        raise RuntimeError("V7 three-pass audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
