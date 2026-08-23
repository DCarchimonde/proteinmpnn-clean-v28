#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select and independently audit the final V9 17 x 100 structure handoff.

This program is intentionally CPU-only and independent of model inference.  It
does not trust the generator's lowercase sequence, eligibility flags, or
summary counts.  Every probability vector and release decision is recomputed
from the persisted all-cyclic-start minima and maxima.  A handoff is written
only when all 17 targets contribute exactly 100 stable, novel, non-cyclic-
duplicate candidates and no target is dominated (>80%) by one physical methyl
position.  A failed target is reported and never padded.
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_stability_v9_1700.json")
THRESHOLD = 0.6
TEMPERATURE = 0.5
QUOTA = 100
MAX_POSITION_SHARE = 0.80
NATURAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
METHYLATABLE_AA = NATURAL_AA - {"P"}
VALID_TOKENS = NATURAL_AA | {token.lower() for token in METHYLATABLE_AA}
FROZEN_TARGETS = (
    "1SFI", "3AV9", "3AVA", "3AVB", "3AVF", "3AVG", "3AVH", "3AVI",
    "3AVJ", "3AVK", "3AVM", "3AVN", "3P8F", "3WNE", "3ZGC", "4K1E",
    "4KEL",
)
EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_"
    "cyclic_stability_worst_start_v9"
)
V11_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_cyclic_native_relative_positions_v11"
)
AUDIT_PROTOCOL = "cyclic_stability_worst_start_heldout_gate_v9"
AUDIT_AUTHORIZATION = "CYCLIC_STABILITY_V9_VALIDATED_FOR_UNIFORM_REGENERATION"
V11_AUDIT_PROTOCOL = "cyclic_native_relative_positions_heldout_gate_v11"
V11_AUDIT_AUTHORIZATION = (
    "CYCLIC_NATIVE_V11_VALIDATED_FOR_RMSD_PRIORITY_REGENERATION"
)
ANNOTATION_MODE = (
    "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
)
RANKING_POLICY = "representation_mean"
RELEASE_POLICY = "representation_min_strict_gt_threshold_zero_disagreement"
CYCLIC_BASE_PROTOCOL = (
    "receptor_visible_all_physical_starts_all_decoder_orders_exact_v9"
)
CYCLIC_BASE_FLOOR_POLICY = (
    "per_target_bottom_1pct_current_pool_outlier_filter_"
    "not_independent_calibration_v9"
)
RMSD_RANKER_PROTOCOL = "rmsd_priority_ranker_v10_six_target_loto_v1"
RMSD_SELECTION_OVERLAY = "rmsd_priority_first_with_evidence_aware_position_gate_v10"
RMSD_VALIDATED_TOP_FRACTION = 0.25
SERIALIZED_PROBABILITY_RECOMPUTE_ATOL = 1e-6
RMSD_DEVELOPMENT_SHA256 = (
    "d754c905e00d03c18ce0610b740c9bd6da09ee0a9e9d5d7ce953dc73d86aad05"
)
V10_POSITION_POLICY_SHA256 = (
    "28b41461138cd719dc0f8e0210e35071fbbb3c3ec7ad13a0f03bd12baff1744b"
)
KNOWN_OUTPUTS = (
    "1700_详细审计.csv",
    "1700_给尚哥_极简.csv",
    "1700_给尚哥_结构输入.fasta",
    "selection_summary_by_target.csv",
    "candidate_validation_problems.csv",
    "v9_1700_release_audit.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


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


def atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    existing = [out_dir / name for name in KNOWN_OUTPUTS if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "V9 release-audit output exists; use --overwrite for this isolated "
            "directory: " + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def parse_json_list(value: Any, field: str) -> List[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a JSON list")
    return parsed


def finite_probability_vector(value: Any, field: str, length: int) -> List[float]:
    parsed = parse_json_list(value, field)
    try:
        vector = [float(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains a non-numeric value") from exc
    if len(vector) != length:
        raise ValueError(f"{field} length {len(vector)} != sequence length {length}")
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in vector):
        raise ValueError(f"{field} contains a non-finite/out-of-range probability")
    return vector


def strict_rounded_pass(value: float, threshold: float = THRESHOLD) -> bool:
    return round(float(value), 8) > float(threshold)


def canonical_rotation(sequence: str) -> str:
    if not sequence:
        raise ValueError("Cannot canonicalize an empty cyclic sequence")
    return min(sequence[index:] + sequence[:index] for index in range(len(sequence)))


def methyl_positions(sequence: str) -> List[int]:
    return [index for index, token in enumerate(sequence, start=1) if token.islower()]


def first_present(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def bool_int(row: Mapping[str, Any], field: str) -> bool:
    try:
        return int(str(row.get(field, "")).strip()) == 1
    except ValueError:
        return False


def validate_candidate(
    source: Mapping[str, Any], expected_targets: set[str]
) -> Tuple[Dict[str, Any] | None, List[str]]:
    row = dict(source)
    errors: List[str] = []
    target = str(row.get("target_name", "")).strip().upper()
    candidate_id = str(row.get("candidate_id", "")).strip()
    sequence = str(row.get("design_seq", "")).strip()
    natural = sequence.upper()
    if target not in expected_targets:
        errors.append("target_not_in_frozen_17")
    if not candidate_id:
        errors.append("empty_candidate_id")
    if not sequence or not set(sequence) <= VALID_TOKENS:
        errors.append("invalid_or_empty_design_sequence")
    if sequence and natural and not set(natural) <= NATURAL_AA:
        errors.append("naturalized_sequence_has_noncanonical_token")
    if str(row.get("design_natural_seq", "")).strip().upper() != natural:
        errors.append("design_natural_seq_mismatch")
    native_sequence = str(row.get("native_seq", "")).strip().upper()
    if not native_sequence or not set(native_sequence) <= NATURAL_AA:
        errors.append("missing_or_invalid_native_sequence")
    elif natural == native_sequence:
        errors.append("candidate_equals_native_sequence")
    elif len(natural) == len(native_sequence) and canonical_rotation(natural) == canonical_rotation(native_sequence):
        errors.append("candidate_equals_native_forward_cyclic_identity")
    try:
        if float(row.get("temperature", "nan")) != TEMPERATURE:
            errors.append("temperature_not_0.5")
    except ValueError:
        errors.append("invalid_temperature")
    try:
        threshold = float(row.get("methyl_threshold", "nan"))
        if threshold != THRESHOLD:
            errors.append("methyl_threshold_not_0.6")
    except ValueError:
        threshold = THRESHOLD
        errors.append("invalid_methyl_threshold")

    if not sequence:
        return None, errors
    recomputed_methyl_positions = methyl_positions(sequence)
    recomputed_recovery = (
        sum(left == right for left, right in zip(native_sequence, natural))
        / len(natural)
        if native_sequence and len(native_sequence) == len(natural)
        else math.nan
    )
    try:
        if int(row.get("design_length", -1)) != len(sequence):
            errors.append("design_length_mismatch")
        if int(row.get("native_length", -1)) != len(native_sequence):
            errors.append("native_length_mismatch")
        if int(row.get("length_match", -1)) != int(
            len(sequence) == len(native_sequence)
        ):
            errors.append("length_match_gate_mismatch")
        if len(sequence) != len(native_sequence):
            errors.append("length_match_gate_not_one")
        if int(row.get("valid_token_gate", -1)) != 1:
            errors.append("valid_token_gate_not_one")
        if int(row.get("design_methyl_count", -1)) != len(
            recomputed_methyl_positions
        ):
            errors.append("design_methyl_count_mismatch")
        observed_methyl_rate = float(row.get("design_methyl_rate", "nan"))
        if not math.isfinite(observed_methyl_rate) or abs(
            observed_methyl_rate
            - len(recomputed_methyl_positions) / len(sequence)
        ) > 1e-12:
            errors.append("design_methyl_rate_mismatch")
        observed_recovery = float(row.get("natural_aa_recovery", "nan"))
        if (
            not math.isfinite(recomputed_recovery)
            or not math.isfinite(observed_recovery)
            or abs(observed_recovery - recomputed_recovery) > 1e-12
        ):
            errors.append("natural_aa_recovery_mismatch")
    except (TypeError, ValueError):
        errors.append("invalid_basic_sequence_diagnostic")
    try:
        means = finite_probability_vector(
            row.get("methyl_probabilities", ""), "methyl_probabilities", len(sequence)
        )
        minima = finite_probability_vector(
            row.get("methyl_probability_representation_min", ""),
            "methyl_probability_representation_min",
            len(sequence),
        )
        maxima = finite_probability_vector(
            row.get("methyl_probability_representation_max", ""),
            "methyl_probability_representation_max",
            len(sequence),
        )
        spans = finite_probability_vector(
            row.get("methyl_probability_representation_span", ""),
            "methyl_probability_representation_span",
            len(sequence),
        )
        representation_std = finite_probability_vector(
            row.get("methyl_probability_representation_std", ""),
            "methyl_probability_representation_std",
            len(sequence),
        )
        order_std = finite_probability_vector(
            row.get("methyl_probability_order_std", ""),
            "methyl_probability_order_std",
            len(sequence),
        )
        representation_by_start_raw = parse_json_list(
            row.get("methyl_probability_representation_by_start", ""),
            "methyl_probability_representation_by_start",
        )
        representation_by_start = [
            [float(value) for value in start_values]
            for start_values in representation_by_start_raw
        ]
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return None, errors

    if any(
        minimum > mean + SERIALIZED_PROBABILITY_RECOMPUTE_ATOL
        or mean > maximum + SERIALIZED_PROBABILITY_RECOMPUTE_ATOL
        for mean, minimum, maximum in zip(means, minima, maxima)
    ):
        errors.append("representation_min_mean_max_order_violation")
    if any(
        abs(span - (maximum - minimum)) > 1e-6
        for span, minimum, maximum in zip(spans, minima, maxima)
    ):
        errors.append("representation_span_does_not_equal_max_minus_min")
    if (
        len(representation_by_start) != len(sequence)
        or any(len(values) != len(sequence) for values in representation_by_start)
        or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for values in representation_by_start
            for value in values
        )
    ):
        errors.append("representation_by_start_matrix_is_not_finite_L_by_L")
    else:
        for position, values in enumerate(zip(*representation_by_start)):
            recomputed_mean = sum(values) / len(values)
            recomputed_min = min(values)
            recomputed_max = max(values)
            recomputed_std = math.sqrt(
                sum((value - recomputed_mean) ** 2 for value in values)
                / len(values)
            )
            if (
                abs(means[position] - recomputed_mean)
                > SERIALIZED_PROBABILITY_RECOMPUTE_ATOL
                or round(minima[position], 8) != round(recomputed_min, 8)
                or round(maxima[position], 8) != round(recomputed_max, 8)
                or abs(representation_std[position] - recomputed_std)
                > SERIALIZED_PROBABILITY_RECOMPUTE_ATOL
            ):
                errors.append("representation_by_start_summary_recompute_mismatch")
                break
    expected_positions = [
        position
        for position, (token, minimum) in enumerate(zip(natural, minima), start=1)
        if token in METHYLATABLE_AA and strict_rounded_pass(minimum, threshold)
    ]
    observed_positions = methyl_positions(sequence)
    disagreements = [
        position
        for position, (minimum, maximum) in enumerate(zip(minima, maxima), start=1)
        if not strict_rounded_pass(minimum, threshold)
        and strict_rounded_pass(maximum, threshold)
    ]
    if not observed_positions:
        errors.append("sequence_contains_no_methyl_token")
    if observed_positions != expected_positions:
        errors.append("lowercase_pattern_not_recomputed_from_representation_min")
    if disagreements:
        errors.append("cyclic_start_threshold_disagreement")
    if any(token == "p" for token in sequence):
        errors.append("lowercase_proline_is_forbidden")

    try:
        persisted_positions = [
            int(value)
            for value in parse_json_list(
                row.get("methyl_positions_1based", ""), "methyl_positions_1based"
            )
        ]
        if persisted_positions != observed_positions:
            errors.append("methyl_positions_1based_mismatch")
    except ValueError as exc:
        errors.append(str(exc))
    try:
        persisted_disagreements = [
            int(value)
            for value in parse_json_list(
                row.get("representation_threshold_disagreement_positions_1based", ""),
                "representation_threshold_disagreement_positions_1based",
            )
        ]
        if persisted_disagreements or persisted_disagreements != disagreements:
            errors.append("persisted_threshold_disagreement_mismatch")
    except ValueError as exc:
        errors.append(str(exc))
    try:
        if int(row.get("representation_threshold_disagreement_count", -1)) != 0:
            errors.append("persisted_threshold_disagreement_count_not_zero")
    except ValueError:
        errors.append("invalid_threshold_disagreement_count")

    required_one_flags = (
        "stable_cyclic_release_gate",
        "passes_methylation_hard_gate",
        "eligible_for_new_permeability_screen",
    )
    for field in required_one_flags:
        if not bool_int(row, field):
            errors.append(f"{field}_not_one")
    required_zero_flags = (
        "seen_in_historical_4115",
        "seen_in_historical_4115_exact",
        "seen_in_historical_4115_naturalized",
        "seen_in_prior_1333",
        "seen_in_prior_1333_exact",
        "seen_in_prior_1333_naturalized",
    )
    for field in required_zero_flags:
        if field not in row:
            errors.append(f"{field}_missing")
            continue
        try:
            if int(str(row.get(field, "0")).strip() or "0") != 0:
                errors.append(f"{field}_not_zero")
        except ValueError:
            errors.append(f"{field}_invalid")

    if str(row.get("annotation_mode", "")) != ANNOTATION_MODE:
        errors.append("annotation_mode_mismatch")
    if str(row.get("annotation_context_policy", "")) != (
        "peptide_chain_only_no_visible_receptor_chains"
    ):
        errors.append("annotation_context_policy_mismatch")
    try:
        if int(row.get("annotation_visible_receptor_chains", -1)) != 0:
            errors.append("annotation_visible_receptor_chains_not_zero")
    except ValueError:
        errors.append("invalid_annotation_visible_receptor_chains")
    if str(row.get("sampling_context_policy", "")) != (
        "native_complex_longest_receptor_visible"
    ):
        errors.append("sampling_context_policy_mismatch")
    if str(row.get("annotation_ranking_probability_policy", "")) != RANKING_POLICY:
        errors.append("ranking_probability_policy_mismatch")
    if str(row.get("annotation_release_probability_policy", "")) != RELEASE_POLICY:
        errors.append("release_probability_policy_mismatch")
    try:
        representation_size = int(row.get("annotation_representation_ensemble_size", -1))
        decoder_size = int(row.get("annotation_decoder_order_ensemble_size", -1))
        order_size = int(row.get("annotation_order_ensemble_size", -1))
        total_size = int(row.get("annotation_total_probability_ensemble_size", -1))
        if representation_size != len(sequence):
            errors.append("representation_ensemble_size_not_peptide_length")
        if decoder_size != len(sequence) or order_size != len(sequence):
            errors.append("decoder_order_ensemble_size_not_peptide_length")
        if total_size != len(sequence) * len(sequence):
            errors.append("total_probability_ensemble_size_not_L_squared")
    except ValueError:
        errors.append("invalid_probability_ensemble_size")

    if str(row.get("cyclic_base_score_protocol", "")) != CYCLIC_BASE_PROTOCOL:
        errors.append("cyclic_base_score_protocol_mismatch")
    if str(row.get("cyclic_base_floor_policy", "")) != CYCLIC_BASE_FLOOR_POLICY:
        errors.append("cyclic_base_floor_policy_mismatch")
    try:
        by_start = [
            float(value)
            for value in parse_json_list(
                row.get("cyclic_base_log_probability_by_start", ""),
                "cyclic_base_log_probability_by_start",
            )
        ]
        start_by_decoder_raw = parse_json_list(
            row.get("cyclic_base_log_probability_start_by_decoder_order", ""),
            "cyclic_base_log_probability_start_by_decoder_order",
        )
        start_by_decoder = [
            [float(value) for value in decoder_values]
            for decoder_values in start_by_decoder_raw
        ]
        base_mean_exact = float(row.get("cyclic_base_log_probability_mean", "nan"))
        base_min_exact = float(row.get("cyclic_base_log_probability_min", "nan"))
        base_max_exact = float(row.get("cyclic_base_log_probability_max", "nan"))
        base_span_exact = float(row.get("cyclic_base_log_probability_span", "nan"))
        base_std_exact = float(row.get("cyclic_base_log_probability_std", "nan"))
        base_floor_exact = float(row.get("cyclic_base_floor", "nan"))
        if len(by_start) != len(sequence) or not all(math.isfinite(value) for value in by_start):
            errors.append("cyclic_base_start_vector_invalid")
        if (
            len(start_by_decoder) != len(sequence)
            or any(len(values) != len(sequence) for values in start_by_decoder)
            or not all(
                math.isfinite(value)
                for values in start_by_decoder
                for value in values
            )
        ):
            errors.append("cyclic_base_start_by_decoder_matrix_invalid")
        elif any(
            abs(start_mean - sum(values) / len(values)) > 1e-6
            for start_mean, values in zip(by_start, start_by_decoder)
        ):
            errors.append("cyclic_base_start_by_decoder_recompute_mismatch")
        if not all(
            math.isfinite(value)
            for value in (
                base_mean_exact,
                base_min_exact,
                base_max_exact,
                base_span_exact,
                base_std_exact,
                base_floor_exact,
            )
        ):
            errors.append("cyclic_base_summary_nonfinite")
        elif by_start:
            recomputed_mean = sum(by_start) / len(by_start)
            recomputed_min = min(by_start)
            recomputed_max = max(by_start)
            recomputed_std = math.sqrt(
                sum((value - recomputed_mean) ** 2 for value in by_start)
                / len(by_start)
            )
            if (
                abs(base_mean_exact - recomputed_mean) > 1e-6
                or abs(base_min_exact - recomputed_min) > 1e-6
                or abs(base_max_exact - recomputed_max) > 1e-6
                or abs(base_span_exact - (recomputed_max - recomputed_min)) > 1e-6
                or abs(base_std_exact - recomputed_std) > 1e-6
            ):
                errors.append("cyclic_base_summary_recompute_mismatch")
            if round(base_mean_exact, 8) < base_floor_exact:
                errors.append("cyclic_base_mean_below_frozen_target_floor")
        if int(row.get("cyclic_base_physical_start_count", -1)) != len(sequence):
            errors.append("cyclic_base_physical_start_count_not_L")
        if int(row.get("cyclic_base_decoder_order_count_per_start", -1)) != len(sequence):
            errors.append("cyclic_base_decoder_order_count_not_L")
        if int(row.get("cyclic_base_total_ensemble_size", -1)) != len(sequence) ** 2:
            errors.append("cyclic_base_total_ensemble_size_not_L_squared")
        if int(row.get("cyclic_base_gate_pass", 0)) != 1:
            errors.append("cyclic_base_gate_pass_not_one")
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid_cyclic_base_evidence:{exc}")

    try:
        base_score = float(row.get("cyclic_base_log_probability_mean", "nan"))
        floor = float(row.get("methyl_site_representation_floor_min", "nan"))
        span = float(row.get("methyl_probability_representation_span_max", "nan"))
        if not math.isfinite(base_score):
            errors.append("nonfinite_base_log_probability_mean")
        if not math.isfinite(floor) or not strict_rounded_pass(floor, threshold):
            errors.append("methyl_site_representation_floor_not_strictly_above_threshold")
        if not math.isfinite(span) or span < 0.0 or span > 1.0:
            errors.append("invalid_representation_span")
        if observed_positions and abs(
            floor - min(minima[position - 1] for position in observed_positions)
        ) > 1e-6:
            errors.append("methyl_site_representation_floor_scalar_mismatch")
        if abs(span - max(spans)) > 1e-6:
            errors.append("representation_span_max_scalar_mismatch")
        representation_std_max = float(
            row.get("methyl_probability_representation_std_max", "nan")
        )
        order_std_max = float(row.get("methyl_probability_order_std_max", "nan"))
        if not math.isfinite(representation_std_max) or abs(
            representation_std_max - max(representation_std)
        ) > 1e-6:
            errors.append("representation_std_max_scalar_mismatch")
        if not math.isfinite(order_std_max) or abs(
            order_std_max - max(order_std)
        ) > 1e-6:
            errors.append("decoder_order_std_max_scalar_mismatch")
    except ValueError:
        base_score, floor, span = float("-inf"), float("-inf"), float("inf")
        errors.append("invalid_ranking_diagnostic")

    if errors:
        return None, errors
    eligible_positions = [
        position
        for position, token in enumerate(natural, start=1)
        if token in METHYLATABLE_AA
    ]
    if not eligible_positions:
        return None, [*errors, "no_methylatable_natural_position"]
    maximum_mean = max(round(means[position - 1], 8) for position in eligible_positions)
    maximum_minimum = max(
        round(minima[position - 1], 8) for position in eligible_positions
    )
    mean_argmax_ties = [
        position
        for position in eligible_positions
        if round(means[position - 1], 8) == maximum_mean
    ]
    minimum_argmax_ties = [
        position
        for position in eligible_positions
        if round(minima[position - 1], 8) == maximum_minimum
    ]
    ranking_mean_argmax = min(mean_argmax_ties)
    release_min_argmax = min(minimum_argmax_ties)
    row.update(
        {
            "target_name": target,
            "candidate_id": candidate_id,
            "design_seq": sequence,
            "design_natural_seq": natural,
            "design_length": len(sequence),
            "native_length": len(native_sequence),
            "length_match": int(len(sequence) == len(native_sequence)),
            "valid_token_gate": 1,
            "design_methyl_count": len(observed_positions),
            "design_methyl_rate": len(observed_positions) / len(sequence),
            "natural_aa_recovery": recomputed_recovery,
            "_methyl_positions": observed_positions,
            "_primary_methyl_position": release_min_argmax,
            "ranking_mean_argmax_position_1based": ranking_mean_argmax,
            "ranking_mean_argmax_tie_count": len(mean_argmax_ties),
            "release_min_argmax_position_1based": release_min_argmax,
            "release_min_argmax_tie_count": len(minimum_argmax_ties),
            "_base_score": base_score,
            "_release_floor": floor,
            "_representation_span": span,
            "_natural_cyclic_key": canonical_rotation(natural),
            "_marked_cyclic_key": canonical_rotation(sequence),
        }
    )
    return row, []


def exclusion_keys(paths: Sequence[Path]) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]]]:
    natural_keys: set[Tuple[str, str]] = set()
    cyclic_keys: set[Tuple[str, str]] = set()
    for path in paths:
        for row_number, row in enumerate(read_csv(path), start=2):
            target = first_present(row, ("target_name", "target", "pdb_id")).upper()
            sequence = first_present(
                row, ("design_seq", "sequence", "design_natural_seq", "fasta")
            ).upper()
            if not target or not sequence or not set(sequence) <= NATURAL_AA:
                raise RuntimeError(
                    f"Exclusion CSV has no valid target/sequence at {path}:{row_number}"
                )
            natural_keys.add((target, sequence))
            cyclic_keys.add((target, canonical_rotation(sequence)))
    return natural_keys, cyclic_keys


def quality_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    try:
        rmsd_lt5 = float(row.get("_rmsd_lt5_score", "nan"))
    except (TypeError, ValueError):
        rmsd_lt5 = math.nan
    if math.isfinite(rmsd_lt5):
        return (
            -rmsd_lt5,
            -float(row["_base_score"]),
            -float(row["_release_floor"]),
            float(row["_representation_span"]),
            str(row["candidate_id"]),
        )
    return (
        -float(row["_base_score"]),
        -float(row["_release_floor"]),
        float(row["_representation_span"]),
        str(row["candidate_id"]),
    )


def deduplicate_cyclic(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in sorted(rows, key=quality_key):
        key = str(row["_natural_cyclic_key"])
        best.setdefault(key, row)
    return sorted(best.values(), key=quality_key)


def projected_site_share(
    site_counts: Mapping[Any, int], total_sites: int, positions: Sequence[Any]
) -> float:
    projected = Counter(site_counts)
    projected.update(positions)
    denominator = total_sites + len(positions)
    return max(projected.values(), default=0) / denominator if denominator else 0.0


def projected_dominant(
    counts: Mapping[Any, int], values: Sequence[Any]
) -> Tuple[Any | None, float]:
    projected = Counter(counts)
    projected.update(values)
    denominator = sum(projected.values())
    if not projected or denominator <= 0:
        return None, 0.0
    maximum = max(projected.values())
    dominant = min(key for key, count in projected.items() if count == maximum)
    return dominant, maximum / denominator


def evidence_aware_position_pass(
    counts: Mapping[int, int], supported_positions: Sequence[int]
) -> bool:
    if not counts:
        return False
    maximum = max(counts.values())
    share = maximum / sum(counts.values())
    dominant_positions = {position for position, count in counts.items() if count == maximum}
    return share <= MAX_POSITION_SHARE or bool(
        dominant_positions & {int(value) for value in supported_positions}
    )


def select_diverse(
    rows: Sequence[Dict[str, Any]],
    quota: int,
    frontier_multiplier: int,
    supported_positions: Sequence[int] = (),
) -> List[Dict[str, Any]]:
    ranked = deduplicate_cyclic(rows)
    # Historical selectors searched only the top ``quota * multiplier`` rows.
    # That could falsely report a quota/concentration failure even when a valid
    # diverse solution existed later in the already-scored pool.  V9 release
    # selection must consider the complete deterministic pool; the retained
    # argument is metadata/backward CLI compatibility only.
    del frontier_multiplier
    rmsd_mode = bool(ranked) and all(
        math.isfinite(float(row.get("_rmsd_lt5_score", "nan"))) for row in ranked
    )
    frontier = (
        ranked[
            : max(
                quota,
                int(math.ceil(len(ranked) * RMSD_VALIDATED_TOP_FRACTION)),
            )
        ]
        if rmsd_mode
        else ranked
    )
    selected: List[Dict[str, Any]] = []
    site_counts: Counter[int] = Counter()
    primary_counts: Counter[int] = Counter()
    ranking_mean_argmax_counts: Counter[int] = Counter()
    release_min_argmax_counts: Counter[int] = Counter()
    methyl_residue_counts: Counter[str] = Counter()
    total_sites = 0
    remaining = list(frontier)
    rank_index = {id(row): index for index, row in enumerate(frontier)}
    supported = {int(value) for value in supported_positions}
    while remaining and len(selected) < quota:
        def selection_balance_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
            sequence = str(row["design_seq"])
            positions = list(row["_methyl_positions"])
            residues = [sequence[position - 1].upper() for position in positions]
            projected_sites = projected_site_share(
                site_counts, total_sites, positions
            )
            projected_residues = projected_site_share(
                methyl_residue_counts, total_sites, residues
            )
            projected_mean_argmax = projected_site_share(
                ranking_mean_argmax_counts,
                len(selected),
                [int(row["ranking_mean_argmax_position_1based"])],
            )
            projected_min_argmax = projected_site_share(
                release_min_argmax_counts,
                len(selected),
                [int(row["release_min_argmax_position_1based"])],
            )
            if rmsd_mode:
                projected_position_maps = (
                    (site_counts, positions),
                    (primary_counts, [int(row["_primary_methyl_position"])]),
                    (
                        ranking_mean_argmax_counts,
                        [int(row["ranking_mean_argmax_position_1based"])],
                    ),
                    (
                        release_min_argmax_counts,
                        [int(row["release_min_argmax_position_1based"])],
                    ),
                )
                position_violations = []
                for counts, new_values in projected_position_maps:
                    dominant, share = projected_dominant(counts, new_values)
                    position_violations.append(
                        max(0.0, share - MAX_POSITION_SHARE)
                        if dominant not in supported
                        else 0.0
                    )
                residue_excess = max(0.0, projected_residues - MAX_POSITION_SHARE)
                excesses = [*position_violations, residue_excess]
                # Scientific priority is the frozen RMSD rank.  It is only
                # overridden when needed to satisfy a non-exempt collapse gate.
                return (
                    sum(value > 0.0 for value in excesses),
                    max(excesses, default=0.0),
                    rank_index[id(row)],
                    max(
                        projected_sites,
                        projected_residues,
                        projected_mean_argmax,
                        projected_min_argmax,
                    ),
                )
            return (
                max(
                    projected_sites,
                    projected_residues,
                    projected_mean_argmax,
                    projected_min_argmax,
                ),
                projected_sites,
                projected_residues,
                projected_mean_argmax,
                projected_min_argmax,
                primary_counts[int(row["_primary_methyl_position"])],
                rank_index[id(row)],
            )

        chosen = min(
            remaining,
            key=selection_balance_key,
        )
        projected = projected_site_share(
            site_counts, total_sites, chosen["_methyl_positions"]
        )
        selected_row = dict(chosen)
        selected_row["selection_order"] = len(selected) + 1
        selected_row["selection_projected_maximum_site_share"] = projected
        selected.append(selected_row)
        site_counts.update(int(value) for value in chosen["_methyl_positions"])
        methyl_residue_counts.update(
            str(chosen["design_seq"])[int(position) - 1].upper()
            for position in chosen["_methyl_positions"]
        )
        total_sites += len(chosen["_methyl_positions"])
        primary_counts[int(chosen["_primary_methyl_position"])] += 1
        ranking_mean_argmax_counts[
            int(chosen["ranking_mean_argmax_position_1based"])
        ] += 1
        release_min_argmax_counts[
            int(chosen["release_min_argmax_position_1based"])
        ] += 1
        remaining.remove(chosen)
    return selected


def target_summary(
    target: str,
    valid_pool: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    quota: int,
    supported_positions: Sequence[int] = (),
) -> Dict[str, Any]:
    sites: Counter[int] = Counter()
    residues: Counter[str] = Counter()
    primary: Counter[int] = Counter()
    ranking_mean_argmax: Counter[int] = Counter()
    release_min_argmax: Counter[int] = Counter()
    for row in selected:
        sequence = str(row["design_seq"])
        sites.update(int(value) for value in row["_methyl_positions"])
        residues.update(sequence[position - 1].upper() for position in row["_methyl_positions"])
        primary[int(row["_primary_methyl_position"])] += 1
        ranking_mean_argmax[int(row["ranking_mean_argmax_position_1based"])] += 1
        release_min_argmax[int(row["release_min_argmax_position_1based"])] += 1
    total_sites = sum(sites.values())
    maximum_site_share = max(sites.values(), default=0) / total_sites if total_sites else 0.0
    maximum_primary_share = (
        max(primary.values(), default=0) / len(selected) if selected else 0.0
    )
    maximum_ranking_mean_argmax_share = (
        max(ranking_mean_argmax.values(), default=0) / len(selected)
        if selected
        else 0.0
    )
    maximum_release_min_argmax_share = (
        max(release_min_argmax.values(), default=0) / len(selected)
        if selected
        else 0.0
    )
    maximum_residue_share = (
        max(residues.values(), default=0) / total_sites if total_sites else 0.0
    )
    natural_sequences = [str(row["design_natural_seq"]) for row in selected]
    position_checks = {
        "all_methyl_sites": evidence_aware_position_pass(sites, supported_positions),
        "primary_position": evidence_aware_position_pass(primary, supported_positions),
        "ranking_mean_argmax": evidence_aware_position_pass(
            ranking_mean_argmax, supported_positions
        ),
        "release_min_argmax": evidence_aware_position_pass(
            release_min_argmax, supported_positions
        ),
    }
    valid_rmsd_scores = [
        float(row["_rmsd_lt5_score"])
        for row in valid_pool
        if math.isfinite(float(row.get("_rmsd_lt5_score", "nan")))
    ]
    selected_rmsd_scores = [
        float(row["_rmsd_lt5_score"])
        for row in selected
        if math.isfinite(float(row.get("_rmsd_lt5_score", "nan")))
    ]
    rmsd_ranked_pool = (
        sorted(valid_pool, key=quality_key) if selected_rmsd_scores else []
    )
    rmsd_validated_frontier_rows = (
        int(math.ceil(len(rmsd_ranked_pool) * RMSD_VALIDATED_TOP_FRACTION))
        if selected_rmsd_scores
        else 0
    )
    rmsd_validated_frontier_ids = {
        str(row["candidate_id"])
        for row in rmsd_ranked_pool[:rmsd_validated_frontier_rows]
    }
    pairwise_identities: List[float] = []
    for left_index, left in enumerate(natural_sequences):
        for right in natural_sequences[left_index + 1 :]:
            if len(left) == len(right) and left:
                pairwise_identities.append(
                    sum(a == b for a, b in zip(left, right)) / len(left)
                )
    return {
        "target_name": target,
        "valid_stable_novel_pool": len(valid_pool),
        "selected": len(selected),
        "required": quota,
        "exact_quota_pass": len(selected) == quota,
        "unique_marked_sequences": len({str(row["design_seq"]) for row in selected}),
        "unique_natural_sequences": len(set(natural_sequences)),
        "unique_forward_cyclic_natural_sequences": len(
            {str(row["_natural_cyclic_key"]) for row in selected}
        ),
        "methyl_sites": total_sites,
        "site_position_counts": json.dumps(dict(sorted(sites.items())), sort_keys=True),
        "primary_position_counts": json.dumps(dict(sorted(primary.items())), sort_keys=True),
        "ranking_mean_argmax_position_counts": json.dumps(
            dict(sorted(ranking_mean_argmax.items())), sort_keys=True
        ),
        "release_min_argmax_position_counts": json.dumps(
            dict(sorted(release_min_argmax.items())), sort_keys=True
        ),
        "methyl_residue_counts": json.dumps(dict(sorted(residues.items())), sort_keys=True),
        "maximum_single_position_share": maximum_site_share,
        "maximum_primary_position_share": maximum_primary_share,
        "maximum_ranking_mean_argmax_position_share": maximum_ranking_mean_argmax_share,
        "maximum_release_min_argmax_position_share": maximum_release_min_argmax_share,
        "maximum_single_methyl_residue_share": maximum_residue_share,
        "methyl_residue_concentration_pass": (
            bool(selected) and maximum_residue_share <= MAX_POSITION_SHARE
        ),
        "position_concentration_pass": (
            bool(selected) and all(position_checks.values())
        ),
        "position_concentration_policy": (
            "evidence_aware_historical_joint_lt5_support"
            if supported_positions
            else "strict_maximum_0.80_no_historical_exemption"
        ),
        "historically_supported_high_concentration_positions_1based": json.dumps(
            sorted({int(value) for value in supported_positions})
        ),
        "position_concentration_component_checks": json.dumps(
            position_checks, sort_keys=True
        ),
        "rmsd_priority_pool_score_mean": (
            sum(valid_rmsd_scores) / len(valid_rmsd_scores)
            if valid_rmsd_scores
            else ""
        ),
        "rmsd_priority_selected_score_min": (
            min(selected_rmsd_scores) if selected_rmsd_scores else ""
        ),
        "rmsd_priority_selected_score_mean": (
            sum(selected_rmsd_scores) / len(selected_rmsd_scores)
            if selected_rmsd_scores
            else ""
        ),
        "rmsd_priority_selected_score_max": (
            max(selected_rmsd_scores) if selected_rmsd_scores else ""
        ),
        "rmsd_priority_validated_top_fraction": (
            RMSD_VALIDATED_TOP_FRACTION if selected_rmsd_scores else ""
        ),
        "rmsd_priority_validated_frontier_rows": rmsd_validated_frontier_rows,
        "rmsd_priority_selected_all_within_validated_top_quartile": (
            bool(selected)
            and bool(selected_rmsd_scores)
            and all(
                str(row["candidate_id"]) in rmsd_validated_frontier_ids
                for row in selected
            )
        ),
        "maximum_pairwise_natural_identity": (
            max(pairwise_identities) if pairwise_identities else 0.0
        ),
        "mean_pairwise_natural_identity": (
            sum(pairwise_identities) / len(pairwise_identities)
            if pairwise_identities
            else 0.0
        ),
    }


def validate_upstream(
    plan_path: Path,
    plan: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
    audit_path: Path,
    audit: Mapping[str, Any],
    model_path: Path | None,
) -> Dict[str, bool]:
    plan_targets = [
        str(item.get("target_name", "")).upper()
        for item in plan.get("targets", [])
        if isinstance(item, Mapping)
    ]
    audit_checks = audit.get("quality_checks", {})
    generation_checks = manifest.get("quality_checks", {})
    heldout_disagreement = (
        audit.get("cyclic_representation_ensemble_heldout", {})
        .get("representation_threshold_disagreement_positions", -1)
    )
    observed_expert_protocol = str(
        manifest.get("model_expert_qc_protocol", "")
    )
    model_is_v11 = observed_expert_protocol == V11_EXPERT_PROTOCOL
    checks = {
        "plan_is_v9_17_target_t05_threshold_06": (
            str(plan.get("protocol", "")).startswith(
                (
                    "temperature_0.5_cyclic_stability_worst_start_v9_",
                    "temperature_0.5_cyclic_native_relative_positions_v11_",
                )
            )
            and float(plan.get("temperature", -1.0)) == TEMPERATURE
            and float(plan.get("methyl_threshold", -1.0)) == THRESHOLD
            and int(plan.get("expected_target_count", -1)) == 17
            and len(plan_targets) == 17
            and len(set(plan_targets)) == 17
            and set(plan_targets) == set(FROZEN_TARGETS)
            and list(plan.get("frozen_targets", [])) == []
            and int(plan.get("final_release_quota_per_target", -1)) == QUOTA
            and plan.get("sampling_context_policy")
            == "native_complex_longest_receptor_visible"
            and plan.get("annotation_context_policy")
            == "peptide_chain_only_no_visible_receptor_chains"
            and plan.get("annotation_ranking_probability_policy") == RANKING_POLICY
            and plan.get("annotation_release_probability_policy") == RELEASE_POLICY
        ),
        "generation_manifest_passes_all_checks": (
            manifest.get("quality_gate") == "PASS"
            and isinstance(generation_checks, Mapping)
            and bool(generation_checks)
            and all(bool(value) for value in generation_checks.values())
        ),
        "generation_protocol_matches_plan": manifest.get("protocol") == plan.get("protocol"),
        "generation_checkpoint_is_v9_worst_start_model": (
            observed_expert_protocol in {EXPERT_PROTOCOL, V11_EXPERT_PROTOCOL}
        ),
        "generation_annotation_stability_passes": (
            manifest.get("annotation_stability_audit", {}).get("quality_gate") == "PASS"
        ),
        "heldout_audit_is_authorized_v9": (
            audit.get("quality_gate") == "PASS"
            and audit.get("protocol")
            == (V11_AUDIT_PROTOCOL if model_is_v11 else AUDIT_PROTOCOL)
            and audit.get("release_authorization")
            == (
                V11_AUDIT_AUTHORIZATION
                if model_is_v11
                else AUDIT_AUTHORIZATION
            )
            and audit.get("model_expert_qc_protocol")
            == observed_expert_protocol
            and isinstance(audit_checks, Mapping)
            and bool(audit_checks)
            and all(bool(value) for value in audit_checks.values())
        ),
        "heldout_audit_matches_plan_bytes": audit.get("plan_sha256") == sha256_file(plan_path),
        "heldout_hard_calls_have_zero_start_disagreement": int(heldout_disagreement) == 0,
        "generation_and_audit_model_hash_match": (
            str(manifest.get("model_sha256", ""))
            and manifest.get("model_sha256") == audit.get("model_sha256")
        ),
    }
    if model_path is not None:
        checks["checkpoint_file_matches_both_manifests"] = (
            model_path.is_file()
            and sha256_file(model_path) == manifest.get("model_sha256")
            and sha256_file(model_path) == audit.get("model_sha256")
        )
    checks["upstream_paths_are_distinct_files"] = (
        plan_path.is_file()
        and manifest_path.is_file()
        and audit_path.is_file()
        and len({plan_path.resolve(), manifest_path.resolve(), audit_path.resolve()}) == 3
    )
    return checks


def clean_selected_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def verify_release_views(
    detail_path: Path,
    concise_path: Path,
    fasta_path: Path,
) -> Dict[str, bool]:
    detailed = read_csv(detail_path)
    concise = read_csv(concise_path)
    fasta_lines = [
        line.strip()
        for line in fasta_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    same_rows = len(detailed) == len(concise) == 1700
    if same_rows:
        same_rows = all(
            str(detail.get(field, "")) == str(short.get(field, ""))
            for detail, short in zip(detailed, concise)
            for field in (
                "final_release_id",
                "candidate_id",
                "target_name",
                "design_seq",
                "design_natural_seq",
                "methyl_positions_1based",
            )
        )
    fasta_matches = len(fasta_lines) == 3400
    if fasta_matches:
        for index, row in enumerate(detailed):
            expected_header = (
                f">{row['final_release_id']}|{row['target_name']}|"
                f"candidate={row['candidate_id']}|marked={row['design_seq']}|"
                f"methyl_positions={row['methyl_positions_1based']}"
            )
            if (
                fasta_lines[2 * index] != expected_header
                or fasta_lines[2 * index + 1] != row["design_natural_seq"]
            ):
                fasta_matches = False
                break
    return {
        "reopened_detailed_and_concise_views_match_exactly": same_rows,
        "reopened_fasta_ids_headers_and_sequences_match_detailed_csv": fasta_matches,
    }


def load_rmsd_priority_overlay(
    scored_csv_path: Path,
    ranker_manifest_path: Path,
    candidate_path: Path,
) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], Dict[str, List[int]], Dict[str, bool]]:
    """Load the optional V10 ranker through a path-specific hash contract."""

    manifest = read_json(ranker_manifest_path)
    rows = read_csv(scored_csv_path)
    inputs = manifest.get("inputs", {})
    artifacts = manifest.get("artifacts", {})
    development = inputs.get("development_csv", {}) if isinstance(inputs, Mapping) else {}
    candidates = inputs.get("candidate_csv", {}) if isinstance(inputs, Mapping) else {}
    scored_artifact = (
        artifacts.get("scored_candidates", {}) if isinstance(artifacts, Mapping) else {}
    )
    manifest_checks = manifest.get("quality_checks", {})
    checks = {
        "v10_rmsd_ranker_manifest_is_authorized_pass": (
            manifest.get("quality_gate") == "PASS"
            and manifest.get("release_status")
            == "AUTHORIZED_FOR_PRESTRUCTURE_PRIORITY_SELECTION"
            and manifest.get("protocol") == RMSD_RANKER_PROTOCOL
            and isinstance(manifest_checks, Mapping)
            and bool(manifest_checks)
            and all(value is True for value in manifest_checks.values())
        ),
        "v10_rmsd_development_is_frozen_476_bytes": (
            isinstance(development, Mapping)
            and development.get("sha256") == RMSD_DEVELOPMENT_SHA256
        ),
        "v10_rmsd_ranker_is_bound_to_exact_base_candidate_bytes": (
            isinstance(candidates, Mapping)
            and candidates.get("sha256") == sha256_file(candidate_path)
        ),
        "v10_rmsd_scored_csv_matches_its_named_manifest_artifact": (
            isinstance(scored_artifact, Mapping)
            and scored_artifact.get("sha256") == sha256_file(scored_csv_path)
        ),
    }
    overlay: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    row_errors = False
    for row in rows:
        key = (
            str(row.get("target_name", "")).strip().upper(),
            str(row.get("candidate_id", "")).strip(),
            str(row.get("design_natural_seq", "")).strip().upper(),
        )
        try:
            lt5 = float(row.get("rmsd_priority_score_joint_lt5", "nan"))
            lt3 = float(
                row.get("rmsd_priority_score_joint_lt3_descriptive", "nan")
            )
            features = parse_json_list(
                row.get("rmsd_priority_feature_vector", ""),
                "rmsd_priority_feature_vector",
            )
            feature_values = [float(value) for value in features]
            valid = (
                all(key)
                and key not in overlay
                and math.isfinite(lt5)
                and math.isfinite(lt3)
                and 0.0 <= lt5 <= 1.0
                and 0.0 <= lt3 <= 1.0
                and len(feature_values) == 16
                and all(math.isfinite(value) for value in feature_values)
                and str(row.get("rmsd_priority_protocol", ""))
                == RMSD_RANKER_PROTOCOL
            )
        except (TypeError, ValueError):
            valid = False
            lt5 = lt3 = math.nan
            feature_values = []
        if not valid:
            row_errors = True
            continue
        overlay[key] = {
            "rmsd_priority_protocol": RMSD_RANKER_PROTOCOL,
            "rmsd_priority_primary_endpoint": str(
                row.get("rmsd_priority_primary_endpoint", "")
            ),
            "rmsd_priority_score_joint_lt5": lt5,
            "rmsd_priority_score_joint_lt3_descriptive": lt3,
            "rmsd_priority_rank_within_target": int(
                row.get("rmsd_priority_rank_within_target", -1)
            ),
            "rmsd_priority_feature_vector": json.dumps(
                feature_values, separators=(",", ":")
            ),
            "rmsd_priority_warning": str(row.get("rmsd_priority_warning", "")),
            "_rmsd_lt5_score": lt5,
            "_rmsd_lt3_score": lt3,
        }
    checks["v10_rmsd_scored_rows_are_unique_finite_and_protocol_exact"] = (
        bool(overlay) and not row_errors and len(overlay) == len(rows)
    )
    support_payload = manifest.get("historical_site_support", {})
    supported_positions: Dict[str, List[int]] = {}
    if isinstance(support_payload, Mapping):
        for target, evidence in support_payload.items():
            if not isinstance(evidence, Mapping):
                continue
            try:
                supported_positions[str(target).upper()] = sorted(
                    {
                        int(value)
                        for value in evidence.get(
                            "supported_high_concentration_positions_1based", []
                        )
                    }
                )
            except (TypeError, ValueError):
                checks["v10_historical_position_support_is_well_formed"] = False
                break
    checks.setdefault(
        "v10_historical_position_support_is_well_formed",
        set(supported_positions) == {"1SFI", "3P8F", "3WNE", "3ZGC", "4K1E", "4KEL"}
        and all(supported_positions.values()),
    )
    return overlay, supported_positions, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--heldout-audit", required=True)
    parser.add_argument("--cyclic-base-manifest", required=True)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--model", required=True)
    parser.add_argument("--exclusion-csv", action="append", default=[])
    parser.add_argument("--rmsd-priority-csv", default="")
    parser.add_argument("--rmsd-priority-manifest", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quota-per-target", type=int, default=QUOTA)
    parser.add_argument(
        "--diversity-frontier-multiplier",
        type=int,
        default=10,
        help=(
            "Deprecated compatibility value; V9 always searches the complete "
            "validated scored pool to avoid false quota failures."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.quota_per_target != QUOTA:
        raise ValueError("The final V9 handoff is frozen to exactly 100 per target")
    if args.diversity_frontier_multiplier <= 0:
        raise ValueError("--diversity-frontier-multiplier must be positive")

    candidates_path = Path(args.candidates).resolve()
    manifest_path = Path(args.generation_manifest).resolve()
    audit_path = Path(args.heldout_audit).resolve()
    cyclic_base_manifest_path = Path(args.cyclic_base_manifest).resolve()
    plan_path = Path(args.plan).resolve()
    model_path = Path(args.model).resolve()
    exclusion_paths = [Path(value).resolve() for value in args.exclusion_csv]
    if bool(args.rmsd_priority_csv) != bool(args.rmsd_priority_manifest):
        raise ValueError(
            "--rmsd-priority-csv and --rmsd-priority-manifest must be supplied together"
        )
    risk_csv_path = (
        Path(args.rmsd_priority_csv).resolve() if args.rmsd_priority_csv else None
    )
    risk_manifest_path = (
        Path(args.rmsd_priority_manifest).resolve()
        if args.rmsd_priority_manifest
        else None
    )
    rmsd_mode = risk_csv_path is not None
    out_dir = Path(args.out_dir).resolve()
    prepare_output(out_dir, args.overwrite)

    plan = read_json(plan_path)
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    cyclic_base_manifest = read_json(cyclic_base_manifest_path)
    targets = [str(item["target_name"]).upper() for item in plan.get("targets", [])]
    expected_targets = set(targets)
    upstream_checks = validate_upstream(
        plan_path, plan, manifest_path, manifest, audit_path, audit, model_path
    )
    cyclic_target_summary = cyclic_base_manifest.get("target_summary", [])
    cyclic_floor_by_target = {
        str(row.get("target_name", "")).upper(): float(row["cyclic_base_floor"])
        for row in cyclic_target_summary
        if isinstance(row, Mapping) and "cyclic_base_floor" in row
    }
    passing_artifact = (
        cyclic_base_manifest.get("artifacts", {}).get("passing_candidates", {})
    )
    cyclic_inputs = cyclic_base_manifest.get("inputs", {})
    upstream_checks["exact_cyclic_base_manifest_is_hash_bound_and_complete"] = (
        cyclic_base_manifest.get("quality_gate") == "PASS"
        and cyclic_base_manifest.get("protocol") == CYCLIC_BASE_PROTOCOL
        and cyclic_base_manifest.get("floor_policy") == CYCLIC_BASE_FLOOR_POLICY
        and cyclic_base_manifest.get("model_sha256") == sha256_file(model_path)
        and cyclic_base_manifest.get("plan_sha256") == sha256_file(plan_path)
        and str(passing_artifact.get("sha256", "")) == sha256_file(candidates_path)
        and cyclic_inputs.get("generation_manifest", {}).get("sha256")
        == sha256_file(manifest_path)
        and cyclic_inputs.get("candidate_csv", {}).get("sha256")
        == manifest.get("methylated_new_candidates_csv_sha256")
        and cyclic_inputs.get("baseline_csv", {}).get("sha256")
        == manifest.get("unique_candidates_csv_sha256")
        and len(cyclic_target_summary) == 17
        and len(cyclic_floor_by_target) == 17
        and set(cyclic_floor_by_target) == set(FROZEN_TARGETS)
        and all(math.isfinite(value) for value in cyclic_floor_by_target.values())
        and all(
            int(row.get("baseline_unique_natural_sequences", 0)) > 0
            and int(row.get("nearest_rank", 0))
            == max(
                1,
                math.ceil(
                    float(row.get("floor_fraction", -1.0))
                    * int(row.get("baseline_unique_natural_sequences", 0))
                ),
            )
            and float(row.get("floor_fraction", -1.0)) == 0.01
            for row in cyclic_target_summary
            if isinstance(row, Mapping)
        )
    )
    exclusion_natural, exclusion_cyclic = exclusion_keys(exclusion_paths)
    exclusion_hashes = {sha256_file(path) for path in exclusion_paths}
    required_exclusion_hashes = {
        str(manifest.get("historical_design_csv_sha256", "")),
        str(manifest.get("prior_handoff_csv_sha256", "")),
    }
    upstream_checks["historical_and_prior_exclusion_files_match_generation_bytes"] = (
        len(exclusion_paths) >= 2
        and "" not in required_exclusion_hashes
        and required_exclusion_hashes <= exclusion_hashes
    )

    risk_overlay: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    supported_positions_by_target: Dict[str, List[int]] = {}
    if risk_csv_path is not None and risk_manifest_path is not None:
        risk_overlay, supported_positions_by_target, risk_checks = (
            load_rmsd_priority_overlay(
                risk_csv_path, risk_manifest_path, candidates_path
            )
        )
        upstream_checks.update(risk_checks)
        upstream_checks["generation_used_frozen_v10_evidence_aware_position_policy"] = (
            manifest.get("position_concentration_policy_sha256")
            == V10_POSITION_POLICY_SHA256
            and manifest.get("annotation_stability_audit", {}).get(
                "concentration_gate_policy"
            )
            == "historical_joint_lt5_supported_position_concentration_v10"
        )
        generation_policy = manifest.get("annotation_stability_audit", {}).get(
            "position_concentration_policy", {}
        )
        generation_support = (
            generation_policy.get("supported_positions_1based_by_target", {})
            if isinstance(generation_policy, Mapping)
            else {}
        )
        normalized_generation_support = {
            str(target).upper(): sorted(int(value) for value in values)
            for target, values in generation_support.items()
        }
        upstream_checks["generation_and_ranker_position_support_are_identical"] = (
            normalized_generation_support == supported_positions_by_target
        )

    problems: List[Dict[str, Any]] = []
    valid_by_target: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    observed_candidate_ids: set[str] = set()
    input_rows = read_csv(candidates_path)
    for row_number, source in enumerate(input_rows, start=2):
        source_with_overlay: Dict[str, Any] = dict(source)
        if rmsd_mode:
            risk_key = (
                str(source.get("target_name", "")).strip().upper(),
                str(source.get("candidate_id", "")).strip(),
                str(source.get("design_natural_seq", "")).strip().upper(),
            )
            overlay = risk_overlay.get(risk_key)
            if overlay is None:
                source_with_overlay["_missing_rmsd_overlay"] = 1
            else:
                source_with_overlay.update(overlay)
        candidate, errors = validate_candidate(source_with_overlay, expected_targets)
        target = str(source.get("target_name", "")).strip().upper()
        candidate_id = str(source.get("candidate_id", "")).strip()
        sequence = str(source.get("design_seq", "")).strip()
        if candidate_id in observed_candidate_ids:
            errors.append("duplicate_candidate_id")
        observed_candidate_ids.add(candidate_id)
        if rmsd_mode and source_with_overlay.get("_missing_rmsd_overlay"):
            errors.append("candidate_missing_from_v10_rmsd_priority_overlay")
        if candidate is not None:
            natural_key = (target, str(candidate["design_natural_seq"]))
            cyclic_key = (target, str(candidate["_natural_cyclic_key"]))
            if natural_key in exclusion_natural:
                errors.append("found_in_external_exclusion_csv_after_naturalization")
            if cyclic_key in exclusion_cyclic:
                errors.append("found_in_external_exclusion_csv_after_forward_cyclic_canonicalization")
            expected_floor = cyclic_floor_by_target.get(target)
            try:
                observed_floor = float(candidate.get("cyclic_base_floor", "nan"))
                if (
                    expected_floor is None
                    or not math.isfinite(observed_floor)
                    or abs(observed_floor - float(expected_floor)) > 1e-8
                ):
                    errors.append("cyclic_base_floor_does_not_match_frozen_manifest")
            except ValueError:
                errors.append("invalid_cyclic_base_floor")
        if errors or candidate is None:
            problems.append(
                {
                    "csv_row": row_number,
                    "target_name": target,
                    "candidate_id": candidate_id,
                    "design_seq": sequence,
                    "problems": ";".join(sorted(set(errors))),
                }
            )
            continue
        valid_by_target[target].append(candidate)

    selected_by_target: Dict[str, List[Dict[str, Any]]] = {}
    selection_pool_by_target: Dict[str, List[Dict[str, Any]]] = {}
    global_natural_keys: set[str] = set()
    global_cyclic_keys: set[str] = set()
    # Resolve the most constrained target first so global cross-receptor
    # duplicate blocking cannot silently starve it after a more abundant target.
    target_selection_order = sorted(
        targets,
        key=lambda target: (
            len(deduplicate_cyclic(valid_by_target.get(target, []))),
            target,
        ),
    )
    for target in target_selection_order:
        pool = deduplicate_cyclic(
            [
                row
                for row in valid_by_target.get(target, [])
                if str(row["design_natural_seq"]) not in global_natural_keys
                and str(row["_natural_cyclic_key"]) not in global_cyclic_keys
            ]
        )
        selection_pool_by_target[target] = pool
        selected = select_diverse(
            pool,
            args.quota_per_target,
            args.diversity_frontier_multiplier,
            supported_positions_by_target.get(target, ()) if rmsd_mode else (),
        )
        selected_by_target[target] = selected
        global_natural_keys.update(str(row["design_natural_seq"]) for row in selected)
        global_cyclic_keys.update(str(row["_natural_cyclic_key"]) for row in selected)

    summaries: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    for target in targets:
        selected = selected_by_target.get(target, [])
        summary = target_summary(
            target,
            selection_pool_by_target.get(target, []),
            selected,
            args.quota_per_target,
            supported_positions_by_target.get(target, ()) if rmsd_mode else (),
        )
        summary["pool_after_global_cross_target_dedup"] = len(
            selection_pool_by_target.get(target, [])
        )
        summary["cyclic_unique_pool_used_for_rmsd_top_quartile"] = len(
            selection_pool_by_target.get(target, [])
        )
        summary["valid_pool_before_global_cross_target_dedup"] = len(
            valid_by_target.get(target, [])
        )
        summaries.append(summary)
        for row in selected:
            enriched = clean_selected_row(row)
            release_prefix = "v10" if rmsd_mode else "v9"
            enriched["final_release_id"] = (
                f"{release_prefix}_{target.lower()}_{len(selected_rows) + 1:04d}"
            )
            selected_rows.append(enriched)

    global_residues: Counter[str] = Counter()
    for row in selected_rows:
        sequence = str(row["design_seq"])
        global_residues.update(token.upper() for token in sequence if token.islower())
    global_sites = sum(global_residues.values())
    maximum_residue_share = (
        max(global_residues.values(), default=0) / global_sites if global_sites else 0.0
    )
    release_checks = {
        **upstream_checks,
        "candidate_input_is_nonempty": bool(input_rows),
        "candidate_rows_have_no_independent_validation_error": not problems,
        "candidate_targets_are_exact_frozen_17": set(valid_by_target) == expected_targets,
        "every_target_has_exactly_100_selected": all(
            int(row["selected"]) == args.quota_per_target for row in summaries
        ),
        "selected_total_is_exactly_1700": len(selected_rows) == 1700,
        "selected_sequences_are_exact_natural_and_cyclic_unique_within_target": all(
            int(row["selected"])
            == int(row["unique_marked_sequences"])
            == int(row["unique_natural_sequences"])
            == int(row["unique_forward_cyclic_natural_sequences"])
            for row in summaries
        ),
        "selected_natural_and_forward_cyclic_sequences_are_unique_across_all_targets": (
            len(selected_rows)
            == len({str(row["design_natural_seq"]) for row in selected_rows})
            == len({canonical_rotation(str(row["design_natural_seq"])) for row in selected_rows})
        ),
        "target_methyl_position_concentration_passes_frozen_policy": all(
            bool(row["position_concentration_pass"]) for row in summaries
        ),
        "no_target_methyl_residue_concentration_exceeds_80_percent": all(
            bool(row["methyl_residue_concentration_pass"]) for row in summaries
        ),
        "global_methyl_residue_concentration_does_not_exceed_80_percent": (
            bool(global_sites) and maximum_residue_share <= MAX_POSITION_SHARE
        ),
        "external_exclusion_files_were_explicitly_loaded": len(exclusion_paths) >= 2,
    }
    if rmsd_mode:
        release_checks[
            "every_target_retains_at_least_400_cyclic_unique_rows_after_global_dedup"
        ] = all(
            int(row["cyclic_unique_pool_used_for_rmsd_top_quartile"]) >= 400
            for row in summaries
        )
        release_checks[
            "selected_fraction_of_each_final_cyclic_unique_pool_is_at_most_25_percent"
        ] = all(
            int(row["selected"])
            / int(row["cyclic_unique_pool_used_for_rmsd_top_quartile"])
            <= RMSD_VALIDATED_TOP_FRACTION
            for row in summaries
            if int(row["cyclic_unique_pool_used_for_rmsd_top_quartile"]) > 0
        ) and len(summaries) == len(targets)
        release_checks[
            "all_selected_rows_are_within_the_validated_top_quartile"
        ] = all(
            bool(row["rmsd_priority_selected_all_within_validated_top_quartile"])
            for row in summaries
        )
        release_checks["every_base_candidate_has_exactly_one_v10_rmsd_score"] = (
            len(risk_overlay) == len(input_rows)
        )
        release_checks["selected_rows_retain_finite_v10_rmsd_priority_scores"] = all(
            math.isfinite(float(row.get("rmsd_priority_score_joint_lt5", "nan")))
            and math.isfinite(
                float(row.get("rmsd_priority_score_joint_lt3_descriptive", "nan"))
            )
            for row in selected_rows
        )
    quality_gate = "PASS" if all(release_checks.values()) else "FAIL"

    summary_fields = list(summaries[0]) if summaries else ["target_name"]
    summary_path = out_dir / "selection_summary_by_target.csv"
    problems_path = out_dir / "candidate_validation_problems.csv"
    atomic_write_csv(summary_path, summaries, summary_fields)
    atomic_write_csv(
        problems_path,
        problems,
        ["csv_row", "target_name", "candidate_id", "design_seq", "problems"],
    )

    release_paths: Dict[str, Dict[str, Any]] = {
        "selection_summary_by_target": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "candidate_validation_problems": {
            "path": str(problems_path),
            "sha256": sha256_file(problems_path),
        },
    }
    if quality_gate == "PASS":
        detailed_rows = sorted(
            selected_rows,
            key=lambda row: (str(row["target_name"]), int(row["selection_order"])),
        )
        detail_path = out_dir / "1700_详细审计.csv"
        concise_path = out_dir / "1700_给尚哥_极简.csv"
        fasta_path = out_dir / "1700_给尚哥_结构输入.fasta"
        atomic_write_csv(detail_path, detailed_rows, union_fields(detailed_rows))
        concise_rows = [
            {
                "final_release_id": row["final_release_id"],
                "candidate_id": row["candidate_id"],
                "target_name": row["target_name"],
                "design_seq": row["design_seq"],
                "design_natural_seq": row["design_natural_seq"],
                "methyl_positions_1based": row["methyl_positions_1based"],
            }
            for row in detailed_rows
        ]
        atomic_write_csv(
            concise_path,
            concise_rows,
            [
                "final_release_id",
                "candidate_id",
                "target_name",
                "design_seq",
                "design_natural_seq",
                "methyl_positions_1based",
            ],
        )
        fasta_lines: List[str] = []
        for row in detailed_rows:
            fasta_lines.append(
                f">{row['final_release_id']}|{row['target_name']}|"
                f"candidate={row['candidate_id']}|marked={row['design_seq']}|"
                f"methyl_positions={row['methyl_positions_1based']}"
            )
            fasta_lines.append(str(row["design_natural_seq"]))
        atomic_write_text(fasta_path, "\n".join(fasta_lines) + "\n")
        release_checks.update(
            verify_release_views(detail_path, concise_path, fasta_path)
        )
        quality_gate = "PASS" if all(release_checks.values()) else "FAIL"
        if quality_gate != "PASS":
            for path in (detail_path, concise_path, fasta_path):
                path.unlink()
        for label, path in (
            ("detailed_audit", detail_path),
            ("shangge_concise", concise_path),
            ("shangge_fasta", fasta_path),
        ):
            if path.is_file():
                release_paths[label] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }

    report = {
        "quality_gate": quality_gate,
        "release_status": (
            "AUTHORIZED_EXACT_17_X_100_STRUCTURE_HANDOFF"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_SEND_TO_SHANGGE"
        ),
        "protocol": "independent_v9_cyclic_stability_17x100_release_audit_v1",
        "selection_overlay": RMSD_SELECTION_OVERLAY if rmsd_mode else "none",
        "rmsd_priority_is_prospective_prediction_not_observed_structure": rmsd_mode,
        "threshold": THRESHOLD,
        "temperature": TEMPERATURE,
        "quota_per_target": args.quota_per_target,
        "expected_targets": targets,
        "quality_checks": release_checks,
        "input_rows": len(input_rows),
        "independently_valid_rows": sum(len(rows) for rows in valid_by_target.values()),
        "problem_rows": len(problems),
        "selected_rows": len(selected_rows) if quality_gate == "PASS" else 0,
        "global_methyl_sites": global_sites,
        "global_methyl_residue_counts": dict(sorted(global_residues.items())),
        "maximum_single_methyl_residue_share": maximum_residue_share,
        "target_summary": summaries,
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "inputs": {
            "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            "generation_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "heldout_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
            "cyclic_base_manifest": {
                "path": str(cyclic_base_manifest_path),
                "sha256": sha256_file(cyclic_base_manifest_path),
            },
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "exclusion_csvs": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in exclusion_paths
            ],
            **(
                {
                    "rmsd_priority_csv": {
                        "path": str(risk_csv_path),
                        "sha256": sha256_file(risk_csv_path),
                    },
                    "rmsd_priority_manifest": {
                        "path": str(risk_manifest_path),
                        "sha256": sha256_file(risk_manifest_path),
                    },
                }
                if risk_csv_path is not None and risk_manifest_path is not None
                else {}
            ),
        },
        "release_artifacts": release_paths,
    }
    atomic_write_json(out_dir / "v9_1700_release_audit.json", report)
    print("===== V9 17 x 100 INDEPENDENT RELEASE AUDIT =====", flush=True)
    print(f"Input rows: {len(input_rows)}", flush=True)
    print(f"Independent problem rows: {len(problems)}", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in release_checks.items() if not passed]
        raise RuntimeError(
            "V9 1700 handoff is blocked; failed checks: " + ", ".join(failed)
        )
    print(f"Released rows: {len(selected_rows)} (17 x 100)", flush=True)
    print(f"Shang-ge concise CSV: {out_dir / '1700_给尚哥_极简.csv'}", flush=True)


if __name__ == "__main__":
    main()
