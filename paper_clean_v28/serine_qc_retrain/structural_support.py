#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provenance-backed interpretation of concentrated methyl-site predictions.

The failed 3AV* targets are not ten unrelated peptides.  They are a homologous
family of eight-residue cyclic backbones.  A deterministic structure-aware
classifier can therefore select the same cyclic position for many candidates
without any decoder-order bug.  This module distinguishes that situation from
an unsupported numerical collapse by comparing the target C-alpha distance
matrix against the pinned, provenance-corrected train/test structures.

Only forward cyclic shifts are considered.  Chain reversal is deliberately
forbidden because it changes peptide direction.  A concentrated target passes
only when its dominant position is closer to a methyl-positive position in the
held-out test set than to every natural-negative test position by a fixed
margin, and the positive match is within a conservative distance-matrix RMSE.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


DEFAULT_CONCENTRATION_SHARE = 0.80
# Structural support is a property of the frozen target backbone, not of the
# number of sampled candidates.  Audit every target with at least one eligible
# methyl site so a quota-sized target such as 3AVB cannot escape the evidence
# check merely because its frozen structure quota is below the old n>=30
# concentration-reporting threshold.
DEFAULT_MINIMUM_SITES = 1
DEFAULT_MAXIMUM_POSITIVE_RMSE_ANGSTROM = 0.35
DEFAULT_MINIMUM_POSITIVE_VS_NEGATIVE_MARGIN_ANGSTROM = 0.05


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def record_name(record: Mapping[str, Any], fallback: int) -> str:
    return str(
        record.get("name")
        or record.get("pdb")
        or record.get("pdb_id")
        or record.get("id")
        or f"record_{fallback}"
    ).upper()


def _coordinate_triplet(value: object) -> Tuple[float, float, float]:
    current = value
    if (
        isinstance(current, list)
        and len(current) == 1
        and isinstance(current[0], list)
    ):
        current = current[0]
    if not isinstance(current, list) or len(current) != 3:
        raise ValueError(f"Invalid coordinate triplet: {value!r}")
    output = tuple(float(component) for component in current)
    if not all(math.isfinite(component) for component in output):
        raise ValueError(f"Non-finite coordinate triplet: {value!r}")
    return output  # type: ignore[return-value]


def ca_coordinates(record: Mapping[str, Any], chain_id: str) -> List[Tuple[float, float, float]]:
    key = f"CA_chain_{chain_id}"
    if key not in record:
        raise ValueError(f"Missing {key} in {record_name(record, 0)}")
    values = record[key]
    if not isinstance(values, list) or not values:
        raise ValueError(f"Empty {key} in {record_name(record, 0)}")
    return [_coordinate_triplet(value) for value in values]


def distance_matrix(
    coordinates: Sequence[Tuple[float, float, float]],
) -> List[List[float]]:
    matrix: List[List[float]] = []
    for left in coordinates:
        row: List[float] = []
        for right in coordinates:
            row.append(
                math.sqrt(
                    (left[0] - right[0]) ** 2
                    + (left[1] - right[1]) ** 2
                    + (left[2] - right[2]) ** 2
                )
            )
        matrix.append(row)
    return matrix


def cyclic_distance_matrix_rmse(
    target: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    shift: int,
) -> float:
    length = len(target)
    if length == 0 or len(reference) != length:
        raise ValueError("Distance matrices must be non-empty and have equal size")
    total = 0.0
    for target_row in range(length):
        reference_row = (target_row - shift) % length
        if len(target[target_row]) != length or len(reference[reference_row]) != length:
            raise ValueError("Distance matrix is not square")
        for target_column in range(length):
            reference_column = (target_column - shift) % length
            delta = (
                float(target[target_row][target_column])
                - float(reference[reference_row][reference_column])
            )
            total += delta * delta
    return math.sqrt(total / (length * length))


def _single_chain(record: Mapping[str, Any]) -> str:
    chain_ids = sorted(
        key[len("seq_chain_") :]
        for key, value in record.items()
        if key.startswith("seq_chain_") and str(value)
    )
    if len(chain_ids) != 1:
        raise ValueError(
            f"Reference {record_name(record, 0)} is not a single peptide chain"
        )
    return chain_ids[0]


def _reference_index(
    rows: Sequence[Mapping[str, Any]], split: str
) -> MutableMapping[int, List[Dict[str, Any]]]:
    result: MutableMapping[int, List[Dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(rows):
        chain_id = _single_chain(record)
        sequence = str(record[f"seq_chain_{chain_id}"])
        coordinates = ca_coordinates(record, chain_id)
        if len(sequence) != len(coordinates):
            raise ValueError(
                f"Sequence/coordinate mismatch in {record_name(record, index)}"
            )
        result[len(sequence)].append(
            {
                "split": split,
                "sample_name": record_name(record, index),
                "sequence": sequence,
                "distance_matrix": distance_matrix(coordinates),
            }
        )
    return result


def _nearest_examples(
    target_matrix: Sequence[Sequence[float]],
    target_position_1based: int,
    references: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    length = len(target_matrix)
    positive: Dict[str, Any] | None = None
    negative: Dict[str, Any] | None = None
    for reference in references:
        sequence = str(reference["sequence"])
        if len(sequence) != length:
            continue
        reference_matrix = reference["distance_matrix"]
        for shift in range(length):
            reference_position = (target_position_1based - 1 - shift) % length
            token = sequence[reference_position]
            if token.upper() == "X":
                continue
            payload = {
                "split": str(reference["split"]),
                "sample_name": str(reference["sample_name"]),
                "distance_matrix_rmse_angstrom": cyclic_distance_matrix_rmse(
                    target_matrix,
                    reference_matrix,  # type: ignore[arg-type]
                    shift,
                ),
                "forward_cyclic_shift": shift,
                "reference_position_1based": reference_position + 1,
                "reference_token": token,
                "reference_sequence": sequence,
                "reference_is_methyl_positive": int(token.islower()),
            }
            destination = "positive" if token.islower() else "negative"
            current = positive if destination == "positive" else negative
            key = (
                float(payload["distance_matrix_rmse_angstrom"]),
                str(payload["sample_name"]),
                int(payload["forward_cyclic_shift"]),
            )
            current_key = (
                (
                    float(current["distance_matrix_rmse_angstrom"]),
                    str(current["sample_name"]),
                    int(current["forward_cyclic_shift"]),
                )
                if current is not None
                else None
            )
            if current_key is None or key < current_key:
                if destination == "positive":
                    positive = payload
                else:
                    negative = payload
    return positive, negative


def audit_dominant_position_structural_support(
    eligible_rows: Sequence[Mapping[str, Any]],
    native_rows: Sequence[Mapping[str, Any]],
    target_manifest_rows: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    concentration_share: float = DEFAULT_CONCENTRATION_SHARE,
    minimum_sites: int = DEFAULT_MINIMUM_SITES,
    maximum_positive_rmse_angstrom: float = (
        DEFAULT_MAXIMUM_POSITIVE_RMSE_ANGSTROM
    ),
    minimum_positive_vs_negative_margin_angstrom: float = (
        DEFAULT_MINIMUM_POSITIVE_VS_NEGATIVE_MARGIN_ANGSTROM
    ),
) -> Dict[str, Any]:
    if not 0.0 < concentration_share < 1.0:
        raise ValueError("concentration_share must be between zero and one")
    if minimum_sites <= 0:
        raise ValueError("minimum_sites must be positive")

    positions_by_target: MutableMapping[str, Counter[int]] = defaultdict(Counter)
    for row in eligible_rows:
        target = str(row.get("target_name", "")).upper()
        sequence = str(row.get("design_seq", ""))
        for position, token in enumerate(sequence, start=1):
            if token.islower():
                positions_by_target[target][position] += 1

    selected_chain_by_target = {
        str(row.get("target_name", "")).upper(): str(row.get("selected_chain", ""))
        for row in target_manifest_rows
        if str(row.get("target_name", "")).strip()
    }
    native_index = {
        record_name(record, index): record
        for index, record in enumerate(native_rows)
    }
    train_by_length = _reference_index(train_records, "train")
    test_by_length = _reference_index(test_records, "test")

    concentrated_rows: List[Dict[str, Any]] = []
    for target in sorted(positions_by_target):
        counts = positions_by_target[target]
        total = int(sum(counts.values()))
        if total < minimum_sites or not counts:
            continue
        dominant_count = max(counts.values())
        dominant_share = dominant_count / total
        if dominant_share <= concentration_share:
            continue
        dominant_position = min(
            position for position, count in counts.items() if count == dominant_count
        )
        if target not in native_index:
            raise ValueError(f"Target {target} is absent from native JSONL")
        selected_chain = selected_chain_by_target.get(target, "")
        if not selected_chain:
            raise ValueError(f"Target {target} has no selected peptide chain")
        coordinates = ca_coordinates(native_index[target], selected_chain)
        target_matrix = distance_matrix(coordinates)
        length = len(coordinates)
        heldout_positive, heldout_negative = _nearest_examples(
            target_matrix,
            dominant_position,
            test_by_length.get(length, []),
        )
        train_positive, train_negative = _nearest_examples(
            target_matrix,
            dominant_position,
            train_by_length.get(length, []),
        )
        heldout_positive_rmse = (
            float(heldout_positive["distance_matrix_rmse_angstrom"])
            if heldout_positive is not None
            else math.inf
        )
        heldout_negative_rmse = (
            float(heldout_negative["distance_matrix_rmse_angstrom"])
            if heldout_negative is not None
            else math.inf
        )
        support_pass = (
            heldout_positive is not None
            and heldout_negative is not None
            and heldout_positive_rmse <= maximum_positive_rmse_angstrom
            and heldout_positive_rmse
            + minimum_positive_vs_negative_margin_angstrom
            <= heldout_negative_rmse
        )
        concentrated_rows.append(
            {
                "target_name": target,
                "selected_chain": selected_chain,
                "peptide_length": length,
                "methyl_sites": total,
                "dominant_position_1based": dominant_position,
                "dominant_position_count": dominant_count,
                "dominant_position_share": dominant_share,
                "heldout_test_nearest_methyl_positive": heldout_positive,
                "heldout_test_nearest_natural_negative": heldout_negative,
                "training_nearest_methyl_positive": train_positive,
                "training_nearest_natural_negative": train_negative,
                "heldout_positive_vs_negative_margin_angstrom": (
                    heldout_negative_rmse - heldout_positive_rmse
                ),
                "structural_support_pass": support_pass,
            }
        )

    quality_gate = (
        "PASS"
        if all(bool(row["structural_support_pass"]) for row in concentrated_rows)
        else "FAIL"
    )
    return {
        "quality_gate": quality_gate,
        "method": "forward_cyclic_ca_distance_matrix_heldout_provenance_support_v1",
        "interpretation": (
            "Absolute target position is a diagnostic, not a decoder-order proxy. "
            "Every target with an eligible concentrated position is audited, including "
            "targets below the legacy n>=30 reporting threshold. A concentrated "
            "position is accepted only when a held-out, provenance-confirmed methyl-"
            "positive backbone match is close and is separated from the nearest "
            "held-out natural-negative match."
        ),
        "concentration_share_threshold": concentration_share,
        "minimum_sites": minimum_sites,
        "maximum_positive_rmse_angstrom": maximum_positive_rmse_angstrom,
        "minimum_positive_vs_negative_margin_angstrom": (
            minimum_positive_vs_negative_margin_angstrom
        ),
        "train_reference_records": len(train_records),
        "heldout_test_reference_records": len(test_records),
        "concentrated_target_count": len(concentrated_rows),
        "concentrated_targets": concentrated_rows,
    }
