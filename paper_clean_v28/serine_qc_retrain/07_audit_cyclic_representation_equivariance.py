#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate a cyclic-start-invariant deployment policy before regeneration.

The V3 checkpoint balanced causal decoder depth, but it did not rotate the
serialized cyclic peptide (sequence plus N/CA/C/O coordinates).  This audit
adds that missing outer ensemble, maps every prediction back to the original
physical residue, and re-runs the frozen held-out test gates.  It also records
how much the un-ensembled checkpoint changes when only the arbitrary cyclic
start is changed.

No checkpoint, candidate CSV, or prior result is modified by this script.  A
PASS authorizes a new, isolated V6 generation that uses the representation
ensemble.  It does not authorize a structure handoff.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_clean_v28.clean_v28_common import (  # noqa: E402
    EXTENDED_AA_ALPHABET,
    NATURAL_AA_ALPHABET,
    N_NATURAL,
    V11_CYCLIC_OFFSET_POLICY,
    V11_MODEL_ARCHITECTURE_PROTOCOL,
    X_INDEX,
    cyclic_known_sequence_methyl_probabilities,
    cyclic_representation_known_sequence_methyl_probabilities,
    featurize_records,
    load_v28_model,
    naturalize_tensor_for_input,
)


DEFAULT_MODEL = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_cyclic_representation_v6"
    / "model"
    / "frankenstein_v28_expert_heads_qc.pt"
)
DEFAULT_TEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_order_balanced_v3"
    / "data"
    / "test_serine_provenance_corrected.jsonl"
)
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_BEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "best_designs.csv"
)
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_representation_v6.json")
DEFAULT_OUT = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_cyclic_representation_v6"
    / "representation_audit"
)
REQUIRED_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_"
    "cyclic_stability_worst_start_v9"
)
V11_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_cyclic_native_relative_positions_v11"
)
SERINE_ONLY_EXPERT_PROTOCOL = (
    "canonical_clean_v28_serine_only_corrected_labels_"
    "cyclic_stability_worst_start_v9"
)
V6_AUDIT_PROTOCOL = "cyclic_stability_worst_start_heldout_gate_v9"
V11_AUDIT_PROTOCOL = "cyclic_native_relative_positions_heldout_gate_v11"
V7_AUDIT_PROTOCOL = (
    "cyclic_stability_worst_start_heldout_gate_v9_serine_only"
)
V6_AUTHORIZATION = (
    "CYCLIC_STABILITY_V9_VALIDATED_FOR_UNIFORM_REGENERATION"
)
V11_AUTHORIZATION = (
    "CYCLIC_NATIVE_V11_VALIDATED_FOR_RMSD_PRIORITY_REGENERATION"
)
V7_AUTHORIZATION = (
    "SERINE_ONLY_CYCLIC_STABILITY_V9_VALIDATED_FOR_REANNOTATION"
)
REQUIRED_TRAINING_REPRESENTATION_POLICY = (
    "all_physical_cyclic_starts_jointly_rotate_sequence_labels_and_"
    "backbone_coordinates_with_residue_index_reset"
)
V11_TRAINING_REPRESENTATION_POLICY = (
    "boundary_marginalized_cyclic_relative_positions_with_all_physical_starts_"
    "retained_as_an_explicit_equivariance_verification_grid"
)
V11_MAXIMUM_EQUIVARIANCE_SPAN = 1e-5
REQUIRED_TRAINING_ORDER_POLICY = (
    "complete_physical_cyclic_start_x_complete_L_decoder_order_grid_"
    "differentiably_meaned_per_start_then_mapped_to_physical_labels"
)
REQUIRED_DEPLOYMENT_POLICY = (
    "all_cyclic_starts_and_all_decoder_orders_mapped_to_physical_"
    "residues_probability_mean_for_ranking_representation_min_for_release"
)
DECODER_ONLY_MODE = "peptide_only_cyclic_order_ensemble_known_natural_sequence"
REPRESENTATION_MODE = (
    "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
)
ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
SUPPORTED_METHYL_BASES = set(NATURAL_AA_ALPHABET) - {"P"}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strict_rounded_probability_pass(value: float, threshold: float) -> bool:
    """Match the exact eight-decimal release decision persisted by generation."""

    numeric = float(value)
    return (
        math.isfinite(numeric)
        and 0.0 <= numeric <= 1.0
        and round(numeric, 8) > float(threshold)
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_name(record: Mapping[str, Any], fallback: int) -> str:
    return str(
        record.get("name")
        or record.get("pdb")
        or record.get("pdb_id")
        or record.get("id")
        or f"record_{fallback}"
    ).upper()


def batches(values: Sequence[Mapping[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(values), batch_size):
        yield [dict(value) for value in values[start : start + batch_size]]


def roc_auc_score_simple(y_true: np.ndarray, probability: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(probability), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and probability[order[end]] == probability[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def binary_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    predicted = np.asarray(
        [
            strict_rounded_probability_pass(value, threshold)
            for value in probability
        ],
        dtype=bool,
    )
    true = y_true == 1
    tp = int(np.sum(predicted & true))
    fp = int(np.sum(predicted & ~true))
    tn = int(np.sum(~predicted & ~true))
    fn = int(np.sum(~predicted & true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "threshold_operator": ">",
        "probability_rounding_policy": "round(prob,8)",
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "pred_methyl_rate": (tp + fp) / len(y_true) if len(y_true) else 0.0,
    }


def metric_summary(
    position_rows: Sequence[Mapping[str, Any]],
    probability_field: str,
    threshold: float,
    auc_probability_field: str | None = None,
) -> Dict[str, Any]:
    y_all = np.asarray(
        [int(row["is_methyl_true"]) for row in position_rows], dtype=np.int64
    )
    probability_all = np.asarray(
        [float(row[probability_field]) for row in position_rows], dtype=np.float64
    )
    auc_field = auc_probability_field or probability_field
    auc_probability_all = np.asarray(
        [float(row[auc_field]) for row in position_rows], dtype=np.float64
    )
    grouped: MutableMapping[str, List[int]] = defaultdict(list)
    for index, row in enumerate(position_rows):
        grouped[str(row["base_token"])].append(index)
    per_residue: List[Dict[str, Any]] = []
    for base in NATURAL_AA_ALPHABET:
        indices = np.asarray(grouped.get(base, []), dtype=np.int64)
        y = y_all[indices] if len(indices) else np.asarray([], dtype=np.int64)
        p = probability_all[indices] if len(indices) else np.asarray([], dtype=np.float64)
        auc_p = (
            auc_probability_all[indices]
            if len(indices)
            else np.asarray([], dtype=np.float64)
        )
        per_residue.append(
            {
                "base_token": base,
                "positions": int(len(indices)),
                "natural_negatives": int(np.sum(y == 0)),
                "methyl_positives": int(np.sum(y == 1)),
                "auc": roc_auc_score_simple(y, auc_p),
                **(binary_metrics(y, p, threshold) if len(indices) else {}),
            }
        )
    supported = [
        row
        for row in per_residue
        if row["base_token"] in SUPPORTED_METHYL_BASES and row["auc"] is not None
    ]
    non_ser_indices = np.asarray(
        [
            index
            for index, row in enumerate(position_rows)
            if str(row["base_token"]) != "S"
        ],
        dtype=np.int64,
    )
    serine = next(row for row in per_residue if row["base_token"] == "S")
    proline = next(row for row in per_residue if row["base_token"] == "P")
    return {
        "positions": len(position_rows),
        "threshold": threshold,
        "threshold_probability_field": probability_field,
        "auc_probability_field": auc_field,
        "overall_auc": roc_auc_score_simple(y_all, auc_probability_all),
        "overall_auc_release_min": roc_auc_score_simple(y_all, probability_all),
        "overall_at_threshold": binary_metrics(y_all, probability_all, threshold),
        "non_ser_auc": roc_auc_score_simple(
            y_all[non_ser_indices], auc_probability_all[non_ser_indices]
        ),
        "non_ser_at_threshold": binary_metrics(
            y_all[non_ser_indices], probability_all[non_ser_indices], threshold
        ),
        "supported_expert_count": len(supported),
        "supported_macro_auc": (
            float(np.mean([float(row["auc"]) for row in supported]))
            if supported
            else None
        ),
        "supported_macro_f1_at_threshold": (
            float(np.mean([float(row["f1"]) for row in supported]))
            if supported
            else None
        ),
        "serine": serine,
        "proline": proline,
        "per_residue": per_residue,
    }


def evaluate_heldout(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    threshold: float,
    temperature: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    position_rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(batches(records, batch_size)):
            packed = featurize_records(batch, device=device, eval_chains="masked")
            if packed is None:
                continue
            tensors, metas = packed
            X, S_label, mask, chain_M, residue_idx, chain_encoding_all, real_pos = tensors
            valid = (
                (mask > 0)
                & (chain_M > 0)
                & (real_pos > 0)
                & (S_label != X_INDEX)
            )
            S_forward = naturalize_tensor_for_input(S_label)
            base_logits, _unused_expert_logits = model(
                X,
                S_forward,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
            )
            pred_base = torch.argmax(base_logits, dim=-1)
            decoder_probability, decoder_order_std = (
                cyclic_known_sequence_methyl_probabilities(
                    model,
                    X,
                    S_forward,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                    temperature=temperature,
                )
            )
            representation = (
                cyclic_representation_known_sequence_methyl_probabilities(
                    model,
                    X,
                    S_forward,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                    temperature=temperature,
                )
            )
            end_to_end_representation = (
                cyclic_representation_known_sequence_methyl_probabilities(
                    model,
                    X,
                    pred_base,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                    temperature=temperature,
                )
            )
            true_base = naturalize_tensor_for_input(S_label)
            for row_index, meta in enumerate(metas):
                positions = torch.where(valid[row_index])[0].cpu().tolist()
                for physical_index, position in enumerate(positions, start=1):
                    base_index = int(true_base[row_index, position].item())
                    target_index = int(S_label[row_index, position].item())
                    minimum = float(
                        representation["representation_min"][row_index, position].item()
                    )
                    maximum = float(
                        representation["representation_max"][row_index, position].item()
                    )
                    end_to_end_minimum = float(
                        end_to_end_representation["representation_min"][
                            row_index, position
                        ].item()
                    )
                    end_to_end_maximum = float(
                        end_to_end_representation["representation_max"][
                            row_index, position
                        ].item()
                    )
                    is_methyl_true = int(target_index >= N_NATURAL)
                    position_rows.append(
                        {
                            "sample_name": meta["name"],
                            "batch_index": batch_index,
                            "position_in_peptide_1based": physical_index,
                            "target_token": EXTENDED_AA_ALPHABET[target_index],
                            "base_token": NATURAL_AA_ALPHABET[base_index],
                            "pred_base_token": NATURAL_AA_ALPHABET[
                                int(pred_base[row_index, position].item())
                            ],
                            "base_correct": int(
                                int(pred_base[row_index, position].item()) == base_index
                            ),
                            "is_methyl_true": is_methyl_true,
                            "probability_decoder_order_only": float(
                                decoder_probability[row_index, position].item()
                            ),
                            "probability_representation_ensemble": float(
                                representation["mean"][row_index, position].item()
                            ),
                            "probability_decoder_order_std": float(
                                decoder_order_std[row_index, position].item()
                            ),
                            "probability_decoder_order_std_mean_across_representations": float(
                                representation["decoder_order_std_mean"][
                                    row_index, position
                                ].item()
                            ),
                            "probability_representation_std": float(
                                representation["representation_std"][
                                    row_index, position
                                ].item()
                            ),
                            "probability_representation_min": minimum,
                            "probability_representation_max": maximum,
                            "probability_label_aware_adversarial": (
                                minimum if is_methyl_true else maximum
                            ),
                            "probability_representation_span": float(
                                representation["representation_span"][
                                    row_index, position
                                ].item()
                            ),
                            "representation_threshold_disagreement": int(
                                not strict_rounded_probability_pass(
                                    minimum, threshold
                                )
                                and strict_rounded_probability_pass(
                                    maximum, threshold
                                )
                            ),
                            "probability_end_to_end_representation_ensemble": float(
                                end_to_end_representation["mean"][
                                    row_index, position
                                ].item()
                            ),
                            "probability_end_to_end_representation_std": float(
                                end_to_end_representation["representation_std"][
                                    row_index, position
                                ].item()
                            ),
                            "probability_end_to_end_representation_min": end_to_end_minimum,
                            "probability_end_to_end_representation_max": end_to_end_maximum,
                            "probability_end_to_end_label_aware_adversarial": (
                                end_to_end_minimum
                                if is_methyl_true
                                else end_to_end_maximum
                            ),
                            "probability_end_to_end_representation_span": float(
                                end_to_end_representation["representation_span"][
                                    row_index, position
                                ].item()
                            ),
                            "end_to_end_representation_threshold_disagreement": int(
                                not strict_rounded_probability_pass(
                                    end_to_end_minimum, threshold
                                )
                                and strict_rounded_probability_pass(
                                    end_to_end_maximum, threshold
                                )
                            ),
                            "annotation_mode": REPRESENTATION_MODE,
                            "annotation_context_policy": ANNOTATION_CONTEXT,
                            "annotation_representation_ensemble_size": len(positions),
                            "annotation_decoder_order_ensemble_size": len(positions),
                        }
                    )
    decoder_summary = metric_summary(
        position_rows, "probability_decoder_order_only", threshold
    )
    representation_summary = metric_summary(
        position_rows, "probability_representation_ensemble", threshold
    )
    spans = [float(row["probability_representation_span"]) for row in position_rows]
    disagreements = sum(
        int(row["representation_threshold_disagreement"]) for row in position_rows
    )
    representation_summary["maximum_probability_representation_span"] = max(spans)
    representation_summary["mean_probability_representation_span"] = float(
        np.mean(spans)
    )
    representation_summary["representation_threshold_disagreement_positions"] = (
        disagreements
    )
    representation_summary["representation_threshold_disagreement_rate"] = (
        disagreements / len(position_rows)
    )
    end_to_end_representation_summary = metric_summary(
        position_rows,
        "probability_end_to_end_representation_ensemble",
        threshold,
    )
    end_to_end_release_floor_summary = metric_summary(
        position_rows,
        "probability_end_to_end_representation_min",
        threshold,
        auc_probability_field="probability_end_to_end_label_aware_adversarial",
    )
    end_to_end_spans = [
        float(row["probability_end_to_end_representation_span"])
        for row in position_rows
    ]
    end_to_end_disagreements = sum(
        int(row["end_to_end_representation_threshold_disagreement"])
        for row in position_rows
    )
    end_to_end_representation_summary.update(
        {
            "maximum_probability_representation_span": max(end_to_end_spans),
            "mean_probability_representation_span": float(
                np.mean(end_to_end_spans)
            ),
            "representation_threshold_disagreement_positions": (
                end_to_end_disagreements
            ),
            "representation_threshold_disagreement_rate": (
                end_to_end_disagreements / len(position_rows)
            ),
        }
    )
    representation_summary["end_to_end_representation_ensemble"] = (
        end_to_end_representation_summary
    )
    representation_summary["end_to_end_release_floor"] = (
        end_to_end_release_floor_summary
    )
    return position_rows, decoder_summary, representation_summary


def selected_chain_index(best_rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
    by_target: MutableMapping[str, set[str]] = defaultdict(set)
    for row in best_rows:
        target = str(row.get("target_name", "")).strip().upper()
        chains = [
            value.strip()
            for value in str(row.get("selected_chains", "")).split(",")
            if value.strip()
        ]
        if target and len(chains) == 1:
            by_target[target].add(chains[0])
    result: Dict[str, str] = {}
    for target, chains in by_target.items():
        if len(chains) != 1:
            raise RuntimeError(
                f"Selected peptide chain is inconsistent for {target}: {sorted(chains)}"
            )
        result[target] = next(iter(chains))
    return result


def peptide_only_record(
    source: Mapping[str, Any], selected_chain: str, natural_sequence: str
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "name": str(source.get("name", "native")),
        "seq": natural_sequence,
        f"seq_chain_{selected_chain}": natural_sequence,
        "masked_list": [selected_chain],
        "visible_list": [],
    }
    for atom_name in ("N", "CA", "C", "O"):
        key = f"{atom_name}_chain_{selected_chain}"
        if key not in source:
            raise RuntimeError(f"Missing native peptide coordinate {key}")
        record[key] = copy.deepcopy(source[key])
    return record


def audit_native_targets(
    model: torch.nn.Module,
    native_rows: Sequence[Mapping[str, Any]],
    selected_chains: Mapping[str, str],
    targets: Sequence[str],
    device: torch.device,
    temperature: float,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    native_index = {
        record_name(row, index): row for index, row in enumerate(native_rows)
    }
    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for target in targets:
            source = native_index[target]
            chain = selected_chains[target]
            sequence = str(source[f"seq_chain_{chain}"]).upper()
            packed = featurize_records(
                [peptide_only_record(source, chain, sequence)],
                device=device,
                eval_chains="masked",
            )
            if packed is None:
                raise RuntimeError(f"Feature construction failed for {target}")
            tensors, _metas = packed
            X, S, mask, chain_M, residue_idx, chain_encoding_all = tensors[:6]
            length = len(sequence)
            expanded_X = []
            expanded_S = []
            expanded_residue_idx = []
            for shift in range(length):
                expanded_X.append(torch.roll(X[0], shifts=-shift, dims=0))
                expanded_S.append(torch.roll(S[0], shifts=-shift, dims=0))
                expanded_residue_idx.append(
                    torch.arange(
                        length,
                        device=device,
                        dtype=residue_idx.dtype,
                    )
                )
            raw_probability, _raw_order_std = (
                cyclic_known_sequence_methyl_probabilities(
                    model,
                    torch.stack(expanded_X, dim=0),
                    torch.stack(expanded_S, dim=0),
                    mask.repeat(length, 1),
                    chain_M.repeat(length, 1),
                    torch.stack(expanded_residue_idx, dim=0),
                    chain_encoding_all.repeat(length, 1),
                    temperature=temperature,
                )
            )
            ensemble = cyclic_representation_known_sequence_methyl_probabilities(
                model,
                X,
                S,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
                temperature=temperature,
            )
            base_logits, _unused_expert_logits = model(
                X,
                S,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
            )
            pred_base = torch.argmax(base_logits, dim=-1)
            end_to_end_ensemble = (
                cyclic_representation_known_sequence_methyl_probabilities(
                    model,
                    X,
                    pred_base,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                    temperature=temperature,
                )
            )
            mapped_by_shift: List[List[float]] = []
            raw_tensor_selected: Counter[int] = Counter()
            mapped_annotation_sets: List[List[int]] = []
            end_to_end_annotation_sets: List[List[int]] = []
            for shift in range(length):
                raw = raw_probability[shift].detach().cpu().tolist()
                mapped = np.roll(np.asarray(raw, dtype=np.float64), shift).tolist()
                mapped_end_to_end = (
                    end_to_end_ensemble["representation_probability_by_start"]
                    [0, shift, :length]
                    .detach()
                    .cpu()
                    .tolist()
                )
                mapped_by_shift.append(mapped)
                raw_selected = [
                    index + 1
                    for index, value in enumerate(raw)
                    if strict_rounded_probability_pass(value, threshold)
                ]
                raw_tensor_selected.update(raw_selected)
                mapped_annotation_sets.append(
                    [
                        index + 1
                        for index, value in enumerate(mapped)
                        if strict_rounded_probability_pass(value, threshold)
                    ]
                )
                end_to_end_annotation_sets.append(
                    [
                        index + 1
                        for index, value in enumerate(mapped_end_to_end)
                        if strict_rounded_probability_pass(value, threshold)
                    ]
                )
                for physical_position, value in enumerate(mapped, start=1):
                    tensor_position = (physical_position - shift - 1) % length + 1
                    end_to_end_value = float(
                        mapped_end_to_end[physical_position - 1]
                    )
                    detail_rows.append(
                        {
                            "target_name": target,
                            "native_sequence": sequence,
                            "representation_left_shift": shift,
                            "physical_position_1based": physical_position,
                            "tensor_position_1based": tensor_position,
                            "base_token": sequence[physical_position - 1],
                            "mapped_probability": value,
                            "mapped_probability_known_sequence": value,
                            "mapped_probability_end_to_end": end_to_end_value,
                            "above_threshold": int(
                                strict_rounded_probability_pass(value, threshold)
                            ),
                            "known_sequence_above_threshold": int(
                                strict_rounded_probability_pass(value, threshold)
                            ),
                            "end_to_end_above_threshold": int(
                                strict_rounded_probability_pass(
                                    end_to_end_value, threshold
                                )
                            ),
                        }
                    )
            recomputed = np.mean(np.asarray(mapped_by_shift), axis=0)
            reported = ensemble["mean"][0].detach().cpu().numpy()
            maximum_recompute_difference = float(np.max(np.abs(recomputed - reported)))
            ensemble_selected = [
                index + 1
                for index, value in enumerate(reported)
                if strict_rounded_probability_pass(value, threshold)
            ]
            most_common_tensor_count = (
                max(raw_tensor_selected.values()) if raw_tensor_selected else 0
            )
            summary_rows.append(
                {
                    "target_name": target,
                    "native_sequence": sequence,
                    "peptide_length": length,
                    "representation_count": length,
                    "ensemble_methyl_positions_1based": json.dumps(ensemble_selected),
                    "raw_mapped_annotation_sets": json.dumps(mapped_annotation_sets),
                    "raw_all_representations_same_physical_annotation": int(
                        len({tuple(value) for value in mapped_annotation_sets}) == 1
                    ),
                    "end_to_end_mapped_annotation_sets": json.dumps(
                        end_to_end_annotation_sets
                    ),
                    "end_to_end_all_representations_same_physical_annotation": int(
                        len(
                            {
                                tuple(value)
                                for value in end_to_end_annotation_sets
                            }
                        )
                        == 1
                    ),
                    "raw_tensor_position_counts_above_threshold": json.dumps(
                        dict(sorted(raw_tensor_selected.items()))
                    ),
                    "raw_maximum_single_tensor_position_share": (
                        most_common_tensor_count / sum(raw_tensor_selected.values())
                        if raw_tensor_selected
                        else 0.0
                    ),
                    "maximum_ensemble_recompute_difference": maximum_recompute_difference,
                    "maximum_known_sequence_representation_span": float(
                        ensemble["representation_span"][0, :length].max().item()
                    ),
                    "maximum_end_to_end_representation_span": float(
                        end_to_end_ensemble["representation_span"]
                        [0, :length]
                        .max()
                        .item()
                    ),
                    "ensemble_probability_vector": json.dumps(
                        [round(float(value), 8) for value in reported]
                    ),
                    "ensemble_representation_span_vector": json.dumps(
                        [
                            round(float(value), 8)
                            for value in ensemble["representation_span"][0]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                    ),
                    "end_to_end_ensemble_probability_vector": json.dumps(
                        [
                            round(float(value), 8)
                            for value in end_to_end_ensemble["mean"][0]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                    ),
                    "end_to_end_ensemble_representation_min_vector": json.dumps(
                        [
                            round(float(value), 8)
                            for value in end_to_end_ensemble[
                                "representation_min"
                            ][0]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                    ),
                    "end_to_end_ensemble_representation_max_vector": json.dumps(
                        [
                            round(float(value), 8)
                            for value in end_to_end_ensemble[
                                "representation_max"
                            ][0]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                    ),
                }
            )
    return detail_rows, summary_rows


def checkpoint_metadata(
    model_path: Path, required_protocol: str = REQUIRED_EXPERT_PROTOCOL
) -> Dict[str, Any]:
    payload = torch.load(model_path, map_location="cpu")
    metadata = (
        dict(payload.get("expert_head_qc_metadata", {}))
        if isinstance(payload, Mapping)
        else {}
    )
    architecture = (
        dict(payload.get("model_architecture_metadata", {}))
        if isinstance(payload, Mapping)
        else {}
    )
    del payload
    is_v11 = required_protocol == V11_EXPERT_PROTOCOL
    v11_base_noninferiority = True
    if is_v11:
        legacy_base = metadata.get("legacy_parent_base_validation", {})
        cyclic_parent_base = metadata.get(
            "cyclic_native_parent_base_validation", {}
        )
        selected_base = metadata.get("selected_base_validation", {})
        try:
            maximum_ce_increase = float(
                metadata["maximum_base_cross_entropy_increase"]
            )
            maximum_accuracy_drop = float(metadata["maximum_base_accuracy_drop"])
            v11_base_noninferiority = (
                math.isfinite(maximum_ce_increase)
                and math.isfinite(maximum_accuracy_drop)
                and 0.0 <= maximum_ce_increase <= 0.05
                and 0.0 <= maximum_accuracy_drop <= 0.02
                and isinstance(legacy_base, Mapping)
                and isinstance(cyclic_parent_base, Mapping)
                and isinstance(selected_base, Mapping)
                and float(selected_base["cross_entropy"])
                <= float(legacy_base["cross_entropy"]) + maximum_ce_increase
                and float(selected_base["cross_entropy"])
                <= float(cyclic_parent_base["cross_entropy"])
                + maximum_ce_increase
                and float(selected_base["accuracy"]) + maximum_accuracy_drop
                >= float(legacy_base["accuracy"])
                and float(selected_base["accuracy"]) + maximum_accuracy_drop
                >= float(cyclic_parent_base["accuracy"])
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            v11_base_noninferiority = False
    if not (
        str(metadata.get("protocol", "")) == required_protocol
        and int(metadata.get("minimum_order_coverage_epochs", 0)) >= 30
        and bool(metadata.get("cyclic_representation_augmentation"))
        and str(metadata.get("training_cyclic_representation_policy", ""))
        == (
            V11_TRAINING_REPRESENTATION_POLICY
            if is_v11
            else REQUIRED_TRAINING_REPRESENTATION_POLICY
        )
        and str(metadata.get("training_decoding_order_policy", ""))
        == REQUIRED_TRAINING_ORDER_POLICY
        and str(metadata.get("deployment_annotation_policy", ""))
        == REQUIRED_DEPLOYMENT_POLICY
        and float(metadata.get("worst_start_bce_weight", 0.0)) > 0.0
        and float(metadata.get("representation_consistency_weight", -1.0)) > 0.0
        and bool(metadata.get("full_physical_start_by_full_decoder_order_grid"))
        and float(metadata.get("training_ensemble_temperature", -1.0)) == 0.5
        and "full_physical_start_x_full_decoder_order_grid"
        in str(metadata.get("training_objective", ""))
        and (
            not is_v11
            or (
                bool(metadata.get("cyclic_relative_positions"))
                and str(metadata.get("model_architecture_protocol", ""))
                == V11_MODEL_ARCHITECTURE_PROTOCOL
                and str(metadata.get("cyclic_offset_policy", ""))
                == V11_CYCLIC_OFFSET_POLICY
                and float(metadata.get("base_sequence_loss_weight", 0.0)) > 0.0
                and float(metadata.get("positional_anchor_weight", 0.0)) > 0.0
                and float(
                    metadata.get("maximum_equivariance_span_tolerance", -1.0)
                )
                > 0.0
                and float(
                    metadata.get("maximum_equivariance_span_tolerance", -1.0)
                )
                <= V11_MAXIMUM_EQUIVARIANCE_SPAN
                and float(
                    metadata.get(
                        "best_epoch_maximum_training_representation_span",
                        float("inf"),
                    )
                )
                <= V11_MAXIMUM_EQUIVARIANCE_SPAN
                and set(metadata.get("trained_cyclic_positional_state_keys", []))
                == {
                    "features.embeddings.linear.weight",
                    "features.embeddings.linear.bias",
                }
                and bool(architecture.get("cyclic_relative_positions"))
                and str(architecture.get("protocol", ""))
                == V11_MODEL_ARCHITECTURE_PROTOCOL
                and str(architecture.get("cyclic_offset_policy", ""))
                == V11_CYCLIC_OFFSET_POLICY
                and v11_base_noninferiority
            )
        )
    ):
        raise RuntimeError(
            "Representation audit requires the requested promoted checkpoint "
            "with its complete cyclic training and architecture contract"
        )
    if required_protocol == SERINE_ONLY_EXPERT_PROTOCOL and not (
        str(metadata.get("expert_scope", "")) == "serine-only"
        and list(metadata.get("active_expert_tokens", [])) == ["S"]
    ):
        raise RuntimeError(
            "V7 representation audit requires an exactly Ser-only checkpoint"
        )
    metadata["model_architecture_metadata"] = architecture
    return metadata


def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).resolve()
    test_path = Path(args.test_jsonl).resolve()
    native_path = Path(args.native_jsonl).resolve()
    best_path = Path(args.best_csv).resolve()
    plan_path = Path(args.plan).resolve()
    out_dir = Path(args.out_dir).resolve()
    for required in (model_path, test_path, native_path, best_path, plan_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if float(args.temperature) != 0.5 or float(args.threshold) != 0.6:
        raise ValueError(
            "The V9 held-out authorization is frozen to temperature 0.5 and "
            "methyl threshold 0.6"
        )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Representation audit output already exists; use --overwrite: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU audit requires --allow-cpu")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise RuntimeError("No CUDA device is available; pass --allow-cpu knowingly")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    required_protocol = str(args.required_expert_protocol)
    if required_protocol not in {
        REQUIRED_EXPERT_PROTOCOL,
        SERINE_ONLY_EXPERT_PROTOCOL,
        V11_EXPERT_PROTOCOL,
    }:
        raise ValueError("Unsupported expert protocol for representation audit")
    serine_only = required_protocol == SERINE_ONLY_EXPERT_PROTOCOL
    cyclic_native_v11 = required_protocol == V11_EXPERT_PROTOCOL
    metadata = checkpoint_metadata(model_path, required_protocol)
    plan = read_json(plan_path)
    targets = [str(row["target_name"]).upper() for row in plan["targets"]]
    frozen = {str(value).upper() for value in plan["frozen_targets"]}
    if len(targets) != int(plan["expected_target_count"]):
        raise RuntimeError("V6 plan target count is inconsistent")
    if set(targets) & frozen:
        raise RuntimeError("V6 generated and frozen target sets overlap")
    selected_chains = selected_chain_index(read_csv(best_path))
    missing_chains = sorted(set(targets) - set(selected_chains))
    if missing_chains:
        raise RuntimeError("Missing selected peptide chains: " + ", ".join(missing_chains))

    print(
        f"Loading promoted cyclic-representation checkpoint: {model_path}",
        flush=True,
    )
    model = load_v28_model(str(model_path), device)
    model.eval()
    test_records = read_jsonl(test_path)
    position_rows, decoder_summary, representation_summary = evaluate_heldout(
        model,
        test_records,
        device,
        int(args.batch_size),
        float(args.threshold),
        float(args.temperature),
    )
    native_detail, native_summary = audit_native_targets(
        model,
        read_jsonl(native_path),
        selected_chains,
        targets,
        device,
        float(args.temperature),
        float(args.threshold),
    )

    release_floor_summary = metric_summary(
        position_rows,
        "probability_representation_min",
        float(args.threshold),
        auc_probability_field="probability_label_aware_adversarial",
    )
    overall = release_floor_summary["overall_at_threshold"]
    serine = release_floor_summary["serine"]
    proline = release_floor_summary["proline"]
    quality_checks = {
        "checkpoint_provenance_and_expert_scope_are_pinned": (
            metadata.get("protocol") == required_protocol
            and (
                not serine_only
                or (
                    metadata.get("expert_scope") == "serine-only"
                    and list(metadata.get("active_expert_tokens", [])) == ["S"]
                )
            )
        ),
        "heldout_test_record_count_is_151": len(test_records) == 151,
        "heldout_test_position_count_is_1505": len(position_rows) == 1505,
        "release_floor_overall_auc_ge_0_85": (
            float(release_floor_summary["overall_auc"]) >= 0.85
        ),
        "overall_fpr_at_0_6_le_0_10": float(overall["false_positive_rate"]) <= 0.10,
        "overall_precision_at_0_6_ge_0_75": float(overall["precision"]) >= 0.75,
        "overall_recall_at_0_6_ge_0_40": float(overall["recall"]) >= 0.40,
        "serine_auc_ge_0_70": (
            serine["auc"] is not None and float(serine["auc"]) >= 0.70
        ),
        "serine_fpr_at_0_6_le_0_25": float(serine["false_positive_rate"]) <= 0.25,
        "serine_recall_at_0_6_ge_0_40": float(serine["recall"]) >= 0.40,
        "proline_fpr_at_0_6_le_0_05": float(proline["false_positive_rate"]) <= 0.05,
        "supported_macro_auc_ge_0_70": (
            release_floor_summary["supported_macro_auc"] is not None
            and float(release_floor_summary["supported_macro_auc"]) >= 0.70
        ),
        "all_17_targets_audited_for_uniform_regeneration": (
            len(native_summary) == len(targets) == 17
        ),
        "every_native_target_used_all_cyclic_starts": all(
            int(row["representation_count"]) == int(row["peptide_length"])
            for row in native_summary
        ),
        "heldout_hard_calls_have_zero_cyclic_start_threshold_disagreement": (
            int(
                representation_summary[
                    "representation_threshold_disagreement_positions"
                ]
            )
            == 0
        ),
        "heldout_end_to_end_hard_calls_have_zero_cyclic_start_threshold_disagreement": (
            int(
                representation_summary["end_to_end_representation_ensemble"][
                    "representation_threshold_disagreement_positions"
                ]
            )
            == 0
        ),
        "v11_heldout_known_sequence_maximum_span_le_1e_5": (
            not cyclic_native_v11
            or float(
                representation_summary["maximum_probability_representation_span"]
            )
            <= V11_MAXIMUM_EQUIVARIANCE_SPAN
        ),
        "v11_heldout_end_to_end_maximum_span_le_1e_5": (
            not cyclic_native_v11
            or float(
                representation_summary["end_to_end_representation_ensemble"]
                ["maximum_probability_representation_span"]
            )
            <= V11_MAXIMUM_EQUIVARIANCE_SPAN
        ),
        "every_native_target_hard_call_is_stable_across_cyclic_starts": all(
            int(row["raw_all_representations_same_physical_annotation"]) == 1
            for row in native_summary
        ),
        "every_native_target_end_to_end_hard_call_is_stable_across_cyclic_starts": all(
            int(
                row[
                    "end_to_end_all_representations_same_physical_annotation"
                ]
            )
            == 1
            for row in native_summary
        ),
        "v11_every_native_known_sequence_maximum_span_le_1e_5": (
            not cyclic_native_v11
            or all(
                float(row["maximum_known_sequence_representation_span"])
                <= V11_MAXIMUM_EQUIVARIANCE_SPAN
                for row in native_summary
            )
        ),
        "v11_every_native_end_to_end_maximum_span_le_1e_5": (
            not cyclic_native_v11
            or all(
                float(row["maximum_end_to_end_representation_span"])
                <= V11_MAXIMUM_EQUIVARIANCE_SPAN
                for row in native_summary
            )
        ),
        "mapped_ensemble_recomputes_from_raw_rotations": all(
            float(row["maximum_ensemble_recompute_difference"]) <= 1e-6
            for row in native_summary
        ),
        "all_probabilities_and_diagnostics_are_finite": all(
            math.isfinite(float(row[field]))
            for row in position_rows
            for field in (
                "probability_decoder_order_only",
                "probability_representation_ensemble",
                "probability_representation_std",
                "probability_representation_span",
                "probability_end_to_end_representation_ensemble",
                "probability_end_to_end_representation_std",
                "probability_end_to_end_representation_min",
                "probability_end_to_end_representation_max",
                "probability_end_to_end_representation_span",
            )
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    authorization = (
        (
            V7_AUTHORIZATION
            if serine_only
            else (V11_AUTHORIZATION if cyclic_native_v11 else V6_AUTHORIZATION)
        )
        if quality_gate == "PASS"
        else "BLOCKED_DO_NOT_REANNOTATE_OR_RELEASE"
    )

    atomic_write_csv(
        out_dir / "heldout_position_probabilities.csv",
        position_rows,
        list(position_rows[0]),
    )
    atomic_write_csv(
        out_dir / "native_target_representation_probabilities.csv",
        native_detail,
        list(native_detail[0]),
    )
    atomic_write_csv(
        out_dir / "native_target_representation_summary.csv",
        native_summary,
        list(native_summary[0]),
    )
    report = {
        "quality_gate": quality_gate,
        "release_authorization": authorization,
        "protocol": (
            V7_AUDIT_PROTOCOL
            if serine_only
            else (V11_AUDIT_PROTOCOL if cyclic_native_v11 else V6_AUDIT_PROTOCOL)
        ),
        "scientific_scope": (
            (
                "V11 removes the artificial peptide cut inside the model by analytically "
                "marginalizing learned relative-position embeddings over every cyclic "
                "start. The outer all-start x all-order grid remains an independent "
                "numerical proof: both known-sequence and end-to-end probability spans "
                "must be <=1e-5 on 1,505 held-out positions and all 17 native targets. "
                "This does not prove post-structure RMSD; later generation, exact base, "
                "RMSD-priority, and returned-structure audits remain mandatory."
                if cyclic_native_v11
                else
                "Outer ensemble jointly rotates sequence and N/CA/C/O coordinates, "
                "resets linear residue indices, maps probabilities back to physical "
                "residues, and uses the mean only for ranking while the all-start "
                "minimum controls release. This repairs the training/deployment cyclic-"
                "grid mismatch and is paired with a checkpoint trained on every cyclic "
                "start. It does not prove that the trained heads eliminated target-level "
                "methyl-site concentration; candidate generation and final selection "
                "enforce that separately."
            )
        ),
        "validation_scope": (
            "Internal development safety audit only. The 151 records were reused "
            "during V3-V9 development and are not a blind publication outer test; "
            "a structure/scaffold-grouped untouched outer set is still required."
        ),
        "quality_checks": quality_checks,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "temperature": float(args.temperature),
        "threshold": float(args.threshold),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "model_expert_qc_protocol": metadata.get("protocol"),
        "model_architecture_protocol": metadata.get(
            "model_architecture_protocol"
        ),
        "cyclic_relative_positions": bool(
            metadata.get("cyclic_relative_positions")
        ),
        "maximum_equivariance_span_tolerance": (
            V11_MAXIMUM_EQUIVARIANCE_SPAN if cyclic_native_v11 else None
        ),
        "test_jsonl": str(test_path),
        "test_jsonl_sha256": sha256_file(test_path),
        "native_jsonl": str(native_path),
        "native_jsonl_sha256": sha256_file(native_path),
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "best_csv": str(best_path),
        "best_csv_sha256": sha256_file(best_path),
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "dependencies": {
            "clean_v28_common": {
                "path": str(REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"),
                "sha256": sha256_file(REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"),
            },
            "model_utils": {
                "path": str(REPO_ROOT / "model_utils.py"),
                "sha256": sha256_file(REPO_ROOT / "model_utils.py"),
            },
            "nmethyl_config": {
                "path": str(REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"),
                "sha256": sha256_file(REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"),
            },
        },
        "inputs": {
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "test_jsonl": {"path": str(test_path), "sha256": sha256_file(test_path)},
            "native_jsonl": {"path": str(native_path), "sha256": sha256_file(native_path)},
            "best_csv": {"path": str(best_path), "sha256": sha256_file(best_path)},
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        },
        "decoder_order_only_heldout": decoder_summary,
        "cyclic_representation_ensemble_heldout": representation_summary,
        "cyclic_representation_release_floor_heldout": release_floor_summary,
        "native_target_summary": native_summary,
        "annotation_mode": REPRESENTATION_MODE,
        "annotation_context_policy": ANNOTATION_CONTEXT,
        "regenerated_targets": targets,
        "frozen_methylated_targets": sorted(frozen),
        "structure_handoff_status": (
            "BLOCKED_PENDING_V7_RESULT_REVIEW"
            if serine_only
            else (
                "BLOCKED_PENDING_V11_GENERATION_AND_STRUCTURE_REVIEW"
                if cyclic_native_v11
                else "BLOCKED_PENDING_V6_RESULT_REVIEW"
            )
        ),
        "artifacts": {
            "heldout_position_probabilities": {
                "path": str(out_dir / "heldout_position_probabilities.csv"),
                "sha256": sha256_file(
                    out_dir / "heldout_position_probabilities.csv"
                ),
            },
            "native_target_representation_probabilities": {
                "path": str(
                    out_dir / "native_target_representation_probabilities.csv"
                ),
                "sha256": sha256_file(
                    out_dir / "native_target_representation_probabilities.csv"
                ),
            },
            "native_target_representation_summary": {
                "path": str(out_dir / "native_target_representation_summary.csv"),
                "sha256": sha256_file(
                    out_dir / "native_target_representation_summary.csv"
                ),
            },
        },
    }
    atomic_write_json(out_dir / "cyclic_representation_audit.json", report)

    print("===== CYCLIC REPRESENTATION AUDIT COMPLETE =====", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    print(f"Held-out positions: {len(position_rows)}", flush=True)
    print(
        "Representation threshold disagreements before averaging: "
        f"{representation_summary['representation_threshold_disagreement_positions']} / "
        f"{len(position_rows)}",
        flush=True,
    )
    print(f"Authorization: {authorization}", flush=True)
    print(f"Report: {out_dir / 'cyclic_representation_audit.json'}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError(
            "Cyclic representation held-out gate failed: " + ", ".join(failed)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--required-expert-protocol",
        default=REQUIRED_EXPERT_PROTOCOL,
        choices=(
            REQUIRED_EXPERT_PROTOCOL,
            SERINE_ONLY_EXPERT_PROTOCOL,
            V11_EXPERT_PROTOCOL,
        ),
    )
    parser.add_argument("--test-jsonl", default=str(DEFAULT_TEST))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--best-csv", default=str(DEFAULT_BEST))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
