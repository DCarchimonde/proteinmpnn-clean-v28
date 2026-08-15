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
    "cyclic_representation_augmented_v6"
)
REQUIRED_TRAINING_REPRESENTATION_POLICY = (
    "all_physical_cyclic_starts_jointly_rotate_sequence_labels_and_"
    "backbone_coordinates_with_residue_index_reset"
)
REQUIRED_TRAINING_ORDER_POLICY = (
    "all_cyclic_sequence_coordinate_starts_with_epoch_indexed_"
    "decoder_rotation_mapped_to_physical_labels"
)
REQUIRED_DEPLOYMENT_POLICY = (
    "all_cyclic_starts_and_all_decoder_orders_mapped_to_physical_"
    "residues_probability_mean"
)
DECODER_ONLY_MODE = "peptide_only_cyclic_order_ensemble_known_natural_sequence"
REPRESENTATION_MODE = (
    "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
)
ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
SUPPORTED_METHYL_BASES = set(NATURAL_AA_ALPHABET) - {"P"}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    predicted = probability > threshold
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
) -> Dict[str, Any]:
    y_all = np.asarray(
        [int(row["is_methyl_true"]) for row in position_rows], dtype=np.int64
    )
    probability_all = np.asarray(
        [float(row[probability_field]) for row in position_rows], dtype=np.float64
    )
    grouped: MutableMapping[str, List[int]] = defaultdict(list)
    for index, row in enumerate(position_rows):
        grouped[str(row["base_token"])].append(index)
    per_residue: List[Dict[str, Any]] = []
    for base in NATURAL_AA_ALPHABET:
        indices = np.asarray(grouped.get(base, []), dtype=np.int64)
        y = y_all[indices] if len(indices) else np.asarray([], dtype=np.int64)
        p = probability_all[indices] if len(indices) else np.asarray([], dtype=np.float64)
        per_residue.append(
            {
                "base_token": base,
                "positions": int(len(indices)),
                "natural_negatives": int(np.sum(y == 0)),
                "methyl_positives": int(np.sum(y == 1)),
                "auc": roc_auc_score_simple(y, p),
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
        "overall_auc": roc_auc_score_simple(y_all, probability_all),
        "overall_at_threshold": binary_metrics(y_all, probability_all, threshold),
        "non_ser_auc": roc_auc_score_simple(
            y_all[non_ser_indices], probability_all[non_ser_indices]
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
                    position_rows.append(
                        {
                            "sample_name": meta["name"],
                            "batch_index": batch_index,
                            "position_in_peptide_1based": physical_index,
                            "target_token": EXTENDED_AA_ALPHABET[target_index],
                            "base_token": NATURAL_AA_ALPHABET[base_index],
                            "is_methyl_true": int(target_index >= N_NATURAL),
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
                            "probability_representation_span": float(
                                representation["representation_span"][
                                    row_index, position
                                ].item()
                            ),
                            "representation_threshold_disagreement": int(
                                minimum <= threshold < maximum
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
            mapped_by_shift: List[List[float]] = []
            raw_tensor_selected: Counter[int] = Counter()
            mapped_annotation_sets: List[List[int]] = []
            for shift in range(length):
                raw = raw_probability[shift].detach().cpu().tolist()
                mapped = np.roll(np.asarray(raw, dtype=np.float64), shift).tolist()
                mapped_by_shift.append(mapped)
                raw_selected = [
                    index + 1 for index, value in enumerate(raw) if value > threshold
                ]
                raw_tensor_selected.update(raw_selected)
                mapped_annotation_sets.append(
                    [index + 1 for index, value in enumerate(mapped) if value > threshold]
                )
                for physical_position, value in enumerate(mapped, start=1):
                    tensor_position = (physical_position - shift - 1) % length + 1
                    detail_rows.append(
                        {
                            "target_name": target,
                            "native_sequence": sequence,
                            "representation_left_shift": shift,
                            "physical_position_1based": physical_position,
                            "tensor_position_1based": tensor_position,
                            "base_token": sequence[physical_position - 1],
                            "mapped_probability": value,
                            "above_threshold": int(value > threshold),
                        }
                    )
            recomputed = np.mean(np.asarray(mapped_by_shift), axis=0)
            reported = ensemble["mean"][0].detach().cpu().numpy()
            maximum_recompute_difference = float(np.max(np.abs(recomputed - reported)))
            ensemble_selected = [
                index + 1 for index, value in enumerate(reported) if value > threshold
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
                    "raw_tensor_position_counts_above_threshold": json.dumps(
                        dict(sorted(raw_tensor_selected.items()))
                    ),
                    "raw_maximum_single_tensor_position_share": (
                        most_common_tensor_count / sum(raw_tensor_selected.values())
                        if raw_tensor_selected
                        else 0.0
                    ),
                    "maximum_ensemble_recompute_difference": maximum_recompute_difference,
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
                }
            )
    return detail_rows, summary_rows


def checkpoint_metadata(model_path: Path) -> Dict[str, Any]:
    payload = torch.load(model_path, map_location="cpu")
    metadata = (
        dict(payload.get("expert_head_qc_metadata", {}))
        if isinstance(payload, Mapping)
        else {}
    )
    del payload
    if not (
        str(metadata.get("protocol", "")) == REQUIRED_EXPERT_PROTOCOL
        and int(metadata.get("minimum_order_coverage_epochs", 0)) >= 30
        and bool(metadata.get("cyclic_representation_augmentation"))
        and str(metadata.get("training_cyclic_representation_policy", ""))
        == REQUIRED_TRAINING_REPRESENTATION_POLICY
        and str(metadata.get("training_decoding_order_policy", ""))
        == REQUIRED_TRAINING_ORDER_POLICY
        and str(metadata.get("deployment_annotation_policy", ""))
        == REQUIRED_DEPLOYMENT_POLICY
    ):
        raise RuntimeError(
            "Representation audit requires a promoted V6 checkpoint trained "
            "with all physical cyclic starts; the decoder-order-only V3 "
            "checkpoint is not sufficient"
        )
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

    metadata = checkpoint_metadata(model_path)
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
        f"Loading promoted cyclic-representation V6 expert checkpoint: {model_path}",
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

    overall = representation_summary["overall_at_threshold"]
    serine = representation_summary["serine"]
    proline = representation_summary["proline"]
    quality_checks = {
        "v3_checkpoint_provenance_is_pinned": (
            metadata.get("protocol") == REQUIRED_EXPERT_PROTOCOL
        ),
        "heldout_test_record_count_is_151": len(test_records) == 151,
        "heldout_test_position_count_is_1505": len(position_rows) == 1505,
        "overall_auc_ge_0_85": float(representation_summary["overall_auc"]) >= 0.85,
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
            representation_summary["supported_macro_auc"] is not None
            and float(representation_summary["supported_macro_auc"]) >= 0.70
        ),
        "all_17_targets_audited_for_uniform_regeneration": (
            len(native_summary) == len(targets) == 17
        ),
        "every_native_target_used_all_cyclic_starts": all(
            int(row["representation_count"]) == int(row["peptide_length"])
            for row in native_summary
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
            )
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    authorization = (
        "REPRESENTATION_ENSEMBLE_VALIDATED_FOR_ISOLATED_V6_REGENERATION"
        if quality_gate == "PASS"
        else "BLOCKED_DO_NOT_REGENERATE_OR_RELEASE"
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
        "protocol": "cyclic_representation_equivariance_heldout_gate_v1",
        "scientific_scope": (
            "Outer ensemble jointly rotates sequence and N/CA/C/O coordinates, resets "
            "linear residue indices, maps probabilities back to physical residues, and "
            "then averages. This repairs arbitrary cyclic-start dependence; it does not "
            "is paired with a checkpoint trained on every cyclic start rather than "
            "claiming that the decoder-order-only V3 checkpoint was equivariant."
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
        "test_jsonl": str(test_path),
        "test_jsonl_sha256": sha256_file(test_path),
        "native_jsonl": str(native_path),
        "native_jsonl_sha256": sha256_file(native_path),
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "decoder_order_only_heldout": decoder_summary,
        "cyclic_representation_ensemble_heldout": representation_summary,
        "native_target_summary": native_summary,
        "annotation_mode": REPRESENTATION_MODE,
        "annotation_context_policy": ANNOTATION_CONTEXT,
        "regenerated_targets": targets,
        "frozen_methylated_targets": sorted(frozen),
        "structure_handoff_status": "BLOCKED_PENDING_V6_RESULT_REVIEW",
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
