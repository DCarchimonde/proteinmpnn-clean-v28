#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrain clean-V28 expert heads on provenance-corrected labels.

The shared ProteinMPNN trunk and the natural-amino-acid base head remain frozen.
The historical V3/V6 mode optimizes all twenty residue experts together.  The
V7 ``serine-only`` mode is narrower: provenance correction changed only the
ordinary-ATOM Ser label, so it optimizes only the Ser expert and requires every
other tensor (including the other nineteen experts) to remain bitwise identical
to the canonical parent.  There is no surrogate network or post-hoc weight
splicing.  Lowercase target tokens are naturalized before every model forward
so the methylation answer can never leak through the sequence embedding.

Every training epoch receives an explicit cyclic designed-position rotation;
the 30-epoch minimum covers every possible relative depth allowed by the frozen
30-residue peptide cap.  The V6 option additionally enumerates every equivalent
physical cyclic start by jointly rotating sequence/labels and N/CA/C/O
coordinates while resetting the linear residue index.  Validation, test
promotion, and downstream annotation then use the matching all-start and
all-decoder-order ensemble mapped back to physical residues.

The corrected 600-record training split is divided deterministically into a
development-train and record-disjoint validation partition.  The original 151
records are not accessed until epoch selection has finished.  Checkpoint
promotion is blocked unless every non-expert tensor is bitwise identical to the
parent and the fixed validation/test gates pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nmethyl.utils.nmethyl_config import (  # noqa: E402
    EXTENDED_AA_ALPHABET,
    METHYL_AA_ALPHABET,
    NATURAL_AA_ALPHABET,
)
from paper_clean_v28.clean_v28_common import (  # noqa: E402
    N_NATURAL,
    X_INDEX,
    binary_metrics,
    cyclic_designed_decoding_order,
    cyclic_known_sequence_methyl_probabilities,
    cyclic_representation_known_sequence_methyl_probabilities,
    featurize_records,
    load_v28_model,
    naturalize_tensor_for_input,
    read_jsonl,
    roc_auc_score_simple,
)


DEFAULT_DATA_DIR = (
    REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_peptide_only_v4" / "data"
)
DEFAULT_OUT = (
    REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_peptide_only_v4" / "model"
)
EXPECTED_TRAIN_COUNTS = {"S": 242, "s": 50, "P": 307, "p": 0}
EXPECTED_TEST_COUNTS = {"S": 62, "s": 12, "P": 83, "p": 0}
SUPPORTED_METHYL_BASES = {token.upper() for token in METHYL_AA_ALPHABET}
ALL_EXPERT_STATE_KEYS = {
    f"experts.{index}.{suffix}"
    for index in range(len(NATURAL_AA_ALPHABET))
    for suffix in ("weight", "bias")
}
ORDER_BALANCED_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_order_balanced_v3"
)
CYCLIC_REPRESENTATION_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_"
    "cyclic_representation_augmented_v6"
)
SERINE_ONLY_CYCLIC_PROTOCOL = (
    "canonical_clean_v28_serine_only_corrected_labels_"
    "cyclic_representation_augmented_v7"
)
SERINE_EXPERT_INDEX = NATURAL_AA_ALPHABET.index("S")
SERINE_EXPERT_STATE_KEYS = {
    f"experts.{SERINE_EXPERT_INDEX}.{suffix}" for suffix in ("weight", "bias")
}
MINIMUM_ORDER_COVERAGE_EPOCHS = 30
VALIDATION_INTERVAL_EPOCHS = 5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hashes(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, str]:
    return {key: tensor_sha256(value) for key, value in sorted(state_dict.items())}


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


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sequence_counts(records: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for key, value in record.items():
            if key.startswith("seq_chain_"):
                counts.update(str(value))
    return counts


def per_base_binary_counts(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Count natural negatives and N-methyl positives for every parent residue."""

    counts = {
        base: {"natural_negative": 0, "methyl_positive": 0}
        for base in NATURAL_AA_ALPHABET
    }
    for record in records:
        for key, value in record.items():
            if not key.startswith("seq_chain_"):
                continue
            for token in str(value):
                if token in NATURAL_AA_ALPHABET:
                    counts[token]["natural_negative"] += 1
                elif token in METHYL_AA_ALPHABET:
                    counts[token.upper()]["methyl_positive"] += 1
    return counts


def cyclic_augmented_per_base_binary_counts(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Count labels after every physical cyclic start is used once.

    A length-L single-chain record contributes L equivalent serializations, so
    each physical label must be observed exactly L times in an augmented epoch.
    """

    counts = {
        base: {"natural_negative": 0, "methyl_positive": 0}
        for base in NATURAL_AA_ALPHABET
    }
    for record in records:
        sequence_keys = [
            key
            for key, value in record.items()
            if key.startswith("seq_chain_") and str(value)
        ]
        if len(sequence_keys) != 1:
            raise RuntimeError(
                "Cyclic representation augmentation requires one peptide chain"
            )
        sequence = str(record[sequence_keys[0]])
        length = len(sequence)
        for token in sequence:
            if token in NATURAL_AA_ALPHABET:
                counts[token]["natural_negative"] += length
            elif token in METHYL_AA_ALPHABET:
                counts[token.upper()]["methyl_positive"] += length
    return counts


def expand_all_cyclic_training_representations(
    X: torch.Tensor,
    S_label: torch.Tensor,
    mask: torch.Tensor,
    chain_M: torch.Tensor,
    residue_idx: torch.Tensor,
    chain_encoding_all: torch.Tensor,
    real_pos: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Jointly rotate coordinates and labels through every cyclic start.

    The model sees a fresh linear ``0..L-1`` residue index for every equivalent
    serialization.  Labels and coordinates are rolled together, so the target
    remains attached to the same physical residue rather than the same tensor
    column.  Padding is retained but never rotated.
    """

    selected_mask = (mask * chain_M) > 0
    if bool(((mask > 0) & ~selected_mask).any()):
        raise RuntimeError(
            "Cyclic representation training requires peptide-only input with "
            "zero visible receptor positions"
        )

    expanded: List[List[torch.Tensor]] = [[] for _ in range(7)]
    for row_index in range(S_label.shape[0]):
        positions = torch.where(selected_mask[row_index])[0]
        length = int(positions.numel())
        if length <= 0:
            raise RuntimeError("Cyclic representation training row is empty")
        chain_values = torch.unique(chain_encoding_all[row_index, positions])
        if int(chain_values.numel()) != 1:
            raise RuntimeError(
                "Cyclic representation training row contains multiple chains"
            )
        canonical_residue_idx = torch.arange(
            length,
            device=residue_idx.device,
            dtype=residue_idx.dtype,
        )
        for shift in range(length):
            row_X = X[row_index].clone()
            row_S = S_label[row_index].clone()
            row_mask = mask[row_index].clone()
            row_chain_M = chain_M[row_index].clone()
            row_residue_idx = residue_idx[row_index].clone()
            row_chain_encoding = chain_encoding_all[row_index].clone()
            row_real_pos = real_pos[row_index].clone()
            row_X[positions] = torch.roll(
                X[row_index, positions], shifts=-shift, dims=0
            )
            row_S[positions] = torch.roll(
                S_label[row_index, positions], shifts=-shift, dims=0
            )
            row_mask[positions] = torch.roll(
                mask[row_index, positions], shifts=-shift, dims=0
            )
            row_chain_M[positions] = torch.roll(
                chain_M[row_index, positions], shifts=-shift, dims=0
            )
            row_real_pos[positions] = torch.roll(
                real_pos[row_index, positions], shifts=-shift, dims=0
            )
            row_residue_idx[positions] = canonical_residue_idx
            row_chain_encoding[positions] = chain_values[0]
            for bucket, value in zip(
                expanded,
                (
                    row_X,
                    row_S,
                    row_mask,
                    row_chain_M,
                    row_residue_idx,
                    row_chain_encoding,
                    row_real_pos,
                ),
            ):
                bucket.append(value)
    return tuple(torch.stack(values, dim=0) for values in expanded)


def record_name(record: Mapping[str, Any], fallback: int) -> str:
    return str(
        record.get("name")
        or record.get("pdb")
        or record.get("pdb_id")
        or record.get("id")
        or f"record_{fallback}"
    )


def require_peptide_only_training_context(
    records: Sequence[Mapping[str, Any]], split_name: str
) -> Dict[str, Any]:
    """Block training if any record contains a visible receptor or second chain."""

    peptide_lengths: List[int] = []
    for index, record in enumerate(records):
        chain_ids = sorted(
            key[len("seq_chain_") :]
            for key in record
            if key.startswith("seq_chain_") and str(record[key])
        )
        masked = [str(value) for value in record.get("masked_list", [])]
        visible = [str(value) for value in record.get("visible_list", [])]
        if (
            len(chain_ids) != 1
            or masked != chain_ids
            or visible
            or int(record.get("num_of_chains", 1)) != 1
        ):
            raise RuntimeError(
                f"{split_name} record {record_name(record, index)} is not a "
                "single masked peptide with zero visible receptor chains"
            )
        peptide_lengths.append(len(str(record[f"seq_chain_{chain_ids[0]}"])))
    if not peptide_lengths or min(peptide_lengths) <= 0:
        raise RuntimeError(f"{split_name} has no non-empty peptide-only records")
    return {
        "rows": len(records),
        "minimum_peptide_length": min(peptide_lengths),
        "maximum_peptide_length": max(peptide_lengths),
        "visible_receptor_chains": 0,
        "chains_per_record": 1,
        "context_policy": "peptide_chain_only_no_visible_receptor_chains",
    }


def deterministic_train_validation_split(
    records: Sequence[Mapping[str, Any]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Create a record-disjoint split with both classes for every supported expert.

    The test set is not involved.  We try deterministic seed offsets until the
    validation subset contains at least one positive and one negative example
    for each of the nineteen supported N-methyl residue types; the remaining
    development records must satisfy the same condition.
    """

    if not 0.05 <= validation_fraction <= 0.40:
        raise ValueError("validation_fraction must be between 0.05 and 0.40")
    if len(records) < 2:
        raise ValueError("At least two training records are required")
    validation_size = max(1, min(len(records) - 1, round(len(records) * validation_fraction)))
    source = [dict(record) for record in records]

    for attempt in range(10_000):
        order = list(range(len(source)))
        random.Random(seed + attempt).shuffle(order)
        validation_indices = set(order[:validation_size])
        development = [row for index, row in enumerate(source) if index not in validation_indices]
        validation = [row for index, row in enumerate(source) if index in validation_indices]
        development_counts = per_base_binary_counts(development)
        validation_counts = per_base_binary_counts(validation)

        supported = sorted(SUPPORTED_METHYL_BASES)
        if all(
            development_counts[base]["natural_negative"] > 0
            and development_counts[base]["methyl_positive"] > 0
            and validation_counts[base]["natural_negative"] > 0
            and validation_counts[base]["methyl_positive"] > 0
            for base in supported
        ):
            development_names = {
                record_name(row, index) for index, row in enumerate(development)
            }
            validation_names = {
                record_name(row, index) for index, row in enumerate(validation)
            }
            if development_names & validation_names:
                raise RuntimeError("Development/validation record-name overlap")
            manifest = {
                "method": "deterministic_record_split_with_per_expert_class_support",
                "seed": seed,
                "accepted_seed_offset": attempt,
                "validation_fraction_requested": validation_fraction,
                "development_records": len(development),
                "validation_records": len(validation),
                "development_record_names": sorted(development_names),
                "validation_record_names": sorted(validation_names),
                "development_counts_by_base": development_counts,
                "validation_counts_by_base": validation_counts,
            }
            return development, validation, manifest

    raise RuntimeError(
        "Could not create a deterministic record-disjoint validation split "
        "with positive/negative support for every methyl expert"
    )


def require_corrected_counts(
    records: Sequence[Mapping[str, Any]], expected: Mapping[str, int], label: str
) -> Counter[str]:
    counts = sequence_counts(records)
    observed = {token: counts[token] for token in expected}
    if observed != dict(expected):
        raise RuntimeError(
            f"{label} is not the pinned provenance-corrected dataset: "
            f"expected {dict(expected)}, observed {observed}"
        )
    return counts


def batches(records: Sequence[Mapping[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield [dict(value) for value in records[start : start + batch_size]]


def normalize_active_base_indices(
    active_base_indices: Sequence[int] | None,
) -> Tuple[int, ...]:
    indices = tuple(
        range(len(NATURAL_AA_ALPHABET))
        if active_base_indices is None
        else sorted({int(value) for value in active_base_indices})
    )
    if not indices or any(
        value < 0 or value >= len(NATURAL_AA_ALPHABET) for value in indices
    ):
        raise ValueError("active expert indices are empty or out of range")
    return indices


def trainable_expert_parameters(
    model: torch.nn.Module,
    active_base_indices: Sequence[int] | None = None,
) -> List[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if active_base_indices is None:
        # Preserve the original V3/V6 all-expert path exactly.
        parameters = list(model.experts.parameters())
    else:
        indices = normalize_active_base_indices(active_base_indices)
        parameters = [
            parameter
            for index in indices
            for parameter in model.experts[index].parameters()
        ]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def positive_weights_by_base(
    records: Sequence[Mapping[str, Any]],
    active_base_indices: Sequence[int] | None = None,
) -> Dict[int, float]:
    counts = per_base_binary_counts(records)
    result: Dict[int, float] = {}
    for base_index in normalize_active_base_indices(active_base_indices):
        base = NATURAL_AA_ALPHABET[base_index]
        negatives = int(counts[base]["natural_negative"])
        positives = int(counts[base]["methyl_positive"])
        if negatives <= 0:
            raise RuntimeError(f"Expert {base} has no natural negative in development training")
        if base in SUPPORTED_METHYL_BASES and positives <= 0:
            raise RuntimeError(f"Supported expert {base} has no methyl positive")
        # Pro has no lowercase p class.  It is trained only as a negative veto,
        # and generation has no natural-to-methyl mapping for P.
        result[base_index] = negatives / positives if positives > 0 else 1.0
    return result


def expert_head_loss(
    expert_logits: torch.Tensor,
    S_label: torch.Tensor,
    valid: torch.Tensor,
    positive_weights: Mapping[int, float],
    active_base_indices: Sequence[int] | None = None,
) -> Tuple[torch.Tensor, Dict[int, Tuple[int, int]]]:
    true_base = naturalize_tensor_for_input(S_label)
    losses: List[torch.Tensor] = []
    coverage: Dict[int, Tuple[int, int]] = {}
    for base_index in normalize_active_base_indices(active_base_indices):
        selected = valid & (true_base == base_index)
        if not bool(selected.any()):
            continue
        labels = (S_label[selected] >= N_NATURAL).to(dtype=torch.float32)
        logits = expert_logits[..., base_index][selected]
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=torch.tensor(
                    float(positive_weights[base_index]), device=expert_logits.device
                ),
            )
        )
        positive_count = int(labels.sum().item())
        coverage[base_index] = (int(labels.numel()) - positive_count, positive_count)
    if not losses:
        raise RuntimeError("No valid expert-head positions in batch")
    # Equal weight per residue expert, rather than allowing common residues to
    # dominate the optimization merely because they have more positions.
    return torch.stack(losses).mean(), coverage


def expert_probability_loss(
    probabilities: torch.Tensor,
    S_label: torch.Tensor,
    valid: torch.Tensor,
    positive_weights: Mapping[int, float],
    active_base_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Balanced BCE on deterministic deployment-ensemble probabilities."""

    true_base = naturalize_tensor_for_input(S_label)
    losses: List[torch.Tensor] = []
    for base_index in normalize_active_base_indices(active_base_indices):
        selected = valid & (true_base == base_index)
        if not bool(selected.any()):
            continue
        labels = (S_label[selected] >= N_NATURAL).to(dtype=torch.float32)
        probability = probabilities[selected].clamp(1e-7, 1.0 - 1e-7)
        positive_weight = float(positive_weights[base_index])
        losses.append(
            -(
                positive_weight * labels * torch.log(probability)
                + (1.0 - labels) * torch.log1p(-probability)
            ).mean()
        )
    if not losses:
        raise RuntimeError("No valid ensemble expert-head positions in batch")
    return torch.stack(losses).mean()


def validation_balanced_bce(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    positive_weights: Mapping[int, float],
    cyclic_representation_ensemble: bool = False,
    active_base_indices: Sequence[int] | None = None,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in batches(records, batch_size):
            packed = featurize_records(batch, device=device, eval_chains="masked")
            if packed is None:
                continue
            tensors, _metas = packed
            X, S_label, mask, chain_M, residue_idx, chain_encoding_all, real_pos = tensors
            valid = (
                (mask > 0)
                & (chain_M > 0)
                & (real_pos > 0)
                & (S_label != X_INDEX)
            )
            S_forward = naturalize_tensor_for_input(S_label)
            if cyclic_representation_ensemble:
                representation = (
                    cyclic_representation_known_sequence_methyl_probabilities(
                        model,
                        X,
                        S_forward,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                        temperature=1.0,
                    )
                )
                probabilities = representation["mean"]
            else:
                probabilities, _order_std = (
                    cyclic_known_sequence_methyl_probabilities(
                        model,
                        X,
                        S_forward,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                        temperature=1.0,
                    )
                )
            loss = expert_probability_loss(
                probabilities,
                S_label,
                valid,
                positive_weights,
                active_base_indices=active_base_indices,
            )
            losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("Validation produced no expert-head loss")
    return float(sum(losses) / len(losses))


def train_all_expert_heads(
    model: torch.nn.Module,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    cyclic_representation_augmentation: bool = False,
    active_base_indices: Sequence[int] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], Dict[str, Any]]:
    active_indices = normalize_active_base_indices(active_base_indices)
    active_state_keys = {
        f"experts.{index}.{suffix}"
        for index in active_indices
        for suffix in ("weight", "bias")
    }
    expert_parameters = trainable_expert_parameters(model, active_indices)
    positive_weights = positive_weights_by_base(train_records, active_indices)
    optimizer = torch.optim.AdamW(
        expert_parameters, lr=learning_rate, weight_decay=1e-4
    )
    history: List[Dict[str, Any]] = []
    best_validation = math.inf
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    # The frozen trunk is always evaluated with dropout/augmentation disabled.
    model.eval()
    expected_counts = (
        cyclic_augmented_per_base_binary_counts(train_records)
        if cyclic_representation_augmentation
        else per_base_binary_counts(train_records)
    )
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_records)))
        random.Random(seed + epoch).shuffle(order)
        shuffled = [train_records[index] for index in order]
        batch_losses: List[float] = []
        epoch_coverage = {
            index: [0, 0] for index in active_indices
        }

        for batch in batches(shuffled, batch_size):
            packed = featurize_records(batch, device=device, eval_chains="masked")
            if packed is None:
                continue
            tensors, _metas = packed
            X, S_label, mask, chain_M, residue_idx, chain_encoding_all, real_pos = tensors
            if cyclic_representation_augmentation:
                (
                    X,
                    S_label,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                    real_pos,
                ) = expand_all_cyclic_training_representations(
                    X,
                    S_label,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                    real_pos,
                )
            valid = (
                (mask > 0)
                & (chain_M > 0)
                & (real_pos > 0)
                & (S_label != X_INDEX)
            )
            optimizer.zero_grad(set_to_none=True)
            S_forward = naturalize_tensor_for_input(S_label)
            decoding_order = cyclic_designed_decoding_order(
                chain_M,
                mask,
                shift=epoch - 1,
            )
            _base_logits, expert_logits = model(
                X,
                S_forward,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
                decoding_order=decoding_order,
            )
            loss, coverage = expert_head_loss(
                expert_logits,
                S_label,
                valid,
                positive_weights,
                active_base_indices=active_indices,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert_parameters, 5.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
            for base_index, (negative_count, positive_count) in coverage.items():
                epoch_coverage[base_index][0] += negative_count
                epoch_coverage[base_index][1] += positive_count

        for base_index in active_indices:
            base = NATURAL_AA_ALPHABET[base_index]
            observed = epoch_coverage[base_index]
            expected = [
                int(expected_counts[base]["natural_negative"]),
                int(expected_counts[base]["methyl_positive"]),
            ]
            if observed != expected:
                raise RuntimeError(
                    f"Epoch {epoch} coverage changed for {base}: "
                    f"expected {expected}, observed {observed}"
                )

        mean_train_loss = float(sum(batch_losses) / len(batch_losses))
        order_coverage_complete = epoch >= MINIMUM_ORDER_COVERAGE_EPOCHS
        should_validate = order_coverage_complete and (
            epoch == MINIMUM_ORDER_COVERAGE_EPOCHS
            or epoch % VALIDATION_INTERVAL_EPOCHS == 0
            or epoch == epochs
        )
        validation_loss = (
            validation_balanced_bce(
                model,
                validation_records,
                device,
                batch_size,
                positive_weights,
                cyclic_representation_ensemble=cyclic_representation_augmentation,
                active_base_indices=active_indices,
            )
            if should_validate
            else None
        )
        improved = (
            validation_loss is not None
            and validation_loss < best_validation - 1e-6
        )
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if key in active_state_keys
            }
            epochs_without_improvement = 0
        elif should_validate:
            epochs_without_improvement += VALIDATION_INTERVAL_EPOCHS

        row = {
            "epoch": epoch,
            "mean_balanced_train_bce": mean_train_loss,
            "validation_balanced_bce": (
                validation_loss if validation_loss is not None else ""
            ),
            "validation_evaluated": int(should_validate),
            "is_best_epoch": int(improved),
            "order_coverage_complete": int(order_coverage_complete),
            "all_cyclic_representations_used": int(
                cyclic_representation_augmentation
            ),
            "epochs_without_improvement": epochs_without_improvement,
            "learning_rate": learning_rate,
        }
        history.append(row)
        validation_text = (
            f"{validation_loss:.6f}" if validation_loss is not None else "DEFERRED"
        )
        best_text = f"{best_validation:.6f}" if math.isfinite(best_validation) else "PENDING"
        print(
            f"Epoch {epoch:03d}/{epochs}: train={mean_train_loss:.6f} "
            f"validation={validation_text} best={best_text}",
            flush=True,
        )
        if (
            epoch >= MINIMUM_ORDER_COVERAGE_EPOCHS
            and epochs_without_improvement >= patience
        ):
            print(f"Early stopping after epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No validation-selected expert-head state was produced")
    current_state = model.state_dict()
    for key, value in best_state.items():
        current_state[key].copy_(value.to(device=current_state[key].device))
    return history, best_state, {
        "best_epoch": best_epoch,
        "best_validation_balanced_bce": best_validation,
        "epochs_ran": len(history),
        "early_stopping_patience": patience,
        "validation_interval_epochs": VALIDATION_INTERVAL_EPOCHS,
        "positive_weights_by_base": {
            NATURAL_AA_ALPHABET[index]: value
            for index, value in positive_weights.items()
        },
        "cyclic_representation_augmentation": bool(
            cyclic_representation_augmentation
        ),
        "active_expert_indices": list(active_indices),
        "active_expert_tokens": [NATURAL_AA_ALPHABET[index] for index in active_indices],
    }


def evaluate(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    threshold: float,
    deployment_temperature: float,
    checkpoint_label: str,
    cyclic_representation_ensemble: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    position_rows: List[Dict[str, Any]] = []
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
            # Prevent target leakage: lowercase methyl labels are evaluation
            # targets only, never inputs to the canonical model trunk.
            S_forward = naturalize_tensor_for_input(S_label)
            if cyclic_representation_ensemble:
                representation = (
                    cyclic_representation_known_sequence_methyl_probabilities(
                        model,
                        X,
                        S_forward,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                        temperature=deployment_temperature,
                    )
                )
                probability = representation["mean"]
                order_std = representation["decoder_order_std_mean"]
                representation_std = representation["representation_std"]
                representation_span = representation["representation_span"]
            else:
                probability, order_std = cyclic_known_sequence_methyl_probabilities(
                    model,
                    X,
                    S_forward,
                    mask,
                    chain_M,
                    residue_idx,
                    chain_encoding_all,
                    temperature=deployment_temperature,
                )
                representation_std = torch.zeros_like(probability)
                representation_span = torch.zeros_like(probability)
            true_base = naturalize_tensor_for_input(S_label)

            for row_index, meta in enumerate(metas):
                ensemble_size = int(valid[row_index].sum().item())
                for position in torch.where(valid[row_index])[0].cpu().tolist():
                    base_index = int(true_base[row_index, position].item())
                    target_index = int(S_label[row_index, position].item())
                    position_rows.append(
                        {
                            "checkpoint": checkpoint_label,
                            "sample_name": meta["name"],
                            "batch_index": batch_index,
                            "position_in_model_0based": position,
                            "target_token": EXTENDED_AA_ALPHABET[target_index],
                            "base_token": NATURAL_AA_ALPHABET[base_index],
                            "is_methyl_true": int(target_index >= N_NATURAL),
                            "probability_methyl_deployment_scaled": float(
                                probability[row_index, position].item()
                            ),
                            "probability_order_std": float(
                                order_std[row_index, position].item()
                            ),
                            "probability_representation_std": float(
                                representation_std[row_index, position].item()
                            ),
                            "probability_representation_span": float(
                                representation_span[row_index, position].item()
                            ),
                            "annotation_mode": (
                                "peptide_only_all_cyclic_starts_and_decoder_orders_"
                                "mapped_to_physical_residues"
                                if cyclic_representation_ensemble
                                else "peptide_only_cyclic_order_ensemble_"
                                "known_natural_sequence"
                            ),
                            "annotation_context_policy": (
                                "peptide_chain_only_no_visible_receptor_chains"
                            ),
                            "annotation_visible_receptor_chains": 0,
                            "annotation_order_ensemble_size": ensemble_size,
                            "annotation_representation_ensemble_size": (
                                ensemble_size if cyclic_representation_ensemble else 1
                            ),
                        }
                    )

    y_all = np.asarray([row["is_methyl_true"] for row in position_rows], dtype=np.int64)
    p_all = np.asarray(
        [row["probability_methyl_deployment_scaled"] for row in position_rows],
        dtype=np.float64,
    )
    order_std_all = np.asarray(
        [row["probability_order_std"] for row in position_rows],
        dtype=np.float64,
    )
    representation_span_all = np.asarray(
        [row["probability_representation_span"] for row in position_rows],
        dtype=np.float64,
    )
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(position_rows):
        grouped_indices[str(row["base_token"])].append(index)

    per_residue: List[Dict[str, Any]] = []
    for base_token in NATURAL_AA_ALPHABET:
        idx = np.asarray(grouped_indices.get(base_token, []), dtype=np.int64)
        y = y_all[idx] if len(idx) else np.asarray([], dtype=np.int64)
        p = p_all[idx] if len(idx) else np.asarray([], dtype=np.float64)
        threshold_metrics = binary_metrics(y, p, [threshold])[0] if len(idx) else {}
        per_residue.append(
            {
                "checkpoint": checkpoint_label,
                "base_token": base_token,
                "positions": int(len(idx)),
                "natural_negatives": int(np.sum(y == 0)),
                "methyl_positives": int(np.sum(y == 1)),
                "auc": roc_auc_score_simple(y, p),
                **threshold_metrics,
            }
        )

    serine = next(row for row in per_residue if row["base_token"] == "S")
    proline = next(row for row in per_residue if row["base_token"] == "P")
    supported_rows = [
        row
        for row in per_residue
        if row["base_token"] in SUPPORTED_METHYL_BASES and row["auc"] is not None
    ]
    non_ser_idx = np.asarray(
        [index for index, row in enumerate(position_rows) if row["base_token"] != "S"],
        dtype=np.int64,
    )
    overall_threshold = binary_metrics(y_all, p_all, [threshold])[0]
    non_ser_threshold = binary_metrics(
        y_all[non_ser_idx], p_all[non_ser_idx], [threshold]
    )[0]
    summary = {
        "checkpoint": checkpoint_label,
        "positions": len(position_rows),
        "threshold": threshold,
        "deployment_temperature": deployment_temperature,
        "overall_auc": roc_auc_score_simple(y_all, p_all),
        "overall_at_threshold": overall_threshold,
        "non_ser_auc": roc_auc_score_simple(y_all[non_ser_idx], p_all[non_ser_idx]),
        "non_ser_at_threshold": non_ser_threshold,
        "supported_expert_count": len(supported_rows),
        "supported_macro_auc": float(
            np.mean([float(row["auc"]) for row in supported_rows])
        ) if supported_rows else None,
        "supported_macro_f1_at_threshold": float(
            np.mean([float(row["f1"]) for row in supported_rows])
        ) if supported_rows else None,
        "maximum_probability_order_std": float(np.max(order_std_all)),
        "mean_probability_order_std": float(np.mean(order_std_all)),
        "maximum_probability_representation_span": float(
            np.max(representation_span_all)
        ),
        "mean_probability_representation_span": float(
            np.mean(representation_span_all)
        ),
        "serine": serine,
        "proline": proline,
    }
    return summary, per_residue, position_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(REPO_ROOT / "frankenstein_v28.pt"))
    parser.add_argument(
        "--train-jsonl",
        default=str(DEFAULT_DATA_DIR / "train_serine_provenance_corrected.jsonl"),
    )
    parser.add_argument(
        "--test-jsonl",
        default=str(DEFAULT_DATA_DIR / "test_serine_provenance_corrected.jsonl"),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--deployment-temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--expert-scope",
        choices=("all", "serine-only"),
        default="all",
        help=(
            "all preserves the historical V3/V6 workflow; serine-only is the "
            "V7 provenance-scoped repair and keeps every non-Ser tensor bitwise "
            "identical to the canonical parent"
        ),
    )
    parser.add_argument(
        "--cyclic-representation-augmentation",
        action="store_true",
        help=(
            "jointly rotate sequence/labels and N/CA/C/O coordinates through "
            "every physical cyclic start during training, and use the matching "
            "all-start deployment ensemble for validation and test"
        ),
    )
    parser.add_argument("--no-fail-on-quality-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serine_only = args.expert_scope == "serine-only"
    if serine_only and not args.cyclic_representation_augmentation:
        raise ValueError(
            "--expert-scope serine-only requires "
            "--cyclic-representation-augmentation"
        )
    active_indices = (
        (SERINE_EXPERT_INDEX,)
        if serine_only
        else tuple(range(len(NATURAL_AA_ALPHABET)))
    )
    expected_changed_keys = (
        SERINE_EXPERT_STATE_KEYS if serine_only else ALL_EXPERT_STATE_KEYS
    )
    if (
        args.epochs < MINIMUM_ORDER_COVERAGE_EPOCHS
        or args.batch_size <= 0
        or args.learning_rate <= 0
        or args.early_stopping_patience <= 0
    ):
        raise ValueError(
            f"epochs must be at least {MINIMUM_ORDER_COVERAGE_EPOCHS}; batch-size, "
            "learning-rate, and early-stopping-patience must be positive"
        )
    if not 0.0 < args.threshold < 1.0 or args.deployment_temperature <= 0.0:
        raise ValueError(
            "threshold must be between zero and one and deployment-temperature "
            "must be positive"
        )

    model_path = Path(args.model_path).resolve()
    train_path = Path(args.train_jsonl).resolve()
    test_path = Path(args.test_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    for required in (model_path, train_path, test_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_deterministic_seed(args.seed)

    train_records = read_jsonl(str(train_path))
    test_records = read_jsonl(str(test_path))
    train_context_audit = require_peptide_only_training_context(train_records, "train")
    test_context_audit = require_peptide_only_training_context(test_records, "test")
    require_corrected_counts(train_records, EXPECTED_TRAIN_COUNTS, "train")
    require_corrected_counts(test_records, EXPECTED_TEST_COUNTS, "test")
    development_records, validation_records, split_manifest = (
        deterministic_train_validation_split(
            train_records,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    )

    print(f"Loading canonical clean-V28 checkpoint: {model_path}", flush=True)
    model = load_v28_model(str(model_path), device)
    before_hashes = state_hashes(model.state_dict())
    parent_summary: Dict[str, Any] | None = None
    parent_positions: List[Dict[str, Any]] | None = None
    baseline_validation_loss = validation_balanced_bce(
        model,
        validation_records,
        device,
        args.batch_size,
        positive_weights_by_base(development_records, active_indices),
        cyclic_representation_ensemble=args.cyclic_representation_augmentation,
        active_base_indices=active_indices,
    )
    history, _best_expert_state, training_selection = train_all_expert_heads(
        model,
        development_records,
        validation_records,
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.early_stopping_patience,
        args.seed,
        cyclic_representation_augmentation=args.cyclic_representation_augmentation,
        active_base_indices=active_indices,
    )
    after_hashes = state_hashes(model.state_dict())
    changed_keys = sorted(
        key for key in before_hashes if before_hashes[key] != after_hashes[key]
    )
    changed_non_expert_keys = sorted(set(changed_keys) - ALL_EXPERT_STATE_KEYS)
    unexpected_changed_keys = sorted(set(changed_keys) - expected_changed_keys)
    unchanged_expected_keys = sorted(expected_changed_keys - set(changed_keys))
    if not serine_only and set(changed_keys) != ALL_EXPERT_STATE_KEYS:
        raise RuntimeError(
            "All-expert state isolation failed: expected exactly the 40 expert "
            f"tensors to change; missing={unchanged_expected_keys}, "
            f"unexpected={unexpected_changed_keys}"
        )
    if serine_only and set(changed_keys) != SERINE_EXPERT_STATE_KEYS:
        raise RuntimeError(
            "Expert-scope state isolation failed: expected exactly "
            f"{sorted(expected_changed_keys)} to change; "
            f"missing={unchanged_expected_keys}, unexpected={unexpected_changed_keys}"
        )

    # The independent test remains untouched until epoch selection is complete.
    # In V7 only, evaluate a freshly reloaded canonical parent now so exact
    # non-Ser preservation can be verified without influencing training.
    if serine_only:
        parent_model = load_v28_model(str(model_path), device)
        parent_summary, _parent_per_residue, parent_positions = evaluate(
            parent_model,
            test_records,
            device,
            args.batch_size,
            args.threshold,
            args.deployment_temperature,
            "canonical_parent_after_training_for_non_ser_invariance",
            cyclic_representation_ensemble=True,
        )
        del parent_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    corrected_summary, corrected_per_residue, corrected_positions = evaluate(
        model,
        test_records,
        device,
        args.batch_size,
        args.threshold,
        args.deployment_temperature,
        (
            "serine_only_qc_cyclic_representation_retrained"
            if serine_only
            else (
                "all_expert_heads_qc_cyclic_representation_retrained"
                if args.cyclic_representation_augmentation
                else "all_expert_heads_qc_retrained"
            )
        ),
        cyclic_representation_ensemble=args.cyclic_representation_augmentation,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = (
        "frankenstein_v28_serine_only_qc.pt"
        if serine_only
        else "frankenstein_v28_expert_heads_qc.pt"
    )
    checkpoint_path = out_dir / checkpoint_name
    candidate_checkpoint_path = out_dir / checkpoint_name.replace(
        ".pt", ".candidate.pt"
    )
    selected_protocol = (
        SERINE_ONLY_CYCLIC_PROTOCOL
        if serine_only
        else (
            CYCLIC_REPRESENTATION_PROTOCOL
            if args.cyclic_representation_augmentation
            else ORDER_BALANCED_PROTOCOL
        )
    )
    training_order_policy = (
        "all_cyclic_sequence_coordinate_starts_with_epoch_indexed_"
        "decoder_rotation_mapped_to_physical_labels"
        if args.cyclic_representation_augmentation
        else "epoch_indexed_cyclic_designed_position_rotation"
    )
    deployment_policy = (
        "all_cyclic_starts_and_all_decoder_orders_mapped_to_physical_"
        "residues_probability_mean"
        if args.cyclic_representation_augmentation
        else "complete_natural_sequence_all_cyclic_rotations_probability_mean"
    )
    checkpoint_payload = {
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "expert_head_qc_metadata": {
            "protocol": selected_protocol,
            "parent_checkpoint_sha256": file_sha256(model_path),
            "train_jsonl_sha256": file_sha256(train_path),
            "test_jsonl_sha256": file_sha256(test_path),
            "changed_state_keys": changed_keys,
            "expert_scope": args.expert_scope,
            "active_expert_tokens": [
                NATURAL_AA_ALPHABET[index] for index in active_indices
            ],
            "preserved_state_key_hashes": {
                key: value
                for key, value in before_hashes.items()
                if key not in expected_changed_keys
            },
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_fraction": args.validation_fraction,
            "early_stopping_patience": args.early_stopping_patience,
            "best_epoch": training_selection["best_epoch"],
            "minimum_order_coverage_epochs": MINIMUM_ORDER_COVERAGE_EPOCHS,
            "training_decoding_order_policy": training_order_policy,
            "training_cyclic_representation_policy": (
                "all_physical_cyclic_starts_jointly_rotate_sequence_labels_and_"
                "backbone_coordinates_with_residue_index_reset"
                if args.cyclic_representation_augmentation
                else "serialized_cyclic_start_fixed"
            ),
            "deployment_annotation_policy": deployment_policy,
            "expert_training_context_policy": (
                "peptide_chain_only_no_visible_receptor_chains"
            ),
            "required_deployment_annotation_context_policy": (
                "peptide_chain_only_no_visible_receptor_chains"
            ),
            "train_context_audit": train_context_audit,
            "test_context_audit": test_context_audit,
            "threshold": args.threshold,
            "deployment_temperature": args.deployment_temperature,
            "seed": args.seed,
            "cyclic_representation_augmentation": bool(
                args.cyclic_representation_augmentation
            ),
        },
    }
    temporary_checkpoint = candidate_checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint_payload, temporary_checkpoint)
    os.replace(temporary_checkpoint, candidate_checkpoint_path)

    # Strictly reload the candidate artifact through the production loader.
    # It is promoted to the production filename only after every quality gate
    # below passes.
    reloaded = load_v28_model(str(candidate_checkpoint_path), device)
    reload_hashes = state_hashes(reloaded.state_dict())
    if reload_hashes != after_hashes:
        raise RuntimeError("Saved checkpoint failed strict state round-trip")

    serine = corrected_summary["serine"]
    proline = corrected_summary["proline"]
    overall_fixed = corrected_summary["overall_at_threshold"]
    corrected_auc = corrected_summary["overall_auc"]
    maximum_non_ser_probability_difference = 0.0
    parent_test_summary: Dict[str, Any] | None = None
    if serine_only:
        if parent_summary is None or parent_positions is None:
            raise RuntimeError("Ser-only parent evaluation was not produced")
        if len(parent_positions) != len(corrected_positions):
            raise RuntimeError("Parent/corrected held-out position counts differ")
        differences: List[float] = []
        for parent_row, corrected_row in zip(parent_positions, corrected_positions):
            identity = (
                str(parent_row["sample_name"]),
                int(parent_row["position_in_model_0based"]),
                str(parent_row["base_token"]),
            )
            corrected_identity = (
                str(corrected_row["sample_name"]),
                int(corrected_row["position_in_model_0based"]),
                str(corrected_row["base_token"]),
            )
            if identity != corrected_identity:
                raise RuntimeError("Parent/corrected held-out row order differs")
            if identity[2] != "S":
                differences.append(
                    abs(
                        float(
                            parent_row["probability_methyl_deployment_scaled"]
                        )
                        - float(
                            corrected_row["probability_methyl_deployment_scaled"]
                        )
                    )
                )
        maximum_non_ser_probability_difference = max(differences, default=0.0)
        parent_test_summary = {
            "overall_auc": parent_summary["overall_auc"],
            "overall_at_threshold": parent_summary["overall_at_threshold"],
            "non_ser_auc": parent_summary["non_ser_auc"],
            "non_ser_at_threshold": parent_summary["non_ser_at_threshold"],
            "serine": parent_summary["serine"],
        }
    quality_checks = {
        "requested_expert_scope_changed_exactly_expected_tensors": (
            set(changed_keys) == expected_changed_keys
        ),
        "all_non_target_tensors_are_bitwise_parent_identical": all(
            before_hashes[key] == after_hashes[key]
            for key in before_hashes
            if key not in expected_changed_keys
        ),
        "serine_only_non_ser_probabilities_are_exactly_preserved": (
            not serine_only or maximum_non_ser_probability_difference == 0.0
        ),
        "train_and_test_are_peptide_only_with_zero_visible_receptors": (
            int(train_context_audit["visible_receptor_chains"]) == 0
            and int(test_context_audit["visible_receptor_chains"]) == 0
            and int(train_context_audit["chains_per_record"]) == 1
            and int(test_context_audit["chains_per_record"]) == 1
        ),
        "record_disjoint_validation_split": not (
            set(split_manifest["development_record_names"])
            & set(split_manifest["validation_record_names"])
        ),
        "validation_bce_improved_over_parent": (
            float(training_selection["best_validation_balanced_bce"])
            < float(baseline_validation_loss)
        ),
        "selected_epoch_has_complete_cyclic_order_coverage": (
            int(training_selection["best_epoch"])
            >= MINIMUM_ORDER_COVERAGE_EPOCHS
        ),
        "requested_cyclic_representation_training_protocol_is_active": (
            not args.cyclic_representation_augmentation
            or (
                selected_protocol
                in {CYCLIC_REPRESENTATION_PROTOCOL, SERINE_ONLY_CYCLIC_PROTOCOL}
                and bool(
                    training_selection.get("cyclic_representation_augmentation")
                )
            )
        ),
        "serine_only_scope_metadata_is_exact": (
            not serine_only
            or training_selection.get("active_expert_tokens") == ["S"]
        ),
        "all_19_supported_experts_present_in_test": (
            int(corrected_summary["supported_expert_count"]) == 19
        ),
        "serine_test_has_both_classes": (
            int(serine["natural_negatives"]) == EXPECTED_TEST_COUNTS["S"]
            and int(serine["methyl_positives"]) == EXPECTED_TEST_COUNTS["s"]
        ),
        "serine_auc_ge_0_70": serine["auc"] is not None and float(serine["auc"]) >= 0.70,
        "serine_deployment_t05_recall_at_0_6_ge_0_40": (
            float(serine["recall"]) >= 0.40
        ),
        "serine_deployment_t05_fpr_at_0_6_le_0_25": (
            float(serine["false_positive_rate"]) <= 0.25
        ),
        "proline_deployment_t05_no_p_fpr_le_0_05": (
            int(proline["methyl_positives"]) == 0
            and float(proline["false_positive_rate"]) <= 0.05
        ),
        "overall_auc_ge_0_85": (
            corrected_auc is not None and float(corrected_auc) >= 0.85
        ),
        "supported_macro_auc_ge_0_70": (
            corrected_summary["supported_macro_auc"] is not None
            and float(corrected_summary["supported_macro_auc"]) >= 0.70
        ),
        "overall_deployment_t05_precision_at_0_6_ge_0_75": (
            float(overall_fixed["precision"]) >= 0.75
        ),
        "overall_deployment_t05_recall_at_0_6_ge_0_40": (
            float(overall_fixed["recall"]) >= 0.40
        ),
        "overall_deployment_t05_fpr_at_0_6_le_0_10": (
            float(overall_fixed["false_positive_rate"]) <= 0.10
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    if quality_gate == "PASS":
        os.replace(candidate_checkpoint_path, checkpoint_path)
        checkpoint_artifact_path = checkpoint_path
    else:
        checkpoint_artifact_path = candidate_checkpoint_path
    manifest = {
        "quality_gate": quality_gate,
        "protocol": selected_protocol,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parent_checkpoint": str(model_path),
        "parent_checkpoint_sha256": file_sha256(model_path),
        "checkpoint_ready_for_generation": quality_gate == "PASS",
        "output_checkpoint": str(checkpoint_path) if quality_gate == "PASS" else None,
        "candidate_checkpoint": str(checkpoint_artifact_path),
        "checkpoint_artifact_sha256": file_sha256(checkpoint_artifact_path),
        "expert_scope": args.expert_scope,
        "active_expert_tokens": [
            NATURAL_AA_ALPHABET[index] for index in active_indices
        ],
        "expected_changed_state_keys": sorted(expected_changed_keys),
        "changed_state_keys": changed_keys,
        "unchanged_state_key_count": len(before_hashes) - len(changed_keys),
        "changed_non_expert_keys": changed_non_expert_keys,
        "alphabet": EXTENDED_AA_ALPHABET,
        "alphabet_size": len(EXTENDED_AA_ALPHABET),
        "proline_policy": (
            "no p output token; P expert is calibrated only on natural-P negatives; "
            "generation contains no P-to-p mapping"
        ),
        "parameter_policy": (
            "shared trunk, sequence embedding, decoder, base head, and nineteen "
            "non-Ser experts are bitwise frozen; only the Ser expert is retrained"
            if serine_only
            else "shared trunk, sequence embedding, decoder, and base head are "
            "bitwise frozen; all 20 expert linear heads are retrained"
        ),
        "label_input_policy": (
            "all methyl target tokens are converted to their natural parent before "
            "every forward pass; labels never enter W_s"
        ),
        "expert_training_context": {
            "train": train_context_audit,
            "test": test_context_audit,
        },
        "training_decoding_order_policy": (
            "all physical cyclic starts jointly rotate sequence, labels, and "
            "N/CA/C/O coordinates with residue_idx reset; an epoch-indexed "
            "decoder rotation is applied to every representation"
            if args.cyclic_representation_augmentation
            else "epoch-indexed cyclic designed-position rotation per batch row; "
            "receptor/padding positions are prefixed, every relative depth is "
            "covered within the 30-epoch minimum, and the exact full order is "
            "passed into the causal decoder"
        ),
        "validation_test_annotation_policy": (
            "all equivalent cyclic sequence/coordinate starts and all decoder "
            "orders are evaluated, then mapped back to physical residues"
            if args.cyclic_representation_augmentation
            else "complete natural sequence scored over every cyclic rotation; "
            "each peptide site appears once at every relative decoder depth"
        ),
        "deployment_gate_policy": (
            f"expert probabilities are sigmoid(logit / {args.deployment_temperature}) "
            "for every cyclic order, then averaged, followed by the exact strict "
            f">{args.threshold} generation decision"
        ),
        "training": {
            "maximum_epochs": args.epochs,
            "minimum_order_coverage_epochs": MINIMUM_ORDER_COVERAGE_EPOCHS,
            "epochs_ran": training_selection["epochs_ran"],
            "best_epoch": training_selection["best_epoch"],
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_fraction": args.validation_fraction,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
            "cyclic_representation_augmentation": bool(
                args.cyclic_representation_augmentation
            ),
            "train_jsonl": str(train_path),
            "train_jsonl_sha256": file_sha256(train_path),
            "test_jsonl": str(test_path),
            "test_jsonl_sha256": file_sha256(test_path),
            "split": split_manifest,
            "baseline_validation_balanced_bce": baseline_validation_loss,
            "selection": training_selection,
        },
        "corrected_test": corrected_summary,
        "canonical_parent_test_before_serine_only_repair": parent_test_summary,
        "maximum_non_ser_probability_difference_from_parent": (
            maximum_non_ser_probability_difference
        ),
        "quality_checks": quality_checks,
    }
    atomic_write_json(out_dir / "expert_heads_retrain_manifest.json", manifest)
    atomic_write_csv(
        out_dir / "training_history.csv", history, list(history[0])
    )
    atomic_write_csv(
        out_dir / "test_metrics_by_residue.csv",
        corrected_per_residue,
        list(corrected_per_residue[0]),
    )
    atomic_write_csv(
        out_dir / "test_position_probabilities.csv",
        corrected_positions,
        list(corrected_positions[0]),
    )

    print(
        "===== CANONICAL SER-ONLY RETRAIN COMPLETE ====="
        if serine_only
        else "===== CANONICAL ALL-EXPERT-HEAD RETRAIN COMPLETE =====",
        flush=True,
    )
    print(f"Quality gate: {quality_gate}", flush=True)
    print(
        f"Changed tensors: {len(changed_keys)} / {len(expected_changed_keys)} expected",
        flush=True,
    )
    if serine_only:
        print(
            "Maximum non-Ser probability difference from parent: "
            f"{maximum_non_ser_probability_difference:.12g}",
            flush=True,
        )
    print(
        "Validation BCE: parent={:.6f}, selected={:.6f} at epoch {}".format(
            baseline_validation_loss,
            training_selection["best_validation_balanced_bce"],
            training_selection["best_epoch"],
        ),
        flush=True,
    )
    print(
        "Ser deployment test (T={:.2f}): AUC={:.4f}, recall={:.4f}, "
        "FPR={:.4f}".format(
            args.deployment_temperature,
            float(serine["auc"]),
            float(serine["recall"]),
            float(serine["false_positive_rate"]),
        ),
        flush=True,
    )
    print(
        "Overall deployment test: AUC={:.4f}, macro-AUC={:.4f}, "
        "precision@0.6={:.4f}, "
        "recall@0.6={:.4f}, FPR@0.6={:.4f}".format(
            float(corrected_summary["overall_auc"]),
            float(corrected_summary["supported_macro_auc"]),
            float(overall_fixed["precision"]),
            float(overall_fixed["recall"]),
            float(overall_fixed["false_positive_rate"]),
        ),
        flush=True,
    )
    if quality_gate == "PASS":
        print(f"Production checkpoint: {checkpoint_path}", flush=True)
    else:
        print(
            f"Diagnostic candidate only (not production): {candidate_checkpoint_path}",
            flush=True,
        )

    if quality_gate != "PASS" and not args.no_fail_on_quality_gate:
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError("Quality gate failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
