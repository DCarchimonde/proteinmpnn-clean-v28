#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
clean_v28_common.py

这个文件只服务于 paper_clean_v28 工作区。
原则：
1. 不改旧代码。
2. 统一使用 nmethyl/utils/nmethyl_config.py 的字母表。
3. 填充位点一律不参与评价。
4. 复合物评价时，模型输入可以包含受体链，但指标只统计被选中的短肽链。
"""

import os
import sys
import json
import copy
import csv
import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
from nmethyl.utils.nmethyl_config import (
    EXTENDED_AA_ALPHABET,
    NATURAL_AA_ALPHABET,
    NMETHYL_TO_NATURAL_MAPPING,
    EXTENDED_AA_TO_INDEX,
)

X_TOKEN = "X"
X_INDEX = EXTENDED_AA_TO_INDEX[X_TOKEN]
N_NATURAL = len(NATURAL_AA_ALPHABET)
V11_MODEL_ARCHITECTURE_PROTOCOL = (
    "proteinmpnn_boundary_marginalized_cyclic_relative_positions_v11"
)
V11_CYCLIC_OFFSET_POLICY = (
    "same_designed_chain_directed_offset_marginalized_over_all_linear_cuts"
)

METHYL_ABS_TO_NAT = {
    int(m_rel) + N_NATURAL: int(n_idx)
    for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items()
}

NAT_TO_METHYL_ABS = {}
for m_abs, n_idx in METHYL_ABS_TO_NAT.items():
    # 如果同一个天然氨基酸有多个甲基 token，保留第一个。
    NAT_TO_METHYL_ABS.setdefault(int(n_idx), int(m_abs))


def complete_decoding_order(
    chain_M: torch.Tensor,
    mask: torch.Tensor,
    selected_orders: Any,
) -> torch.Tensor:
    """Build a full decoder permutation from one order per designed chain.

    Receptor and padding positions are always placed before designed peptide
    positions so their known sequence remains visible.  ``selected_orders`` may
    be a rank-2 tensor when every row has the same peptide length, or a sequence
    of rank-1 tensors for variable-length training batches.
    """

    if chain_M.ndim != 2 or mask.ndim != 2 or chain_M.shape != mask.shape:
        raise ValueError("chain_M and mask must be same-shaped rank-2 tensors")
    if torch.is_tensor(selected_orders):
        if selected_orders.ndim != 2 or selected_orders.shape[0] != chain_M.shape[0]:
            raise ValueError("selected_orders tensor has an incompatible shape")
        order_rows = [selected_orders[index] for index in range(chain_M.shape[0])]
    else:
        order_rows = list(selected_orders)
        if len(order_rows) != chain_M.shape[0]:
            raise ValueError("selected_orders row count does not match chain_M")

    selected_mask = (chain_M * mask) > 0.0
    full_orders: List[torch.Tensor] = []
    for row_index, requested in enumerate(order_rows):
        expected_selected = torch.nonzero(
            selected_mask[row_index], as_tuple=False
        ).squeeze(-1)
        requested = requested.to(device=chain_M.device, dtype=torch.long).reshape(-1)
        if requested.numel() != expected_selected.numel() or not torch.equal(
            torch.sort(requested).values,
            expected_selected,
        ):
            raise ValueError(
                f"Selected decoding order is not the designed-position permutation "
                f"for batch row {row_index}"
            )
        prefix = torch.nonzero(
            ~selected_mask[row_index], as_tuple=False
        ).squeeze(-1)
        full = torch.cat([prefix, requested], dim=0)
        expected_full = torch.arange(
            chain_M.shape[1], device=chain_M.device, dtype=torch.long
        )
        if full.numel() != expected_full.numel() or not torch.equal(
            torch.sort(full).values,
            expected_full,
        ):
            raise RuntimeError("Constructed decoding order is not a full permutation")
        full_orders.append(full)
    return torch.stack(full_orders, dim=0)


def random_designed_decoding_order(
    chain_M: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return one explicit random designed-position order per batch row."""

    selected_mask = (chain_M * mask) > 0.0
    requested = []
    for row_index in range(chain_M.shape[0]):
        positions = torch.nonzero(
            selected_mask[row_index], as_tuple=False
        ).squeeze(-1)
        if positions.numel() <= 0:
            raise ValueError(f"Batch row {row_index} has no designed positions")
        requested.append(
            positions[torch.randperm(positions.numel(), device=chain_M.device)]
        )
    return complete_decoding_order(chain_M, mask, requested)


def cyclic_designed_decoding_order(
    chain_M: torch.Tensor,
    mask: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """Return the requested cyclic rotation for every designed chain."""

    if shift < 0:
        raise ValueError("shift must be non-negative")
    selected_mask = (chain_M * mask) > 0.0
    requested = []
    for row_index in range(chain_M.shape[0]):
        positions = torch.nonzero(
            selected_mask[row_index], as_tuple=False
        ).squeeze(-1)
        if positions.numel() <= 0:
            raise ValueError(f"Batch row {row_index} has no designed positions")
        requested.append(torch.roll(positions, shifts=-(shift % len(positions))))
    return complete_decoding_order(chain_M, mask, requested)


def cyclic_known_sequence_methyl_probabilities(
    model: nn.Module,
    X: torch.Tensor,
    S_natural: torch.Tensor,
    mask: torch.Tensor,
    chain_M: torch.Tensor,
    residue_idx: torch.Tensor,
    chain_encoding_all: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score a complete natural sequence with a depth-balanced order ensemble.

    For a peptide of length ``L``, all ``L`` cyclic rotations are evaluated.
    Every site therefore appears exactly once at every relative decoder depth.
    The returned mean is the only probability used for methyl annotation; the
    standard deviation is an order-sensitivity diagnostic.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if S_natural.shape != chain_M.shape or mask.shape != chain_M.shape:
        raise ValueError("S_natural, chain_M, and mask shapes must match")
    selected_mask = (chain_M * mask) > 0.0
    selected_rows = [
        torch.nonzero(selected_mask[index], as_tuple=False).squeeze(-1)
        for index in range(chain_M.shape[0])
    ]
    lengths = [int(value.numel()) for value in selected_rows]
    if not lengths or min(lengths) <= 0:
        raise ValueError("Every batch row must contain at least one designed position")

    safe_base = S_natural.clone()
    invalid_base = (safe_base < 0) | (safe_base >= N_NATURAL)
    if bool((invalid_base & selected_mask).any()):
        raise RuntimeError("Designed sequence contains a noncanonical natural token")
    safe_base[invalid_base] = 0

    probability_sum = torch.zeros_like(S_natural, dtype=torch.float32)
    probability_square_sum = torch.zeros_like(S_natural, dtype=torch.float32)
    probability_count = torch.zeros_like(S_natural, dtype=torch.float32)
    for shift in range(max(lengths)):
        active_rows = torch.tensor(
            [shift < length for length in lengths],
            device=chain_M.device,
            dtype=torch.bool,
        )
        requested = [
            torch.roll(positions, shifts=-shift) if shift < len(positions) else positions
            for positions in selected_rows
        ]
        decoding_order = complete_decoding_order(
            chain_M,
            mask,
            requested,
        )
        _base_logits, expert_logits = model(
            X,
            S_natural,
            mask,
            chain_M,
            residue_idx,
            chain_encoding_all,
            decoding_order=decoding_order,
        )
        selected_logits = torch.gather(
            expert_logits,
            -1,
            safe_base.unsqueeze(-1),
        ).squeeze(-1)
        probabilities = torch.sigmoid(selected_logits / temperature)
        contribution = selected_mask & active_rows.unsqueeze(-1)
        contribution_float = contribution.to(dtype=probabilities.dtype)
        probability_sum += probabilities * contribution_float
        probability_square_sum += probabilities.square() * contribution_float
        probability_count += contribution_float

    expected_count = torch.tensor(
        lengths, device=chain_M.device, dtype=probability_count.dtype
    ).unsqueeze(-1).expand_as(probability_count)
    if not torch.equal(
        probability_count[selected_mask],
        expected_count[selected_mask],
    ):
        raise RuntimeError("Cyclic order ensemble did not balance decoder depth")
    safe_count = probability_count.clamp_min(1.0)
    mean = probability_sum / safe_count
    variance = (probability_square_sum / safe_count - mean.square()).clamp_min(0.0)
    return mean, torch.sqrt(variance)


def cyclic_representation_known_sequence_methyl_probabilities(
    model: nn.Module,
    X: torch.Tensor,
    S_natural: torch.Tensor,
    mask: torch.Tensor,
    chain_M: torch.Tensor,
    residue_idx: torch.Tensor,
    chain_encoding_all: torch.Tensor,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    """Average every physical cyclic start *and* every decoder-depth rotation.

    ``cyclic_known_sequence_methyl_probabilities`` balances only the causal
    decoder order while keeping the serialized peptide start fixed.  A cyclic
    peptide, however, can be serialized from any residue.  ProteinMPNN's
    relative positional features treat the first/last array boundary specially,
    so decoder-order balancing alone cannot rule out an absolute tensor-index
    artefact.

    This function evaluates all equivalent cyclic serializations.  For each
    representation it jointly rolls the natural sequence and N/CA/C/O
    coordinates, resets the peptide residue indices to ``0..L-1``, evaluates
    the complete decoder-order ensemble, and finally maps the probabilities
    back to the original physical residues before averaging.  The returned mean
    is therefore invariant to which residue was chosen as array position 1.

    ``mean``/``ranking_mean`` is a ranking statistic only.  A release caller
    must use ``representation_min``/``release_floor`` with a strict threshold
    and must reject any position whose min/max straddles that threshold.  This
    prevents an averaged score from hiding a hard-call change across equivalent
    cyclic serializations.

    The deployment contract is deliberately narrow: every non-padding position
    must belong to one designed peptide chain and there may be no visible
    receptor positions.  That is exactly the expert-head train/test and final
    annotation context used by the Ser-QC workflow.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if X.ndim != 4 or S_natural.ndim != 2:
        raise ValueError("X must be rank 4 and S_natural must be rank 2")
    if (
        X.shape[:2] != S_natural.shape
        or mask.shape != S_natural.shape
        or chain_M.shape != S_natural.shape
        or residue_idx.shape != S_natural.shape
        or chain_encoding_all.shape != S_natural.shape
    ):
        raise ValueError("cyclic representation tensors have incompatible shapes")

    selected_mask = (chain_M * mask) > 0.0
    if bool(((mask > 0.0) & ~selected_mask).any()):
        raise ValueError(
            "cyclic representation ensemble requires peptide-only input with "
            "no visible receptor positions"
        )
    selected_rows = [
        torch.nonzero(selected_mask[index], as_tuple=False).squeeze(-1)
        for index in range(chain_M.shape[0])
    ]
    lengths = [int(value.numel()) for value in selected_rows]
    if not lengths or min(lengths) <= 0:
        raise ValueError("Every batch row must contain at least one peptide position")

    expanded_X: List[torch.Tensor] = []
    expanded_S: List[torch.Tensor] = []
    expanded_mask: List[torch.Tensor] = []
    expanded_chain_M: List[torch.Tensor] = []
    expanded_residue_idx: List[torch.Tensor] = []
    expanded_chain_encoding: List[torch.Tensor] = []
    representation_map: List[Tuple[int, int]] = []

    for row_index, positions in enumerate(selected_rows):
        length = int(positions.numel())
        chain_values = torch.unique(chain_encoding_all[row_index, positions])
        if int(chain_values.numel()) != 1:
            raise ValueError("Every cyclic representation row must contain one peptide chain")
        canonical_residue_idx = torch.arange(
            length,
            device=residue_idx.device,
            dtype=residue_idx.dtype,
        )
        for shift in range(length):
            row_X = X[row_index].clone()
            row_S = S_natural[row_index].clone()
            row_residue_idx = residue_idx[row_index].clone()
            row_chain_encoding = chain_encoding_all[row_index].clone()
            row_X[positions] = torch.roll(
                X[row_index, positions], shifts=-shift, dims=0
            )
            row_S[positions] = torch.roll(
                S_natural[row_index, positions], shifts=-shift, dims=0
            )
            # A new cyclic serialization starts its linear residue numbering at
            # zero.  Rolling the old residue_idx values would preserve the old
            # artificial boundary and would not test representation bias.
            row_residue_idx[positions] = canonical_residue_idx
            row_chain_encoding[positions] = chain_values[0]
            expanded_X.append(row_X)
            expanded_S.append(row_S)
            expanded_mask.append(mask[row_index].clone())
            expanded_chain_M.append(chain_M[row_index].clone())
            expanded_residue_idx.append(row_residue_idx)
            expanded_chain_encoding.append(row_chain_encoding)
            representation_map.append((row_index, shift))

    expanded_probability, expanded_order_std = (
        cyclic_known_sequence_methyl_probabilities(
            model=model,
            X=torch.stack(expanded_X, dim=0),
            S_natural=torch.stack(expanded_S, dim=0),
            mask=torch.stack(expanded_mask, dim=0),
            chain_M=torch.stack(expanded_chain_M, dim=0),
            residue_idx=torch.stack(expanded_residue_idx, dim=0),
            chain_encoding_all=torch.stack(expanded_chain_encoding, dim=0),
            temperature=temperature,
        )
    )

    probability_sum = torch.zeros_like(S_natural, dtype=torch.float32)
    probability_square_sum = torch.zeros_like(S_natural, dtype=torch.float32)
    probability_min = torch.full_like(
        S_natural, float("inf"), dtype=torch.float32
    )
    probability_max = torch.full_like(
        S_natural, float("-inf"), dtype=torch.float32
    )
    decoder_order_std_sum = torch.zeros_like(S_natural, dtype=torch.float32)
    representation_count = torch.zeros_like(S_natural, dtype=torch.float32)
    mapped_probability_by_start = torch.full(
        (S_natural.shape[0], max(lengths), S_natural.shape[1]),
        float("nan"),
        device=S_natural.device,
        dtype=torch.float32,
    )

    for expanded_index, (row_index, shift) in enumerate(representation_map):
        positions = selected_rows[row_index]
        # The representation was rolled left by ``shift``.  Rolling the output
        # right maps every value back to its original physical residue.
        mapped_probability = torch.roll(
            expanded_probability[expanded_index, positions],
            shifts=shift,
            dims=0,
        )
        mapped_order_std = torch.roll(
            expanded_order_std[expanded_index, positions],
            shifts=shift,
            dims=0,
        )
        probability_sum[row_index, positions] += mapped_probability
        probability_square_sum[row_index, positions] += mapped_probability.square()
        probability_min[row_index, positions] = torch.minimum(
            probability_min[row_index, positions], mapped_probability
        )
        probability_max[row_index, positions] = torch.maximum(
            probability_max[row_index, positions], mapped_probability
        )
        decoder_order_std_sum[row_index, positions] += mapped_order_std
        representation_count[row_index, positions] += 1.0
        mapped_probability_by_start[row_index, shift, positions] = mapped_probability

    expected_count = torch.tensor(
        lengths,
        device=representation_count.device,
        dtype=representation_count.dtype,
    ).unsqueeze(-1).expand_as(representation_count)
    if not torch.equal(
        representation_count[selected_mask], expected_count[selected_mask]
    ):
        raise RuntimeError("Cyclic representation ensemble coverage is incomplete")

    safe_count = representation_count.clamp_min(1.0)
    mean = probability_sum / safe_count
    variance = (
        probability_square_sum / safe_count - mean.square()
    ).clamp_min(0.0)
    representation_std = torch.sqrt(variance)
    representation_span = probability_max - probability_min
    decoder_order_std_mean = decoder_order_std_sum / safe_count

    zero = torch.zeros_like(mean)
    ranking_mean = torch.where(selected_mask, mean, zero)
    release_floor = torch.where(selected_mask, probability_min, zero)
    return {
        "mean": ranking_mean,
        "ranking_mean": ranking_mean,
        "representation_std": torch.where(selected_mask, representation_std, zero),
        "representation_min": release_floor,
        "release_floor": release_floor,
        "representation_max": torch.where(selected_mask, probability_max, zero),
        "representation_span": torch.where(selected_mask, representation_span, zero),
        "decoder_order_std_mean": torch.where(
            selected_mask, decoder_order_std_mean, zero
        ),
        "representation_count": representation_count,
        "representation_probability_by_start": mapped_probability_by_start,
    }


def peptide_only_annotation_tensors(
    X: torch.Tensor,
    S_natural: torch.Tensor,
    mask: torch.Tensor,
    chain_M: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract the designed peptide as the single-chain expert context.

    The corrected expert heads are trained and validated on peptide-only JSONL
    records.  Base-sequence sampling may still use a receptor-conditioned
    complex, but final N-methyl annotation must use the same single-chain input
    domain as expert-head training.  This helper removes every visible receptor
    position and rebuilds the exact single-chain residue/chain indices that
    :func:`featurize_records` creates for a peptide-only record.

    All rows must contain the same number of selected peptide residues.  That is
    the deployment case here: batches are constructed for one target at a time.
    """

    if X.ndim != 4 or S_natural.ndim != 2:
        raise ValueError("X must be rank 4 and S_natural must be rank 2")
    if (
        X.shape[:2] != S_natural.shape
        or mask.shape != S_natural.shape
        or chain_M.shape != S_natural.shape
    ):
        raise ValueError("X, S_natural, mask, and chain_M batch/length shapes must match")

    selected_mask = (chain_M * mask) > 0.0
    selected_rows = [
        torch.nonzero(selected_mask[index], as_tuple=False).squeeze(-1)
        for index in range(S_natural.shape[0])
    ]
    lengths = [int(positions.numel()) for positions in selected_rows]
    if not lengths or min(lengths) <= 0:
        raise ValueError("Every batch row must contain at least one designed peptide residue")
    if len(set(lengths)) != 1:
        raise ValueError("Peptide-only annotation batches must have one peptide length")

    X_peptide = torch.stack(
        [X[index, positions] for index, positions in enumerate(selected_rows)],
        dim=0,
    )
    S_peptide = torch.stack(
        [S_natural[index, positions] for index, positions in enumerate(selected_rows)],
        dim=0,
    )
    peptide_mask = torch.ones_like(S_peptide, dtype=mask.dtype)
    peptide_chain_M = torch.ones_like(S_peptide, dtype=chain_M.dtype)
    residue_idx = torch.arange(
        lengths[0], device=S_natural.device, dtype=torch.long
    ).unsqueeze(0).expand(S_natural.shape[0], -1).clone()
    chain_encoding = torch.zeros_like(residue_idx)
    return (
        X_peptide,
        S_peptide,
        peptide_mask,
        peptide_chain_M,
        residue_idx,
        chain_encoding,
    )


class RobustHierarchicalProteinMPNN(ProteinMPNN):
    """与 v28 / frankenstein_v28.pt 对齐的模型结构。"""

    def __init__(
        self,
        hidden_dim: int = 128,
        augment_eps: float = 0.0,
        cyclic_relative_positions: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_letters=21,
            hidden_dim=hidden_dim,
            vocab=21,
            k_neighbors=48,
            augment_eps=augment_eps,
            **kwargs,
        )
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.W_out_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)),
        )
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))
        ])
        self.cyclic_relative_positions = bool(cyclic_relative_positions)

    def set_cyclic_relative_positions(self, enabled: bool) -> None:
        """Enable the V11 cyclic-native feature path without changing weights."""

        self.cyclic_relative_positions = bool(enabled)

    def forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        decoding_order=None,
    ):
        cyclic_mask = (
            chain_M * mask if self.cyclic_relative_positions else None
        )
        E, E_idx = self.features(
            X,
            mask,
            residue_idx,
            chain_encoding_all,
            cyclic_mask=cyclic_mask,
        )
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_E, E_idx, mask, mask_attend, use_reentrant=False
            )

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        if decoding_order is None:
            decoding_order = torch.argsort(chain_M + 0.0001)
        else:
            decoding_order = decoding_order.to(device=X.device, dtype=torch.long)
            if decoding_order.shape != chain_M.shape:
                raise ValueError(
                    "decoding_order shape must match chain_M: "
                    f"{tuple(decoding_order.shape)} != {tuple(chain_M.shape)}"
                )
            expected = torch.arange(
                chain_M.shape[1], device=X.device, dtype=torch.long
            ).unsqueeze(0).expand_as(decoding_order)
            if not torch.equal(torch.sort(decoding_order, dim=1).values, expected):
                raise ValueError("Every decoding_order row must be a full permutation")

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_ESV, mask, use_reentrant=False
            )

        logits_base = self.W_out_base(h_V)
        logits_experts = torch.cat([expert(h_V) for expert in self.experts], dim=-1)
        return logits_base, logits_experts


def load_v28_model(model_path: str, device: torch.device) -> RobustHierarchicalProteinMPNN:
    """严格加载 frankenstein_v28.pt。不做切片，不做防弹加载。"""
    # Workflow callers hash-pin these local checkpoints, which also carry
    # non-tensor provenance metadata.  Keep full-payload loading explicit and
    # avoid PyTorch's warning about the future default changing.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    architecture_metadata = (
        dict(checkpoint.get("model_architecture_metadata", {}))
        if isinstance(checkpoint, dict)
        else {}
    )
    cyclic_relative_positions = bool(
        architecture_metadata.get("cyclic_relative_positions", False)
    )
    if cyclic_relative_positions and (
        str(architecture_metadata.get("protocol", ""))
        != V11_MODEL_ARCHITECTURE_PROTOCOL
        or str(architecture_metadata.get("cyclic_offset_policy", ""))
        != V11_CYCLIC_OFFSET_POLICY
    ):
        raise RuntimeError(
            "V11 cyclic-relative checkpoint metadata is incomplete or unsupported"
        )
    model = RobustHierarchicalProteinMPNN(
        augment_eps=0.0,
        cyclic_relative_positions=cyclic_relative_positions,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "模型权重没有严格对齐。\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}\n"
            "最终论文流程不能用防弹切片加载。请确认 frankenstein_v28.pt 和模型结构一致。"
        )
    model.eval()
    return model


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_record_name(record: Dict[str, Any], idx: int = 0) -> str:
    return str(record.get("name") or record.get("pdb") or record.get("pdb_id") or record.get("id") or f"sample_{idx}")


def chain_ids_from_record(record: Dict[str, Any]) -> List[str]:
    ordered = []
    for cid in record.get("masked_list", []):
        if f"seq_chain_{cid}" in record and cid not in ordered:
            ordered.append(cid)
    for cid in record.get("visible_list", []):
        if f"seq_chain_{cid}" in record and cid not in ordered:
            ordered.append(cid)
    for key in sorted(record.keys()):
        if key.startswith("seq_chain_"):
            cid = key.replace("seq_chain_", "")
            if cid not in ordered:
                ordered.append(cid)
    return ordered


def choose_eval_chains(
    record: Dict[str, Any],
    eval_chains: str,
    max_peptide_len: int = 30,
    chain_ids: Optional[str] = None,
) -> List[str]:
    all_chains = chain_ids_from_record(record)
    if eval_chains == "masked":
        selected = [c for c in record.get("masked_list", []) if c in all_chains]
        return selected if selected else (["A"] if "A" in all_chains else list(all_chains))
    if eval_chains == "short":
        return [c for c in all_chains if 0 < len(record.get(f"seq_chain_{c}", "")) <= max_peptide_len]
    if eval_chains == "all":
        return list(all_chains)
    if eval_chains == "chain":
        wanted = [x.strip() for x in (chain_ids or "").split(",") if x.strip()]
        return [c for c in wanted if c in all_chains]
    raise ValueError(f"未知 eval_chains: {eval_chains}")


def prepare_record_for_eval(
    record: Dict[str, Any],
    eval_chains: str,
    max_peptide_len: int = 30,
    chain_ids: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    b = copy.deepcopy(record)
    all_chains = chain_ids_from_record(b)
    selected = choose_eval_chains(b, eval_chains, max_peptide_len, chain_ids)
    visible = [c for c in all_chains if c not in set(selected)]

    # 为了复合物短肽评价，短肽设为 masked，其余受体链作为 visible。
    # 单体 masked 模式也这样明确写出，避免旧文件里 masked_list 缺失。
    b["masked_list"] = selected
    b["visible_list"] = visible
    order = selected + visible
    b["seq"] = "".join(b.get(f"seq_chain_{c}", "") for c in order)

    chain_info = []
    for c in order:
        tag = "M" if c in selected else "V"
        chain_info.append(f"{c}:{len(b.get(f'seq_chain_{c}', ''))}{tag}")

    meta = {
        "name": get_record_name(b),
        "selected_chains": ",".join(selected),
        "visible_chains": ",".join(visible),
        "chain_info": ";".join(chain_info),
        "selected_length": sum(len(b.get(f"seq_chain_{c}", "")) for c in selected),
        "total_length": sum(len(b.get(f"seq_chain_{c}", "")) for c in order),
    }
    return b, meta


def naturalize_indices_np(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.int64).copy()
    for m_abs, n_idx in METHYL_ABS_TO_NAT.items():
        out[out == m_abs] = n_idx
    out[(out >= N_NATURAL) & (out != X_INDEX)] = 0
    return out


def naturalize_tensor_for_input(S: torch.Tensor) -> torch.Tensor:
    out = S.clone()
    for m_abs, n_idx in METHYL_ABS_TO_NAT.items():
        out[out == int(m_abs)] = int(n_idx)
    bad = (out >= N_NATURAL) & (out != X_INDEX)
    out[bad] = 0
    return out


def token_to_natural_token(token: str) -> str:
    if token in NATURAL_AA_ALPHABET:
        return token
    if token in EXTENDED_AA_TO_INDEX and token != X_TOKEN:
        idx = EXTENDED_AA_TO_INDEX[token]
        if idx in METHYL_ABS_TO_NAT:
            return NATURAL_AA_ALPHABET[METHYL_ABS_TO_NAT[idx]]
    return X_TOKEN


def naturalize_sequence(seq: str) -> str:
    return "".join(token_to_natural_token(ch) for ch in seq)


def methyl_binary_from_indices(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int64)
    return (arr >= N_NATURAL).astype(np.int64)


def idx_to_sequence(indices: List[int], natural_only: bool = False) -> str:
    alphabet = NATURAL_AA_ALPHABET if natural_only else EXTENDED_AA_ALPHABET
    chars = []
    for x in indices:
        x = int(x)
        chars.append(alphabet[x] if 0 <= x < len(alphabet) else "?")
    return "".join(chars)


def featurize_records(
    records: List[Dict[str, Any]],
    device: torch.device,
    eval_chains: str,
    max_peptide_len: int = 30,
    chain_ids: Optional[str] = None,
):
    """
    干净特征构建。

    返回：
    X, S, mask, chain_M, residue_idx, chain_encoding_all, real_pos, meta

    重要定义：
    - mask：所有真实残基为 1，包括受体链和短肽链。只给模型看真实位置。
    - chain_M：被评价或被设计的链为 1。
    - real_pos：所有真实残基为 1。主要用于审计。
    - 指标统计位点：mask & chain_M & S != X。
    """
    prepared = []
    metas = []
    for r in records:
        b, m = prepare_record_for_eval(r, eval_chains, max_peptide_len, chain_ids)
        if m["total_length"] > 0 and m["selected_length"] > 0:
            prepared.append(b)
            metas.append(m)

    if not prepared:
        return None

    B = len(prepared)
    L_max = max(len(b["seq"]) for b in prepared)

    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.full([B, L_max], X_INDEX, dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    real_pos = np.zeros([B, L_max], dtype=np.float32)

    for i, b in enumerate(prepared):
        chain_order = b.get("masked_list", []) + b.get("visible_list", [])
        l_p = 0
        for c_i, c_id in enumerate(chain_order):
            seq = b.get(f"seq_chain_{c_id}", "")
            if not seq:
                continue

            coords_by_atom = []
            min_len = len(seq)
            for atom_name in ["N", "CA", "C", "O"]:
                arr = np.asarray(b.get(f"{atom_name}_chain_{c_id}", []), dtype=np.float32)
                coords_by_atom.append(arr)
                min_len = min(min_len, len(arr))

            if min_len <= 0:
                continue

            N, CA, C, O = [a[:min_len] for a in coords_by_atom]

            # 有些数据 CA 和 O 可能交换，保留旧代码里的修正。
            if len(N) > 0 and len(CA) > 0 and len(O) > 0:
                dist_n_ca = np.linalg.norm(N[:1] - CA[:1])
                dist_n_o = np.linalg.norm(N[:1] - O[:1])
                if dist_n_o < dist_n_ca and dist_n_o < 1.6:
                    CA, O = O, CA

            X[i, l_p:l_p + min_len, 0, :] = N
            X[i, l_p:l_p + min_len, 1, :] = CA
            X[i, l_p:l_p + min_len, 2, :] = C[:min_len]
            X[i, l_p:l_p + min_len, 3, :] = O

            seq_part = seq[:min_len]
            S[i, l_p:l_p + min_len] = [EXTENDED_AA_TO_INDEX.get(aa, X_INDEX) for aa in seq_part]
            mask[i, l_p:l_p + min_len] = 1.0
            real_pos[i, l_p:l_p + min_len] = 1.0
            if c_id in b.get("masked_list", []):
                chain_M[i, l_p:l_p + min_len] = 1.0
            residue_idx[i, l_p:l_p + min_len] = np.arange(min_len) + c_i * 100
            chain_encoding_all[i, l_p:l_p + min_len] = c_i
            l_p += min_len

    tensors = [
        torch.from_numpy(X).to(device=device, dtype=torch.float32),
        torch.from_numpy(S).to(device=device, dtype=torch.long),
        torch.from_numpy(mask).to(device=device, dtype=torch.float32),
        torch.from_numpy(chain_M).to(device=device, dtype=torch.float32),
        torch.from_numpy(residue_idx).to(device=device, dtype=torch.long),
        torch.from_numpy(chain_encoding_all).to(device=device, dtype=torch.long),
        torch.from_numpy(real_pos).to(device=device, dtype=torch.float32),
    ]
    return tensors, metas


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, thresholds: List[float]) -> List[Dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float32)
    rows = []
    for thr in thresholds:
        # Release decisions throughout V9/V10 use a strict threshold after an
        # explicit eight-decimal normalization.  Evaluation must use the same
        # contract so values such as 0.600000004 cannot change class merely
        # because one CSV retained more floating-point digits.
        pred = (np.round(prob.astype(np.float64), 8) > float(thr)).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({
            "threshold": thr,
            "threshold_operator": ">",
            "probability_rounding_policy": "round(prob,8)",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_positive_rate": fpr,
            "pred_methyl_rate": float(pred.mean()) if len(pred) else 0.0,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })
    return rows


def average_precision_score_simple(
    y_true: np.ndarray, scores: np.ndarray
) -> Optional[float]:
    """Tie-aware non-interpolated average precision without sklearn."""

    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        true_positives += int(np.sum(group == 1))
        false_positives += int(np.sum(group == 0))
        recall = true_positives / n_pos
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(average_precision)


def roc_auc_score_simple(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """不用 sklearn 的二分类曲线下面积。没有正负两类时返回 None。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    # 处理并列分数：同分取平均 rank。
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def parse_fasta(path: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


def sequence_recovery(true_seq: str, pred_seq: str, naturalize: bool = True) -> Optional[float]:
    if naturalize:
        true_seq = naturalize_sequence(true_seq)
        pred_seq = naturalize_sequence(pred_seq)
    if len(true_seq) != len(pred_seq) or len(true_seq) == 0:
        return None
    return sum(a == b for a, b in zip(true_seq, pred_seq)) / len(true_seq)


def methyl_count(seq: str) -> int:
    return sum(1 for ch in seq if ch in EXTENDED_AA_TO_INDEX and EXTENDED_AA_TO_INDEX[ch] >= N_NATURAL and ch != X_TOKEN)
