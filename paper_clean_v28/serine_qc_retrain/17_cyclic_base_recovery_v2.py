#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cyclic-start ProteinMPNN re-audit and conditional joint recovery for V8.

This revision is intentionally additive.  It never rewrites the immutable
31,500-row V6-derived baseline or the completed V8 round-6 search evidence.
The legacy strict hits are first re-audited with a receptor-conditioned base
score that jointly rotates peptide coordinates and sequence, resets peptide
residue indices, and averages every physical cyclic start and every decoder
order.  Only if that corrected re-audit still releases no candidate does the
program run a fixed six-round methyl/base dual-objective search.

Release gates remain unchanged: rounded methyl probability strictly greater
than 0.6, the exact cyclic-start ProteinMPNN 1st-percentile floor, independent
batch-one re-scoring, and exact/forward-cyclic novelty.  Exhausting the fixed
budget without a real candidate is an explicit failure, never an abstention or
fabricated success.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
LEGACY_SEARCH_PATH = SCRIPT_PATH.with_name("14_directed_recovery_search_v8.py")
FRONTIER_V3_PATH = SCRIPT_PATH.with_name("20_full_frontier_recovery_v3.py")
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"
DEFAULT_MODEL = V8_ROOT / "model" / "frankenstein_v28_source_scoped_hybrid_v8.pt"
DEFAULT_MODEL_MANIFEST = V8_ROOT / "model" / "expert_source_composition_manifest.json"
DEFAULT_REPRESENTATION = V8_ROOT / "representation_audit" / "cyclic_representation_audit.json"
DEFAULT_BASELINE = V8_ROOT / "generation_baseline"
DEFAULT_LEGACY_SEARCH = V8_ROOT / "directed_search"
DEFAULT_OUT = V8_ROOT / "directed_search_cyclic_base_v2"
DEFAULT_PRIOR_V2 = DEFAULT_OUT
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

V2_SEARCH_PROTOCOL = "cyclic_start_base_pareto_recovery_v8_v2"
V2_INFLIGHT_PROTOCOL = "hash_pinned_v2_round_inflight_resume_v1"
V2_BASE_POLICY = (
    "native_complex_receptor_visible_joint_peptide_coordinate_sequence_roll_"
    "residue_index_reset_all_physical_starts_all_decoder_orders_mean"
)
V2_SEED = 20260818
THRESHOLD = 0.6
TEMPERATURE = 0.5
BASE_PERCENTILE = 0.01
RESCORE_TOLERANCE = 2e-6
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
METHYLATABLE_AA = set(NATURAL_AA) - {"P"}
EXPECTED_LEGACY_ROUNDS = 6
EXPECTED_BEAM_WIDTH = 512
EXPECTED_OFFSPRING = 4096
EXPECTED_SHORTLIST = 4096
EXPECTED_RELEASE_LIMIT = 200
EXPECTED_V3_LEGACY_BRIDGE = 16_384


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def artifact(path: Path) -> Dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    os.replace(temporary, destination)


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


def atomic_write_gzip_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with __import__("io").TextIOWrapper(
                compressed, encoding="utf-8", newline=""
            ) as text:
                writer = csv.DictWriter(
                    text, fieldnames=list(fields), extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def physical_argmax_summary(
    sequence: str, probabilities: Sequence[float]
) -> Dict[str, Any]:
    """Return an explicit physical-position summary for an annotation vector."""

    natural = str(sequence).upper()
    values = [float(value) for value in probabilities]
    if (
        not natural
        or len(natural) != len(values)
        or not set(natural) <= set(NATURAL_AA)
        or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
    ):
        raise ValueError("Invalid physical methyl probability vector")
    eligible = [index for index, token in enumerate(natural) if token in METHYLATABLE_AA]
    if eligible:
        best = max(eligible, key=lambda index: (values[index], -index))
        maximum = values[best]
    else:
        best = max(range(len(natural)), key=lambda index: (values[index], -index))
        maximum = 0.0
    total = sum(values)
    return {
        "physical_argmax_position_1based": best + 1,
        "physical_argmax_residue": natural[best],
        "physical_argmax_probability": maximum,
        "physical_probability_vector": json.dumps(values),
        "physical_probability_mass_fraction": json.dumps(
            [round(value / total, 8) if total > 0.0 else 0.0 for value in values]
        ),
        "strict_physical_positions_1based": json.dumps(
            [
                index + 1
                for index in eligible
                if round(values[index], 8) > THRESHOLD
            ]
        ),
    }


class CyclicBasePlausibilityScorer:
    """Receptor-conditioned base score over all cyclic starts and all orders.

    For physical start ``r`` the peptide N/CA/C/O coordinates and natural
    sequence are jointly rolled left by ``r`` while receptor tensors remain
    fixed.  Peptide residue indices are reset to ``0..L-1``.  The mean log
    probability is then averaged over every complete decoder-order rotation.
    The final score is the mean over all physical starts.
    """

    def __init__(
        self,
        model: Any,
        device: Any,
        target_records: Mapping[str, Mapping[str, Any]],
        batch_size: int,
        torch_module: Any,
        functional: Any,
        common: Mapping[str, Any],
        progress_class: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.target_records = target_records
        self.batch_size = int(batch_size)
        self.torch = torch_module
        self.functional = functional
        self.common = common
        self.progress_class = progress_class
        self.features: Dict[str, Tuple[Any, ...]] = {}

    def _features(self, target: str) -> Tuple[Any, ...]:
        if target not in self.features:
            packed = self.common["featurize_records"](
                [self.target_records[target]],
                device=self.device,
                eval_chains="masked",
                max_peptide_len=30,
            )
            if packed is None:
                raise RuntimeError(f"Complex feature construction failed for {target}")
            tensors, _metas = packed
            self.features[target] = tuple(tensors[:6])
        return self.features[target]

    def score_detailed(
        self,
        target: str,
        sequences: Sequence[str],
        stage: str = "cyclic-start base plausibility",
    ) -> Dict[str, Dict[str, Any]]:
        unique = sorted(set(str(value).upper() for value in sequences))
        result: Dict[str, Dict[str, Any]] = {}
        progress = self.progress_class(f"{target} {stage}", len(unique), unit="seq")
        alphabet = self.common["NATURAL_AA_ALPHABET"]
        X, S_true, mask, chain_M, residue_idx, chain_encoding = self._features(target)
        selected = self.torch.nonzero(
            (chain_M[0] * mask[0]) > 0.0, as_tuple=False
        ).squeeze(-1)
        length = int(selected.numel())
        if length <= 0:
            raise RuntimeError(f"No designed peptide positions for {target}")
        canonical_residue_idx = self.torch.arange(
            length, device=self.device, dtype=residue_idx.dtype
        )
        peptide_X = X[0, selected]
        for sequence_batch in chunks(unique, self.batch_size):
            if any(len(sequence) != length for sequence in sequence_batch):
                raise RuntimeError(f"Cyclic base plausibility length mismatch for {target}")
            current = len(sequence_batch)
            natural = self.torch.tensor(
                [[alphabet.index(token) for token in sequence] for sequence in sequence_batch],
                device=self.device,
                dtype=self.torch.long,
            )
            representation_scores: List[Any] = []
            with self.torch.no_grad():
                for representation_shift in range(length):
                    rolled_natural = self.torch.roll(
                        natural, shifts=-representation_shift, dims=1
                    )
                    Xb = X.repeat(current, 1, 1, 1).clone()
                    Sb = S_true.repeat(current, 1).clone()
                    maskb = mask.repeat(current, 1)
                    chainb = chain_M.repeat(current, 1)
                    residueb = residue_idx.repeat(current, 1).clone()
                    encodingb = chain_encoding.repeat(current, 1)
                    Xb[:, selected] = self.torch.roll(
                        peptide_X, shifts=-representation_shift, dims=0
                    ).unsqueeze(0)
                    Sb[:, selected] = rolled_natural
                    residueb[:, selected] = canonical_residue_idx.unsqueeze(0)
                    order_total = self.torch.zeros(current, device=self.device)
                    for order_shift in range(length):
                        requested = selected.roll(shifts=-order_shift).unsqueeze(0).repeat(
                            current, 1
                        )
                        order = self.common["complete_decoding_order"](
                            chainb, maskb, requested
                        )
                        base_logits, _experts = self.model(
                            Xb,
                            Sb,
                            maskb,
                            chainb,
                            residueb,
                            encodingb,
                            decoding_order=order,
                        )
                        log_probability = self.functional.log_softmax(base_logits, dim=-1)
                        selected_log_probability = log_probability[:, selected].gather(
                            -1, rolled_natural.unsqueeze(-1)
                        ).squeeze(-1)
                        order_total += selected_log_probability.mean(dim=1)
                    representation_scores.append(order_total / length)
            matrix = self.torch.stack(representation_scores, dim=1)
            means = matrix.mean(dim=1)
            minimums = matrix.min(dim=1).values
            maximums = matrix.max(dim=1).values
            standard_deviations = matrix.std(dim=1, unbiased=False)
            vectors = matrix.detach().cpu().tolist()
            for row_index, sequence in enumerate(sequence_batch):
                values = [float(value) for value in vectors[row_index]]
                summary = {
                    "cyclic_base_log_probability_mean": float(means[row_index].item()),
                    "cyclic_base_log_probability_min": float(minimums[row_index].item()),
                    "cyclic_base_log_probability_max": float(maximums[row_index].item()),
                    "cyclic_base_log_probability_span": float(
                        maximums[row_index].item() - minimums[row_index].item()
                    ),
                    "cyclic_base_log_probability_std": float(
                        standard_deviations[row_index].item()
                    ),
                    "cyclic_base_physical_start_scores": json.dumps(values),
                    "cyclic_base_physical_start_count": length,
                    "cyclic_base_decoder_order_count_per_start": length,
                    "cyclic_base_total_ensemble_size": length * length,
                    "cyclic_base_context_policy": V2_BASE_POLICY,
                }
                numeric_values = [
                    float(summary[key])
                    for key in (
                        "cyclic_base_log_probability_mean",
                        "cyclic_base_log_probability_min",
                        "cyclic_base_log_probability_max",
                        "cyclic_base_log_probability_span",
                        "cyclic_base_log_probability_std",
                    )
                ]
                if not all(math.isfinite(value) for value in [*values, *numeric_values]):
                    raise RuntimeError(f"Non-finite cyclic base score for {target}:{sequence}")
                if not (
                    len(values) == length
                    and abs(sum(values) / length - numeric_values[0]) <= 2e-6
                    and abs(min(values) - numeric_values[1]) <= 2e-6
                    and abs(max(values) - numeric_values[2]) <= 2e-6
                    and numeric_values[3] >= 0.0
                    and numeric_values[4] >= 0.0
                ):
                    raise RuntimeError(f"Inconsistent cyclic base score for {target}:{sequence}")
                result[sequence] = summary
            progress.update(len(sequence_batch))
        progress.close()
        return result

    def score(
        self, target: str, sequences: Sequence[str], stage: str = "cyclic base"
    ) -> Dict[str, float]:
        return {
            sequence: float(row["cyclic_base_log_probability_mean"])
            for sequence, row in self.score_detailed(target, sequences, stage).items()
        }


def pareto_front(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Exact non-dominated front for methyl maximum and cyclic base mean."""

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row["maximum_probability"]),
            -float(row["cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    front: List[Dict[str, Any]] = []
    best_base = float("-inf")
    for row in ordered:
        base = float(row["cyclic_base_log_probability_mean"])
        if base > best_base:
            front.append(row)
            best_base = base
    return front


def deterministic_diversity_fill(
    ranked: Sequence[Mapping[str, Any]],
    selected: List[Dict[str, Any]],
    width: int,
) -> List[Dict[str, Any]]:
    """Fill a beam by max-min Hamming distance with deterministic ties.

    The original V2 implementation recomputed every candidate-to-selected
    Hamming distance after every insertion.  At the production round-one
    shape (roughly 30,000 candidates and 3,700 seeds) that turns a small
    seven-residue max-min problem into hundreds of billions of Python-level
    comparisons.  Cache each candidate's current nearest distance instead:
    initialize the cache in bounded NumPy blocks, then update it only against
    the newly selected sequence.  This is exactly the same greedy objective
    and tie order; only the implementation cost changes.
    """

    selected_by_sequence = {str(row["sequence"]): row for row in selected}
    candidates = [
        dict(row)
        for row in ranked
        if str(row["sequence"]) not in selected_by_sequence
    ]
    rank = {str(row["sequence"]): index for index, row in enumerate(ranked)}
    if not candidates or len(selected_by_sequence) >= width:
        return list(selected_by_sequence.values())[:width]

    started = time.monotonic()
    additions_required = max(0, int(width) - len(selected_by_sequence))
    if len(candidates) >= 1000 and additions_required:
        print(
            "V2 deterministic diversity fill: "
            f"candidates={len(candidates):,}; "
            f"selected={len(selected_by_sequence):,}; "
            f"additions<={additions_required:,}",
            flush=True,
        )
    candidate_sequences = [str(row["sequence"]) for row in candidates]
    prior_sequences = list(selected_by_sequence)
    uniform_length = (
        len({len(sequence) for sequence in [*candidate_sequences, *prior_sequences]})
        == 1
    )
    numpy_module = None
    candidate_tokens = None
    if prior_sequences and uniform_length:
        try:
            numpy_module = __import__("numpy")
            length = len(candidate_sequences[0])
            candidate_tokens = numpy_module.frombuffer(
                "".join(candidate_sequences).encode("ascii"),
                dtype=numpy_module.uint8,
            ).reshape(len(candidate_sequences), length)
            prior_tokens = numpy_module.frombuffer(
                "".join(prior_sequences).encode("ascii"),
                dtype=numpy_module.uint8,
            ).reshape(len(prior_sequences), length)
            minimum_distances = numpy_module.full(
                len(candidate_sequences), length + 1, dtype=numpy_module.int16
            )
            candidate_block_size = 4096
            prior_block_size = 256
            for candidate_start in range(
                0, len(candidate_sequences), candidate_block_size
            ):
                candidate_stop = min(
                    candidate_start + candidate_block_size,
                    len(candidate_sequences),
                )
                block = candidate_tokens[candidate_start:candidate_stop]
                block_minimum = numpy_module.full(
                    len(block), length + 1, dtype=numpy_module.int16
                )
                for prior_start in range(0, len(prior_sequences), prior_block_size):
                    prior_stop = min(
                        prior_start + prior_block_size, len(prior_sequences)
                    )
                    distances = numpy_module.count_nonzero(
                        block[:, None, :]
                        != prior_tokens[None, prior_start:prior_stop, :],
                        axis=2,
                    )
                    block_minimum = numpy_module.minimum(
                        block_minimum, distances.min(axis=1)
                    )
                minimum_distances[candidate_start:candidate_stop] = block_minimum
        except (ImportError, UnicodeEncodeError, ValueError):
            numpy_module = None
            candidate_tokens = None

    if prior_sequences and candidate_tokens is None:
        minimum_distances = [
            min(
                sum(left != right for left, right in zip(sequence, prior))
                for prior in prior_sequences
            )
            for sequence in candidate_sequences
        ]
    elif not prior_sequences:
        minimum_distances = None

    active = [True] * len(candidates)
    while any(active) and len(selected_by_sequence) < width:
        active_indices = [index for index, enabled in enumerate(active) if enabled]
        if not selected_by_sequence:
            chosen_index = active_indices[0]
        else:
            best_distance = max(
                int(minimum_distances[index]) for index in active_indices
            )
            distance_ties = [
                index
                for index in active_indices
                if int(minimum_distances[index]) == best_distance
            ]
            best_rank = min(rank[candidate_sequences[index]] for index in distance_ties)
            rank_ties = [
                index
                for index in distance_ties
                if rank[candidate_sequences[index]] == best_rank
            ]
            chosen_index = max(
                rank_ties,
                key=lambda index: (candidate_sequences[index], -index),
            )
        chosen = candidates[chosen_index]
        chosen_sequence = candidate_sequences[chosen_index]
        selected_by_sequence[chosen_sequence] = chosen
        active[chosen_index] = False
        if not any(active) or len(selected_by_sequence) >= width:
            continue
        if candidate_tokens is not None and numpy_module is not None:
            distances = numpy_module.count_nonzero(
                candidate_tokens != candidate_tokens[chosen_index], axis=1
            )
            if minimum_distances is None:
                minimum_distances = distances
            else:
                minimum_distances = numpy_module.minimum(
                    minimum_distances, distances
                )
        else:
            if minimum_distances is None:
                minimum_distances = [
                    sum(
                        left != right
                        for left, right in zip(sequence, chosen_sequence)
                    )
                    for sequence in candidate_sequences
                ]
            else:
                for index, sequence in enumerate(candidate_sequences):
                    if active[index]:
                        minimum_distances[index] = min(
                            int(minimum_distances[index]),
                            sum(
                                left != right
                                for left, right in zip(sequence, chosen_sequence)
                            ),
                        )
    if len(candidates) >= 1000 and additions_required:
        print(
            "V2 deterministic diversity fill: complete; "
            f"selected={len(selected_by_sequence):,}; "
            f"elapsed={time.monotonic() - started:.2f}s",
            flush=True,
        )
    return list(selected_by_sequence.values())[:width]


def select_dual_objective_beam(
    rows: Sequence[Mapping[str, Any]], width: int, length: int, floor: float
) -> List[Dict[str, Any]]:
    """Fixed-quota beam retaining both hard-gate neighborhoods and diversity."""

    unique = {str(row["sequence"]): dict(row) for row in rows}
    values = list(unique.values())
    methyl_order = sorted(
        values,
        key=lambda row: (
            -float(row["maximum_probability"]),
            -float(row["cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    base_order = sorted(
        values,
        key=lambda row: (
            -float(row["cyclic_base_log_probability_mean"]),
            -float(row["maximum_probability"]),
            str(row["sequence"]),
        ),
    )
    selected: Dict[str, Dict[str, Any]] = {}

    def add(rows_to_add: Sequence[Mapping[str, Any]], limit: int) -> None:
        for row in rows_to_add[: max(0, int(limit))]:
            selected.setdefault(str(row["sequence"]), dict(row))

    strict = [row for row in methyl_order if int(row["passes_strict_probability"])]
    base_pass = [
        row
        for row in methyl_order
        if float(row["cyclic_base_log_probability_mean"]) >= float(floor)
    ]
    joint = [
        row
        for row in strict
        if float(row["cyclic_base_log_probability_mean"]) >= float(floor)
    ]
    add(joint, max(1, width // 4))
    add(base_pass, max(1, width // 4))
    add(sorted(strict, key=lambda row: (
        -float(row["cyclic_base_log_probability_mean"]),
        -float(row["maximum_probability"]),
        str(row["sequence"]),
    )), max(1, width // 5))
    add(base_order, max(1, width // 5))
    add(pareto_front(values), width)
    per_position = max(1, width // max(1, 8 * length))
    for position in range(1, length + 1):
        add(
            [
                row
                for row in methyl_order
                if int(row["argmax_position_1based"]) == position
            ],
            per_position,
        )
    ranked_pool = methyl_order[: max(width * 12, width)] + base_order[: max(width * 12, width)]
    deduplicated_pool = list(
        {str(row["sequence"]): row for row in ranked_pool}.values()
    )
    return deterministic_diversity_fill(
        deduplicated_pool, list(selected.values())[:width], width
    )


def select_methyl_screen_shortlist(
    rows: Sequence[Mapping[str, Any]], limit: int, length: int
) -> List[str]:
    """Deterministic shortlist using methyl score and parent-base bridge signal."""

    unique = {str(row["sequence"]): dict(row) for row in rows}
    values = list(unique.values())
    methyl = sorted(
        values,
        key=lambda row: (
            -int(row["passes_strict_probability"]),
            -float(row["maximum_probability"]),
            -float(row.get("parent_cyclic_base_log_probability_mean", -1e30)),
            str(row["sequence"]),
        ),
    )
    parent_bridge = sorted(
        values,
        key=lambda row: (
            -float(row.get("parent_cyclic_base_log_probability_mean", -1e30)),
            -float(row["maximum_probability"]),
            str(row["sequence"]),
        ),
    )
    selected: Dict[str, Mapping[str, Any]] = {}

    def add(group: Sequence[Mapping[str, Any]], count: int) -> None:
        for row in group[:count]:
            selected.setdefault(str(row["sequence"]), row)

    add(methyl, max(1, limit // 2))
    add(parent_bridge, max(1, limit // 3))
    per_position = max(1, limit // max(1, 12 * length))
    for position in range(1, length + 1):
        add(
            [row for row in methyl if int(row["argmax_position_1based"]) == position],
            per_position,
        )
    if len(selected) < limit:
        ranked = [row for row in methyl if str(row["sequence"]) not in selected]
        seed_rows = [dict(row) for row in selected.values()]
        filled = deterministic_diversity_fill(
            [*seed_rows, *ranked[: max(limit * 8, limit)]], seed_rows, limit
        )
        selected = {str(row["sequence"]): row for row in filled}
    return list(selected)[:limit]


def v2_round_provenance(
    beam: Sequence[Mapping[str, Any]],
    round_index: int,
    offspring_per_round: int,
    numpy_module: Any,
) -> Dict[str, Dict[str, Any]]:
    generated: Dict[str, Dict[str, Any]] = {}
    for parent_row in beam:
        parent = str(parent_row["sequence"])
        for position in range(len(parent)):
            for token in NATURAL_AA:
                if token == parent[position]:
                    continue
                sequence = parent[:position] + token + parent[position + 1 :]
                generated.setdefault(
                    sequence,
                    {
                        "generation_kind": "complete_single_mutant",
                        "parent_sequence": parent,
                        "edit_distance": 1,
                        "mutation_positions_1based": json.dumps([position + 1]),
                        "rng_seed": "",
                        "rng_draw_index": "",
                    },
                )
    rng = numpy_module.random.Generator(
        numpy_module.random.PCG64(
            numpy_module.random.SeedSequence([V2_SEED, int(round_index)])
        )
    )
    parents = [str(row["sequence"]) for row in beam]
    if not parents:
        raise RuntimeError("V2 beam is empty")
    length = len(parents[0])
    for draw_index in range(int(offspring_per_round)):
        parent = parents[int(rng.integers(0, len(parents)))]
        mutation_count = int(rng.integers(2, min(5, length + 1)))
        positions = sorted(
            int(value)
            for value in rng.choice(length, size=mutation_count, replace=False)
        )
        child = list(parent)
        for position in positions:
            alternatives = [token for token in NATURAL_AA if token != child[position]]
            child[position] = alternatives[int(rng.integers(0, len(alternatives)))]
        sequence = "".join(child)
        generated.setdefault(
            sequence,
            {
                "generation_kind": "fixed_seed_multi_mutant",
                "parent_sequence": parent,
                "edit_distance": mutation_count,
                "mutation_positions_1based": json.dumps(
                    [position + 1 for position in positions]
                ),
                "rng_seed": f"{V2_SEED}:{round_index}",
                "rng_draw_index": draw_index,
            },
        )
    return generated


def v2_round_context(
    config_digest: str,
    round_index: int,
    beam: Sequence[Mapping[str, Any]],
    seen: Sequence[str],
    to_score: Sequence[str],
    provenance: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Hash every deterministic input that can affect one V2 search round."""

    return {
        "protocol": V2_INFLIGHT_PROTOCOL,
        "config_sha256": str(config_digest),
        "round_index": int(round_index),
        "beam_sha256": stable_json_sha256({"rows": list(beam)}),
        "seen_sequence_sha256": hashlib.sha256(
            ("\n".join(sorted(set(seen))) + "\n").encode("ascii")
        ).hexdigest(),
        "to_score_sha256": hashlib.sha256(
            ("\n".join(to_score) + "\n").encode("ascii")
        ).hexdigest(),
        "to_score_count": len(to_score),
        "provenance_sha256": stable_json_sha256(
            {"rows": {sequence: provenance[sequence] for sequence in to_score}}
        ),
    }


def validate_v2_methyl_screen_rows(
    old: Any,
    rows: Sequence[Mapping[str, Any]],
    target: str,
    stage: str,
    to_score: Sequence[str],
    provenance: Mapping[str, Mapping[str, Any]],
    parent_base: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Validate and type-restore a hash-pinned in-flight methyl screen."""

    expected_sequences = list(to_score)
    observed_sequences = [
        str(row.get("sequence", "")).upper() for row in rows
    ]
    if observed_sequences != expected_sequences:
        raise RuntimeError("V2 in-flight methyl screen sequence/order mismatch")
    normalized: List[Dict[str, Any]] = []
    for raw, sequence in zip(rows, expected_sequences):
        row = old.normalize_search_ledger_row(raw)
        old.validate_search_ledger_row(
            row,
            target,
            sequence,
            stage,
            provenance[sequence],
        )
        parent = str(provenance[sequence]["parent_sequence"])
        try:
            observed_parent_base = float(
                row["parent_cyclic_base_log_probability_mean"]
            )
            expected_parent_base = float(parent_base[parent])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Malformed V2 parent-base bridge score: {sequence}"
            ) from exc
        if not (
            math.isfinite(observed_parent_base)
            and observed_parent_base == expected_parent_base
        ):
            raise RuntimeError(
                f"V2 parent-base bridge score mismatch: {sequence}"
            )
        row["parent_cyclic_base_log_probability_mean"] = observed_parent_base
        normalized.append(row)
    return normalized


def validate_v2_cyclic_base_rows(
    old: Any,
    rows: Sequence[Mapping[str, Any]],
    target: str,
    stage: str,
    shortlist_sequences: Sequence[str],
    provenance: Mapping[str, Mapping[str, Any]],
    parent_base: Mapping[str, float],
    length: int,
) -> List[Dict[str, Any]]:
    """Validate and type-restore a hash-pinned in-flight base shortlist."""

    normalized = validate_v2_methyl_screen_rows(
        old,
        rows,
        target,
        stage,
        shortlist_sequences,
        provenance,
        parent_base,
    )
    for row in normalized:
        sequence = str(row["sequence"])
        try:
            values = [
                float(value)
                for value in json.loads(
                    str(row["cyclic_base_physical_start_scores"])
                )
            ]
            mean = float(row["cyclic_base_log_probability_mean"])
            minimum = float(row["cyclic_base_log_probability_min"])
            maximum = float(row["cyclic_base_log_probability_max"])
            span = float(row["cyclic_base_log_probability_span"])
            standard_deviation = float(row["cyclic_base_log_probability_std"])
            start_count = int(row["cyclic_base_physical_start_count"])
            order_count = int(
                row["cyclic_base_decoder_order_count_per_start"]
            )
            ensemble_size = int(row["cyclic_base_total_ensemble_size"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Malformed V2 cyclic-base in-flight row: {sequence}"
            ) from exc
        recomputed_mean = sum(values) / length if values else float("nan")
        recomputed_std = (
            math.sqrt(
                sum((value - recomputed_mean) ** 2 for value in values) / length
            )
            if values
            else float("nan")
        )
        if not (
            len(values) == length
            and all(
                math.isfinite(value)
                for value in [
                    *values,
                    mean,
                    minimum,
                    maximum,
                    span,
                    standard_deviation,
                ]
            )
            and abs(recomputed_mean - mean) <= 2e-6
            and abs(min(values) - minimum) <= 2e-6
            and abs(max(values) - maximum) <= 2e-6
            and abs((maximum - minimum) - span) <= 2e-6
            and abs(recomputed_std - standard_deviation) <= 2e-6
            and start_count == length
            and order_count == length
            and ensemble_size == length * length
            and str(row.get("cyclic_base_context_policy", ""))
            == V2_BASE_POLICY
        ):
            raise RuntimeError(
                f"Inconsistent V2 cyclic-base in-flight row: {sequence}"
            )
        row.update(
            {
                "cyclic_base_log_probability_mean": mean,
                "cyclic_base_log_probability_min": minimum,
                "cyclic_base_log_probability_max": maximum,
                "cyclic_base_log_probability_span": span,
                "cyclic_base_log_probability_std": standard_deviation,
                "cyclic_base_physical_start_count": start_count,
                "cyclic_base_decoder_order_count_per_start": order_count,
                "cyclic_base_total_ensemble_size": ensemble_size,
            }
        )
    return normalized


def validate_declared_artifact(declared: Mapping[str, Any], expected: Path) -> None:
    if not (
        expected.is_file()
        and Path(str(declared.get("path", ""))).resolve() == expected.resolve()
        and str(declared.get("sha256", "")) == sha256_file(expected)
    ):
        raise RuntimeError(f"Legacy artifact is absent, moved, or stale: {expected}")


def validate_and_reconstruct_legacy(
    old: Any,
    legacy_dir: Path,
    model_path: Path,
    baseline: Path,
    baseline_unique: Sequence[Mapping[str, Any]],
    numpy_module: Any,
) -> Tuple[
    Dict[str, Any],
    set[str],
    Dict[str, Dict[str, Any]],
    List[Dict[str, Any]],
]:
    manifest_path = legacy_dir / "directed_search_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    false_checks = {
        name
        for name, passed in dict(manifest.get("quality_checks") or {}).items()
        if not passed
    }
    if not (
        manifest.get("protocol") == old.V8_SEARCH_PROTOCOL
        and manifest.get("quality_gate") == "FAIL"
        and false_checks
        == {"all_missing_targets_have_a_novel_plausible_strict_candidate"}
        and manifest.get("missing_targets_before_search") == ["3ZGC"]
        and manifest.get("missing_targets_after_search") == ["3ZGC"]
        and int(manifest.get("released_candidates", -1)) == 0
        and int(dict(manifest.get("strict_probability_hit_counts") or {}).get("3ZGC", -1))
        > 0
        and manifest.get("model_sha256") == sha256_file(model_path)
        and manifest.get("baseline_manifest_sha256")
        == sha256_file(baseline / "generation_manifest.json")
        and dict(manifest.get("config") or {}).get("input_hashes", {}).get(
            "search_program"
        )
        == sha256_file(LEGACY_SEARCH_PATH)
    ):
        raise RuntimeError("Legacy V8 failure is not the exact recoverable 3ZGC state")
    artifacts = dict(manifest.get("artifacts") or {})
    for key, filename in {
        "controls": "mandatory_length_6_7_controls.csv",
        "plausibility": "qualified_candidate_plausibility_and_novelty.csv",
        "directed_candidates": "directed_candidates.csv",
        "trace": "search_trace_by_round.csv",
    }.items():
        validate_declared_artifact(dict(artifacts.get(key) or {}), legacy_dir / filename)
    ledger_artifacts = dict(artifacts.get("search_ledgers") or {})
    checkpoint_artifacts = dict(artifacts.get("checkpoints") or {})
    expected_ledgers = ["3zgc_round_00_initial.csv.gz"] + [
        f"3zgc_round_{index:02d}.csv.gz"
        for index in range(1, EXPECTED_LEGACY_ROUNDS + 1)
    ]
    expected_checkpoints = [
        f"3zgc_round_{index:02d}.json.gz"
        for index in range(1, EXPECTED_LEGACY_ROUNDS + 1)
    ]
    if set(ledger_artifacts) != set(expected_ledgers) or set(checkpoint_artifacts) != set(
        expected_checkpoints
    ):
        raise RuntimeError("Legacy ledger/checkpoint inventory is incomplete")
    for filename in expected_ledgers:
        validate_declared_artifact(
            dict(ledger_artifacts[filename]), legacy_dir / filename
        )
    for filename in expected_checkpoints:
        validate_declared_artifact(
            dict(checkpoint_artifacts[filename]), legacy_dir / "checkpoints" / filename
        )
    ranked = old.top_ranked_sequences(baseline_unique, "3ZGC")
    initial = [
        old.HISTORICAL_CONTROLS["3ZGC"]["sequence"],
        old.NATIVE_CONTROLS["3ZGC"],
        ranked[0],
    ]
    anchors = old.select_diverse_sequences(ranked[:128], 34, initial=initial)
    initial_provenance = old.zgc_initial_anchor_provenance(anchors, ranked[0])
    checkpoints = [legacy_dir / "checkpoints" / name for name in expected_checkpoints]
    completed, seen, _beam, qualified, _trace = old.reconstruct_and_validate_zgc_resume(
        legacy_dir,
        checkpoints,
        str(manifest.get("checkpoint_config_sha256", manifest.get("config_sha256", ""))),
        EXPECTED_BEAM_WIDTH,
        initial_provenance,
        EXPECTED_OFFSPRING,
        numpy_module,
        None,
        validate_model_scores=False,
    )
    expected_seen = int(dict(manifest.get("evaluated_sequence_counts") or {}).get("3ZGC", -1))
    expected_qualified = int(
        dict(manifest.get("strict_probability_hit_counts") or {}).get("3ZGC", -1)
    )
    if completed != EXPECTED_LEGACY_ROUNDS or len(seen) != expected_seen or len(qualified) != expected_qualified:
        raise RuntimeError("Legacy V8 ledgers do not reconstruct to manifest counts")
    # V2 used the complete ledger only to reconstruct ``seen`` and retained
    # solely the 2,881 strict rows.  V3 must preserve the other 265,484 model
    # observations as possible bridge states instead of making them
    # permanently unreachable.  Reloading is cheap compared with GPU scoring,
    # and the artifact hashes and exact round reconstruction were checked
    # immediately above.
    all_rows: List[Dict[str, Any]] = []
    all_sequences: set[str] = set()
    for filename in expected_ledgers:
        rows = [
            old.normalize_search_ledger_row(row)
            for row in old.read_gzip_csv(legacy_dir / filename)
        ]
        for row in rows:
            sequence = str(row.get("sequence", "")).upper()
            if not sequence or sequence in all_sequences:
                raise RuntimeError("Legacy V8 full ledger has a duplicate sequence")
            all_sequences.add(sequence)
            all_rows.append(row)
    if all_sequences != seen or len(all_rows) != expected_seen:
        raise RuntimeError("Legacy V8 full-ledger materialization differs from replay")
    return manifest, seen, qualified, all_rows


def exclusion_sets(
    old: Any,
    historical_rows: Sequence[Mapping[str, Any]],
    prior_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    target: str,
) -> Dict[str, set[str]]:
    historical_natural, historical_cyclic = old.exclusion_keys(historical_rows, target)
    prior_natural, prior_cyclic = old.exclusion_keys(prior_rows, target)
    pool_natural, pool_cyclic = old.exclusion_keys(baseline_rows, target)
    native_natural = {old.NATIVE_CONTROLS[target]}
    return {
        "historical_natural": historical_natural,
        "historical_cyclic": historical_cyclic,
        "prior_natural": prior_natural,
        "prior_cyclic": prior_cyclic,
        "pool_natural": pool_natural,
        "pool_cyclic": pool_cyclic,
        "native_natural": native_natural,
        "native_cyclic": {old.forward_cyclic_identity(value) for value in native_natural},
    }


def duplicate_reason(old: Any, sequence: str, sets: Mapping[str, set[str]]) -> str:
    cyclic = old.forward_cyclic_identity(sequence)
    for label in ("historical", "prior", "pool", "native"):
        if sequence in sets[f"{label}_natural"] or cyclic in sets[f"{label}_cyclic"]:
            return {
                "historical": "historical_4115",
                "prior": "prior_1333",
                "pool": "current_31500_pool",
                "native": "native",
            }[label]
    return ""


def evaluate_candidates(
    *,
    old: Any,
    target: str,
    candidates: Mapping[str, Mapping[str, Any]],
    base_scores: Mapping[str, Mapping[str, Any]],
    floor: float,
    full_payload: Mapping[str, Mapping[str, Any]],
    novelty_sets: Mapping[str, set[str]],
    batch_one_scorer: Any,
    selected_chain: str,
    max_release: int,
    id_prefix: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    evidence: List[Dict[str, Any]] = []
    releases: List[Dict[str, Any]] = []
    accepted_cyclic: set[str] = set()
    ordered = sorted(
        candidates,
        key=lambda sequence: (
            -float(candidates[sequence]["maximum_probability"]),
            -float(base_scores[sequence]["cyclic_base_log_probability_mean"]),
            sequence,
        ),
    )
    progress = old.ProgressBar(
        f"{target} V2 eligibility + true batch-one audit", len(ordered), unit="candidate"
    )
    for sequence in ordered:
        row = dict(candidates[sequence])
        base = dict(base_scores[sequence])
        payload = dict(full_payload[sequence])
        probabilities = [
            float(value) for value in json.loads(str(payload["methyl_probabilities"]))
        ]
        point = physical_argmax_summary(sequence, probabilities)
        search_maximum = float(row["maximum_probability"])
        full_maximum = float(point["physical_argmax_probability"])
        full_difference = abs(full_maximum - search_maximum)
        base_pass = float(base["cyclic_base_log_probability_mean"]) >= floor
        reason = duplicate_reason(old, sequence, novelty_sets)
        cyclic_key = old.forward_cyclic_identity(sequence)
        if not reason and cyclic_key in accepted_cyclic:
            reason = "accepted_forward_cyclic_equivalent"
        strict_full = (
            int(payload["design_methyl_count"]) > 0
            and old.strict_rounded_pass(full_maximum)
        )
        preeligible = (
            not reason
            and base_pass
            and strict_full
            and full_difference <= RESCORE_TOLERANCE
        )
        evidence_row = {
            "target_name": target,
            "sequence": sequence,
            "search_stage": row.get("search_stage", ""),
            "search_maximum_probability": search_maximum,
            "qualified_full_maximum_probability": full_maximum,
            "qualified_full_rescore_absolute_difference": full_difference,
            **base,
            "cyclic_base_plausibility_floor_1pct": floor,
            "passes_cyclic_base_plausibility": int(base_pass),
            "duplicate_reason": reason,
            "pre_batch_one_release_eligible": int(preeligible),
            **point,
            "predicted_methyl_positions_1based": payload["methyl_positions_1based"],
            "methyl_probability_representation_min": payload[
                "methyl_probability_representation_min"
            ],
            "methyl_probability_representation_max": payload[
                "methyl_probability_representation_max"
            ],
            "methyl_probability_representation_span": payload[
                "methyl_probability_representation_span"
            ],
            "methyl_probability_representation_span_max": payload[
                "methyl_probability_representation_span_max"
            ],
            "representation_threshold_disagreement_positions_1based": payload[
                "representation_threshold_disagreement_positions_1based"
            ],
            "annotation_representation_ensemble_size": payload[
                "annotation_representation_ensemble_size"
            ],
            "annotation_decoder_order_ensemble_size": payload[
                "annotation_decoder_order_ensemble_size"
            ],
            "batch_one_checked": 0,
            "batch_one_maximum_probability": "",
            "batch_one_rescore_absolute_difference": "",
            "release_eligible": 0,
        }
        if preeligible and len(releases) < max_release:
            independent = batch_one_scorer.score_full(
                target,
                [sequence],
                stage="V2 independent batch-one release audit",
                show_progress=False,
            )[sequence]
            independent_values = [
                float(value)
                for value in json.loads(str(independent["methyl_probabilities"]))
            ]
            independent_point = physical_argmax_summary(sequence, independent_values)
            independent_max = float(independent_point["physical_argmax_probability"])
            independent_difference = abs(independent_max - full_maximum)
            batch_pass = (
                int(independent["design_methyl_count"]) > 0
                and old.strict_rounded_pass(independent_max)
                and independent_difference <= RESCORE_TOLERANCE
                and int(independent_point["physical_argmax_position_1based"])
                == int(point["physical_argmax_position_1based"])
            )
            evidence_row.update(
                {
                    "batch_one_checked": 1,
                    "batch_one_maximum_probability": independent_max,
                    "batch_one_rescore_absolute_difference": independent_difference,
                    "release_eligible": int(batch_pass),
                }
            )
            if batch_pass:
                accepted_cyclic.add(cyclic_key)
                releases.append(
                    {
                        "candidate_id": f"{id_prefix}_{len(releases) + 1:04d}",
                        "target_name": target,
                        "selected_chain": selected_chain,
                        "design_seq": independent["design_seq"],
                        "design_natural_seq": sequence,
                        "native_seq": old.NATIVE_CONTROLS[target],
                        "native_length": len(sequence),
                        "design_length": len(sequence),
                        "length_match": 1,
                        "valid_token_gate": 1,
                        "temperature": TEMPERATURE,
                        "methyl_threshold": THRESHOLD,
                        "strict_threshold_operator": ">",
                        "candidate_origin": "V8_V2_CYCLIC_BASE_RECOVERY",
                        "search_stage": row.get("search_stage", ""),
                        "search_maximum_probability": search_maximum,
                        "qualified_full_maximum_probability": full_maximum,
                        "qualified_full_rescore_absolute_difference": full_difference,
                        "batch_one_maximum_probability": independent_max,
                        "batch_rescore_absolute_difference": independent_difference,
                        "base_log_probability_mean": "",
                        "base_log_probability_mean_all_orders": base[
                            "cyclic_base_log_probability_mean"
                        ],
                        **base,
                        "base_plausibility_context_policy": V2_BASE_POLICY,
                        "base_plausibility_floor_1pct": floor,
                        "forward_cyclic_identity": cyclic_key,
                        "control_or_candidate": "NOVEL_RECOVERY_CANDIDATE",
                        **independent,
                        **independent_point,
                        "sampling_context_policy": (
                            "DETERMINISTIC_DUAL_OBJECTIVE_SEARCH_NO_AUTOREGRESSIVE_SAMPLING"
                        ),
                        "sampling_path_annotation_status": "NOT_APPLICABLE_DIRECTED_SEARCH",
                        "sampling_path_methyl_probabilities": "",
                        "decoding_order_absolute": "",
                        "seed": "",
                        "draw_index_within_seed": "",
                        "occurrence_count": 1,
                        "seeds_observed": "DIRECTED_SEARCH_NOT_SAMPLED",
                        "seen_in_historical_4115_exact": 0,
                        "seen_in_historical_4115_naturalized": 0,
                        "seen_in_historical_4115": 0,
                        "seen_in_prior_1333_exact": 0,
                        "seen_in_prior_1333_naturalized": 0,
                        "seen_in_prior_1333": 0,
                        "passes_methylation_hard_gate": 1,
                        "eligible_for_new_permeability_screen": 0,
                        "permeability_screen_authorized_in_this_release": 0,
                        "permeability_eligibility_status": (
                            "DEFERRED_PENDING_GLOBAL_AND_CYCLIC_RMSD_LT_3A"
                        ),
                        "eligible_for_manual_structure_review": 1,
                        "permeability_id": "",
                    }
                )
        evidence.append(evidence_row)
        progress.update(1)
    progress.close()
    return evidence, releases


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("V8 V2 recovery requires NumPy and PyTorch") from exc
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    old = load_module("v8_legacy_search_for_cyclic_v2", LEGACY_SEARCH_PATH)
    frontier_v3 = (
        load_module("v8_full_frontier_recovery_v3", FRONTIER_V3_PATH)
        if bool(args.frontier_v3)
        else None
    )
    protocol = (
        frontier_v3.V3_SEARCH_PROTOCOL if frontier_v3 is not None else V2_SEARCH_PROTOCOL
    )

    model_path = Path(args.model_path).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    representation_path = Path(args.representation_audit).resolve()
    baseline = Path(args.baseline_run_dir).resolve()
    legacy_dir = Path(args.legacy_search_dir).resolve()
    plan_path = Path(args.plan).resolve()
    native_path = Path(args.native_jsonl).resolve()
    historical_path = Path(args.historical_designs_csv).resolve()
    prior_path = Path(args.prior_handoff_csv).resolve()
    prior_v2_dir = Path(args.prior_v2_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    immutable = [
        model_path,
        model_manifest_path,
        representation_path,
        baseline,
        legacy_dir,
        plan_path,
        native_path,
        historical_path,
        prior_path,
        SCRIPT_PATH,
        LEGACY_SEARCH_PATH,
    ]
    if frontier_v3 is not None:
        immutable.extend((prior_v2_dir, FRONTIER_V3_PATH))
    if any(old.paths_overlap(out_dir, path) for path in immutable):
        raise ValueError("V2 output overlaps an immutable input")
    for required in (
        model_path,
        model_manifest_path,
        representation_path,
        plan_path,
        native_path,
        historical_path,
        prior_path,
        LEGACY_SEARCH_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    baseline_manifest, baseline_unique, baseline_target_rows = old.validate_baseline(
        baseline,
        model_path,
        model_manifest_path,
        representation_path,
        plan_path,
        native_path,
        historical_path,
        prior_path,
    )
    model_manifest_payload = read_json(model_manifest_path)
    expert_source_by_residue = {
        str(token): str(source)
        for token, source in dict(
            model_manifest_payload.get("expert_source_by_residue") or {}
        ).items()
    }
    source_quality_checks = dict(model_manifest_payload.get("quality_checks") or {})
    serine_provenance_gate = (
        set(expert_source_by_residue) == set(NATURAL_AA)
        and expert_source_by_residue.get("S") == "v7_serine"
        and all(
            expert_source_by_residue.get(token) == "v6_non_ser"
            for token in NATURAL_AA
            if token != "S"
        )
        and source_quality_checks.get("all_shared_tensors_are_canonical_bitwise_identical")
        is True
        and source_quality_checks.get("all_non_ser_experts_are_v6_bitwise_identical")
        is True
        and source_quality_checks.get("serine_expert_is_v7_bitwise_identical")
        is True
        and source_quality_checks.get("every_ser_probability_is_inherited_from_v7")
        is True
    )
    if not serine_provenance_gate:
        raise RuntimeError("V8 s-to-S/Ser expert provenance gate is absent or stale")
    missing_targets = [
        str(value).upper()
        for value in baseline_manifest.get("targets_without_signature_candidate", [])
    ]
    if missing_targets != ["3ZGC"]:
        raise RuntimeError(
            "V2 is frozen for the observed single missing target 3ZGC; "
            f"found {missing_targets}"
        )
    selected_chains = {
        str(row["target_name"]).upper(): str(row["selected_chain"])
        for row in baseline_target_rows
    }
    legacy_manifest, legacy_seen, legacy_qualified, legacy_all_rows = validate_and_reconstruct_legacy(
        old, legacy_dir, model_path, baseline, baseline_unique, np
    )
    prior_v2_manifest: Optional[Dict[str, Any]] = None
    prior_v2_screen_rows: List[Dict[str, Any]] = []
    prior_v2_exact_rows: List[Dict[str, Any]] = []
    prior_v2_seen: set[str] = set()
    if frontier_v3 is not None:
        (
            prior_v2_manifest,
            prior_v2_screen_rows,
            prior_v2_exact_rows,
            prior_v2_seen,
        ) = (
            frontier_v3.validate_prior_v2_failure(
                prior_dir=prior_v2_dir,
                expected_model_sha256=sha256_file(model_path),
                expected_baseline_manifest_sha256=sha256_file(
                    baseline / "generation_manifest.json"
                ),
                expected_legacy_manifest_sha256=sha256_file(
                    legacy_dir / "directed_search_manifest.json"
                ),
                read_gzip_csv=old.read_gzip_csv,
            )
        )
    config = {
        "protocol": protocol,
        "legacy_manifest_sha256": sha256_file(
            legacy_dir / "directed_search_manifest.json"
        ),
        "model_sha256": sha256_file(model_path),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "representation_audit_sha256": sha256_file(representation_path),
        "baseline_manifest_sha256": sha256_file(
            baseline / "generation_manifest.json"
        ),
        "baseline_unique_sha256": sha256_file(baseline / "unique_candidates.csv"),
        "plan_sha256": sha256_file(plan_path),
        "native_sha256": sha256_file(native_path),
        "historical_sha256": sha256_file(historical_path),
        "prior_sha256": sha256_file(prior_path),
        "legacy_search_program_sha256": sha256_file(LEGACY_SEARCH_PATH),
        "v2_search_program_sha256": sha256_file(SCRIPT_PATH),
        "threshold": THRESHOLD,
        "strict_operator": ">",
        "temperature": TEMPERATURE,
        "base_percentile": BASE_PERCENTILE,
        "base_policy": V2_BASE_POLICY,
        "physical_representation_count": 7,
        "decoder_orders_per_representation": 7,
        "conditional_rounds": int(args.rounds),
        "beam_width": int(args.beam_width),
        "offspring_per_round": int(args.offspring_per_round),
        "multi_mutation_count_range_inclusive": [2, 4],
        "cyclic_base_shortlist_per_round": int(args.shortlist_per_round),
        "seed": V2_SEED,
        "methyl_batch_size": int(args.batch_size),
        "cyclic_base_batch_size": int(args.base_batch_size),
        "maximum_release_per_target": int(args.max_release_per_target),
        "requested_device": str(args.device),
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_device_name": (
            str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "full_conditional_budget_no_early_stop": True,
    }
    if frontier_v3 is not None:
        config.update(
            {
                "frontier_v3_program_sha256": sha256_file(FRONTIER_V3_PATH),
                "prior_v2_manifest_sha256": sha256_file(
                    prior_v2_dir / "cyclic_base_recovery_manifest.json"
                ),
                "prior_v2_config_sha256": prior_v2_manifest["config_sha256"],
                "prior_v2_methyl_screen_rows_available_to_frontier": len(
                    prior_v2_screen_rows
                ),
                "legacy_full_ledger_rows_available_to_frontier": len(
                    legacy_all_rows
                ),
                "legacy_bridge_exact_score_budget": int(
                    args.legacy_bridge_size
                ),
                "surrogate_protocol": frontier_v3.V3_SURROGATE_PROTOCOL,
                "surrogate_release_authority": "NONE_ACQUISITION_ONLY",
            }
        )
    config_digest = stable_json_sha256(config)
    manifest_path = out_dir / "cyclic_base_recovery_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("config_sha256") != config_digest:
            raise RuntimeError("Existing recovery output belongs to a different configuration")
        if existing.get("quality_gate") == "PASS":
            for declared in dict(existing.get("artifacts") or {}).values():
                if isinstance(declared, dict) and "path" in declared:
                    validate_declared_artifact(declared, Path(str(declared["path"])))
            print("V8 recovery: reused hash-valid PASS result", flush=True)
            return
    elif out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError("Recovery output exists; pass --resume after inspection")
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_v2_manifest_copy: Optional[Path] = None
    prior_v2_trace_copy: Optional[Path] = None
    if frontier_v3 is not None:
        prior_v2_manifest_copy = out_dir / "prior_v2_failure_manifest.json"
        prior_v2_trace_copy = out_dir / "prior_v2_failure_search_trace.csv"
        atomic_copy_file(
            prior_v2_dir / "cyclic_base_recovery_manifest.json",
            prior_v2_manifest_copy,
        )
        atomic_copy_file(
            prior_v2_dir / "search_trace_by_round.csv",
            prior_v2_trace_copy,
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU V2 recovery requires --allow-cpu")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise RuntimeError("No CUDA device is available")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    reannotator = old.load_module("v8_reannotator_for_cyclic_v2", old.REANNOTATOR_PATH)
    generator = old.load_module("v8_generator_for_cyclic_v2", old.GENERATOR_PATH)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from paper_clean_v28.clean_v28_common import (  # pylint: disable=import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        NAT_TO_METHYL_ABS,
        NATURAL_AA_ALPHABET,
        complete_decoding_order,
        cyclic_representation_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
    )

    native_rows = generator.read_jsonl(native_path)
    native_index = {
        generator.record_name(row, index): row for index, row in enumerate(native_rows)
    }
    model = load_v28_model(str(model_path), device)
    model.eval()
    common = {
        "EXTENDED_AA_ALPHABET": EXTENDED_AA_ALPHABET,
        "NAT_TO_METHYL_ABS": NAT_TO_METHYL_ABS,
        "NATURAL_AA_ALPHABET": NATURAL_AA_ALPHABET,
        "complete_decoding_order": complete_decoding_order,
        "cyclic_representation_known_sequence_methyl_probabilities": (
            cyclic_representation_known_sequence_methyl_probabilities
        ),
        "featurize_records": featurize_records,
    }
    methyl_scorer = old.MethylScorer(
        model,
        device,
        native_index,
        selected_chains,
        int(args.batch_size),
        torch,
        common,
        reannotator,
    )
    batch_one_scorer = old.MethylScorer(
        model, device, native_index, selected_chains, 1, torch, common, reannotator
    )
    target_records, _ = generator.prepare_target_records(
        native_rows, selected_chains, sorted(old.ALLOWED_RECOVERY_TARGETS)
    )
    base_scorer = CyclicBasePlausibilityScorer(
        model,
        device,
        target_records,
        int(args.base_batch_size),
        torch,
        functional,
        common,
        old.ProgressBar,
    )

    target = "3ZGC"
    baseline_sequences = sorted(
        {
            str(row["design_natural_seq"]).upper()
            for row in baseline_unique
            if str(row["target_name"]).upper() == target
        }
    )
    baseline_base = base_scorer.score_detailed(
        target, baseline_sequences, "V2 baseline cyclic-start plausibility floor"
    )
    floor = old.nearest_rank_percentile(
        [float(row["cyclic_base_log_probability_mean"]) for row in baseline_base.values()],
        BASE_PERCENTILE,
    )
    baseline_rows = [
        {"target_name": target, "sequence": sequence, **baseline_base[sequence]}
        for sequence in baseline_sequences
    ]
    baseline_base_path = out_dir / "baseline_cyclic_start_plausibility.csv"
    atomic_write_csv(baseline_base_path, baseline_rows, list(baseline_rows[0]))

    legacy_sequences = sorted(legacy_qualified)
    destination_minimal = methyl_scorer.score_minimal(
        target, legacy_sequences, "legacy strict-hit destination replay V2"
    )
    destination_legacy: Dict[str, Dict[str, Any]] = {}
    for sequence in legacy_sequences:
        source = legacy_qualified[sequence]
        observed = destination_minimal[sequence]
        difference = abs(
            float(source["maximum_probability"])
            - float(observed["maximum_probability"])
        )
        if not (
            difference <= RESCORE_TOLERANCE
            and int(source["argmax_position_1based"])
            == int(observed["argmax_position_1based"])
            and str(source["argmax_residue"]) == str(observed["argmax_residue"])
            and int(observed["passes_strict_probability"]) == 1
        ):
            raise RuntimeError(
                f"Legacy strict hit is not reproduced on destination: {sequence}"
            )
        destination_legacy[sequence] = {
            **observed,
            "search_stage": "legacy_strict_hit_cyclic_base_reaudit",
            "legacy_search_stage": source["search_stage"],
            "legacy_destination_probability_difference": difference,
        }
    legacy_base = base_scorer.score_detailed(
        target, legacy_sequences, "legacy strict-hit cyclic-start base re-audit"
    )
    legacy_full = methyl_scorer.score_full(
        target, legacy_sequences, stage="legacy strict-hit physical-position re-audit"
    )
    novelty = exclusion_sets(
        old,
        read_csv(historical_path),
        read_csv(prior_path),
        baseline_unique,
        target,
    )
    legacy_evidence, legacy_releases = evaluate_candidates(
        old=old,
        target=target,
        candidates=destination_legacy,
        base_scores=legacy_base,
        floor=floor,
        full_payload=legacy_full,
        novelty_sets=novelty,
        batch_one_scorer=batch_one_scorer,
        selected_chain=selected_chains[target],
        max_release=int(args.max_release_per_target),
        id_prefix="v8v2_3zgc_reaudit",
    )
    legacy_evidence_path = out_dir / "legacy_strict_hit_cyclic_reaudit.csv"
    atomic_write_csv(
        legacy_evidence_path,
        legacy_evidence,
        list(legacy_evidence[0]),
    )

    legacy_bridge_rows: List[Dict[str, Any]] = []
    legacy_bridge_path: Optional[Path] = None
    surrogate_report_path: Optional[Path] = None
    surrogate: Optional[Any] = None
    surrogate_training_rows: List[Dict[str, Any]] = []
    if frontier_v3 is not None:
        # Exact V2 rows, the baseline, and all 2,881 strict re-audits provide a
        # large frozen training set for acquisition ranking.  The surrogate
        # never decides release; its only role is deciding which previously
        # discarded legacy rows deserve the expensive exact cyclic-base score.
        surrogate_training_rows = [
            *baseline_rows,
            *[
                {
                    "sequence": sequence,
                    **legacy_base[sequence],
                }
                for sequence in legacy_sequences
            ],
            *prior_v2_exact_rows,
        ]
        surrogate = frontier_v3.KmerBaseSurrogate(NATURAL_AA, 7)
        initial_surrogate_report = surrogate.fit(surrogate_training_rows)
        prior_v2_exact_sequences = {
            str(row["sequence"]) for row in prior_v2_exact_rows
        }
        baseline_sequence_set = set(baseline_sequences)
        legacy_baseline_overlap = set(legacy_seen) & baseline_sequence_set
        pre_v3_frontier_rows = [
            *[
                row
                for row in legacy_all_rows
                if str(row["sequence"]) not in baseline_sequence_set
            ],
            *[
                row
                for row in prior_v2_screen_rows
                if str(row["sequence"]) not in prior_v2_exact_sequences
            ],
        ]
        if len({str(row["sequence"]) for row in pre_v3_frontier_rows}) != len(
            pre_v3_frontier_rows
        ):
            raise RuntimeError("V3 pre-search frontier contains duplicate sequences")
        selected_bridge = frontier_v3.select_surrogate_frontier(
            rows=pre_v3_frontier_rows,
            surrogate=surrogate,
            limit=int(args.legacy_bridge_size),
            length=7,
            floor=floor,
            diversity_fill=deterministic_diversity_fill,
            exclude_strict=True,
        )
        bridge_sequences = [str(row["sequence"]).upper() for row in selected_bridge]
        if set(bridge_sequences) & set(legacy_sequences):
            raise RuntimeError("V3 non-strict bridge overlaps legacy strict hits")
        if set(bridge_sequences) & baseline_sequence_set:
            raise RuntimeError("V3 exact bridge wastes budget on an exact baseline row")
        bridge_selection_path = out_dir / "pre_v3_full_frontier_selection.csv.gz"
        legacy_bridge_path = out_dir / "pre_v3_full_frontier_cyclic_base.csv.gz"
        bridge_state_path = out_dir / "v3_frontier_state.json"
        reusable_bridge = False
        if args.resume and bridge_state_path.is_file():
            bridge_state = read_json(bridge_state_path)
            reusable_bridge = (
                bridge_state.get("config_sha256") == config_digest
                and bridge_state.get("selection_filename")
                == bridge_selection_path.name
                and bridge_state.get("exact_filename") == legacy_bridge_path.name
                and bridge_selection_path.is_file()
                and legacy_bridge_path.is_file()
                and bridge_state.get("selection_sha256")
                == sha256_file(bridge_selection_path)
                and bridge_state.get("exact_sha256")
                == sha256_file(legacy_bridge_path)
                and bridge_state.get("selection_sequence_sha256")
                == hashlib.sha256(
                    ("\n".join(bridge_sequences) + "\n").encode("ascii")
                ).hexdigest()
            )
        if reusable_bridge:
            persisted_selection = old.read_gzip_csv(bridge_selection_path)
            if [
                str(row.get("sequence", "")).upper()
                for row in persisted_selection
            ] != bridge_sequences:
                raise RuntimeError("V3 reusable bridge selection order changed")
            legacy_bridge_rows = frontier_v3.validate_exact_frontier_rows(
                old.read_gzip_csv(legacy_bridge_path),
                bridge_sequences,
                V2_BASE_POLICY,
                7,
            )
            print(
                f"V3 full legacy frontier: reused {len(legacy_bridge_rows):,} "
                "hash-pinned exact bridge rows",
                flush=True,
            )
        else:
            bridge_minimal = methyl_scorer.score_minimal(
                target,
                bridge_sequences,
                "V3 full-legacy frontier destination replay",
            )
            source_bridge = {
                str(row["sequence"]).upper(): row for row in selected_bridge
            }
            replayed_selection: List[Dict[str, Any]] = []
            for sequence in bridge_sequences:
                source = source_bridge[sequence]
                observed = bridge_minimal[sequence]
                difference = abs(
                    float(source["maximum_probability"])
                    - float(observed["maximum_probability"])
                )
                if not (
                    difference <= RESCORE_TOLERANCE
                    and int(source["argmax_position_1based"])
                    == int(observed["argmax_position_1based"])
                    and str(source["argmax_residue"])
                    == str(observed["argmax_residue"])
                    and int(source["passes_strict_probability"])
                    == int(observed["passes_strict_probability"])
                    == 0
                ):
                    raise RuntimeError(
                        "V3 legacy bridge is not reproduced on destination: "
                        + sequence
                    )
                replayed_selection.append(
                    {
                        **source,
                        **observed,
                        "search_stage": "V3 full-legacy frontier exact bridge",
                        "legacy_destination_probability_difference": difference,
                    }
                )
            atomic_write_gzip_csv(
                bridge_selection_path,
                replayed_selection,
                list(replayed_selection[0]),
            )
            exact_bridge = base_scorer.score_detailed(
                target,
                bridge_sequences,
                "V3 full-legacy frontier exact cyclic base",
            )
            replayed_by_sequence = {
                str(row["sequence"]): row for row in replayed_selection
            }
            legacy_bridge_rows = [
                {
                    **replayed_by_sequence[sequence],
                    **exact_bridge[sequence],
                }
                for sequence in bridge_sequences
            ]
            atomic_write_gzip_csv(
                legacy_bridge_path,
                legacy_bridge_rows,
                list(legacy_bridge_rows[0]),
            )
            atomic_write_json(
                bridge_state_path,
                {
                    "protocol": frontier_v3.V3_SEARCH_PROTOCOL,
                    "config_sha256": config_digest,
                    "selection_filename": bridge_selection_path.name,
                    "selection_sha256": sha256_file(bridge_selection_path),
                    "exact_filename": legacy_bridge_path.name,
                    "exact_sha256": sha256_file(legacy_bridge_path),
                    "selection_sequence_sha256": hashlib.sha256(
                        ("\n".join(bridge_sequences) + "\n").encode("ascii")
                    ).hexdigest(),
                },
            )
        surrogate_training_rows.extend(legacy_bridge_rows)
        after_bridge_surrogate_report = surrogate.fit(surrogate_training_rows)
        surrogate_report_path = out_dir / "v3_surrogate_audit.json"
        atomic_write_json(
            surrogate_report_path,
            {
                "protocol": frontier_v3.V3_SURROGATE_PROTOCOL,
                "initial_fit": initial_surrogate_report,
                "after_exact_legacy_bridge_fit": after_bridge_surrogate_report,
                "legacy_frontier_rows_available": len(legacy_all_rows),
                "legacy_rows_already_exact_in_baseline": len(
                    legacy_baseline_overlap
                ),
                "prior_v2_methyl_screen_rows_available": len(
                    prior_v2_screen_rows
                ),
                "prior_v2_exact_rows_reused": len(prior_v2_exact_rows),
                "non_exact_pre_v3_frontier_rows_available": len(
                    pre_v3_frontier_rows
                ),
                "legacy_strict_rows_separately_audited": len(legacy_sequences),
                "legacy_non_strict_bridge_rows_exactly_scored": len(
                    legacy_bridge_rows
                ),
                "selection": frontier_v3.frontier_summary(
                    selected_bridge, floor
                ),
                "hard_gate_threshold": THRESHOLD,
                "hard_gate_cyclic_base_floor": floor,
                "surrogate_release_authority": "NONE",
            },
        )

    conditional_search_ran = not legacy_releases
    search_evidence: List[Dict[str, Any]] = []
    search_releases: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    screening_paths: List[Path] = []
    shortlist_paths: List[Path] = []
    if conditional_search_ran:
        round_stage_prefix = (
            "v3_full_frontier" if frontier_v3 is not None else "v2_joint"
        )
        baseline_methyl = methyl_scorer.score_minimal(
            target, baseline_sequences, "V2 baseline joint-search anchors"
        )
        initial_rows: List[Dict[str, Any]] = []
        for sequence in baseline_sequences:
            initial_rows.append(
                {
                    **baseline_methyl[sequence],
                    **baseline_base[sequence],
                    "search_stage": "V2 baseline joint-search anchors",
                }
            )
        for sequence in legacy_sequences:
            initial_rows.append(
                {
                    **destination_legacy[sequence],
                    **legacy_base[sequence],
                }
            )
        if frontier_v3 is not None:
            initial_rows.extend(legacy_bridge_rows)
            initial_rows.extend(prior_v2_exact_rows)
        beam = (
            frontier_v3.select_exact_dual_objective_beam(
                rows=initial_rows,
                limit=int(args.beam_width),
                length=7,
                floor=floor,
                diversity_fill=deterministic_diversity_fill,
            )
            if frontier_v3 is not None
            else select_dual_objective_beam(
                initial_rows, int(args.beam_width), 7, floor
            )
        )
        seen = set(legacy_seen) | set(baseline_sequences) | set(prior_v2_seen)
        all_joint_hits: Dict[str, Dict[str, Any]] = {}
        for row in initial_rows:
            if (
                int(row["passes_strict_probability"]) == 1
                and float(row["cyclic_base_log_probability_mean"]) >= floor
            ):
                all_joint_hits[str(row["sequence"])] = row
        start_round = 1
        state_path = out_dir / "v2_resume_state.json"
        inflight_path = out_dir / "v2_inflight_round_state.json"
        if args.resume and state_path.is_file():
            state = read_json(state_path)
            completed_round = int(state.get("completed_round", -1))
            if not (
                state.get("config_sha256") == config_digest
                and 1 <= completed_round <= int(args.rounds)
            ):
                raise RuntimeError("V2 resume state has a different configuration/round")
            expected_screen_hashes = dict(state.get("methyl_screen_sha256") or {})
            expected_shortlist_hashes = dict(
                state.get("cyclic_base_shortlist_sha256") or {}
            )
            expected_round_names = {
                f"v2_round_{index:02d}_methyl_screen.csv.gz"
                for index in range(1, completed_round + 1)
            }
            expected_shortlist_names = {
                f"v2_round_{index:02d}_cyclic_base.csv.gz"
                for index in range(1, completed_round + 1)
            }
            if set(expected_screen_hashes) != expected_round_names or set(
                expected_shortlist_hashes
            ) != expected_shortlist_names:
                raise RuntimeError("V2 resume state lacks a complete round artifact map")
            reconstructed_seen = (
                set(legacy_seen) | set(baseline_sequences) | set(prior_v2_seen)
            )
            for index in range(1, completed_round + 1):
                screen_path = out_dir / f"v2_round_{index:02d}_methyl_screen.csv.gz"
                shortlist_path = out_dir / f"v2_round_{index:02d}_cyclic_base.csv.gz"
                if not (
                    screen_path.is_file()
                    and shortlist_path.is_file()
                    and sha256_file(screen_path)
                    == expected_screen_hashes[screen_path.name]
                    and sha256_file(shortlist_path)
                    == expected_shortlist_hashes[shortlist_path.name]
                ):
                    raise RuntimeError(
                        f"V2 resume round-{index} artifact is absent or stale"
                    )
                screen_rows = old.read_gzip_csv(screen_path)
                screen_sequences = [
                    str(row.get("sequence", "")).upper() for row in screen_rows
                ]
                if (
                    any(not sequence for sequence in screen_sequences)
                    or len(screen_sequences) != len(set(screen_sequences))
                    or set(screen_sequences) & reconstructed_seen
                ):
                    raise RuntimeError(
                        f"V2 resume round-{index} methyl screen is duplicated/malformed"
                    )
                reconstructed_seen.update(screen_sequences)
                screening_paths.append(screen_path)
                shortlist_paths.append(shortlist_path)
                if frontier_v3 is not None:
                    persisted_exact = old.read_gzip_csv(shortlist_path)
                    persisted_sequences = [
                        str(row.get("sequence", "")).upper()
                        for row in persisted_exact
                    ]
                    surrogate_training_rows.extend(
                        frontier_v3.validate_exact_frontier_rows(
                            persisted_exact,
                            persisted_sequences,
                            V2_BASE_POLICY,
                            7,
                        )
                    )
            if frontier_v3 is not None:
                surrogate.fit(surrogate_training_rows)
            observed_seen_hash = hashlib.sha256(
                ("\n".join(sorted(reconstructed_seen)) + "\n").encode("ascii")
            ).hexdigest()
            if not (
                len(reconstructed_seen) == int(state.get("seen_sequence_count", -1))
                and observed_seen_hash == str(state.get("seen_sequence_sha256", ""))
            ):
                raise RuntimeError("V2 resume seen-set hash/count mismatch")
            resumed_beam = [dict(row) for row in state.get("beam", [])]
            resumed_hits = [dict(row) for row in state.get("joint_hits", [])]
            resumed_trace = [dict(row) for row in state.get("trace", [])]
            if not (
                len(resumed_beam) == int(args.beam_width)
                and len({str(row.get("sequence", "")) for row in resumed_beam})
                == len(resumed_beam)
                and all(
                    len(str(row.get("sequence", ""))) == 7
                    and set(str(row.get("sequence", ""))) <= set(NATURAL_AA)
                    and math.isfinite(float(row["maximum_probability"]))
                    and math.isfinite(
                        float(row["cyclic_base_log_probability_mean"])
                    )
                    for row in resumed_beam
                )
                and len(resumed_trace) == completed_round
                and {str(row.get("stage", "")) for row in resumed_trace}
                == {
                    f"{round_stage_prefix}_round_{index:02d}"
                    for index in range(1, completed_round + 1)
                }
                and len({str(row.get("sequence", "")) for row in resumed_hits})
                == len(resumed_hits)
                and all(
                    int(row["passes_strict_probability"]) == 1
                    and float(row["cyclic_base_log_probability_mean"]) >= floor
                    for row in resumed_hits
                )
            ):
                raise RuntimeError("V2 resume beam/joint-hit/trace state is malformed")
            seen = reconstructed_seen
            beam = resumed_beam
            all_joint_hits = {
                str(row["sequence"]): row for row in resumed_hits
            }
            trace_rows = resumed_trace
            start_round = completed_round + 1
            print(
                f"V2 conditional search: resumed after round {completed_round}",
                flush=True,
            )
            if inflight_path.is_file():
                stale_inflight = read_json(inflight_path)
                if (
                    stale_inflight.get("config_sha256") == config_digest
                    and int(stale_inflight.get("round_index", -1))
                    <= completed_round
                ):
                    inflight_path.unlink()
        for round_index in range(start_round, int(args.rounds) + 1):
            provenance = v2_round_provenance(
                beam, round_index, int(args.offspring_per_round), np
            )
            to_score = sorted(set(provenance) - seen)
            parent_base = {
                str(row["sequence"]): float(row["cyclic_base_log_probability_mean"])
                for row in beam
            }
            methyl_stage = (
                f"V3 full-frontier methyl screen round {round_index:02d}"
                if frontier_v3 is not None
                else f"V2 methyl screen round {round_index:02d}"
            )
            screen_path = out_dir / f"v2_round_{round_index:02d}_methyl_screen.csv.gz"
            shortlist_path = out_dir / f"v2_round_{round_index:02d}_cyclic_base.csv.gz"
            round_context = v2_round_context(
                config_digest,
                round_index,
                beam,
                sorted(seen),
                to_score,
                provenance,
            )
            inflight: Optional[Dict[str, Any]] = None
            if args.resume and inflight_path.is_file():
                inflight = read_json(inflight_path)
                if any(
                    inflight.get(key) != value
                    for key, value in round_context.items()
                ):
                    raise RuntimeError(
                        "V2 in-flight round state belongs to a different context"
                    )
                if inflight.get("phase") not in {
                    "methyl_screen_complete",
                    "cyclic_base_shortlist_complete",
                }:
                    raise RuntimeError("V2 in-flight round phase is malformed")
                if not (
                    inflight.get("methyl_screen_filename") == screen_path.name
                    and screen_path.is_file()
                    and sha256_file(screen_path)
                    == inflight.get("methyl_screen_sha256")
                ):
                    raise RuntimeError("V2 in-flight methyl screen is absent or stale")
                screen_rows = validate_v2_methyl_screen_rows(
                    old,
                    old.read_gzip_csv(screen_path),
                    target,
                    methyl_stage,
                    to_score,
                    provenance,
                    parent_base,
                )
                print(
                    f"V2 round {round_index:02d}: reused hash-pinned "
                    f"methyl screen ({len(screen_rows):,} rows)",
                    flush=True,
                )
            else:
                minimal = methyl_scorer.score_minimal(
                    target, to_score, methyl_stage
                )
                screen_rows = []
                for sequence in to_score:
                    source = provenance[sequence]
                    parent = str(source["parent_sequence"])
                    screen_rows.append(
                        {
                            **minimal[sequence],
                            **source,
                            "parent_cyclic_base_log_probability_mean": parent_base[
                                parent
                            ],
                        }
                    )
                atomic_write_gzip_csv(
                    screen_path,
                    screen_rows,
                    list(screen_rows[0]) if screen_rows else ["sequence"],
                )
                inflight = {
                    **round_context,
                    "phase": "methyl_screen_complete",
                    "methyl_screen_filename": screen_path.name,
                    "methyl_screen_sha256": sha256_file(screen_path),
                }
                atomic_write_json(inflight_path, inflight)
            screening_paths.append(screen_path)
            if frontier_v3 is not None:
                selected_frontier_rows = frontier_v3.select_surrogate_frontier(
                    rows=screen_rows,
                    surrogate=surrogate,
                    limit=int(args.shortlist_per_round),
                    length=7,
                    floor=floor,
                    diversity_fill=deterministic_diversity_fill,
                )
                shortlist_sequences = [
                    str(row["sequence"]) for row in selected_frontier_rows
                ]
                selected_frontier_by_sequence = {
                    str(row["sequence"]): row for row in selected_frontier_rows
                }
            else:
                shortlist_sequences = select_methyl_screen_shortlist(
                    screen_rows, int(args.shortlist_per_round), 7
                )
                selected_frontier_by_sequence = {}
            if (
                inflight is not None
                and inflight.get("phase") == "cyclic_base_shortlist_complete"
            ):
                expected_shortlist_hash = hashlib.sha256(
                    ("\n".join(shortlist_sequences) + "\n").encode("ascii")
                ).hexdigest()
                if not (
                    inflight.get("cyclic_base_shortlist_filename")
                    == shortlist_path.name
                    and inflight.get("cyclic_base_shortlist_sequence_sha256")
                    == expected_shortlist_hash
                    and shortlist_path.is_file()
                    and sha256_file(shortlist_path)
                    == inflight.get("cyclic_base_shortlist_sha256")
                ):
                    raise RuntimeError(
                        "V2 in-flight cyclic-base shortlist is absent or stale"
                    )
                shortlist_rows = validate_v2_cyclic_base_rows(
                    old,
                    old.read_gzip_csv(shortlist_path),
                    target,
                    methyl_stage,
                    shortlist_sequences,
                    provenance,
                    parent_base,
                    7,
                )
                print(
                    f"V2 round {round_index:02d}: reused hash-pinned "
                    f"cyclic-base shortlist ({len(shortlist_rows):,} rows)",
                    flush=True,
                )
            else:
                detailed = base_scorer.score_detailed(
                    target,
                    shortlist_sequences,
                    (
                        f"V3 full-frontier cyclic-base shortlist round "
                        f"{round_index:02d}"
                        if frontier_v3 is not None
                        else f"V2 cyclic-base shortlist round {round_index:02d}"
                    ),
                )
                screen_by_sequence = {
                    str(row["sequence"]): row for row in screen_rows
                }
                if frontier_v3 is not None:
                    screen_by_sequence.update(selected_frontier_by_sequence)
                shortlist_rows = [
                    {**screen_by_sequence[sequence], **detailed[sequence]}
                    for sequence in shortlist_sequences
                ]
                atomic_write_gzip_csv(
                    shortlist_path,
                    shortlist_rows,
                    list(shortlist_rows[0]) if shortlist_rows else ["sequence"],
                )
                inflight = {
                    **round_context,
                    "phase": "cyclic_base_shortlist_complete",
                    "methyl_screen_filename": screen_path.name,
                    "methyl_screen_sha256": sha256_file(screen_path),
                    "cyclic_base_shortlist_filename": shortlist_path.name,
                    "cyclic_base_shortlist_sha256": sha256_file(shortlist_path),
                    "cyclic_base_shortlist_sequence_sha256": hashlib.sha256(
                        ("\n".join(shortlist_sequences) + "\n").encode("ascii")
                    ).hexdigest(),
                }
                atomic_write_json(inflight_path, inflight)
            shortlist_paths.append(shortlist_path)
            if frontier_v3 is not None:
                surrogate_training_rows.extend(shortlist_rows)
                round_surrogate_report = surrogate.fit(surrogate_training_rows)
                current_surrogate_audit = read_json(surrogate_report_path)
                current_surrogate_audit[
                    f"after_v3_round_{round_index:02d}_fit"
                ] = round_surrogate_report
                atomic_write_json(surrogate_report_path, current_surrogate_audit)
            for row in shortlist_rows:
                if (
                    int(row["passes_strict_probability"]) == 1
                    and float(row["cyclic_base_log_probability_mean"]) >= floor
                ):
                    all_joint_hits[str(row["sequence"])] = row
            combined = {str(row["sequence"]): row for row in beam}
            combined.update({str(row["sequence"]): row for row in shortlist_rows})
            beam = (
                frontier_v3.select_exact_dual_objective_beam(
                    rows=list(combined.values()),
                    limit=int(args.beam_width),
                    length=7,
                    floor=floor,
                    diversity_fill=deterministic_diversity_fill,
                )
                if frontier_v3 is not None
                else select_dual_objective_beam(
                    list(combined.values()), int(args.beam_width), 7, floor
                )
            )
            seen.update(to_score)
            trace_rows.append(
                {
                    "target_name": target,
                    "stage": f"{round_stage_prefix}_round_{round_index:02d}",
                    "generated_unique": len(provenance),
                    "newly_methyl_scored": len(to_score),
                    "cyclic_base_shortlist": len(shortlist_rows),
                    "strict_methyl_in_shortlist": sum(
                        int(row["passes_strict_probability"]) for row in shortlist_rows
                    ),
                    "joint_hard_gate_hits_cumulative": len(all_joint_hits),
                    "beam_base_pass": sum(
                        float(row["cyclic_base_log_probability_mean"]) >= floor
                        for row in beam
                    ),
                    "beam_strict_methyl": sum(
                        int(row["passes_strict_probability"]) for row in beam
                    ),
                    "beam_joint_hard_gate": sum(
                        int(row["passes_strict_probability"]) == 1
                        and float(row["cyclic_base_log_probability_mean"]) >= floor
                        for row in beam
                    ),
                    "maximum_probability": max(
                        (float(row["maximum_probability"]) for row in beam),
                        default=0.0,
                    ),
                    "maximum_cyclic_base_log_probability_mean": max(
                        (
                            float(row["cyclic_base_log_probability_mean"])
                            for row in beam
                        ),
                        default=float("-inf"),
                    ),
                }
            )
            atomic_write_json(
                state_path,
                {
                    "config_sha256": config_digest,
                    "completed_round": round_index,
                    "seen_sequence_sha256": hashlib.sha256(
                        ("\n".join(sorted(seen)) + "\n").encode("ascii")
                    ).hexdigest(),
                    "seen_sequence_count": len(seen),
                    "beam": beam,
                    "joint_hits": list(all_joint_hits.values()),
                    "trace": trace_rows,
                    "methyl_screen_sha256": {
                        path.name: sha256_file(path) for path in screening_paths
                    },
                    "cyclic_base_shortlist_sha256": {
                        path.name: sha256_file(path) for path in shortlist_paths
                    },
                },
            )
            if inflight_path.is_file():
                inflight_path.unlink()
        joint_sequences = sorted(all_joint_hits)
        if joint_sequences:
            joint_minimal = {
                sequence: {
                    **all_joint_hits[sequence],
                    "search_stage": all_joint_hits[sequence].get(
                        "search_stage",
                        (
                            "V3 full-frontier dual-objective search"
                            if frontier_v3 is not None
                            else "V2 dual-objective search"
                        ),
                    ),
                }
                for sequence in joint_sequences
            }
            joint_base = {
                sequence: {
                    key: value
                    for key, value in all_joint_hits[sequence].items()
                    if key.startswith("cyclic_base_")
                }
                for sequence in joint_sequences
            }
            joint_full = methyl_scorer.score_full(
                target,
                joint_sequences,
                stage=(
                    "V3 joint-hit physical-position audit"
                    if frontier_v3 is not None
                    else "V2 joint-hit physical-position audit"
                ),
            )
            search_evidence, search_releases = evaluate_candidates(
                old=old,
                target=target,
                candidates=joint_minimal,
                base_scores=joint_base,
                floor=floor,
                full_payload=joint_full,
                novelty_sets=novelty,
                batch_one_scorer=batch_one_scorer,
                selected_chain=selected_chains[target],
                max_release=int(args.max_release_per_target),
                id_prefix=(
                    "v8v3_3zgc_joint"
                    if frontier_v3 is not None
                    else "v8v2_3zgc_joint"
                ),
            )

    release_rows = legacy_releases if legacy_releases else search_releases
    combined_evidence = [*legacy_evidence, *search_evidence]
    evidence_path = out_dir / "cyclic_base_plausibility_and_position_evidence.csv"
    atomic_write_csv(
        evidence_path,
        combined_evidence,
        list(combined_evidence[0]),
    )
    candidates_path = out_dir / "directed_candidates.csv"
    atomic_write_csv(
        candidates_path,
        release_rows,
        list(release_rows[0])
        if release_rows
        else ["candidate_id", "target_name", "design_seq", "design_natural_seq"],
    )
    trace_path = out_dir / "search_trace_by_round.csv"
    atomic_write_csv(
        trace_path,
        trace_rows,
        list(trace_rows[0])
        if trace_rows
        else [
            "target_name",
            "stage",
            "generated_unique",
            "newly_methyl_scored",
            "cyclic_base_shortlist",
        ],
    )

    control_rows: List[Dict[str, Any]] = []
    for control_target in sorted(old.ALLOWED_RECOVERY_TARGETS):
        sequences = [
            old.HISTORICAL_CONTROLS[control_target]["sequence"],
            old.NATIVE_CONTROLS[control_target],
        ]
        full = methyl_scorer.score_full(
            control_target, sequences, stage="V2 mandatory length control"
        )
        base = base_scorer.score_detailed(
            control_target, sequences, "V2 mandatory length control cyclic base"
        )
        for control_type, sequence in zip(("withdrawn_historical", "native"), sequences):
            probabilities = json.loads(str(full[sequence]["methyl_probabilities"]))
            control_rows.append(
                {
                    "target_name": control_target,
                    "selected_chain": selected_chains[control_target],
                    "control_type": control_type,
                    "natural_sequence": sequence,
                    "length": len(sequence),
                    **physical_argmax_summary(sequence, probabilities),
                    **base[sequence],
                    "methyl_probabilities": full[sequence]["methyl_probabilities"],
                    "methyl_probability_representation_min": full[sequence][
                        "methyl_probability_representation_min"
                    ],
                    "methyl_probability_representation_max": full[sequence][
                        "methyl_probability_representation_max"
                    ],
                    "methyl_probability_representation_span": full[sequence][
                        "methyl_probability_representation_span"
                    ],
                    "release_eligibility": "CONTROL_ONLY_NEVER_RELEASE",
                }
            )
    controls_path = out_dir / "mandatory_length_6_7_controls.csv"
    atomic_write_csv(controls_path, control_rows, list(control_rows[0]))

    full_budget_complete = (
        not conditional_search_ran
        or len(trace_rows) == int(args.rounds)
        and {str(row["stage"]) for row in trace_rows}
        == {
            f"{round_stage_prefix}_round_{index:02d}"
            for index in range(1, int(args.rounds) + 1)
        }
    )
    exact_search_rows = [
        *legacy_bridge_rows,
        *prior_v2_exact_rows,
        *[
            row
            for path in shortlist_paths
            for row in old.read_gzip_csv(path)
        ],
    ]
    quality_checks = {
        "legacy_failure_is_hash_pinned_and_exactly_reconstructed": True,
        "s_to_S_source_scoped_model_and_representation_are_hash_pinned": (
            serine_provenance_gate
        ),
        "legacy_strict_hits_are_destination_methyl_rescored": (
            len(destination_legacy) == len(legacy_qualified)
        ),
        "cyclic_base_uses_joint_coordinate_sequence_roll_and_residue_index_reset": True,
        "cyclic_base_averages_all_physical_starts_and_all_decoder_orders": all(
            int(row["cyclic_base_total_ensemble_size"]) == len(row["sequence"]) ** 2
            for row in [*combined_evidence, *exact_search_rows]
        ),
        "baseline_and_candidates_use_identical_cyclic_base_policy": all(
            row["cyclic_base_context_policy"] == V2_BASE_POLICY
            for row in [*baseline_rows, *combined_evidence, *exact_search_rows]
        ),
        "physical_position_vectors_and_argmax_are_persisted": all(
            len(json.loads(str(row["physical_probability_vector"])))
            == len(str(row["sequence"]))
            and 1
            <= int(row["physical_argmax_position_1based"])
            <= len(str(row["sequence"]))
            for row in combined_evidence
        ),
        "conditional_search_policy_is_obeyed": (
            conditional_search_ran == (len(legacy_releases) == 0)
        ),
        "conditional_fixed_budget_completed_without_early_stop": full_budget_complete,
        "strict_threshold_remains_greater_than_0_6": all(
            old.strict_rounded_pass(float(row["batch_one_maximum_probability"]))
            for row in release_rows
        ),
        "every_release_passes_cyclic_base_floor": all(
            float(row["cyclic_base_log_probability_mean"]) >= floor
            for row in release_rows
        ),
        "every_release_passes_independent_batch_one": all(
            float(row["batch_rescore_absolute_difference"]) <= RESCORE_TOLERANCE
            for row in release_rows
        ),
        "at_least_one_real_3zgc_candidate_is_released": bool(release_rows),
        "no_threshold_relaxation_formal_abstention_handoff_or_permeability": True,
    }
    if frontier_v3 is not None:
        quality_checks.update(
            {
                "all_268365_legacy_rows_are_represented_by_exact_baseline_or_frontier_selection": (
                    len(legacy_all_rows) == len(legacy_seen) == 268_365
                    and {str(row["sequence"]) for row in legacy_all_rows}
                    == set(legacy_seen)
                    and legacy_baseline_overlap <= baseline_sequence_set
                ),
                "all_159329_v2_screen_rows_are_reused_as_exact_or_frontier": (
                    len(prior_v2_screen_rows) == len(prior_v2_seen) == 159_329
                    and {str(row["sequence"]) for row in prior_v2_screen_rows}
                    == set(prior_v2_seen)
                    and prior_v2_exact_sequences <= set(prior_v2_seen)
                ),
                "non_strict_legacy_bridge_received_exact_cyclic_base_scores": (
                    len(legacy_bridge_rows) == int(args.legacy_bridge_size)
                    and all(
                        int(row["passes_strict_probability"]) == 0
                        and row["cyclic_base_context_policy"] == V2_BASE_POLICY
                        and str(row["sequence"]) not in baseline_sequence_set
                        for row in legacy_bridge_rows
                    )
                ),
                "completed_v2_failure_is_hash_pinned_and_reused_not_rerun": (
                    prior_v2_manifest is not None
                    and int(prior_v2_manifest["conditional_rounds_completed"]) == 6
                    and len(prior_v2_exact_rows) == 6 * 4096
                ),
                "surrogate_is_acquisition_only_and_never_a_release_gate": (
                    surrogate_report_path is not None
                    and surrogate_report_path.is_file()
                    and read_json(surrogate_report_path).get(
                        "surrogate_release_authority"
                    )
                    == "NONE"
                ),
            }
        )
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    artifacts: Dict[str, Any] = {
        "baseline_cyclic_plausibility": artifact(baseline_base_path),
        "legacy_strict_hit_reaudit": artifact(legacy_evidence_path),
        "plausibility_and_position_evidence": artifact(evidence_path),
        "directed_candidates": artifact(candidates_path),
        "trace": artifact(trace_path),
        "controls": artifact(controls_path),
    }
    if screening_paths:
        artifacts["conditional_methyl_screens"] = {
            path.name: artifact(path) for path in screening_paths
        }
        artifacts["conditional_cyclic_base_shortlists"] = {
            path.name: artifact(path) for path in shortlist_paths
        }
        artifacts["resume_state"] = artifact(out_dir / "v2_resume_state.json")
    if frontier_v3 is not None:
        artifacts.update(
            {
                "pre_v3_full_frontier_selection": artifact(
                    out_dir / "pre_v3_full_frontier_selection.csv.gz"
                ),
                "pre_v3_full_frontier_exact_cyclic_base": artifact(
                    legacy_bridge_path
                ),
                "frontier_state": artifact(out_dir / "v3_frontier_state.json"),
                "surrogate_audit": artifact(surrogate_report_path),
                "prior_v2_failure_manifest": artifact(
                    prior_v2_manifest_copy
                ),
                "prior_v2_failure_search_trace": artifact(prior_v2_trace_copy),
            }
        )
    manifest = {
        "quality_gate": quality_gate,
        "release_status": (
            (
                "READY_FOR_INDEPENDENT_V3_FINAL_AUDIT_NO_STRUCTURE_HANDOFF"
                if frontier_v3 is not None
                else "READY_FOR_INDEPENDENT_V2_FINAL_AUDIT_NO_STRUCTURE_HANDOFF"
            )
            if quality_gate == "PASS"
            else (
                "BLOCKED_FIXED_V3_FULL_FRONTIER_BUDGET_DID_NOT_RECOVER_3ZGC"
                if frontier_v3 is not None
                else "BLOCKED_FIXED_V2_BUDGET_DID_NOT_RECOVER_3ZGC"
            )
        ),
        "protocol": protocol,
        "config": config,
        "config_sha256": config_digest,
        "model_sha256": sha256_file(model_path),
        "baseline_manifest_sha256": sha256_file(
            baseline / "generation_manifest.json"
        ),
        "legacy_manifest_sha256": sha256_file(
            legacy_dir / "directed_search_manifest.json"
        ),
        "legacy_evaluated_sequences": len(legacy_seen),
        "legacy_strict_hits_reaudited": len(legacy_sequences),
        "legacy_full_frontier_rows": (
            len(legacy_all_rows) if frontier_v3 is not None else 0
        ),
        "legacy_rows_already_exact_in_baseline": (
            len(legacy_baseline_overlap) if frontier_v3 is not None else 0
        ),
        "prior_v2_methyl_screen_rows_reused": len(prior_v2_screen_rows),
        "legacy_non_strict_bridge_rows_exactly_scored": len(legacy_bridge_rows),
        "cyclic_base_floor_1pct": floor,
        "conditional_search_ran": conditional_search_ran,
        "conditional_rounds_completed": len(trace_rows),
        "released_candidates": len(release_rows),
        "released_candidate_counts": dict(
            sorted(Counter(str(row["target_name"]) for row in release_rows).items())
        ),
        "missing_targets_before_search": [target],
        "missing_targets_after_search": [] if release_rows else [target],
        "targets_formally_abstained": [],
        "structure_handoff_status": "NOT_CREATED_PENDING_MANUAL_REVIEW",
        "permeability_status": "DEFERRED_UNTIL_STRUCTURE_GATES_PASS",
        "quality_checks": quality_checks,
        "artifacts": artifacts,
        "legacy_manifest_summary": {
            "quality_gate": legacy_manifest["quality_gate"],
            "strict_probability_hit_counts": legacy_manifest[
                "strict_probability_hit_counts"
            ],
            "evaluated_sequence_counts": legacy_manifest[
                "evaluated_sequence_counts"
            ],
        },
        "serine_provenance_gate": {
            "quality_gate": "PASS",
            "literal_normalization": "lowercase_design_s_naturalizes_to_uppercase_S",
            "expert_source_by_residue": expert_source_by_residue,
            "shared_tensor_source": "canonical",
            "serine_expert_source": "v7_serine",
            "non_ser_expert_source": "v6_non_ser",
            "canonical_checkpoint_sha256": model_manifest_payload[
                "canonical_checkpoint_sha256"
            ],
            "v6_checkpoint_sha256": model_manifest_payload["v6_checkpoint_sha256"],
            "v7_checkpoint_sha256": model_manifest_payload["v7_checkpoint_sha256"],
            "bitwise_quality_checks": {
                name: source_quality_checks[name]
                for name in (
                    "all_shared_tensors_are_canonical_bitwise_identical",
                    "all_non_ser_experts_are_v6_bitwise_identical",
                    "serine_expert_is_v7_bitwise_identical",
                    "every_ser_probability_is_inherited_from_v7",
                )
            },
        },
    }
    atomic_write_json(manifest_path, manifest)
    version_label = "V3 FULL FRONTIER" if frontier_v3 is not None else "V2"
    print(
        f"===== V8 CYCLIC-START BASE RECOVERY {version_label} COMPLETE =====",
        flush=True,
    )
    print(f"Quality gate: {quality_gate}", flush=True)
    print(f"Legacy strict hits re-audited: {len(legacy_sequences):,}", flush=True)
    print(f"Corrected cyclic-base floor: {floor:.8f}", flush=True)
    print(f"Conditional joint search ran: {conditional_search_ran}", flush=True)
    print(f"Released 3ZGC candidates: {len(release_rows)}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError(
            f"V8 {version_label} recovery failed honestly: " + ", ".join(failed)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-manifest", default=str(DEFAULT_MODEL_MANIFEST))
    parser.add_argument("--representation-audit", default=str(DEFAULT_REPRESENTATION))
    parser.add_argument("--baseline-run-dir", default=str(DEFAULT_BASELINE))
    parser.add_argument("--legacy-search-dir", default=str(DEFAULT_LEGACY_SEARCH))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--historical-designs-csv", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--prior-handoff-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--prior-v2-dir", default=str(DEFAULT_PRIOR_V2))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-batch-size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=EXPECTED_LEGACY_ROUNDS)
    parser.add_argument("--beam-width", type=int, default=EXPECTED_BEAM_WIDTH)
    parser.add_argument("--offspring-per-round", type=int, default=EXPECTED_OFFSPRING)
    parser.add_argument("--shortlist-per-round", type=int, default=EXPECTED_SHORTLIST)
    parser.add_argument("--max-release-per-target", type=int, default=EXPECTED_RELEASE_LIMIT)
    parser.add_argument(
        "--legacy-bridge-size", type=int, default=EXPECTED_V3_LEGACY_BRIDGE
    )
    parser.add_argument("--frontier-v3", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = {
        "--batch-size": (args.batch_size, 64),
        "--base-batch-size": (args.base_batch_size, 32),
        "--rounds": (args.rounds, EXPECTED_LEGACY_ROUNDS),
        "--beam-width": (args.beam_width, EXPECTED_BEAM_WIDTH),
        "--offspring-per-round": (args.offspring_per_round, EXPECTED_OFFSPRING),
        "--shortlist-per-round": (args.shortlist_per_round, EXPECTED_SHORTLIST),
        "--max-release-per-target": (
            args.max_release_per_target,
            EXPECTED_RELEASE_LIMIT,
        ),
    }
    if args.frontier_v3:
        frozen["--legacy-bridge-size"] = (
            args.legacy_bridge_size,
            EXPECTED_V3_LEGACY_BRIDGE,
        )
    changed = [name for name, (observed, expected) in frozen.items() if observed != expected]
    if changed:
        raise ValueError("V8 V2 numerical protocol is frozen: " + ", ".join(changed))
    run(args)


if __name__ == "__main__":
    main()
