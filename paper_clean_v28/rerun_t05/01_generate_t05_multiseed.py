#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the fixed T=0.5 multiseed rerun.

The sampler preserves the historical V28 base/expert decision rule from
``DCarchimonde/ProteinMPNN:nmethyl/generate_100_seqs_robust.py`` while fixing
both known train/inference mismatches:

1. randomly permute the designed peptide positions for every draw;
2. sample the natural amino acid from the base head at the fixed temperature;
3. pass that exact random order into the model's causal decoder mask;
4. after the complete natural sequence has been sampled, remove the visible
   receptor and annotate every peptide site from a deterministic cyclic-order
   ensemble at the same temperature, matching the expert-head training domain;
5. use the all-start ensemble mean only for ranking, and emit a lowercase
   N-methyl token only when the worst cyclic-start probability, rounded to the
   persisted eight-decimal contract, is strictly greater than the frozen
   threshold;
6. block release of any sequence with a physical position whose equivalent
   cyclic starts disagree on the threshold hard call.

The emitted lowercase annotation is stored separately from the autoregressive
model context.  Only the sampled natural parent residue is fed into subsequent
decoder steps.  A complete natural sequence therefore has one deterministic
annotation regardless of which random path generated it.

This version adds reproducible seeds, strict clean-V28 checkpoint loading,
target-wise sampling budgets, complete provenance, exact old-pool exclusion,
and an optional permeability-model input manifest.  ``--defer-permeability-``
``until-structure`` enforces the structure-first collaboration workflow by
omitting every permeability input until the structure gate has been run.
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
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan.json")
DEFAULT_MODEL = REPO_ROOT / "frankenstein_v28.pt"
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_BEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "best_designs.csv"
)
DEFAULT_OLD = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "all_designs.csv"
)
EXPECTED_PRIOR_HANDOFF_ROWS = 1_333
REQUIRED_ORDER_BALANCED_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_order_balanced_v3"
)
REQUIRED_CYCLIC_REPRESENTATION_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_"
    "cyclic_stability_worst_start_v9"
)
REQUIRED_CYCLIC_REPRESENTATION_TRAINING_POLICY = (
    "all_physical_cyclic_starts_jointly_rotate_sequence_labels_and_"
    "backbone_coordinates_with_residue_index_reset"
)
REQUIRED_CYCLIC_REPRESENTATION_ORDER_POLICY = (
    "complete_physical_cyclic_start_x_complete_L_decoder_order_grid_"
    "differentiably_meaned_per_start_then_mapped_to_physical_labels"
)
REQUIRED_CYCLIC_REPRESENTATION_DEPLOYMENT_POLICY = (
    "all_cyclic_starts_and_all_decoder_orders_mapped_to_physical_"
    "residues_probability_mean_for_ranking_representation_min_for_release"
)
PEPTIDE_ONLY_ANNOTATION_MODE = (
    "peptide_only_cyclic_order_ensemble_known_natural_sequence"
)
CYCLIC_REPRESENTATION_ANNOTATION_MODE = (
    "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
)
PEPTIDE_ONLY_ANNOTATION_CONTEXT = (
    "peptide_chain_only_no_visible_receptor_chains"
)
REPRESENTATION_AUDIT_PROTOCOL = "cyclic_stability_worst_start_heldout_gate_v9"
REPRESENTATION_AUDIT_AUTHORIZATION = (
    "CYCLIC_STABILITY_V9_VALIDATED_FOR_UNIFORM_REGENERATION"
)
SAMPLING_CONTEXT_POLICY = "native_complex_longest_receptor_visible"
DEFAULT_OUT = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "rerun_temperature_0.5_multiseed"
)

NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
VALID_DESIGN_TOKENS = set(NATURAL_AA + NATURAL_AA.lower())
KNOWN_OUTPUTS = (
    "all_candidates.csv",
    "unique_candidates.csv",
    "methylated_new_candidates.csv",
    "permeability_input.csv",
    "permeability_input_manifest.csv",
    "target_manifest.csv",
    "generation_summary_by_target.csv",
    "generation_manifest.json",
)
PERMEABILITY_OUTPUTS = (
    "permeability_input.csv",
    "permeability_input_manifest.csv",
)


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
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_target_offset(target_name: str) -> int:
    """Return a stable small integer so targets remain seed-independent."""
    digest = hashlib.sha256(target_name.upper().encode("ascii")).hexdigest()
    return int(digest[:8], 16) % 100_000


def chain_ids_from_record(record: Mapping[str, Any]) -> List[str]:
    ordered: List[str] = []
    for chain_id in list(record.get("masked_list", [])) + list(record.get("visible_list", [])):
        if f"seq_chain_{chain_id}" in record and chain_id not in ordered:
            ordered.append(str(chain_id))
    for key in record:
        if key.startswith("seq_chain_"):
            chain_id = key[len("seq_chain_") :]
            if chain_id not in ordered:
                ordered.append(chain_id)
    return ordered


def record_name(record: Mapping[str, Any], fallback_index: int = 0) -> str:
    return str(
        record.get("name")
        or record.get("pdb")
        or record.get("pdb_id")
        or record.get("id")
        or f"sample_{fallback_index}"
    ).upper()


def naturalize(sequence: str) -> str:
    return "".join(ch.upper() for ch in str(sequence))


def methyl_positions_1based(sequence: str) -> List[int]:
    return [index for index, token in enumerate(str(sequence), start=1) if token.islower()]


def strict_rounded_probability_pass(value: float, threshold: float = 0.6) -> bool:
    """Apply the persisted eight-decimal strict-threshold contract."""

    numeric = float(value)
    return (
        math.isfinite(numeric)
        and 0.0 <= numeric <= 1.0
        and round(numeric, 8) > float(threshold)
    )


def stable_cyclic_release_gate(row: Mapping[str, Any]) -> bool:
    """Fail closed unless the saved lowercase pattern is all-start stable.

    The ensemble mean remains a ranking statistic.  Release requires the
    representation minimum to pass at every lowercase site and requires zero
    threshold-straddling physical positions anywhere in the sequence.
    """

    sequence = str(row.get("design_seq", ""))
    if not sequence:
        return False
    try:
        threshold = float(row.get("methyl_threshold", 0.6))
        means = [float(value) for value in json.loads(str(row["methyl_probabilities"]))]
        minima = [
            float(value)
            for value in json.loads(
                str(row["methyl_probability_representation_min"])
            )
        ]
        maxima = [
            float(value)
            for value in json.loads(
                str(row["methyl_probability_representation_max"])
            )
        ]
        spans = [
            float(value)
            for value in json.loads(
                str(row["methyl_probability_representation_span"])
            )
        ]
        standard_deviations = [
            float(value)
            for value in json.loads(
                str(row["methyl_probability_representation_std"])
            )
        ]
        probability_by_start = [
            [float(value) for value in values]
            for values in json.loads(
                str(row["methyl_probability_representation_by_start"])
            )
        ]
        representation_size = int(row["annotation_representation_ensemble_size"])
        decoder_size = int(row["annotation_decoder_order_ensemble_size"])
        total_size = int(row["annotation_total_probability_ensemble_size"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not (
        len(means)
        == len(minima)
        == len(maxima)
        == len(spans)
        == len(standard_deviations)
        == len(sequence)
        and all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for values in (means, minima, maxima, spans, standard_deviations)
            for value in values
        )
        and all(
            minimum <= mean + 1e-7 and mean <= maximum + 1e-7
            for mean, minimum, maximum in zip(means, minima, maxima)
        )
        and all(
            abs(span - (maximum - minimum)) <= 1e-6
            for span, minimum, maximum in zip(spans, minima, maxima)
        )
        and representation_size > 0
        and decoder_size == len(sequence)
        and total_size == representation_size * decoder_size
        and (
            str(row.get("annotation_mode", ""))
            != CYCLIC_REPRESENTATION_ANNOTATION_MODE
            or representation_size == len(sequence)
        )
        and len(probability_by_start) == representation_size
        and all(len(values) == len(sequence) for values in probability_by_start)
        and all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for values in probability_by_start
            for value in values
        )
        and all(
            round(mean, 8) == round(sum(values) / len(values), 8)
            and round(minimum, 8) == round(min(values), 8)
            and round(maximum, 8) == round(max(values), 8)
            for mean, minimum, maximum, values in zip(
                means,
                minima,
                maxima,
                zip(*probability_by_start),
            )
        )
    ):
        return False
    stable_positions = [
        index
        for index, (token, minimum) in enumerate(zip(sequence, minima), start=1)
        if token.upper() != "P" and strict_rounded_probability_pass(minimum, threshold)
    ]
    disagreement_positions = [
        index
        for index, (minimum, maximum) in enumerate(zip(minima, maxima), start=1)
        if not strict_rounded_probability_pass(minimum, threshold)
        and strict_rounded_probability_pass(maximum, threshold)
    ]
    return bool(stable_positions) and not disagreement_positions and (
        methyl_positions_1based(sequence) == stable_positions
    )


def sequence_recovery(native: str, design: str) -> float:
    native_natural = naturalize(native)
    design_natural = naturalize(design)
    if not native_natural or len(native_natural) != len(design_natural):
        return math.nan
    return sum(a == b for a, b in zip(native_natural, design_natural)) / len(native_natural)


def validate_plan(plan: Mapping[str, Any], seeds_override: Sequence[int] | None = None) -> Dict[str, Any]:
    required = {"temperature", "methyl_threshold", "seeds", "targets", "frozen_targets"}
    missing = required - set(plan)
    if missing:
        raise ValueError(f"Plan is missing fields: {sorted(missing)}")
    if float(plan["temperature"]) != 0.5:
        raise ValueError("This protocol is frozen to temperature 0.5")
    if float(plan["methyl_threshold"]) != 0.6:
        raise ValueError("This protocol is frozen to methyl threshold 0.6")

    seeds = [int(value) for value in (seeds_override or plan["seeds"])]
    if not seeds or len(seeds) != len(set(seeds)) or any(seed <= 0 for seed in seeds):
        raise ValueError("Seeds must be unique positive integers")

    targets = list(plan["targets"])
    names = [str(item["target_name"]).upper() for item in targets]
    expected_target_count = int(plan.get("expected_target_count", 13))
    if expected_target_count <= 0:
        raise ValueError("expected_target_count must be positive")
    if len(targets) != expected_target_count or len(names) != len(set(names)):
        raise ValueError(
            "The rerun plan must contain exactly "
            f"{expected_target_count} unique targets"
        )
    frozen = {str(value).upper() for value in plan["frozen_targets"]}
    if set(names) & frozen:
        raise ValueError("A frozen target is also present in the rerun list")

    for item in targets:
        if int(item["sequences_per_seed"]) <= 0 or int(item["structure_quota"]) <= 0:
            raise ValueError(f"Invalid target budget: {item}")

    expected_raw = len(seeds) * sum(int(item["sequences_per_seed"]) for item in targets)
    expected_handoff = sum(int(item["structure_quota"]) for item in targets)
    if str(plan.get("protocol", "")).startswith(
        "temperature_0.5_cyclic_stability_worst_start_v9_"
    ):
        v9_contract = {
            "sampling_context_policy": SAMPLING_CONTEXT_POLICY,
            "annotation_context_policy": PEPTIDE_ONLY_ANNOTATION_CONTEXT,
            "annotation_ranking_probability_policy": "representation_mean",
            "annotation_release_probability_policy": (
                "representation_min_strict_gt_threshold_zero_disagreement"
            ),
        }
        mismatched = [
            field
            for field, expected in v9_contract.items()
            if str(plan.get(field, "")) != expected
        ]
        if mismatched:
            raise ValueError(
                "V9 plan policy mismatch: " + ", ".join(sorted(mismatched))
            )
        if (
            expected_target_count != 17
            or frozen
            or int(plan.get("final_release_quota_per_target", -1)) != 100
            or int(plan.get("initial_stable_pool_quota_per_target", -1)) <= 100
            or any(
                int(item["structure_quota"])
                != int(plan["initial_stable_pool_quota_per_target"])
                for item in targets
            )
        ):
            raise ValueError(
                "V9 plan requires 17 regenerated targets, no frozen targets, "
                "final quota 100, and a uniform stable pool quota above 100"
            )
    return {
        "seeds": seeds,
        "targets": targets,
        "target_names": names,
        "expected_target_count": expected_target_count,
        "expected_raw_candidates": expected_raw,
        "planned_structure_handoff": expected_handoff,
    }


def selected_chain_index(best_rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
    by_target: MutableMapping[str, set[str]] = defaultdict(set)
    for row in best_rows:
        target = str(row.get("target_name", "")).strip().upper()
        chains = [value.strip() for value in str(row.get("selected_chains", "")).split(",") if value.strip()]
        if target and len(chains) == 1:
            by_target[target].add(chains[0])
    result: Dict[str, str] = {}
    for target, chains in by_target.items():
        if len(chains) != 1:
            raise ValueError(f"Selected peptide chain is inconsistent for {target}: {sorted(chains)}")
        result[target] = next(iter(chains))
    return result


def prepare_target_records(
    native_rows: Sequence[Mapping[str, Any]],
    selected_chains: Mapping[str, str],
    target_names: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    native_index = {record_name(row, index): dict(row) for index, row in enumerate(native_rows)}
    prepared: Dict[str, Dict[str, Any]] = {}
    manifest: List[Dict[str, Any]] = []

    for target in target_names:
        if target not in native_index:
            raise ValueError(f"Target {target} is absent from the native JSONL")
        if target not in selected_chains:
            raise ValueError(f"Target {target} has no unique auto_single peptide chain")

        source = native_index[target]
        chain_ids = chain_ids_from_record(source)
        peptide_chain = selected_chains[target]
        if peptide_chain not in chain_ids:
            raise ValueError(f"Peptide chain {peptide_chain} is absent for {target}")

        receptor_candidates = [chain for chain in chain_ids if chain != peptide_chain]
        if not receptor_candidates:
            raise ValueError(f"No receptor chain is available for {target}")
        # Historical inference_complexes.jsonl retained only the longest receptor
        # chain. Keep that exact generation context while separately preserving
        # all remaining chains for the later structure-prediction manifest.
        generation_receptor = max(
            receptor_candidates,
            key=lambda chain: len(str(source.get(f"seq_chain_{chain}", ""))),
        )

        target_record = copy.deepcopy(source)
        target_record["masked_list"] = [peptide_chain]
        target_record["visible_list"] = [generation_receptor]
        prepared[target] = target_record

        receptor_sequences = {
            chain: str(source.get(f"seq_chain_{chain}", ""))
            for chain in receptor_candidates
        }
        peptide_sequence = str(source.get(f"seq_chain_{peptide_chain}", ""))
        manifest.append(
            {
                "target_name": target,
                "selected_chain": peptide_chain,
                "generation_receptor_chain": generation_receptor,
                "structure_receptor_chains": ",".join(receptor_candidates),
                "all_chain_ids": ",".join(chain_ids),
                "native_peptide_seq": peptide_sequence,
                "native_peptide_natural_seq": naturalize(peptide_sequence),
                "native_peptide_length": len(peptide_sequence),
                "generation_receptor_length": len(
                    str(source.get(f"seq_chain_{generation_receptor}", ""))
                ),
                "receptor_sequences_json": json.dumps(
                    receptor_sequences, ensure_ascii=False, sort_keys=True
                ),
            }
        )

    return prepared, manifest


def old_design_keys(
    path: Path,
) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]]]:
    exact_keys: set[Tuple[str, str]] = set()
    natural_keys: set[Tuple[str, str]] = set()
    for row in read_csv(path):
        target = str(row.get("target_name", "")).strip().upper()
        sequence = str(row.get("design_seq", "")).strip()
        if target and sequence:
            exact_keys.add((target, sequence))
            natural_keys.add((target, naturalize(sequence)))
    return exact_keys, natural_keys


def validate_prior_handoff(
    path: Path,
) -> Tuple[List[Dict[str, str]], set[Tuple[str, str]], set[Tuple[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    if len(rows) != EXPECTED_PRIOR_HANDOFF_ROWS:
        raise RuntimeError(
            "Prior handoff row count changed: expected "
            f"{EXPECTED_PRIOR_HANDOFF_ROWS}, observed {len(rows)}"
        )
    for row_number, row in enumerate(rows, start=2):
        if not str(row.get("target_name", "")).strip() or not str(
            row.get("design_seq", "")
        ).strip():
            raise RuntimeError(
                f"Prior handoff has an empty target_name or design_seq at CSV row {row_number}"
            )
    exact_keys, natural_keys = old_design_keys(path)
    return rows, exact_keys, natural_keys


def ensure_output_scope(out_dir: Path, overwrite: bool) -> None:
    existing = [out_dir / name for name in KNOWN_OUTPUTS if (out_dir / name).exists()]
    if existing and not overwrite:
        shown = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "The output already contains a completed/partial run. Use --overwrite "
            "only when intentionally regenerating the same isolated run:\n" + shown
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def torch_seed_all(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def repeat_batch(tensor: Any, batch_size: int) -> Any:
    repeats = [batch_size] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*repeats)


def generate_batch(
    model: Any,
    features: Sequence[Any],
    batch_size: int,
    temperature: float,
    methyl_threshold: float,
    torch_module: Any,
    functional: Any,
    extended_alphabet: str,
    x_index: int,
    natural_to_methyl: Mapping[int, int],
    complete_order_fn: Any,
    ensemble_probability_fn: Any,
    peptide_only_tensors_fn: Any,
) -> List[Dict[str, Any]]:
    X, S_true, mask, chain_M, residue_idx, chain_encoding_all = features[:6]
    X = repeat_batch(X, batch_size)
    S_context = repeat_batch(S_true, batch_size).clone()
    mask = repeat_batch(mask, batch_size)
    chain_M = repeat_batch(chain_M, batch_size)
    residue_idx = repeat_batch(residue_idx, batch_size)
    chain_encoding_all = repeat_batch(chain_encoding_all, batch_size)

    masked_positions = torch_module.nonzero(chain_M[0].eq(1.0), as_tuple=False).squeeze(-1)
    if masked_positions.ndim == 0:
        masked_positions = masked_positions.unsqueeze(0)
    if masked_positions.numel() == 0:
        raise ValueError("The prepared record has no designed peptide positions")

    # The model trunk never receives a methyl token. Naturalize any unusual
    # receptor-side annotation too, then keep emitted annotations in a separate
    # tensor used only for output serialization.
    for natural_index, methyl_index in natural_to_methyl.items():
        S_context[S_context == int(methyl_index)] = int(natural_index)
    S_context[:, masked_positions] = x_index
    orders = torch_module.stack(
        [masked_positions[torch_module.randperm(masked_positions.numel(), device=X.device)] for _ in range(batch_size)],
        dim=0,
    )
    row_indices = torch_module.arange(batch_size, device=X.device)
    position_to_relative = {
        int(position.item()): relative for relative, position in enumerate(masked_positions)
    }
    peptide_length = int(masked_positions.numel())
    base_log_prob = torch_module.zeros((batch_size, peptide_length), device=X.device)
    sampled_log_prob = torch_module.zeros((batch_size, peptide_length), device=X.device)
    sampling_path_methyl_probability = torch_module.zeros(
        (batch_size, peptide_length), device=X.device
    )
    full_orders = complete_order_fn(chain_M, mask, orders)

    # clean_v28_common still uses torch.utils.checkpoint in forward. no_grad is
    # compatible with that implementation and matches the historical sampler;
    # inference_mode can reject tensors saved internally by checkpoint.
    with torch_module.no_grad():
        for step in range(peptide_length):
            positions = orders[:, step]
            logits_base, logits_experts = model(
                X,
                S_context,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
                decoding_order=full_orders,
            )
            current_logits = logits_base[row_indices, positions]
            scaled_log_probs = functional.log_softmax(current_logits / temperature, dim=-1)
            sampled_base = torch_module.multinomial(
                scaled_log_probs.exp(), num_samples=1
            ).squeeze(-1)
            unscaled_log_probs = functional.log_softmax(current_logits, dim=-1)
            expert_logits = logits_experts[row_indices, positions, sampled_base]
            current_methyl_probability = torch_module.sigmoid(
                expert_logits / temperature
            )

            S_context[row_indices, positions] = sampled_base

            relative_positions = torch_module.tensor(
                [position_to_relative[int(value)] for value in positions.detach().cpu().tolist()],
                device=X.device,
                dtype=torch_module.long,
            )
            base_log_prob[row_indices, relative_positions] = unscaled_log_probs.gather(
                1, sampled_base.unsqueeze(-1)
            ).squeeze(-1)
            sampled_log_prob[row_indices, relative_positions] = scaled_log_probs.gather(
                1, sampled_base.unsqueeze(-1)
            ).squeeze(-1)
            sampling_path_methyl_probability[
                row_indices, relative_positions
            ] = current_methyl_probability

        (
            annotation_X,
            annotation_S,
            annotation_mask,
            annotation_chain_M,
            annotation_residue_idx,
            annotation_chain_encoding,
        ) = peptide_only_tensors_fn(X, S_context, mask, chain_M)
        ensemble_result = ensemble_probability_fn(
            model=model,
            X=annotation_X,
            S_natural=annotation_S,
            mask=annotation_mask,
            chain_M=annotation_chain_M,
            residue_idx=annotation_residue_idx,
            chain_encoding_all=annotation_chain_encoding,
            temperature=temperature,
        )
        if isinstance(ensemble_result, Mapping):
            final_methyl_probability = ensemble_result["mean"]
            order_probability_std = ensemble_result["decoder_order_std_mean"]
            representation_probability_std = ensemble_result["representation_std"]
            representation_probability_min = ensemble_result["representation_min"]
            representation_probability_max = ensemble_result["representation_max"]
            representation_probability_span = ensemble_result["representation_span"]
            representation_count = ensemble_result["representation_count"]
            representation_probability_by_start = ensemble_result[
                "representation_probability_by_start"
            ]
            annotation_mode = CYCLIC_REPRESENTATION_ANNOTATION_MODE
        else:
            final_methyl_probability, order_probability_std = ensemble_result
            representation_probability_std = torch_module.zeros_like(
                final_methyl_probability
            )
            representation_probability_min = final_methyl_probability
            representation_probability_max = final_methyl_probability
            representation_probability_span = torch_module.zeros_like(
                final_methyl_probability
            )
            representation_count = torch_module.ones_like(final_methyl_probability)
            representation_probability_by_start = final_methyl_probability.unsqueeze(1)
            annotation_mode = PEPTIDE_ONLY_ANNOTATION_MODE

    S_output = S_context.clone()
    final_natural_tokens = S_context[:, masked_positions]
    final_output_tokens = final_natural_tokens.clone()
    # Compare the exact same eight-decimal value that is serialized to CSV.
    # Keeping the rounding in float32 can turn the nearest float32 value above
    # 0.6 into a different decision than Python's persisted-value audit.
    persisted_release_probability = (
        torch_module.round(representation_probability_min.double() * 1.0e8)
        / 1.0e8
    )
    for natural_index, methyl_index in natural_to_methyl.items():
        use_methyl = final_natural_tokens.eq(int(natural_index)) & (
            persisted_release_probability.gt(methyl_threshold)
        )
        final_output_tokens = torch_module.where(
            use_methyl,
            torch_module.full_like(final_output_tokens, int(methyl_index)),
            final_output_tokens,
        )
    S_output[:, masked_positions] = final_output_tokens

    results: List[Dict[str, Any]] = []
    peptide_tokens = S_output[:, masked_positions].detach().cpu().tolist()
    base_lp = base_log_prob.detach().cpu().tolist()
    sampled_lp = sampled_log_prob.detach().cpu().tolist()
    methyl_p = final_methyl_probability.detach().cpu().tolist()
    methyl_order_std = order_probability_std.detach().cpu().tolist()
    methyl_representation_std = representation_probability_std.detach().cpu().tolist()
    methyl_representation_min = representation_probability_min.detach().cpu().tolist()
    methyl_representation_max = representation_probability_max.detach().cpu().tolist()
    methyl_representation_span = representation_probability_span.detach().cpu().tolist()
    representation_counts = representation_count.detach().cpu().tolist()
    representation_probability_by_start_cpu = (
        representation_probability_by_start.detach().cpu().tolist()
    )
    sampling_path_p = sampling_path_methyl_probability.detach().cpu().tolist()
    orders_cpu = orders.detach().cpu().tolist()
    for index in range(batch_size):
        sequence = "".join(extended_alphabet[int(token)] for token in peptide_tokens[index])
        representation_disagreement_positions = [
            position + 1
            for position, (minimum, maximum) in enumerate(
                zip(
                    methyl_representation_min[index],
                    methyl_representation_max[index],
                )
            )
            if not strict_rounded_probability_pass(float(minimum), methyl_threshold)
            and strict_rounded_probability_pass(float(maximum), methyl_threshold)
        ]
        methyl_site_probabilities = [
            float(methyl_p[index][position])
            for position, token in enumerate(sequence)
            if token.islower()
        ]
        methyl_site_representation_floors = [
            float(methyl_representation_min[index][position])
            for position, token in enumerate(sequence)
            if token.islower()
        ]
        payload = {
                "design_seq": sequence,
                "base_log_probability_sum": float(sum(base_lp[index])),
                "base_log_probability_mean": float(sum(base_lp[index]) / peptide_length),
                "sampling_log_probability_sum": float(sum(sampled_lp[index])),
                "sampling_log_probability_mean": float(sum(sampled_lp[index]) / peptide_length),
                "methyl_probability_min": float(min(methyl_p[index])),
                "methyl_probability_mean": float(sum(methyl_p[index]) / peptide_length),
                "methyl_probability_max": float(max(methyl_p[index])),
                "methyl_site_probability_min": (
                    float(min(methyl_site_probabilities))
                    if methyl_site_probabilities
                    else ""
                ),
                "methyl_site_probability_mean": (
                    float(sum(methyl_site_probabilities) / len(methyl_site_probabilities))
                    if methyl_site_probabilities
                    else ""
                ),
                "methyl_site_probability_max": (
                    float(max(methyl_site_probabilities))
                    if methyl_site_probabilities
                    else ""
                ),
                "methyl_site_representation_floor_min": (
                    float(min(methyl_site_representation_floors))
                    if methyl_site_representation_floors
                    else ""
                ),
                "methyl_site_representation_floor_mean": (
                    float(
                        sum(methyl_site_representation_floors)
                        / len(methyl_site_representation_floors)
                    )
                    if methyl_site_representation_floors
                    else ""
                ),
                "methyl_site_representation_floor_max": (
                    float(max(methyl_site_representation_floors))
                    if methyl_site_representation_floors
                    else ""
                ),
                "methyl_probabilities": json.dumps(
                    [round(float(value), 8) for value in methyl_p[index]]
                ),
                "methyl_probability_order_std": json.dumps(
                    [round(float(value), 8) for value in methyl_order_std[index]]
                ),
                "methyl_probability_order_std_max": float(
                    max(methyl_order_std[index])
                ),
                "methyl_probability_representation_std": json.dumps(
                    [
                        round(float(value), 8)
                        for value in methyl_representation_std[index]
                    ]
                ),
                "methyl_probability_representation_std_max": float(
                    max(methyl_representation_std[index])
                ),
                "methyl_probability_representation_min": json.dumps(
                    [
                        round(float(value), 8)
                        for value in methyl_representation_min[index]
                    ]
                ),
                "methyl_probability_representation_max": json.dumps(
                    [
                        round(float(value), 8)
                        for value in methyl_representation_max[index]
                    ]
                ),
                "methyl_probability_representation_span": json.dumps(
                    [
                        round(float(value), 8)
                        for value in methyl_representation_span[index]
                    ]
                ),
                "methyl_probability_representation_by_start": json.dumps(
                    [
                        [round(float(value), 8) for value in start_values]
                        for start_values in representation_probability_by_start_cpu[index][
                            : int(round(max(representation_counts[index])))
                        ]
                    ]
                ),
                "methyl_probability_representation_span_max": float(
                    max(methyl_representation_span[index])
                ),
                "representation_threshold_disagreement_positions_1based": json.dumps(
                    representation_disagreement_positions
                ),
                "representation_threshold_disagreement_count": len(
                    representation_disagreement_positions
                ),
                "sampling_path_methyl_probabilities": json.dumps(
                    [round(float(value), 8) for value in sampling_path_p[index]]
                ),
                "annotation_mode": annotation_mode,
                "annotation_context_policy": PEPTIDE_ONLY_ANNOTATION_CONTEXT,
                "annotation_visible_receptor_chains": 0,
                "sampling_context_policy": SAMPLING_CONTEXT_POLICY,
                "annotation_order_ensemble_size": peptide_length,
                "annotation_decoder_order_ensemble_size": peptide_length,
                "annotation_representation_ensemble_size": int(
                    round(max(representation_counts[index]))
                ),
                "annotation_total_probability_ensemble_size": (
                    peptide_length
                    * int(round(max(representation_counts[index])))
                ),
                "annotation_ranking_probability_policy": "representation_mean",
                "annotation_release_probability_policy": (
                    "representation_min_strict_gt_threshold_zero_disagreement"
                ),
                "decoding_order_absolute": json.dumps([int(value) for value in orders_cpu[index]]),
            }
        payload["stable_cyclic_release_gate"] = int(
            stable_cyclic_release_gate({**payload, "methyl_threshold": methyl_threshold})
        )
        results.append(payload)
    return results


def write_seed_fasta(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    lines: List[str] = []
    for row in rows:
        sequence = str(row["design_seq"])
        if sequence in seen:
            continue
        seen.add(sequence)
        lines.append(
            ">"
            + str(row["candidate_id"])
            + " | T=0.5 | Thr=0.6"
            + f" | seed={row['seed']} | effective_seed={row['effective_seed']}"
        )
        lines.append(sequence)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(temp, path)


FINAL_ANNOTATION_FIELDS = (
    "design_seq",
    "methyl_probability_min",
    "methyl_probability_mean",
    "methyl_probability_max",
    "methyl_site_probability_min",
    "methyl_site_probability_mean",
    "methyl_site_probability_max",
    "methyl_site_representation_floor_min",
    "methyl_site_representation_floor_mean",
    "methyl_site_representation_floor_max",
    "methyl_probabilities",
    "methyl_probability_order_std",
    "methyl_probability_order_std_max",
    "methyl_probability_representation_std",
    "methyl_probability_representation_std_max",
    "methyl_probability_representation_min",
    "methyl_probability_representation_max",
    "methyl_probability_representation_span",
    "methyl_probability_representation_by_start",
    "methyl_probability_representation_span_max",
    "representation_threshold_disagreement_positions_1based",
    "representation_threshold_disagreement_count",
    "annotation_mode",
    "annotation_context_policy",
    "annotation_visible_receptor_chains",
    "sampling_context_policy",
    "annotation_order_ensemble_size",
    "annotation_decoder_order_ensemble_size",
    "annotation_representation_ensemble_size",
    "annotation_total_probability_ensemble_size",
    "annotation_ranking_probability_policy",
    "annotation_release_probability_policy",
    "stable_cyclic_release_gate",
)


def canonicalize_repeated_natural_annotations(
    rows: Sequence[MutableMapping[str, Any]],
    preferred_candidate_ids: set[str] | None = None,
) -> Dict[str, int]:
    """Give every repeated target/natural sequence one exact annotation payload.

    CUDA kernels can differ by a few last-place bits when an identical sequence
    is scored in batches with different neighbours.  That is not a biological
    difference, but it used to trip the repeated-probability gate.  Selecting one
    deterministic payload per target/natural sequence makes the persisted result
    exact and also guarantees that aggregation never treats numerical noise as a
    distinct compound.  Quota-resume callers may supply the immutable source-row
    IDs so a newly sampled duplicate can never rewrite previously audited evidence.
    """

    grouped: MutableMapping[Tuple[str, str], List[MutableMapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (str(row["target_name"]).upper(), str(row["design_natural_seq"]).upper())
        ].append(row)

    repeated_groups = 0
    rewritten_rows = 0
    preferred_candidate_ids = preferred_candidate_ids or set()
    for occurrences in grouped.values():
        representative = min(
            occurrences,
            key=lambda item: (
                0
                if str(item.get("candidate_id", "")) in preferred_candidate_ids
                else 1,
                str(item.get("candidate_id", "")),
            ),
        )
        payload = {
            field: representative[field]
            for field in FINAL_ANNOTATION_FIELDS
            if field in representative
        }
        if len(occurrences) > 1:
            repeated_groups += 1
        for row in occurrences:
            before = tuple(str(row.get(field, "")) for field in payload)
            for field, value in payload.items():
                row[field] = value
            sequence = str(row["design_seq"])
            positions = methyl_positions_1based(sequence)
            row["design_natural_seq"] = naturalize(sequence)
            row["design_methyl_count"] = len(positions)
            row["design_methyl_rate"] = len(positions) / len(sequence)
            row["methyl_positions_1based"] = json.dumps(positions)
            after = tuple(str(row.get(field, "")) for field in payload)
            if before != after:
                rewritten_rows += 1
    return {
        "unique_target_natural_sequence_groups": len(grouped),
        "repeated_target_natural_sequence_groups": repeated_groups,
        "rows_rewritten_to_canonical_payload": rewritten_rows,
    }


def aggregate_unique_candidates(
    rows: Sequence[Mapping[str, Any]],
    old_exact_keys: set[Tuple[str, str]],
    old_natural_keys: set[Tuple[str, str]] | None = None,
    prior_exact_keys: set[Tuple[str, str]] | None = None,
    prior_natural_keys: set[Tuple[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    old_natural_keys = old_natural_keys or {
        (target, naturalize(sequence)) for target, sequence in old_exact_keys
    }
    prior_exact_keys = prior_exact_keys or set()
    prior_natural_keys = prior_natural_keys or {
        (target, naturalize(sequence)) for target, sequence in prior_exact_keys
    }
    grouped: MutableMapping[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_name"]), str(row["design_seq"]))].append(row)

    result: List[Dict[str, Any]] = []
    for (target, sequence), occurrences in sorted(grouped.items()):
        representative = max(
            occurrences,
            key=lambda item: (
                float(item["base_log_probability_mean"]),
                -int(item["seed"]),
                -int(item["draw_index_within_seed"]),
            ),
        )
        row = dict(representative)
        row["occurrence_count"] = len(occurrences)
        row["seeds_observed"] = ";".join(
            str(value) for value in sorted({int(item["seed"]) for item in occurrences})
        )
        exact_seen = (target, sequence) in old_exact_keys
        natural_seen = (target, naturalize(sequence)) in old_natural_keys
        prior_exact_seen = (target, sequence) in prior_exact_keys
        prior_natural_seen = (target, naturalize(sequence)) in prior_natural_keys
        row["seen_in_historical_4115_exact"] = int(exact_seen)
        row["seen_in_historical_4115_naturalized"] = int(natural_seen)
        row["seen_in_historical_4115"] = int(exact_seen or natural_seen)
        row["seen_in_prior_1333_exact"] = int(prior_exact_seen)
        row["seen_in_prior_1333_naturalized"] = int(prior_natural_seen)
        row["seen_in_prior_1333"] = int(prior_exact_seen or prior_natural_seen)
        stable_release = stable_cyclic_release_gate(row)
        row["stable_cyclic_release_gate"] = int(stable_release)
        row["passes_methylation_hard_gate"] = int(
            int(row["design_methyl_count"]) > 0 and stable_release
        )
        row["eligible_for_new_permeability_screen"] = int(
            int(row["design_methyl_count"]) > 0
            and stable_release
            and not exact_seen
            and not natural_seen
            and not prior_exact_seen
            and not prior_natural_seen
        )
        result.append(row)
    return result


def build_permeability_rows(
    eligible: Sequence[Dict[str, Any]],
    all_native_manifest: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model_input: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []
    target_serial: Counter[str] = Counter()

    for source in sorted(eligible, key=lambda row: (str(row["target_name"]), str(row["design_seq"]))):
        row = dict(source)
        target = str(row["target_name"]).upper()
        serial = target_serial[target]
        target_serial[target] += 1
        permeability_id = f"{target.lower()}_{serial}_{row['design_seq']}_model"
        methyl_index = json.dumps(methyl_positions_1based(str(row["design_seq"])))
        row["permeability_id"] = permeability_id
        enriched.append(row)
        model_input.append(
            {
                "id": permeability_id,
                "fasta": row["design_natural_seq"],
                "methy_index": methyl_index,
            }
        )
        manifest.append(
            {
                "id": permeability_id,
                "record_type": "candidate",
                "target_name": target,
                "candidate_id": row["candidate_id"],
                "design_seq": row["design_seq"],
                "design_natural_seq": row["design_natural_seq"],
                "methy_index": methyl_index,
            }
        )

    for native_index, source in enumerate(sorted(all_native_manifest, key=lambda row: str(row["target_name"]))):
        target = str(source["target_name"]).upper()
        native_sequence = str(source["native_peptide_natural_seq"])
        permeability_id = f"{target.lower()}_{9_000_000 + native_index}_{native_sequence}_model"
        model_input.append({"id": permeability_id, "fasta": native_sequence, "methy_index": "[]"})
        manifest.append(
            {
                "id": permeability_id,
                "record_type": "native",
                "target_name": target,
                "candidate_id": "",
                "design_seq": native_sequence,
                "design_natural_seq": native_sequence,
                "methy_index": "[]",
            }
        )
    return model_input, manifest, enriched


def native_manifest_all_targets(
    native_rows: Sequence[Mapping[str, Any]], selected_chains: Mapping[str, str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, source in enumerate(native_rows):
        target = record_name(source, index)
        peptide_chain = selected_chains.get(target)
        if not peptide_chain:
            continue
        native_sequence = str(source.get(f"seq_chain_{peptide_chain}", ""))
        if not native_sequence:
            continue
        rows.append(
            {
                "target_name": target,
                "selected_chain": peptide_chain,
                "native_peptide_natural_seq": naturalize(native_sequence),
            }
        )
    if len(rows) != 17:
        raise ValueError(f"Expected 17 native permeability baselines, found {len(rows)}")
    return rows


def audit_annotation_stability(
    raw_rows: Sequence[Mapping[str, Any]],
    eligible_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Detect the two failure signatures that invalidated earlier reruns."""
    observed_annotation_modes = {
        str(row.get("annotation_mode", "")) for row in raw_rows
    }
    expected_annotation_mode = (
        next(iter(observed_annotation_modes))
        if len(observed_annotation_modes) == 1
        else ""
    )
    repeated: MutableMapping[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        repeated[
            (str(row["target_name"]), str(row["design_natural_seq"]))
        ].append(row)

    repeated_groups = [rows for rows in repeated.values() if len(rows) > 1]
    inconsistent_groups = []
    probability_disagreement_groups = []
    for rows in repeated_groups:
        annotations = {str(row["design_seq"]) for row in rows}
        if len(annotations) != 1:
            inconsistent_groups.append(rows)
        # The persisted V4 result uses one canonical payload for every repeated
        # target/natural sequence.  Exact JSON equality is therefore expected.
        reference_probability_json = str(rows[0].get("methyl_probabilities", ""))
        disagree = any(
            str(row.get("methyl_probabilities", "")) != reference_probability_json
            for row in rows[1:]
        )
        if disagree:
            probability_disagreement_groups.append(rows)

    site_positions: Counter[int] = Counter()
    site_residues: Counter[str] = Counter()
    target_site_positions: MutableMapping[str, Counter[int]] = defaultdict(Counter)
    target_site_residues: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    order_std_maxima: List[float] = []
    representation_std_maxima: List[float] = []
    representation_span_maxima: List[float] = []
    representation_disagreement_counts: List[int] = []
    unstable_release_candidate_ids: List[str] = []
    for row in eligible_rows:
        if not stable_cyclic_release_gate(row):
            unstable_release_candidate_ids.append(str(row.get("candidate_id", "")))
        target = str(row["target_name"])
        sequence = str(row["design_seq"])
        for position, token in enumerate(sequence, start=1):
            if token.islower():
                site_positions[position] += 1
                site_residues[token.upper()] += 1
                target_site_positions[target][position] += 1
                target_site_residues[target][token.upper()] += 1
        order_std_maxima.append(float(row["methyl_probability_order_std_max"]))
        representation_std_maxima.append(
            float(row.get("methyl_probability_representation_std_max", 0.0))
        )
        representation_span_maxima.append(
            float(row.get("methyl_probability_representation_span_max", 0.0))
        )
        representation_disagreement_counts.append(
            int(row.get("representation_threshold_disagreement_count", 0))
        )

    total_sites = int(sum(site_positions.values()))
    max_position_share = (
        max(site_positions.values()) / total_sites if total_sites else 0.0
    )
    max_residue_share = (
        max(site_residues.values()) / total_sites if total_sites else 0.0
    )
    per_target_concentration = []
    for target in sorted(set(target_site_positions) | set(target_site_residues)):
        target_total = int(sum(target_site_positions[target].values()))
        target_position_share = (
            max(target_site_positions[target].values()) / target_total
            if target_total
            else 0.0
        )
        target_residue_share = (
            max(target_site_residues[target].values()) / target_total
            if target_total
            else 0.0
        )
        per_target_concentration.append(
            {
                "target_name": target,
                "methyl_sites": target_total,
                "site_position_counts": dict(sorted(target_site_positions[target].items())),
                "site_residue_counts": dict(sorted(target_site_residues[target].items())),
                "maximum_single_position_share": target_position_share,
                "maximum_single_residue_share": target_residue_share,
                "concentration_gate_applies": target_total >= 30,
                "position_gate_pass": target_total < 30 or target_position_share <= 0.80,
                "residue_gate_pass": target_total < 30 or target_residue_share <= 0.80,
            }
        )

    # A target-local >80% physical-position concentration is a hard scientific
    # stop.  Structural homology may be reviewed as evidence, but it cannot
    # silently override this release gate.
    concentration_gate_applies = total_sites >= 100
    concentration_diagnostics = {
        "no_single_position_exceeds_80_percent_of_sites": (
            not concentration_gate_applies or max_position_share <= 0.80
        ),
        "no_single_residue_exceeds_80_percent_of_sites": (
            not concentration_gate_applies or max_residue_share <= 0.80
        ),
        "no_target_has_single_residue_above_80_percent_when_n_ge_30": all(
            bool(row["residue_gate_pass"]) for row in per_target_concentration
        ),
        "no_target_has_single_position_above_80_percent_when_n_ge_30": all(
            bool(row["position_gate_pass"]) for row in per_target_concentration
        ),
    }
    quality_checks = {
        "one_uniform_supported_peptide_only_annotation_mode_is_recorded": (
            expected_annotation_mode
            in {
                PEPTIDE_ONLY_ANNOTATION_MODE,
                CYCLIC_REPRESENTATION_ANNOTATION_MODE,
            }
        ),
        "peptide_only_annotation_context_recorded_for_every_raw_row": all(
            str(row.get("annotation_mode", ""))
            == expected_annotation_mode
            and str(row.get("annotation_context_policy", ""))
            == PEPTIDE_ONLY_ANNOTATION_CONTEXT
            and int(row.get("annotation_visible_receptor_chains", -1)) == 0
            for row in raw_rows
        ),
        "cyclic_representation_ensemble_size_matches_peptide_length_when_enabled": (
            expected_annotation_mode != CYCLIC_REPRESENTATION_ANNOTATION_MODE
            or all(
                int(row.get("annotation_representation_ensemble_size", -1))
                == int(row.get("design_length", -2))
                for row in raw_rows
            )
        ),
        "repeated_final_natural_sequences_have_identical_annotations": (
            len(inconsistent_groups) == 0
        ),
        "repeated_final_natural_sequences_have_matching_probabilities": (
            len(probability_disagreement_groups) == 0
        ),
        "every_eligible_candidate_is_stable_across_all_cyclic_starts": (
            len(unstable_release_candidate_ids) == 0
            and all(count == 0 for count in representation_disagreement_counts)
        ),
        "no_single_position_exceeds_80_percent_of_sites": (
            not concentration_gate_applies or max_position_share <= 0.80
        ),
        "no_target_has_single_position_above_80_percent_when_n_ge_30": all(
            bool(row["position_gate_pass"]) for row in per_target_concentration
        ),
        "no_single_residue_exceeds_80_percent_of_sites": (
            not concentration_gate_applies or max_residue_share <= 0.80
        ),
        "no_target_has_single_residue_above_80_percent_when_n_ge_30": all(
            bool(row["residue_gate_pass"]) for row in per_target_concentration
        ),
    }
    return {
        "quality_gate": "PASS" if all(quality_checks.values()) else "FAIL",
        "quality_checks": quality_checks,
        "raw_repeated_target_natural_sequence_groups": len(repeated_groups),
        "raw_inconsistent_annotation_groups": len(inconsistent_groups),
        "raw_probability_disagreement_groups": len(probability_disagreement_groups),
        "eligible_methyl_sites": total_sites,
        "eligible_site_position_counts": dict(sorted(site_positions.items())),
        "eligible_site_residue_counts": dict(sorted(site_residues.items())),
        "maximum_single_position_share": max_position_share,
        "maximum_single_residue_share": max_residue_share,
        "concentration_gate_applies": concentration_gate_applies,
        "concentration_diagnostics": concentration_diagnostics,
        "concentration_gate_policy": (
            "HARD_BLOCK_ABOVE_80_PERCENT_PENDING_INDEPENDENT_MANUAL_"
            "SCIENTIFIC_RELEASE; structural homology alone cannot override"
        ),
        "unstable_release_candidate_count": len(unstable_release_candidate_ids),
        "unstable_release_candidate_ids_first_100": unstable_release_candidate_ids[:100],
        "per_target_concentration": per_target_concentration,
        "maximum_candidate_order_probability_std": (
            max(order_std_maxima) if order_std_maxima else 0.0
        ),
        "mean_candidate_order_probability_std_max": (
            sum(order_std_maxima) / len(order_std_maxima)
            if order_std_maxima
            else 0.0
        ),
        "maximum_candidate_representation_probability_std": (
            max(representation_std_maxima) if representation_std_maxima else 0.0
        ),
        "mean_candidate_representation_probability_std_max": (
            sum(representation_std_maxima) / len(representation_std_maxima)
            if representation_std_maxima
            else 0.0
        ),
        "maximum_candidate_representation_probability_span": (
            max(representation_span_maxima) if representation_span_maxima else 0.0
        ),
        "eligible_candidates_with_pre_ensemble_threshold_disagreement": sum(
            count > 0 for count in representation_disagreement_counts
        ),
        "observed_annotation_mode": expected_annotation_mode,
    }


def run_generation(args: argparse.Namespace, plan: Dict[str, Any], validated: Dict[str, Any]) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "Generation requires numpy and torch in the active Python environment"
        ) from exc

    clean_dir = REPO_ROOT / "paper_clean_v28"
    if str(clean_dir) not in sys.path:
        sys.path.insert(0, str(clean_dir))
    from clean_v28_common import (  # pylint: disable=import-error,import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        EXTENDED_AA_TO_INDEX,
        NAT_TO_METHYL_ABS,
        complete_decoding_order,
        cyclic_known_sequence_methyl_probabilities,
        cyclic_representation_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
        peptide_only_annotation_tensors,
    )

    model_path = Path(args.model_path).resolve()
    native_path = Path(args.native_jsonl).resolve()
    best_path = Path(args.best_csv).resolve()
    old_path = Path(args.old_designs_csv).resolve()
    prior_path = (
        Path(args.prior_designs_csv).resolve() if args.prior_designs_csv else None
    )
    out_dir = Path(args.out_dir).resolve()
    for required in (model_path, native_path, best_path, old_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    protocol_name = str(plan.get("protocol", ""))
    is_v9_cyclic_stability_plan = protocol_name.startswith(
        "temperature_0.5_cyclic_stability_worst_start_v9_"
    )
    if is_v9_cyclic_stability_plan and not args.cyclic_representation_ensemble:
        raise RuntimeError(
            "A V9 cyclic-stability plan requires "
            "--cyclic-representation-ensemble; decoder-only annotation is forbidden"
        )
    requires_expert_qc = (
        "all_expert_qc" in str(plan.get("protocol", ""))
        or is_v9_cyclic_stability_plan
        or bool(args.cyclic_representation_ensemble)
    )
    if requires_expert_qc:
        if prior_path is None or not prior_path.is_file():
            raise FileNotFoundError(
                "This recovery protocol requires the prior 1,333-row handoff CSV "
                "for hard duplicate exclusion"
            )
    checkpoint_metadata: Dict[str, Any] = {}
    if requires_expert_qc:
        checkpoint_payload = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint_payload, Mapping):
            checkpoint_metadata = dict(
                checkpoint_payload.get("expert_head_qc_metadata", {})
            )
        observed_protocol = str(checkpoint_metadata.get("protocol", ""))
        if args.cyclic_representation_ensemble:
            metadata_is_complete = (
                observed_protocol
                == REQUIRED_CYCLIC_REPRESENTATION_EXPERT_PROTOCOL
                and int(checkpoint_metadata.get("minimum_order_coverage_epochs", 0))
                >= 30
                and bool(
                    checkpoint_metadata.get("cyclic_representation_augmentation")
                )
                and str(
                    checkpoint_metadata.get(
                        "training_cyclic_representation_policy", ""
                    )
                )
                == REQUIRED_CYCLIC_REPRESENTATION_TRAINING_POLICY
                and str(
                    checkpoint_metadata.get("training_decoding_order_policy", "")
                )
                == REQUIRED_CYCLIC_REPRESENTATION_ORDER_POLICY
                and str(
                    checkpoint_metadata.get("deployment_annotation_policy", "")
                )
                == REQUIRED_CYCLIC_REPRESENTATION_DEPLOYMENT_POLICY
                and float(
                    checkpoint_metadata.get("worst_start_bce_weight", 0.0)
                )
                > 0.0
                and float(
                    checkpoint_metadata.get(
                        "representation_consistency_weight", -1.0
                    )
                )
                > 0.0
                and bool(
                    checkpoint_metadata.get(
                        "full_physical_start_by_full_decoder_order_grid"
                    )
                )
                and float(
                    checkpoint_metadata.get("training_ensemble_temperature", -1.0)
                )
                == 0.5
                and "full_physical_start_x_full_decoder_order_grid"
                in str(checkpoint_metadata.get("training_objective", ""))
            )
            expected_protocol = REQUIRED_CYCLIC_REPRESENTATION_EXPERT_PROTOCOL
        else:
            metadata_is_complete = (
                observed_protocol == REQUIRED_ORDER_BALANCED_EXPERT_PROTOCOL
                and int(checkpoint_metadata.get("minimum_order_coverage_epochs", 0))
                >= 30
                and str(
                    checkpoint_metadata.get("training_decoding_order_policy", "")
                )
                == "epoch_indexed_cyclic_designed_position_rotation"
                and str(
                    checkpoint_metadata.get("deployment_annotation_policy", "")
                )
                == "complete_natural_sequence_all_cyclic_rotations_probability_mean"
            )
            expected_protocol = REQUIRED_ORDER_BALANCED_EXPERT_PROTOCOL
        if not metadata_is_complete:
            raise RuntimeError(
                "Generation is blocked because the expert checkpoint was not "
                "trained and promoted with the required training/deployment "
                "protocol metadata. Expected "
                f"{expected_protocol!r}, observed "
                f"{observed_protocol or '<missing>'!r}. Rerun expert-head "
                "retraining before generation."
            )
        del checkpoint_payload
    model_sha256 = sha256_file(model_path)
    representation_audit: Dict[str, Any] = {}
    representation_audit_path: Path | None = None
    if args.cyclic_representation_ensemble:
        if not args.representation_audit_json:
            raise ValueError(
                "--representation-audit-json is required with "
                "--cyclic-representation-ensemble"
            )
        representation_audit_path = Path(args.representation_audit_json).resolve()
        if not representation_audit_path.is_file():
            raise FileNotFoundError(representation_audit_path)
        representation_audit = read_json(representation_audit_path)
        audit_quality_checks = representation_audit.get("quality_checks", {})
        v9_audit_contract = (
            not is_v9_cyclic_stability_plan
            or (
                isinstance(audit_quality_checks, Mapping)
                and bool(audit_quality_checks)
                and all(bool(value) for value in audit_quality_checks.values())
                and str(
                    representation_audit.get("annotation_context_policy", "")
                )
                == PEPTIDE_ONLY_ANNOTATION_CONTEXT
                and float(representation_audit.get("temperature", -1.0))
                == float(plan["temperature"])
                and float(representation_audit.get("threshold", -1.0))
                == float(plan["methyl_threshold"])
            )
        )
        audit_is_authorized = (
            str(representation_audit.get("quality_gate", "")) == "PASS"
            and v9_audit_contract
            and str(representation_audit.get("protocol", ""))
            == REPRESENTATION_AUDIT_PROTOCOL
            and str(representation_audit.get("release_authorization", ""))
            == REPRESENTATION_AUDIT_AUTHORIZATION
            and str(representation_audit.get("model_sha256", "")) == model_sha256
            and str(representation_audit.get("plan_sha256", ""))
            == sha256_file(Path(args.plan).resolve())
            and str(representation_audit.get("annotation_mode", ""))
            == CYCLIC_REPRESENTATION_ANNOTATION_MODE
        )
        if not audit_is_authorized:
            raise RuntimeError(
                "Cyclic-representation generation is blocked because its held-out "
                "audit is absent, failed, or belongs to different model/plan bytes"
            )
    annotation_probability_fn = (
        cyclic_representation_known_sequence_methyl_probabilities
        if args.cyclic_representation_ensemble
        else cyclic_known_sequence_methyl_probabilities
    )
    annotation_mode = (
        CYCLIC_REPRESENTATION_ANNOTATION_MODE
        if args.cyclic_representation_ensemble
        else PEPTIDE_ONLY_ANNOTATION_MODE
    )
    ensure_output_scope(out_dir, args.overwrite)
    if args.defer_permeability_until_structure and args.overwrite:
        # ``--overwrite`` is an explicit request to replace this isolated run.
        # Do not let permeability inputs from an older non-structure-first run
        # survive and masquerade as outputs of the new protocol.
        for output_name in PERMEABILITY_OUTPUTS:
            stale_path = out_dir / output_name
            if stale_path.is_file():
                stale_path.unlink()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU generation is intentionally blocked; pass --allow-cpu to override")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise RuntimeError(
                "No CUDA device is available. Activate the GPU torch environment, or pass --allow-cpu knowingly."
            )

    best_rows = read_csv(best_path)
    selected_chains = selected_chain_index(best_rows)
    native_rows = read_jsonl(native_path)
    target_records, target_manifest = prepare_target_records(
        native_rows, selected_chains, validated["target_names"]
    )
    all_native = native_manifest_all_targets(native_rows, selected_chains)
    old_exact_keys, old_natural_keys = old_design_keys(old_path)
    if prior_path is not None:
        prior_rows, prior_exact_keys, prior_natural_keys = validate_prior_handoff(
            prior_path
        )
    else:
        prior_rows, prior_exact_keys, prior_natural_keys = [], set(), set()
    plan_by_target = {
        str(item["target_name"]).upper(): dict(item) for item in validated["targets"]
    }

    print(f"Loading strict clean-V28 checkpoint: {model_path}", flush=True)
    model = load_v28_model(str(model_path), device)
    model.eval()
    print(f"Device: {device}", flush=True)
    print(f"Targets: {len(validated['target_names'])}", flush=True)
    print(f"Seeds: {validated['seeds']}", flush=True)
    print(f"Expected raw draws: {validated['expected_raw_candidates']}", flush=True)

    raw_rows: List[Dict[str, Any]] = []
    rows_by_target_seed: MutableMapping[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    temperature = float(plan["temperature"])
    methyl_threshold = float(plan["methyl_threshold"])

    for target_index, target in enumerate(validated["target_names"]):
        metadata = next(row for row in target_manifest if row["target_name"] == target)
        packed = featurize_records(
            [target_records[target]],
            device=device,
            eval_chains="masked",
            max_peptide_len=30,
        )
        if packed is None:
            raise RuntimeError(f"Feature construction failed for {target}")
        features, feature_meta = packed
        if int(feature_meta[0]["selected_length"]) != int(metadata["native_peptide_length"]):
            raise RuntimeError(f"Peptide coordinate/sequence length mismatch for {target}")

        per_seed = int(plan_by_target[target]["sequences_per_seed"])
        for base_seed in validated["seeds"]:
            effective_seed = int(base_seed) * 100_000 + stable_target_offset(target)
            torch_seed_all(torch, effective_seed)
            produced = 0
            while produced < per_seed:
                current_batch = min(int(args.batch_size), per_seed - produced)
                try:
                    generated = generate_batch(
                        model=model,
                        features=features,
                        batch_size=current_batch,
                        temperature=temperature,
                        methyl_threshold=methyl_threshold,
                        torch_module=torch,
                        functional=F,
                        extended_alphabet=EXTENDED_AA_ALPHABET,
                        x_index=int(EXTENDED_AA_TO_INDEX["X"]),
                        natural_to_methyl=NAT_TO_METHYL_ABS,
                        complete_order_fn=complete_decoding_order,
                        ensemble_probability_fn=annotation_probability_fn,
                        peptide_only_tensors_fn=peptide_only_annotation_tensors,
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise RuntimeError(
                            f"CUDA out of memory for batch_size={args.batch_size}. "
                            "Rerun the isolated output with a smaller -BatchSize and -Force."
                        ) from exc
                    raise

                for batch_offset, generated_row in enumerate(generated):
                    draw_index = produced + batch_offset + 1
                    sequence = str(generated_row["design_seq"])
                    candidate_id = f"t05_{target.lower()}_s{base_seed}_{draw_index:04d}"
                    methyl_positions = methyl_positions_1based(sequence)
                    row: Dict[str, Any] = {
                        "candidate_id": candidate_id,
                        "target_name": target,
                        "temperature": temperature,
                        "methyl_threshold": methyl_threshold,
                        "seed": int(base_seed),
                        "effective_seed": effective_seed,
                        "draw_index_within_seed": draw_index,
                        "selected_chain": metadata["selected_chain"],
                        "generation_receptor_chain": metadata["generation_receptor_chain"],
                        "structure_receptor_chains": metadata["structure_receptor_chains"],
                        "native_seq": metadata["native_peptide_seq"],
                        "design_seq": sequence,
                        "design_natural_seq": naturalize(sequence),
                        "native_length": int(metadata["native_peptide_length"]),
                        "design_length": len(sequence),
                        "length_match": int(len(sequence) == int(metadata["native_peptide_length"])),
                        "valid_token_gate": int(bool(sequence) and set(sequence) <= VALID_DESIGN_TOKENS),
                        "natural_aa_recovery": sequence_recovery(
                            str(metadata["native_peptide_seq"]), sequence
                        ),
                        "design_methyl_count": len(methyl_positions),
                        "design_methyl_rate": len(methyl_positions) / len(sequence),
                        "methyl_positions_1based": json.dumps(methyl_positions),
                        "current_problem": plan_by_target[target]["current_problem"],
                        "planned_structure_quota": int(plan_by_target[target]["structure_quota"]),
                        **generated_row,
                    }
                    raw_rows.append(row)
                    rows_by_target_seed[(target, int(base_seed))].append(row)
                produced += current_batch
            print(
                f"[{target}] seed={base_seed}: generated {per_seed}",
                flush=True,
            )

    if len(raw_rows) != int(validated["expected_raw_candidates"]):
        raise RuntimeError(
            f"Raw-count quality gate failed: {len(raw_rows)} != {validated['expected_raw_candidates']}"
        )
    invalid_rows = [
        row for row in raw_rows if not int(row["length_match"]) or not int(row["valid_token_gate"])
    ]
    if invalid_rows:
        raise RuntimeError(f"Sequence quality gate failed for {len(invalid_rows)} generated rows")

    canonicalization = canonicalize_repeated_natural_annotations(raw_rows)

    for (target, seed), seed_rows in sorted(rows_by_target_seed.items()):
        write_seed_fasta(
            out_dir / "generated_fastas" / f"seed_{seed}" / f"{target.lower()}_designs.fasta",
            seed_rows,
        )

    unique_rows = aggregate_unique_candidates(
        raw_rows,
        old_exact_keys,
        old_natural_keys,
        prior_exact_keys,
        prior_natural_keys,
    )
    eligible_rows = [row for row in unique_rows if int(row["eligible_for_new_permeability_screen"])]
    permeability_input: List[Dict[str, Any]] = []
    permeability_manifest: List[Dict[str, Any]] = []
    eligible_id_by_key: Dict[Tuple[str, str], str] = {}
    if not args.defer_permeability_until_structure:
        permeability_input, permeability_manifest, eligible_with_ids = build_permeability_rows(
            eligible_rows, all_native
        )
        eligible_id_by_key = {
            (str(row["target_name"]), str(row["design_seq"])): str(row["permeability_id"])
            for row in eligible_with_ids
        }
    for row in unique_rows:
        row["permeability_id"] = eligible_id_by_key.get(
            (str(row["target_name"]), str(row["design_seq"])), ""
        )

    raw_fields = list(raw_rows[0].keys())
    unique_extra = [
        "occurrence_count",
        "seeds_observed",
        "seen_in_historical_4115",
        "seen_in_historical_4115_exact",
        "seen_in_historical_4115_naturalized",
        "seen_in_prior_1333",
        "seen_in_prior_1333_exact",
        "seen_in_prior_1333_naturalized",
        "passes_methylation_hard_gate",
        "eligible_for_new_permeability_screen",
        "permeability_id",
    ]
    unique_fields = raw_fields + [field for field in unique_extra if field not in raw_fields]
    atomic_write_csv(out_dir / "all_candidates.csv", raw_rows, raw_fields)
    atomic_write_csv(out_dir / "unique_candidates.csv", unique_rows, unique_fields)
    atomic_write_csv(
        out_dir / "methylated_new_candidates.csv",
        [row for row in unique_rows if int(row["eligible_for_new_permeability_screen"])],
        unique_fields,
    )
    if not args.defer_permeability_until_structure:
        atomic_write_csv(
            out_dir / "permeability_input.csv",
            permeability_input,
            ["id", "fasta", "methy_index"],
        )
        atomic_write_csv(
            out_dir / "permeability_input_manifest.csv",
            permeability_manifest,
            [
                "id",
                "record_type",
                "target_name",
                "candidate_id",
                "design_seq",
                "design_natural_seq",
                "methy_index",
            ],
        )
    atomic_write_csv(
        out_dir / "target_manifest.csv",
        target_manifest,
        list(target_manifest[0].keys()),
    )

    summary_rows: List[Dict[str, Any]] = []
    for target in validated["target_names"]:
        target_raw = [row for row in raw_rows if row["target_name"] == target]
        target_unique = [row for row in unique_rows if row["target_name"] == target]
        target_eligible = [
            row for row in target_unique if int(row["eligible_for_new_permeability_screen"])
        ]
        target_plan = plan_by_target[target]
        summary_rows.append(
            {
                "target_name": target,
                "current_problem": target_plan["current_problem"],
                "raw_generated": len(target_raw),
                "unique_generated": len(target_unique),
                "unique_methylated": sum(
                    int(row["passes_methylation_hard_gate"]) for row in target_unique
                ),
                "historical_4115_hits": sum(
                    int(row["seen_in_historical_4115"]) for row in target_unique
                ),
                "prior_1333_hits": sum(
                    int(row["seen_in_prior_1333"]) for row in target_unique
                ),
                "new_methylated_for_permeability": len(target_eligible),
                "planned_structure_quota": int(target_plan["structure_quota"]),
                "enough_candidates_before_permeability": int(
                    len(target_eligible) >= int(target_plan["structure_quota"])
                ),
            }
        )
    atomic_write_csv(
        out_dir / "generation_summary_by_target.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )

    annotation_audit = audit_annotation_stability(raw_rows, eligible_rows)
    targets_below_quota = [
        row["target_name"]
        for row in summary_rows
        if not int(row["enough_candidates_before_permeability"])
    ]
    generation_quality_checks = {
        **dict(annotation_audit["quality_checks"]),
        "every_target_meets_pre_structure_candidate_quota": not targets_below_quota,
    }
    generation_quality_gate = (
        "PASS" if all(generation_quality_checks.values()) else "FAIL"
    )

    manifest_payload = {
        "quality_gate": generation_quality_gate,
        "quality_checks": generation_quality_checks,
        "protocol": plan["protocol"],
        "temperature": temperature,
        "methyl_threshold": methyl_threshold,
        "seeds": validated["seeds"],
        "batch_size": int(args.batch_size),
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "model_expert_qc_protocol": checkpoint_metadata.get("protocol"),
        "native_jsonl": str(native_path),
        "best_csv": str(best_path),
        "historical_design_csv": str(old_path),
        "historical_design_csv_sha256": sha256_file(old_path),
        "historical_exact_design_keys": len(old_exact_keys),
        "historical_naturalized_design_keys": len(old_natural_keys),
        "prior_handoff_csv": str(prior_path) if prior_path is not None else None,
        "prior_handoff_csv_sha256": (
            sha256_file(prior_path) if prior_path is not None else None
        ),
        "prior_handoff_rows": len(prior_rows),
        "prior_handoff_exact_design_keys": len(prior_exact_keys),
        "prior_handoff_naturalized_design_keys": len(prior_natural_keys),
        "raw_candidates_expected": int(validated["expected_raw_candidates"]),
        "raw_candidates_generated": len(raw_rows),
        "unique_candidates": len(unique_rows),
        "all_candidates_csv_sha256": sha256_file(out_dir / "all_candidates.csv"),
        "unique_candidates_csv_sha256": sha256_file(
            out_dir / "unique_candidates.csv"
        ),
        "methylated_new_candidates_csv_sha256": sha256_file(
            out_dir / "methylated_new_candidates.csv"
        ),
        "new_methylated_candidates_for_permeability": len(eligible_rows),
        "workflow_order": (
            "STRUCTURE_FIRST_THEN_PERMEABILITY"
            if args.defer_permeability_until_structure
            else "GENERATION_THEN_PERMEABILITY_INPUT"
        ),
        "permeability_status": (
            "DEFERRED_UNTIL_STRUCTURE_RETURNS"
            if args.defer_permeability_until_structure
            else "INPUT_READY_NOT_PREDICTED"
        ),
        "native_permeability_controls": (
            0 if args.defer_permeability_until_structure else len(all_native)
        ),
        "permeability_input_rows": len(permeability_input),
        "planned_structure_handoff": int(validated["planned_structure_handoff"]),
        "targets_below_pre_permeability_quota": targets_below_quota,
        "frozen_targets_not_regenerated": plan["frozen_targets"],
        "sampler_definition": (
            "one explicit random peptide-position order is shared by the outer "
            "sampling loop and causal decoder mask; natural base sampled at T=0.5; "
            "after the complete natural sequence is available, visible receptor chains "
            "are removed and expert probabilities are averaged over every causal "
            "decoder-depth rotation"
            + (
                " and every joint sequence/coordinate cyclic start after mapping "
                "back to physical residues"
                if args.cyclic_representation_ensemble
                else " while the serialized cyclic start remains fixed"
            )
            + "; the ensemble mean is retained for ranking, but a lowercase "
            "methyl token is released only when the mapped-back probability "
            "minimum across every cyclic start is strictly >0.6 and no start "
            "straddles the threshold; only natural parents enter model context"
        ),
        "generation_decoding_order_policy": (
            "explicit random designed-position permutation, receptor/padding prefix; "
            "the exact same full permutation is passed to every model forward"
        ),
        "annotation_order_policy": (
            "complete-natural-sequence cyclic ensemble; every peptide site occurs "
            "once at every relative decoder depth"
        ),
        "annotation_representation_policy": (
            "all equivalent cyclic starts jointly rotate sequence and N/CA/C/O "
            "coordinates, reset linear residue indices, and map probabilities back "
            "to physical residues before averaging"
            if args.cyclic_representation_ensemble
            else "serialized cyclic start fixed; decoder-order ensemble only"
        ),
        "sampling_context_policy": SAMPLING_CONTEXT_POLICY,
        "annotation_mode": annotation_mode,
        "annotation_context_policy": PEPTIDE_ONLY_ANNOTATION_CONTEXT,
        "annotation_visible_receptor_chains": 0,
        "train_deployment_context_match": True,
        "cyclic_representation_ensemble_enabled": bool(
            args.cyclic_representation_ensemble
        ),
        "cyclic_representation_heldout_audit": (
            {
                "path": str(representation_audit_path),
                "sha256": sha256_file(representation_audit_path),
                "quality_gate": representation_audit.get("quality_gate"),
                "protocol": representation_audit.get("protocol"),
                "release_authorization": representation_audit.get(
                    "release_authorization"
                ),
                "model_sha256": representation_audit.get("model_sha256"),
                "plan_sha256": representation_audit.get("plan_sha256"),
                "temperature": representation_audit.get("temperature"),
                "threshold": representation_audit.get("threshold"),
                "annotation_mode": representation_audit.get("annotation_mode"),
                "annotation_context_policy": representation_audit.get(
                    "annotation_context_policy"
                ),
            }
            if representation_audit_path is not None
            else None
        ),
        "annotation_payload_canonicalization": canonicalization,
        "annotation_stability_audit": annotation_audit,
        "autoregressive_input_policy": (
            "natural-only model context; lowercase expert annotations are output-only"
        ),
        "permeability_definition_pending": (
            "candidate prediction must be strictly greater than the same-model native-peptide prediction"
        ),
    }
    atomic_write_json(out_dir / "generation_manifest.json", manifest_payload)

    print("\n===== GENERATION COMPLETE =====", flush=True)
    print(f"Raw candidates: {len(raw_rows)}", flush=True)
    print(f"Unique candidates: {len(unique_rows)}", flush=True)
    print(f"New methylated candidates: {len(eligible_rows)}", flush=True)
    if args.defer_permeability_until_structure:
        print("Permeability: DEFERRED_UNTIL_STRUCTURE_RETURNS", flush=True)
    else:
        print(f"Permeability input: {out_dir / 'permeability_input.csv'}", flush=True)
    print(f"Quality gate: {manifest_payload['quality_gate']}", flush=True)
    if generation_quality_gate != "PASS":
        failed = [
            name for name, passed in generation_quality_checks.items() if not passed
        ]
        raise RuntimeError(
            "Generation annotation/coverage quality gate failed; handoff is blocked: "
            + ", ".join(failed)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--native_jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--best_csv", default=str(DEFAULT_BEST))
    parser.add_argument("--old_designs_csv", default=str(DEFAULT_OLD))
    parser.add_argument("--prior_designs_csv")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate-prior-designs-only", action="store_true")
    parser.add_argument(
        "--cyclic-representation-ensemble",
        action="store_true",
        help=(
            "jointly rotate peptide sequence/coordinates through every cyclic start, "
            "map probabilities back to physical residues, and average"
        ),
    )
    parser.add_argument(
        "--representation-audit-json",
        help=(
            "required PASS report produced by "
            "07_audit_cyclic_representation_equivariance.py"
        ),
    )
    parser.add_argument(
        "--defer-permeability-until-structure",
        action="store_true",
        help=(
            "enforce the executed structure-first workflow by not writing "
            "permeability_input.csv or its manifest"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be positive")
    plan = read_json(Path(args.plan).resolve())
    validated = validate_plan(plan, args.seeds)
    if args.validate_prior_designs_only:
        if not args.prior_designs_csv:
            raise ValueError("--prior_designs_csv is required for prior-only validation")
        rows, exact_keys, natural_keys = validate_prior_handoff(
            Path(args.prior_designs_csv).resolve()
        )
        print(
            json.dumps(
                {
                    "quality_gate": "PASS",
                    "rows": len(rows),
                    "exact_target_sequence_keys": len(exact_keys),
                    "naturalized_target_sequence_keys": len(natural_keys),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.plan_only:
        print(
            json.dumps(
                {
                    "protocol": plan["protocol"],
                    "temperature": plan["temperature"],
                    "methyl_threshold": plan["methyl_threshold"],
                    "seeds": validated["seeds"],
                    "rerun_targets": validated["target_names"],
                    "frozen_targets": plan["frozen_targets"],
                    "expected_target_count": validated["expected_target_count"],
                    "expected_raw_candidates": validated["expected_raw_candidates"],
                    "planned_structure_handoff": validated["planned_structure_handoff"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_generation(args, plan, validated)


if __name__ == "__main__":
    main()
