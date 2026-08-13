#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the fixed T=0.5 multiseed rerun.

The sampler preserves the historical V28 base/expert decision rule from
``DCarchimonde/ProteinMPNN:nmethyl/generate_100_seqs_robust.py`` while fixing
one train/inference mismatch:

1. randomly permute the designed peptide positions for every draw;
2. sample the natural amino acid from the base head at the fixed temperature;
3. query the sampled amino-acid expert at that same temperature;
4. emit the lowercase N-methyl token when the expert probability is strictly
   greater than the frozen methylation threshold.

The emitted lowercase annotation is stored separately from the autoregressive
model context.  Only the sampled natural parent residue is fed into subsequent
decoder steps, exactly matching the leakage-free expert-head training input.

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
) -> List[Dict[str, Any]]:
    X, S_true, mask, chain_M, residue_idx, chain_encoding_all = features[:6]
    X = repeat_batch(X, batch_size)
    S_context = repeat_batch(S_true, batch_size).clone()
    S_output = S_context.clone()
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
    S_output.copy_(S_context)
    S_context[:, masked_positions] = x_index
    S_output[:, masked_positions] = x_index
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
    methyl_probability = torch_module.zeros((batch_size, peptide_length), device=X.device)

    # clean_v28_common still uses torch.utils.checkpoint in forward. no_grad is
    # compatible with that implementation and matches the historical sampler;
    # inference_mode can reject tensors saved internally by checkpoint.
    with torch_module.no_grad():
        for step in range(peptide_length):
            positions = orders[:, step]
            logits_base, logits_experts = model(
                X, S_context, mask, chain_M, residue_idx, chain_encoding_all
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

            final_token = sampled_base.clone()
            for natural_index, methyl_index in natural_to_methyl.items():
                use_methyl = sampled_base.eq(int(natural_index)) & current_methyl_probability.gt(
                    methyl_threshold
                )
                final_token = torch_module.where(
                    use_methyl,
                    torch_module.full_like(final_token, int(methyl_index)),
                    final_token,
                )
            S_context[row_indices, positions] = sampled_base
            S_output[row_indices, positions] = final_token

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
            methyl_probability[row_indices, relative_positions] = current_methyl_probability

    results: List[Dict[str, Any]] = []
    peptide_tokens = S_output[:, masked_positions].detach().cpu().tolist()
    base_lp = base_log_prob.detach().cpu().tolist()
    sampled_lp = sampled_log_prob.detach().cpu().tolist()
    methyl_p = methyl_probability.detach().cpu().tolist()
    orders_cpu = orders.detach().cpu().tolist()
    for index in range(batch_size):
        sequence = "".join(extended_alphabet[int(token)] for token in peptide_tokens[index])
        methyl_site_probabilities = [
            float(methyl_p[index][position])
            for position, token in enumerate(sequence)
            if token.islower()
        ]
        results.append(
            {
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
                "methyl_probabilities": json.dumps(
                    [round(float(value), 8) for value in methyl_p[index]]
                ),
                "decoding_order_absolute": json.dumps([int(value) for value in orders_cpu[index]]),
            }
        )
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
        row["passes_methylation_hard_gate"] = int(int(row["design_methyl_count"]) > 0)
        row["eligible_for_new_permeability_screen"] = int(
            int(row["design_methyl_count"]) > 0
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
        featurize_records,
        load_v28_model,
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
    if "all_expert_qc" in str(plan.get("protocol", "")):
        if prior_path is None or not prior_path.is_file():
            raise FileNotFoundError(
                "This recovery protocol requires the prior 1,333-row handoff CSV "
                "for hard duplicate exclusion"
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

    manifest_payload = {
        "quality_gate": "PASS",
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
        "model_sha256": sha256_file(model_path),
        "native_jsonl": str(native_path),
        "best_csv": str(best_path),
        "historical_design_csv": str(old_path),
        "historical_exact_design_keys": len(old_exact_keys),
        "historical_naturalized_design_keys": len(old_natural_keys),
        "prior_handoff_csv": str(prior_path) if prior_path is not None else None,
        "prior_handoff_rows": len(prior_rows),
        "prior_handoff_exact_design_keys": len(prior_exact_keys),
        "prior_handoff_naturalized_design_keys": len(prior_natural_keys),
        "raw_candidates_expected": int(validated["expected_raw_candidates"]),
        "raw_candidates_generated": len(raw_rows),
        "unique_candidates": len(unique_rows),
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
        "targets_below_pre_permeability_quota": [
            row["target_name"]
            for row in summary_rows
            if not int(row["enough_candidates_before_permeability"])
        ],
        "frozen_targets_not_regenerated": plan["frozen_targets"],
        "sampler_definition": (
            "random peptide-position order; natural base sampled at T=0.5; "
            "sampled-base expert sigmoid(logit/T)>0.6 emits lowercase methyl token; "
            "only the natural parent is fed back into later decoder steps"
        ),
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
