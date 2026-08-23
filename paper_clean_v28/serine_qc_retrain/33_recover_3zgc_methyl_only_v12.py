#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recover exactly 100 strict V11 methylation candidates for 3ZGC.

This is a pre-structure sequence search.  It deliberately does not calculate,
predict, or rank RMSD, and it does not require the historical ProteinMPNN base
floor.  Candidates are released only after the promoted V11 model reproduces
an explicit representation-minimum probability strictly greater than 0.6 at
at least one methylatable site, with zero cyclic-start threshold disagreement.

The search first replays every available historical/current 3ZGC sequence under
the *same* V11 checkpoint.  It then explores complete single mutants and a
fixed deterministic multi-mutant budget around the best full-sequence beam.
Only 100 independently batch-one-confirmed, novel cyclic identities are written
to the release CSV.  Scored search rows are diagnostics, not structures and not
handoff rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
V8_SEARCH_PATH = SCRIPT_PATH.with_name("14_directed_recovery_search_v8.py")
REANNOTATOR_PATH = SCRIPT_PATH.with_name("10_reannotate_v6_pool_serine_only_v7.py")
GENERATOR_PATH = (
    REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
)
V11_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "cyclic_native_v11_1700_monomer"
DEFAULT_MODEL = V11_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
DEFAULT_AUDIT = V11_ROOT / "representation_audit" / "cyclic_representation_audit.json"
DEFAULT_GENERATION = V11_ROOT / "generation"
DEFAULT_PLAN = SCRIPT_PATH.with_name(
    "target_plan_v11_cyclic_native_rmsd_priority_1700.json"
)
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_BEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "best_designs.csv"
)
DEFAULT_HISTORICAL = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "all_designs.csv"
)
DEFAULT_PRIOR = REPO_ROOT / "v9_inputs" / "methylated_new_candidates.csv"
DEFAULT_RMSD_HISTORY = REPO_ROOT / "v10_inputs" / "six_non3av_t05_joint_rmsd_476.csv"
DEFAULT_OUT = V11_ROOT / "v12_methyl_only" / "3zgc_directed_search"

TARGET = "3ZGC"
NATIVE_SEQUENCE = "GDEETGE"
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
METHYLATABLE_AA = set(NATURAL_AA) - {"P"}
THRESHOLD = 0.6
TEMPERATURE = 0.5
QUOTA = 100
PROBABILITY_ATOL = 2e-6
V11_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_cyclic_native_relative_positions_v11"
)
V11_AUDIT_PROTOCOL = "cyclic_native_relative_positions_heldout_gate_v11"
SEARCH_PROTOCOL = "v12_3zgc_complete_sequence_methyl_only_directed_search_v1"

SEQUENCE_FIELDS = (
    "design_natural_seq",
    "sequence",
    "design_seq",
    "fasta",
)
TARGET_FIELDS = ("target_name", "target", "pdb_id")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
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


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = union_fields(rows)
    if not fields:
        fields = ["target_name", "sequence", "status"]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def strict_pass(value: float, threshold: float = THRESHOLD) -> bool:
    numeric = float(value)
    return (
        math.isfinite(numeric)
        and 0.0 <= numeric <= 1.0
        and round(numeric, 8) > float(threshold)
    )


def canonical_rotation(sequence: str) -> str:
    natural = str(sequence).upper()
    if not natural:
        raise ValueError("Cannot canonicalize an empty sequence")
    return min(natural[index:] + natural[:index] for index in range(len(natural)))


def normalized_sequence(row: Mapping[str, Any]) -> str:
    for field in SEQUENCE_FIELDS:
        value = str(row.get(field, "")).strip()
        if value:
            natural = value.upper()
            if set(natural) <= set(NATURAL_AA):
                return natural
    return ""


def normalized_target(row: Mapping[str, Any]) -> str:
    for field in TARGET_FIELDS:
        value = str(row.get(field, "")).strip().upper()
        if value:
            return value
    return ""


def seed_sequences_from_rows(
    rows: Sequence[Mapping[str, Any]],
    source: str,
) -> Tuple[set[str], List[Dict[str, Any]]]:
    sequences: set[str] = set()
    problems: List[Dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        if normalized_target(row) != TARGET:
            continue
        sequence = normalized_sequence(row)
        if not sequence:
            problems.append(
                {
                    "source": source,
                    "row_number": row_number,
                    "problem": "missing_or_invalid_natural_sequence",
                }
            )
            continue
        if len(sequence) != len(NATIVE_SEQUENCE):
            problems.append(
                {
                    "source": source,
                    "row_number": row_number,
                    "sequence": sequence,
                    "problem": "length_not_equal_to_3zgc_native",
                }
            )
            continue
        sequences.add(sequence)
    return sequences, problems


def actionable_max(sequence: str, values: Sequence[float]) -> Tuple[float, int]:
    candidates = [
        (float(value), index)
        for index, (token, value) in enumerate(zip(sequence, values))
        if token in METHYLATABLE_AA
    ]
    if not candidates:
        return 0.0, 0
    value, index = max(candidates, key=lambda pair: (pair[0], -pair[1]))
    return value, index + 1


def release_rank(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(row["release_floor_maximum_probability"]),
        -float(row["ranking_mean_maximum_probability"]),
        float(row["methyl_probability_representation_span_max"]),
        str(row["design_natural_seq"]),
    )


def select_novel_release_rows(
    rows: Sequence[Mapping[str, Any]],
    excluded_natural: set[str],
    excluded_cyclic: set[str],
    quota: int = QUOTA,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    accepted_cyclic: set[str] = set()
    for source in sorted((dict(row) for row in rows), key=release_rank):
        natural = str(source.get("design_natural_seq", "")).upper()
        cyclic = canonical_rotation(natural)
        reason = ""
        if natural in excluded_natural:
            reason = "previously_generated_or_historical_exact"
        elif cyclic in excluded_cyclic:
            reason = "previously_generated_or_historical_forward_cyclic"
        elif cyclic in accepted_cyclic:
            reason = "selected_forward_cyclic_duplicate"
        if reason:
            source["v12_rejection_reason"] = reason
            rejected.append(source)
            continue
        if len(accepted) >= int(quota):
            source["v12_rejection_reason"] = "beyond_exact_100_release_quota"
            rejected.append(source)
            continue
        accepted_cyclic.add(cyclic)
        source["forward_cyclic_identity"] = cyclic
        source["v12_release_rank"] = len(accepted) + 1
        accepted.append(source)
    return accepted, rejected


class V11ReleaseFloorScorer:
    """Full-sequence V11 scorer whose search objective is the release minimum."""

    def __init__(self, v8_module: Any, *args: Any, **kwargs: Any) -> None:
        self.v8 = v8_module
        self.delegate = v8_module.MethylScorer(*args, **kwargs)

    def score_objective(
        self, target: str, sequences: Sequence[str], stage: str
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        unique = sorted(set(str(value).upper() for value in sequences))
        progress = self.v8.ProgressBar(f"{target} {stage}", len(unique), unit="seq")
        for batch in self.v8.chunks(unique, self.delegate.batch_size):
            representation = self.delegate._representations(target, batch)
            for row_index, sequence in enumerate(batch):
                minimum = [
                    round(float(value), 8)
                    for value in representation["min"][row_index].detach().cpu().tolist()
                ]
                mean = [
                    round(float(value), 8)
                    for value in representation["mean"][row_index].detach().cpu().tolist()
                ]
                maximum, position = actionable_max(sequence, minimum)
                mean_maximum, _mean_position = actionable_max(sequence, mean)
                result[sequence] = {
                    "target_name": TARGET,
                    "sequence": sequence,
                    "search_stage": stage,
                    "maximum_probability": maximum,
                    "release_floor_maximum_probability": maximum,
                    "ranking_mean_maximum_probability": mean_maximum,
                    "argmax_position_1based": position,
                    "argmax_residue": sequence[position - 1] if position else "",
                    "passes_strict_probability": int(position > 0 and strict_pass(maximum)),
                }
            progress.update(len(batch))
        progress.close()
        return result

    def score_full(
        self,
        target: str,
        sequences: Sequence[str],
        stage: str,
        *,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        return self.delegate.score_full(
            target, sequences, stage=stage, show_progress=show_progress
        )


def validate_v11_contract(
    torch_module: Any,
    model_path: Path,
    audit_path: Path,
    plan_path: Path,
) -> Dict[str, Any]:
    checkpoint = torch_module.load(model_path, map_location="cpu")
    metadata = (
        dict(checkpoint.get("expert_head_qc_metadata", {}))
        if isinstance(checkpoint, Mapping)
        else {}
    )
    del checkpoint
    required_metadata = (
        str(metadata.get("protocol", "")) == V11_EXPERT_PROTOCOL
        and bool(metadata.get("cyclic_relative_positions"))
        and float(metadata.get("worst_start_bce_weight", 0.0)) > 0.0
        and float(metadata.get("representation_consistency_weight", 0.0)) > 0.0
        and float(metadata.get("base_sequence_loss_weight", 0.0)) > 0.0
        and float(metadata.get("positional_anchor_weight", 0.0)) > 0.0
        and bool(metadata.get("full_physical_start_by_full_decoder_order_grid"))
    )
    if not required_metadata:
        raise RuntimeError("3ZGC V12 search requires the promoted cyclic-native V11 checkpoint")
    audit = read_json(audit_path)
    checks = dict(audit.get("quality_checks") or {})
    if not (
        audit.get("quality_gate") == "PASS"
        and audit.get("protocol") == V11_AUDIT_PROTOCOL
        and audit.get("model_sha256") == sha256_file(model_path)
        and audit.get("plan_sha256") == sha256_file(plan_path)
        and checks
        and all(value is True for value in checks.values())
        and float(audit.get("temperature", -1.0)) == TEMPERATURE
        and float(audit.get("threshold", -1.0)) == THRESHOLD
    ):
        raise RuntimeError("3ZGC V12 search requires the exact all-PASS V11 audit")
    return {"checkpoint_metadata": metadata, "audit": audit}


def source_rows_and_hashes(paths: Sequence[Path]) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Any]]:
    rows: Dict[str, List[Dict[str, str]]] = {}
    records: Dict[str, Any] = {}
    for index, path in enumerate(paths):
        resolved = path.resolve()
        label = f"seed_source_{index:02d}_{resolved.name}"
        rows[label] = read_csv(resolved)
        records[label] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "rows": len(rows[label]),
        }
    return rows, records


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("V12 directed search requires numpy and torch") from exc

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model_path = Path(args.model).resolve()
    audit_path = Path(args.representation_audit).resolve()
    generation_dir = Path(args.generation_dir).resolve()
    generation_manifest_path = generation_dir / "generation_manifest.json"
    generation_all_path = generation_dir / "all_candidates.csv"
    plan_path = Path(args.v11_plan).resolve()
    native_path = Path(args.native_jsonl).resolve()
    best_path = Path(args.best_csv).resolve()
    historical_path = Path(args.historical_csv).resolve()
    prior_path = Path(args.prior_csv).resolve()
    rmsd_history_path = Path(args.rmsd_history_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    extra_seed_paths = [Path(value).resolve() for value in args.seed_csv]
    required = (
        model_path,
        audit_path,
        generation_manifest_path,
        generation_all_path,
        plan_path,
        native_path,
        best_path,
        historical_path,
        prior_path,
        rmsd_history_path,
        V8_SEARCH_PATH,
        REANNOTATOR_PATH,
        GENERATOR_PATH,
        *extra_seed_paths,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if out_dir in required or any(out_dir == path.parent for path in required):
        raise ValueError("V12 output directory must not overlap an immutable input")
    out_dir.mkdir(parents=True, exist_ok=True)

    contract = validate_v11_contract(torch, model_path, audit_path, plan_path)
    generation_manifest = read_json(generation_manifest_path)
    generation_artifacts = dict(generation_manifest.get("artifacts") or {})
    all_record = dict(generation_artifacts.get("all_candidates") or {})
    allowed_generation_failures = {
        "every_target_meets_pre_structure_candidate_quota",
        "every_target_meets_final_release_diversity_reserve",
        "no_single_position_exceeds_80_percent_of_sites",
        "no_single_residue_exceeds_80_percent_of_sites",
        "no_target_has_single_residue_above_80_percent_when_n_ge_30",
        "no_target_has_unsupported_single_position_above_80_percent_when_n_ge_30",
    }
    generation_false = {
        name
        for name, passed in dict(generation_manifest.get("quality_checks") or {}).items()
        if not bool(passed)
    }
    if not (
        generation_manifest.get("model_sha256") == sha256_file(model_path)
        and float(generation_manifest.get("methyl_threshold", -1.0)) == THRESHOLD
        and float(generation_manifest.get("temperature", -1.0)) == TEMPERATURE
        and generation_false
        and generation_false <= allowed_generation_failures
        and str(all_record.get("sha256", "")) == sha256_file(generation_all_path)
    ):
        raise RuntimeError(
            "V12 search accepts only the preserved V11 pool whose sole failures are "
            "quota/diversity diagnostics"
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU search requires --allow-cpu")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise RuntimeError("No CUDA device; pass --allow-cpu only for a deliberate slow run")

    v8 = load_module("v12_v8_search_primitives", V8_SEARCH_PATH)
    reannotator = load_module("v12_reannotator", REANNOTATOR_PATH)
    generator = load_module("v12_generator", GENERATOR_PATH)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from paper_clean_v28.clean_v28_common import (  # pylint: disable=import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        NAT_TO_METHYL_ABS,
        NATURAL_AA_ALPHABET,
        cyclic_representation_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
    )

    best_rows = generator.read_csv(best_path)
    selected_chains = generator.selected_chain_index(best_rows)
    if str(selected_chains.get(TARGET, "")) != "C":
        raise RuntimeError("Frozen 3ZGC peptide chain is no longer C")
    native_rows = generator.read_jsonl(native_path)
    native_index = {
        generator.record_name(row, index): row for index, row in enumerate(native_rows)
    }
    if TARGET not in native_index:
        raise RuntimeError("3ZGC is absent from native JSONL")
    model = load_v28_model(str(model_path), device)
    model.eval()
    common = {
        "EXTENDED_AA_ALPHABET": EXTENDED_AA_ALPHABET,
        "NAT_TO_METHYL_ABS": NAT_TO_METHYL_ABS,
        "NATURAL_AA_ALPHABET": NATURAL_AA_ALPHABET,
        "cyclic_representation_known_sequence_methyl_probabilities": (
            cyclic_representation_known_sequence_methyl_probabilities
        ),
        "featurize_records": featurize_records,
    }
    scorer = V11ReleaseFloorScorer(
        v8,
        model,
        device,
        native_index,
        selected_chains,
        int(args.batch_size),
        torch,
        common,
        reannotator,
    )

    seed_paths = [
        generation_all_path,
        historical_path,
        prior_path,
        rmsd_history_path,
        *extra_seed_paths,
    ]
    rows_by_source, source_records = source_rows_and_hashes(seed_paths)
    source_sequences: MutableMapping[str, set[str]] = defaultdict(set)
    seed_problems: List[Dict[str, Any]] = []
    for source, rows in rows_by_source.items():
        sequences, problems = seed_sequences_from_rows(rows, source)
        source_sequences[source].update(sequences)
        seed_problems.extend(problems)
    seed_union = set().union(*source_sequences.values(), {NATIVE_SEQUENCE})
    if not seed_union:
        raise RuntimeError("No valid 3ZGC seed sequence was found")

    excluded_natural: set[str] = set()
    excluded_cyclic: set[str] = set()
    for path in (generation_all_path, historical_path, prior_path):
        sequences, _problems = seed_sequences_from_rows(read_csv(path), path.name)
        excluded_natural.update(sequences)
        excluded_cyclic.update(canonical_rotation(sequence) for sequence in sequences)
    excluded_natural.add(NATIVE_SEQUENCE)
    excluded_cyclic.add(canonical_rotation(NATIVE_SEQUENCE))

    seed_scores = scorer.score_objective(TARGET, sorted(seed_union), "same-model seed replay")
    seed_replay_rows: List[Dict[str, Any]] = []
    for sequence, score in seed_scores.items():
        sources = sorted(
            source for source, sequences in source_sequences.items() if sequence in sequences
        )
        seed_replay_rows.append(
            {
                **score,
                "seed_sources": json.dumps(sources, ensure_ascii=False),
                "excluded_from_new_release": int(
                    sequence in excluded_natural
                    or canonical_rotation(sequence) in excluded_cyclic
                ),
            }
        )
    atomic_write_csv(out_dir / "3zgc_same_v11_seed_replay.csv", seed_replay_rows)
    atomic_write_csv(out_dir / "3zgc_seed_input_problems.csv", seed_problems)

    beam = v8.select_beam(seed_scores, int(args.beam_width), len(NATIVE_SEQUENCE))
    seen = set(seed_scores)
    strict_objective: Dict[str, Dict[str, Any]] = {
        sequence: row
        for sequence, row in seed_scores.items()
        if int(row["passes_strict_probability"])
    }
    trace_rows: List[Dict[str, Any]] = [
        {
            "round": 0,
            "stage": "same_model_seed_replay",
            "seed_sequences": len(seed_union),
            "newly_scored": len(seed_scores),
            "cumulative_scored": len(seen),
            "strict_objective_hits": len(strict_objective),
            "best_release_floor_probability": max(
                float(row["maximum_probability"]) for row in seed_scores.values()
            ),
        }
    ]

    confirmed_by_sequence: Dict[str, Dict[str, Any]] = {}
    for round_index in range(1, int(args.rounds) + 1):
        provenance = v8.zgc_round_provenance(
            beam,
            round_index,
            int(args.random_offspring_per_round),
            np,
        )
        to_score = sorted(set(provenance) - seen)
        scored = scorer.score_objective(
            TARGET, to_score, f"release-floor beam round {round_index:02d}"
        )
        for sequence, row in scored.items():
            row.update(provenance[sequence])
        atomic_write_csv(
            out_dir / f"3zgc_search_round_{round_index:02d}.csv",
            list(scored.values()),
        )
        seen.update(scored)
        strict_objective.update(
            {
                sequence: row
                for sequence, row in scored.items()
                if int(row["passes_strict_probability"])
            }
        )
        combined = {str(row["sequence"]): dict(row) for row in beam}
        combined.update(scored)
        beam = v8.select_beam(combined, int(args.beam_width), len(NATIVE_SEQUENCE))

        candidates_to_confirm = [
            sequence
            for sequence in strict_objective
            if sequence not in confirmed_by_sequence
            and sequence not in excluded_natural
            and canonical_rotation(sequence) not in excluded_cyclic
        ]
        if candidates_to_confirm:
            full = scorer.score_full(
                TARGET,
                candidates_to_confirm,
                stage=f"round {round_index:02d} strict-hit confirmation",
            )
            for sequence, payload in full.items():
                if not v8.stable_cyclic_methyl_release_gate(payload, sequence):
                    continue
                minimum = json.loads(
                    str(payload["methyl_probability_representation_min"])
                )
                mean = json.loads(str(payload["methyl_probabilities"]))
                floor_maximum, floor_position = actionable_max(sequence, minimum)
                mean_maximum, _mean_position = actionable_max(sequence, mean)
                confirmed_by_sequence[sequence] = {
                    "target_name": TARGET,
                    "candidate_id": f"v12_3zgc_search_{len(confirmed_by_sequence)+1:05d}",
                    "selected_chain": "C",
                    "native_seq": NATIVE_SEQUENCE,
                    "native_length": len(NATIVE_SEQUENCE),
                    "design_length": len(sequence),
                    "temperature": TEMPERATURE,
                    "methyl_threshold": THRESHOLD,
                    "strict_threshold_operator": ">",
                    "candidate_origin": "V12_COMPLETE_SEQUENCE_DIRECTED_SEARCH",
                    "search_stage": strict_objective[sequence]["search_stage"],
                    "release_floor_maximum_probability": floor_maximum,
                    "release_floor_argmax_position_1based": floor_position,
                    "ranking_mean_maximum_probability": mean_maximum,
                    "length_match": 1,
                    "valid_token_gate": 1,
                    "passes_methylation_hard_gate": 1,
                    "eligible_for_manual_structure_review": 1,
                    "prestructure_base_gate_used": 0,
                    "prestructure_rmsd_available": 0,
                    "prestructure_rmsd_rank_used": 0,
                    "rmsd_status": "NOT_AVAILABLE_UNTIL_SHANGGE_RETURNS_STRUCTURES",
                    **payload,
                }

        tentative, _rejected = select_novel_release_rows(
            list(confirmed_by_sequence.values()),
            excluded_natural,
            excluded_cyclic,
            int(args.quota),
        )
        trace_rows.append(
            {
                "round": round_index,
                "stage": f"release_floor_beam_round_{round_index:02d}",
                "generated_unique": len(provenance),
                "newly_scored": len(scored),
                "cumulative_scored": len(seen),
                "strict_objective_hits": len(strict_objective),
                "confirmed_novel_strict_hits": len(confirmed_by_sequence),
                "exact_release_rows_available": len(tentative),
                "best_release_floor_probability": max(
                    float(row["maximum_probability"]) for row in beam
                ),
            }
        )
        atomic_write_csv(out_dir / "3zgc_search_trace.csv", trace_rows)
        atomic_write_json(
            out_dir / "3zgc_search_checkpoint.json",
            {
                "protocol": SEARCH_PROTOCOL,
                "completed_round": round_index,
                "seen_sequence_count": len(seen),
                "strict_objective_hit_count": len(strict_objective),
                "confirmed_novel_strict_hit_count": len(confirmed_by_sequence),
                "beam": beam,
                "model_sha256": sha256_file(model_path),
                "audit_sha256": sha256_file(audit_path),
            },
        )
        print(
            f"[3ZGC] round={round_index}, scored={len(seen):,}, "
            f"confirmed novel strict={len(confirmed_by_sequence)}, "
            f"release-ready={len(tentative)}/{int(args.quota)}",
            flush=True,
        )
        if len(tentative) >= int(args.quota):
            break

    # Keep every novel cyclic identity in rank order for the independent
    # replay.  If one row is numerically inconsistent, the next verified row
    # may fill the quota; search-batch rows never receive grandfather status.
    preliminarily_selected, rejected = select_novel_release_rows(
        list(confirmed_by_sequence.values()),
        excluded_natural,
        excluded_cyclic,
        max(len(confirmed_by_sequence), int(args.quota)),
    )

    # A fresh batch-one scorer is the final authority.  Search-batch output is
    # never copied directly into the handoff pool.
    batch_one = V11ReleaseFloorScorer(
        v8,
        model,
        device,
        native_index,
        selected_chains,
        1,
        torch,
        common,
        reannotator,
    )
    independently_confirmed: List[Dict[str, Any]] = []
    replay_failures: List[Dict[str, Any]] = []
    for source in preliminarily_selected:
        if len(independently_confirmed) >= int(args.quota):
            break
        sequence = str(source["design_natural_seq"]).upper()
        payload = batch_one.score_full(
            TARGET,
            [sequence],
            stage="independent batch-one final confirmation",
            show_progress=False,
        )[sequence]
        if not v8.stable_cyclic_methyl_release_gate(payload, sequence):
            replay_failures.append(
                {"sequence": sequence, "problem": "batch_one_release_gate_failed"}
            )
            continue
        replay_minimum = json.loads(
            str(payload["methyl_probability_representation_min"])
        )
        replay_floor, replay_position = actionable_max(sequence, replay_minimum)
        delta = abs(
            replay_floor - float(source["release_floor_maximum_probability"])
        )
        if not math.isfinite(delta) or delta > PROBABILITY_ATOL:
            replay_failures.append(
                {
                    "sequence": sequence,
                    "problem": "batch_one_probability_mismatch",
                    "absolute_difference": delta,
                }
            )
            continue
        row = dict(source)
        row.update(payload)
        row.update(
            {
                "release_floor_maximum_probability": replay_floor,
                "release_floor_argmax_position_1based": replay_position,
                "independent_batch_one_replay": "PASS",
                "independent_batch_one_probability_absolute_difference": delta,
                "v12_release_rank": len(independently_confirmed) + 1,
                "candidate_id": f"v12_3zgc_{len(independently_confirmed)+1:03d}",
            }
        )
        independently_confirmed.append(row)

    quality_checks = {
        "promoted_v11_checkpoint_and_full_grid_audit_are_pinned": True,
        "search_objective_is_representation_minimum_not_mean_only": True,
        "methyl_threshold_remains_strictly_greater_than_0_6": True,
        "no_prestructure_base_gate_was_used": True,
        "no_predicted_or_observed_rmsd_was_used": True,
        "all_released_rows_are_novel_exact_and_forward_cyclic_identities": (
            len({str(row["design_natural_seq"]) for row in independently_confirmed})
            == len(independently_confirmed)
            and len({str(row["forward_cyclic_identity"]) for row in independently_confirmed})
            == len(independently_confirmed)
        ),
        "all_released_rows_pass_independent_batch_one_replay": (
            len(independently_confirmed) == int(args.quota)
            and all(
                row.get("independent_batch_one_replay") == "PASS"
                for row in independently_confirmed
            )
        ),
        "exactly_100_3zgc_rows_are_released": (
            len(independently_confirmed) == int(args.quota) == QUOTA
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    diagnostic_rows = sorted(
        list(confirmed_by_sequence.values()), key=release_rank
    )
    atomic_write_csv(out_dir / "3zgc_confirmed_strict_diagnostics.csv", diagnostic_rows)
    atomic_write_csv(out_dir / "3zgc_rejected_or_surplus_diagnostics.csv", rejected)
    atomic_write_csv(out_dir / "3zgc_batch_one_failures.csv", replay_failures)
    if quality_gate == "PASS":
        atomic_write_csv(out_dir / "3zgc_exact_100_methylated.csv", independently_confirmed)

    output_artifacts: Dict[str, Any] = {
        "same_v11_seed_replay": {
            "path": str(out_dir / "3zgc_same_v11_seed_replay.csv"),
            "sha256": sha256_file(out_dir / "3zgc_same_v11_seed_replay.csv"),
        },
        "search_trace": {
            "path": str(out_dir / "3zgc_search_trace.csv"),
            "sha256": sha256_file(out_dir / "3zgc_search_trace.csv"),
        },
        "confirmed_strict_diagnostics": {
            "path": str(out_dir / "3zgc_confirmed_strict_diagnostics.csv"),
            "sha256": sha256_file(out_dir / "3zgc_confirmed_strict_diagnostics.csv"),
        },
    }
    if quality_gate == "PASS":
        output_artifacts["exact_100_release"] = {
            "path": str(out_dir / "3zgc_exact_100_methylated.csv"),
            "sha256": sha256_file(out_dir / "3zgc_exact_100_methylated.csv"),
        }

    manifest = {
        "quality_gate": quality_gate,
        "release_status": (
            "AUTHORIZED_3ZGC_EXACT_100_METHYLATION_ONLY_PRESTRUCTURE_ROWS"
            if quality_gate == "PASS"
            else "BLOCKED_NO_3ZGC_HANDOFF_CREATED"
        ),
        "protocol": SEARCH_PROTOCOL,
        "target": TARGET,
        "quota": int(args.quota),
        "temperature": TEMPERATURE,
        "threshold": THRESHOLD,
        "strict_threshold_operator": ">",
        "selection_scope": "METHYLATION_ONLY_BEFORE_STRUCTURE_PREDICTION",
        "base_score_policy": "NOT_A_RELEASE_GATE_AND_NOT_COMPUTED",
        "rmsd_policy": "UNAVAILABLE_UNTIL_SHANGGE_RETURNS_STRUCTURES",
        "search": {
            "rounds_requested": int(args.rounds),
            "rounds_completed": len(trace_rows) - 1,
            "beam_width": int(args.beam_width),
            "random_offspring_per_round": int(args.random_offspring_per_round),
            "full_sequences_scored": len(seen),
            "strict_objective_hits": len(strict_objective),
            "confirmed_novel_strict_hits": len(confirmed_by_sequence),
            "released_rows": len(independently_confirmed),
            "important_interpretation": (
                "scored sequence variants are internal model evaluations, not generated structures"
            ),
        },
        "seed_source_counts": {
            source: len(sequences) for source, sequences in source_sequences.items()
        },
        "quality_checks": quality_checks,
        "artifacts": output_artifacts,
        "inputs": {
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "representation_audit": {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
            },
            "v11_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "generation_manifest": {
                "path": str(generation_manifest_path),
                "sha256": sha256_file(generation_manifest_path),
                "false_checks": sorted(generation_false),
            },
            "seed_csvs": source_records,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        },
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "checkpoint_protocol": contract["checkpoint_metadata"].get("protocol"),
    }
    atomic_write_json(out_dir / "3zgc_methyl_only_search_manifest.json", manifest)
    print("===== V12 3ZGC METHYL-ONLY SEARCH COMPLETE =====", flush=True)
    print(f"Full sequences scored: {len(seen):,}", flush=True)
    print(f"Strict novel candidates: {len(confirmed_by_sequence):,}", flush=True)
    print(f"Independent exact release: {len(independently_confirmed)}/{QUOTA}", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError(
            "V12 3ZGC search completed without an exact-100 release: "
            + ", ".join(failed)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--representation-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--generation-dir", default=str(DEFAULT_GENERATION))
    parser.add_argument("--v11-plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--best-csv", default=str(DEFAULT_BEST))
    parser.add_argument("--historical-csv", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--prior-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--rmsd-history-csv", default=str(DEFAULT_RMSD_HISTORY))
    parser.add_argument("--seed-csv", action="append", default=[])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--quota", type=int, default=QUOTA)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=512)
    parser.add_argument("--random-offspring-per-round", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.quota) != QUOTA:
        raise ValueError("V12 3ZGC release quota is fixed to exactly 100")
    if any(
        int(value) <= 0
        for value in (
            args.rounds,
            args.beam_width,
            args.random_offspring_per_round,
            args.batch_size,
        )
    ):
        raise ValueError("Search sizes must be positive")
    run(args)


if __name__ == "__main__":
    main()
