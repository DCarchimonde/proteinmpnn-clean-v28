#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reannotate the preserved V6 natural-sequence pool with an audited expert model.

V6 changed all twenty expert heads even though the provenance repair changed
only Ser labels.  This recovery does not sample another base sequence.  It
hash-pins and retains all 31,500 receptor-conditioned V6 natural sequences and
their base-model path statistics, then scores each unique target/natural pair
once with the supplied cyclic-representation ensemble in the peptide-only context.
Defaults reproduce the original Ser-only V7 stage; explicit protocol/scope
arguments authorize the source-scoped V8 reuse without duplicating scoring code.

Formal target abstention is deliberately not supported.  The quality gate
requires at least one strict-threshold, novelty-filtered methylated candidate
for every one of the seventeen targets.  Structure quotas remain diagnostics;
no structure handoff is created by this stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
COMMON_PATH = REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"
MODEL_UTILS_PATH = REPO_ROOT / "model_utils.py"
NMETHYL_CONFIG_PATH = REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"
GENERATOR_PATH = (
    REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
)
V6_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_cyclic_representation_v6"
V7_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_serine_only_cyclic_v7"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_representation_v6.json")
DEFAULT_MODEL = V7_ROOT / "model" / "frankenstein_v28_serine_only_qc.pt"
DEFAULT_EXPERT_MANIFEST = V7_ROOT / "model" / "expert_heads_retrain_manifest.json"
DEFAULT_REPRESENTATION_AUDIT = (
    V7_ROOT / "representation_audit" / "cyclic_representation_audit.json"
)
DEFAULT_SOURCE_RUN = V6_ROOT / "generation"
DEFAULT_OUT = V7_ROOT / "generation"
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_OLD = (
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

V7_EXPERT_PROTOCOL = (
    "canonical_clean_v28_serine_only_corrected_labels_"
    "cyclic_representation_augmented_v7"
)
V7_REPRESENTATION_AUDIT_PROTOCOL = (
    "cyclic_representation_equivariance_heldout_gate_v2_serine_only"
)
V7_REPRESENTATION_AUTHORIZATION = (
    "SERINE_ONLY_REPAIR_VALIDATED_FOR_ISOLATED_V7_REANNOTATION"
)
V7_GENERATION_PROTOCOL = (
    "temperature_0.5_serine_only_cyclic_v7_reannotation_of_preserved_v6_pool"
)
ANNOTATION_MODE = (
    "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
)
ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
SAMPLING_CONTEXT = "native_complex_longest_receptor_visible"
EXPECTED_SOURCE_RAW_ROWS = 31_500
EXPECTED_SOURCE_TARGETS = 17
EXPECTED_HISTORICAL_ROWS = 4_115
EXPECTED_SOURCE_ALL_SHA256 = (
    "1ab4791c09a1b2428b1a84894d13bb8c4049ba580df05bebd93c263a2e4e634c"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "067a22a2175c97cf483e64967168eefc676389e302c9acc79a66c70e8290711f"
)
VALID_NATURAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_generator_module() -> Any:
    spec = importlib.util.spec_from_file_location("serine_v7_base_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator module: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path is equal to or contains the other."""

    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                result.append(field)
    return result


def chunks(values: Sequence[str], size: int) -> Sequence[Sequence[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def peptide_only_record(
    source: Mapping[str, Any], selected_chain: str, natural_sequence: str
) -> Dict[str, Any]:
    native_sequence = str(source.get(f"seq_chain_{selected_chain}", ""))
    if not native_sequence or len(native_sequence) != len(natural_sequence):
        raise RuntimeError(
            f"Peptide geometry/sequence length mismatch for chain {selected_chain}"
        )
    record: Dict[str, Any] = {
        "name": str(source.get("name", "peptide_only")),
        "seq": natural_sequence,
        f"seq_chain_{selected_chain}": natural_sequence,
        "masked_list": [selected_chain],
        "visible_list": [],
        "num_of_chains": 1,
    }
    for atom_name in ("N", "CA", "C", "O"):
        key = f"{atom_name}_chain_{selected_chain}"
        if key not in source:
            raise RuntimeError(f"Missing peptide coordinate: {key}")
        record[key] = copy.deepcopy(source[key])
    return record


def annotation_payload(
    natural_sequence: str,
    representation: Mapping[str, Any],
    row_index: int,
    threshold: float,
    natural_alphabet: str,
    extended_alphabet: str,
    natural_to_methyl: Mapping[int, int],
) -> Dict[str, Any]:
    def vector(name: str) -> List[float]:
        return [
            round(float(value), 8)
            for value in representation[name][row_index].detach().cpu().tolist()
        ]

    probability = vector("mean")
    order_std = vector("decoder_order_std_mean")
    representation_std = vector("representation_std")
    representation_min = vector("representation_min")
    representation_max = vector("representation_max")
    representation_span = vector("representation_span")
    if not all(
        len(values) == len(natural_sequence)
        for values in (
            probability,
            order_std,
            representation_std,
            representation_min,
            representation_max,
            representation_span,
        )
    ):
        raise RuntimeError("Cyclic reannotation vector length mismatch")
    if not all(
        math.isfinite(value)
        for values in (
            probability,
            order_std,
            representation_std,
            representation_min,
            representation_max,
            representation_span,
        )
        for value in values
    ):
        raise RuntimeError("Cyclic reannotation contains a non-finite value")
    if any(
        value < 0.0 or value > 1.0
        for values in (probability, representation_min, representation_max)
        for value in values
    ):
        raise RuntimeError("Cyclic reannotation probability is outside [0, 1]")
    if any(
        value < 0.0
        for values in (order_std, representation_std, representation_span)
        for value in values
    ):
        raise RuntimeError("Cyclic reannotation dispersion is negative")
    if any(
        minimum > mean + 1e-7
        or mean > maximum + 1e-7
        or abs((maximum - minimum) - span) > 2e-6
        for mean, minimum, maximum, span in zip(
            probability,
            representation_min,
            representation_max,
            representation_span,
        )
    ):
        raise RuntimeError("Cyclic reannotation min/mean/max/span is inconsistent")

    output_tokens: List[str] = []
    for token, value in zip(natural_sequence, probability):
        natural_index = natural_alphabet.index(token)
        methyl_index = natural_to_methyl.get(natural_index)
        output_tokens.append(
            extended_alphabet[int(methyl_index)]
            if methyl_index is not None and value > threshold
            else token
        )
    design_sequence = "".join(output_tokens)
    methyl_positions = [
        index for index, token in enumerate(design_sequence, start=1) if token.islower()
    ]
    methyl_site_probabilities = [probability[index - 1] for index in methyl_positions]
    disagreement_positions = [
        index
        for index, (minimum, maximum) in enumerate(
            zip(representation_min, representation_max), start=1
        )
        if minimum <= threshold < maximum
    ]
    representation_count = int(
        round(
            max(
                representation["representation_count"][row_index]
                .detach()
                .cpu()
                .tolist()
            )
        )
    )
    return {
        "design_seq": design_sequence,
        "design_natural_seq": natural_sequence,
        "design_methyl_count": len(methyl_positions),
        "design_methyl_rate": len(methyl_positions) / len(natural_sequence),
        "methyl_positions_1based": json.dumps(methyl_positions),
        "methyl_probability_min": min(probability),
        "methyl_probability_mean": sum(probability) / len(probability),
        "methyl_probability_max": max(probability),
        "methyl_site_probability_min": (
            min(methyl_site_probabilities) if methyl_site_probabilities else ""
        ),
        "methyl_site_probability_mean": (
            sum(methyl_site_probabilities) / len(methyl_site_probabilities)
            if methyl_site_probabilities
            else ""
        ),
        "methyl_site_probability_max": (
            max(methyl_site_probabilities) if methyl_site_probabilities else ""
        ),
        "methyl_probabilities": json.dumps(probability),
        "methyl_probability_order_std": json.dumps(order_std),
        "methyl_probability_order_std_max": max(order_std),
        "methyl_probability_representation_std": json.dumps(representation_std),
        "methyl_probability_representation_std_max": max(representation_std),
        "methyl_probability_representation_min": json.dumps(representation_min),
        "methyl_probability_representation_max": json.dumps(representation_max),
        "methyl_probability_representation_span": json.dumps(representation_span),
        "methyl_probability_representation_span_max": max(representation_span),
        "representation_threshold_disagreement_positions_1based": json.dumps(
            disagreement_positions
        ),
        "representation_threshold_disagreement_count": len(disagreement_positions),
        "sampling_path_methyl_probabilities": "",
        "sampling_path_annotation_status": (
            "NOT_REUSED_V6_EXPERT_PATH_PROBABILITIES_BASE_SEQUENCE_ONLY_RETAINED"
        ),
        "annotation_mode": ANNOTATION_MODE,
        "annotation_context_policy": ANNOTATION_CONTEXT,
        "annotation_visible_receptor_chains": 0,
        "sampling_context_policy": SAMPLING_CONTEXT,
        "annotation_order_ensemble_size": len(natural_sequence),
        "annotation_decoder_order_ensemble_size": len(natural_sequence),
        "annotation_representation_ensemble_size": representation_count,
    }


def validate_source_rows(
    rows: Sequence[Mapping[str, str]],
    target_names: Sequence[str],
    expected_raw_rows: int = EXPECTED_SOURCE_RAW_ROWS,
    expected_target_count: int = EXPECTED_SOURCE_TARGETS,
) -> Dict[str, Any]:
    if len(rows) != expected_raw_rows:
        raise RuntimeError(
            f"V6 source row count changed: {len(rows)} != {expected_raw_rows}"
        )
    expected_targets = {str(value).upper() for value in target_names}
    observed_targets = {str(row.get("target_name", "")).upper() for row in rows}
    if observed_targets != expected_targets or len(observed_targets) != expected_target_count:
        raise RuntimeError("V6 source target set is not the frozen 17-target pool")
    candidate_ids = [str(row.get("candidate_id", "")) for row in rows]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("V6 source candidate IDs are empty or duplicated")

    chains: MutableMapping[str, set[str]] = defaultdict(set)
    unique_pairs: set[Tuple[str, str]] = set()
    for row in rows:
        target = str(row["target_name"]).upper()
        natural = str(row.get("design_natural_seq", "")).upper()
        if (
            not natural
            or not set(natural) <= VALID_NATURAL_AA
            or natural != str(row.get("design_seq", "")).upper()
            or int(row.get("design_length", -1)) != len(natural)
            or int(row.get("native_length", -1)) != len(natural)
        ):
            raise RuntimeError(f"Invalid V6 natural row: {row.get('candidate_id')}")
        chains[target].add(str(row.get("selected_chain", "")))
        unique_pairs.add((target, natural))
    invalid_chains = {
        target: sorted(values)
        for target, values in chains.items()
        if len(values) != 1 or not next(iter(values), "")
    }
    if invalid_chains:
        raise RuntimeError(f"V6 selected peptide chains are inconsistent: {invalid_chains}")
    return {
        "selected_chain_by_target": {
            target: next(iter(values)) for target, values in chains.items()
        },
        "unique_target_natural_sequence_groups": len(unique_pairs),
        "repeated_raw_rows": len(rows) - len(unique_pairs),
    }


def score_unique_sequences(
    model: Any,
    device: Any,
    native_index: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, str]],
    selected_chain_by_target: Mapping[str, str],
    temperature: float,
    threshold: float,
    batch_size: int,
    torch_module: Any,
    common: Mapping[str, Any],
    run_label: str = "SERINE-ONLY V7",
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    by_target: MutableMapping[str, set[str]] = defaultdict(set)
    for row in rows:
        by_target[str(row["target_name"]).upper()].add(
            str(row["design_natural_seq"]).upper()
        )

    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    natural_alphabet = str(common["NATURAL_AA_ALPHABET"])
    for target in sorted(by_target):
        sequences = sorted(by_target[target])
        selected_chain = selected_chain_by_target[target]
        geometry_record = peptide_only_record(
            native_index[target], selected_chain, sequences[0]
        )
        packed = common["featurize_records"](
            [geometry_record], device=device, eval_chains="masked"
        )
        if packed is None:
            raise RuntimeError(f"Peptide-only feature construction failed for {target}")
        tensors, metas = packed
        X, _S, mask, chain_M, residue_idx, chain_encoding_all = tensors[:6]
        if int(metas[0]["selected_length"]) != len(sequences[0]):
            raise RuntimeError(f"Peptide-only annotation context leaked for {target}")

        for sequence_batch in chunks(sequences, batch_size):
            current_batch = len(sequence_batch)
            S_natural = torch_module.tensor(
                [
                    [natural_alphabet.index(token) for token in sequence]
                    for sequence in sequence_batch
                ],
                device=device,
                dtype=torch_module.long,
            )
            with torch_module.no_grad():
                representation = common[
                    "cyclic_representation_known_sequence_methyl_probabilities"
                ](
                    model=model,
                    X=X.repeat(current_batch, 1, 1, 1),
                    S_natural=S_natural,
                    mask=mask.repeat(current_batch, 1),
                    chain_M=chain_M.repeat(current_batch, 1),
                    residue_idx=residue_idx.repeat(current_batch, 1),
                    chain_encoding_all=chain_encoding_all.repeat(current_batch, 1),
                    temperature=temperature,
                )
            for index, sequence in enumerate(sequence_batch):
                result[(target, sequence)] = annotation_payload(
                    sequence,
                    representation,
                    index,
                    threshold,
                    natural_alphabet,
                    str(common["EXTENDED_AA_ALPHABET"]),
                    common["NAT_TO_METHYL_ABS"],
                )
        print(
            f"[{target}] {run_label} annotations: {len(sequences)} unique natural sequences",
            flush=True,
        )
    return result


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(f"{args.run_label} reannotation requires numpy and torch") from exc
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = load_generator_module()
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

    plan_path = Path(args.plan).resolve()
    model_path = Path(args.model_path).resolve()
    expert_manifest_path = Path(args.expert_manifest).resolve()
    representation_audit_path = Path(args.representation_audit_json).resolve()
    source_run = Path(args.source_run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    native_path = Path(args.native_jsonl).resolve()
    old_path = Path(args.old_designs_csv).resolve()
    prior_path = Path(args.prior_designs_csv).resolve()
    immutable_inputs = (
        plan_path,
        model_path,
        expert_manifest_path,
        representation_audit_path,
        source_run,
        native_path,
        old_path,
        prior_path,
        SCRIPT_PATH,
        GENERATOR_PATH,
        COMMON_PATH,
        MODEL_UTILS_PATH,
        NMETHYL_CONFIG_PATH,
    )
    overlapping = [path for path in immutable_inputs if paths_overlap(out_dir, path)]
    if overlapping:
        raise ValueError(
            "Reannotation output overlaps an immutable input: "
            + ", ".join(str(path) for path in overlapping)
        )
    source_paths = {
        "all": source_run / "all_candidates.csv",
        "manifest": source_run / "generation_manifest.json",
        "target_manifest": source_run / "target_manifest.csv",
    }
    for required in (
        plan_path,
        model_path,
        expert_manifest_path,
        representation_audit_path,
        native_path,
        old_path,
        prior_path,
        GENERATOR_PATH,
        COMMON_PATH,
        MODEL_UTILS_PATH,
        NMETHYL_CONFIG_PATH,
        *source_paths.values(),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.run_label} generation output already exists: {out_dir}"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = read_json(plan_path)
    validated = generator.validate_plan(plan)
    source_manifest = read_json(source_paths["manifest"])
    source_rows = generator.read_csv(source_paths["all"])
    source_all_sha256 = sha256_file(source_paths["all"])
    source_manifest_sha256 = sha256_file(source_paths["manifest"])
    if source_all_sha256 != str(args.expected_source_all_sha256):
        raise RuntimeError(
            "V6 all_candidates.csv differs from the audited uploaded pool: "
            + source_all_sha256
        )
    if source_manifest_sha256 != str(args.expected_source_manifest_sha256):
        raise RuntimeError(
            "V6 generation_manifest.json differs from the audited uploaded pool: "
            + source_manifest_sha256
        )
    if not (
        str(source_manifest.get("quality_gate", "")) == "PASS"
        and sorted(
            str(value).upper()
            for value in source_manifest.get("targets_formally_abstained", [])
        )
        == ["3ZGC"]
        and int(source_manifest.get("raw_candidates_generated", -1))
        == EXPECTED_SOURCE_RAW_ROWS
    ):
        raise RuntimeError("V6 source is not the audited 31,500-row abstention result")
    source_validation = validate_source_rows(source_rows, validated["target_names"])

    expert_manifest = read_json(expert_manifest_path)
    representation_audit = read_json(representation_audit_path)
    checkpoint_sha256 = sha256_file(model_path)
    expected_active_tokens = json.loads(str(args.expected_active_expert_tokens_json))
    if not isinstance(expected_active_tokens, list) or not all(
        isinstance(value, str) for value in expected_active_tokens
    ):
        raise ValueError("--expected-active-expert-tokens-json must be a JSON list")
    if not (
        expert_manifest.get("quality_gate") == "PASS"
        and expert_manifest.get("protocol") == args.expected_expert_protocol
        and expert_manifest.get("expert_scope") == args.expected_expert_scope
        and expert_manifest.get("active_expert_tokens") == expected_active_tokens
        and expert_manifest.get("checkpoint_artifact_sha256") == checkpoint_sha256
    ):
        raise RuntimeError(
            "Expert model manifest is absent, failed, stale, or wrong-scope"
        )
    if not (
        representation_audit.get("quality_gate") == "PASS"
        and representation_audit.get("protocol")
        == args.expected_representation_protocol
        and representation_audit.get("release_authorization")
        == args.expected_representation_authorization
        and representation_audit.get("model_sha256") == checkpoint_sha256
        and representation_audit.get("plan_sha256") == sha256_file(plan_path)
    ):
        raise RuntimeError("Cyclic-representation audit is absent, failed, or stale")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU scoring requires --allow-cpu")
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

    native_rows = generator.read_jsonl(native_path)
    native_names = [
        str(
            row.get("name")
            or row.get("pdb")
            or row.get("pdb_id")
            or row.get("id")
            or ""
        ).upper()
        for row in native_rows
    ]
    expected_native_names = {
        str(value).upper() for value in validated["target_names"]
    }
    if (
        len(native_rows) != EXPECTED_SOURCE_TARGETS
        or any(not name for name in native_names)
        or len(set(native_names)) != len(native_names)
        or set(native_names) != expected_native_names
    ):
        raise RuntimeError(
            "Native JSONL must contain exactly one named record for each planned target"
        )
    native_index = dict(zip(native_names, native_rows))

    print(f"Loading promoted {args.run_label} checkpoint: {model_path}", flush=True)
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
    annotation_by_key = score_unique_sequences(
        model,
        device,
        native_index,
        source_rows,
        source_validation["selected_chain_by_target"],
        float(plan["temperature"]),
        float(plan["methyl_threshold"]),
        int(args.batch_size),
        torch,
        common,
        args.run_label,
    )

    raw_rows: List[Dict[str, Any]] = []
    for source in source_rows:
        target = str(source["target_name"]).upper()
        natural = str(source["design_natural_seq"]).upper()
        row = dict(source)
        row["source_v6_design_seq"] = str(source.get("design_seq", ""))
        row["source_v6_methyl_probabilities"] = str(
            source.get("methyl_probabilities", "")
        )
        row["source_v6_sampling_path_methyl_probabilities"] = str(
            source.get("sampling_path_methyl_probabilities", "")
        )
        row.update(annotation_by_key[(target, natural)])
        row[f"{args.summary_score_label}_reannotation_source"] = (
            "PRESERVED_V6_NATURAL_SEQUENCE"
        )
        row["length_match"] = int(len(natural) == int(row["native_length"]))
        row["valid_token_gate"] = 1
        raw_rows.append(row)

    old_rows = generator.read_csv(old_path)
    old_targets = {
        str(row.get("target_name", "")).strip().upper() for row in old_rows
    }
    if (
        len(old_rows) != EXPECTED_HISTORICAL_ROWS
        or old_targets != {str(value).upper() for value in validated["target_names"]}
        or any(
            not str(row.get("target_name", "")).strip()
            or not str(row.get("design_seq", "")).strip()
            for row in old_rows
        )
    ):
        raise RuntimeError("Historical 4,115-row design exclusion set changed")
    old_exact, old_natural = generator.old_design_keys(old_path)
    prior_rows, prior_exact, prior_natural = generator.validate_prior_handoff(prior_path)
    unique_rows = generator.aggregate_unique_candidates(
        raw_rows, old_exact, old_natural, prior_exact, prior_natural
    )
    eligible_rows = [
        row
        for row in unique_rows
        if int(row["eligible_for_new_permeability_screen"])
    ]
    for row in unique_rows:
        row["permeability_id"] = ""

    raw_fields = union_fields(raw_rows)
    unique_fields = union_fields(unique_rows)
    generator.atomic_write_csv(out_dir / "all_candidates.csv", raw_rows, raw_fields)
    generator.atomic_write_csv(
        out_dir / "unique_candidates.csv", unique_rows, unique_fields
    )
    generator.atomic_write_csv(
        out_dir / "methylated_new_candidates.csv", eligible_rows, unique_fields
    )

    plan_by_target = {
        str(item["target_name"]).upper(): dict(item) for item in validated["targets"]
    }
    raw_count = Counter(str(row["target_name"]).upper() for row in raw_rows)
    unique_count = Counter(str(row["target_name"]).upper() for row in unique_rows)
    eligible_count = Counter(str(row["target_name"]).upper() for row in eligible_rows)
    summary_rows: List[Dict[str, Any]] = []
    target_manifest_rows: List[Dict[str, Any]] = []
    for target in validated["target_names"]:
        target_unique = [
            row for row in unique_rows if str(row["target_name"]).upper() == target
        ]
        target_source_rows = [
            row for row in source_rows if str(row["target_name"]).upper() == target
        ]
        quota = int(plan_by_target[target]["structure_quota"])
        coverage = int(eligible_count[target] > 0)
        methyl_residue_counts: Counter[str] = Counter(
            token.upper()
            for row in target_unique
            for token in str(row["design_seq"])
            if token.islower()
        )
        summary_rows.append(
            {
                "target_name": target,
                "raw_natural_sequences_retained": raw_count[target],
                "unique_reannotated": unique_count[target],
                "source_v6_raw_rows_with_methyl_call": sum(
                    int(any(token.islower() for token in str(row["design_seq"])))
                    for row in target_source_rows
                ),
                "source_v6_maximum_recorded_probability": max(
                    (float(row.get("methyl_probability_max", 0.0)) for row in target_source_rows),
                    default=0.0,
                ),
                "unique_methylated": sum(
                    int(row["passes_methylation_hard_gate"])
                    for row in target_unique
                ),
                "novel_methylated_candidates": eligible_count[target],
                f"{args.summary_score_label}_maximum_recorded_probability": max(
                    (float(row.get("methyl_probability_max", 0.0)) for row in target_unique),
                    default=0.0,
                ),
                f"{args.summary_score_label}_methyl_residue_counts": json.dumps(
                    dict(sorted(methyl_residue_counts.items()))
                ),
                "minimum_signature_candidate_required": 1,
                "has_signature_candidate": coverage,
                "planned_structure_quota": quota,
                "meets_planned_structure_quota": int(eligible_count[target] >= quota),
            }
        )
        target_manifest_rows.append(
            {
                "target_name": target,
                "selected_chain": source_validation["selected_chain_by_target"][target],
                "raw_rows_retained": raw_count[target],
                "unique_rows": unique_count[target],
                "novel_methylated_candidates": eligible_count[target],
                "target_release_status": (
                    "CANDIDATE_FOUND_PENDING_MANUAL_AND_STRUCTURE_REVIEW"
                    if coverage
                    else "BLOCKED_NO_SIGNATURE_CANDIDATE"
                ),
                "formal_abstention": 0,
            }
        )
    generator.atomic_write_csv(
        out_dir / "generation_summary_by_target.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    generator.atomic_write_csv(
        out_dir / "target_manifest.csv",
        target_manifest_rows,
        list(target_manifest_rows[0]),
    )

    annotation_audit = generator.audit_annotation_stability(raw_rows, eligible_rows)
    uncovered_targets = [
        row["target_name"] for row in summary_rows if not int(row["has_signature_candidate"])
    ]
    targets_below_quota = [
        row["target_name"]
        for row in summary_rows
        if not int(row["meets_planned_structure_quota"])
    ]
    quality_checks = {
        **dict(annotation_audit["quality_checks"]),
        "source_v6_pool_hash_count_and_target_set_are_pinned": True,
        "expert_checkpoint_scope_protocol_and_representation_audit_pass": True,
        "every_unique_target_natural_sequence_scored_exactly_once": (
            len(annotation_by_key)
            == int(source_validation["unique_target_natural_sequence_groups"])
        ),
        "all_31500_source_natural_rows_are_retained": (
            len(raw_rows) == EXPECTED_SOURCE_RAW_ROWS
        ),
        "no_formal_target_abstention_is_used": True,
        "every_target_has_at_least_one_novel_methylated_signature_candidate": (
            not uncovered_targets
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    noncoverage_checks = {
        name: passed
        for name, passed in quality_checks.items()
        if name
        != "every_target_has_at_least_one_novel_methylated_signature_candidate"
    }
    recovery_eligible = bool(
        args.permit_missing_targets_for_recovery
        and uncovered_targets
        and all(noncoverage_checks.values())
        and not quality_checks[
            "every_target_has_at_least_one_novel_methylated_signature_candidate"
        ]
    )
    artifacts = {
        name: {
            "path": str(out_dir / filename),
            "sha256": sha256_file(out_dir / filename),
        }
        for name, filename in {
            "all": "all_candidates.csv",
            "unique": "unique_candidates.csv",
            "eligible": "methylated_new_candidates.csv",
            "target_manifest": "target_manifest.csv",
            "target_summary": "generation_summary_by_target.csv",
        }.items()
    }
    manifest = {
        "quality_gate": quality_gate,
        "release_status": (
            "READY_FOR_MANUAL_SCIENTIFIC_REVIEW_NO_STRUCTURE_HANDOFF"
            if quality_gate == "PASS"
            else (
                "BLOCKED_BASELINE_VALID_DIRECTED_RECOVERY_REQUIRED"
                if recovery_eligible
                else "BLOCKED_MODEL_OR_INTEGRITY_FAILURE_DO_NOT_SEARCH"
            )
        ),
        "quality_checks": quality_checks,
        "protocol": args.output_protocol,
        "recovery_mode": str(args.recovery_mode),
        "scientific_reason": str(args.scientific_reason),
        "temperature": float(plan["temperature"]),
        "methyl_threshold": float(plan["methyl_threshold"]),
        "strict_threshold_operator": ">",
        "device": str(device),
        "scoring_batch_size": int(args.batch_size),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "deterministic_runtime": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.version() is not None
                else None
            ),
        },
        "model_path": str(model_path),
        "model_sha256": checkpoint_sha256,
        "reannotator_program_sha256": sha256_file(SCRIPT_PATH),
        "generator_program_sha256": sha256_file(GENERATOR_PATH),
        "common_program_sha256": sha256_file(COMMON_PATH),
        "model_utils_program_sha256": sha256_file(MODEL_UTILS_PATH),
        "nmethyl_config_program_sha256": sha256_file(NMETHYL_CONFIG_PATH),
        "model_expert_qc_protocol": args.expected_expert_protocol,
        "expert_scope": args.expected_expert_scope,
        "summary_score_label": args.summary_score_label,
        "expert_manifest": str(expert_manifest_path),
        "expert_manifest_sha256": sha256_file(expert_manifest_path),
        "cyclic_representation_heldout_audit": {
            "path": str(representation_audit_path),
            "sha256": sha256_file(representation_audit_path),
            "quality_gate": representation_audit["quality_gate"],
            "protocol": representation_audit["protocol"],
            "release_authorization": representation_audit["release_authorization"],
            "model_sha256": representation_audit["model_sha256"],
            "plan_sha256": representation_audit["plan_sha256"],
        },
        "source_v6_run_dir": str(source_run),
        "source_v6_all_candidates_sha256": source_all_sha256,
        "source_v6_generation_manifest_sha256": source_manifest_sha256,
        "source_v6_formal_abstention_was_not_carried_forward": ["3ZGC"],
        "source_v6_natural_rows_retained": len(raw_rows),
        "unique_target_natural_sequences_rescored": len(annotation_by_key),
        "unique_candidates": len(unique_rows),
        "new_methylated_candidates_for_structure_review": len(eligible_rows),
        "new_methylated_candidates_for_permeability": len(eligible_rows),
        "raw_candidates_generated": len(raw_rows),
        "raw_candidates_expected": EXPECTED_SOURCE_RAW_ROWS,
        "expected_target_count": EXPECTED_SOURCE_TARGETS,
        "targets_with_signature_candidate": EXPECTED_SOURCE_TARGETS - len(uncovered_targets),
        "targets_without_signature_candidate": uncovered_targets,
        "directed_recovery_eligible": recovery_eligible,
        "targets_below_planned_structure_quota_diagnostic": targets_below_quota,
        "targets_formally_abstained": [],
        "effective_structure_target_count": EXPECTED_SOURCE_TARGETS - len(uncovered_targets),
        "workflow_order": "MODEL_REPAIR_THEN_MANUAL_REVIEW_THEN_STRUCTURE_THEN_PERMEABILITY",
        "structure_handoff_status": "NOT_CREATED_PENDING_MANUAL_SCIENTIFIC_REVIEW",
        "permeability_status": "DEFERRED_UNTIL_RETURNED_STRUCTURES_PASS_GATE",
        "permeability_input_rows": 0,
        "annotation_mode": ANNOTATION_MODE,
        "annotation_context_policy": ANNOTATION_CONTEXT,
        "annotation_visible_receptor_chains": 0,
        "sampling_context_policy": SAMPLING_CONTEXT,
        "base_sampling_reused": True,
        "expert_sampling_path_probabilities_reused": False,
        "annotation_stability_audit": annotation_audit,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "plan_target_count": len(validated["target_names"]),
        "native_jsonl": str(native_path),
        "native_jsonl_sha256": sha256_file(native_path),
        "native_jsonl_records": len(native_rows),
        "historical_design_csv": str(old_path),
        "historical_design_csv_sha256": sha256_file(old_path),
        "historical_design_rows": len(old_rows),
        "prior_handoff_csv": str(prior_path),
        "prior_handoff_csv_sha256": sha256_file(prior_path),
        "prior_handoff_rows": len(prior_rows),
        "candidate_artifacts": artifacts,
    }
    generator.atomic_write_json(out_dir / "generation_manifest.json", manifest)

    print(f"===== {args.run_label} REANNOTATION COMPLETE =====", flush=True)
    print(f"V6 natural rows retained: {len(raw_rows)}", flush=True)
    print(f"Unique target/natural sequences rescored: {len(annotation_by_key)}", flush=True)
    print(f"Novel methylated candidates: {len(eligible_rows)}", flush=True)
    print(
        f"Target coverage: {EXPECTED_SOURCE_TARGETS - len(uncovered_targets)}/"
        f"{EXPECTED_SOURCE_TARGETS}",
        flush=True,
    )
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS" and not recovery_eligible:
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError(
            f"{args.run_label} reannotation is blocked; no abstention or release was created. "
            "Failed checks: " + ", ".join(failed)
        )
    if recovery_eligible:
        print(
            "Baseline integrity passed; directed recovery is required for: "
            + ", ".join(uncovered_targets),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--expert-manifest", default=str(DEFAULT_EXPERT_MANIFEST))
    parser.add_argument(
        "--representation-audit-json", default=str(DEFAULT_REPRESENTATION_AUDIT)
    )
    parser.add_argument("--source-run-dir", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--old-designs-csv", default=str(DEFAULT_OLD))
    parser.add_argument("--prior-designs-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--expected-source-all-sha256", default=EXPECTED_SOURCE_ALL_SHA256
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=EXPECTED_SOURCE_MANIFEST_SHA256,
    )
    parser.add_argument("--expected-expert-protocol", default=V7_EXPERT_PROTOCOL)
    parser.add_argument("--expected-expert-scope", default="serine-only")
    parser.add_argument(
        "--expected-active-expert-tokens-json", default='["S"]'
    )
    parser.add_argument(
        "--expected-representation-protocol",
        default=V7_REPRESENTATION_AUDIT_PROTOCOL,
    )
    parser.add_argument(
        "--expected-representation-authorization",
        default=V7_REPRESENTATION_AUTHORIZATION,
    )
    parser.add_argument("--output-protocol", default=V7_GENERATION_PROTOCOL)
    parser.add_argument(
        "--recovery-mode",
        default=(
            "SERINE_ONLY_RETRAIN_THEN_REANNOTATE_PRESERVED_V6_NATURAL_POOL_"
            "NO_RESAMPLING_NO_ABSTENTION"
        ),
    )
    parser.add_argument("--run-label", default="SERINE-ONLY V7")
    parser.add_argument("--summary-score-label", default="v7")
    parser.add_argument(
        "--scientific-reason",
        default=(
            "The provenance correction changed only Ser labels. V7 restores the "
            "canonical parent for every non-Ser expert and retrains only Ser."
        ),
    )
    parser.add_argument("--permit-missing-targets-for-recovery", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if not str(args.summary_score_label).replace("_", "").isalnum():
        raise ValueError("--summary-score-label must be alphanumeric/underscore")
    run(args)


if __name__ == "__main__":
    main()
