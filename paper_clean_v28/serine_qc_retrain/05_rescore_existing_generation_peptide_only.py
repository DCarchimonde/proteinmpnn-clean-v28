#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recover the V3 Ser QC generation without retraining or resampling.

The V3 natural base sequences were sampled correctly in receptor-conditioned
complexes, but their final expert-head annotation was evaluated in that complex
context even though the corrected expert heads were trained and tested on
single peptide chains.  This recovery keeps every sampled natural sequence and
path statistic, scores each unique target/natural sequence exactly once in the
training-matched peptide-only context, and writes an isolated V4 result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
GENERATOR_PATH = REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_structure_failures.json")
DEFAULT_MODEL = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_order_balanced_v3"
    / "model"
    / "frankenstein_v28_expert_heads_qc.pt"
)
DEFAULT_SOURCE_RUN = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_order_balanced_v3"
    / "generation"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_peptide_only_v4"
    / "generation"
)
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
EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_order_balanced_v3"
)
ANNOTATION_MODE = "peptide_only_cyclic_order_ensemble_known_natural_sequence"
ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
SAMPLING_CONTEXT = "native_complex_longest_receptor_visible"
VALID_NATURAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_SOURCE_ALL_CANDIDATES_SHA256 = (
    "8cef556a39884cc7d063c7850f0ca1f3886eb0afcada62d1a9c8112ca044a0a6"
)
EXPECTED_SOURCE_GENERATION_MANIFEST_SHA256 = (
    "ae3de9689eebc8ded67af121229ad264f99002630efe1a301e3e612af042dc95"
)


def load_generator_module() -> Any:
    spec = importlib.util.spec_from_file_location("serine_v4_base_generator", GENERATOR_PATH)
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


def batches(values: Sequence[str], batch_size: int) -> Sequence[Sequence[str]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def peptide_only_record(
    source: Mapping[str, Any], selected_chain: str, natural_sequence: str
) -> Dict[str, Any]:
    native_sequence = str(source.get(f"seq_chain_{selected_chain}", ""))
    if not native_sequence:
        raise RuntimeError(f"Selected peptide chain {selected_chain} is absent")
    if len(native_sequence) != len(natural_sequence):
        raise RuntimeError(
            f"Peptide length mismatch for chain {selected_chain}: "
            f"{len(natural_sequence)} != {len(native_sequence)}"
        )
    record: Dict[str, Any] = {
        "name": str(source.get("name", "peptide_only")),
        "seq": natural_sequence,
        f"seq_chain_{selected_chain}": natural_sequence,
        "masked_list": [selected_chain],
        "visible_list": [],
    }
    for atom_name in ("N", "CA", "C", "O"):
        key = f"{atom_name}_chain_{selected_chain}"
        if key not in source:
            raise RuntimeError(f"Missing peptide coordinates: {key}")
        record[key] = copy.deepcopy(source[key])
    return record


def annotation_payload(
    natural_sequence: str,
    probabilities: Sequence[float],
    order_std: Sequence[float],
    threshold: float,
    natural_alphabet: str,
    extended_alphabet: str,
    natural_to_methyl: Mapping[int, int],
) -> Dict[str, Any]:
    if len(natural_sequence) != len(probabilities) or len(probabilities) != len(order_std):
        raise RuntimeError("Annotation vector length differs from the natural peptide length")

    # Persist first, then apply the threshold to the persisted values.  This
    # prevents a value infinitesimally above 0.6 from serializing as exactly 0.6
    # while still emitting a lowercase token.
    rounded_probability = [round(float(value), 8) for value in probabilities]
    rounded_std = [round(float(value), 8) for value in order_std]
    output_tokens: List[str] = []
    for token, probability in zip(natural_sequence, rounded_probability):
        natural_index = natural_alphabet.index(token)
        methyl_index = natural_to_methyl.get(natural_index)
        if methyl_index is not None and probability > threshold:
            output_tokens.append(extended_alphabet[int(methyl_index)])
        else:
            output_tokens.append(token)
    design_sequence = "".join(output_tokens)
    methyl_positions = [
        index for index, token in enumerate(design_sequence, start=1) if token.islower()
    ]
    methyl_probabilities = [
        rounded_probability[index - 1] for index in methyl_positions
    ]
    return {
        "design_seq": design_sequence,
        "design_natural_seq": natural_sequence,
        "design_methyl_count": len(methyl_positions),
        "design_methyl_rate": len(methyl_positions) / len(design_sequence),
        "methyl_positions_1based": json.dumps(methyl_positions),
        "methyl_probability_min": min(rounded_probability),
        "methyl_probability_mean": sum(rounded_probability) / len(rounded_probability),
        "methyl_probability_max": max(rounded_probability),
        "methyl_site_probability_min": min(methyl_probabilities) if methyl_probabilities else "",
        "methyl_site_probability_mean": (
            sum(methyl_probabilities) / len(methyl_probabilities)
            if methyl_probabilities
            else ""
        ),
        "methyl_site_probability_max": max(methyl_probabilities) if methyl_probabilities else "",
        "methyl_probabilities": json.dumps(rounded_probability),
        "methyl_probability_order_std": json.dumps(rounded_std),
        "methyl_probability_order_std_max": max(rounded_std),
        "annotation_mode": ANNOTATION_MODE,
        "annotation_context_policy": ANNOTATION_CONTEXT,
        "annotation_visible_receptor_chains": 0,
        "sampling_context_policy": SAMPLING_CONTEXT,
        "annotation_order_ensemble_size": len(natural_sequence),
    }


def validate_source_rows(
    rows: Sequence[Mapping[str, str]],
    validated_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = int(validated_plan["expected_raw_candidates"])
    if len(rows) != expected:
        raise RuntimeError(f"V3 source raw count changed: expected {expected}, observed {len(rows)}")
    expected_targets = set(validated_plan["target_names"])
    observed_targets = {str(row.get("target_name", "")).upper() for row in rows}
    if observed_targets != expected_targets:
        raise RuntimeError(
            "V3 source target set differs from the frozen plan: "
            f"expected {sorted(expected_targets)}, observed {sorted(observed_targets)}"
        )
    ids = [str(row.get("candidate_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise RuntimeError("V3 source candidate IDs are empty or duplicated")

    selected_by_target: MutableMapping[str, set[str]] = defaultdict(set)
    natural_groups: set[Tuple[str, str]] = set()
    draws_by_target_seed: MutableMapping[Tuple[str, int], set[int]] = defaultdict(set)
    plan_by_target = {
        str(item["target_name"]).upper(): dict(item)
        for item in validated_plan["targets"]
    }
    planned_seeds = {int(value) for value in validated_plan["seeds"]}
    for row in rows:
        target = str(row["target_name"]).upper()
        selected_by_target[target].add(str(row.get("selected_chain", "")))
        natural_sequence = str(row.get("design_natural_seq", "")).upper()
        source_design = str(row.get("design_seq", ""))
        seed = int(row.get("seed", -1))
        draw_index = int(row.get("draw_index_within_seed", -1))
        if (
            not natural_sequence
            or not set(natural_sequence) <= VALID_NATURAL_AA
            or natural_sequence != source_design.upper()
            or int(row.get("design_length", -1)) != len(natural_sequence)
            or int(row.get("native_length", -1)) != len(natural_sequence)
            or seed not in planned_seeds
            or draw_index <= 0
        ):
            raise RuntimeError(f"Invalid V3 natural sequence for {row.get('candidate_id')}")
        try:
            decoding_order = [
                int(value) for value in json.loads(str(row["decoding_order_absolute"]))
            ]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid V3 decoding order for {row.get('candidate_id')}"
            ) from exc
        if (
            len(decoding_order) != len(natural_sequence)
            or len(set(decoding_order)) != len(decoding_order)
            or sorted(decoding_order) != list(range(len(natural_sequence)))
        ):
            raise RuntimeError(f"Invalid V3 decoding order for {row.get('candidate_id')}")
        draws_by_target_seed[(target, seed)].add(draw_index)
        natural_groups.add((target, natural_sequence))
    inconsistent = {
        target: sorted(chains)
        for target, chains in selected_by_target.items()
        if len(chains) != 1 or not next(iter(chains), "")
    }
    if inconsistent:
        raise RuntimeError(f"V3 selected peptide chains are inconsistent: {inconsistent}")
    for target, config in plan_by_target.items():
        expected_draws = set(range(1, int(config["sequences_per_seed"]) + 1))
        for seed in planned_seeds:
            if draws_by_target_seed[(target, seed)] != expected_draws:
                raise RuntimeError(
                    f"V3 draw coverage changed for {target} seed {seed}"
                )
    return {
        "selected_chain_by_target": {
            target: next(iter(chains)) for target, chains in selected_by_target.items()
        },
        "unique_target_natural_sequence_groups": len(natural_groups),
        "repeated_raw_rows": len(rows) - len(natural_groups),
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
        peptide_length = int(metas[0]["selected_length"])
        if peptide_length != len(sequences[0]) or int(metas[0]["total_length"]) != peptide_length:
            raise RuntimeError(f"Visible receptor leaked into the annotation context for {target}")

        for sequence_batch in batches(sequences, batch_size):
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
                probabilities, order_std = common[
                    "cyclic_known_sequence_methyl_probabilities"
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
                    probabilities[index].detach().cpu().tolist(),
                    order_std[index].detach().cpu().tolist(),
                    threshold,
                    natural_alphabet,
                    str(common["EXTENDED_AA_ALPHABET"]),
                    common["NAT_TO_METHYL_ABS"],
                )
        print(
            f"[{target}] peptide-only annotations: {len(sequences)} unique natural sequences",
            flush=True,
        )
    return result


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Peptide-only recovery requires numpy and torch") from exc

    generator = load_generator_module()
    clean_dir = REPO_ROOT / "paper_clean_v28"
    if str(clean_dir) not in sys.path:
        sys.path.insert(0, str(clean_dir))
    from clean_v28_common import (  # pylint: disable=import-error,import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        NAT_TO_METHYL_ABS,
        NATURAL_AA_ALPHABET,
        cyclic_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
    )

    plan_path = Path(args.plan).resolve()
    model_path = Path(args.model_path).resolve()
    source_run = Path(args.source_run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    native_path = Path(args.native_jsonl).resolve()
    old_path = Path(args.old_designs_csv).resolve()
    prior_path = Path(args.prior_designs_csv).resolve()
    source_paths = {
        "all": source_run / "all_candidates.csv",
        "manifest": source_run / "generation_manifest.json",
        "target_manifest": source_run / "target_manifest.csv",
    }
    for required in (
        plan_path,
        model_path,
        native_path,
        old_path,
        prior_path,
        *source_paths.values(),
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    plan = generator.read_json(plan_path)
    validated = generator.validate_plan(plan)
    source_manifest = generator.read_json(source_paths["manifest"])
    source_rows = generator.read_csv(source_paths["all"])
    source_all_sha256 = sha256_file(source_paths["all"])
    source_manifest_sha256 = sha256_file(source_paths["manifest"])
    if (
        source_all_sha256 != EXPECTED_SOURCE_ALL_CANDIDATES_SHA256
        or source_manifest_sha256 != EXPECTED_SOURCE_GENERATION_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "V3 source files differ from the uploaded audited review bundle; "
            f"all_candidates={source_all_sha256}, manifest={source_manifest_sha256}"
        )
    source_validation = validate_source_rows(source_rows, validated)
    if not (
        str(source_manifest.get("protocol", ""))
        == "temperature_0.5_all_expert_qc_order_balanced_structure_failure_recovery_v3"
        and str(source_manifest.get("model_expert_qc_protocol", "")) == EXPERT_PROTOCOL
        and str(source_manifest.get("workflow_order", ""))
        == "STRUCTURE_FIRST_THEN_PERMEABILITY"
        and str(source_manifest.get("permeability_status", ""))
        == "DEFERRED_UNTIL_STRUCTURE_RETURNS"
        and "exact same full permutation is passed to every model forward"
        in str(source_manifest.get("generation_decoding_order_policy", ""))
    ):
        raise RuntimeError(
            "Source run is not the audited order-balanced V3 generation expected by recovery"
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    metadata = (
        dict(checkpoint.get("expert_head_qc_metadata", {}))
        if isinstance(checkpoint, Mapping)
        else {}
    )
    del checkpoint
    if not (
        str(metadata.get("protocol", "")) == EXPERT_PROTOCOL
        and int(metadata.get("minimum_order_coverage_epochs", 0)) >= 30
        and str(metadata.get("training_decoding_order_policy", ""))
        == "epoch_indexed_cyclic_designed_position_rotation"
        and str(metadata.get("deployment_annotation_policy", ""))
        == "complete_natural_sequence_all_cyclic_rotations_probability_mean"
    ):
        raise RuntimeError("Recovery requires the promoted order-balanced V3 expert checkpoint")
    model_sha256 = sha256_file(model_path)
    if str(source_manifest.get("model_sha256", "")) != model_sha256:
        raise RuntimeError(
            "The V3 source rows and checkpoint do not match: "
            f"source={source_manifest.get('model_sha256')}, checkpoint={model_sha256}"
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU recovery is blocked unless --allow-cpu is explicit")
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

    generator.ensure_output_scope(out_dir, args.overwrite)
    if args.overwrite:
        for output_name in generator.KNOWN_OUTPUTS:
            stale_path = out_dir / output_name
            if stale_path.is_file():
                stale_path.unlink()
    native_rows = generator.read_jsonl(native_path)
    native_index = {
        generator.record_name(row, index): row for index, row in enumerate(native_rows)
    }
    missing_targets = sorted(set(validated["target_names"]) - set(native_index))
    if missing_targets:
        raise RuntimeError("Targets missing from native JSONL: " + ", ".join(missing_targets))

    print(f"Loading promoted expert checkpoint: {model_path}", flush=True)
    model = load_v28_model(str(model_path), device)
    model.eval()
    common = {
        "EXTENDED_AA_ALPHABET": EXTENDED_AA_ALPHABET,
        "NAT_TO_METHYL_ABS": NAT_TO_METHYL_ABS,
        "NATURAL_AA_ALPHABET": NATURAL_AA_ALPHABET,
        "cyclic_known_sequence_methyl_probabilities": cyclic_known_sequence_methyl_probabilities,
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
    )

    raw_rows: List[Dict[str, Any]] = []
    rows_by_target_seed: MutableMapping[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for source in source_rows:
        target = str(source["target_name"]).upper()
        natural_sequence = str(source["design_natural_seq"]).upper()
        row: Dict[str, Any] = dict(source)
        row.update(annotation_by_key[(target, natural_sequence)])
        row["length_match"] = int(len(natural_sequence) == int(row["native_length"]))
        row["valid_token_gate"] = 1
        raw_rows.append(row)
        rows_by_target_seed[(target, int(row["seed"]))].append(row)

    old_exact, old_natural = generator.old_design_keys(old_path)
    prior_rows, prior_exact, prior_natural = generator.validate_prior_handoff(prior_path)
    unique_rows = generator.aggregate_unique_candidates(
        raw_rows, old_exact, old_natural, prior_exact, prior_natural
    )
    eligible_rows = [
        row for row in unique_rows if int(row["eligible_for_new_permeability_screen"])
    ]
    for row in unique_rows:
        row["permeability_id"] = ""

    for (target, seed), seed_rows in sorted(rows_by_target_seed.items()):
        generator.write_seed_fasta(
            out_dir / "generated_fastas" / f"seed_{seed}" / f"{target.lower()}_designs.fasta",
            seed_rows,
        )

    raw_fields = list(raw_rows[0])
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
    generator.atomic_write_csv(out_dir / "all_candidates.csv", raw_rows, raw_fields)
    generator.atomic_write_csv(out_dir / "unique_candidates.csv", unique_rows, unique_fields)
    generator.atomic_write_csv(
        out_dir / "methylated_new_candidates.csv", eligible_rows, unique_fields
    )

    target_manifest = generator.read_csv(source_paths["target_manifest"])
    for row in target_manifest:
        row["sampling_context_policy"] = SAMPLING_CONTEXT
        row["annotation_context_policy"] = ANNOTATION_CONTEXT
        row["annotation_visible_receptor_chains"] = 0
    generator.atomic_write_csv(
        out_dir / "target_manifest.csv", target_manifest, list(target_manifest[0])
    )

    plan_by_target = {
        str(item["target_name"]).upper(): dict(item) for item in validated["targets"]
    }
    raw_count_by_target = Counter(str(row["target_name"]).upper() for row in raw_rows)
    unique_count_by_target = Counter(str(row["target_name"]).upper() for row in unique_rows)
    eligible_count_by_target = Counter(str(row["target_name"]).upper() for row in eligible_rows)
    summary_rows: List[Dict[str, Any]] = []
    for target in validated["target_names"]:
        target_unique = [row for row in unique_rows if str(row["target_name"]).upper() == target]
        quota = int(plan_by_target[target]["structure_quota"])
        summary_rows.append(
            {
                "target_name": target,
                "current_problem": plan_by_target[target]["current_problem"],
                "raw_generated": raw_count_by_target[target],
                "unique_generated": unique_count_by_target[target],
                "unique_methylated": sum(
                    int(row["passes_methylation_hard_gate"]) for row in target_unique
                ),
                "historical_4115_hits": sum(
                    int(row["seen_in_historical_4115"]) for row in target_unique
                ),
                "prior_1333_hits": sum(
                    int(row["seen_in_prior_1333"]) for row in target_unique
                ),
                "new_methylated_for_permeability": eligible_count_by_target[target],
                "planned_structure_quota": quota,
                "enough_candidates_before_permeability": int(
                    eligible_count_by_target[target] >= quota
                ),
            }
        )
    generator.atomic_write_csv(
        out_dir / "generation_summary_by_target.csv",
        summary_rows,
        list(summary_rows[0]),
    )

    annotation_audit = generator.audit_annotation_stability(raw_rows, eligible_rows)
    targets_below_quota = [
        row["target_name"]
        for row in summary_rows
        if not int(row["enough_candidates_before_permeability"])
    ]
    quality_checks = {
        **dict(annotation_audit["quality_checks"]),
        "source_v3_raw_count_and_target_set_validated": True,
        "source_v3_checkpoint_sha256_matches": True,
        "every_unique_target_natural_sequence_scored_exactly_once": (
            len(annotation_by_key)
            == int(source_validation["unique_target_natural_sequence_groups"])
        ),
        "every_target_meets_pre_structure_candidate_quota": not targets_below_quota,
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    manifest = {
        "quality_gate": quality_gate,
        "quality_checks": quality_checks,
        "protocol": "temperature_0.5_peptide_only_annotation_context_recovery_v4",
        "recovery_mode": "RESCORE_EXISTING_V3_NATURAL_SEQUENCES_NO_RETRAIN_NO_RESAMPLING",
        "scientific_reason": (
            "V3 base sampling and novelty checks passed, but its expert annotation used a "
            "receptor-visible complex outside the peptide-only expert train/test domain"
        ),
        "temperature": float(plan["temperature"]),
        "methyl_threshold": float(plan["methyl_threshold"]),
        "seeds": validated["seeds"],
        "batch_size": int(args.batch_size),
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "deterministic_sorted_unique_scoring": True,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "model_expert_qc_protocol": metadata.get("protocol"),
        "source_v3_run_dir": str(source_run),
        "source_v3_generation_manifest": str(source_paths["manifest"]),
        "source_v3_generation_manifest_sha256": source_manifest_sha256,
        "source_v3_all_candidates_sha256": source_all_sha256,
        "source_v3_quality_gate": source_manifest.get("quality_gate"),
        "source_v3_natural_sequences_retained": len(raw_rows),
        "unique_target_natural_sequences_rescored": len(annotation_by_key),
        "annotation_payload_canonicalization": {
            **source_validation,
            "selected_chain_by_target": source_validation["selected_chain_by_target"],
            "canonical_payload_source": "one peptide-only score per sorted target+natural sequence",
        },
        "native_jsonl": str(native_path),
        "historical_design_csv": str(old_path),
        "historical_exact_design_keys": len(old_exact),
        "historical_naturalized_design_keys": len(old_natural),
        "prior_handoff_csv": str(prior_path),
        "prior_handoff_rows": len(prior_rows),
        "prior_handoff_exact_design_keys": len(prior_exact),
        "prior_handoff_naturalized_design_keys": len(prior_natural),
        "raw_candidates_expected": int(validated["expected_raw_candidates"]),
        "raw_candidates_generated": len(raw_rows),
        "unique_candidates": len(unique_rows),
        "new_methylated_candidates_for_permeability": len(eligible_rows),
        "workflow_order": "STRUCTURE_FIRST_THEN_PERMEABILITY",
        "permeability_status": "DEFERRED_UNTIL_STRUCTURE_RETURNS",
        "permeability_input_rows": 0,
        "native_permeability_controls": 0,
        "planned_structure_handoff": int(validated["planned_structure_handoff"]),
        "targets_below_pre_permeability_quota": targets_below_quota,
        "frozen_targets_not_regenerated": plan["frozen_targets"],
        "sampling_context_policy": SAMPLING_CONTEXT,
        "sampling_context_retained_from_v3": True,
        "annotation_mode": ANNOTATION_MODE,
        "annotation_context_policy": ANNOTATION_CONTEXT,
        "annotation_visible_receptor_chains": 0,
        "train_deployment_context_match": True,
        "annotation_order_policy": (
            "complete-natural-sequence cyclic ensemble; every peptide site occurs "
            "once at every relative decoder depth"
        ),
        "annotation_stability_audit": annotation_audit,
        "autoregressive_input_policy": (
            "natural-only V3 sampling context retained; V4 lowercase annotations are output-only"
        ),
        "permeability_definition_pending": (
            "candidate prediction must be strictly greater than the same-model native-peptide prediction"
        ),
    }
    generator.atomic_write_json(out_dir / "generation_manifest.json", manifest)

    print("\n===== PEPTIDE-ONLY V4 RECOVERY COMPLETE =====", flush=True)
    print(f"Raw natural sequences retained: {len(raw_rows)}", flush=True)
    print(f"Unique target/natural sequences rescored: {len(annotation_by_key)}", flush=True)
    print(f"New methylated candidates: {len(eligible_rows)}", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError(
            "V4 peptide-only recovery quality gate failed; outputs were preserved: "
            + ", ".join(failed)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--source-run-dir", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--old-designs-csv", default=str(DEFAULT_OLD))
    parser.add_argument("--prior-designs-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--batch-size", type=int, default=32)
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
