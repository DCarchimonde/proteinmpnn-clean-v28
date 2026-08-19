#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Finish the peptide-only recovery with deterministic quota top-up.

V4 correctly re-annotated every audited V3 natural sequence, but it deliberately
did not sample replacements when a target remained below its frozen structure
quota.  V5 retains the exact V4 pool and generates additional rows only for a
shortfall target.  Reserve seeds and a finite per-target budget are explicit;
the quota is never lowered and candidates are never manufactured by editing an
annotation or deleting a concentrated position.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
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
    / "serine_qc_peptide_only_v4"
    / "generation"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_structural_support_v5"
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
REQUIRED_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_order_balanced_v3"
)
V4_PROTOCOL = "temperature_0.5_peptide_only_annotation_context_recovery_v4"
V5_PROTOCOL = "temperature_0.5_structural_support_adaptive_quota_recovery_v5"
ANNOTATION_MODE = "peptide_only_cyclic_order_ensemble_known_natural_sequence"
ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
SAMPLING_CONTEXT = "native_complex_longest_receptor_visible"
ALLOWED_V4_FAILED_CHECKS = {
    "no_single_position_exceeds_80_percent_of_sites",
    "no_single_residue_exceeds_80_percent_of_sites",
    "no_target_has_single_position_above_80_percent_when_n_ge_30",
    "no_target_has_single_residue_above_80_percent_when_n_ge_30",
    "every_target_meets_pre_structure_candidate_quota",
}
DEFAULT_RESERVE_SEEDS = (
    606,
    707,
    808,
    909,
    1111,
    1212,
    1313,
    1414,
    1515,
    1616,
    1717,
    1818,
)


def load_generator_module() -> Any:
    spec = importlib.util.spec_from_file_location("serine_v5_base_generator", GENERATOR_PATH)
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


def false_checks(checks: Mapping[str, Any]) -> List[str]:
    return sorted(name for name, passed in checks.items() if not bool(passed))


def validate_v4_source(manifest: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if str(manifest.get("protocol", "")) != V4_PROTOCOL:
        raise RuntimeError("Quota top-up requires the peptide-only V4 recovery source")
    if str(manifest.get("model_expert_qc_protocol", "")) != REQUIRED_EXPERT_PROTOCOL:
        raise RuntimeError("V4 source does not use the promoted order-balanced expert checkpoint")
    if not (
        str(manifest.get("annotation_mode", "")) == ANNOTATION_MODE
        and str(manifest.get("annotation_context_policy", "")) == ANNOTATION_CONTEXT
        and int(manifest.get("annotation_visible_receptor_chains", -1)) == 0
        and bool(manifest.get("train_deployment_context_match"))
    ):
        raise RuntimeError("V4 source annotation context is not training-matched peptide-only")
    if int(manifest.get("raw_candidates_generated", -1)) != len(raw_rows):
        raise RuntimeError("V4 source manifest/raw count mismatch")
    observed_false = false_checks(dict(manifest.get("quality_checks", {})))
    unexpected_false = sorted(set(observed_false) - ALLOWED_V4_FAILED_CHECKS)
    if unexpected_false:
        raise RuntimeError(
            "V4 has failures that quota top-up is not allowed to bypass: "
            + ", ".join(unexpected_false)
        )
    return {
        "source_quality_gate": str(manifest.get("quality_gate", "")),
        "source_false_checks": observed_false,
        "source_false_checks_allowed_for_v5": not unexpected_false,
    }


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def eligible_pool(
    generator: Any,
    raw_rows: Sequence[Mapping[str, Any]],
    old_exact: set[Tuple[str, str]],
    old_natural: set[Tuple[str, str]],
    prior_exact: set[Tuple[str, str]],
    prior_natural: set[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    unique_rows = generator.aggregate_unique_candidates(
        raw_rows,
        old_exact,
        old_natural,
        prior_exact,
        prior_natural,
    )
    eligible_rows = [
        row
        for row in unique_rows
        if int(row["eligible_for_new_permeability_screen"])
    ]
    return unique_rows, eligible_rows


def eligible_count_by_target(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(row["target_name"]).upper() for row in rows)


def checkpoint_metadata(torch_module: Any, model_path: Path) -> Dict[str, Any]:
    checkpoint = torch_module.load(model_path, map_location="cpu")
    metadata = (
        dict(checkpoint.get("expert_head_qc_metadata", {}))
        if isinstance(checkpoint, Mapping)
        else {}
    )
    del checkpoint
    if not (
        str(metadata.get("protocol", "")) == REQUIRED_EXPERT_PROTOCOL
        and int(metadata.get("minimum_order_coverage_epochs", 0)) >= 30
        and str(metadata.get("training_decoding_order_policy", ""))
        == "epoch_indexed_cyclic_designed_position_rotation"
        and str(metadata.get("deployment_annotation_policy", ""))
        == "complete_natural_sequence_all_cyclic_rotations_probability_mean"
    ):
        raise RuntimeError("V5 requires the promoted order-balanced V3 checkpoint")
    return metadata


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("V5 quota top-up requires numpy and torch") from exc

    generator = load_generator_module()
    clean_dir = REPO_ROOT / "paper_clean_v28"
    if str(clean_dir) not in sys.path:
        sys.path.insert(0, str(clean_dir))
    from clean_v28_common import (  # pylint: disable=import-error,import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        EXTENDED_AA_TO_INDEX,
        NAT_TO_METHYL_ABS,
        complete_decoding_order,
        cyclic_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
        peptide_only_annotation_tensors,
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
    source_validation = validate_v4_source(source_manifest, source_rows)
    metadata = checkpoint_metadata(torch, model_path)
    model_sha256 = sha256_file(model_path)
    if str(source_manifest.get("model_sha256", "")) != model_sha256:
        raise RuntimeError("V4 source and V5 top-up checkpoint SHA256 differ")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU top-up is blocked unless --allow-cpu is explicit")
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
            stale = out_dir / output_name
            if stale.is_file():
                stale.unlink()
        generated_fastas = out_dir / "generated_fastas"
        if generated_fastas.is_dir():
            shutil.rmtree(generated_fastas)

    target_manifest = generator.read_csv(source_paths["target_manifest"])
    selected_chains = {
        str(row["target_name"]).upper(): str(row["selected_chain"])
        for row in target_manifest
    }
    native_rows = generator.read_jsonl(native_path)
    target_records, regenerated_target_manifest = generator.prepare_target_records(
        native_rows,
        selected_chains,
        validated["target_names"],
    )
    regenerated_metadata = {
        str(row["target_name"]).upper(): row
        for row in regenerated_target_manifest
    }
    old_exact, old_natural = generator.old_design_keys(old_path)
    prior_rows, prior_exact, prior_natural = generator.validate_prior_handoff(prior_path)
    plan_by_target = {
        str(item["target_name"]).upper(): dict(item)
        for item in validated["targets"]
    }

    raw_rows: List[Dict[str, Any]] = []
    for source in source_rows:
        row = dict(source)
        row["source_recovery_stage"] = "V4_RESCORED_V3_POOL"
        raw_rows.append(row)

    unique_rows, eligible_rows = eligible_pool(
        generator,
        raw_rows,
        old_exact,
        old_natural,
        prior_exact,
        prior_natural,
    )
    initial_counts = eligible_count_by_target(eligible_rows)
    shortfalls = [
        target
        for target in validated["target_names"]
        if initial_counts[target] < int(plan_by_target[target]["structure_quota"])
    ]

    print(f"Loading promoted expert checkpoint: {model_path}", flush=True)
    model = load_v28_model(str(model_path), device)
    model.eval()
    reserve_seeds = [int(value) for value in args.reserve_seeds]
    if (
        not reserve_seeds
        or len(reserve_seeds) != len(set(reserve_seeds))
        or any(value <= 0 for value in reserve_seeds)
        or set(reserve_seeds) & {int(value) for value in validated["seeds"]}
    ):
        raise ValueError("Reserve seeds must be unique, positive, and distinct from V3 seeds")

    topup_rows_by_target: Counter[str] = Counter()
    topup_seeds_by_target: MutableMapping[str, List[int]] = defaultdict(list)
    topup_checkpoints: List[Dict[str, Any]] = []
    temperature = float(plan["temperature"])
    threshold = float(plan["methyl_threshold"])
    for target in shortfalls:
        target_config = plan_by_target[target]
        quota = int(target_config["structure_quota"])
        goal = quota + int(args.quota_margin)
        metadata_row = regenerated_metadata[target]
        packed = featurize_records(
            [target_records[target]],
            device=device,
            eval_chains="masked",
            max_peptide_len=30,
        )
        if packed is None:
            raise RuntimeError(f"Feature construction failed for top-up target {target}")
        features, feature_meta = packed
        if int(feature_meta[0]["selected_length"]) != int(
            metadata_row["native_peptide_length"]
        ):
            raise RuntimeError(f"Peptide coordinate/sequence mismatch for {target}")

        print(
            f"[{target}] initial eligible={initial_counts[target]}, quota={quota}, goal={goal}",
            flush=True,
        )
        target_draws = 0
        current_count = initial_counts[target]
        for reserve_seed in reserve_seeds:
            if current_count >= goal or target_draws >= int(args.max_topup_draws_per_target):
                break
            effective_seed = reserve_seed * 100_000 + generator.stable_target_offset(target)
            generator.torch_seed_all(torch, effective_seed)
            topup_seeds_by_target[target].append(reserve_seed)
            produced_for_seed = 0
            while (
                produced_for_seed < int(args.draws_per_reserve_seed)
                and target_draws < int(args.max_topup_draws_per_target)
                and current_count < goal
            ):
                checkpoint_draws = min(
                    int(args.check_interval_draws),
                    int(args.draws_per_reserve_seed) - produced_for_seed,
                    int(args.max_topup_draws_per_target) - target_draws,
                )
                checkpoint_produced = 0
                while checkpoint_produced < checkpoint_draws:
                    current_batch = min(
                        int(args.batch_size),
                        checkpoint_draws - checkpoint_produced,
                    )
                    generated = generator.generate_batch(
                        model=model,
                        features=features,
                        batch_size=current_batch,
                        temperature=temperature,
                        methyl_threshold=threshold,
                        torch_module=torch,
                        functional=functional,
                        extended_alphabet=EXTENDED_AA_ALPHABET,
                        x_index=int(EXTENDED_AA_TO_INDEX["X"]),
                        natural_to_methyl=NAT_TO_METHYL_ABS,
                        complete_order_fn=complete_decoding_order,
                        ensemble_probability_fn=cyclic_known_sequence_methyl_probabilities,
                        peptide_only_tensors_fn=peptide_only_annotation_tensors,
                    )
                    for batch_offset, generated_row in enumerate(generated):
                        draw_index = produced_for_seed + batch_offset + 1
                        sequence = str(generated_row["design_seq"])
                        methyl_positions = generator.methyl_positions_1based(sequence)
                        row: Dict[str, Any] = {
                            "candidate_id": (
                                f"t05v5_{target.lower()}_s{reserve_seed}_{draw_index:04d}"
                            ),
                            "target_name": target,
                            "temperature": temperature,
                            "methyl_threshold": threshold,
                            "seed": reserve_seed,
                            "effective_seed": effective_seed,
                            "draw_index_within_seed": draw_index,
                            "selected_chain": metadata_row["selected_chain"],
                            "generation_receptor_chain": metadata_row[
                                "generation_receptor_chain"
                            ],
                            "structure_receptor_chains": metadata_row[
                                "structure_receptor_chains"
                            ],
                            "native_seq": metadata_row["native_peptide_seq"],
                            "design_seq": sequence,
                            "design_natural_seq": generator.naturalize(sequence),
                            "native_length": int(metadata_row["native_peptide_length"]),
                            "design_length": len(sequence),
                            "length_match": int(
                                len(sequence) == int(metadata_row["native_peptide_length"])
                            ),
                            "valid_token_gate": int(
                                bool(sequence)
                                and set(sequence) <= generator.VALID_DESIGN_TOKENS
                            ),
                            "natural_aa_recovery": generator.sequence_recovery(
                                str(metadata_row["native_peptide_seq"]), sequence
                            ),
                            "design_methyl_count": len(methyl_positions),
                            "design_methyl_rate": len(methyl_positions) / len(sequence),
                            "methyl_positions_1based": json.dumps(methyl_positions),
                            "current_problem": target_config["current_problem"],
                            "planned_structure_quota": quota,
                            "source_recovery_stage": "V5_ADAPTIVE_QUOTA_TOPUP",
                            **generated_row,
                        }
                        raw_rows.append(row)
                        topup_rows_by_target[target] += 1
                    checkpoint_produced += current_batch
                    produced_for_seed += current_batch
                    target_draws += current_batch

                generator.canonicalize_repeated_natural_annotations(raw_rows)
                unique_rows, eligible_rows = eligible_pool(
                    generator,
                    raw_rows,
                    old_exact,
                    old_natural,
                    prior_exact,
                    prior_natural,
                )
                current_count = eligible_count_by_target(eligible_rows)[target]
                topup_checkpoints.append(
                    {
                        "target_name": target,
                        "reserve_seed": reserve_seed,
                        "draws_from_seed": produced_for_seed,
                        "cumulative_topup_draws": target_draws,
                        "eligible_candidates": current_count,
                        "goal": goal,
                    }
                )
                print(
                    f"[{target}] reserve seed={reserve_seed}, draws={target_draws}, "
                    f"eligible={current_count}/{goal}",
                    flush=True,
                )

    canonicalization = generator.canonicalize_repeated_natural_annotations(raw_rows)
    unique_rows, eligible_rows = eligible_pool(
        generator,
        raw_rows,
        old_exact,
        old_natural,
        prior_exact,
        prior_natural,
    )
    final_counts = eligible_count_by_target(eligible_rows)

    candidate_ids = [str(row.get("candidate_id", "")) for row in raw_rows]
    invalid_rows = [
        row
        for row in raw_rows
        if not int(row.get("length_match", 0)) or not int(row.get("valid_token_gate", 0))
    ]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("V5 raw candidate IDs are empty or duplicated")

    rows_by_target_seed: MutableMapping[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        rows_by_target_seed[
            (str(row["target_name"]).upper(), int(row["seed"]))
        ].append(row)
    for (target, seed), seed_rows in sorted(rows_by_target_seed.items()):
        generator.write_seed_fasta(
            out_dir
            / "generated_fastas"
            / f"seed_{seed}"
            / f"{target.lower()}_designs.fasta",
            seed_rows,
        )

    for row in unique_rows:
        row["permeability_id"] = ""
    raw_fields = union_fields(raw_rows)
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

    target_manifest_index = {
        str(row["target_name"]).upper(): dict(row) for row in target_manifest
    }
    for target in validated["target_names"]:
        row = target_manifest_index[target]
        row["sampling_context_policy"] = SAMPLING_CONTEXT
        row["annotation_context_policy"] = ANNOTATION_CONTEXT
        row["annotation_visible_receptor_chains"] = 0
        row["v5_topup_draws"] = int(topup_rows_by_target[target])
        row["v5_topup_seeds"] = ";".join(
            str(value) for value in topup_seeds_by_target[target]
        )
    final_target_manifest = [target_manifest_index[target] for target in validated["target_names"]]
    generator.atomic_write_csv(
        out_dir / "target_manifest.csv",
        final_target_manifest,
        union_fields(final_target_manifest),
    )

    raw_counts = Counter(str(row["target_name"]).upper() for row in raw_rows)
    unique_counts = Counter(str(row["target_name"]).upper() for row in unique_rows)
    summary_rows: List[Dict[str, Any]] = []
    for target in validated["target_names"]:
        target_unique = [
            row for row in unique_rows if str(row["target_name"]).upper() == target
        ]
        quota = int(plan_by_target[target]["structure_quota"])
        summary_rows.append(
            {
                "target_name": target,
                "current_problem": plan_by_target[target]["current_problem"],
                "raw_generated": raw_counts[target],
                "source_v4_raw_retained": sum(
                    str(row.get("source_recovery_stage", "")) == "V4_RESCORED_V3_POOL"
                    and str(row["target_name"]).upper() == target
                    for row in raw_rows
                ),
                "v5_topup_raw_generated": int(topup_rows_by_target[target]),
                "unique_generated": unique_counts[target],
                "unique_methylated": sum(
                    int(row["passes_methylation_hard_gate"]) for row in target_unique
                ),
                "historical_4115_hits": sum(
                    int(row["seen_in_historical_4115"]) for row in target_unique
                ),
                "prior_1333_hits": sum(
                    int(row["seen_in_prior_1333"]) for row in target_unique
                ),
                "new_methylated_for_permeability": final_counts[target],
                "planned_structure_quota": quota,
                "enough_candidates_before_permeability": int(
                    final_counts[target] >= quota
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
        target
        for target in validated["target_names"]
        if final_counts[target] < int(plan_by_target[target]["structure_quota"])
    ]
    source_ids = {str(row["candidate_id"]) for row in source_rows}
    final_ids = {str(row["candidate_id"]) for row in raw_rows}
    source_natural_by_id = {
        str(row["candidate_id"]): str(row["design_natural_seq"])
        for row in source_rows
    }
    final_natural_by_id = {
        str(row["candidate_id"]): str(row["design_natural_seq"])
        for row in raw_rows
    }
    quality_checks = {
        **dict(annotation_audit["quality_checks"]),
        "v4_source_has_no_unapproved_failures": bool(
            source_validation["source_false_checks_allowed_for_v5"]
        ),
        "every_v4_raw_candidate_is_retained": source_ids <= final_ids,
        "every_v4_natural_sequence_is_retained_byte_for_value": all(
            final_natural_by_id.get(candidate_id) == natural_sequence
            for candidate_id, natural_sequence in source_natural_by_id.items()
        ),
        "all_v5_raw_rows_have_unique_ids_and_valid_lengths": (
            len(candidate_ids) == len(set(candidate_ids)) and not invalid_rows
        ),
        "every_target_meets_pre_structure_candidate_quota": not targets_below_quota,
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    manifest = {
        "quality_gate": quality_gate,
        "quality_checks": quality_checks,
        "protocol": V5_PROTOCOL,
        "recovery_mode": (
            "RETAIN_EXACT_V4_POOL_AND_ADAPTIVELY_SAMPLE_ONLY_QUOTA_SHORTFALL_TARGETS"
        ),
        "scientific_reason": (
            "V4 fixed annotation context. Absolute site concentration is assessed "
            "independently against held-out provenance-confirmed backbone support; "
            "V5 adds natural sequences only when a frozen target quota is short."
        ),
        "temperature": temperature,
        "methyl_threshold": threshold,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "model_expert_qc_protocol": metadata.get("protocol"),
        "source_v4_run_dir": str(source_run),
        "source_v4_generation_manifest": str(source_paths["manifest"]),
        "source_v4_generation_manifest_sha256": sha256_file(source_paths["manifest"]),
        "source_v4_all_candidates_sha256": sha256_file(source_paths["all"]),
        "source_v4_quality_gate": source_validation["source_quality_gate"],
        "source_v4_false_checks": source_validation["source_false_checks"],
        "source_v4_raw_candidates_retained": len(source_rows),
        "adaptive_topup_raw_candidates": len(raw_rows) - len(source_rows),
        "adaptive_topup_initial_shortfall_targets": shortfalls,
        "adaptive_topup_rows_by_target": dict(sorted(topup_rows_by_target.items())),
        "adaptive_topup_seeds_by_target": {
            target: values for target, values in sorted(topup_seeds_by_target.items())
        },
        "adaptive_topup_checkpoints": topup_checkpoints,
        "adaptive_topup_budget": {
            "reserve_seeds": reserve_seeds,
            "draws_per_reserve_seed": int(args.draws_per_reserve_seed),
            "maximum_draws_per_target": int(args.max_topup_draws_per_target),
            "check_interval_draws": int(args.check_interval_draws),
            "quota_margin": int(args.quota_margin),
        },
        "native_jsonl": str(native_path),
        "historical_design_csv": str(old_path),
        "historical_exact_design_keys": len(old_exact),
        "historical_naturalized_design_keys": len(old_natural),
        "prior_handoff_csv": str(prior_path),
        "prior_handoff_rows": len(prior_rows),
        "prior_handoff_exact_design_keys": len(prior_exact),
        "prior_handoff_naturalized_design_keys": len(prior_natural),
        "raw_candidates_expected": len(raw_rows),
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
        "annotation_mode": ANNOTATION_MODE,
        "annotation_context_policy": ANNOTATION_CONTEXT,
        "annotation_visible_receptor_chains": 0,
        "train_deployment_context_match": True,
        "annotation_payload_canonicalization": canonicalization,
        "annotation_stability_audit": annotation_audit,
        "absolute_concentration_policy": (
            "diagnostic in generation; independent three-pass audit requires "
            "held-out provenance-backed structural support"
        ),
        "autoregressive_input_policy": (
            "natural-only model context; lowercase expert annotations are output-only"
        ),
        "permeability_definition_pending": (
            "candidate prediction must be strictly greater than the same-model "
            "native-peptide prediction"
        ),
    }
    generator.atomic_write_json(out_dir / "generation_manifest.json", manifest)

    print("\n===== STRUCTURAL-SUPPORT V5 RECOVERY COMPLETE =====", flush=True)
    print(f"V4 raw rows retained: {len(source_rows)}", flush=True)
    print(f"Adaptive top-up rows: {len(raw_rows) - len(source_rows)}", flush=True)
    print(f"Final new methylated candidates: {len(eligible_rows)}", flush=True)
    print(f"Targets below quota: {targets_below_quota}", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        failed = false_checks(quality_checks)
        raise RuntimeError(
            "V5 adaptive quota recovery failed; outputs were preserved: "
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--draws-per-reserve-seed", type=int, default=1_000)
    parser.add_argument("--max-topup-draws-per-target", type=int, default=10_000)
    parser.add_argument("--check-interval-draws", type=int, default=200)
    parser.add_argument("--quota-margin", type=int, default=5)
    parser.add_argument(
        "--reserve-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESERVE_SEEDS),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.draws_per_reserve_seed <= 0
        or args.max_topup_draws_per_target <= 0
        or args.check_interval_draws <= 0
        or args.quota_margin < 0
    ):
        raise ValueError("Batch and top-up budgets must be positive; quota margin cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
