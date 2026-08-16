#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resume a completed V6 generation by sampling quota-shortfall targets only.

The expensive V6 expert-head retraining, held-out cyclic-representation audit,
and the original 19,500 draws are immutable inputs to this recovery.  The
script reads the preserved failed generation, identifies targets whose final
novel methylated pool is below the frozen structure quota, and samples only
those targets with disjoint reserve seeds.  It never lowers the methylation
threshold, edits annotations, or discards an original V6 row.

All replacement CSV files are written atomically and the manifest is written
last.  A failed top-up therefore remains blocked while retaining every row for
another diagnostic pass.
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
GENERATOR_PATH = (
    REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
)
V6_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_cyclic_representation_v6"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_representation_v6.json")
DEFAULT_MODEL = V6_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
DEFAULT_SOURCE_RUN = V6_ROOT / "generation"
DEFAULT_REPRESENTATION_AUDIT = (
    V6_ROOT / "representation_audit" / "cyclic_representation_audit.json"
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
REPRESENTATION_AUDIT_PROTOCOL = "cyclic_representation_equivariance_heldout_gate_v1"
REPRESENTATION_AUDIT_AUTHORIZATION = (
    "REPRESENTATION_ENSEMBLE_VALIDATED_FOR_ISOLATED_V6_REGENERATION"
)
ANNOTATION_MODE = (
    "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
)
ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
SAMPLING_CONTEXT = "native_complex_longest_receptor_visible"
RECOVERY_MODE = (
    "RETAIN_COMPLETE_V6_RUN_AND_ADAPTIVELY_SAMPLE_ONLY_QUOTA_SHORTFALL_TARGETS"
)
INITIAL_STAGE = "V6_INITIAL_FULL_REGENERATION"
TOPUP_STAGE = "V6_ADAPTIVE_QUOTA_TOPUP"
ALLOWED_SOURCE_FAILED_CHECKS = {"every_target_meets_pre_structure_candidate_quota"}
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
    1919,
    2020,
    2121,
    2222,
    2323,
    2424,
    2525,
    2626,
    2727,
    2828,
    2929,
    3030,
)


def load_generator_module() -> Any:
    spec = importlib.util.spec_from_file_location("serine_v6_base_generator", GENERATOR_PATH)
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
        and bool(metadata.get("cyclic_representation_augmentation"))
        and str(metadata.get("training_cyclic_representation_policy", ""))
        == REQUIRED_TRAINING_REPRESENTATION_POLICY
        and str(metadata.get("training_decoding_order_policy", ""))
        == REQUIRED_TRAINING_ORDER_POLICY
        and str(metadata.get("deployment_annotation_policy", ""))
        == REQUIRED_DEPLOYMENT_POLICY
    ):
        raise RuntimeError(
            "V6 quota resume requires the promoted cyclic-representation checkpoint"
        )
    return metadata


def validate_representation_audit(
    report: Mapping[str, Any],
    report_path: Path,
    model_sha256: str,
    plan_path: Path,
) -> None:
    if not (
        str(report.get("quality_gate", "")) == "PASS"
        and str(report.get("protocol", "")) == REPRESENTATION_AUDIT_PROTOCOL
        and str(report.get("release_authorization", ""))
        == REPRESENTATION_AUDIT_AUTHORIZATION
        and str(report.get("model_sha256", "")) == model_sha256
        and str(report.get("plan_sha256", "")) == sha256_file(plan_path)
        and str(report.get("annotation_mode", "")) == ANNOTATION_MODE
    ):
        raise RuntimeError(
            "V6 quota resume is blocked because the held-out representation audit "
            "is absent, failed, or belongs to different model/plan bytes"
        )
    if not report_path.is_file():
        raise FileNotFoundError(report_path)


def validate_source_manifest(
    manifest: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    plan_path: Path,
    model_sha256: str,
    representation_audit_path: Path,
) -> Dict[str, Any]:
    if str(manifest.get("protocol", "")) != str(plan.get("protocol", "")):
        raise RuntimeError("V6 source manifest and frozen plan protocols differ")
    if str(manifest.get("model_sha256", "")) != model_sha256:
        raise RuntimeError("V6 source generation and checkpoint SHA256 differ")
    if str(manifest.get("model_expert_qc_protocol", "")) != REQUIRED_EXPERT_PROTOCOL:
        raise RuntimeError("V6 source generation used the wrong expert protocol")
    if not (
        str(manifest.get("annotation_mode", "")) == ANNOTATION_MODE
        and str(manifest.get("annotation_context_policy", "")) == ANNOTATION_CONTEXT
        and int(manifest.get("annotation_visible_receptor_chains", -1)) == 0
        and bool(manifest.get("train_deployment_context_match"))
        and bool(manifest.get("cyclic_representation_ensemble_enabled"))
    ):
        raise RuntimeError("V6 source annotation is not the cyclic peptide-only policy")
    if int(manifest.get("raw_candidates_generated", -1)) != len(raw_rows):
        raise RuntimeError("V6 source manifest/raw row count mismatch")
    pinned = dict(manifest.get("cyclic_representation_heldout_audit") or {})
    if not (
        str(pinned.get("quality_gate", "")) == "PASS"
        and str(pinned.get("protocol", "")) == REPRESENTATION_AUDIT_PROTOCOL
        and str(pinned.get("release_authorization", ""))
        == REPRESENTATION_AUDIT_AUTHORIZATION
        and str(pinned.get("model_sha256", "")) == model_sha256
        and str(pinned.get("plan_sha256", "")) == sha256_file(plan_path)
        and str(pinned.get("sha256", "")) == sha256_file(representation_audit_path)
    ):
        raise RuntimeError("V6 source does not pin the passed representation audit")
    observed_false = false_checks(dict(manifest.get("quality_checks", {})))
    source_quality_gate = str(manifest.get("quality_gate", ""))
    if source_quality_gate not in {"PASS", "FAIL"}:
        raise RuntimeError("V6 source manifest has no valid quality-gate state")
    if (source_quality_gate == "PASS") == bool(observed_false):
        raise RuntimeError(
            "V6 source manifest quality-gate state contradicts its recorded checks"
        )
    unexpected = sorted(set(observed_false) - ALLOWED_SOURCE_FAILED_CHECKS)
    if unexpected:
        raise RuntimeError(
            "V6 source has failures quota top-up may not bypass: " + ", ".join(unexpected)
        )
    return {
        "source_quality_gate": source_quality_gate,
        "source_false_checks": observed_false,
        "source_false_checks_allowed": not unexpected,
    }


def validate_source_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    validated: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reject a partial, mixed-policy, or already-corrupted V6 source pool."""
    target_names = [str(value).upper() for value in validated["target_names"]]
    target_set = set(target_names)
    candidate_ids = [str(row.get("candidate_id", "")) for row in rows]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("V6 source candidate IDs are empty or duplicated")

    invalid_targets = sorted(
        {
            str(row.get("target_name", "")).upper()
            for row in rows
            if str(row.get("target_name", "")).upper() not in target_set
        }
    )
    if invalid_targets:
        raise RuntimeError(
            "V6 source contains targets outside the frozen plan: "
            + ", ".join(invalid_targets)
        )

    invalid_semantic_rows = []
    for row in rows:
        try:
            valid = (
                int(row.get("length_match", 0)) == 1
                and int(row.get("valid_token_gate", 0)) == 1
                and str(row.get("annotation_mode", "")) == ANNOTATION_MODE
                and str(row.get("annotation_context_policy", ""))
                == ANNOTATION_CONTEXT
                and int(row.get("annotation_visible_receptor_chains", -1)) == 0
                and int(row.get("annotation_representation_ensemble_size", -1))
                == int(row.get("design_length", -2))
                and float(row.get("methyl_threshold", "nan"))
                == float(plan["methyl_threshold"])
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalid_semantic_rows.append(str(row.get("candidate_id", "<missing>")))
            if len(invalid_semantic_rows) >= 10:
                break
    if invalid_semantic_rows:
        raise RuntimeError(
            "V6 source contains invalid or mixed-policy rows: "
            + ", ".join(invalid_semantic_rows)
        )

    stage_counts = Counter(str(row.get("source_recovery_stage", "")) for row in rows)
    expected_initial_by_target = {
        str(item["target_name"]).upper(): int(item["sequences_per_seed"])
        * len(validated["seeds"])
        for item in validated["targets"]
    }
    observed_initial_by_target: Counter[str] = Counter()
    is_resumed_source = str(manifest.get("recovery_mode", "")) == RECOVERY_MODE
    if is_resumed_source:
        unexpected_stages = sorted(set(stage_counts) - {INITIAL_STAGE, TOPUP_STAGE})
        if unexpected_stages:
            raise RuntimeError(
                "V6 resumed source contains an unknown recovery stage: "
                + ", ".join(repr(value) for value in unexpected_stages)
            )
        for row in rows:
            if str(row.get("source_recovery_stage", "")) == INITIAL_STAGE:
                observed_initial_by_target[str(row["target_name"]).upper()] += 1
        if (
            stage_counts[INITIAL_STAGE]
            != int(manifest.get("source_v6_raw_candidates_retained", -1))
            or stage_counts[TOPUP_STAGE]
            != int(manifest.get("adaptive_topup_raw_candidates", -1))
        ):
            raise RuntimeError("V6 source recovery-stage accounting is inconsistent")
    else:
        if set(stage_counts) != {""}:
            raise RuntimeError("Initial V6 source unexpectedly contains recovery stages")
        observed_initial_by_target.update(
            str(row["target_name"]).upper() for row in rows
        )

    if dict(observed_initial_by_target) != expected_initial_by_target:
        raise RuntimeError(
            "V6 source does not retain the complete frozen full-regeneration pool"
        )
    return {
        "source_candidate_ids_unique": True,
        "source_semantic_rows_valid": True,
        "source_initial_rows_by_target": dict(observed_initial_by_target),
        "source_stage_counts": dict(stage_counts),
    }


def backup_source_files(out_dir: Path) -> Path:
    backup_dir = out_dir / "pre_quota_resume_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "generation_manifest.json",
        "generation_summary_by_target.csv",
        "all_candidates.csv",
        "unique_candidates.csv",
        "methylated_new_candidates.csv",
        "target_manifest.csv",
    ):
        source = out_dir / name
        destination = backup_dir / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    return backup_dir


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("V6 quota resume requires numpy and torch") from exc

    generator = load_generator_module()
    clean_dir = REPO_ROOT / "paper_clean_v28"
    if str(clean_dir) not in sys.path:
        sys.path.insert(0, str(clean_dir))
    from clean_v28_common import (  # pylint: disable=import-error,import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        EXTENDED_AA_TO_INDEX,
        NAT_TO_METHYL_ABS,
        complete_decoding_order,
        cyclic_representation_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
        peptide_only_annotation_tensors,
    )

    plan_path = Path(args.plan).resolve()
    model_path = Path(args.model_path).resolve()
    source_run = Path(args.source_run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    audit_path = Path(args.representation_audit_json).resolve()
    native_path = Path(args.native_jsonl).resolve()
    old_path = Path(args.old_designs_csv).resolve()
    prior_path = Path(args.prior_designs_csv).resolve()
    if source_run != out_dir:
        raise ValueError("V6 quota recovery is intentionally in-place; source and output differ")
    source_paths = {
        "all": source_run / "all_candidates.csv",
        "manifest": source_run / "generation_manifest.json",
        "target_manifest": source_run / "target_manifest.csv",
    }
    for required in (
        plan_path,
        model_path,
        audit_path,
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
    model_sha256 = sha256_file(model_path)
    metadata = checkpoint_metadata(torch, model_path)
    representation_audit = generator.read_json(audit_path)
    validate_representation_audit(
        representation_audit,
        audit_path,
        model_sha256,
        plan_path,
    )
    source_validation = validate_source_manifest(
        source_manifest,
        source_rows,
        plan,
        plan_path,
        model_sha256,
        audit_path,
    )
    source_row_validation = validate_source_rows(
        source_rows,
        source_manifest,
        plan,
        validated,
    )

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
        str(row["target_name"]).upper(): row for row in regenerated_target_manifest
    }
    old_exact, old_natural = generator.old_design_keys(old_path)
    prior_rows, prior_exact, prior_natural = generator.validate_prior_handoff(prior_path)
    source_unique_rows, source_eligible_rows = eligible_pool(
        generator,
        source_rows,
        old_exact,
        old_natural,
        prior_exact,
        prior_natural,
    )
    source_annotation_audit = generator.audit_annotation_stability(
        source_rows,
        source_eligible_rows,
    )
    if str(source_annotation_audit.get("quality_gate", "")) != "PASS":
        raise RuntimeError(
            "V6 source row audit failed before quota sampling: "
            + ", ".join(
                false_checks(dict(source_annotation_audit.get("quality_checks", {})))
            )
        )
    plan_by_target = {
        str(item["target_name"]).upper(): dict(item) for item in validated["targets"]
    }

    raw_rows: List[Dict[str, Any]] = []
    for source in source_rows:
        row = dict(source)
        if not str(row.get("source_recovery_stage", "")):
            row["source_recovery_stage"] = INITIAL_STAGE
        raw_rows.append(row)
    source_ids = {str(row["candidate_id"]) for row in raw_rows}
    source_payload_by_id = {
        str(row["candidate_id"]): dict(row) for row in source_rows
    }
    source_natural_by_id = {
        str(row["candidate_id"]): str(row["design_natural_seq"]) for row in raw_rows
    }
    initial_full_rows = int(
        source_manifest.get("source_v6_raw_candidates_retained", len(source_rows))
        if str(source_manifest.get("recovery_mode", "")) == RECOVERY_MODE
        else len(source_rows)
    )

    generator.canonicalize_repeated_natural_annotations(raw_rows)
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
    recorded_shortfalls = {
        str(value).upper()
        for value in source_manifest.get("targets_below_pre_permeability_quota", [])
    }
    if recorded_shortfalls != set(shortfalls):
        raise RuntimeError(
            "V6 source manifest shortfall list does not match recomputed candidates"
        )
    recorded_quota_failure = (
        "every_target_meets_pre_structure_candidate_quota"
        in source_validation["source_false_checks"]
    )
    if recorded_quota_failure != bool(shortfalls):
        raise RuntimeError(
            "V6 source quota check does not match recomputed candidate shortfalls"
        )
    if not shortfalls:
        print("V6 already meets every target quota; no top-up was performed.", flush=True)
        return

    print(
        "Quota-shortfall targets: "
        + ", ".join(
            f"{target} ({initial_counts[target]}/"
            f"{int(plan_by_target[target]['structure_quota'])})"
            for target in shortfalls
        ),
        flush=True,
    )

    reserve_seeds = [int(value) for value in args.reserve_seeds]
    if (
        not reserve_seeds
        or len(reserve_seeds) != len(set(reserve_seeds))
        or any(value <= 0 for value in reserve_seeds)
        or set(reserve_seeds) & {int(value) for value in validated["seeds"]}
    ):
        raise ValueError("Reserve seeds must be unique, positive, and disjoint from V6 seeds")

    print(f"Loading promoted V6 checkpoint: {model_path}", flush=True)
    model = load_v28_model(str(model_path), device)
    model.eval()
    existing_topup_rows = [
        row for row in raw_rows if str(row.get("source_recovery_stage", "")) == TOPUP_STAGE
    ]
    topup_rows_by_target: Counter[str] = Counter(
        str(row["target_name"]).upper() for row in existing_topup_rows
    )
    topup_seeds_by_target: MutableMapping[str, List[int]] = defaultdict(list)
    for row in existing_topup_rows:
        target = str(row["target_name"]).upper()
        seed = int(row["seed"])
        if seed not in topup_seeds_by_target[target]:
            topup_seeds_by_target[target].append(seed)
    topup_checkpoints: List[Dict[str, Any]] = list(
        source_manifest.get("adaptive_topup_checkpoints", [])
    )
    all_initial_shortfalls = list(
        source_manifest.get("adaptive_topup_initial_shortfall_targets", shortfalls)
    )
    for target in shortfalls:
        if target not in all_initial_shortfalls:
            all_initial_shortfalls.append(target)

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

        current_count = initial_counts[target]
        target_draws_this_run = 0
        remaining_target_budget = max(
            0,
            int(args.max_topup_draws_per_target)
            - int(topup_rows_by_target[target]),
        )
        used_seeds = set(topup_seeds_by_target[target])
        print(
            f"[{target}] existing eligible={current_count}, quota={quota}, goal={goal}, "
            f"remaining fixed budget={remaining_target_budget}",
            flush=True,
        )
        for reserve_seed in reserve_seeds:
            if current_count >= goal or target_draws_this_run >= remaining_target_budget:
                break
            if reserve_seed in used_seeds:
                continue
            effective_seed = reserve_seed * 100_000 + generator.stable_target_offset(target)
            generator.torch_seed_all(torch, effective_seed)
            topup_seeds_by_target[target].append(reserve_seed)
            used_seeds.add(reserve_seed)
            produced_for_seed = 0
            while (
                produced_for_seed < int(args.draws_per_reserve_seed)
                and target_draws_this_run < remaining_target_budget
                and current_count < goal
            ):
                checkpoint_draws = min(
                    int(args.check_interval_draws),
                    int(args.draws_per_reserve_seed) - produced_for_seed,
                    remaining_target_budget - target_draws_this_run,
                )
                checkpoint_produced = 0
                while checkpoint_produced < checkpoint_draws:
                    current_batch = min(
                        int(args.batch_size), checkpoint_draws - checkpoint_produced
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
                        ensemble_probability_fn=(
                            cyclic_representation_known_sequence_methyl_probabilities
                        ),
                        peptide_only_tensors_fn=peptide_only_annotation_tensors,
                    )
                    for batch_offset, generated_row in enumerate(generated):
                        draw_index = produced_for_seed + batch_offset + 1
                        sequence = str(generated_row["design_seq"])
                        methyl_positions = generator.methyl_positions_1based(sequence)
                        row: Dict[str, Any] = {
                            "candidate_id": (
                                f"t05v6topup_{target.lower()}_s{reserve_seed}_"
                                f"{draw_index:04d}"
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
                            "source_recovery_stage": TOPUP_STAGE,
                            **generated_row,
                        }
                        raw_rows.append(row)
                        topup_rows_by_target[target] += 1
                    checkpoint_produced += current_batch
                    produced_for_seed += current_batch
                    target_draws_this_run += current_batch

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
                        "draws_this_resume": target_draws_this_run,
                        "total_topup_rows_for_target": topup_rows_by_target[target],
                        "eligible_candidates": current_count,
                        "goal": goal,
                    }
                )
                print(
                    f"[{target}] reserve seed={reserve_seed}, "
                    f"new draws={target_draws_this_run}, eligible={current_count}/{goal}",
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
        raise RuntimeError("V6 resumed raw candidate IDs are empty or duplicated")

    final_natural_by_id = {
        str(row["candidate_id"]): str(row["design_natural_seq"]) for row in raw_rows
    }
    final_row_by_id = {str(row["candidate_id"]): row for row in raw_rows}
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

    target_manifest_index = {
        str(row["target_name"]).upper(): dict(row) for row in target_manifest
    }
    for target in validated["target_names"]:
        row = target_manifest_index[target]
        row["sampling_context_policy"] = SAMPLING_CONTEXT
        row["annotation_context_policy"] = ANNOTATION_CONTEXT
        row["annotation_visible_receptor_chains"] = 0
        row["v6_topup_draws"] = int(topup_rows_by_target[target])
        row["v6_topup_seeds"] = ";".join(
            str(value) for value in topup_seeds_by_target[target]
        )
    final_target_manifest = [
        target_manifest_index[target] for target in validated["target_names"]
    ]

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
                "source_v6_raw_retained": sum(
                    str(row.get("source_recovery_stage", "")) == INITIAL_STAGE
                    and str(row["target_name"]).upper() == target
                    for row in raw_rows
                ),
                "v6_topup_raw_generated": int(topup_rows_by_target[target]),
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
                "enough_candidates_before_permeability": int(final_counts[target] >= quota),
            }
        )

    annotation_audit = generator.audit_annotation_stability(raw_rows, eligible_rows)
    targets_below_quota = [
        target
        for target in validated["target_names"]
        if final_counts[target] < int(plan_by_target[target]["structure_quota"])
    ]
    stage_counts = Counter(str(row.get("source_recovery_stage", "")) for row in raw_rows)
    quality_checks = {
        **dict(annotation_audit["quality_checks"]),
        "source_v6_has_no_unapproved_failures": bool(
            source_validation["source_false_checks_allowed"]
        ),
        "every_pre_resume_v6_row_is_retained": source_ids <= set(candidate_ids),
        "every_pre_resume_natural_sequence_is_retained": all(
            final_natural_by_id.get(candidate_id) == natural_sequence
            for candidate_id, natural_sequence in source_natural_by_id.items()
        ),
        "every_pre_resume_v6_payload_field_is_retained": all(
            all(str(final_row_by_id[candidate_id].get(field, "")) == str(value)
                for field, value in source_payload.items())
            for candidate_id, source_payload in source_payload_by_id.items()
        ),
        "all_resumed_rows_have_unique_ids_and_valid_lengths": (
            len(candidate_ids) == len(set(candidate_ids)) and not invalid_rows
        ),
        "initial_and_topup_stage_accounting_is_exact": (
            stage_counts[INITIAL_STAGE] == initial_full_rows
            and stage_counts[TOPUP_STAGE] == len(raw_rows) - initial_full_rows
        ),
        "every_topup_row_uses_cyclic_representation_annotation": all(
            str(row.get("annotation_mode", "")) == ANNOTATION_MODE
            and int(row.get("annotation_representation_ensemble_size", -1))
            == int(row.get("design_length", -2))
            for row in raw_rows
            if str(row.get("source_recovery_stage", "")) == TOPUP_STAGE
        ),
        "every_target_meets_pre_structure_candidate_quota": not targets_below_quota,
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"

    previous_manifest_sha256 = sha256_file(source_paths["manifest"])
    previous_all_sha256 = sha256_file(source_paths["all"])
    backup_dir = backup_source_files(out_dir)
    backup_manifest_sha256 = sha256_file(backup_dir / "generation_manifest.json")
    backup_all_sha256 = sha256_file(backup_dir / "all_candidates.csv")
    generator.atomic_write_csv(out_dir / "all_candidates.csv", raw_rows, raw_fields)
    generator.atomic_write_csv(out_dir / "unique_candidates.csv", unique_rows, unique_fields)
    generator.atomic_write_csv(
        out_dir / "methylated_new_candidates.csv", eligible_rows, unique_fields
    )
    generator.atomic_write_csv(
        out_dir / "target_manifest.csv",
        final_target_manifest,
        union_fields(final_target_manifest),
    )
    generator.atomic_write_csv(
        out_dir / "generation_summary_by_target.csv",
        summary_rows,
        list(summary_rows[0]),
    )

    rows_by_target_seed: MutableMapping[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        rows_by_target_seed[(str(row["target_name"]).upper(), int(row["seed"]))].append(row)
    for (target, seed), seed_rows in sorted(rows_by_target_seed.items()):
        generator.write_seed_fasta(
            out_dir / "generated_fastas" / f"seed_{seed}" / f"{target.lower()}_designs.fasta",
            seed_rows,
        )

    manifest = dict(source_manifest)
    manifest.update(
        {
            "quality_gate": quality_gate,
            "quality_checks": quality_checks,
            "protocol": plan["protocol"],
            "recovery_mode": RECOVERY_MODE,
            "scientific_reason": (
                "The complete representation-invariant V6 run is retained. Reserve "
                "sampling is restricted to targets below their frozen structure quota."
            ),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "source_v6_generation_manifest_sha256_before_resume": previous_manifest_sha256,
            "source_v6_all_candidates_sha256_before_resume": previous_all_sha256,
            "source_v6_backup_dir": str(backup_dir),
            "source_v6_initial_backup_manifest_sha256": backup_manifest_sha256,
            "source_v6_initial_backup_all_candidates_sha256": backup_all_sha256,
            "source_v6_quality_gate_before_resume": source_validation[
                "source_quality_gate"
            ],
            "source_v6_false_checks_before_resume": source_validation[
                "source_false_checks"
            ],
            "source_v6_row_validation": source_row_validation,
            "source_v6_annotation_audit_before_resume": source_annotation_audit,
            "source_v6_unique_candidates_before_resume": len(source_unique_rows),
            "source_v6_eligible_candidates_before_resume": len(source_eligible_rows),
            "source_v6_raw_candidates_retained": initial_full_rows,
            "adaptive_topup_raw_candidates": len(raw_rows) - initial_full_rows,
            "adaptive_topup_initial_shortfall_targets": all_initial_shortfalls,
            "adaptive_topup_rows_by_target": dict(sorted(topup_rows_by_target.items())),
            "adaptive_topup_seeds_by_target": {
                target: values
                for target, values in sorted(topup_seeds_by_target.items())
            },
            "adaptive_topup_checkpoints": topup_checkpoints,
            "adaptive_topup_budget": {
                "reserve_seeds": reserve_seeds,
                "draws_per_reserve_seed": int(args.draws_per_reserve_seed),
                "maximum_draws_per_target_total": int(
                    args.max_topup_draws_per_target
                ),
                "maximum_draws_per_target_per_resume": int(
                    args.max_topup_draws_per_target
                ),
                "check_interval_draws": int(args.check_interval_draws),
                "quota_margin": int(args.quota_margin),
            },
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "model_expert_qc_protocol": metadata.get("protocol"),
            "representation_audit_json": str(audit_path),
            "representation_audit_sha256": sha256_file(audit_path),
            "raw_candidates_expected": len(raw_rows),
            "raw_candidates_generated": len(raw_rows),
            "unique_candidates": len(unique_rows),
            "new_methylated_candidates_for_permeability": len(eligible_rows),
            "targets_below_pre_permeability_quota": targets_below_quota,
            "annotation_payload_canonicalization": canonicalization,
            "annotation_stability_audit": annotation_audit,
            "permeability_status": "DEFERRED_UNTIL_STRUCTURE_RETURNS",
            "permeability_input_rows": 0,
            "native_permeability_controls": 0,
        }
    )
    generator.atomic_write_json(out_dir / "generation_manifest.json", manifest)

    print("\n===== V6 QUOTA RESUME COMPLETE =====", flush=True)
    print(f"Original V6 rows retained: {initial_full_rows}", flush=True)
    print(f"Adaptive top-up rows: {len(raw_rows) - initial_full_rows}", flush=True)
    print(f"Final new methylated candidates: {len(eligible_rows)}", flush=True)
    print(f"Targets below quota: {targets_below_quota}", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        raise RuntimeError(
            "V6 adaptive quota recovery failed; all outputs were preserved: "
            + ", ".join(false_checks(quality_checks))
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--source-run-dir", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--out-dir", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument(
        "--representation-audit-json", default=str(DEFAULT_REPRESENTATION_AUDIT)
    )
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--old-designs-csv", default=str(DEFAULT_OLD))
    parser.add_argument("--prior-designs-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--draws-per-reserve-seed", type=int, default=1_000)
    parser.add_argument("--max-topup-draws-per-target", type=int, default=12_000)
    parser.add_argument("--check-interval-draws", type=int, default=200)
    parser.add_argument("--quota-margin", type=int, default=5)
    parser.add_argument(
        "--reserve-seeds", type=int, nargs="+", default=list(DEFAULT_RESERVE_SEEDS)
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
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
        raise ValueError(
            "Batch and top-up budgets must be positive; quota margin cannot be negative"
        )
    run(args)


if __name__ == "__main__":
    main()
