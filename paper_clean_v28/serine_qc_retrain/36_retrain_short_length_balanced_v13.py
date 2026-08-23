#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V13: repair the short-peptide blind spot without changing V11 geometry.

V11 removed the artificial cyclic cut, but its deterministic inner split put
no length-6 records and only two length-7 records in validation.  Those lengths
include the two weakest generation targets (3WNE and 3ZGC).  This program starts
from the promoted V11 cyclic-native checkpoint, keeps the trunk, base head and
cyclic positional projection bitwise frozen, and fine-tunes only the twenty
methyl expert heads using:

* a deterministic record-disjoint split with every peptide length represented
  on both sides as well as positive/negative support for every expert;
* fixed, declared repeat weights for scarce short records (real labels only);
* the unchanged full cyclic-start x decoder-order objective and strict T=0.5
  deployment representation.

The 151-record historical diagnostic set is opened only after epoch selection.
It is reported but never used to promote the checkpoint.  Actual target-level
generation yield is a separate V13 gate; this script cannot declare the final
17 x 100 handoff successful.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
LEGACY_TRAINER_PATH = SCRIPT_PATH.with_name(
    "02_retrain_canonical_expert_heads.py"
)
V11_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "cyclic_native_v11_1700_monomer"
DEFAULT_MODEL = V11_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
DEFAULT_TRAIN = REPO_ROOT / "v9_inputs" / "train_serine_provenance_corrected.jsonl"
DEFAULT_TEST = REPO_ROOT / "v9_inputs" / "test_serine_provenance_corrected.jsonl"
DEFAULT_OUT = (
    REPO_ROOT / "paper_clean_v28_outputs" / "methyl_yield_v13_1700" / "model"
)
V11_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_cyclic_native_relative_positions_v11"
)
V13_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_cyclic_native_"
    "short_length_balanced_v13"
)
V13_SPLIT_PROTOCOL = (
    "record_disjoint_length_stratified_per_expert_class_supported_v13"
)
V13_TRAINING_REPRESENTATION_POLICY = (
    "boundary_marginalized_cyclic_relative_positions_with_all_physical_starts_"
    "retained_as_an_explicit_equivariance_verification_grid"
)
V13_REPEAT_FACTORS = {6: 5, 7: 4, 8: 2}
V13_SHORT_LENGTHS = (6, 7)


def load_legacy_trainer() -> Any:
    spec = importlib.util.spec_from_file_location("v13_legacy_trainer", LEGACY_TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {LEGACY_TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peptide_length(record: Mapping[str, Any]) -> int:
    masked = [str(value) for value in record.get("masked_list", [])]
    if len(masked) != 1:
        raise RuntimeError("V13 expects exactly one masked peptide chain per record")
    sequence = str(record.get(f"seq_chain_{masked[0]}", ""))
    if not sequence:
        raise RuntimeError("V13 encountered an empty peptide sequence")
    return len(sequence)


def counts_by_length(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        str(length): count
        for length, count in sorted(Counter(peptide_length(row) for row in records).items())
    }


def deterministic_length_stratified_split(
    records: Sequence[Mapping[str, Any]],
    validation_fraction: float,
    seed: int,
    per_base_binary_counts_fn: Any,
    supported_methyl_bases: Sequence[str],
    record_name_fn: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Split records with length coverage and per-expert class support."""

    if not 0.05 <= float(validation_fraction) <= 0.40:
        raise ValueError("validation_fraction must be between 0.05 and 0.40")
    source = [dict(row) for row in records]
    groups: Dict[int, List[int]] = defaultdict(list)
    for index, row in enumerate(source):
        groups[peptide_length(row)].append(index)
    if any(len(indices) < 2 for indices in groups.values()):
        raise RuntimeError("Every peptide length needs at least two records")
    target_size = max(1, min(len(source) - 1, round(len(source) * validation_fraction)))

    for attempt in range(10_000):
        rng = random.Random(seed + attempt)
        validation_indices: set[int] = set()
        remaining_by_length: Dict[int, List[int]] = {}
        for length, original_indices in sorted(groups.items()):
            indices = list(original_indices)
            rng.shuffle(indices)
            minimum_validation = 2 if len(indices) >= 4 else 1
            requested = max(
                minimum_validation,
                min(len(indices) - 1, round(len(indices) * validation_fraction)),
            )
            validation_indices.update(indices[:requested])
            remaining_by_length[length] = indices[requested:]

        if len(validation_indices) < target_size:
            fill = [
                index
                for length in sorted(remaining_by_length)
                for index in remaining_by_length[length][:-1]
                if index not in validation_indices
            ]
            rng.shuffle(fill)
            validation_indices.update(fill[: target_size - len(validation_indices)])
        elif len(validation_indices) > target_size:
            removable = list(validation_indices)
            rng.shuffle(removable)
            for index in removable:
                length = peptide_length(source[index])
                length_validation_count = sum(
                    peptide_length(source[candidate]) == length
                    for candidate in validation_indices
                )
                if length_validation_count > 1 and len(validation_indices) > target_size:
                    validation_indices.remove(index)

        if len(validation_indices) != target_size:
            continue
        development = [
            row for index, row in enumerate(source) if index not in validation_indices
        ]
        validation = [
            row for index, row in enumerate(source) if index in validation_indices
        ]
        if set(counts_by_length(development)) != set(counts_by_length(source)):
            continue
        if set(counts_by_length(validation)) != set(counts_by_length(source)):
            continue
        if any(
            sum(peptide_length(row) == length for row in validation) < 2
            for length, indices in groups.items()
            if len(indices) >= 4
        ):
            continue
        development_counts = per_base_binary_counts_fn(development)
        validation_counts = per_base_binary_counts_fn(validation)
        if not all(
            development_counts[base]["natural_negative"] > 0
            and development_counts[base]["methyl_positive"] > 0
            and validation_counts[base]["natural_negative"] > 0
            and validation_counts[base]["methyl_positive"] > 0
            for base in sorted(supported_methyl_bases)
        ):
            continue

        development_names = {
            record_name_fn(row, index) for index, row in enumerate(development)
        }
        validation_names = {
            record_name_fn(row, index) for index, row in enumerate(validation)
        }
        if development_names & validation_names:
            raise RuntimeError("Development/validation record-name overlap")
        return development, validation, {
            "protocol": V13_SPLIT_PROTOCOL,
            "seed": int(seed),
            "accepted_seed_offset": attempt,
            "validation_fraction_requested": float(validation_fraction),
            "development_records": len(development),
            "validation_records": len(validation),
            "all_records_by_length": counts_by_length(source),
            "development_records_by_length": counts_by_length(development),
            "validation_records_by_length": counts_by_length(validation),
            "development_record_names": sorted(development_names),
            "validation_record_names": sorted(validation_names),
            "development_counts_by_base": development_counts,
            "validation_counts_by_base": validation_counts,
        }
    raise RuntimeError(
        "Could not create the V13 record-disjoint split with both length and "
        "per-expert class coverage"
    )


def repeat_factor(length: int) -> int:
    return int(V13_REPEAT_FACTORS.get(int(length), 1))


def length_balanced_records(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    effective = Counter()
    for row in records:
        length = peptide_length(row)
        factor = repeat_factor(length)
        for _ in range(factor):
            expanded.append(dict(row))
        effective[length] += factor
    return expanded, {
        "repeat_factors_by_length": {
            str(length): repeat_factor(length)
            for length in sorted({peptide_length(row) for row in records})
        },
        "source_records_by_length": counts_by_length(records),
        "effective_records_by_length": {
            str(length): count for length, count in sorted(effective.items())
        },
        "effective_records": len(expanded),
        "policy": "deterministic_real_record_repetition_no_synthetic_labels",
    }


def validation_bce_by_length(
    trainer: Any,
    model: Any,
    records: Sequence[Mapping[str, Any]],
    device: Any,
    batch_size: int,
    positive_weights: Mapping[int, float],
    worst_start_bce_weight: float,
    representation_consistency_weight: float,
    temperature: float,
) -> Dict[str, float]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[peptide_length(row)].append(dict(row))
    return {
        str(length): trainer.validation_balanced_bce(
            model,
            rows,
            device,
            batch_size,
            positive_weights,
            cyclic_representation_ensemble=True,
            active_base_indices=tuple(range(len(trainer.NATURAL_AA_ALPHABET))),
            worst_start_bce_weight=worst_start_bce_weight,
            representation_consistency_weight=representation_consistency_weight,
            ensemble_temperature=temperature,
            base_sequence_loss_weight=0.0,
        )
        for length, rows in sorted(grouped.items())
    }


def arithmetic_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(sum(float(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN))
    parser.add_argument("--test-jsonl", default=str(DEFAULT_TEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--deployment-temperature", type=float, default=0.5)
    parser.add_argument("--worst-start-bce-weight", type=float, default=1.0)
    parser.add_argument("--representation-consistency-weight", type=float, default=0.25)
    parser.add_argument("--minimum-short-bce-improvement", type=float, default=0.005)
    parser.add_argument("--maximum-long-bce-increase", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 30 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("V13 requires >=30 epochs and positive batch size/learning rate")
    if args.deployment_temperature != 0.5 or args.threshold != 0.6:
        raise ValueError("V13 is frozen to T=0.5 and strict methyl threshold 0.6")
    if args.minimum_short_bce_improvement <= 0.0 or args.maximum_long_bce_increase < 0.0:
        raise ValueError("V13 BCE improvement/tolerance controls are invalid")

    model_path = Path(args.model_path).resolve()
    train_path = Path(args.train_jsonl).resolve()
    test_path = Path(args.test_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    for required in (model_path, train_path, test_path, LEGACY_TRAINER_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = load_legacy_trainer()
    trainer.set_deterministic_seed(args.seed)

    parent_payload = torch.load(model_path, map_location="cpu", weights_only=False)
    parent_qc = dict(parent_payload.get("expert_head_qc_metadata", {}))
    parent_architecture = dict(parent_payload.get("model_architecture_metadata", {}))
    if (
        parent_qc.get("protocol") != V11_EXPERT_PROTOCOL
        or not bool(parent_architecture.get("cyclic_relative_positions"))
        or parent_architecture.get("protocol") != trainer.V11_MODEL_ARCHITECTURE_PROTOCOL
    ):
        raise RuntimeError("V13 must start from the promoted cyclic-native V11 checkpoint")
    del parent_payload

    train_records = trainer.read_jsonl(str(train_path))
    trainer.require_peptide_only_training_context(train_records, "train")
    trainer.require_corrected_counts(train_records, trainer.EXPECTED_TRAIN_COUNTS, "train")
    development, validation, split_manifest = deterministic_length_stratified_split(
        train_records,
        args.validation_fraction,
        args.seed,
        trainer.per_base_binary_counts,
        sorted(trainer.SUPPORTED_METHYL_BASES),
        trainer.record_name,
    )
    weighted_development, development_weighting = length_balanced_records(development)
    weighted_validation, validation_weighting = length_balanced_records(validation)
    active_indices = tuple(range(len(trainer.NATURAL_AA_ALPHABET)))
    positive_weights = trainer.positive_weights_by_base(weighted_development, active_indices)

    print(f"Loading promoted V11 checkpoint: {model_path}", flush=True)
    model = trainer.load_v28_model(str(model_path), device)
    if not bool(model.cyclic_relative_positions):
        raise RuntimeError("Loaded V11 model is not in cyclic-relative mode")
    before_hashes = trainer.state_hashes(model.state_dict())
    parent_validation_by_length = validation_bce_by_length(
        trainer,
        model,
        validation,
        device,
        args.batch_size,
        positive_weights,
        args.worst_start_bce_weight,
        args.representation_consistency_weight,
        args.deployment_temperature,
    )
    parent_weighted_validation = trainer.validation_balanced_bce(
        model,
        weighted_validation,
        device,
        args.batch_size,
        positive_weights,
        cyclic_representation_ensemble=True,
        active_base_indices=active_indices,
        worst_start_bce_weight=args.worst_start_bce_weight,
        representation_consistency_weight=args.representation_consistency_weight,
        ensemble_temperature=args.deployment_temperature,
    )

    history, _best_state, selection = trainer.train_all_expert_heads(
        model,
        weighted_development,
        weighted_validation,
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.early_stopping_patience,
        args.seed,
        cyclic_representation_augmentation=True,
        active_base_indices=active_indices,
        worst_start_bce_weight=args.worst_start_bce_weight,
        representation_consistency_weight=args.representation_consistency_weight,
        ensemble_temperature=args.deployment_temperature,
        train_cyclic_positional_encoding=False,
    )
    after_hashes = trainer.state_hashes(model.state_dict())
    changed_keys = sorted(
        key for key in before_hashes if before_hashes[key] != after_hashes[key]
    )
    selected_validation_by_length = validation_bce_by_length(
        trainer,
        model,
        validation,
        device,
        args.batch_size,
        positive_weights,
        args.worst_start_bce_weight,
        args.representation_consistency_weight,
        args.deployment_temperature,
    )
    selected_weighted_validation = trainer.validation_balanced_bce(
        model,
        weighted_validation,
        device,
        args.batch_size,
        positive_weights,
        cyclic_representation_ensemble=True,
        active_base_indices=active_indices,
        worst_start_bce_weight=args.worst_start_bce_weight,
        representation_consistency_weight=args.representation_consistency_weight,
        ensemble_temperature=args.deployment_temperature,
    )
    available_short = [
        str(length) for length in V13_SHORT_LENGTHS
        if str(length) in parent_validation_by_length
    ]
    parent_short = arithmetic_mean(
        [parent_validation_by_length[length] for length in available_short]
    )
    selected_short = arithmetic_mean(
        [selected_validation_by_length[length] for length in available_short]
    )
    long_lengths = [
        length for length in parent_validation_by_length
        if int(length) not in V13_SHORT_LENGTHS
    ]
    parent_long = arithmetic_mean(
        [parent_validation_by_length[length] for length in long_lengths]
    )
    selected_long = arithmetic_mean(
        [selected_validation_by_length[length] for length in long_lengths]
    )

    test_records = trainer.read_jsonl(str(test_path))
    trainer.require_peptide_only_training_context(test_records, "test")
    trainer.require_corrected_counts(test_records, trainer.EXPECTED_TEST_COUNTS, "test")
    parent_model = trainer.load_v28_model(str(model_path), device)
    parent_test, _parent_residue, _parent_positions = trainer.evaluate(
        parent_model,
        test_records,
        device,
        args.batch_size,
        args.threshold,
        args.deployment_temperature,
        "v11_parent_internal_diagnostic",
        cyclic_representation_ensemble=True,
    )
    selected_test, selected_residue, selected_positions = trainer.evaluate(
        model,
        test_records,
        device,
        args.batch_size,
        args.threshold,
        args.deployment_temperature,
        "v13_short_length_balanced_internal_diagnostic",
        cyclic_representation_ensemble=True,
    )
    del parent_model

    quality_checks = {
        "parent_is_promoted_cyclic_native_v11": True,
        "split_has_every_length_on_both_sides": (
            set(split_manifest["development_records_by_length"])
            == set(split_manifest["validation_records_by_length"])
            == set(split_manifest["all_records_by_length"])
        ),
        "only_twenty_expert_heads_changed": (
            set(changed_keys) == set(trainer.ALL_EXPERT_STATE_KEYS)
        ),
        "weighted_validation_bce_improved": (
            selected_weighted_validation < parent_weighted_validation
        ),
        "short_length_validation_bce_improved_materially": (
            selected_short
            <= parent_short - float(args.minimum_short_bce_improvement)
        ),
        "long_length_validation_bce_noninferior": (
            selected_long
            <= parent_long + float(args.maximum_long_bce_increase)
        ),
        "selected_training_representation_span_le_1e_5": (
            float(selection["best_epoch_maximum_training_representation_span"])
            <= 1e-5
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    out_dir.mkdir(parents=True, exist_ok=True)
    production_path = out_dir / "frankenstein_v28_short_length_balanced_v13.pt"
    candidate_path = out_dir / "frankenstein_v28_short_length_balanced_v13.candidate.pt"
    checkpoint_payload = {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "model_architecture_metadata": parent_architecture,
        "expert_head_qc_metadata": {
            "protocol": V13_EXPERT_PROTOCOL,
            "parent_expert_protocol": V11_EXPERT_PROTOCOL,
            "parent_checkpoint_sha256": sha256_file(model_path),
            "train_jsonl_sha256": sha256_file(train_path),
            "test_jsonl_sha256": sha256_file(test_path),
            "changed_state_keys": changed_keys,
            "expert_scope": "all",
            "active_expert_tokens": list(trainer.NATURAL_AA_ALPHABET),
            "preserved_state_key_hashes": {
                key: value
                for key, value in before_hashes.items()
                if key not in trainer.ALL_EXPERT_STATE_KEYS
            },
            "minimum_order_coverage_epochs": trainer.MINIMUM_ORDER_COVERAGE_EPOCHS,
            "cyclic_representation_augmentation": True,
            "training_cyclic_representation_policy": V13_TRAINING_REPRESENTATION_POLICY,
            "training_decoding_order_policy": (
                "complete_physical_cyclic_start_x_complete_L_decoder_order_grid_"
                "differentiably_meaned_per_start_then_mapped_to_physical_labels"
            ),
            "deployment_annotation_policy": (
                "all_cyclic_starts_and_all_decoder_orders_mapped_to_physical_"
                "residues_probability_mean_for_ranking_representation_min_for_release"
            ),
            "worst_start_bce_weight": float(args.worst_start_bce_weight),
            "representation_consistency_weight": float(
                args.representation_consistency_weight
            ),
            "training_ensemble_temperature": float(args.deployment_temperature),
            "full_physical_start_by_full_decoder_order_grid": True,
            "training_objective": selection["training_objective"],
            "model_architecture_protocol": trainer.V11_MODEL_ARCHITECTURE_PROTOCOL,
            "cyclic_relative_positions": True,
            "cyclic_offset_policy": trainer.V11_CYCLIC_OFFSET_POLICY,
            "trained_cyclic_positional_state_keys": [],
            "inherited_cyclic_positional_state_keys": sorted(
                trainer.V11_POSITIONAL_STATE_KEYS
            ),
            "base_sequence_loss_weight": 0.0,
            "positional_anchor_weight": 0.0,
            "maximum_equivariance_span_tolerance": 1e-5,
            "best_epoch_maximum_training_representation_span": float(
                selection["best_epoch_maximum_training_representation_span"]
            ),
            "v11_base_noninferiority_inherited_bitwise": True,
            "threshold": float(args.threshold),
            "deployment_temperature": float(args.deployment_temperature),
            "split_protocol": V13_SPLIT_PROTOCOL,
            "length_repeat_factors": {
                str(key): value for key, value in sorted(V13_REPEAT_FACTORS.items())
            },
            "short_lengths": list(V13_SHORT_LENGTHS),
        },
    }
    temporary = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    torch.save(checkpoint_payload, temporary)
    os.replace(temporary, candidate_path)
    reloaded = trainer.load_v28_model(str(candidate_path), device)
    if trainer.state_hashes(reloaded.state_dict()) != after_hashes:
        raise RuntimeError("V13 candidate failed strict state round-trip")
    if quality_gate == "PASS":
        os.replace(candidate_path, production_path)
        artifact_path = production_path
    else:
        artifact_path = candidate_path

    manifest = {
        "quality_gate": quality_gate,
        "checkpoint_ready_for_generation": quality_gate == "PASS",
        "protocol": V13_EXPERT_PROTOCOL,
        "quality_checks": quality_checks,
        "device": str(device),
        "parent_checkpoint": str(model_path),
        "parent_checkpoint_sha256": sha256_file(model_path),
        "output_checkpoint": str(production_path) if quality_gate == "PASS" else None,
        "checkpoint_artifact": str(artifact_path),
        "checkpoint_artifact_sha256": sha256_file(artifact_path),
        "artifacts": {
            "promoted_checkpoint" if quality_gate == "PASS" else "blocked_candidate": {
                "path": str(artifact_path),
                "sha256": sha256_file(artifact_path),
            }
        },
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "dependencies": {
            "legacy_trainer": {
                "path": str(LEGACY_TRAINER_PATH),
                "sha256": sha256_file(LEGACY_TRAINER_PATH),
            }
        },
        "inputs": {
            "parent_checkpoint": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "train_jsonl": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "test_jsonl": {"path": str(test_path), "sha256": sha256_file(test_path)},
        },
        "changed_state_keys": changed_keys,
        "split": split_manifest,
        "development_weighting": development_weighting,
        "validation_weighting": validation_weighting,
        "parent_validation_bce_by_length": parent_validation_by_length,
        "selected_validation_bce_by_length": selected_validation_by_length,
        "parent_short_length_macro_bce": parent_short,
        "selected_short_length_macro_bce": selected_short,
        "parent_long_length_macro_bce": parent_long,
        "selected_long_length_macro_bce": selected_long,
        "parent_weighted_validation_bce": parent_weighted_validation,
        "selected_weighted_validation_bce": selected_weighted_validation,
        "training_selection": selection,
        "internal_development_audit_not_blind_outer_test": {
            "parent_v11": parent_test,
            "selected_v13": selected_test,
        },
        "structure_handoff_status": "BLOCKED_PENDING_FIXED_BUDGET_GENERATION_YIELD_GATE",
    }
    trainer.atomic_write_json(out_dir / "v13_short_length_retrain_manifest.json", manifest)
    trainer.atomic_write_csv(out_dir / "training_history.csv", history, list(history[0]))
    trainer.atomic_write_csv(
        out_dir / "internal_test_metrics_by_residue.csv",
        selected_residue,
        list(selected_residue[0]),
    )
    trainer.atomic_write_csv(
        out_dir / "internal_test_position_probabilities.csv",
        selected_positions,
        list(selected_positions[0]),
    )
    print("===== V13 SHORT-LENGTH REPAIR COMPLETE =====", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    print(
        f"Short-length validation BCE: {parent_short:.6f} -> {selected_short:.6f}",
        flush=True,
    )
    print(
        f"Weighted validation BCE: {parent_weighted_validation:.6f} -> "
        f"{selected_weighted_validation:.6f}",
        flush=True,
    )
    print(f"Checkpoint artifact: {artifact_path}", flush=True)
    if quality_gate != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
