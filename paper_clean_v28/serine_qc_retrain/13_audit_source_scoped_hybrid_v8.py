#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cyclic-representation audit for the source-scoped V8 checkpoint.

The frozen 151-record set is a paired internal audit set, not a newly blind
publication test.  This stage nevertheless provides the exact safety checks
needed before any target-directed sequence search: model/source pinning,
V6-noninferior sensitivity, all-start physical remapping, and explicit short
peptide (length 6/7) stratification.
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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
AUDITOR_PATH = SCRIPT_PATH.with_name("07_audit_cyclic_representation_equivariance.py")
COMMON_PATH = REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"
MODEL_UTILS_PATH = REPO_ROOT / "model_utils.py"
NMETHYL_CONFIG_PATH = REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"
DEFAULT_MODEL = V8_ROOT / "model" / "frankenstein_v28_source_scoped_hybrid_v8.pt"
DEFAULT_MODEL_MANIFEST = V8_ROOT / "model" / "expert_source_composition_manifest.json"
DEFAULT_TEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_order_balanced_v3"
    / "data"
    / "test_serine_provenance_corrected.jsonl"
)
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_BEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "best_designs.csv"
)
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_representation_v6.json")
DEFAULT_OUT = V8_ROOT / "representation_audit"

V8_EXPERT_PROTOCOL = (
    "canonical_shared_v6_non_ser_v7_ser_cyclic_representation_hybrid_v8"
)
V8_AUDIT_PROTOCOL = (
    "cyclic_representation_frozen_audit_source_scoped_hybrid_v8"
)
V8_AUTHORIZATION = "SOURCE_SCOPED_HYBRID_V8_AUTHORIZED_FOR_DIRECTED_RECOVERY"
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
    "residues_probability_mean_for_ranking_representation_min_for_release"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_auditor_module() -> Any:
    spec = importlib.util.spec_from_file_location("source_scoped_v8_base_auditor", AUDITOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load representation auditor: {AUDITOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record_lengths(records: Sequence[Mapping[str, Any]], auditor: Any) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for index, record in enumerate(records):
        name = auditor.record_name(record, index)
        sequences = [
            str(value)
            for key, value in record.items()
            if str(key).startswith("seq_chain_") and str(value)
        ]
        if len(sequences) != 1:
            raise RuntimeError(f"Frozen test record {name} is not peptide-only")
        result[name] = len(sequences[0])
    return result


def length_strata(
    position_rows: Sequence[Mapping[str, Any]],
    lengths: Mapping[str, int],
    auditor: Any,
    threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    strata = ((6,), (7,), (6, 7), tuple(sorted(set(lengths.values()))))
    seen: set[tuple[int, ...]] = set()
    for values in strata:
        key = tuple(values)
        if key in seen:
            continue
        seen.add(key)
        subset = [
            row
            for row in position_rows
            if int(lengths[str(row["sample_name"]).upper()]) in set(values)
        ]
        if not subset:
            rows.append(
                {
                    "lengths": ";".join(str(value) for value in values),
                    "records": sum(length in set(values) for length in lengths.values()),
                    "positions": 0,
                    "methyl_positives": 0,
                    "natural_negatives": 0,
                    "auc": "",
                    "precision_at_0_6": "",
                    "recall_at_0_6": "",
                    "f1_at_0_6": "",
                    "false_positive_rate_at_0_6": "",
                }
            )
            continue
        summary = auditor.metric_summary(
            subset, "probability_representation_ensemble", threshold
        )
        fixed = summary["overall_at_threshold"]
        rows.append(
            {
                "lengths": ";".join(str(value) for value in values),
                "records": sum(length in set(values) for length in lengths.values()),
                "positions": len(subset),
                "methyl_positives": int(fixed["tp"]) + int(fixed["fn"]),
                "natural_negatives": int(fixed["tn"]) + int(fixed["fp"]),
                "auc": summary["overall_auc"],
                "precision_at_0_6": fixed["precision"],
                "recall_at_0_6": fixed["recall"],
                "f1_at_0_6": fixed["f1"],
                "false_positive_rate_at_0_6": fixed["false_positive_rate"],
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("V8 representation audit requires PyTorch") from exc
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = Path(args.model_path).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    test_path = Path(args.test_jsonl).resolve()
    native_path = Path(args.native_jsonl).resolve()
    best_path = Path(args.best_csv).resolve()
    plan_path = Path(args.plan).resolve()
    out_dir = Path(args.out_dir).resolve()
    immutable_inputs = (
        model_path,
        model_manifest_path,
        test_path,
        native_path,
        best_path,
        plan_path,
        SCRIPT_PATH,
        AUDITOR_PATH,
        COMMON_PATH,
        MODEL_UTILS_PATH,
        NMETHYL_CONFIG_PATH,
    )
    if any(paths_overlap(out_dir, path) for path in immutable_inputs):
        raise ValueError("V8 representation output overlaps an immutable input")
    for required in (
        model_path,
        model_manifest_path,
        test_path,
        native_path,
        best_path,
        plan_path,
        AUDITOR_PATH,
        COMMON_PATH,
        MODEL_UTILS_PATH,
        NMETHYL_CONFIG_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"V8 representation output already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_manifest = read_json(model_manifest_path)
    source_files_are_current = True
    for path_field, hash_field in (
        ("canonical_checkpoint", "canonical_checkpoint_sha256"),
        ("v6_checkpoint", "v6_checkpoint_sha256"),
        ("v6_manifest", "v6_manifest_sha256"),
        ("v7_checkpoint", "v7_checkpoint_sha256"),
        ("v7_manifest", "v7_manifest_sha256"),
    ):
        source_path = Path(str(model_manifest.get(path_field, ""))).resolve()
        try:
            source_path.relative_to(REPO_ROOT.resolve())
        except ValueError:
            source_files_are_current = False
            break
        if not source_path.is_file() or sha256_file(source_path) != str(
            model_manifest.get(hash_field, "")
        ):
            source_files_are_current = False
            break
    if not (
        model_manifest.get("quality_gate") == "PASS"
        and model_manifest.get("protocol") == V8_EXPERT_PROTOCOL
        and int(model_manifest.get("audit_batch_size", -1)) == 8
        and model_manifest.get("checkpoint_artifact_sha256") == sha256_file(model_path)
        and model_manifest.get("test_jsonl_sha256") == sha256_file(test_path)
        and model_manifest.get("composer_program_sha256")
        == sha256_file(SCRIPT_PATH.with_name("12_compose_source_scoped_hybrid_v8.py"))
        and model_manifest.get("trainer_program_sha256")
        == sha256_file(SCRIPT_PATH.with_name("02_retrain_canonical_expert_heads.py"))
        and model_manifest.get("common_program_sha256") == sha256_file(COMMON_PATH)
        and model_manifest.get("model_utils_program_sha256")
        == sha256_file(MODEL_UTILS_PATH)
        and model_manifest.get("nmethyl_config_program_sha256")
        == sha256_file(NMETHYL_CONFIG_PATH)
        and source_files_are_current
    ):
        raise RuntimeError("V8 model composition manifest is absent, failed, or stale")

    auditor = load_auditor_module()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = (
        dict(payload.get("expert_head_qc_metadata", {}))
        if isinstance(payload, Mapping)
        else {}
    )
    state_dict = (
        dict(payload.get("model_state_dict", {}))
        if isinstance(payload, Mapping)
        else {}
    )
    observed_state_hashes = {
        key: tensor_sha256(value) for key, value in sorted(state_dict.items())
    }
    state_source_by_key = dict(metadata.get("state_source_by_key") or {})
    expected_state_source = {}
    for key in state_dict:
        if key in {"experts.15.weight", "experts.15.bias"}:
            expected_state_source[key] = "v7_serine"
        elif key.startswith("experts."):
            expected_state_source[key] = "v6_non_ser"
        else:
            expected_state_source[key] = "canonical_shared"
    del payload
    if not (
        metadata.get("protocol") == V8_EXPERT_PROTOCOL
        and metadata.get("expert_scope") == "residue-source-scoped-hybrid"
        and metadata.get("optimization_performed") is False
        and list(metadata.get("active_expert_tokens", []))
        == list("ACDEFGHIKLMNPQRSTVWY")
        and not {
            "changed_state_keys",
            "preserved_state_key_hashes",
            "parent_checkpoint_sha256",
            "train_jsonl_sha256",
            "best_epoch",
        }
        & set(metadata)
        and dict(metadata.get("source_checkpoint_sha256") or {})
        == {
            "canonical": model_manifest.get("canonical_checkpoint_sha256"),
            "v6": model_manifest.get("v6_checkpoint_sha256"),
            "v7": model_manifest.get("v7_checkpoint_sha256"),
        }
        and dict(metadata.get("source_manifest_sha256") or {})
        == {
            "v6": model_manifest.get("v6_manifest_sha256"),
            "v7": model_manifest.get("v7_manifest_sha256"),
        }
        and dict(metadata.get("composed_state_key_sha256") or {})
        == observed_state_hashes
        and state_source_by_key == expected_state_source
        and int(metadata.get("minimum_order_coverage_epochs", 0)) >= 30
        and bool(metadata.get("cyclic_representation_augmentation"))
        and metadata.get("training_cyclic_representation_policy")
        == REQUIRED_TRAINING_REPRESENTATION_POLICY
        and metadata.get("training_decoding_order_policy")
        == REQUIRED_TRAINING_ORDER_POLICY
        and metadata.get("deployment_annotation_policy") == REQUIRED_DEPLOYMENT_POLICY
    ):
        raise RuntimeError("V8 checkpoint metadata does not prove cyclic source compatibility")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU audit requires --allow-cpu")
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

    plan = read_json(plan_path)
    targets = [str(row["target_name"]).upper() for row in plan["targets"]]
    if len(targets) != 17 or len(set(targets)) != 17 or any(not name for name in targets):
        raise RuntimeError("V8 target plan must contain exactly 17 unique named targets")
    selected_chains = auditor.selected_chain_index(auditor.read_csv(best_path))
    if set(targets) - set(selected_chains):
        raise RuntimeError("Selected-chain index is incomplete for V8 targets")
    test_records = auditor.read_jsonl(test_path)
    native_records = auditor.read_jsonl(native_path)
    native_names = [
        auditor.record_name(record, index)
        for index, record in enumerate(native_records)
    ]
    if (
        len(native_records) != 17
        or len(set(native_names)) != 17
        or any(not name for name in native_names)
        or set(native_names) != set(targets)
    ):
        raise RuntimeError(
            "Native audit input must contain exactly one named record per planned target"
        )
    model = auditor.load_v28_model(str(model_path), device)
    model.eval()
    position_rows, decoder_summary, representation_summary = auditor.evaluate_heldout(
        model,
        test_records,
        device,
        int(args.batch_size),
        float(args.threshold),
        float(args.temperature),
    )
    native_detail, native_summary = auditor.audit_native_targets(
        model,
        native_records,
        selected_chains,
        targets,
        device,
        float(args.temperature),
        float(args.threshold),
    )
    lengths = record_lengths(test_records, auditor)
    short_metrics = length_strata(
        position_rows, lengths, auditor, float(args.threshold)
    )

    v6_summary = dict(model_manifest["v6_test"])
    v8_composition_summary = dict(model_manifest["v8_test"])
    current_fixed = representation_summary["overall_at_threshold"]
    release_floor_summary = auditor.metric_summary(
        position_rows, "probability_representation_min", float(args.threshold)
    )
    release_floor_fixed = release_floor_summary["overall_at_threshold"]
    release_floor_serine = release_floor_summary["serine"]
    v6_fixed = v6_summary["overall_at_threshold"]
    composition_fixed = v8_composition_summary["overall_at_threshold"]
    serine = representation_summary["serine"]
    v6_serine = v6_summary["serine"]
    composition_serine = v8_composition_summary["serine"]
    serine_tradeoff = dict(
        dict(model_manifest.get("metric_gate_provenance") or {}).get(
            "serine_auc_tradeoff"
        )
        or {}
    )
    try:
        recorded_serine_auc_delta = float(serine_tradeoff["v8_minus_v6_auc"])
        composition_serine_auc_delta = float(composition_serine["auc"]) - float(
            v6_serine["auc"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("V8 model manifest lacks the Ser AUC trade-off audit") from exc
    confusion_fields = ("tp", "tn", "fp", "fn")
    proline = representation_summary["proline"]
    native_lengths = {
        str(row["target_name"]): int(row["peptide_length"]) for row in native_summary
    }
    native_detail_is_finite = all(
        math.isfinite(float(row["mapped_probability"]))
        and 0.0 <= float(row["mapped_probability"]) <= 1.0
        for row in native_detail
    )
    native_summary_is_finite = True
    for row in native_summary:
        try:
            length = int(row["peptide_length"])
            probabilities = [
                float(value)
                for value in json.loads(str(row["ensemble_probability_vector"]))
            ]
            spans = [
                float(value)
                for value in json.loads(
                    str(row["ensemble_representation_span_vector"])
                )
            ]
            scalars = (
                float(row["raw_maximum_single_tensor_position_share"]),
                float(row["maximum_ensemble_recompute_difference"]),
            )
            valid = (
                len(probabilities) == length
                and len(spans) == length
                and all(
                    math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in probabilities
                )
                and all(math.isfinite(value) and value >= 0.0 for value in spans)
                and all(math.isfinite(value) and value >= 0.0 for value in scalars)
                and scalars[0] <= 1.0
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
        native_summary_is_finite = native_summary_is_finite and valid
    quality_checks = {
        "model_composition_and_source_hashes_are_pinned": True,
        "frozen_audit_has_151_records_and_1505_positions": (
            len(test_records) == 151 and len(position_rows) == 1505
        ),
        "representation_metrics_recompute_composition_metrics": (
            int(current_fixed["tp"]) == int(composition_fixed["tp"])
            and int(current_fixed["tn"]) == int(composition_fixed["tn"])
            and int(current_fixed["fp"]) == int(composition_fixed["fp"])
            and int(current_fixed["fn"]) == int(composition_fixed["fn"])
        ),
        "recall_at_0_6_is_non_inferior_to_v6": (
            float(current_fixed["recall"]) + 1e-12 >= float(v6_fixed["recall"])
        ),
        "f1_at_0_6_is_non_inferior_to_v6": (
            float(current_fixed["f1"]) + 1e-12 >= float(v6_fixed["f1"])
        ),
        "serine_threshold_operating_point_is_non_degrading_vs_v6_and_recomputes_composition": (
            all(
                int(serine[field]) == int(composition_serine[field])
                for field in confusion_fields
            )
            and int(serine["tp"]) >= int(v6_serine["tp"])
            and int(serine["tn"]) >= int(v6_serine["tn"])
            and int(serine["fp"]) <= int(v6_serine["fp"])
            and int(serine["fn"]) <= int(v6_serine["fn"])
        ),
        "serine_auc_recomputes_composition_audit": (
            serine["auc"] is not None
            and composition_serine["auc"] is not None
            and abs(float(serine["auc"]) - float(composition_serine["auc"]))
            <= 1e-12
        ),
        "serine_auc_tradeoff_is_carried_forward_without_redefinition": (
            math.isfinite(recorded_serine_auc_delta)
            and abs(recorded_serine_auc_delta - composition_serine_auc_delta)
            <= 1e-12
            and serine_tradeoff.get("auc_gate_policy")
            == (
                "report observed V8-minus-V6 AUC exactly; do not assert zero-margin "
                "non-inferiority post hoc; retain the absolute Ser-AUC safety floor"
            )
        ),
        "overall_auc_ge_0_85": float(representation_summary["overall_auc"]) >= 0.85,
        "overall_precision_at_0_6_ge_0_75": float(current_fixed["precision"]) >= 0.75,
        "overall_fpr_at_0_6_le_0_10": float(current_fixed["false_positive_rate"]) <= 0.10,
        "release_floor_precision_at_0_6_ge_0_75": (
            float(release_floor_fixed["precision"]) >= 0.75
        ),
        "release_floor_fpr_at_0_6_le_0_10": (
            float(release_floor_fixed["false_positive_rate"]) <= 0.10
        ),
        "release_floor_serine_recall_at_0_6_ge_0_40": (
            float(release_floor_serine["recall"]) >= 0.40
        ),
        "heldout_hard_calls_have_zero_cyclic_start_threshold_disagreement": (
            int(
                representation_summary[
                    "representation_threshold_disagreement_positions"
                ]
            )
            == 0
        ),
        "serine_auc_ge_0_70": serine["auc"] is not None and float(serine["auc"]) >= 0.70,
        "serine_recall_at_0_6_ge_0_40": float(serine["recall"]) >= 0.40,
        "serine_fpr_at_0_6_le_0_25": float(serine["false_positive_rate"]) <= 0.25,
        "proline_fpr_at_0_6_le_0_05": float(proline["false_positive_rate"]) <= 0.05,
        "length_6_and_7_are_explicitly_stratified": (
            {row["lengths"] for row in short_metrics} >= {"6", "7", "6;7"}
        ),
        "native_3wne_and_3zgc_lengths_are_6_and_7": (
            native_lengths.get("3WNE") == 6 and native_lengths.get("3ZGC") == 7
        ),
        "all_17_native_targets_use_every_cyclic_start": (
            len(native_summary) == 17
            and len(native_lengths) == 17
            and set(native_lengths) == set(targets)
            and all(
                int(row["representation_count"]) == int(row["peptide_length"])
                for row in native_summary
            )
        ),
        "all_17_native_target_hard_calls_are_stable_across_cyclic_starts": (
            len(native_summary) == 17
            and all(
                int(row["raw_all_representations_same_physical_annotation"]) == 1
                for row in native_summary
            )
        ),
        "mapped_ensemble_recomputes_from_raw_rotations": all(
            float(row["maximum_ensemble_recompute_difference"]) <= 1e-6
            for row in native_summary
        ),
        "all_probabilities_and_diagnostics_are_finite": all(
            math.isfinite(float(row[field]))
            for row in position_rows
            for field in (
                "probability_decoder_order_only",
                "probability_representation_ensemble",
                "probability_representation_std",
                "probability_representation_span",
            )
        )
        and all(
            0.0 <= float(row["probability_decoder_order_only"]) <= 1.0
            and 0.0 <= float(row["probability_representation_ensemble"]) <= 1.0
            and float(row["probability_representation_std"]) >= 0.0
            and float(row["probability_representation_span"]) >= 0.0
            for row in position_rows
        )
        and native_detail_is_finite
        and native_summary_is_finite,
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    authorization = (
        V8_AUTHORIZATION if quality_gate == "PASS" else "BLOCKED_DO_NOT_SEARCH_OR_RELEASE"
    )

    frozen_position_path = out_dir / "frozen_test_position_probabilities.csv"
    length_metrics_path = out_dir / "frozen_test_metrics_by_length.csv"
    native_detail_path = out_dir / "native_target_representation_probabilities.csv"
    native_summary_path = out_dir / "native_target_representation_summary.csv"
    atomic_write_csv(
        frozen_position_path,
        position_rows,
        list(position_rows[0]),
    )
    atomic_write_csv(
        length_metrics_path,
        short_metrics,
        list(short_metrics[0]),
    )
    atomic_write_csv(
        native_detail_path,
        native_detail,
        list(native_detail[0]),
    )
    atomic_write_csv(
        native_summary_path,
        native_summary,
        list(native_summary[0]),
    )
    report = {
        "quality_gate": quality_gate,
        "release_authorization": authorization,
        "protocol": V8_AUDIT_PROTOCOL,
        "test_reuse_limitation": model_manifest["test_reuse_limitation"],
        "development_status": model_manifest["development_status"],
        "serine_auc_tradeoff": serine_tradeoff,
        "release_floor_metrics": release_floor_summary,
        "quality_checks": quality_checks,
        "device": str(device),
        "audit_batch_size": int(args.batch_size),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "representation_auditor_program_sha256": sha256_file(SCRIPT_PATH),
        "equivariance_auditor_program_sha256": sha256_file(AUDITOR_PATH),
        "common_program_sha256": sha256_file(COMMON_PATH),
        "model_utils_program_sha256": sha256_file(MODEL_UTILS_PATH),
        "nmethyl_config_program_sha256": sha256_file(NMETHYL_CONFIG_PATH),
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
        "temperature": float(args.temperature),
        "threshold": float(args.threshold),
        "strict_threshold_operator": ">",
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "model_expert_qc_protocol": V8_EXPERT_PROTOCOL,
        "test_jsonl": str(test_path),
        "test_jsonl_sha256": sha256_file(test_path),
        "native_jsonl": str(native_path),
        "native_jsonl_sha256": sha256_file(native_path),
        "best_csv": str(best_path),
        "best_csv_sha256": sha256_file(best_path),
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "decoder_order_only_frozen_test": decoder_summary,
        "cyclic_representation_frozen_test": representation_summary,
        "frozen_test_metrics_by_length": short_metrics,
        "native_target_summary": native_summary,
        "annotation_mode": auditor.REPRESENTATION_MODE,
        "annotation_context_policy": auditor.ANNOTATION_CONTEXT,
        "structure_handoff_status": "BLOCKED_PENDING_SEARCH_AND_MANUAL_REVIEW",
        "artifacts": {
            "frozen_test_positions": {
                "path": str(frozen_position_path),
                "sha256": sha256_file(frozen_position_path),
            },
            "length_metrics": {
                "path": str(length_metrics_path),
                "sha256": sha256_file(length_metrics_path),
            },
            "native_probabilities": {
                "path": str(native_detail_path),
                "sha256": sha256_file(native_detail_path),
            },
            "native_summary": {
                "path": str(native_summary_path),
                "sha256": sha256_file(native_summary_path),
            },
        },
    }
    atomic_write_json(out_dir / "cyclic_representation_audit.json", report)

    print("===== SOURCE-SCOPED V8 CYCLIC REPRESENTATION AUDIT =====", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    print(f"Authorization: {authorization}", flush=True)
    print(
        "Recall/F1 @0.6: "
        f"{float(current_fixed['recall']):.4f} / {float(current_fixed['f1']):.4f}",
        flush=True,
    )
    if quality_gate != "PASS":
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError("V8 representation audit failed: " + ", ".join(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-manifest", default=str(DEFAULT_MODEL_MANIFEST))
    parser.add_argument("--test-jsonl", default=str(DEFAULT_TEST))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--best-csv", default=str(DEFAULT_BEST))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.batch_size) != 8:
        raise ValueError("V8 representation audit is frozen to --batch-size 8")
    if float(args.temperature) != 0.5 or float(args.threshold) != 0.6:
        raise ValueError("V8 audit is frozen to T=0.5 and strict threshold >0.6")
    run(args)


if __name__ == "__main__":
    main()
