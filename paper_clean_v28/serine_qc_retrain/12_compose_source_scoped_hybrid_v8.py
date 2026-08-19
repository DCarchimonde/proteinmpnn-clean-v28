#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compose and audit the source-scoped cyclic expert checkpoint (V8).

V7 correctly repaired the Ser provenance label, but it also restored the other
nineteen expert heads to the old canonical checkpoint.  On the unchanged
1,505-position frozen audit test this deliberately discarded 77 true positives.
That loss is not caused by Ser training: with the shared trunk frozen, every
expert is an independent linear head and its loss is selected only at positions
whose natural parent is that expert's residue.

V8 therefore applies one deterministic, provenance-defined source rule:

* shared trunk, embeddings, decoder and base head: canonical clean-V28;
* Ser expert weight and bias: provenance-corrected Ser-only V7;
* every non-Ser expert weight and bias: corrected-label, cyclic-trained V6.

This is not a metric-tuned blend.  No weights are averaged and there are no new
hyperparameters.  The output is promoted only if a paired frozen-test audit
proves that every Ser probability is exactly inherited from V7, every non-Ser
probability is exactly inherited from V6, and V8 is non-inferior to V6 for
recall and F1 at the frozen strict ``>0.6`` decision rule.  Ser AUC is reported
as an explicit post-hoc trade-off rather than forced through a zero-margin
non-inferiority claim: V8 inherits the V7 Ser ranking exactly, while its Ser
threshold operating point must not lose true positives or add false positives
relative to V6.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TRAINER_PATH = SCRIPT_PATH.with_name("02_retrain_canonical_expert_heads.py")
COMMON_PATH = REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"
MODEL_UTILS_PATH = REPO_ROOT / "model_utils.py"
NMETHYL_CONFIG_PATH = REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"
V6_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_cyclic_representation_v6"
V7_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_serine_only_cyclic_v7"
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"

DEFAULT_CANONICAL = REPO_ROOT / "frankenstein_v28.pt"
DEFAULT_V6_MODEL = V6_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
DEFAULT_V6_MANIFEST = V6_ROOT / "model" / "expert_heads_retrain_manifest.json"
DEFAULT_V7_MODEL = V7_ROOT / "model" / "frankenstein_v28_serine_only_qc.pt"
DEFAULT_V7_MANIFEST = V7_ROOT / "model" / "expert_heads_retrain_manifest.json"
DEFAULT_TEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_order_balanced_v3"
    / "data"
    / "test_serine_provenance_corrected.jsonl"
)
DEFAULT_OUT = V8_ROOT / "model"

V6_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_"
    "cyclic_representation_augmented_v6"
)
V7_EXPERT_PROTOCOL = (
    "canonical_clean_v28_serine_only_corrected_labels_"
    "cyclic_representation_augmented_v7"
)
V8_EXPERT_PROTOCOL = (
    "canonical_shared_v6_non_ser_v7_ser_cyclic_representation_hybrid_v8"
)
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SERINE_INDEX = NATURAL_AA_ALPHABET.index("S")
SERINE_KEYS = {
    f"experts.{SERINE_INDEX}.weight",
    f"experts.{SERINE_INDEX}.bias",
}
EXPERT_KEYS = {
    f"experts.{index}.{suffix}"
    for index in range(len(NATURAL_AA_ALPHABET))
    for suffix in ("weight", "bias")
}
PROBABILITY_TOLERANCE = 1e-7
METRIC_TOLERANCE = 1e-12
SERINE_AUC_SAFETY_FLOOR = 0.70


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


def load_trainer_module() -> Any:
    spec = importlib.util.spec_from_file_location("source_scoped_v8_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load trainer module: {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extract_state_dict(payload: Any) -> MutableMapping[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("model_state_dict"), Mapping):
        return dict(payload["model_state_dict"])
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError("Checkpoint payload is not a state dictionary")


def source_for_state_key(key: str) -> str:
    """Return the frozen V8 source rule for one state key."""

    if key in SERINE_KEYS:
        return "v7_serine"
    if key in EXPERT_KEYS:
        return "v6_non_ser"
    return "canonical_shared"


def compose_state_dict(
    canonical: Mapping[str, Any],
    v6: Mapping[str, Any],
    v7: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compose V8 without averaging or mutating any source tensor."""

    key_sets = {tuple(sorted(value)) for value in (canonical, v6, v7)}
    if len(key_sets) != 1:
        raise RuntimeError("Canonical, V6, and V7 state-key sets differ")
    result: Dict[str, Any] = {}
    for key in sorted(canonical):
        source = source_for_state_key(key)
        value = (
            v7[key]
            if source == "v7_serine"
            else v6[key]
            if source == "v6_non_ser"
            else canonical[key]
        )
        result[key] = value.detach().cpu().clone() if hasattr(value, "detach") else value
    return result


def tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_hashes(state: Mapping[str, Any]) -> Dict[str, str]:
    return {key: tensor_sha256(value) for key, value in sorted(state.items())}


def validate_source_manifests(
    v6_manifest: Mapping[str, Any],
    v7_manifest: Mapping[str, Any],
    v6_model: Path,
    v7_model: Path,
    test_path: Path,
) -> None:
    if not (
        v6_manifest.get("quality_gate") == "PASS"
        and v6_manifest.get("protocol") == V6_EXPERT_PROTOCOL
        and v6_manifest.get("checkpoint_artifact_sha256") == sha256_file(v6_model)
    ):
        raise RuntimeError("V6 source model is absent, failed, stale, or wrong-protocol")
    if not (
        v7_manifest.get("quality_gate") == "PASS"
        and v7_manifest.get("protocol") == V7_EXPERT_PROTOCOL
        and v7_manifest.get("expert_scope") == "serine-only"
        and list(v7_manifest.get("active_expert_tokens", [])) == ["S"]
        and v7_manifest.get("checkpoint_artifact_sha256") == sha256_file(v7_model)
    ):
        raise RuntimeError("V7 source model is absent, failed, stale, or not Ser-only")
    test_sha = sha256_file(test_path)
    source_test_hashes = {
        str(dict(manifest.get("training") or {}).get("test_jsonl_sha256", ""))
        for manifest in (v6_manifest, v7_manifest)
    }
    if source_test_hashes != {test_sha}:
        raise RuntimeError("V6/V7 did not use the exact same corrected held-out test")


def row_identity(row: Mapping[str, Any]) -> Tuple[str, int, str, int]:
    return (
        str(row["sample_name"]),
        int(row["position_in_model_0based"]),
        str(row["base_token"]),
        int(row["is_methyl_true"]),
    )


def validate_finite_position_rows(
    rows: Sequence[Mapping[str, Any]], label: str
) -> None:
    fields = (
        "probability_methyl_deployment_scaled",
        "probability_order_std",
        "probability_representation_std",
        "probability_representation_span",
    )
    for row_number, row in enumerate(rows, start=1):
        try:
            values = {field: float(row[field]) for field in fields}
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{label} frozen-test row {row_number} lacks a numeric probability field"
            ) from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(
                f"{label} frozen-test row {row_number} contains NaN/Inf"
            )
        probability = values["probability_methyl_deployment_scaled"]
        if not 0.0 <= probability <= 1.0 or any(
            values[field] < 0.0 for field in fields[1:]
        ):
            raise RuntimeError(
                f"{label} frozen-test row {row_number} is outside its valid range"
            )


def inheritance_differences(
    v6_rows: Sequence[Mapping[str, Any]],
    v7_rows: Sequence[Mapping[str, Any]],
    v8_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    if not (len(v6_rows) == len(v7_rows) == len(v8_rows)):
        raise RuntimeError("Source/hybrid frozen-test position counts differ")
    maximum_ser = 0.0
    maximum_non_ser = 0.0
    for v6_row, v7_row, v8_row in zip(v6_rows, v7_rows, v8_rows):
        identities = {row_identity(row) for row in (v6_row, v7_row, v8_row)}
        if len(identities) != 1:
            raise RuntimeError("Source/hybrid frozen-test row identities differ")
        base = str(v8_row["base_token"])
        inherited = v7_row if base == "S" else v6_row
        difference = abs(
            float(v8_row["probability_methyl_deployment_scaled"])
            - float(inherited["probability_methyl_deployment_scaled"])
        )
        if not math.isfinite(difference):
            raise RuntimeError("Non-finite source/hybrid inheritance difference")
        if base == "S":
            maximum_ser = max(maximum_ser, difference)
        else:
            maximum_non_ser = max(maximum_non_ser, difference)
    return {
        "maximum_ser_probability_difference_from_v7": maximum_ser,
        "maximum_non_ser_probability_difference_from_v6": maximum_non_ser,
    }


def metric_comparison_rows(
    v6_summary: Mapping[str, Any],
    v7_summary: Mapping[str, Any],
    v8_summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    paths = {
        "overall_auc": lambda row: row["overall_auc"],
        "supported_macro_auc": lambda row: row["supported_macro_auc"],
        "precision_at_0_6": lambda row: row["overall_at_threshold"]["precision"],
        "recall_at_0_6": lambda row: row["overall_at_threshold"]["recall"],
        "f1_at_0_6": lambda row: row["overall_at_threshold"]["f1"],
        "false_positive_rate_at_0_6": lambda row: row["overall_at_threshold"][
            "false_positive_rate"
        ],
        "true_positives_at_0_6": lambda row: row["overall_at_threshold"]["tp"],
        "false_negatives_at_0_6": lambda row: row["overall_at_threshold"]["fn"],
        "false_positives_at_0_6": lambda row: row["overall_at_threshold"]["fp"],
        "true_negatives_at_0_6": lambda row: row["overall_at_threshold"]["tn"],
        "non_ser_auc": lambda row: row["non_ser_auc"],
        "non_ser_recall_at_0_6": lambda row: row["non_ser_at_threshold"]["recall"],
        "non_ser_f1_at_0_6": lambda row: row["non_ser_at_threshold"]["f1"],
        "serine_auc": lambda row: row["serine"]["auc"],
        "serine_recall_at_0_6": lambda row: row["serine"]["recall"],
        "serine_fpr_at_0_6": lambda row: row["serine"]["false_positive_rate"],
    }
    rows: List[Dict[str, Any]] = []
    for metric, accessor in paths.items():
        v6_value = accessor(v6_summary)
        v7_value = accessor(v7_summary)
        v8_value = accessor(v8_summary)
        rows.append(
            {
                "metric": metric,
                "v6": v6_value,
                "v7": v7_value,
                "v8": v8_value,
                "v8_minus_v6": float(v8_value) - float(v6_value),
                "v8_minus_v7": float(v8_value) - float(v7_value),
            }
        )
    return rows


def confusion_counts(row: Mapping[str, Any], label: str) -> Dict[str, int]:
    """Read and internally validate one frozen threshold-confusion payload."""

    try:
        counts = {name: int(row[name]) for name in ("tp", "tn", "fp", "fn")}
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} lacks integer TP/TN/FP/FN counts") from exc
    if any(value < 0 for value in counts.values()):
        raise RuntimeError(f"{label} contains a negative confusion count")
    return counts


def serine_auc_tradeoff_audit(
    v6_summary: Mapping[str, Any],
    v7_summary: Mapping[str, Any],
    v8_summary: Mapping[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """Record the observed Ser ranking trade-off without redefining the data.

    All three models are evaluated in the same process, on the same hash-pinned
    rows and with the same batch/temperature/threshold.  AUC is threshold-free;
    it is therefore disclosed as an observed post-hoc comparison, not converted
    into a new zero-margin promotion gate after seeing the result.  The actual
    deployment decision remains protected by a non-degrading Ser threshold
    confusion (TP/TN cannot decrease; FP/FN cannot increase) plus the
    pre-existing absolute AUC safety floor.
    """

    summaries = {"v6": v6_summary, "v7": v7_summary, "v8": v8_summary}
    models: Dict[str, Any] = {}
    supports = set()
    for label, summary in summaries.items():
        try:
            serine = dict(summary["serine"])
            auc = float(serine["auc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{label.upper()} Ser audit summary is incomplete") from exc
        if not math.isfinite(auc) or not 0.0 <= auc <= 1.0:
            raise RuntimeError(f"{label.upper()} Ser AUC is non-finite or out of range")
        counts = confusion_counts(serine, f"{label.upper()} runtime Ser")
        positives = counts["tp"] + counts["fn"]
        negatives = counts["tn"] + counts["fp"]
        supports.add((positives, negatives))
        models[label] = {
            "auc": auc,
            "threshold_confusion": counts,
            "methyl_positives": positives,
            "natural_negatives": negatives,
        }
    if len(supports) != 1:
        raise RuntimeError("V6/V7/V8 Ser class supports differ")
    positives, negatives = next(iter(supports))
    pair_count = positives * negatives
    if pair_count <= 0:
        raise RuntimeError("Ser AUC audit requires both positive and negative examples")
    for values in models.values():
        values["auc_positive_negative_pair_equivalent"] = (
            float(values["auc"]) * pair_count
        )

    auc_delta = float(models["v8"]["auc"]) - float(models["v6"]["auc"])
    if auc_delta < -METRIC_TOLERANCE:
        direction = "lower"
    elif auc_delta > METRIC_TOLERANCE:
        direction = "higher"
    else:
        direction = "equal_within_tolerance"
    return {
        "basis": (
            "same-run paired replay of hash-pinned V6/V7/V8 checkpoints on the "
            "same 151-record, 1505-position internal audit set"
        ),
        "threshold": float(threshold),
        "strict_threshold_operator": ">",
        "positive_negative_pair_count": pair_count,
        "models": models,
        "v8_minus_v6_auc": auc_delta,
        "v8_minus_v6_auc_positive_negative_pair_equivalent": (
            auc_delta * pair_count
        ),
        "v8_auc_direction_vs_v6": direction,
        "v8_threshold_confusion_matches_v6": (
            models["v8"]["threshold_confusion"]
            == models["v6"]["threshold_confusion"]
        ),
        "v8_threshold_confusion_is_non_degrading_vs_v6": (
            models["v8"]["threshold_confusion"]["tp"]
            >= models["v6"]["threshold_confusion"]["tp"]
            and models["v8"]["threshold_confusion"]["tn"]
            >= models["v6"]["threshold_confusion"]["tn"]
            and models["v8"]["threshold_confusion"]["fp"]
            <= models["v6"]["threshold_confusion"]["fp"]
            and models["v8"]["threshold_confusion"]["fn"]
            <= models["v6"]["threshold_confusion"]["fn"]
        ),
        "v8_threshold_confusion_matches_v7": (
            models["v8"]["threshold_confusion"]
            == models["v7"]["threshold_confusion"]
        ),
        "v8_auc_matches_v7_within_tolerance": (
            abs(float(models["v8"]["auc"]) - float(models["v7"]["auc"]))
            <= METRIC_TOLERANCE
        ),
        "auc_gate_policy": (
            "report observed V8-minus-V6 AUC exactly; do not assert zero-margin "
            "non-inferiority post hoc; retain the absolute Ser-AUC safety floor"
        ),
        "interpretation": (
            "V8 inherits the provenance-corrected V7 Ser ranking. Any observed "
            "V8-minus-V6 Ser-AUC difference is an explicit internal trade-off; "
            "publication claims require a new outer split or blind evaluation."
        ),
    }


def serine_auc_tradeoff_rows(audit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create a reviewable CSV for the Ser AUC and threshold decision audit."""

    models = audit["models"]
    accessors = {
        "serine_auc": lambda row: row["auc"],
        "serine_auc_positive_negative_pair_equivalent": lambda row: row[
            "auc_positive_negative_pair_equivalent"
        ],
        "serine_tp_at_0_6": lambda row: row["threshold_confusion"]["tp"],
        "serine_tn_at_0_6": lambda row: row["threshold_confusion"]["tn"],
        "serine_fp_at_0_6": lambda row: row["threshold_confusion"]["fp"],
        "serine_fn_at_0_6": lambda row: row["threshold_confusion"]["fn"],
    }
    threshold_roles = {
        "serine_tp_at_0_6": "v8_must_be_greater_than_or_equal_to_v6",
        "serine_tn_at_0_6": "v8_must_be_greater_than_or_equal_to_v6",
        "serine_fp_at_0_6": "v8_must_be_less_than_or_equal_to_v6",
        "serine_fn_at_0_6": "v8_must_be_less_than_or_equal_to_v6",
    }
    rows = []
    for metric, accessor in accessors.items():
        v6_value = accessor(models["v6"])
        v7_value = accessor(models["v7"])
        v8_value = accessor(models["v8"])
        is_auc = metric.startswith("serine_auc")
        rows.append(
            {
                "metric": metric,
                "v6": v6_value,
                "v7": v7_value,
                "v8": v8_value,
                "v8_minus_v6": float(v8_value) - float(v6_value),
                "promotion_role": (
                    "report_only_with_absolute_safety_floor"
                    if is_auc
                    else threshold_roles[metric]
                ),
                "audit_basis": audit["basis"],
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("V8 composition requires PyTorch") from exc
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    canonical_path = Path(args.canonical_model).resolve()
    v6_model_path = Path(args.v6_model).resolve()
    v6_manifest_path = Path(args.v6_manifest).resolve()
    v7_model_path = Path(args.v7_model).resolve()
    v7_manifest_path = Path(args.v7_manifest).resolve()
    test_path = Path(args.test_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    immutable_inputs = (
        canonical_path,
        v6_model_path,
        v6_manifest_path,
        v7_model_path,
        v7_manifest_path,
        test_path,
        SCRIPT_PATH,
        TRAINER_PATH,
        COMMON_PATH,
        MODEL_UTILS_PATH,
        NMETHYL_CONFIG_PATH,
    )
    if any(paths_overlap(out_dir, path) for path in immutable_inputs):
        raise ValueError("V8 model output overlaps an immutable input")
    for required in (
        canonical_path,
        v6_model_path,
        v6_manifest_path,
        v7_model_path,
        v7_manifest_path,
        test_path,
        TRAINER_PATH,
        COMMON_PATH,
        MODEL_UTILS_PATH,
        NMETHYL_CONFIG_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest_path = out_dir / "expert_source_composition_manifest.json"
    production_path = out_dir / "frankenstein_v28_source_scoped_hybrid_v8.pt"
    candidate_path = out_dir / "frankenstein_v28_source_scoped_hybrid_v8.candidate.pt"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"V8 model output already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    v6_manifest = read_json(v6_manifest_path)
    v7_manifest = read_json(v7_manifest_path)
    validate_source_manifests(
        v6_manifest, v7_manifest, v6_model_path, v7_model_path, test_path
    )

    # These are hash-verified local workflow artifacts and contain provenance
    # metadata in addition to tensors.  State the trusted full-payload policy
    # explicitly so PyTorch does not emit its future-default warning.
    canonical_payload = torch.load(
        canonical_path, map_location="cpu", weights_only=False
    )
    v6_payload = torch.load(v6_model_path, map_location="cpu", weights_only=False)
    v7_payload = torch.load(v7_model_path, map_location="cpu", weights_only=False)
    canonical_state = extract_state_dict(canonical_payload)
    v6_state = extract_state_dict(v6_payload)
    v7_state = extract_state_dict(v7_payload)
    for label, state in (
        ("canonical", canonical_state),
        ("V6", v6_state),
        ("V7", v7_state),
    ):
        nonfinite = [
            key
            for key, value in state.items()
            if hasattr(value, "is_floating_point")
            and value.is_floating_point()
            and not bool(torch.isfinite(value).all().item())
        ]
        if nonfinite:
            raise RuntimeError(
                f"{label} checkpoint contains non-finite tensors: "
                + ", ".join(nonfinite[:5])
            )
    canonical_hashes = state_hashes(canonical_state)
    v6_hashes = state_hashes(v6_state)
    v7_hashes = state_hashes(v7_state)

    shared_keys = sorted(set(canonical_state) - EXPERT_KEYS)
    if any(
        canonical_hashes[key] != v6_hashes[key]
        or canonical_hashes[key] != v7_hashes[key]
        for key in shared_keys
    ):
        raise RuntimeError("V6/V7 shared model tensors are not canonical-identical")

    hybrid_state = compose_state_dict(canonical_state, v6_state, v7_state)
    hybrid_hashes = state_hashes(hybrid_state)
    source_hash_checks = {
        key: hybrid_hashes[key]
        == (
            v7_hashes[key]
            if source_for_state_key(key) == "v7_serine"
            else v6_hashes[key]
            if source_for_state_key(key) == "v6_non_ser"
            else canonical_hashes[key]
        )
        for key in hybrid_state
    }
    if not all(source_hash_checks.values()):
        raise RuntimeError("V8 state composition violated its source map")

    v6_metadata = dict(v6_payload.get("expert_head_qc_metadata", {}))
    v7_metadata = dict(v7_payload.get("expert_head_qc_metadata", {}))
    required_metadata = (
        "training_cyclic_representation_policy",
        "training_decoding_order_policy",
        "deployment_annotation_policy",
        "expert_training_context_policy",
        "required_deployment_annotation_context_policy",
    )
    if any(
        not v6_metadata.get(name)
        or v6_metadata.get(name) != v7_metadata.get(name)
        for name in required_metadata
    ):
        raise RuntimeError("V6/V7 cyclic training or deployment policies differ")
    # Do not copy a source checkpoint's single-training-run metadata wholesale:
    # V7's preserved/changed-key and parent/training fields would become false as
    # soon as the 19 V6 expert heads are inserted.  V8 records only policies
    # proven equal across both sources plus an explicit tensor provenance map.
    hybrid_metadata = {
        "protocol": V8_EXPERT_PROTOCOL,
        "expert_scope": "residue-source-scoped-hybrid",
        "composition_rule": (
            "canonical shared tensors; V6 non-Ser experts; V7 Ser expert; "
            "no averaging and no new optimization"
        ),
        "optimization_performed": False,
        "source_protocols": {
            "canonical_shared": "canonical_clean_v28",
            "v6_non_ser": V6_EXPERT_PROTOCOL,
            "v7_serine": V7_EXPERT_PROTOCOL,
        },
        "source_checkpoint_sha256": {
            "canonical": sha256_file(canonical_path),
            "v6": sha256_file(v6_model_path),
            "v7": sha256_file(v7_model_path),
        },
        "source_manifest_sha256": {
            "v6": sha256_file(v6_manifest_path),
            "v7": sha256_file(v7_manifest_path),
        },
        "active_expert_tokens": list(NATURAL_AA_ALPHABET),
        "serine_expert_index": SERINE_INDEX,
        "expert_source_by_residue": {
            token: ("v7_serine" if token == "S" else "v6_non_ser")
            for token in NATURAL_AA_ALPHABET
        },
        "state_source_by_key": {
            key: source_for_state_key(key) for key in sorted(hybrid_state)
        },
        "composed_state_key_sha256": hybrid_hashes,
        "minimum_order_coverage_epochs": min(
            int(v6_metadata.get("minimum_order_coverage_epochs", 0)),
            int(v7_metadata.get("minimum_order_coverage_epochs", 0)),
        ),
        "training_decoding_order_policy": v7_metadata[
            "training_decoding_order_policy"
        ],
        "training_cyclic_representation_policy": v7_metadata[
            "training_cyclic_representation_policy"
        ],
        "deployment_annotation_policy": v7_metadata[
            "deployment_annotation_policy"
        ],
        "expert_training_context_policy": v7_metadata.get(
            "expert_training_context_policy"
        ),
        "required_deployment_annotation_context_policy": v7_metadata.get(
            "required_deployment_annotation_context_policy"
        ),
        "threshold": float(args.threshold),
        "deployment_temperature": float(args.temperature),
        "cyclic_representation_augmentation": True,
    }
    checkpoint_payload = {
        "model_state_dict": hybrid_state,
        "expert_head_qc_metadata": hybrid_metadata,
    }
    temporary = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    torch.save(checkpoint_payload, temporary)
    os.replace(temporary, candidate_path)

    trainer = load_trainer_module()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU V8 audit requires --allow-cpu")
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

    test_records = trainer.read_jsonl(str(test_path))
    models = {
        "v6": trainer.load_v28_model(str(v6_model_path), device),
        "v7": trainer.load_v28_model(str(v7_model_path), device),
        "v8": trainer.load_v28_model(str(candidate_path), device),
    }
    evaluated: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    for label, model in models.items():
        model.eval()
        evaluated[label] = trainer.evaluate(
            model,
            test_records,
            device,
            int(args.batch_size),
            float(args.threshold),
            float(args.temperature),
            f"source_scoped_comparison_{label}",
            cyclic_representation_ensemble=True,
        )
    v6_summary, _v6_by_residue, v6_positions = evaluated["v6"]
    v7_summary, _v7_by_residue, v7_positions = evaluated["v7"]
    v8_summary, v8_by_residue, v8_positions = evaluated["v8"]
    validate_finite_position_rows(v6_positions, "V6")
    validate_finite_position_rows(v7_positions, "V7")
    validate_finite_position_rows(v8_positions, "V8")
    inheritance = inheritance_differences(v6_positions, v7_positions, v8_positions)
    comparison = metric_comparison_rows(v6_summary, v7_summary, v8_summary)

    v6_fixed = v6_summary["overall_at_threshold"]
    v8_fixed = v8_summary["overall_at_threshold"]
    v6_ser = v6_summary["serine"]
    v8_ser = v8_summary["serine"]
    v6_non_ser = v6_summary["non_ser_at_threshold"]
    v8_non_ser = v8_summary["non_ser_at_threshold"]
    serine_tradeoff = serine_auc_tradeoff_audit(
        v6_summary, v7_summary, v8_summary, float(args.threshold)
    )
    quality_checks = {
        "all_shared_tensors_are_canonical_bitwise_identical": all(
            hybrid_hashes[key] == canonical_hashes[key] for key in shared_keys
        ),
        "all_non_ser_experts_are_v6_bitwise_identical": all(
            hybrid_hashes[key] == v6_hashes[key]
            for key in sorted(EXPERT_KEYS - SERINE_KEYS)
        ),
        "serine_expert_is_v7_bitwise_identical": all(
            hybrid_hashes[key] == v7_hashes[key] for key in sorted(SERINE_KEYS)
        ),
        "frozen_audit_test_is_the_same_151_records_and_1505_positions": (
            len(test_records) == 151 and len(v8_positions) == 1505
        ),
        "every_non_ser_probability_is_inherited_from_v6": (
            inheritance["maximum_non_ser_probability_difference_from_v6"]
            <= PROBABILITY_TOLERANCE
        ),
        "every_ser_probability_is_inherited_from_v7": (
            inheritance["maximum_ser_probability_difference_from_v7"]
            <= PROBABILITY_TOLERANCE
        ),
        "recall_at_0_6_is_non_inferior_to_v6": (
            float(v8_fixed["recall"]) + METRIC_TOLERANCE
            >= float(v6_fixed["recall"])
        ),
        "f1_at_0_6_is_non_inferior_to_v6": (
            float(v8_fixed["f1"]) + METRIC_TOLERANCE >= float(v6_fixed["f1"])
        ),
        "non_ser_recall_at_0_6_is_non_inferior_to_v6": (
            float(v8_non_ser["recall"]) + METRIC_TOLERANCE
            >= float(v6_non_ser["recall"])
        ),
        "non_ser_f1_at_0_6_is_non_inferior_to_v6": (
            float(v8_non_ser["f1"]) + METRIC_TOLERANCE
            >= float(v6_non_ser["f1"])
        ),
        "serine_threshold_operating_point_is_non_degrading_vs_v6": (
            bool(
                serine_tradeoff[
                    "v8_threshold_confusion_is_non_degrading_vs_v6"
                ]
            )
        ),
        "serine_threshold_confusion_is_inherited_from_v7": (
            bool(serine_tradeoff["v8_threshold_confusion_matches_v7"])
        ),
        "serine_auc_is_inherited_from_v7": (
            bool(serine_tradeoff["v8_auc_matches_v7_within_tolerance"])
        ),
        "serine_auc_tradeoff_is_explicitly_recorded": (
            math.isfinite(float(serine_tradeoff["v8_minus_v6_auc"]))
            and math.isfinite(
                float(
                    serine_tradeoff[
                        "v8_minus_v6_auc_positive_negative_pair_equivalent"
                    ]
                )
            )
            and int(serine_tradeoff["positive_negative_pair_count"]) > 0
        ),
        "serine_auc_ge_0_70": (
            v8_ser["auc"] is not None
            and float(v8_ser["auc"]) >= SERINE_AUC_SAFETY_FLOOR
        ),
        "overall_auc_ge_0_85": float(v8_summary["overall_auc"]) >= 0.85,
        "overall_precision_at_0_6_ge_0_75": float(v8_fixed["precision"]) >= 0.75,
        "overall_fpr_at_0_6_le_0_10": float(v8_fixed["false_positive_rate"]) <= 0.10,
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    checkpoint_artifact = candidate_path
    if quality_gate == "PASS":
        os.replace(candidate_path, production_path)
        checkpoint_artifact = production_path

    comparison_path = out_dir / "v6_v7_v8_metric_comparison.csv"
    serine_tradeoff_path = out_dir / "serine_auc_tradeoff_audit.csv"
    residue_metrics_path = out_dir / "test_metrics_by_residue.csv"
    position_path = out_dir / "test_position_probabilities.csv"
    atomic_write_csv(
        comparison_path,
        comparison,
        list(comparison[0]),
    )
    serine_tradeoff_rows = serine_auc_tradeoff_rows(serine_tradeoff)
    atomic_write_csv(
        serine_tradeoff_path,
        serine_tradeoff_rows,
        list(serine_tradeoff_rows[0]),
    )
    atomic_write_csv(
        residue_metrics_path,
        v8_by_residue,
        list(v8_by_residue[0]),
    )
    atomic_write_csv(
        position_path,
        v8_positions,
        list(v8_positions[0]),
    )
    manifest = {
        "quality_gate": quality_gate,
        "protocol": V8_EXPERT_PROTOCOL,
        "release_status": (
            "READY_FOR_HYBRID_REPRESENTATION_AUDIT"
            if quality_gate == "PASS"
            else "BLOCKED_SOURCE_COMPOSITION_OR_OPERATIONAL_GATE_FAILURE"
        ),
        "scientific_reason": (
            "Ser provenance repair and cyclic representation retraining are distinct "
            "interventions. Independent linear heads use the newest justified source "
            "for their own residue without changing the canonical shared network."
        ),
        "development_status": "POST_HOC_INTERNAL_RECOVERY_CANDIDATE_NOT_BLIND_PUBLICATION_MODEL",
        "selection_policy": (
            "The residue-source rule was chosen after diagnosing the V6/V7 paired-audit "
            "recall loss, then frozen before composing/evaluating V8. It uses no averaging, "
            "threshold change, search-result feedback, or fitted coefficient, but remains "
            "post hoc and requires a new outer/blind evaluation for publication claims."
        ),
        "test_reuse_limitation": (
            "The 151-record set has already been inspected during V3/V6/V7 development. "
            "It is a frozen paired audit set for this internal recovery, not a new blind "
            "publication test. Final paper claims require a new outer split or blind set."
        ),
        "threshold": float(args.threshold),
        "strict_threshold_operator": ">",
        "deployment_temperature": float(args.temperature),
        "device": str(device),
        "audit_batch_size": int(args.batch_size),
        "composer_program_sha256": sha256_file(SCRIPT_PATH),
        "trainer_program_sha256": sha256_file(TRAINER_PATH),
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
        "canonical_checkpoint": str(canonical_path),
        "canonical_checkpoint_sha256": sha256_file(canonical_path),
        "v6_checkpoint": str(v6_model_path),
        "v6_checkpoint_sha256": sha256_file(v6_model_path),
        "v6_manifest": str(v6_manifest_path),
        "v6_manifest_sha256": sha256_file(v6_manifest_path),
        "v7_checkpoint": str(v7_model_path),
        "v7_checkpoint_sha256": sha256_file(v7_model_path),
        "v7_manifest": str(v7_manifest_path),
        "v7_manifest_sha256": sha256_file(v7_manifest_path),
        "checkpoint_ready_for_representation_audit": quality_gate == "PASS",
        "output_checkpoint": str(production_path) if quality_gate == "PASS" else None,
        "candidate_checkpoint": str(checkpoint_artifact),
        "checkpoint_artifact_sha256": sha256_file(checkpoint_artifact),
        "expert_scope": "residue-source-scoped-hybrid",
        "active_expert_tokens": list(NATURAL_AA_ALPHABET),
        "test_jsonl": str(test_path),
        "test_jsonl_sha256": sha256_file(test_path),
        "expert_source_counts": {
            "canonical_shared_tensors": len(shared_keys),
            "v6_non_ser_expert_tensors": len(EXPERT_KEYS - SERINE_KEYS),
            "v7_ser_expert_tensors": len(SERINE_KEYS),
        },
        "expert_source_by_residue": {
            token: ("v7_serine" if token == "S" else "v6_non_ser")
            for token in NATURAL_AA_ALPHABET
        },
        "probability_inheritance_audit": inheritance,
        "metric_gate_provenance": {
            "paired_runtime_replay_policy": (
                "One same-process V6/V7/V8 replay proves position-level source "
                "inheritance, a non-degrading Ser threshold operating point, "
                "overall Recall/F1 non-inferiority, and absolute safety floors. "
                "Ser AUC is reported as an observed post-hoc trade-off, not a "
                "zero-margin gate."
            ),
            "serine_auc_tradeoff": serine_tradeoff,
        },
        "v6_test": v6_summary,
        "v7_test": v7_summary,
        "v8_test": v8_summary,
        "quality_checks": quality_checks,
        "artifacts": {
            "metric_comparison": {
                "path": str(comparison_path),
                "sha256": sha256_file(comparison_path),
            },
            "serine_auc_tradeoff_audit": {
                "path": str(serine_tradeoff_path),
                "sha256": sha256_file(serine_tradeoff_path),
            },
            "metrics_by_residue": {
                "path": str(residue_metrics_path),
                "sha256": sha256_file(residue_metrics_path),
            },
            "position_probabilities": {
                "path": str(position_path),
                "sha256": sha256_file(position_path),
            },
        },
    }
    atomic_write_json(manifest_path, manifest)

    print("===== SOURCE-SCOPED CYCLIC EXPERT V8 COMPOSITION COMPLETE =====", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    print(
        "V6 -> V7 -> V8 recall@0.6: "
        f"{float(v6_fixed['recall']):.4f} -> "
        f"{float(v7_summary['overall_at_threshold']['recall']):.4f} -> "
        f"{float(v8_fixed['recall']):.4f}",
        flush=True,
    )
    print(
        "V6 -> V7 -> V8 F1@0.6: "
        f"{float(v6_fixed['f1']):.4f} -> "
        f"{float(v7_summary['overall_at_threshold']['f1']):.4f} -> "
        f"{float(v8_fixed['f1']):.4f}",
        flush=True,
    )
    print(
        "Ser AUC V6 -> V7 -> V8 (reported trade-off, not zero-margin gate): "
        f"{float(v6_ser['auc']):.6f}/"
        f"{float(v7_summary['serine']['auc']):.6f}/"
        f"{float(v8_ser['auc']):.6f}; V8-V6="
        f"{float(serine_tradeoff['v8_minus_v6_auc']):+.6f}; "
        "pair-equivalent="
        f"{float(serine_tradeoff['v8_minus_v6_auc_positive_negative_pair_equivalent']):+.6f}",
        flush=True,
    )
    print(
        "Ser threshold operating point non-degrading vs V6: "
        f"{bool(serine_tradeoff['v8_threshold_confusion_is_non_degrading_vs_v6'])}; "
        "exactly inherited from V7: "
        f"{bool(serine_tradeoff['v8_threshold_confusion_matches_v7'])}",
        flush=True,
    )
    print(f"Checkpoint: {checkpoint_artifact}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError("V8 source composition failed: " + ", ".join(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-model", default=str(DEFAULT_CANONICAL))
    parser.add_argument("--v6-model", default=str(DEFAULT_V6_MODEL))
    parser.add_argument("--v6-manifest", default=str(DEFAULT_V6_MANIFEST))
    parser.add_argument("--v7-model", default=str(DEFAULT_V7_MODEL))
    parser.add_argument("--v7-manifest", default=str(DEFAULT_V7_MANIFEST))
    parser.add_argument("--test-jsonl", default=str(DEFAULT_TEST))
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
        raise ValueError("V8 paired audit is frozen to --batch-size 8")
    if float(args.temperature) != 0.5 or float(args.threshold) != 0.6:
        raise ValueError("V8 is frozen to T=0.5 and strict threshold >0.6")
    run(args)


if __name__ == "__main__":
    main()
