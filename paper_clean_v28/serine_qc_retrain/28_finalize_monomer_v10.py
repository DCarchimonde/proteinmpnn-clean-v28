#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Finalize the corrected 151-monomer V10 sequence and methylation audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from rmsd_ranker_v10 import binary_auc


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_ORIGINAL = REPO_ROOT / "v10_inputs" / "monomer_corrected_1505_original_v28.csv"
EXPECTED_ORIGINAL_SHA256 = "c9c709521b83523c82dd83eb376da0d7f88be3147521d5c80526b6306f92fc62"
EXPECTED_PARENT_SHA256 = "bab7b8a010114fc52c749fab1914d9d8ae561ddca45d6d7a0fbec3f9f5ac5b2e"
EXPECTED_POSITIONS = 1505
EXPECTED_SAMPLES = 151
EXPECTED_POSITIVES = 261
EXPECTED_NATIVE_COMPLEX_POSITIONS = 157
THRESHOLD = 0.6
SENSITIVITY_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
PROTOCOL = "corrected_monomer_cyclic_stability_and_base_freeze_audit_v10"
KNOWN_OUTPUTS = (
    "monomer_v10_metrics.csv",
    "monomer_v10_threshold_curves.csv",
    "monomer_v10_by_residue.csv",
    "monomer_v10_by_company_rosetta_panel.csv",
    "monomer_v10_per_sample.csv",
    "monomer_v10_paired_original_v28_comparison.csv",
    "native17_v10_all_negative_control.csv",
    "monomer_v10_position_comparison_1505.csv",
    "monomer_v10_design_manifest_151.csv",
    "monomer_v10_structure_input_if_reprediction_needed.fasta",
    "monomer_v10_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def nested_hash(payload: Mapping[str, Any], section: str, label: str) -> str:
    section_value = payload.get(section, {})
    if not isinstance(section_value, Mapping):
        return ""
    record = section_value.get(label, {})
    if not isinstance(record, Mapping):
        return ""
    return str(record.get("sha256", ""))


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                result.append(field)
    return result


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=union_fields(rows), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def strict_pass(value: Any, threshold: float = THRESHOLD) -> bool:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Invalid methylation probability")
    return round(probability, 8) > float(threshold)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    score = np.asarray(scores, dtype=float)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return math.nan
    order = np.argsort(-score, kind="stable")
    sorted_y = y[order]
    sorted_score = score[order]
    tp = fp = 0
    previous_recall = 0.0
    result = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_score[end] == sorted_score[start]:
            end += 1
        tp += int(np.sum(sorted_y[start:end] == 1))
        fp += int(np.sum(sorted_y[start:end] == 0))
        recall = tp / positives
        precision = tp / (tp + fp)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(result)


def classification_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = THRESHOLD,
) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    pred = np.asarray(
        [strict_pass(value, threshold=threshold) for value in values], dtype=int
    )
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "positions": len(y),
        "positives": int(np.sum(y)),
        "negatives": int(np.sum(y == 0)),
        "roc_auc": binary_auc(y, values),
        "average_precision": average_precision(y, values),
        "threshold": float(threshold),
        "threshold_operator": ">",
        "probability_rounding_policy": "round(prob,8)",
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "predicted_methyl_rate": float(np.mean(pred)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def exact_mcnemar_p_value(old_only_correct: int, new_only_correct: int) -> float:
    """Two-sided exact McNemar/binomial p-value without SciPy."""

    discordant = int(old_only_correct) + int(new_only_correct)
    if discordant == 0:
        return 1.0
    tail = min(int(old_only_correct), int(new_only_correct))
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * probability)


def source_panel(row: Mapping[str, Any]) -> str:
    value = str(row.get("source_panel", "")).strip()
    if value:
        return value
    sample = str(row.get("sample_name", ""))
    if sample.startswith("Me_"):
        return "Rosetta-2023"
    if sample.startswith("pdb_"):
        return "Rosetta-2025"
    return "unknown_company_rosetta_panel"


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    old_score_field: str,
    new_score_field: str,
    *,
    replicates: int = 1000,
    seed: int = 20260822,
) -> Dict[str, Tuple[float, float]]:
    """Paired sample-cluster bootstrap CIs for key classification deltas."""

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_name"])].append(row)
    sample_names = sorted(grouped)
    rng = np.random.default_rng(seed)
    metric_names = (
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "predicted_methyl_rate",
    )
    deltas: Dict[str, List[float]] = {name: [] for name in metric_names}
    for _replicate in range(replicates):
        sampled = rng.choice(sample_names, size=len(sample_names), replace=True)
        bootstrap_rows = [row for name in sampled for row in grouped[str(name)]]
        labels = [int(row["is_methyl_true_corrected"]) for row in bootstrap_rows]
        old_scores = [float(row[old_score_field]) for row in bootstrap_rows]
        new_scores = [float(row[new_score_field]) for row in bootstrap_rows]
        old_metrics = classification_metrics(labels, old_scores)
        new_metrics = classification_metrics(labels, new_scores)
        for metric in metric_names:
            delta = float(new_metrics[metric]) - float(old_metrics[metric])
            if math.isfinite(delta):
                deltas[metric].append(delta)
    return {
        metric: (
            float(np.quantile(values, 0.025)) if values else math.nan,
            float(np.quantile(values, 0.975)) if values else math.nan,
        )
        for metric, values in deltas.items()
    }


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    existing = [out_dir / name for name in KNOWN_OUTPUTS if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "V10 monomer output exists; use a new directory or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10-position-csv", required=True)
    parser.add_argument("--v10-eval-manifest", required=True)
    parser.add_argument("--parent-position-csv", required=True)
    parser.add_argument("--parent-eval-manifest", required=True)
    parser.add_argument("--cyclic-position-csv", required=True)
    parser.add_argument("--native-position-csv", required=True)
    parser.add_argument("--cyclic-audit-manifest", required=True)
    parser.add_argument("--original-v28-corrected-csv", default=str(DEFAULT_ORIGINAL))
    parser.add_argument("--v10-model", required=True)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    v10_position_path = Path(args.v10_position_csv).resolve()
    v10_eval_manifest_path = Path(args.v10_eval_manifest).resolve()
    parent_position_path = Path(args.parent_position_csv).resolve()
    parent_eval_manifest_path = Path(args.parent_eval_manifest).resolve()
    cyclic_position_path = Path(args.cyclic_position_csv).resolve()
    native_position_path = Path(args.native_position_csv).resolve()
    cyclic_audit_path = Path(args.cyclic_audit_manifest).resolve()
    original_path = Path(args.original_v28_corrected_csv).resolve()
    v10_model_path = Path(args.v10_model).resolve()
    parent_model_path = Path(args.parent_model).resolve()
    out_dir = Path(args.out_dir).resolve()
    prepare_output(out_dir, args.overwrite)

    original_rows = read_csv(original_path)
    v10_all = read_csv(v10_position_path)
    v10_rows = [
        row for row in v10_all if row.get("input_mode") == "strict_naturalized_input"
    ]
    parent_all = read_csv(parent_position_path)
    parent_rows = [
        row
        for row in parent_all
        if row.get("input_mode") == "strict_naturalized_input"
    ]
    cyclic_rows = read_csv(cyclic_position_path)
    native_raw_rows = read_csv(native_position_path)
    audit = read_json(cyclic_audit_path)
    eval_manifest = read_json(v10_eval_manifest_path)
    parent_eval_manifest = read_json(parent_eval_manifest_path)

    def original_key(row: Mapping[str, Any]) -> Tuple[str, str, int]:
        return (
            str(row.get("sample_name", "")),
            str(row.get("selected_chains", "")),
            int(row.get("position_in_model", -1)),
        )

    def cyclic_key(row: Mapping[str, Any]) -> Tuple[str, int]:
        return (
            str(row.get("sample_name", "")),
            int(row.get("position_in_peptide_1based", -1)) - 1,
        )

    original_index = {original_key(row): row for row in original_rows}
    cyclic_index = {cyclic_key(row): row for row in cyclic_rows}
    v10_index = {original_key(row): row for row in v10_rows}
    parent_index = {original_key(row): row for row in parent_rows}
    expected_cyclic_keys = {(key[0], key[2]) for key in v10_index}
    native_groups: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in native_raw_rows:
        native_groups[
            (
                str(row.get("target_name", "")).upper(),
                int(row.get("physical_position_1based", -1)),
            )
        ].append(row)
    native_control_rows: List[Dict[str, Any]] = []
    native_group_contract_valid = True
    for (target, position), rows in sorted(native_groups.items()):
        try:
            sequence = str(rows[0]["native_sequence"]).upper()
            known_values = [
                float(
                    row.get(
                        "mapped_probability_known_sequence",
                        row.get("mapped_probability", "nan"),
                    )
                )
                for row in rows
            ]
            e2e_values = [
                float(row["mapped_probability_end_to_end"]) for row in rows
            ]
            representation_ids = {
                int(row["representation_left_shift"]) for row in rows
            }
            valid = (
                bool(target)
                and 1 <= position <= len(sequence)
                and len(rows) == len(sequence)
                and representation_ids == set(range(len(sequence)))
                and all(
                    str(row["native_sequence"]).upper() == sequence
                    and str(row["base_token"]).upper()
                    == sequence[position - 1]
                    for row in rows
                )
                and all(
                    math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in known_values + e2e_values
                )
            )
        except (KeyError, TypeError, ValueError):
            valid = False
            sequence = ""
            known_values = []
            e2e_values = []
        native_group_contract_valid = native_group_contract_valid and valid
        if not valid:
            continue
        known_minimum, known_maximum = min(known_values), max(known_values)
        e2e_minimum, e2e_maximum = min(e2e_values), max(e2e_values)
        native_control_rows.append(
            {
                "target_name": target,
                "native_sequence": sequence,
                "physical_position_1based": position,
                "base_token": sequence[position - 1],
                "is_methyl_true": 0,
                "representation_count": len(rows),
                "known_representation_mean": float(np.mean(known_values)),
                "known_representation_min": known_minimum,
                "known_representation_max": known_maximum,
                "known_representation_span": known_maximum - known_minimum,
                "known_threshold_disagreement": int(
                    strict_pass(known_minimum) != strict_pass(known_maximum)
                ),
                "known_stable_methyl_call": int(strict_pass(known_minimum)),
                "e2e_representation_mean": float(np.mean(e2e_values)),
                "e2e_representation_min": e2e_minimum,
                "e2e_representation_max": e2e_maximum,
                "e2e_representation_span": e2e_maximum - e2e_minimum,
                "e2e_threshold_disagreement": int(
                    strict_pass(e2e_minimum) != strict_pass(e2e_maximum)
                ),
                "e2e_stable_methyl_call": int(strict_pass(e2e_minimum)),
            }
        )
    audit_checks = audit.get("quality_checks", {})
    checks: Dict[str, bool] = {
        "original_corrected_csv_sha256_is_frozen": (
            sha256_file(original_path) == EXPECTED_ORIGINAL_SHA256
        ),
        "canonical_parent_model_sha256_is_frozen": (
            sha256_file(parent_model_path) == EXPECTED_PARENT_SHA256
        ),
        "cyclic_audit_is_authorized_pass_for_exact_v10_model": (
            audit.get("quality_gate") == "PASS"
            and audit.get("model_sha256") == sha256_file(v10_model_path)
            and isinstance(audit_checks, Mapping)
            and bool(audit_checks)
            and all(value is True for value in audit_checks.values())
        ),
        "single_representation_eval_is_bound_to_exact_v10_model_data_and_position_csv": (
            eval_manifest.get("quality_gate") == "PASS"
            and eval_manifest.get("protocol")
            == "clean_v28_three_input_mode_evaluation_provenance_v2_deterministic"
            and nested_hash(eval_manifest, "inputs", "model")
            == sha256_file(v10_model_path)
            and nested_hash(eval_manifest, "inputs", "data_jsonl")
            == nested_hash(audit, "inputs", "test_jsonl")
            and nested_hash(eval_manifest, "artifacts", "position_predictions")
            == sha256_file(v10_position_path)
            and int(eval_manifest.get("seed", -1)) == 0
            and bool(eval_manifest.get("deterministic_cudnn"))
        ),
        "deterministic_parent_eval_is_bound_to_exact_parent_model_data_and_position_csv": (
            parent_eval_manifest.get("quality_gate") == "PASS"
            and parent_eval_manifest.get("protocol")
            == "clean_v28_three_input_mode_evaluation_provenance_v2_deterministic"
            and nested_hash(parent_eval_manifest, "inputs", "model")
            == sha256_file(parent_model_path)
            and nested_hash(parent_eval_manifest, "inputs", "data_jsonl")
            == nested_hash(audit, "inputs", "test_jsonl")
            and nested_hash(
                parent_eval_manifest, "artifacts", "position_predictions"
            )
            == sha256_file(parent_position_path)
            and int(parent_eval_manifest.get("seed", -1))
            == int(eval_manifest.get("seed", -2))
            and int(parent_eval_manifest.get("batch_size", -1))
            == int(eval_manifest.get("batch_size", -2))
            and str(parent_eval_manifest.get("program", {}).get("sha256", ""))
            == str(eval_manifest.get("program", {}).get("sha256", ""))
            and bool(parent_eval_manifest.get("deterministic_cudnn"))
        ),
        "native_position_table_matches_named_cyclic_audit_artifact": (
            nested_hash(
                audit,
                "artifacts",
                "native_target_representation_probabilities",
            )
            == sha256_file(native_position_path)
        ),
        "heldout_cyclic_position_table_matches_named_audit_artifact": (
            nested_hash(audit, "artifacts", "heldout_position_probabilities")
            == sha256_file(cyclic_position_path)
        ),
        "all_four_position_tables_have_exactly_1505_unique_rows": (
            len(original_rows) == len(original_index) == EXPECTED_POSITIONS
            and len(v10_rows) == len(v10_index) == EXPECTED_POSITIONS
            and len(parent_rows) == len(parent_index) == EXPECTED_POSITIONS
            and len(cyclic_rows) == len(cyclic_index) == EXPECTED_POSITIONS
        ),
        "v10_and_original_position_keys_match_exactly": set(v10_index) == set(original_index),
        "v10_and_deterministic_parent_position_keys_match_exactly": (
            set(v10_index) == set(parent_index)
        ),
        "cyclic_and_v10_sample_position_keys_match_exactly": (
            set(cyclic_index) == expected_cyclic_keys
        ),
        "native17_control_has_exactly_157_selected_peptide_positions_and_17_targets": (
            native_group_contract_valid
            and len(native_control_rows) == EXPECTED_NATIVE_COMPLEX_POSITIONS
            and len({row["target_name"] for row in native_control_rows}) == 17
        ),
        "native17_known_and_e2e_hard_calls_have_zero_start_disagreement": (
            bool(native_control_rows)
            and all(
                int(row["known_threshold_disagreement"]) == 0
                and int(row["e2e_threshold_disagreement"]) == 0
                for row in native_control_rows
            )
        ),
    }

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "V10 monomer input/provenance gate failed before metric computation: "
            + ", ".join(failed)
        )

    comparison_rows: List[Dict[str, Any]] = []
    for key in sorted(v10_index):
        v10 = v10_index[key]
        parent = parent_index[key]
        original = original_index[key]
        cyclic = cyclic_index[(key[0], key[2])]
        corrected_true = int(original["is_methyl_true_corrected"])
        original_known_call = int(
            strict_pass(original["prob_methyl_known_sequence"])
        )
        original_e2e_call = int(
            strict_pass(original["prob_methyl_end_to_end"])
        )
        v10_known_call = int(
            strict_pass(cyclic["probability_representation_min"])
        )
        v10_e2e_call = int(
            strict_pass(cyclic["probability_end_to_end_representation_min"])
        )
        comparison_rows.append(
            {
                "sample_name": key[0],
                "source_panel": source_panel(original),
                "selected_chains": key[1],
                "position_in_model": key[2],
                "true_base_token": v10["true_base_token"],
                "target_token_corrected": original["target_token_corrected"],
                "is_methyl_true_corrected": corrected_true,
                "original_v28_pred_base_token": original["pred_base_token"],
                "deterministic_parent_pred_base_token": parent["pred_base_token"],
                "v10_pred_base_token": v10["pred_base_token"],
                "base_prediction_unchanged": int(
                    parent["pred_base_token"] == v10["pred_base_token"]
                ),
                "original_v28_known_probability_single_representation": original[
                    "prob_methyl_known_sequence"
                ],
                "v10_known_probability_single_representation": v10[
                    "prob_methyl_known_sequence"
                ],
                "original_v28_e2e_probability_single_representation": original[
                    "prob_methyl_end_to_end"
                ],
                "v10_e2e_probability_single_representation": v10[
                    "prob_methyl_end_to_end"
                ],
                "v10_known_representation_min": cyclic[
                    "probability_representation_min"
                ],
                "v10_known_representation_max": cyclic[
                    "probability_representation_max"
                ],
                "v10_known_representation_span": cyclic[
                    "probability_representation_span"
                ],
                "v10_known_threshold_disagreement": cyclic[
                    "representation_threshold_disagreement"
                ],
                "v10_e2e_representation_min": cyclic[
                    "probability_end_to_end_representation_min"
                ],
                "v10_e2e_representation_max": cyclic[
                    "probability_end_to_end_representation_max"
                ],
                "v10_e2e_representation_span": cyclic[
                    "probability_end_to_end_representation_span"
                ],
                "v10_e2e_threshold_disagreement": cyclic[
                    "end_to_end_representation_threshold_disagreement"
                ],
                "original_v28_known_methyl_call": original_known_call,
                "original_v28_e2e_methyl_call": original_e2e_call,
                "original_v28_known_methyl_state_correct": int(
                    original_known_call == corrected_true
                ),
                "original_v28_e2e_methyl_state_correct": int(
                    original_e2e_call == corrected_true
                ),
                "original_v28_end_to_end_exact_methyl_residue_recovered": int(
                    float(
                        original.get(
                            "end_to_end_exact_methyl_residue_recovered", 0
                        )
                    )
                ),
                "original_v28_end_to_end_extended_token_correct": int(
                    float(
                        original.get("end_to_end_extended_token_correct", 0)
                    )
                ),
                "v10_known_stable_methyl_call": v10_known_call,
                "v10_e2e_stable_methyl_call": v10_e2e_call,
                "v10_known_methyl_state_correct": int(
                    v10_known_call == corrected_true
                ),
                "v10_e2e_methyl_state_correct": int(
                    v10_e2e_call == corrected_true
                ),
                "v10_end_to_end_exact_methyl_residue_recovered": int(
                    corrected_true == 1
                    and v10_e2e_call == 1
                    and v10["pred_base_token"] == v10["true_base_token"]
                ),
                "v10_end_to_end_extended_token_correct": int(
                    v10["pred_base_token"] == v10["true_base_token"]
                    and strict_pass(
                        cyclic["probability_end_to_end_representation_min"]
                    )
                    == bool(corrected_true)
                ),
            }
        )

    checks["all_corrected_ground_truth_labels_match_cyclic_audit"] = bool(comparison_rows) and all(
        int(row["is_methyl_true_corrected"])
        == int(cyclic_index[(row["sample_name"], int(row["position_in_model"]))]["is_methyl_true"])
        for row in comparison_rows
    )
    checks["base_head_predictions_match_deterministic_parent_for_all_1505_positions"] = (
        len(comparison_rows) == EXPECTED_POSITIONS
        and all(int(row["base_prediction_unchanged"]) == 1 for row in comparison_rows)
    )
    checks["deterministic_parent_and_v10_ground_truth_tokens_match"] = (
        len(comparison_rows) == EXPECTED_POSITIONS
        and all(
            parent_index[
                (
                    str(row["sample_name"]),
                    str(row["selected_chains"]),
                    int(row["position_in_model"]),
                )
            ]["true_base_token"]
            == row["true_base_token"]
            for row in comparison_rows
        )
    )
    checks["corrected_positive_count_is_261"] = (
        sum(int(row["is_methyl_true_corrected"]) for row in comparison_rows)
        == EXPECTED_POSITIVES
    )
    checks["known_sequence_cyclic_start_disagreement_is_zero"] = all(
        int(row["v10_known_threshold_disagreement"]) == 0 for row in comparison_rows
    )
    checks["end_to_end_cyclic_start_disagreement_is_zero"] = all(
        int(row["v10_e2e_threshold_disagreement"]) == 0 for row in comparison_rows
    )

    labels = [int(row["is_methyl_true_corrected"]) for row in comparison_rows]
    known_scores = [float(row["v10_known_representation_min"]) for row in comparison_rows]
    e2e_scores = [float(row["v10_e2e_representation_min"]) for row in comparison_rows]
    known_metrics = classification_metrics(labels, known_scores) if labels else {}
    e2e_metrics = classification_metrics(labels, e2e_scores) if labels else {}
    base_recovery = (
        sum(
            row["v10_pred_base_token"] == row["true_base_token"]
            for row in comparison_rows
        )
        / len(comparison_rows)
        if comparison_rows
        else math.nan
    )
    extended_recovery = (
        sum(int(row["v10_end_to_end_extended_token_correct"]) for row in comparison_rows)
        / len(comparison_rows)
        if comparison_rows
        else math.nan
    )
    metric_rows = [
        {"scope": "monomer_v10", "task": "base_identity", "metric": "base_recovery", "value": base_recovery},
        {"scope": "monomer_v10", "task": "end_to_end", "metric": "extended_token_recovery", "value": extended_recovery},
    ]
    for task, metrics in (("known_sequence_stable_floor", known_metrics), ("end_to_end_stable_floor", e2e_metrics)):
        for metric, value in metrics.items():
            metric_rows.append(
                {"scope": "monomer_v10", "task": task, "metric": metric, "value": value}
            )

    exact_methyl_count = sum(
        int(row["v10_end_to_end_exact_methyl_residue_recovered"])
        for row in comparison_rows
    )
    exact_methyl_rate = (
        exact_methyl_count / EXPECTED_POSITIVES if comparison_rows else math.nan
    )
    metric_rows.extend(
        [
            {
                "scope": "monomer_v10",
                "task": "end_to_end",
                "metric": "exact_methylated_residue_recovery_count",
                "value": exact_methyl_count,
            },
            {
                "scope": "monomer_v10",
                "task": "end_to_end",
                "metric": "exact_methylated_residue_recovery_rate_among_true_methyl_sites",
                "value": exact_methyl_rate,
            },
        ]
    )

    threshold_rows: List[Dict[str, Any]] = []
    for task, scores in (
        ("known_sequence_stable_floor", known_scores),
        ("end_to_end_stable_floor", e2e_scores),
    ):
        for threshold in SENSITIVITY_THRESHOLDS:
            row = {
                "scope": "overall",
                "task": task,
                **classification_metrics(labels, scores, threshold),
            }
            threshold_rows.append(row)

    by_residue_rows: List[Dict[str, Any]] = []
    scopes: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("overall", comparison_rows),
        (
            "serine",
            [row for row in comparison_rows if row["true_base_token"] == "S"],
        ),
        (
            "non_serine",
            [row for row in comparison_rows if row["true_base_token"] != "S"],
        ),
    ]
    scopes.extend(
        (
            f"residue_{residue}",
            [
                row
                for row in comparison_rows
                if row["true_base_token"] == residue
            ],
        )
        for residue in NATURAL_AA
    )
    for scope, scoped_rows in scopes:
        if not scoped_rows:
            continue
        scoped_labels = [
            int(row["is_methyl_true_corrected"]) for row in scoped_rows
        ]
        scoped_base_recovery = sum(
            row["v10_pred_base_token"] == row["true_base_token"]
            for row in scoped_rows
        ) / len(scoped_rows)
        for task, score_field in (
            ("known_sequence_stable_floor", "v10_known_representation_min"),
            ("end_to_end_stable_floor", "v10_e2e_representation_min"),
        ):
            metrics = classification_metrics(
                scoped_labels,
                [float(row[score_field]) for row in scoped_rows],
            )
            by_residue_rows.append(
                {
                    "scope": scope,
                    "task": task,
                    "base_recovery": scoped_base_recovery,
                    **metrics,
                }
            )

    panel_rows: List[Dict[str, Any]] = []
    for panel in sorted({str(row["source_panel"]) for row in comparison_rows}):
        scoped_rows = [row for row in comparison_rows if row["source_panel"] == panel]
        scoped_labels = [
            int(row["is_methyl_true_corrected"]) for row in scoped_rows
        ]
        panel_record: Dict[str, Any] = {
            "source_panel": panel,
            "monomers": len({str(row["sample_name"]) for row in scoped_rows}),
            "positions": len(scoped_rows),
            "positives": sum(scoped_labels),
            "base_recovery": sum(
                row["v10_pred_base_token"] == row["true_base_token"]
                for row in scoped_rows
            )
            / len(scoped_rows),
        }
        for prefix, score_field in (
            ("known", "v10_known_representation_min"),
            ("e2e", "v10_e2e_representation_min"),
        ):
            metrics = classification_metrics(
                scoped_labels,
                [float(row[score_field]) for row in scoped_rows],
            )
            for metric, value in metrics.items():
                panel_record[f"{prefix}_{metric}"] = value
        panel_rows.append(panel_record)

    paired_rows: List[Dict[str, Any]] = []
    for task, old_score_field, new_score_field, old_correct_field, new_correct_field in (
        (
            "known_sequence",
            "original_v28_known_probability_single_representation",
            "v10_known_representation_min",
            "original_v28_known_methyl_state_correct",
            "v10_known_methyl_state_correct",
        ),
        (
            "end_to_end",
            "original_v28_e2e_probability_single_representation",
            "v10_e2e_representation_min",
            "original_v28_e2e_methyl_state_correct",
            "v10_e2e_methyl_state_correct",
        ),
    ):
        old_metrics = classification_metrics(
            labels,
            [float(row[old_score_field]) for row in comparison_rows],
        )
        new_metrics = classification_metrics(
            labels,
            [float(row[new_score_field]) for row in comparison_rows],
        )
        bootstrap = paired_cluster_bootstrap(
            comparison_rows, old_score_field, new_score_field
        )
        old_only = sum(
            int(row[old_correct_field]) == 1 and int(row[new_correct_field]) == 0
            for row in comparison_rows
        )
        new_only = sum(
            int(row[old_correct_field]) == 0 and int(row[new_correct_field]) == 1
            for row in comparison_rows
        )
        for metric in (
            "roc_auc",
            "average_precision",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "predicted_methyl_rate",
        ):
            lower, upper = bootstrap[metric]
            paired_rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "original_v28_value": old_metrics[metric],
                    "v10_value": new_metrics[metric],
                    "v10_minus_original": float(new_metrics[metric])
                    - float(old_metrics[metric]),
                    "paired_sample_cluster_bootstrap_delta_ci95_low": lower,
                    "paired_sample_cluster_bootstrap_delta_ci95_high": upper,
                    "bootstrap_replicates": 1000,
                    "mcnemar_old_only_correct": old_only,
                    "mcnemar_v10_only_correct": new_only,
                    "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(
                        old_only, new_only
                    ),
                }
            )

    native_labels = [0] * len(native_control_rows)
    native_known_scores = [
        float(row["known_representation_min"]) for row in native_control_rows
    ]
    native_e2e_scores = [
        float(row["e2e_representation_min"]) for row in native_control_rows
    ]
    native_known_metrics = (
        classification_metrics(native_labels, native_known_scores)
        if native_labels
        else {}
    )
    native_e2e_metrics = (
        classification_metrics(native_labels, native_e2e_scores)
        if native_labels
        else {}
    )
    for task, metrics in (
        ("native17_known_sequence_all_negative_control", native_known_metrics),
        ("native17_end_to_end_all_negative_control", native_e2e_metrics),
    ):
        for metric, value in metrics.items():
            metric_rows.append(
                {
                    "scope": "17_native_complex_peptide_positions",
                    "task": task,
                    "metric": metric,
                    "value": value,
                }
            )

    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        by_sample[str(row["sample_name"])].append(row)
    design_rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    fasta_lines: List[str] = []
    for sample_name, rows in sorted(by_sample.items()):
        rows.sort(key=lambda row: int(row["position_in_model"]))
        reference = "".join(str(row["true_base_token"]) for row in rows)
        known = "".join(
            str(row["true_base_token"]).lower()
            if int(row["v10_known_stable_methyl_call"])
            else str(row["true_base_token"]).upper()
            for row in rows
        )
        e2e = "".join(
            str(row["v10_pred_base_token"]).lower()
            if int(row["v10_e2e_stable_methyl_call"])
            else str(row["v10_pred_base_token"]).upper()
            for row in rows
        )
        known_positions = [index for index, token in enumerate(known, start=1) if token.islower()]
        e2e_positions = [index for index, token in enumerate(e2e, start=1) if token.islower()]
        per_sample_rows.append(
            {
                "sample_name": sample_name,
                "source_panel": rows[0]["source_panel"],
                "selected_chains": rows[0]["selected_chains"],
                "positions": len(rows),
                "true_methyl_positions": sum(
                    int(row["is_methyl_true_corrected"]) for row in rows
                ),
                "base_correct_positions": sum(
                    row["v10_pred_base_token"] == row["true_base_token"]
                    for row in rows
                ),
                "base_recovery": sum(
                    row["v10_pred_base_token"] == row["true_base_token"]
                    for row in rows
                )
                / len(rows),
                "known_stable_methyl_calls": len(known_positions),
                "e2e_stable_methyl_calls": len(e2e_positions),
                "known_methyl_state_correct_positions": sum(
                    int(row["v10_known_methyl_state_correct"]) for row in rows
                ),
                "e2e_methyl_state_correct_positions": sum(
                    int(row["v10_e2e_methyl_state_correct"]) for row in rows
                ),
                "e2e_exact_methyl_residue_recovered": sum(
                    int(row["v10_end_to_end_exact_methyl_residue_recovered"])
                    for row in rows
                ),
                "e2e_extended_token_correct_positions": sum(
                    int(row["v10_end_to_end_extended_token_correct"])
                    for row in rows
                ),
                "known_representation_span_mean": float(
                    np.mean(
                        [float(row["v10_known_representation_span"]) for row in rows]
                    )
                ),
                "known_representation_span_max": max(
                    float(row["v10_known_representation_span"]) for row in rows
                ),
                "e2e_representation_span_mean": float(
                    np.mean(
                        [float(row["v10_e2e_representation_span"]) for row in rows]
                    )
                ),
                "e2e_representation_span_max": max(
                    float(row["v10_e2e_representation_span"]) for row in rows
                ),
                "known_threshold_disagreement_positions": sum(
                    int(row["v10_known_threshold_disagreement"]) for row in rows
                ),
                "e2e_threshold_disagreement_positions": sum(
                    int(row["v10_e2e_threshold_disagreement"]) for row in rows
                ),
            }
        )
        design_rows.append(
            {
                "sample_name": sample_name,
                "source_panel": rows[0]["source_panel"],
                "selected_chains": rows[0]["selected_chains"],
                "threshold": THRESHOLD,
                "threshold_operator": ">",
                "probability_rounding_policy": "round(prob,8)",
                "reference_natural_sequence": reference,
                "known_sequence_stable_methyl_design": known,
                "known_sequence_methyl_positions_1based": ",".join(map(str, known_positions)),
                "known_sequence_methyl_count": len(known_positions),
                "e2e_stable_methyl_design": e2e,
                "e2e_natural_sequence_for_structure_prediction": e2e.upper(),
                "e2e_methyl_positions_1based": ",".join(map(str, e2e_positions)),
                "e2e_methyl_count": len(e2e_positions),
                "e2e_has_at_least_one_predicted_stable_methylation": int(bool(e2e_positions)),
                "naturalized_variant4_pdb_reuse_status": (
                    "ELIGIBLE_PENDING_WINDOWS_151_FILE_SEQUENCE_AUDIT"
                ),
                "explicit_methyl_variant3_pdb_reuse_status": (
                    "NOT_AUTHORIZED_UNLESS_MARKED_SEQUENCE_ALSO_MATCHES"
                ),
            }
        )
        fasta_lines.extend(
            [
                f">v10_monomer_{sample_name}|marked={e2e}|methyl_positions={','.join(map(str, e2e_positions))}",
                e2e.upper(),
            ]
        )
    checks["design_manifest_has_exactly_151_monomers"] = len(design_rows) == EXPECTED_SAMPLES
    checks["every_monomer_design_has_nonempty_canonical_natural_sequence"] = all(
        row["e2e_natural_sequence_for_structure_prediction"]
        and row["e2e_natural_sequence_for_structure_prediction"].isupper()
        for row in design_rows
    )

    position_out = out_dir / "monomer_v10_position_comparison_1505.csv"
    metric_out = out_dir / "monomer_v10_metrics.csv"
    threshold_out = out_dir / "monomer_v10_threshold_curves.csv"
    residue_out = out_dir / "monomer_v10_by_residue.csv"
    panel_out = out_dir / "monomer_v10_by_company_rosetta_panel.csv"
    sample_out = out_dir / "monomer_v10_per_sample.csv"
    paired_out = out_dir / "monomer_v10_paired_original_v28_comparison.csv"
    native_control_out = out_dir / "native17_v10_all_negative_control.csv"
    design_out = out_dir / "monomer_v10_design_manifest_151.csv"
    fasta_out = out_dir / "monomer_v10_structure_input_if_reprediction_needed.fasta"
    atomic_write_csv(position_out, comparison_rows)
    atomic_write_csv(metric_out, metric_rows)
    atomic_write_csv(threshold_out, threshold_rows)
    atomic_write_csv(residue_out, by_residue_rows)
    atomic_write_csv(panel_out, panel_rows)
    atomic_write_csv(sample_out, per_sample_rows)
    atomic_write_csv(paired_out, paired_rows)
    atomic_write_csv(native_control_out, native_control_rows)
    atomic_write_csv(design_out, design_rows)
    atomic_write_text(fasta_out, "\n".join(fasta_lines) + "\n")
    checks["reopened_outputs_have_exact_frozen_row_counts"] = (
        len(read_csv(position_out)) == EXPECTED_POSITIONS
        and len(read_csv(design_out)) == EXPECTED_SAMPLES
        and len(read_csv(sample_out)) == EXPECTED_SAMPLES
        and len(read_csv(threshold_out)) == 2 * len(SENSITIVITY_THRESHOLDS)
        and len(read_csv(residue_out)) == 46
        and len(read_csv(panel_out)) == 2
        and len(read_csv(paired_out)) == 16
        and len(read_csv(native_control_out)) == EXPECTED_NATIVE_COMPLEX_POSITIONS
    )
    quality_gate = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "quality_gate": quality_gate,
        "release_status": (
            "MONOMER_SEQUENCE_AUDIT_PASS_WINDOWS_STRUCTURE_RECALCULATION_PENDING"
            if quality_gate == "PASS"
            else "BLOCKED_MONOMER_V10_AUDIT_FAILED"
        ),
        "protocol": PROTOCOL,
        "data_source": (
            "751 company-generated Rosetta theoretical monomers; corrected split "
            "600 train / 151 internal development audit"
        ),
        "validation_scope": (
            "internal paired audit, not a new blind experimental-structure test"
        ),
        "quality_checks": checks,
        "sample_count": len(design_rows),
        "position_count": len(comparison_rows),
        "corrected_positive_count": sum(labels),
        "base_recovery": base_recovery,
        "end_to_end_extended_token_recovery": extended_recovery,
        "end_to_end_exact_methylated_residue_recovery_count": exact_methyl_count,
        "end_to_end_exact_methylated_residue_recovery_rate": exact_methyl_rate,
        "known_sequence_stable_floor_metrics": known_metrics,
        "end_to_end_stable_floor_metrics": e2e_metrics,
        "native17_known_sequence_all_negative_control": native_known_metrics,
        "native17_end_to_end_all_negative_control": native_e2e_metrics,
        "samples_with_at_least_one_end_to_end_stable_methyl_call": sum(
            int(row["e2e_has_at_least_one_predicted_stable_methylation"])
            for row in design_rows
        ),
        "structure_reuse_contract": {
            "naturalized_reference_variant2": "reuse_allowed",
            "naturalized_old_v28_e2e_variant4": (
                "reuse_allowed_only_after Windows verifies all 151 PDB filename/sequence "
                "records match this manifest; base freeze alone is necessary but not sufficient"
            ),
            "explicit_methyl_variant3": (
                "do not reuse when the V10 marked sequence differs; regenerate or report missing"
            ),
        },
        "inputs": {
            "v10_position_csv": {"path": str(v10_position_path), "sha256": sha256_file(v10_position_path)},
            "v10_eval_manifest": {"path": str(v10_eval_manifest_path), "sha256": sha256_file(v10_eval_manifest_path)},
            "parent_position_csv": {"path": str(parent_position_path), "sha256": sha256_file(parent_position_path)},
            "parent_eval_manifest": {"path": str(parent_eval_manifest_path), "sha256": sha256_file(parent_eval_manifest_path)},
            "cyclic_position_csv": {"path": str(cyclic_position_path), "sha256": sha256_file(cyclic_position_path)},
            "native_position_csv": {"path": str(native_position_path), "sha256": sha256_file(native_position_path)},
            "cyclic_audit_manifest": {"path": str(cyclic_audit_path), "sha256": sha256_file(cyclic_audit_path)},
            "original_v28_corrected_csv": {"path": str(original_path), "sha256": sha256_file(original_path)},
            "v10_model": {"path": str(v10_model_path), "sha256": sha256_file(v10_model_path)},
            "parent_model": {"path": str(parent_model_path), "sha256": sha256_file(parent_model_path)},
        },
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "dependencies": {
            "monomer_evaluator": {
                "path": str(eval_manifest.get("program", {}).get("path", "")),
                "sha256": str(eval_manifest.get("program", {}).get("sha256", "")),
            },
            **{
                f"monomer_evaluator_{name}": {
                    "path": str(record.get("path", "")),
                    "sha256": str(record.get("sha256", "")),
                }
                for name, record in (
                    eval_manifest.get("dependencies", {}).items()
                    if isinstance(eval_manifest.get("dependencies", {}), Mapping)
                    else []
                )
                if isinstance(record, Mapping)
            },
            "rmsd_ranker_module": {
                "path": str(SCRIPT_PATH.with_name("rmsd_ranker_v10.py")),
                "sha256": sha256_file(SCRIPT_PATH.with_name("rmsd_ranker_v10.py")),
            },
        },
        "artifacts": {
            "position_comparison": {"path": str(position_out), "sha256": sha256_file(position_out)},
            "metrics": {"path": str(metric_out), "sha256": sha256_file(metric_out)},
            "threshold_curves": {"path": str(threshold_out), "sha256": sha256_file(threshold_out)},
            "by_residue": {"path": str(residue_out), "sha256": sha256_file(residue_out)},
            "by_company_rosetta_panel": {"path": str(panel_out), "sha256": sha256_file(panel_out)},
            "per_sample": {"path": str(sample_out), "sha256": sha256_file(sample_out)},
            "paired_original_v28_comparison": {"path": str(paired_out), "sha256": sha256_file(paired_out)},
            "native17_all_negative_control": {"path": str(native_control_out), "sha256": sha256_file(native_control_out)},
            "design_manifest": {"path": str(design_out), "sha256": sha256_file(design_out)},
            "structure_fasta_if_needed": {"path": str(fasta_out), "sha256": sha256_file(fasta_out)},
        },
    }
    atomic_write_json(out_dir / "monomer_v10_manifest.json", report)
    print("===== V10 MONOMER AUDIT =====", flush=True)
    print(f"Positions: {len(comparison_rows)}; monomers: {len(design_rows)}", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("V10 monomer audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
