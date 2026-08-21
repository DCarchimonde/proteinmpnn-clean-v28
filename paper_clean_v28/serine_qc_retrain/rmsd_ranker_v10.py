#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low-capacity, target-held-out structural-risk ranker for V10.

The methyl expert heads do not change ProteinMPNN's natural-amino-acid
sampling distribution.  This module therefore provides a deliberately small
and auditable *post-generation* model.  It only uses attributes available
before a candidate structure is predicted and never uses a target identifier
as a feature.  Hyperparameters are frozen in source; validation holds out one
entire target at a time.

The ranker is a class-balanced, L2-regularised logistic regression trained by
deterministic full-batch Adam using NumPy.  The joint <5 A endpoint is primary.
The much rarer joint <3 A endpoint is retained as a descriptive secondary
score and can never by itself authorize V10 release.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_NAMES = (
    "peptide_length",
    "natural_aa_recovery",
    "methyl_count",
    "methyl_rate",
    "normalized_sequence_entropy",
    "maximum_residue_fraction",
    "glycine_fraction",
    "proline_fraction",
    "cysteine_fraction",
    "charged_fraction_DEKR",
    "hydrophobic_fraction_AVILMFWY",
    "aromatic_fraction_FWY",
    "polar_fraction_STNQ",
    "basic_fraction_KRH",
    "acidic_fraction_DE",
    "net_charge_proxy_fraction",
)
MODEL_PROTOCOL = "target_heldout_low_capacity_logistic_joint_rmsd_v10"
PRIMARY_LABEL = "joint_lt5"
SECONDARY_LABEL = "joint_lt3"
SEED = 20260821
L2_PENALTY = 10.0
LEARNING_RATE = 0.03
EPOCHS = 3000
TOP_FRACTION = 0.25


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite")
    return result

def _fraction(counts: Counter[str], residues: str, length: int) -> float:
    return sum(counts[token] for token in residues) / length


def sequence_features(row: Mapping[str, Any]) -> np.ndarray:
    """Return the frozen, target-agnostic pre-structure feature vector."""

    marked = str(row.get("design_seq", "")).strip()
    natural = str(row.get("design_natural_seq", "")).strip().upper()
    if not natural:
        natural = marked.upper()
    if not natural or any(token not in NATURAL_AA for token in natural):
        raise ValueError("design_natural_seq is empty or noncanonical")
    if marked and marked.upper() != natural:
        raise ValueError("design_seq and design_natural_seq disagree after naturalization")
    length = len(natural)
    native = str(row.get("native_seq", "")).strip().upper()
    if native and len(native) != length:
        raise ValueError("native_seq and design sequence lengths differ")

    if native:
        recovery = sum(a == b for a, b in zip(native, natural)) / length
    else:
        recovery = _finite_float(row.get("natural_aa_recovery"), "natural_aa_recovery")
    persisted_recovery = row.get("natural_aa_recovery", "")
    if str(persisted_recovery).strip():
        observed = _finite_float(persisted_recovery, "natural_aa_recovery")
        if abs(observed - recovery) > 5e-8:
            raise ValueError("natural_aa_recovery does not recompute")

    lowercase_count = sum(token.islower() for token in marked)
    if not marked:
        lowercase_count = int(_finite_float(row.get("design_methyl_count"), "design_methyl_count"))
    persisted_count = row.get("design_methyl_count", "")
    if str(persisted_count).strip() and int(_finite_float(persisted_count, "design_methyl_count")) != lowercase_count:
        raise ValueError("design_methyl_count does not recompute")
    methyl_rate = lowercase_count / length
    persisted_rate = row.get("design_methyl_rate", "")
    if str(persisted_rate).strip():
        observed_rate = _finite_float(persisted_rate, "design_methyl_rate")
        if abs(observed_rate - methyl_rate) > 5e-8:
            raise ValueError("design_methyl_rate does not recompute")

    counts: Counter[str] = Counter(natural)
    fractions = np.asarray([counts[token] / length for token in NATURAL_AA], dtype=float)
    nonzero = fractions[fractions > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(len(NATURAL_AA)))
    features = np.asarray(
        [
            float(length),
            recovery,
            float(lowercase_count),
            methyl_rate,
            entropy,
            float(np.max(fractions)),
            counts["G"] / length,
            counts["P"] / length,
            counts["C"] / length,
            _fraction(counts, "DEKR", length),
            _fraction(counts, "AVILMFWY", length),
            _fraction(counts, "FWY", length),
            _fraction(counts, "STNQ", length),
            _fraction(counts, "KRH", length),
            _fraction(counts, "DE", length),
            (_fraction(counts, "KR", length) - _fraction(counts, "DE", length)),
        ],
        dtype=float,
    )
    if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
        raise ValueError("pre-structure feature vector is invalid")
    return features


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot build features for an empty row set")
    return np.vstack([sequence_features(row) for row in rows])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def train_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2_penalty: float = L2_PENALTY,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
) -> Dict[str, Any]:
    """Fit one deterministic class-balanced logistic model."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Feature/label shape mismatch")
    if not np.all(np.isfinite(x)) or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("Features or labels are invalid")
    positives = int(np.sum(y == 1.0))
    negatives = int(np.sum(y == 0.0))
    if positives == 0 or negatives == 0:
        raise ValueError("Both label classes are required")

    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1], dtype=float)
    bias = 0.0
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = 0.0
    second_b = 0.0
    sample_weight = np.where(
        y == 1.0,
        len(y) / (2.0 * positives),
        len(y) / (2.0 * negatives),
    )

    for step in range(1, int(epochs) + 1):
        probabilities = _sigmoid(z @ weights + bias)
        errors = (probabilities - y) * sample_weight
        gradient_w = z.T @ errors / len(y) + l2_penalty * weights / len(y)
        gradient_b = float(np.mean(errors))
        first_w = 0.9 * first_w + 0.1 * gradient_w
        second_w = 0.999 * second_w + 0.001 * gradient_w * gradient_w
        first_b = 0.9 * first_b + 0.1 * gradient_b
        second_b = 0.999 * second_b + 0.001 * gradient_b * gradient_b
        corrected_first_w = first_w / (1.0 - 0.9**step)
        corrected_second_w = second_w / (1.0 - 0.999**step)
        corrected_first_b = first_b / (1.0 - 0.9**step)
        corrected_second_b = second_b / (1.0 - 0.999**step)
        weights -= learning_rate * corrected_first_w / (np.sqrt(corrected_second_w) + 1e-8)
        bias -= learning_rate * corrected_first_b / (math.sqrt(corrected_second_b) + 1e-8)

    return {
        "protocol": MODEL_PROTOCOL,
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "training_rows": len(y),
        "training_positives": positives,
        "training_negatives": negatives,
        "l2_penalty": l2_penalty,
        "learning_rate": learning_rate,
        "epochs": int(epochs),
        "seed": SEED,
    }


def predict_logistic(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    if model.get("protocol") != MODEL_PROTOCOL:
        raise ValueError("Unexpected RMSD-ranker model protocol")
    if tuple(model.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("RMSD-ranker feature contract mismatch")
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Prediction feature width mismatch")
    result = _sigmoid(((x - mean) / scale) @ weights + float(model["bias"]))
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("RMSD-ranker emitted an invalid score")
    return result


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware Mann-Whitney ROC AUC without an sklearn dependency."""

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and s[order[end]] == s[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (np.sum(ranks[y == 1]) - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def leave_one_target_out(
    rows: Sequence[Mapping[str, Any]], label_field: str
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Return out-of-target predictions and per-target diagnostics."""

    x = feature_matrix(rows)
    labels = np.asarray([int(str(row[label_field]).strip()) for row in rows], dtype=int)
    targets = np.asarray([str(row["target_name"]).strip().upper() for row in rows])
    unique_targets = sorted(set(targets.tolist()))
    if len(unique_targets) < 3:
        raise ValueError("At least three targets are required for target-held-out validation")
    predictions = np.full(len(rows), np.nan, dtype=float)
    diagnostics: List[Dict[str, Any]] = []
    for target in unique_targets:
        heldout = targets == target
        training = ~heldout
        model = train_logistic(x[training], labels[training])
        predictions[heldout] = predict_logistic(model, x[heldout])
        target_labels = labels[heldout]
        target_scores = predictions[heldout]
        k = max(1, int(math.ceil(len(target_labels) * TOP_FRACTION)))
        ranking = np.argsort(-target_scores, kind="stable")[:k]
        diagnostics.append(
            {
                "target_name": target,
                "heldout_rows": int(np.sum(heldout)),
                "heldout_positives": int(np.sum(target_labels)),
                "heldout_rate": float(np.mean(target_labels)),
                "heldout_auc": binary_auc(target_labels, target_scores),
                "top_quartile_rows": k,
                "top_quartile_positives": int(np.sum(target_labels[ranking])),
                "top_quartile_rate": float(np.mean(target_labels[ranking])),
            }
        )
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Target-held-out predictions are incomplete")
    return predictions, diagnostics


def cross_validation_summary(
    rows: Sequence[Mapping[str, Any]], label_field: str
) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, Any]]]:
    predictions, diagnostics = leave_one_target_out(rows, label_field)
    labels = np.asarray([int(str(row[label_field]).strip()) for row in rows], dtype=int)
    targets = np.asarray([str(row["target_name"]).strip().upper() for row in rows])
    selected_indices: List[int] = []
    for target in sorted(set(targets.tolist())):
        indices = np.flatnonzero(targets == target)
        k = max(1, int(math.ceil(len(indices) * TOP_FRACTION)))
        local_order = np.argsort(-predictions[indices], kind="stable")[:k]
        selected_indices.extend(indices[local_order].tolist())
    baseline_rate = float(np.mean(labels))
    top_rate = float(np.mean(labels[selected_indices]))
    summary = {
        "label": label_field,
        "validation": "leave_one_entire_target_out",
        "rows": len(rows),
        "targets": len(set(targets.tolist())),
        "positives": int(np.sum(labels)),
        "baseline_rate": baseline_rate,
        "pooled_oof_auc": binary_auc(labels, predictions),
        "top_fraction_within_each_target": TOP_FRACTION,
        "top_fraction_rows": len(selected_indices),
        "top_fraction_positives": int(np.sum(labels[selected_indices])),
        "top_fraction_rate": top_rate,
        "absolute_enrichment": top_rate - baseline_rate,
        "relative_enrichment": top_rate / baseline_rate if baseline_rate else math.nan,
    }
    return summary, predictions, diagnostics


def historical_site_support(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Derive narrow exemptions for positions with prior <5 A support.

    An exemption is only eligible when a position appeared in at least 80% of
    a target's historical candidates, had at least three joint <5 A successes,
    and its conditional success rate was at least 75% of that target's overall
    rate.  Targets absent from this dataset receive no exemption.
    """

    by_target: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(str(row["target_name"]).upper(), []).append(row)
    result: Dict[str, Any] = {}
    for target, target_rows in sorted(by_target.items()):
        occurrences: Counter[int] = Counter()
        successes: Counter[int] = Counter()
        total_successes = sum(int(str(row[PRIMARY_LABEL])) for row in target_rows)
        target_rate = total_successes / len(target_rows)
        for row in target_rows:
            positions = [
                int(value)
                for value in str(row.get("methyl_positions_1based", "")).split(",")
                if value.strip()
            ]
            for position in set(positions):
                occurrences[position] += 1
                successes[position] += int(str(row[PRIMARY_LABEL]))
        position_rows = []
        supported: List[int] = []
        for position in sorted(occurrences):
            count = occurrences[position]
            pass_count = successes[position]
            conditional_rate = pass_count / count
            presence_rate = count / len(target_rows)
            qualifies = (
                presence_rate >= 0.80
                and pass_count >= 3
                and conditional_rate >= 0.75 * target_rate
            )
            if qualifies:
                supported.append(position)
            position_rows.append(
                {
                    "position_1based": position,
                    "candidate_count": count,
                    "candidate_presence_rate": presence_rate,
                    "joint_lt5_success_count": pass_count,
                    "joint_lt5_success_rate_given_position": conditional_rate,
                    "high_concentration_exemption_eligible": qualifies,
                }
            )
        result[target] = {
            "historical_rows": len(target_rows),
            "historical_joint_lt5_successes": total_successes,
            "historical_joint_lt5_rate": target_rate,
            "supported_high_concentration_positions_1based": supported,
            "position_evidence": position_rows,
        }
    return result
