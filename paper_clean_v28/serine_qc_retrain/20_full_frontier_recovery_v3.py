#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic full-frontier helpers for the V8 3ZGC V3 recovery.

V2 retained the 2,881 legacy strict methyl hits and the 996 baseline anchors,
but treated the other 265,484 hash-valid legacy observations only as a ``seen``
set.  Those rows could therefore neither seed the dual-objective beam nor be
used as bridge states between methyl confidence and cyclic-base plausibility.

This module repairs that search-space omission.  It does not change either
hard gate.  A deterministic sequence surrogate is fitted only to prioritize
expensive exact cyclic-base evaluations; every releasable row is still scored
by the frozen ProteinMPNN model and independently replayed by the existing
release audit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


V3_SEARCH_PROTOCOL = "full_legacy_frontier_cyclic_base_recovery_v8_v3"
V3_SURROGATE_PROTOCOL = "deterministic_position_adjacent_pair_backfit_v1"
V3_PRIOR_PROTOCOL = "cyclic_start_base_pareto_recovery_v8_v2"
V3_EXPECTED_PRIOR_FALSE_CHECK = "at_least_one_real_3zgc_candidate_is_released"
V3_DEFAULT_LEGACY_BRIDGE = 16_384
V3_SURROGATE_SHRINKAGE = 8.0
V3_SURROGATE_ITERATIONS = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def artifact_leaves(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if set(value) >= {"path", "sha256"}:
            yield value
        else:
            for child in value.values():
                yield from artifact_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from artifact_leaves(child)


def validate_prior_v2_failure(
    *,
    prior_dir: Path,
    expected_model_sha256: str,
    expected_baseline_manifest_sha256: str,
    expected_legacy_manifest_sha256: str,
    read_gzip_csv: Any,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    set[str],
]:
    """Validate and load the completed V2 failure as immutable V3 evidence."""

    prior_dir = prior_dir.resolve()
    manifest_path = prior_dir / "cyclic_base_recovery_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    false_checks = {
        str(name)
        for name, passed in dict(manifest.get("quality_checks") or {}).items()
        if not passed
    }
    if not (
        manifest.get("protocol") == V3_PRIOR_PROTOCOL
        and manifest.get("quality_gate") == "FAIL"
        and false_checks == {V3_EXPECTED_PRIOR_FALSE_CHECK}
        and manifest.get("release_status")
        == "BLOCKED_FIXED_V2_BUDGET_DID_NOT_RECOVER_3ZGC"
        and int(manifest.get("conditional_rounds_completed", -1)) == 6
        and int(manifest.get("legacy_strict_hits_reaudited", -1)) == 2881
        and int(manifest.get("released_candidates", -1)) == 0
        and manifest.get("missing_targets_after_search") == ["3ZGC"]
        and manifest.get("model_sha256") == expected_model_sha256
        and manifest.get("baseline_manifest_sha256")
        == expected_baseline_manifest_sha256
        and manifest.get("legacy_manifest_sha256")
        == expected_legacy_manifest_sha256
    ):
        raise RuntimeError("Prior V2 directory is not the exact completed 3ZGC failure")

    leaves = list(artifact_leaves(manifest.get("artifacts")))
    if not leaves:
        raise RuntimeError("Prior V2 manifest has no declared artifacts")
    for leaf in leaves:
        path = Path(str(leaf.get("path", ""))).resolve()
        try:
            path.relative_to(prior_dir)
        except ValueError as exc:
            raise RuntimeError(f"Prior V2 artifact escapes its directory: {path}") from exc
        if not path.is_file() or sha256_file(path) != str(leaf.get("sha256", "")):
            raise RuntimeError(f"Prior V2 artifact is absent or stale: {path}")

    exact_rows: Dict[str, Dict[str, Any]] = {}
    screen_rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    artifacts = dict(manifest["artifacts"])
    for leaf in dict(artifacts["conditional_methyl_screens"]).values():
        for raw in read_gzip_csv(Path(str(leaf["path"]))):
            row = dict(raw)
            sequence = str(row.get("sequence", "")).upper()
            maximum = float(row.get("maximum_probability", float("nan")))
            argmax = int(row.get("argmax_position_1based", -1))
            strict = int(row.get("passes_strict_probability", -1))
            if not (
                len(sequence) == 7
                and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
                and sequence not in seen
                and math.isfinite(maximum)
                and 0.0 <= maximum <= 1.0
                and 1 <= argmax <= 7
                and str(row.get("argmax_residue", "")) == sequence[argmax - 1]
                and strict == int(round(maximum, 8) > 0.6)
            ):
                raise RuntimeError("Prior V2 methyl screens are duplicated or malformed")
            seen.add(sequence)
            row.update(
                {
                    "sequence": sequence,
                    "maximum_probability": maximum,
                    "argmax_position_1based": argmax,
                    "passes_strict_probability": strict,
                }
            )
            screen_rows.append(row)
    for leaf in dict(artifacts["conditional_cyclic_base_shortlists"]).values():
        for row in read_gzip_csv(Path(str(leaf["path"]))):
            sequence = str(row.get("sequence", "")).upper()
            if not sequence or sequence in exact_rows:
                raise RuntimeError("Prior V2 cyclic-base rows are duplicated or malformed")
            normalized = dict(row)
            normalized["sequence"] = sequence
            for key in (
                "maximum_probability",
                "cyclic_base_log_probability_mean",
                "cyclic_base_log_probability_min",
                "cyclic_base_log_probability_max",
                "cyclic_base_log_probability_span",
                "cyclic_base_log_probability_std",
            ):
                normalized[key] = float(normalized[key])
            for key in (
                "argmax_position_1based",
                "passes_strict_probability",
                "cyclic_base_physical_start_count",
                "cyclic_base_decoder_order_count_per_start",
                "cyclic_base_total_ensemble_size",
            ):
                normalized[key] = int(normalized[key])
            exact_rows[sequence] = normalized
    if len(seen) != sum(
        int(row["newly_methyl_scored"])
        for row in read_csv_rows(prior_dir / "search_trace_by_round.csv")
    ):
        raise RuntimeError("Prior V2 seen-set count does not match its trace")
    if len(exact_rows) != 6 * 4096:
        raise RuntimeError("Prior V2 exact cyclic-base inventory is incomplete")
    strict_screen = {
        str(row["sequence"])
        for row in screen_rows
        if int(row["passes_strict_probability"]) == 1
    }
    if not strict_screen <= set(exact_rows):
        raise RuntimeError("Prior V2 exact inventory discarded a strict methyl row")
    return manifest, screen_rows, list(exact_rows.values()), seen


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class KmerBaseSurrogate:
    """Small deterministic additive model used only for acquisition ranking."""

    def __init__(
        self,
        alphabet: str,
        length: int,
        *,
        shrinkage: float = V3_SURROGATE_SHRINKAGE,
        iterations: int = V3_SURROGATE_ITERATIONS,
    ) -> None:
        if len(set(alphabet)) != len(alphabet) or length <= 0:
            raise ValueError("Invalid surrogate alphabet or sequence length")
        if shrinkage < 0.0 or iterations <= 0:
            raise ValueError("Invalid surrogate regularization")
        self.alphabet = alphabet
        self.length = int(length)
        self.shrinkage = float(shrinkage)
        self.iterations = int(iterations)
        self.index = {token: idx for idx, token in enumerate(alphabet)}
        self.global_mean = 0.0
        self.position = np.zeros((length, len(alphabet)), dtype=np.float64)
        self.adjacent = np.zeros(
            (max(0, length - 1), len(alphabet), len(alphabet)), dtype=np.float64
        )
        self.report: Dict[str, Any] = {}

    def _encode(self, sequences: Sequence[str]) -> np.ndarray:
        encoded = np.empty((len(sequences), self.length), dtype=np.int16)
        for row_index, sequence in enumerate(sequences):
            sequence = str(sequence).upper()
            if len(sequence) != self.length or any(
                token not in self.index for token in sequence
            ):
                raise ValueError(f"Invalid surrogate sequence: {sequence}")
            encoded[row_index] = [self.index[token] for token in sequence]
        return encoded

    def _predict_encoded(self, encoded: np.ndarray) -> np.ndarray:
        result = np.full(len(encoded), self.global_mean, dtype=np.float64)
        rows = np.arange(len(encoded))
        for position in range(self.length):
            result += self.position[position, encoded[:, position]]
        for position in range(self.length - 1):
            result += self.adjacent[
                position,
                encoded[:, position],
                encoded[:, position + 1],
            ]
        return result

    def _fit_arrays(self, encoded: np.ndarray, target: np.ndarray) -> None:
        alphabet_size = len(self.alphabet)
        self.global_mean = float(target.mean())
        self.position.fill(0.0)
        self.adjacent.fill(0.0)
        rows = np.arange(len(encoded))
        for _iteration in range(self.iterations):
            prediction = self._predict_encoded(encoded)
            for position in range(self.length):
                old = self.position[position, encoded[:, position]].copy()
                residual = target - (prediction - old)
                for token in range(alphabet_size):
                    mask = encoded[:, position] == token
                    count = int(mask.sum())
                    value = (
                        float(residual[mask].sum()) / (count + self.shrinkage)
                        if count
                        else 0.0
                    )
                    self.position[position, token] = value
                prediction = self._predict_encoded(encoded)
            for position in range(self.length - 1):
                old = self.adjacent[
                    position,
                    encoded[:, position],
                    encoded[:, position + 1],
                ].copy()
                residual = target - (prediction - old)
                pair_code = (
                    encoded[:, position].astype(np.int32) * alphabet_size
                    + encoded[:, position + 1]
                )
                sums = np.bincount(
                    pair_code,
                    weights=residual,
                    minlength=alphabet_size * alphabet_size,
                )
                counts = np.bincount(
                    pair_code, minlength=alphabet_size * alphabet_size
                )
                values = sums / (counts + self.shrinkage)
                values[counts == 0] = 0.0
                self.adjacent[position] = values.reshape(
                    alphabet_size, alphabet_size
                )
                prediction = self._predict_encoded(encoded)

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        by_sequence: Dict[str, float] = {}
        for row in rows:
            sequence = str(row["sequence"]).upper()
            value = float(row["cyclic_base_log_probability_mean"])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite exact base value: {sequence}")
            prior = by_sequence.setdefault(sequence, value)
            if abs(prior - value) > 2e-6:
                raise RuntimeError(f"Conflicting exact base values: {sequence}")
        sequences = sorted(by_sequence)
        if len(sequences) < 100:
            raise RuntimeError("V3 surrogate requires at least 100 exact rows")
        target = np.asarray([by_sequence[sequence] for sequence in sequences])
        encoded = self._encode(sequences)
        validation = np.asarray(
            [
                hashlib.sha256(sequence.encode("ascii")).digest()[0] % 5 == 0
                for sequence in sequences
            ],
            dtype=bool,
        )
        if int(validation.sum()) < 10 or int((~validation).sum()) < 50:
            raise RuntimeError("V3 surrogate deterministic split is too small")
        self._fit_arrays(encoded[~validation], target[~validation])
        predicted = self._predict_encoded(encoded[validation])
        residual = predicted - target[validation]
        correlation = float(
            np.corrcoef(predicted, target[validation])[0, 1]
            if float(np.std(predicted)) > 0.0
            and float(np.std(target[validation])) > 0.0
            else 0.0
        )
        report = {
            "protocol": V3_SURROGATE_PROTOCOL,
            "exact_unique_training_rows": len(sequences),
            "deterministic_validation_rows": int(validation.sum()),
            "validation_mae": float(np.mean(np.abs(residual))),
            "validation_rmse": float(np.sqrt(np.mean(residual**2))),
            "validation_pearson": correlation,
            "shrinkage": self.shrinkage,
            "iterations": self.iterations,
            "release_decisions_use_surrogate": False,
            "surrogate_use": "ACQUISITION_RANKING_ONLY",
        }
        self._fit_arrays(encoded, target)
        self.report = report
        return dict(report)

    def predict(self, sequences: Sequence[str]) -> Dict[str, float]:
        unique = sorted(set(str(sequence).upper() for sequence in sequences))
        encoded = self._encode(unique)
        values = self._predict_encoded(encoded)
        return {sequence: float(value) for sequence, value in zip(unique, values)}


def pareto_predicted_rows(
    rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, float]
) -> List[Dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row["maximum_probability"]),
            -float(predictions[str(row["sequence"])]),
            str(row["sequence"]),
        ),
    )
    front: List[Dict[str, Any]] = []
    best_base = float("-inf")
    for row in ordered:
        predicted = float(predictions[str(row["sequence"])])
        if predicted > best_base:
            row["surrogate_cyclic_base_log_probability_mean"] = predicted
            front.append(row)
            best_base = predicted
    return front


def select_surrogate_frontier(
    *,
    rows: Sequence[Mapping[str, Any]],
    surrogate: KmerBaseSurrogate,
    limit: int,
    length: int,
    floor: float,
    diversity_fill: Any,
    exclude_strict: bool = False,
) -> List[Dict[str, Any]]:
    """Select an exact-score shortlist without discarding either frontier."""

    unique = {str(row["sequence"]).upper(): dict(row) for row in rows}
    values = list(unique.values())
    if exclude_strict:
        values = [row for row in values if not int(row["passes_strict_probability"])]
    if limit <= 0:
        raise ValueError("Invalid V3 frontier limit")
    if len(values) <= limit:
        predictions = surrogate.predict([str(row["sequence"]) for row in values])
        for row in values:
            row["surrogate_cyclic_base_log_probability_mean"] = predictions[
                str(row["sequence"])
            ]
        return values
    predictions = surrogate.predict([str(row["sequence"]) for row in values])
    for row in values:
        row["surrogate_cyclic_base_log_probability_mean"] = predictions[
            str(row["sequence"])
        ]

    methyl_order = sorted(
        values,
        key=lambda row: (
            -int(row["passes_strict_probability"]),
            -float(row["maximum_probability"]),
            -float(row["surrogate_cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    base_order = sorted(
        values,
        key=lambda row: (
            -float(row["surrogate_cyclic_base_log_probability_mean"]),
            -float(row["maximum_probability"]),
            str(row["sequence"]),
        ),
    )
    base_values = np.asarray(
        [float(row["surrogate_cyclic_base_log_probability_mean"]) for row in values]
    )
    scale = max(0.25, float(np.quantile(base_values, 0.90) - np.quantile(base_values, 0.10)))
    acquisition = sorted(
        values,
        key=lambda row: (
            max(
                max(0.0, 0.6 - float(row["maximum_probability"])) / 0.4,
                max(
                    0.0,
                    floor
                    - float(row["surrogate_cyclic_base_log_probability_mean"]),
                )
                / scale,
            ),
            -(float(row["maximum_probability"]) - 0.6),
            -float(row["surrogate_cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    selected: Dict[str, Dict[str, Any]] = {}

    def add_row(row: Mapping[str, Any]) -> None:
        if len(selected) < limit:
            selected.setdefault(str(row["sequence"]), dict(row))

    def add_all(group: Sequence[Mapping[str, Any]]) -> None:
        for row in group:
            add_row(row)

    strict_rows = [row for row in methyl_order if int(row["passes_strict_probability"])]
    if len(strict_rows) > limit:
        raise RuntimeError(
            "V3 exact-score width is smaller than the strict-methyl inventory; "
            "refusing to discard a hard-gate hit"
        )
    # Hard-gate hits and physical-position coverage are mandatory.  The former
    # implementation inserted several oversized objective groups and then
    # sliced the insertion-ordered dictionary to ``limit``.  That could
    # silently remove the later Pareto/position groups even though the source
    # code appeared to add them.  Reserve them explicitly before balancing the
    # remaining objective rankings.
    add_all(strict_rows)
    per_position = max(1, limit // max(1, 10 * length))
    position_orders: List[List[Dict[str, Any]]] = []
    for position in range(1, length + 1):
        position_orders.append(
            [
                row
                for row in methyl_order
                if int(row["argmax_position_1based"]) == position
            ]
        )
    for rank in range(per_position):
        for group in position_orders:
            if rank < len(group):
                add_row(group[rank])

    pareto_order = pareto_predicted_rows(values, predictions)
    objective_orders = [methyl_order, base_order, acquisition, pareto_order]
    diversity_reserve = min(max(1, limit // 8), max(0, limit - len(selected)))
    objective_target = max(len(selected), limit - diversity_reserve)
    cursors = [0] * len(objective_orders)
    while len(selected) < objective_target:
        progressed = False
        for order_index, order in enumerate(objective_orders):
            while cursors[order_index] < len(order):
                row = order[cursors[order_index]]
                cursors[order_index] += 1
                sequence = str(row["sequence"])
                if sequence in selected:
                    continue
                add_row(row)
                progressed = True
                break
            if len(selected) >= objective_target:
                break
        if not progressed:
            break

    # Diversity is a final fill, not a fourth full objective scan.  Passing the
    # entire 400k-row frontier to max-min Hamming makes its initialization
    # quadratic in an already-large seed.  Interleave the three deterministic
    # objective rankings and retain eight alternatives per missing slot (at
    # least 4,096).  This preserves objective balance while bounding the exact
    # same max-min rule to a useful candidate pool.
    missing = max(0, limit - len(selected))
    pool_limit = min(
        max(0, len(values) - len(selected)), max(4096, missing * 8)
    )
    selected_sequences = set(selected)
    ranked_pool: List[Dict[str, Any]] = []
    pooled: set[str] = set()
    for index in range(max(len(order) for order in objective_orders)):
        for order in objective_orders:
            if index >= len(order):
                continue
            row = order[index]
            sequence = str(row["sequence"])
            if sequence in selected_sequences or sequence in pooled:
                continue
            pooled.add(sequence)
            ranked_pool.append(row)
            if len(ranked_pool) >= pool_limit:
                break
        if len(ranked_pool) >= pool_limit:
            break
    filled = diversity_fill(ranked_pool, list(selected.values()), limit)
    if len(filled) != limit or len({str(row["sequence"]) for row in filled}) != limit:
        raise RuntimeError("V3 frontier selection did not produce its frozen width")
    return filled


def select_exact_dual_objective_beam(
    *,
    rows: Sequence[Mapping[str, Any]],
    limit: int,
    length: int,
    floor: float,
    diversity_fill: Any,
) -> List[Dict[str, Any]]:
    """Balance the exact methyl/base beam without insertion-order truncation."""

    unique = {str(row["sequence"]).upper(): dict(row) for row in rows}
    values = list(unique.values())
    if limit <= 0 or len(values) < limit:
        raise ValueError("V3 exact beam requires at least its frozen width")
    methyl_order = sorted(
        values,
        key=lambda row: (
            -int(row["passes_strict_probability"]),
            -float(row["maximum_probability"]),
            -float(row["cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    base_order = sorted(
        values,
        key=lambda row: (
            -float(row["cyclic_base_log_probability_mean"]),
            -float(row["maximum_probability"]),
            str(row["sequence"]),
        ),
    )
    base_values = np.asarray(
        [float(row["cyclic_base_log_probability_mean"]) for row in values]
    )
    scale = max(
        0.25,
        float(np.quantile(base_values, 0.90) - np.quantile(base_values, 0.10)),
    )
    acquisition = sorted(
        values,
        key=lambda row: (
            max(
                max(0.0, 0.6 - float(row["maximum_probability"])) / 0.4,
                max(0.0, floor - float(row["cyclic_base_log_probability_mean"]))
                / scale,
            ),
            -(float(row["maximum_probability"]) - 0.6),
            -float(row["cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    strict_bridge = sorted(
        [row for row in values if int(row["passes_strict_probability"])],
        key=lambda row: (
            -float(row["cyclic_base_log_probability_mean"]),
            -float(row["maximum_probability"]),
            str(row["sequence"]),
        ),
    )
    base_pass_bridge = sorted(
        [
            row
            for row in values
            if float(row["cyclic_base_log_probability_mean"]) >= floor
        ],
        key=lambda row: (
            -float(row["maximum_probability"]),
            -float(row["cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    pareto_order = sorted(
        (dict(row) for row in values),
        key=lambda row: (
            -float(row["maximum_probability"]),
            -float(row["cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    pareto: List[Dict[str, Any]] = []
    best_base = float("-inf")
    for row in pareto_order:
        value = float(row["cyclic_base_log_probability_mean"])
        if value > best_base:
            pareto.append(row)
            best_base = value

    selected: Dict[str, Dict[str, Any]] = {}

    def add_row(row: Mapping[str, Any]) -> None:
        if len(selected) < limit:
            selected.setdefault(str(row["sequence"]), dict(row))

    joint = [
        row
        for row in acquisition
        if int(row["passes_strict_probability"])
        and float(row["cyclic_base_log_probability_mean"]) >= floor
    ]
    for row in joint[: max(1, limit // 8)]:
        add_row(row)

    per_position = max(1, limit // max(1, 16 * length))
    position_orders = [
        [
            row
            for row in methyl_order
            if int(row["argmax_position_1based"]) == position
        ]
        for position in range(1, length + 1)
    ]
    for rank in range(per_position):
        for group in position_orders:
            if rank < len(group):
                add_row(group[rank])

    objective_orders = [
        methyl_order,
        base_order,
        acquisition,
        strict_bridge,
        base_pass_bridge,
        pareto,
    ]
    diversity_reserve = min(max(1, limit // 8), max(0, limit - len(selected)))
    objective_target = max(len(selected), limit - diversity_reserve)
    cursors = [0] * len(objective_orders)
    while len(selected) < objective_target:
        progressed = False
        for order_index, order in enumerate(objective_orders):
            while cursors[order_index] < len(order):
                row = order[cursors[order_index]]
                cursors[order_index] += 1
                if str(row["sequence"]) in selected:
                    continue
                add_row(row)
                progressed = True
                break
            if len(selected) >= objective_target:
                break
        if not progressed:
            break

    missing = max(0, limit - len(selected))
    pool_limit = min(
        max(0, len(values) - len(selected)), max(2048, missing * 8)
    )
    ranked_pool: List[Dict[str, Any]] = []
    pooled: set[str] = set()
    for index in range(max(len(order) for order in objective_orders)):
        for order in objective_orders:
            if index >= len(order):
                continue
            row = order[index]
            sequence = str(row["sequence"])
            if sequence in selected or sequence in pooled:
                continue
            pooled.add(sequence)
            ranked_pool.append(row)
            if len(ranked_pool) >= pool_limit:
                break
        if len(ranked_pool) >= pool_limit:
            break
    filled = diversity_fill(ranked_pool, list(selected.values()), limit)
    if len(filled) != limit or len({str(row["sequence"]) for row in filled}) != limit:
        raise RuntimeError("V3 exact beam did not produce its frozen width")
    return filled


def frontier_summary(
    selected: Sequence[Mapping[str, Any]], floor: float
) -> Dict[str, Any]:
    probabilities = [float(row["maximum_probability"]) for row in selected]
    predicted = [
        float(row["surrogate_cyclic_base_log_probability_mean"]) for row in selected
    ]
    return {
        "selected_rows": len(selected),
        "strict_methyl_rows": sum(int(row["passes_strict_probability"]) for row in selected),
        "surrogate_base_pass_rows": sum(value >= floor for value in predicted),
        "maximum_methyl_probability": max(probabilities),
        "maximum_surrogate_base": max(predicted),
        "argmax_position_counts": dict(
            sorted(Counter(int(row["argmax_position_1based"]) for row in selected).items())
        ),
        "surrogate_is_never_a_release_gate": True,
    }


def validate_exact_frontier_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_sequences: Sequence[str],
    base_policy: str,
    length: int,
) -> List[Dict[str, Any]]:
    """Type-restore and validate a resumable exact V3 frontier artifact."""

    if [str(row.get("sequence", "")).upper() for row in rows] != list(
        expected_sequences
    ):
        raise RuntimeError("V3 frontier exact-score sequence/order mismatch")
    normalized: List[Dict[str, Any]] = []
    for raw, sequence in zip(rows, expected_sequences):
        row = dict(raw)
        row["sequence"] = sequence
        for key in (
            "maximum_probability",
            "surrogate_cyclic_base_log_probability_mean",
            "cyclic_base_log_probability_mean",
            "cyclic_base_log_probability_min",
            "cyclic_base_log_probability_max",
            "cyclic_base_log_probability_span",
            "cyclic_base_log_probability_std",
        ):
            row[key] = float(row[key])
        for key in (
            "argmax_position_1based",
            "passes_strict_probability",
            "cyclic_base_physical_start_count",
            "cyclic_base_decoder_order_count_per_start",
            "cyclic_base_total_ensemble_size",
        ):
            row[key] = int(row[key])
        vector = [
            float(value)
            for value in json.loads(str(row["cyclic_base_physical_start_scores"]))
        ]
        mean = float(row["cyclic_base_log_probability_mean"])
        if not (
            len(sequence) == length
            and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY")
            and len(vector) == length
            and row["cyclic_base_context_policy"] == base_policy
            and int(row["cyclic_base_physical_start_count"]) == length
            and int(row["cyclic_base_decoder_order_count_per_start"]) == length
            and int(row["cyclic_base_total_ensemble_size"]) == length * length
            and all(math.isfinite(value) for value in vector)
            and abs(sum(vector) / length - mean) <= 2e-6
            and abs(min(vector) - float(row["cyclic_base_log_probability_min"]))
            <= 2e-6
            and abs(max(vector) - float(row["cyclic_base_log_probability_max"]))
            <= 2e-6
        ):
            raise RuntimeError(f"Malformed V3 exact frontier row: {sequence}")
        normalized.append(row)
    return normalized
