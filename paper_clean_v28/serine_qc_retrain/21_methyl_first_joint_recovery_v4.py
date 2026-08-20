#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final methyl-first, fixed-budget recovery for V8 target 3ZGC.

V3 exhaustively preserved the historical/V2/V3 evidence but found no row that
passed both frozen hard gates.  V4 is deliberately narrower: it reuses every
hash-pinned exact score, builds deterministic acquisition-only surrogates, and
spends one final bounded methyl screen on crossover/local-lattice candidates.

The release policy cannot be relaxed.  A released or advisor-review row must
have explicit representation min/max/span/std, a representation minimum
strictly greater than 0.6 at every lowercase site, zero cyclic-start threshold
disagreement, and the exact minimum-derived lowercase pattern.  A release must
also pass the frozen 3ZGC cyclic-base floor, independent batch-one agreement,
and exact/forward-cyclic novelty.  Representation means have acquisition and
ranking authority only.

If the joint gate remains empty, V4 does *not* fabricate a release.  It writes a
separate advisor-review table containing only independently replayed methyl
hard-gate hits, ranked by distance to the unchanged base floor.  Non-methylated
rows can never enter that table.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
LEGACY_SEARCH_PATH = SCRIPT_PATH.with_name("14_directed_recovery_search_v8.py")
V2_SEARCH_PATH = SCRIPT_PATH.with_name("17_cyclic_base_recovery_v2.py")
V3_HELPER_PATH = SCRIPT_PATH.with_name("20_full_frontier_recovery_v3.py")
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"
DEFAULT_MODEL = V8_ROOT / "model" / "frankenstein_v28_source_scoped_hybrid_v8.pt"
DEFAULT_MODEL_MANIFEST = V8_ROOT / "model" / "expert_source_composition_manifest.json"
DEFAULT_REPRESENTATION = V8_ROOT / "representation_audit" / "cyclic_representation_audit.json"
DEFAULT_BASELINE = V8_ROOT / "generation_baseline"
DEFAULT_LEGACY = V8_ROOT / "directed_search"
DEFAULT_PRIOR_V2 = V8_ROOT / "directed_search_cyclic_base_v2"
DEFAULT_PRIOR_V3 = V8_ROOT / "directed_search_cyclic_base_v3_full_frontier"
DEFAULT_OUT = V8_ROOT / "directed_search_methyl_first_v4"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_representation_v6.json")
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_HISTORICAL = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "all_designs.csv"
)
DEFAULT_PRIOR_HANDOFF = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "rerun_temperature_0.5_multiseed"
    / "methylated_new_candidates.csv"
)

V4_PROTOCOL = "methyl_first_joint_feasibility_recovery_v8_v4"
V4_ACQUISITION_PROTOCOL = "deterministic_methyl_logit_plus_exact_base_surrogate_v1"
V4_EXPECTED_PRIOR_FALSE_CHECK = "at_least_one_real_3zgc_candidate_is_released"
THRESHOLD = 0.6
BASE_PERCENTILE = 0.01
RESCORE_TOLERANCE = 2e-6
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
METHYL_SCREEN_BUDGET = 24_576
EXACT_BASE_BUDGET = 2_048
ADVISOR_NEAR_MISS_LIMIT = 10
MAX_RELEASE = 200


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def artifact(path: Path) -> Dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


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


def validate_artifacts_under(manifest: Mapping[str, Any], root: Path) -> None:
    leaves = list(artifact_leaves(manifest.get("artifacts")))
    if not leaves:
        raise RuntimeError(f"Manifest has no artifact leaves: {root}")
    for leaf in leaves:
        path = Path(str(leaf.get("path", ""))).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Artifact escapes its immutable root: {path}") from exc
        if not path.is_file() or sha256_file(path) != str(leaf.get("sha256", "")):
            raise RuntimeError(f"Artifact is absent or stale: {path}")


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_gzip_csv(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def declared_paths(group: Mapping[str, Any]) -> List[Path]:
    return [Path(str(group[key]["path"])).resolve() for key in sorted(group)]


def validate_prior_failures(
    *,
    prior_v2_dir: Path,
    prior_v3_dir: Path,
    model_path: Path,
    baseline_manifest_path: Path,
    legacy_manifest_path: Path,
    v3: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate the exact completed V2 and V3 zero-release states."""

    v2_path = prior_v2_dir / "cyclic_base_recovery_manifest.json"
    v3_path = prior_v3_dir / "cyclic_base_recovery_manifest.json"
    if not v2_path.is_file() or not v3_path.is_file():
        raise FileNotFoundError(v2_path if not v2_path.is_file() else v3_path)
    v2_manifest = read_json(v2_path)
    v3_manifest = read_json(v3_path)
    v2_false = {
        str(name)
        for name, passed in dict(v2_manifest.get("quality_checks") or {}).items()
        if not passed
    }
    v3_false = {
        str(name)
        for name, passed in dict(v3_manifest.get("quality_checks") or {}).items()
        if not passed
    }
    common_hashes = (
        sha256_file(model_path),
        sha256_file(baseline_manifest_path),
        sha256_file(legacy_manifest_path),
    )
    if not (
        v2_manifest.get("protocol") == v3.V3_PRIOR_PROTOCOL
        and v2_manifest.get("quality_gate") == "FAIL"
        and v2_false == {V4_EXPECTED_PRIOR_FALSE_CHECK}
        and v2_manifest.get("release_status")
        == "BLOCKED_FIXED_V2_BUDGET_DID_NOT_RECOVER_3ZGC"
        and int(v2_manifest.get("conditional_rounds_completed", -1)) == 6
        and int(v2_manifest.get("released_candidates", -1)) == 0
        and v2_manifest.get("missing_targets_after_search") == ["3ZGC"]
        and (
            v2_manifest.get("model_sha256"),
            v2_manifest.get("baseline_manifest_sha256"),
            v2_manifest.get("legacy_manifest_sha256"),
        )
        == common_hashes
    ):
        raise RuntimeError("Prior V2 is not the exact hash-pinned zero-release failure")
    if not (
        v3_manifest.get("protocol") == v3.V3_SEARCH_PROTOCOL
        and v3_manifest.get("quality_gate") == "FAIL"
        and v3_false == {V4_EXPECTED_PRIOR_FALSE_CHECK}
        and v3_manifest.get("release_status")
        == "BLOCKED_FIXED_V3_FULL_FRONTIER_BUDGET_DID_NOT_RECOVER_3ZGC"
        and int(v3_manifest.get("conditional_rounds_completed", -1)) == 6
        and int(v3_manifest.get("legacy_strict_hits_reaudited", -1)) == 2881
        and int(v3_manifest.get("legacy_full_frontier_rows", -1)) == 268365
        and int(v3_manifest.get("prior_v2_methyl_screen_rows_reused", -1))
        == 159329
        and int(v3_manifest.get("legacy_non_strict_bridge_rows_exactly_scored", -1))
        == 16384
        and int(v3_manifest.get("released_candidates", -1)) == 0
        and v3_manifest.get("missing_targets_after_search") == ["3ZGC"]
        and (
            v3_manifest.get("model_sha256"),
            v3_manifest.get("baseline_manifest_sha256"),
            v3_manifest.get("legacy_manifest_sha256"),
        )
        == common_hashes
        and dict(v3_manifest.get("config") or {}).get("prior_v2_manifest_sha256")
        == sha256_file(v2_path)
        and abs(float(v3_manifest.get("cyclic_base_floor_1pct", float("nan")))
                - float(v2_manifest.get("cyclic_base_floor_1pct", float("nan"))))
        <= 2e-6
    ):
        raise RuntimeError("Prior V3 is not the exact hash-pinned zero-release failure")
    validate_artifacts_under(v2_manifest, prior_v2_dir)
    validate_artifacts_under(v3_manifest, prior_v3_dir)
    return v2_manifest, v3_manifest


def normalize_methyl_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    sequence = str(result.get("sequence", "")).upper()
    probability = float(
        result.get(
            "maximum_probability",
            result.get(
                "qualified_full_maximum_probability",
                result.get(
                    "physical_argmax_probability",
                    result.get("search_maximum_probability", float("nan")),
                ),
            ),
        )
    )
    position = int(
        result.get(
            "argmax_position_1based",
            result.get("physical_argmax_position_1based", -1),
        )
    )
    residue = str(
        result.get(
            "argmax_residue",
            result.get("physical_argmax_residue", ""),
        )
    )
    strict = int(
        result.get(
            "passes_strict_probability",
            int(round(probability, 8) > THRESHOLD)
            if math.isfinite(probability)
            else -1,
        )
    )
    if not (
        len(sequence) == 7
        and set(sequence) <= set(NATURAL_AA)
        and math.isfinite(probability)
        and 0.0 <= probability <= 1.0
        and 1 <= position <= 7
        and residue == sequence[position - 1]
        and strict == int(round(probability, 8) > THRESHOLD)
    ):
        raise RuntimeError(f"Malformed prior methyl row: {sequence}")
    result.update(
        {
            "sequence": sequence,
            "maximum_probability": probability,
            "argmax_position_1based": position,
            "argmax_residue": residue,
            "passes_strict_probability": strict,
        }
    )
    return result


def normalize_exact_row(row: Mapping[str, Any], base_policy: str) -> Dict[str, Any]:
    result = dict(row)
    sequence = str(result.get("sequence", result.get("design_natural_seq", ""))).upper()
    base = float(result.get("cyclic_base_log_probability_mean", float("nan")))
    if not (
        len(sequence) == 7
        and set(sequence) <= set(NATURAL_AA)
        and math.isfinite(base)
        and str(result.get("cyclic_base_context_policy", "")) == base_policy
    ):
        raise RuntimeError(f"Malformed prior exact cyclic-base row: {sequence}")
    result["sequence"] = sequence
    for key in (
        "cyclic_base_log_probability_mean",
        "cyclic_base_log_probability_min",
        "cyclic_base_log_probability_max",
        "cyclic_base_log_probability_span",
        "cyclic_base_log_probability_std",
    ):
        result[key] = float(result[key])
    return result


def merge_methyl_rows(
    destination: MutableMapping[str, Dict[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    source: str,
) -> None:
    for raw in rows:
        row = normalize_methyl_row(raw)
        sequence = str(row["sequence"])
        row.setdefault("v4_evidence_source", source)
        prior = destination.get(sequence)
        if prior is not None:
            if not (
                abs(float(prior["maximum_probability"])
                    - float(row["maximum_probability"])) <= RESCORE_TOLERANCE
                and int(prior["argmax_position_1based"])
                == int(row["argmax_position_1based"])
                and str(prior["argmax_residue"]) == str(row["argmax_residue"])
            ):
                raise RuntimeError(f"Conflicting prior methyl score: {sequence}")
            continue
        destination[sequence] = row


def merge_exact_rows(
    destination: MutableMapping[str, Dict[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    source: str,
    base_policy: str,
) -> None:
    for raw in rows:
        row = normalize_exact_row(raw, base_policy)
        sequence = str(row["sequence"])
        row.setdefault("v4_exact_source", source)
        prior = destination.get(sequence)
        if prior is not None:
            if abs(
                float(prior["cyclic_base_log_probability_mean"])
                - float(row["cyclic_base_log_probability_mean"])
            ) > RESCORE_TOLERANCE:
                raise RuntimeError(f"Conflicting prior cyclic-base score: {sequence}")
            continue
        destination[sequence] = row


def load_prior_inventories(
    *,
    v2_manifest: Mapping[str, Any],
    v3_manifest: Mapping[str, Any],
    prior_v3_dir: Path,
    legacy_rows: Sequence[Mapping[str, Any]],
    base_policy: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    methyl: Dict[str, Dict[str, Any]] = {}
    exact: Dict[str, Dict[str, Any]] = {}
    merge_methyl_rows(methyl, legacy_rows, "legacy_full_268365")
    v2_artifacts = dict(v2_manifest["artifacts"])
    for path in declared_paths(dict(v2_artifacts["conditional_methyl_screens"])):
        merge_methyl_rows(methyl, read_gzip_csv(path), "prior_v2_methyl_screen")
    for path in declared_paths(dict(v2_artifacts["conditional_cyclic_base_shortlists"])):
        rows = read_gzip_csv(path)
        merge_methyl_rows(methyl, rows, "prior_v2_exact_shortlist")
        merge_exact_rows(exact, rows, "prior_v2_exact_shortlist", base_policy)

    v3_artifacts = dict(v3_manifest["artifacts"])
    for path in declared_paths(dict(v3_artifacts["conditional_methyl_screens"])):
        merge_methyl_rows(methyl, read_gzip_csv(path), "prior_v3_methyl_screen")
    for path in declared_paths(dict(v3_artifacts["conditional_cyclic_base_shortlists"])):
        rows = read_gzip_csv(path)
        merge_methyl_rows(methyl, rows, "prior_v3_exact_shortlist")
        merge_exact_rows(exact, rows, "prior_v3_exact_shortlist", base_policy)

    baseline_rows = read_csv(prior_v3_dir / "baseline_cyclic_start_plausibility.csv")
    legacy_strict = read_csv(prior_v3_dir / "legacy_strict_hit_cyclic_reaudit.csv")
    bridge = read_gzip_csv(prior_v3_dir / "pre_v3_full_frontier_cyclic_base.csv.gz")
    merge_exact_rows(exact, baseline_rows, "v3_baseline_996", base_policy)
    merge_methyl_rows(methyl, legacy_strict, "v3_legacy_strict_reaudit")
    merge_exact_rows(exact, legacy_strict, "v3_legacy_strict_reaudit", base_policy)
    merge_methyl_rows(methyl, bridge, "v3_exact_legacy_bridge")
    merge_exact_rows(exact, bridge, "v3_exact_legacy_bridge", base_policy)
    return methyl, exact


def logit(probability: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def exact_joint_rows(
    methyl: Mapping[str, Mapping[str, Any]],
    exact: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sequence in sorted(set(methyl) & set(exact)):
        rows.append({**methyl[sequence], **exact[sequence], "sequence": sequence})
    return rows


def joint_deficit(row: Mapping[str, Any], floor: float) -> float:
    methyl_deficit = max(0.0, THRESHOLD - float(row["maximum_probability"])) / 0.10
    base_deficit = max(
        0.0, float(floor) - float(row["cyclic_base_log_probability_mean"])
    ) / 0.50
    return max(methyl_deficit, base_deficit)


def pareto_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row["maximum_probability"]),
            -float(row["cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    result: List[Dict[str, Any]] = []
    best_base = float("-inf")
    for row in ordered:
        base = float(row["cyclic_base_log_probability_mean"])
        if base > best_base:
            result.append(row)
            best_base = base
    return result


def residue_palettes(
    rows: Sequence[Mapping[str, Any]],
    width: int = 6,
    fallback_rows: Sequence[Mapping[str, Any]] = (),
) -> List[List[str]]:
    palettes: List[List[str]] = []
    for position in range(7):
        counts = Counter(str(row["sequence"])[position] for row in rows)
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        fallback_counts = Counter(
            str(row["sequence"])[position] for row in fallback_rows
        )
        for token in sorted(
            fallback_counts, key=lambda value: (-fallback_counts[value], value)
        ):
            if token not in ordered:
                ordered.append(token)
        for token in NATURAL_AA:
            if token not in ordered:
                ordered.append(token)
        palettes.append(ordered[:width])
    return palettes


def generate_candidate_pool(
    rows: Sequence[Mapping[str, Any]], seen: set[str], floor: float
) -> Dict[str, Dict[str, Any]]:
    """Construct a deterministic local/crossover lattice around both gates."""

    strict = sorted(
        [row for row in rows if int(row["passes_strict_probability"]) == 1],
        key=lambda row: (
            -float(row["cyclic_base_log_probability_mean"]),
            -float(row["maximum_probability"]),
            str(row["sequence"]),
        ),
    )
    base_pass = sorted(
        [row for row in rows if float(row["cyclic_base_log_probability_mean"]) >= floor],
        key=lambda row: (
            -float(row["maximum_probability"]),
            -float(row["cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    boundary = sorted(
        rows,
        key=lambda row: (
            joint_deficit(row, floor),
            -float(row["maximum_probability"]),
            -float(row["cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    pareto = pareto_rows(rows)
    if not strict or not base_pass:
        raise RuntimeError("V4 requires both observed single-gate frontiers")

    anchors: Dict[str, Dict[str, Any]] = {}
    for label, group, limit in (
        ("strict_anchor", strict, 192),
        ("base_anchor", base_pass, 192),
        ("boundary_anchor", boundary, 256),
        ("pareto_anchor", pareto, 128),
    ):
        for row in group[:limit]:
            anchors.setdefault(str(row["sequence"]), {**row, "anchor_source": label})

    pool: Dict[str, Dict[str, Any]] = {}

    def add(sequence: str, origin: str, parent: str = "") -> None:
        sequence = str(sequence).upper()
        if sequence in seen or sequence in anchors or len(sequence) != 7:
            return
        if set(sequence) > set(NATURAL_AA):
            return
        row = pool.setdefault(
            sequence,
            {
                "sequence": sequence,
                "candidate_generation_origins": [],
                "representative_parent": parent,
            },
        )
        origins = row["candidate_generation_origins"]
        if origin not in origins:
            origins.append(origin)

    # Complete Hamming-1 neighborhoods of the most informative anchors.
    local_anchors = list(anchors.values())[:512]
    for anchor in local_anchors:
        parent = str(anchor["sequence"])
        for position in range(7):
            for token in NATURAL_AA:
                if token != parent[position]:
                    add(
                        parent[:position] + token + parent[position + 1 :],
                        "complete_hamming1",
                        parent,
                    )

    # Mutations restricted to residue palettes learned from the opposite gate.
    palette_rows = [*strict[:128], *base_pass[:128], *boundary[:256], *pareto[:128]]
    # Six residues per position make a bounded 6^7 lattice.  Fallback filling
    # from the complete exact inventory prevents a collapsed one-motif lattice
    # after the prior V2/V3 seen set removes local Hamming neighborhoods.
    palettes = residue_palettes(palette_rows, width=6, fallback_rows=rows)
    for anchor in [*strict[:128], *boundary[:128]]:
        parent = str(anchor["sequence"])
        alternatives = [
            [token for token in palettes[position] if token != parent[position]][:4]
            for position in range(7)
        ]
        for position in range(7):
            for token in alternatives[position]:
                child = list(parent)
                child[position] = token
                add("".join(child), "palette_hamming1", parent)
        for left, right in itertools.combinations(range(7), 2):
            for token_left in alternatives[left]:
                for token_right in alternatives[right]:
                    child = list(parent)
                    child[left] = token_left
                    child[right] = token_right
                    add("".join(child), "palette_hamming2", parent)

    # Enumerate every path vertex between the strongest methyl and base anchors.
    crossover_right = list(
        {str(row["sequence"]): row for row in [*base_pass[:48], *boundary[:48]]}.values()
    )
    for left_row in strict[:32]:
        left = str(left_row["sequence"])
        for right_row in crossover_right:
            right = str(right_row["sequence"])
            differences = [index for index in range(7) if left[index] != right[index]]
            if not differences:
                continue
            for mask in range(1, (1 << len(differences)) - 1):
                child = list(left)
                for bit, position in enumerate(differences):
                    if mask & (1 << bit):
                        child[position] = right[position]
                add("".join(child), "strict_base_crossover", left)

    # A compact positional lattice finds combinations not lying on a single pair path.
    for tokens in itertools.product(*palettes):
        add("".join(tokens), "frontier_residue_lattice")

    for row in pool.values():
        row["candidate_generation_origins"] = json.dumps(
            sorted(row["candidate_generation_origins"]), separators=(",", ":")
        )
    return pool


def select_screen_rows(
    rows: Sequence[Mapping[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    if len(rows) < limit or limit <= 0:
        raise RuntimeError("V4 generated pool is smaller than the frozen methyl budget")
    values = [dict(row) for row in rows]
    joint = sorted(
        values,
        key=lambda row: (
            max(
                max(0.0, THRESHOLD - float(row["predicted_methyl_probability"]))
                / 0.10,
                max(
                    0.0,
                    float(row["cyclic_base_floor"])
                    - float(row["predicted_cyclic_base_log_probability_mean"]),
                )
                / 0.50,
            ),
            -float(row["predicted_methyl_probability"]),
            -float(row["predicted_cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    methyl = sorted(
        values,
        key=lambda row: (
            -float(row["predicted_methyl_probability"]),
            -float(row["predicted_cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    methyl_bridge = sorted(
        [row for row in values if float(row["predicted_methyl_probability"]) >= 0.40],
        key=lambda row: (
            -float(row["predicted_cyclic_base_log_probability_mean"]),
            -float(row["predicted_methyl_probability"]),
            str(row["sequence"]),
        ),
    )
    base_bridge = sorted(
        [
            row
            for row in values
            if float(row["predicted_cyclic_base_log_probability_mean"])
            >= float(row["cyclic_base_floor"]) - 0.50
        ],
        key=lambda row: (
            -float(row["predicted_methyl_probability"]),
            -float(row["predicted_cyclic_base_log_probability_mean"]),
            str(row["sequence"]),
        ),
    )
    predicted_pareto_scored = pareto_rows(
        [
            {
                **row,
                "maximum_probability": row["predicted_methyl_probability"],
                "cyclic_base_log_probability_mean": row[
                    "predicted_cyclic_base_log_probability_mean"
                ],
            }
            for row in values
        ]
    )
    original_by_sequence = {str(row["sequence"]): row for row in values}
    predicted_pareto = [
        original_by_sequence[str(row["sequence"])]
        for row in predicted_pareto_scored
    ]
    orders = [joint, methyl, methyl_bridge, base_bridge, predicted_pareto]
    selected: Dict[str, Dict[str, Any]] = {}
    cursors = [0] * len(orders)
    while len(selected) < limit:
        progressed = False
        for index, order in enumerate(orders):
            while cursors[index] < len(order):
                row = order[cursors[index]]
                cursors[index] += 1
                sequence = str(row["sequence"])
                if sequence in selected:
                    continue
                selected[sequence] = row
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    if len(selected) != limit:
        raise RuntimeError("V4 methyl-screen selection did not fill its frozen budget")
    return list(selected.values())


def select_strict_exact_rows(
    rows: Sequence[Mapping[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    strict = [dict(row) for row in rows if int(row["passes_strict_probability"]) == 1]
    if len(strict) <= limit:
        return sorted(strict, key=lambda row: str(row["sequence"]))
    orders = [
        sorted(
            strict,
            key=lambda row: (
                -float(row["predicted_cyclic_base_log_probability_mean"]),
                -float(row["maximum_probability"]),
                str(row["sequence"]),
            ),
        ),
        sorted(
            strict,
            key=lambda row: (
                -float(row["maximum_probability"]),
                -float(row["predicted_cyclic_base_log_probability_mean"]),
                str(row["sequence"]),
            ),
        ),
    ]
    selected: Dict[str, Dict[str, Any]] = {}
    cursors = [0, 0]
    while len(selected) < limit:
        for index, order in enumerate(orders):
            while str(order[cursors[index]]["sequence"]) in selected:
                cursors[index] += 1
            row = order[cursors[index]]
            cursors[index] += 1
            selected[str(row["sequence"])] = row
            if len(selected) >= limit:
                break
    return list(selected.values())


def build_runtime(
    *,
    model_path: Path,
    native_path: Path,
    baseline_target_rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    base_batch_size: int,
    device_name: str,
    allow_cpu: bool,
    old: Any,
    v2: Any,
) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("V4 requires PyTorch") from exc
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif device_name == "cpu":
        if not allow_cpu:
            raise RuntimeError("CPU V4 requires --allow-cpu")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("No CUDA device is available")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    reannotator = old.load_module("v8_reannotator_for_v4", old.REANNOTATOR_PATH)
    generator = old.load_module("v8_generator_for_v4", old.GENERATOR_PATH)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from paper_clean_v28.clean_v28_common import (  # pylint: disable=import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        NAT_TO_METHYL_ABS,
        NATURAL_AA_ALPHABET,
        complete_decoding_order,
        cyclic_representation_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
    )

    selected_chains = {
        str(row["target_name"]).upper(): str(row["selected_chain"])
        for row in baseline_target_rows
    }
    native_rows = generator.read_jsonl(native_path)
    native_index = {
        generator.record_name(row, index): row for index, row in enumerate(native_rows)
    }
    model = load_v28_model(str(model_path), device)
    model.eval()
    common = {
        "EXTENDED_AA_ALPHABET": EXTENDED_AA_ALPHABET,
        "NAT_TO_METHYL_ABS": NAT_TO_METHYL_ABS,
        "NATURAL_AA_ALPHABET": NATURAL_AA_ALPHABET,
        "complete_decoding_order": complete_decoding_order,
        "cyclic_representation_known_sequence_methyl_probabilities": (
            cyclic_representation_known_sequence_methyl_probabilities
        ),
        "featurize_records": featurize_records,
    }
    target_records, _ = generator.prepare_target_records(
        native_rows, selected_chains, ["3ZGC"]
    )
    methyl = old.MethylScorer(
        model,
        device,
        native_index,
        selected_chains,
        int(batch_size),
        torch,
        common,
        reannotator,
    )
    batch_one = old.MethylScorer(
        model,
        device,
        native_index,
        selected_chains,
        1,
        torch,
        common,
        reannotator,
    )
    base = v2.CyclicBasePlausibilityScorer(
        model,
        device,
        target_records,
        int(base_batch_size),
        torch,
        functional,
        common,
        old.ProgressBar,
    )
    return {
        "torch": torch,
        "device": device,
        "methyl": methyl,
        "batch_one": batch_one,
        "base": base,
        "selected_chains": selected_chains,
    }


def stable_methyl_review_rows(
    *,
    ranked: Sequence[Mapping[str, Any]],
    limit: int,
    floor: float,
    old: Any,
    v2: Any,
    methyl_scorer: Any,
    batch_one_scorer: Any,
    novelty_sets: Mapping[str, set[str]],
) -> List[Dict[str, Any]]:
    """Return only independently replayed methyl-hard-gate/base-fail rows."""

    selected: List[Dict[str, Any]] = []
    accepted_cyclic: set[str] = set()
    for raw in ranked:
        if len(selected) >= limit:
            break
        row = dict(raw)
        sequence = str(row["sequence"])
        base = float(row["cyclic_base_log_probability_mean"])
        if (
            int(row["passes_strict_probability"]) != 1
            or not old.strict_rounded_pass(float(row["maximum_probability"]))
            or base >= floor
            or v2.duplicate_reason(old, sequence, novelty_sets)
        ):
            continue
        cyclic = old.forward_cyclic_identity(sequence)
        if cyclic in accepted_cyclic:
            continue
        full = methyl_scorer.score_full(
            "3ZGC", [sequence], stage="V4 methylated near-miss full annotation"
        )[sequence]
        replay = batch_one_scorer.score_full(
            "3ZGC",
            [sequence],
            stage="V4 methylated near-miss independent batch-one",
            show_progress=False,
        )[sequence]
        full_values = [float(value) for value in json.loads(str(full["methyl_probabilities"]))]
        replay_values = [
            float(value) for value in json.loads(str(replay["methyl_probabilities"]))
        ]
        full_point = v2.physical_argmax_summary(sequence, full_values)
        replay_point = v2.physical_argmax_summary(sequence, replay_values)
        full_max = float(full_point["physical_argmax_probability"])
        replay_max = float(replay_point["physical_argmax_probability"])
        full_release_floor_max = old.release_floor_actionable_max(full, sequence)
        replay_release_floor_max = old.release_floor_actionable_max(replay, sequence)
        methyl_positions = [
            index
            for index, token in enumerate(str(replay["design_seq"]), start=1)
            if token.islower()
        ]
        stable = (
            old.stable_cyclic_methyl_release_gate(full, sequence)
            and old.stable_cyclic_methyl_release_gate(replay, sequence)
            and bool(methyl_positions)
            and abs(full_max - replay_max) <= RESCORE_TOLERANCE
            and abs(full_release_floor_max - replay_release_floor_max)
            <= RESCORE_TOLERANCE
            and int(full_point["physical_argmax_position_1based"])
            == int(replay_point["physical_argmax_position_1based"])
            and str(full_point["physical_argmax_residue"])
            == str(replay_point["physical_argmax_residue"])
            and int(replay_point["physical_argmax_position_1based"]) in methyl_positions
        )
        if not stable:
            continue
        accepted_cyclic.add(cyclic)
        selected.append(
            {
                "candidate_id": f"v8v4_3zgc_methyl_nearmiss_{len(selected)+1:02d}",
                "target_name": "3ZGC",
                "design_seq": replay["design_seq"],
                "design_natural_seq": sequence,
                "predicted_methyl_positions_1based": replay["methyl_positions_1based"],
                "methyl_positions_1based": replay["methyl_positions_1based"],
                "design_methyl_count": replay["design_methyl_count"],
                "methyl_threshold": THRESHOLD,
                "strict_threshold_operator": ">",
                "batch_one_maximum_probability": replay_max,
                "batch_one_release_floor_maximum_probability": (
                    replay_release_floor_max
                ),
                "batch_one_argmax_position_1based": replay_point[
                    "physical_argmax_position_1based"
                ],
                "batch_one_argmax_residue": replay_point["physical_argmax_residue"],
                "batch_rescore_absolute_difference": abs(full_max - replay_max),
                "batch_one_release_floor_rescore_absolute_difference": abs(
                    full_release_floor_max - replay_release_floor_max
                ),
                "passes_methylation_hard_gate": 1,
                "cyclic_base_log_probability_mean": base,
                "cyclic_base_floor_1pct": floor,
                "base_gap_to_floor": floor - base,
                "passes_cyclic_base_hard_gate": 0,
                "passes_joint_hard_gate": 0,
                "review_class": "METHYL_HARD_GATE_PASS_BASE_GATE_FAIL",
                "advisor_review_status": "REVIEW_ONLY_NOT_FULLY_QUALIFIED",
                "released_candidate": 0,
                "candidate_origin": row.get("v4_exact_source", "prior_exact_inventory"),
                "forward_cyclic_identity": cyclic,
                "methylation_claim_scope": "FROZEN_V8_MODEL_PREDICTION_NOT_EXPERIMENTAL_CONFIRMATION",
                "methyl_probabilities": replay["methyl_probabilities"],
                "methyl_probability_representation_min": replay[
                    "methyl_probability_representation_min"
                ],
                "methyl_probability_representation_max": replay[
                    "methyl_probability_representation_max"
                ],
                "methyl_probability_representation_span": replay[
                    "methyl_probability_representation_span"
                ],
                "methyl_probability_representation_span_max": replay[
                    "methyl_probability_representation_span_max"
                ],
                "methyl_probability_representation_std": replay[
                    "methyl_probability_representation_std"
                ],
                "methyl_probability_representation_std_max": replay[
                    "methyl_probability_representation_std_max"
                ],
                "representation_threshold_disagreement_positions_1based": replay[
                    "representation_threshold_disagreement_positions_1based"
                ],
                "representation_threshold_disagreement_count": replay[
                    "representation_threshold_disagreement_count"
                ],
                "stable_cyclic_release_gate": 1,
                "annotation_release_probability_policy": (
                    "representation_min_strict_gt_threshold_zero_disagreement"
                ),
            }
        )
    return selected


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("V4 requires NumPy and PyTorch") from exc
    old = load_module("v8_legacy_search_for_v4", LEGACY_SEARCH_PATH)
    v2 = load_module("v8_cyclic_base_for_v4", V2_SEARCH_PATH)
    v3 = load_module("v8_full_frontier_for_v4", V3_HELPER_PATH)

    model_path = Path(args.model_path).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    representation_path = Path(args.representation_audit).resolve()
    baseline = Path(args.baseline_run_dir).resolve()
    legacy_dir = Path(args.legacy_search_dir).resolve()
    prior_v2_dir = Path(args.prior_v2_dir).resolve()
    prior_v3_dir = Path(args.prior_v3_dir).resolve()
    plan_path = Path(args.plan).resolve()
    native_path = Path(args.native_jsonl).resolve()
    historical_path = Path(args.historical_designs_csv).resolve()
    prior_handoff_path = Path(args.prior_handoff_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    immutable = (
        model_path,
        model_manifest_path,
        representation_path,
        baseline,
        legacy_dir,
        prior_v2_dir,
        prior_v3_dir,
        plan_path,
        native_path,
        historical_path,
        prior_handoff_path,
        SCRIPT_PATH,
        LEGACY_SEARCH_PATH,
        V2_SEARCH_PATH,
        V3_HELPER_PATH,
    )
    if any(old.paths_overlap(out_dir, path) for path in immutable):
        raise ValueError("V4 output overlaps an immutable input")
    for required in (
        model_path,
        model_manifest_path,
        representation_path,
        plan_path,
        native_path,
        historical_path,
        prior_handoff_path,
        legacy_dir / "directed_search_manifest.json",
        baseline / "generation_manifest.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    baseline_manifest, baseline_unique, baseline_target_rows = old.validate_baseline(
        baseline,
        model_path,
        model_manifest_path,
        representation_path,
        plan_path,
        native_path,
        historical_path,
        prior_handoff_path,
    )
    if baseline_manifest.get("targets_without_signature_candidate") != ["3ZGC"]:
        raise RuntimeError("V4 is frozen for the sole missing target 3ZGC")
    v2_manifest, v3_manifest = validate_prior_failures(
        prior_v2_dir=prior_v2_dir,
        prior_v3_dir=prior_v3_dir,
        model_path=model_path,
        baseline_manifest_path=baseline / "generation_manifest.json",
        legacy_manifest_path=legacy_dir / "directed_search_manifest.json",
        v3=v3,
    )
    floor = float(v3_manifest["cyclic_base_floor_1pct"])
    base_policy = str(dict(v3_manifest["config"])["base_policy"])
    if not (
        float(dict(v3_manifest["config"])["threshold"]) == THRESHOLD
        and float(dict(v3_manifest["config"])["base_percentile"]) == BASE_PERCENTILE
        and base_policy == v2.V2_BASE_POLICY
    ):
        raise RuntimeError("V4 refuses a changed methyl/base hard gate")

    legacy_manifest, legacy_seen, _legacy_qualified, legacy_rows = (
        v2.validate_and_reconstruct_legacy(
            old, legacy_dir, model_path, baseline, baseline_unique, np
        )
    )
    methyl_inventory, exact_inventory = load_prior_inventories(
        v2_manifest=v2_manifest,
        v3_manifest=v3_manifest,
        prior_v3_dir=prior_v3_dir,
        legacy_rows=legacy_rows,
        base_policy=base_policy,
    )
    joined = exact_joint_rows(methyl_inventory, exact_inventory)
    if any(
        int(row["passes_strict_probability"]) == 1
        and float(row["cyclic_base_log_probability_mean"]) >= floor
        for row in joined
    ):
        raise RuntimeError("Prior V3 claimed zero joint hits but exact inventory has one")

    v2_screen_seen = {
        str(row["sequence"])
        for path in declared_paths(
            dict(dict(v2_manifest["artifacts"])["conditional_methyl_screens"])
        )
        for row in read_gzip_csv(path)
    }
    v3_screen_seen = {
        str(row["sequence"])
        for path in declared_paths(
            dict(dict(v3_manifest["artifacts"])["conditional_methyl_screens"])
        )
        for row in read_gzip_csv(path)
    }
    baseline_sequences = {
        str(row["design_natural_seq"]).upper()
        for row in baseline_unique
        if str(row["target_name"]).upper() == "3ZGC"
    }
    seen = set(legacy_seen) | v2_screen_seen | v3_screen_seen | baseline_sequences

    base_surrogate = v3.KmerBaseSurrogate(NATURAL_AA, 7)
    base_report = base_surrogate.fit(list(exact_inventory.values()))
    methyl_surrogate_rows = [
        {
            "sequence": sequence,
            "cyclic_base_log_probability_mean": logit(
                float(row["maximum_probability"])
            ),
        }
        for sequence, row in methyl_inventory.items()
    ]
    methyl_surrogate = v3.KmerBaseSurrogate(NATURAL_AA, 7)
    methyl_report = methyl_surrogate.fit(methyl_surrogate_rows)
    pool = generate_candidate_pool(joined, seen, floor)
    pool_sequences = sorted(pool)
    predicted_base = base_surrogate.predict(pool_sequences)
    predicted_methyl_logit = methyl_surrogate.predict(pool_sequences)
    predicted_rows = [
        {
            **pool[sequence],
            "predicted_methyl_probability": sigmoid(predicted_methyl_logit[sequence]),
            "predicted_methyl_logit": predicted_methyl_logit[sequence],
            "predicted_cyclic_base_log_probability_mean": predicted_base[sequence],
            "cyclic_base_floor": floor,
            "surrogate_release_authority": "NONE_ACQUISITION_ONLY",
        }
        for sequence in pool_sequences
    ]
    selected = select_screen_rows(predicted_rows, int(args.methyl_screen_budget))

    config = {
        "protocol": V4_PROTOCOL,
        "model_sha256": sha256_file(model_path),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "baseline_manifest_sha256": sha256_file(baseline / "generation_manifest.json"),
        "legacy_manifest_sha256": sha256_file(legacy_dir / "directed_search_manifest.json"),
        "prior_v2_manifest_sha256": sha256_file(
            prior_v2_dir / "cyclic_base_recovery_manifest.json"
        ),
        "prior_v3_manifest_sha256": sha256_file(
            prior_v3_dir / "cyclic_base_recovery_manifest.json"
        ),
        "legacy_search_program_sha256": sha256_file(LEGACY_SEARCH_PATH),
        "v2_program_sha256": sha256_file(V2_SEARCH_PATH),
        "v3_program_sha256": sha256_file(V3_HELPER_PATH),
        "v4_program_sha256": sha256_file(SCRIPT_PATH),
        "threshold": THRESHOLD,
        "strict_operator": ">",
        "base_floor": floor,
        "base_percentile": BASE_PERCENTILE,
        "base_policy": base_policy,
        "methyl_screen_budget": int(args.methyl_screen_budget),
        "exact_base_budget": int(args.exact_base_budget),
        "advisor_near_miss_limit": int(args.advisor_near_miss_limit),
        "maximum_release": int(args.max_release),
        "acquisition_protocol": V4_ACQUISITION_PROTOCOL,
        "surrogate_release_authority": "NONE",
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "requested_device": str(args.device),
        "cuda_device_name": (
            str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
        ),
        "deterministic_algorithms": True,
    }
    config_sha256 = stable_json_sha256(config)
    manifest_path = out_dir / "methyl_first_v4_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("config_sha256") != config_sha256:
            raise RuntimeError("Existing V4 output belongs to a different configuration")
        if existing.get("execution_audit_gate") == "PASS":
            validate_artifacts_under(existing, out_dir)
            print("V4: reused hash-valid completed result", flush=True)
            return
    elif out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError("V4 output exists; pass --resume after inspection")
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_path = out_dir / "v4_methyl_screen_selection.csv.gz"
    screen_path = out_dir / "v4_methyl_screen.csv.gz"
    exact_path = out_dir / "v4_strict_methyl_exact_cyclic_base.csv.gz"
    state_path = out_dir / "v4_resume_state.json"
    selected_sequences = [str(row["sequence"]) for row in selected]
    selection_sha = hashlib.sha256(
        ("\n".join(selected_sequences) + "\n").encode("ascii")
    ).hexdigest()
    persisted_selection = [
        {**row, "selection_rank": index}
        for index, row in enumerate(selected, start=1)
    ]
    v2.atomic_write_gzip_csv(
        selection_path, persisted_selection, list(persisted_selection[0])
    )

    runtime = build_runtime(
        model_path=model_path,
        native_path=native_path,
        baseline_target_rows=baseline_target_rows,
        batch_size=int(args.batch_size),
        base_batch_size=int(args.base_batch_size),
        device_name=str(args.device),
        allow_cpu=bool(args.allow_cpu),
        old=old,
        v2=v2,
    )
    reusable_screen = False
    state: Dict[str, Any] = {}
    if args.resume and state_path.is_file():
        state = read_json(state_path)
        reusable_screen = (
            state.get("config_sha256") == config_sha256
            and state.get("selection_sequence_sha256") == selection_sha
            and screen_path.is_file()
            and state.get("methyl_screen_sha256") == sha256_file(screen_path)
            and state.get("phase") in {"methyl_screen_complete", "exact_base_complete"}
        )
    if reusable_screen:
        screen_rows = [normalize_methyl_row(row) for row in read_gzip_csv(screen_path)]
        if [str(row["sequence"]) for row in screen_rows] != selected_sequences:
            raise RuntimeError("Reusable V4 methyl screen sequence/order changed")
        print(f"V4: reused {len(screen_rows):,} hash-pinned methyl scores", flush=True)
    else:
        observed = runtime["methyl"].score_minimal(
            "3ZGC", selected_sequences, "V4 final methyl-first screen"
        )
        by_sequence = {str(row["sequence"]): row for row in selected}
        screen_rows = [
            {**by_sequence[sequence], **observed[sequence]}
            for sequence in selected_sequences
        ]
        v2.atomic_write_gzip_csv(screen_path, screen_rows, list(screen_rows[0]))
        state = {
            "protocol": V4_PROTOCOL,
            "config_sha256": config_sha256,
            "phase": "methyl_screen_complete",
            "selection_sequence_sha256": selection_sha,
            "selection_sha256": sha256_file(selection_path),
            "methyl_screen_sha256": sha256_file(screen_path),
            "methyl_screen_rows": len(screen_rows),
            "strict_methyl_rows": sum(
                int(row["passes_strict_probability"]) for row in screen_rows
            ),
        }
        v2.atomic_write_json(state_path, state)

    strict_shortlist = select_strict_exact_rows(
        screen_rows, int(args.exact_base_budget)
    )
    strict_sequences = [str(row["sequence"]) for row in strict_shortlist]
    reusable_exact = (
        bool(strict_sequences)
        and reusable_screen
        and state.get("phase") == "exact_base_complete"
        and exact_path.is_file()
        and state.get("exact_base_sha256") == sha256_file(exact_path)
        and state.get("exact_sequence_sha256")
        == hashlib.sha256(("\n".join(strict_sequences) + "\n").encode("ascii")).hexdigest()
    )
    new_exact_rows: List[Dict[str, Any]] = []
    if strict_sequences and reusable_exact:
        new_exact_rows = [
            normalize_exact_row(row, base_policy) for row in read_gzip_csv(exact_path)
        ]
        if [str(row["sequence"]) for row in new_exact_rows] != strict_sequences:
            raise RuntimeError("Reusable V4 exact-score sequence/order changed")
        print(f"V4: reused {len(new_exact_rows):,} strict exact base scores", flush=True)
    elif strict_sequences:
        detailed = runtime["base"].score_detailed(
            "3ZGC", strict_sequences, "V4 strict-methyl exact cyclic base"
        )
        strict_by_sequence = {str(row["sequence"]): row for row in strict_shortlist}
        new_exact_rows = [
            {**strict_by_sequence[sequence], **detailed[sequence]}
            for sequence in strict_sequences
        ]
        v2.atomic_write_gzip_csv(exact_path, new_exact_rows, list(new_exact_rows[0]))
        state.update(
            {
                "phase": "exact_base_complete",
                "exact_base_sha256": sha256_file(exact_path),
                "exact_sequence_sha256": hashlib.sha256(
                    ("\n".join(strict_sequences) + "\n").encode("ascii")
                ).hexdigest(),
                "strict_exact_rows": len(new_exact_rows),
            }
        )
        v2.atomic_write_json(state_path, state)
    else:
        v2.atomic_write_gzip_csv(
            exact_path,
            [],
            [
                "sequence",
                "maximum_probability",
                "passes_strict_probability",
                "cyclic_base_log_probability_mean",
            ],
        )

    for row in new_exact_rows:
        sequence = str(row["sequence"])
        methyl_inventory[sequence] = normalize_methyl_row(row)
        exact_inventory[sequence] = normalize_exact_row(row, base_policy)
    all_joined = exact_joint_rows(methyl_inventory, exact_inventory)
    novelty = v2.exclusion_sets(
        old,
        read_csv(historical_path),
        read_csv(prior_handoff_path),
        baseline_unique,
        "3ZGC",
    )
    joint_rows = [
        row
        for row in all_joined
        if int(row["passes_strict_probability"]) == 1
        and float(row["cyclic_base_log_probability_mean"]) >= floor
        and not v2.duplicate_reason(old, str(row["sequence"]), novelty)
    ]
    joint_candidates = {
        str(row["sequence"]): {**row, "search_stage": "V4 methyl-first joint search"}
        for row in joint_rows
    }
    joint_base = {
        sequence: {
            key: value
            for key, value in row.items()
            if key.startswith("cyclic_base_")
        }
        for sequence, row in joint_candidates.items()
    }
    joint_full = (
        runtime["methyl"].score_full(
            "3ZGC", sorted(joint_candidates), stage="V4 joint full annotation"
        )
        if joint_candidates
        else {}
    )
    joint_evidence, releases = (
        v2.evaluate_candidates(
            old=old,
            target="3ZGC",
            candidates=joint_candidates,
            base_scores=joint_base,
            floor=floor,
            full_payload=joint_full,
            novelty_sets=novelty,
            batch_one_scorer=runtime["batch_one"],
            selected_chain=runtime["selected_chains"]["3ZGC"],
            max_release=int(args.max_release),
            id_prefix="v8v4_3zgc_joint",
        )
        if joint_candidates
        else ([], [])
    )
    for row in releases:
        row["candidate_origin"] = "V8_V4_METHYL_FIRST_JOINT_RECOVERY"
        row["methylation_claim_scope"] = (
            "FROZEN_V8_MODEL_PREDICTION_NOT_EXPERIMENTAL_CONFIRMATION"
        )

    strict_ranked = sorted(
        [
            row
            for row in all_joined
            if int(row["passes_strict_probability"]) == 1
            and float(row["cyclic_base_log_probability_mean"]) < floor
        ],
        key=lambda row: (
            floor - float(row["cyclic_base_log_probability_mean"]),
            -float(row["maximum_probability"]),
            str(row["sequence"]),
        ),
    )
    near_misses = (
        stable_methyl_review_rows(
            ranked=strict_ranked,
            limit=int(args.advisor_near_miss_limit),
            floor=floor,
            old=old,
            v2=v2,
            methyl_scorer=runtime["methyl"],
            batch_one_scorer=runtime["batch_one"],
            novelty_sets=novelty,
        )
        if not releases
        else []
    )

    release_path = out_dir / "released_joint_candidates.csv"
    near_miss_path = out_dir / "methylated_base_near_miss_for_shangge_review.csv"
    joint_evidence_path = out_dir / "joint_candidate_evidence.csv"
    surrogate_path = out_dir / "v4_surrogate_and_selection_audit.json"
    v2.atomic_write_csv(
        release_path,
        releases,
        list(releases[0])
        if releases
        else ["candidate_id", "target_name", "design_seq", "design_natural_seq"],
    )
    v2.atomic_write_csv(
        near_miss_path,
        near_misses,
        list(near_misses[0])
        if near_misses
        else [
            "candidate_id",
            "target_name",
            "design_seq",
            "design_natural_seq",
            "passes_methylation_hard_gate",
            "passes_cyclic_base_hard_gate",
        ],
    )
    v2.atomic_write_csv(
        joint_evidence_path,
        joint_evidence,
        list(joint_evidence[0])
        if joint_evidence
        else ["target_name", "sequence", "release_eligible"],
    )
    v2.atomic_write_json(
        surrogate_path,
        {
            "protocol": V4_ACQUISITION_PROTOCOL,
            "base_surrogate": base_report,
            "methyl_logit_surrogate": methyl_report,
            "surrogate_release_authority": "NONE",
            "prior_unique_methyl_rows": len(methyl_inventory),
            "prior_unique_exact_base_rows": len(exact_inventory) - len(new_exact_rows),
            "generated_pool_rows": len(pool),
            "selected_methyl_screen_rows": len(selected),
            "selected_sequence_sha256": selection_sha,
            "new_strict_methyl_rows": len(strict_shortlist),
            "new_exact_base_rows": len(new_exact_rows),
        },
    )

    execution_checks = {
        "prior_v2_and_v3_zero_release_failures_are_hash_pinned": True,
        "methyl_threshold_is_unchanged_strictly_greater_than_0_6": True,
        "cyclic_base_floor_and_policy_are_unchanged": True,
        "all_prior_expensive_scores_are_reused_not_recomputed": True,
        "v4_methyl_screen_uses_the_complete_fixed_budget": (
            len(selected) == int(args.methyl_screen_budget)
            and len(screen_rows) == int(args.methyl_screen_budget)
        ),
        "v4_selected_sequences_are_new_to_all_prior_searches": not (
            set(selected_sequences) & seen
        ),
        "exact_base_is_applied_only_to_strict_methyl_hits": all(
            int(row["passes_strict_probability"]) == 1 for row in new_exact_rows
        ),
        "exact_base_budget_is_not_exceeded": len(new_exact_rows)
        <= int(args.exact_base_budget),
        "surrogates_have_no_release_authority": True,
        "every_release_passes_both_hard_gates": all(
            int(row["passes_methylation_hard_gate"]) == 1
            and old.stable_cyclic_methyl_release_gate(
                row, str(row["design_natural_seq"])
            )
            and float(row["base_log_probability_mean_all_orders"]) >= floor
            and any(token.islower() for token in str(row["design_seq"]))
            for row in releases
        ),
        "every_advisor_review_row_is_methylated_and_base_fail_only": all(
            int(row["passes_methylation_hard_gate"]) == 1
            and int(row["passes_cyclic_base_hard_gate"]) == 0
            and int(row["passes_joint_hard_gate"]) == 0
            and int(row["design_methyl_count"]) > 0
            and any(token.islower() for token in str(row["design_seq"]))
            and old.stable_cyclic_methyl_release_gate(
                row, str(row["design_natural_seq"])
            )
            and float(
                row["batch_one_release_floor_rescore_absolute_difference"]
            )
            <= RESCORE_TOLERANCE
            and float(row["cyclic_base_log_probability_mean"]) < floor
            for row in near_misses
        ),
        "zero_joint_outcome_has_nonempty_methylated_review_fallback": bool(releases)
        or bool(near_misses),
        "no_non_methylated_candidate_can_enter_advisor_review": all(
            int(row["design_methyl_count"]) > 0
            and bool(json.loads(str(row["predicted_methyl_positions_1based"])))
            for row in near_misses
        ),
    }
    if not all(execution_checks.values()):
        failed = [name for name, passed in execution_checks.items() if not passed]
        raise RuntimeError("V4 execution/integrity audit failed: " + ", ".join(failed))

    scientific_gate = "PASS" if releases else "FAIL"
    workflow_status = (
        "COMPLETE_JOINT_RECOVERY_READY_FOR_INDEPENDENT_AUDIT"
        if releases
        else "COMPLETE_ZERO_JOINT_HITS_METHYLATED_NEAR_MISS_REVIEW_ONLY"
    )
    manifest = {
        "quality_gate": scientific_gate,
        "scientific_joint_gate": scientific_gate,
        "execution_audit_gate": "PASS",
        "workflow_completion_status": workflow_status,
        "protocol": V4_PROTOCOL,
        "config": config,
        "config_sha256": config_sha256,
        "methylation_claim_scope": (
            "MODEL_PREDICTED_N_METHYLATION_UNDER_FROZEN_V8_NOT_EXPERIMENTAL_PROOF"
        ),
        "frozen_methyl_threshold": THRESHOLD,
        "strict_threshold_operator": ">",
        "frozen_cyclic_base_floor_1pct": floor,
        "prior_seen_sequences": len(seen),
        "generated_pool_sequences": len(pool),
        "new_methyl_screen_rows": len(screen_rows),
        "new_strict_methyl_hits": sum(
            int(row["passes_strict_probability"]) for row in screen_rows
        ),
        "new_exact_base_rows": len(new_exact_rows),
        "released_joint_candidates": len(releases),
        "methylated_base_near_miss_review_rows": len(near_misses),
        "formal_scientific_interpretation": (
            "AT_LEAST_ONE_3ZGC_CANDIDATE_PASSED_BOTH_FROZEN_HARD_GATES"
            if releases
            else "NO_3ZGC_CANDIDATE_PASSED_BOTH_FROZEN_HARD_GATES_AFTER_V4; "
            "SEPARATE REVIEW TABLE CONTAINS ONLY METHYL-HARD-GATE-PASS BASE NEAR MISSES"
        ),
        "execution_checks": execution_checks,
        "artifacts": {
            "selection": artifact(selection_path),
            "methyl_screen": artifact(screen_path),
            "strict_exact_base": artifact(exact_path),
            "resume_state": artifact(state_path),
            "surrogate_audit": artifact(surrogate_path),
            "joint_evidence": artifact(joint_evidence_path),
            "released_joint_candidates": artifact(release_path),
            "methylated_base_near_miss_review": artifact(near_miss_path),
        },
        "prior_evidence": {
            "v2_manifest": artifact(
                prior_v2_dir / "cyclic_base_recovery_manifest.json"
            ),
            "v3_manifest": artifact(
                prior_v3_dir / "cyclic_base_recovery_manifest.json"
            ),
            "legacy_manifest": artifact(
                legacy_dir / "directed_search_manifest.json"
            ),
        },
    }
    v2.atomic_write_json(manifest_path, manifest)
    print("===== V8 METHYL-FIRST JOINT RECOVERY V4 COMPLETE =====", flush=True)
    print("Execution audit gate: PASS", flush=True)
    print(f"Scientific joint gate: {scientific_gate}", flush=True)
    print(f"New methyl screen: {len(screen_rows):,}", flush=True)
    print(
        "New strict methyl hits: "
        f"{sum(int(row['passes_strict_probability']) for row in screen_rows):,}",
        flush=True,
    )
    print(f"Released joint candidates: {len(releases)}", flush=True)
    print(f"Methylated base-near-miss review rows: {len(near_misses)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-manifest", default=str(DEFAULT_MODEL_MANIFEST))
    parser.add_argument("--representation-audit", default=str(DEFAULT_REPRESENTATION))
    parser.add_argument("--baseline-run-dir", default=str(DEFAULT_BASELINE))
    parser.add_argument("--legacy-search-dir", default=str(DEFAULT_LEGACY))
    parser.add_argument("--prior-v2-dir", default=str(DEFAULT_PRIOR_V2))
    parser.add_argument("--prior-v3-dir", default=str(DEFAULT_PRIOR_V3))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--historical-designs-csv", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--prior-handoff-csv", default=str(DEFAULT_PRIOR_HANDOFF))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-batch-size", type=int, default=32)
    parser.add_argument("--methyl-screen-budget", type=int, default=METHYL_SCREEN_BUDGET)
    parser.add_argument("--exact-base-budget", type=int, default=EXACT_BASE_BUDGET)
    parser.add_argument(
        "--advisor-near-miss-limit", type=int, default=ADVISOR_NEAR_MISS_LIMIT
    )
    parser.add_argument("--max-release", type=int, default=MAX_RELEASE)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = {
        "--batch-size": (args.batch_size, 64),
        "--base-batch-size": (args.base_batch_size, 32),
        "--methyl-screen-budget": (args.methyl_screen_budget, METHYL_SCREEN_BUDGET),
        "--exact-base-budget": (args.exact_base_budget, EXACT_BASE_BUDGET),
        "--advisor-near-miss-limit": (
            args.advisor_near_miss_limit,
            ADVISOR_NEAR_MISS_LIMIT,
        ),
        "--max-release": (args.max_release, MAX_RELEASE),
    }
    changed = [name for name, (value, expected) in frozen.items() if value != expected]
    if changed:
        raise ValueError("V8 V4 numerical protocol is frozen: " + ", ".join(changed))
    run(args)


if __name__ == "__main__":
    main()
