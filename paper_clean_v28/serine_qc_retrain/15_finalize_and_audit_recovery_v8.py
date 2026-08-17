#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Finalize and independently audit the immutable V8 recovery overlay.

The 31,500-row baseline directory is never rewritten.  This stage combines its
already-novel eligible rows with independently re-scored directed candidates in
a new review-only directory, then performs integrity, annotation/position, and
novelty/workflow passes.  It does not create a structure handoff or permeability
input and it does not claim that a sequence-level signature has passed RMSD.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
SEARCH_PATH = SCRIPT_PATH.with_name("14_directed_recovery_search_v8.py")
V7_AUDITOR_PATH = SCRIPT_PATH.with_name("11_triple_audit_serine_only_v7.py")
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"
DEFAULT_MODEL = V8_ROOT / "model" / "frankenstein_v28_source_scoped_hybrid_v8.pt"
DEFAULT_MODEL_MANIFEST = V8_ROOT / "model" / "expert_source_composition_manifest.json"
DEFAULT_REPRESENTATION = V8_ROOT / "representation_audit" / "cyclic_representation_audit.json"
DEFAULT_BASELINE = V8_ROOT / "generation_baseline"
DEFAULT_SEARCH = V8_ROOT / "directed_search"
DEFAULT_OUT = V8_ROOT / "generation_recovered"
DEFAULT_AUDIT_OUT = V8_ROOT / "triple_audit_recovered"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_representation_v6.json")
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_HISTORICAL = (
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

V8_EXPERT_PROTOCOL = (
    "canonical_shared_v6_non_ser_v7_ser_cyclic_representation_hybrid_v8"
)
V8_REPRESENTATION_PROTOCOL = (
    "cyclic_representation_frozen_audit_source_scoped_hybrid_v8"
)
V8_REPRESENTATION_AUTHORIZATION = (
    "SOURCE_SCOPED_HYBRID_V8_AUTHORIZED_FOR_DIRECTED_RECOVERY"
)
V8_SEARCH_PROTOCOL = "deterministic_missing_target_directed_recovery_v8"
V8_FINAL_PROTOCOL = "immutable_baseline_plus_directed_recovery_overlay_v8"
V8_AUDIT_PROTOCOL = "independent_three_pass_source_scoped_recovery_v8"
V8_MODEL_ARTIFACT_FILENAMES = {
    "metric_comparison": "v6_v7_v8_metric_comparison.csv",
    "serine_auc_tradeoff_audit": "serine_auc_tradeoff_audit.csv",
    "metrics_by_residue": "test_metrics_by_residue.csv",
    "position_probabilities": "test_position_probabilities.csv",
}
EXPECTED_BASELINE_ROWS = 31_500
EXPECTED_TARGETS = 17
THRESHOLD = 0.6
TEMPERATURE = 0.5
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
METHYLATABLE_AA = set(NATURAL_AA) - {"P"}
AV_FAMILY = {
    "3AV9",
    "3AVA",
    "3AVB",
    "3AVF",
    "3AVG",
    "3AVH",
    "3AVI",
    "3AVJ",
    "3AVK",
    "3AVM",
    "3AVN",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path is equal to or contains the other."""

    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def record_name(record: Mapping[str, Any], fallback: int) -> str:
    return str(
        record.get("name")
        or record.get("pdb")
        or record.get("pdb_id")
        or record.get("id")
        or f"record_{fallback}"
    ).upper()


def artifact_leaves(value: Any) -> List[Mapping[str, Any]]:
    leaves: List[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            leaves.append(value)
        else:
            for child in value.values():
                leaves.extend(artifact_leaves(child))
    elif isinstance(value, list):
        for child in value:
            leaves.extend(artifact_leaves(child))
    return leaves


def artifacts_are_hash_pinned_under(value: Any, root: Path) -> bool:
    leaves = artifact_leaves(value)
    if not leaves:
        return False
    for leaf in leaves:
        path = Path(str(leaf["path"])).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            return False
        if not path.is_file() or sha256_file(path) != str(leaf["sha256"]):
            return False
    return True


def artifact_matches_exact_path(value: Any, expected_path: Path) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        return False
    declared = Path(str(value["path"])).resolve()
    expected = expected_path.resolve()
    return (
        declared == expected
        and expected.is_file()
        and sha256_file(expected) == str(value["sha256"])
    )


def artifact_map_matches_exact_paths(
    artifacts: Any, expected: Mapping[str, Path]
) -> bool:
    return (
        isinstance(artifacts, Mapping)
        and set(artifacts) == set(expected)
        and all(
            artifact_matches_exact_path(artifacts[name], path)
            for name, path in expected.items()
        )
    )


def declared_path_hash_is_current(
    payload: Mapping[str, Any],
    path_field: str,
    hash_field: str,
    expected_path: Any = None,
) -> bool:
    declared = Path(str(payload.get(path_field, ""))).resolve()
    try:
        declared.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False
    return (
        declared.is_file()
        and (expected_path is None or declared == Path(expected_path).resolve())
        and sha256_file(declared) == str(payload.get(hash_field, ""))
    )


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                result.append(field)
    return result


def parse_json_list(value: object, field: str, row_id: str) -> List[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{row_id}: invalid {field}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{row_id}: {field} is not a list")
    return parsed


def natural_sequence(row: Mapping[str, Any]) -> str:
    return str(row.get("design_natural_seq") or row.get("design_seq") or "").upper()


def strict_rounded_pass(value: float, threshold: float = THRESHOLD) -> bool:
    numeric = float(value)
    return (
        math.isfinite(numeric)
        and 0.0 <= numeric <= 1.0
        and round(numeric, 8) > float(threshold)
    )


def methyl_positions(sequence: str) -> List[int]:
    return [index for index, token in enumerate(sequence, start=1) if token.islower()]


def exact_and_natural_keys(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]]]:
    exact: set[Tuple[str, str]] = set()
    natural: set[Tuple[str, str]] = set()
    for row in rows:
        target = str(row.get("target_name", "")).upper()
        design = str(row.get("design_seq", ""))
        sequence = natural_sequence(row)
        if target and design:
            exact.add((target, design))
        if target and sequence:
            natural.add((target, sequence))
    return exact, natural


def canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {str(key): str(value) for key, value in sorted(row.items())}
        for row in sorted(
            rows,
            key=lambda row: (
                str(row.get("target_name", "")),
                str(row.get("candidate_id", "")),
                str(row.get("design_seq", "")),
            ),
        )
    ]
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def project_preserved_source_fields(
    row: Mapping[str, Any], source_fields: Iterable[str]
) -> Dict[str, Any]:
    """Project an overlay row back to the immutable baseline field values."""

    projected = {field: row.get(field, "") for field in source_fields}
    eligibility = "eligible_for_new_permeability_screen"
    source_eligibility = "source_eligible_for_new_permeability_screen"
    if eligibility in projected and source_eligibility in row:
        projected[eligibility] = row.get(source_eligibility, "")
    return projected


def validate_annotation_row(row: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    row_id = str(row.get("candidate_id", ""))
    sequence = str(row.get("design_seq", ""))
    natural = natural_sequence(row)
    try:
        probabilities = [
            float(value)
            for value in parse_json_list(
                row.get("methyl_probabilities"), "methyl_probabilities", row_id
            )
        ]
        positions = [
            int(value)
            for value in parse_json_list(
                row.get("methyl_positions_1based"), "methyl_positions_1based", row_id
            )
        ]
        minima = [
            float(value)
            for value in parse_json_list(
                row.get("methyl_probability_representation_min"),
                "representation_min",
                row_id,
            )
        ]
        maxima = [
            float(value)
            for value in parse_json_list(
                row.get("methyl_probability_representation_max"),
                "representation_max",
                row_id,
            )
        ]
        spans = [
            float(value)
            for value in parse_json_list(
                row.get("methyl_probability_representation_span"),
                "representation_span",
                row_id,
            )
        ]
        order_std = [
            float(value)
            for value in parse_json_list(
                row.get("methyl_probability_order_std"),
                "order_std",
                row_id,
            )
        ]
        representation_std = [
            float(value)
            for value in parse_json_list(
                row.get("methyl_probability_representation_std"),
                "representation_std",
                row_id,
            )
        ]
        methyl_count = int(row.get("design_methyl_count", -1))
        methyl_rate = float(row.get("design_methyl_rate", "nan"))
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    if not row_id or not sequence or natural != sequence.upper():
        errors.append(f"{row_id}: ID/sequence/natural mismatch")
    if not sequence:
        return errors
    if not set(natural) <= set(NATURAL_AA) or "p" in sequence:
        errors.append(f"{row_id}: invalid token or forbidden methyl-Pro")
    if positions != methyl_positions(sequence):
        errors.append(f"{row_id}: methyl position mismatch")
    if methyl_count != len(positions):
        errors.append(f"{row_id}: methyl count mismatch")
    vectors = (
        probabilities,
        minima,
        maxima,
        spans,
        order_std,
        representation_std,
    )
    if any(len(values) != len(sequence) for values in vectors):
        errors.append(f"{row_id}: vector length mismatch")
        return errors
    if not all(math.isfinite(value) for values in vectors for value in values):
        errors.append(f"{row_id}: non-finite annotation vector")
        return errors
    if any(
        value < 0.0 or value > 1.0
        for values in (probabilities, minima, maxima)
        for value in values
    ) or any(
        value < 0.0
        for values in (spans, order_std, representation_std)
        for value in values
    ):
        errors.append(f"{row_id}: annotation vector outside its valid range")
    if any(value != round(value, 8) for value in probabilities):
        errors.append(f"{row_id}: methyl probability was not persisted at 8 decimals")
    if not math.isfinite(methyl_rate) or abs(
        methyl_rate - (len(positions) / len(sequence) if sequence else 0.0)
    ) > 1e-12:
        errors.append(f"{row_id}: methyl rate mismatch")
    for index, (token, probability) in enumerate(zip(sequence, probabilities), start=1):
        expected = token.upper() in METHYLATABLE_AA and strict_rounded_pass(probability)
        if token.islower() != expected:
            errors.append(f"{row_id}: strict threshold mismatch at {index}")
    if any(
        minimum > mean + 1e-7
        or mean > maximum + 1e-7
        or abs((maximum - minimum) - span) > 2e-6
        for mean, minimum, maximum, span in zip(probabilities, minima, maxima, spans)
    ):
        errors.append(f"{row_id}: representation min/mean/max/span conflict")
    summary_expectations = {
        "methyl_probability_min": min(probabilities),
        "methyl_probability_mean": sum(probabilities) / len(probabilities),
        "methyl_probability_max": max(probabilities),
        "methyl_probability_order_std_max": max(order_std),
        "methyl_probability_representation_std_max": max(representation_std),
        "methyl_probability_representation_span_max": max(spans),
    }
    for field, expected in summary_expectations.items():
        try:
            observed = float(row.get(field, "nan"))
        except (TypeError, ValueError):
            observed = float("nan")
        if not math.isfinite(observed) or abs(observed - expected) > 2e-6:
            errors.append(f"{row_id}: {field} summary mismatch")
    site_probabilities = [probabilities[index - 1] for index in positions]
    site_fields = (
        "methyl_site_probability_min",
        "methyl_site_probability_mean",
        "methyl_site_probability_max",
    )
    if site_probabilities:
        site_expected = (
            min(site_probabilities),
            sum(site_probabilities) / len(site_probabilities),
            max(site_probabilities),
        )
        for field, expected in zip(site_fields, site_expected):
            try:
                observed = float(row.get(field, "nan"))
            except (TypeError, ValueError):
                observed = float("nan")
            if not math.isfinite(observed) or abs(observed - expected) > 2e-6:
                errors.append(f"{row_id}: {field} summary mismatch")
    elif any(str(row.get(field, "")).strip() for field in site_fields):
        errors.append(f"{row_id}: empty methyl-site summary mismatch")
    if (
        int(row.get("annotation_decoder_order_ensemble_size", -1)) != len(sequence)
        or int(row.get("annotation_representation_ensemble_size", -1)) != len(sequence)
    ):
        errors.append(f"{row_id}: ensemble size mismatch")
    return errors


def validate_eligible_candidate_row(
    row: Mapping[str, Any], expected_permeability_flag: int
) -> List[str]:
    row_id = str(row.get("candidate_id", ""))
    errors: List[str] = []
    try:
        natural = natural_sequence(row)
        probabilities = [
            float(value)
            for value in parse_json_list(
                row.get("methyl_probabilities"),
                "methyl_probabilities",
                row_id,
            )
        ]
        actionable = [
            probability
            for token, probability in zip(natural, probabilities)
            if token in METHYLATABLE_AA
        ]
        hard_gate = (
            len(probabilities) == len(natural)
            and all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in probabilities
            )
            and int(row.get("design_methyl_count", 0)) > 0
            and bool(methyl_positions(str(row.get("design_seq", ""))))
            and bool(actionable)
            and any(strict_rounded_pass(value) for value in actionable)
            and int(row.get("passes_methylation_hard_gate", 0)) == 1
            and int(row.get("eligible_for_new_permeability_screen", -1))
            == int(expected_permeability_flag)
            and int(row.get("valid_token_gate", 0)) == 1
            and int(row.get("length_match", 0)) == 1
            and int(row.get("occurrence_count", 0)) >= 1
            and all(
                int(row.get(field, 1)) == 0
                for field in (
                    "seen_in_historical_4115_exact",
                    "seen_in_historical_4115_naturalized",
                    "seen_in_historical_4115",
                    "seen_in_prior_1333_exact",
                    "seen_in_prior_1333_naturalized",
                    "seen_in_prior_1333",
                )
            )
        )
    except (TypeError, ValueError):
        hard_gate = False
    if not hard_gate:
        errors.append(f"{row_id}: eligible-candidate hard gate/novelty mismatch")
    return errors


def concentration_audit(
    final_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    native_rows: Sequence[Mapping[str, Any]],
    selected_chains: Mapping[str, str],
    v7_auditor: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, bool]]:
    positions_by_target: MutableMapping[str, Counter[int]] = defaultdict(Counter)
    residues_by_target: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    positions_all: Counter[int] = Counter()
    residues_all: Counter[str] = Counter()
    for row in final_rows:
        target = str(row["target_name"]).upper()
        sequence = str(row["design_seq"])
        for position in methyl_positions(sequence):
            residue = sequence[position - 1].upper()
            positions_by_target[target][position] += 1
            residues_by_target[target][residue] += 1
            positions_all[position] += 1
            residues_all[residue] += 1

    sampling_steps_by_target: MutableMapping[str, Counter[int]] = defaultdict(Counter)
    sampling_steps_all: Counter[int] = Counter()
    for row in baseline_rows:
        order_value = str(row.get("decoding_order_absolute", ""))
        if not order_value:
            continue
        order = [int(value) for value in json.loads(order_value)]
        sorted_order = sorted(order)
        target = str(row["target_name"]).upper()
        for position in methyl_positions(str(row["design_seq"])):
            absolute = sorted_order[position - 1]
            step = order.index(absolute) + 1
            sampling_steps_by_target[target][step] += 1
            sampling_steps_all[step] += 1

    targets = sorted({str(row["target_name"]).upper() for row in final_rows})
    concentration_rows: List[Dict[str, Any]] = []
    for target in ["ALL", *targets]:
        positions = positions_all if target == "ALL" else positions_by_target[target]
        residues = residues_all if target == "ALL" else residues_by_target[target]
        steps = sampling_steps_all if target == "ALL" else sampling_steps_by_target[target]
        total = sum(positions.values())
        concentration_rows.append(
            {
                "target_name": target,
                "methyl_sites": total,
                "site_position_counts": json.dumps(dict(sorted(positions.items()))),
                "site_residue_counts": json.dumps(dict(sorted(residues.items()))),
                "baseline_sampling_step_counts": json.dumps(dict(sorted(steps.items()))),
                "dominant_position_1based": (
                    max(positions, key=positions.get) if positions else ""
                ),
                "maximum_single_position_share": (
                    max(positions.values()) / total if total else 0.0
                ),
                "maximum_single_residue_share": (
                    max(residues.values()) / total if total else 0.0
                ),
                "maximum_single_baseline_sampling_step_share": (
                    max(steps.values()) / sum(steps.values()) if steps else 0.0
                ),
            }
        )

    dominant = {
        target: max(positions_by_target[target], key=positions_by_target[target].get)
        for target in sorted(AV_FAMILY)
        if positions_by_target[target]
    }
    complete_map = set(dominant) == AV_FAMILY
    universal = complete_map and len(set(dominant.values())) == 1
    if universal:
        av_alignment = v7_auditor.av_family_alignment_audit(
            native_rows, selected_chains, dominant
        )
        av_alignment["dominant_position_by_target"] = dominant
    else:
        av_alignment = {
            "quality_gate": "PASS" if complete_map else "FAIL",
            "method": "complete_dominant_position_map_then_conditional_geometry_alignment",
            "reason": (
                "No universal absolute position collapse; complete physical-position map retained."
                if complete_map
                else "One or more 3AV targets have no dominant methyl position."
            ),
            "dominant_position_by_target": dominant,
            "targets": [],
        }
    global_row = next(row for row in concentration_rows if row["target_name"] == "ALL")
    checks = {
        "no_global_position_collapse_above_80_percent": (
            float(global_row["maximum_single_position_share"]) <= 0.80
        ),
        "no_global_residue_collapse_above_80_percent": (
            float(global_row["maximum_single_residue_share"]) <= 0.80
        ),
        "no_baseline_sampling_step_collapse_above_80_percent": all(
            float(row["maximum_single_baseline_sampling_step_share"]) <= 0.80
            for row in concentration_rows
            if int(row["methyl_sites"]) >= 30
        ),
        "all_3av_targets_have_a_dominant_physical_position": complete_map,
        "universal_3av_position_is_geometry_supported_when_present": (
            av_alignment["quality_gate"] == "PASS"
        ),
    }
    return concentration_rows, av_alignment, checks


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("V8 final audit requires NumPy and PyTorch") from exc
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = Path(args.model_path).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    representation_path = Path(args.representation_audit).resolve()
    baseline = Path(args.baseline_run_dir).resolve()
    search_dir = Path(args.search_dir).resolve()
    plan_path = Path(args.plan).resolve()
    native_path = Path(args.native_jsonl).resolve()
    historical_path = Path(args.historical_designs_csv).resolve()
    prior_path = Path(args.prior_handoff_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    audit_out = Path(args.audit_out_dir).resolve()
    immutable_inputs = (
        model_path,
        model_manifest_path,
        representation_path,
        baseline,
        search_dir,
        plan_path,
        native_path,
        historical_path,
        prior_path,
        SCRIPT_PATH,
        SEARCH_PATH,
        V7_AUDITOR_PATH,
        REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py",
        REPO_ROOT / "model_utils.py",
        REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py",
    )
    for label, writable in (("recovered generation", out_dir), ("recovered audit", audit_out)):
        overlapping = [path for path in immutable_inputs if paths_overlap(writable, path)]
        if overlapping:
            raise ValueError(
                f"{label} output overlaps an immutable input: "
                + ", ".join(str(path) for path in overlapping)
            )
    if paths_overlap(out_dir, audit_out):
        raise ValueError("Recovered generation and audit outputs overlap")
    required = (
        model_path,
        model_manifest_path,
        representation_path,
        baseline / "all_candidates.csv",
        baseline / "unique_candidates.csv",
        baseline / "methylated_new_candidates.csv",
        baseline / "target_manifest.csv",
        baseline / "generation_summary_by_target.csv",
        baseline / "generation_manifest.json",
        search_dir / "directed_candidates.csv",
        search_dir / "directed_search_manifest.json",
        search_dir / "mandatory_length_6_7_controls.csv",
        plan_path,
        native_path,
        historical_path,
        prior_path,
        SEARCH_PATH,
        V7_AUDITOR_PATH,
        REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py",
        REPO_ROOT / "model_utils.py",
        REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (out_dir.exists() and any(out_dir.iterdir())) or (
        audit_out.exists() and any(audit_out.iterdir())
    ):
        if not args.overwrite:
            raise FileExistsError("Recovered generation/audit output already exists")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_out.mkdir(parents=True, exist_ok=True)

    model_manifest = read_json(model_manifest_path)
    representation = read_json(representation_path)
    baseline_manifest = read_json(baseline / "generation_manifest.json")
    search_manifest = read_json(search_dir / "directed_search_manifest.json")
    declared_model_sources = {
        "canonical": (
            Path(str(model_manifest.get("canonical_checkpoint", ""))),
            str(model_manifest.get("canonical_checkpoint_sha256", "")),
        ),
        "v6_checkpoint": (
            Path(str(model_manifest.get("v6_checkpoint", ""))),
            str(model_manifest.get("v6_checkpoint_sha256", "")),
        ),
        "v6_manifest": (
            Path(str(model_manifest.get("v6_manifest", ""))),
            str(model_manifest.get("v6_manifest_sha256", "")),
        ),
        "v7_checkpoint": (
            Path(str(model_manifest.get("v7_checkpoint", ""))),
            str(model_manifest.get("v7_checkpoint_sha256", "")),
        ),
        "v7_manifest": (
            Path(str(model_manifest.get("v7_manifest", ""))),
            str(model_manifest.get("v7_manifest_sha256", "")),
        ),
    }
    model_sources_are_current = True
    for source_path, expected_hash in declared_model_sources.values():
        resolved_source = source_path.resolve()
        try:
            resolved_source.relative_to(REPO_ROOT.resolve())
        except ValueError:
            model_sources_are_current = False
            break
        if not resolved_source.is_file() or sha256_file(resolved_source) != expected_hash:
            model_sources_are_current = False
            break
    model_artifacts = dict(model_manifest.get("artifacts") or {})
    if not (
        model_manifest.get("quality_gate") == "PASS"
        and model_manifest.get("protocol") == V8_EXPERT_PROTOCOL
        and int(model_manifest.get("audit_batch_size", -1)) == 8
        and model_manifest.get("checkpoint_artifact_sha256") == sha256_file(model_path)
        and model_sources_are_current
        and model_manifest.get("composer_program_sha256")
        == sha256_file(SEARCH_PATH.with_name("12_compose_source_scoped_hybrid_v8.py"))
        and model_manifest.get("trainer_program_sha256")
        == sha256_file(SEARCH_PATH.with_name("02_retrain_canonical_expert_heads.py"))
        and model_manifest.get("common_program_sha256")
        == sha256_file(REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py")
        and model_manifest.get("model_utils_program_sha256")
        == sha256_file(REPO_ROOT / "model_utils.py")
        and model_manifest.get("nmethyl_config_program_sha256")
        == sha256_file(REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py")
        and artifact_map_matches_exact_paths(
            model_artifacts,
            {
                name: model_manifest_path.parent / filename
                for name, filename in V8_MODEL_ARTIFACT_FILENAMES.items()
            },
        )
    ):
        raise RuntimeError("V8 model manifest failed or is stale")
    representation_artifacts = dict(representation.get("artifacts") or {})
    representation_best_path = Path(str(representation.get("best_csv", ""))).resolve()
    try:
        representation_best_path.relative_to(REPO_ROOT.resolve())
        representation_best_is_current = (
            representation_best_path.is_file()
            and sha256_file(representation_best_path)
            == str(representation.get("best_csv_sha256", ""))
        )
    except ValueError:
        representation_best_is_current = False
    if not (
        representation.get("quality_gate") == "PASS"
        and representation.get("protocol") == V8_REPRESENTATION_PROTOCOL
        and representation.get("release_authorization")
        == V8_REPRESENTATION_AUTHORIZATION
        and int(representation.get("audit_batch_size", -1)) == 8
        and representation.get("model_sha256") == sha256_file(model_path)
        and representation.get("model_manifest_sha256")
        == sha256_file(model_manifest_path)
        and representation.get("representation_auditor_program_sha256")
        == sha256_file(SEARCH_PATH.with_name("13_audit_source_scoped_hybrid_v8.py"))
        and representation.get("equivariance_auditor_program_sha256")
        == sha256_file(
            SEARCH_PATH.with_name("07_audit_cyclic_representation_equivariance.py")
        )
        and representation.get("common_program_sha256")
        == sha256_file(REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py")
        and representation.get("model_utils_program_sha256")
        == sha256_file(REPO_ROOT / "model_utils.py")
        and representation.get("nmethyl_config_program_sha256")
        == sha256_file(REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py")
        and declared_path_hash_is_current(
            representation, "test_jsonl", "test_jsonl_sha256"
        )
        and declared_path_hash_is_current(
            representation,
            "native_jsonl",
            "native_jsonl_sha256",
            native_path,
        )
        and declared_path_hash_is_current(
            representation, "best_csv", "best_csv_sha256"
        )
        and declared_path_hash_is_current(
            representation, "plan", "plan_sha256", plan_path
        )
        and artifact_map_matches_exact_paths(
            representation_artifacts,
            {
                "frozen_test_positions": representation_path.parent
                / "frozen_test_position_probabilities.csv",
                "length_metrics": representation_path.parent
                / "frozen_test_metrics_by_length.csv",
                "native_probabilities": representation_path.parent
                / "native_target_representation_probabilities.csv",
                "native_summary": representation_path.parent
                / "native_target_representation_summary.csv",
            },
        )
        and representation_best_is_current
    ):
        raise RuntimeError("V8 representation audit failed or is stale")
    if not (
        search_manifest.get("quality_gate") == "PASS"
        and search_manifest.get("protocol") == V8_SEARCH_PROTOCOL
        and search_manifest.get("model_sha256") == sha256_file(model_path)
        and search_manifest.get("baseline_manifest_sha256")
        == sha256_file(baseline / "generation_manifest.json")
    ):
        raise RuntimeError("V8 directed search failed or is stale")
    baseline_missing = {
        str(value).upper()
        for value in baseline_manifest.get("targets_without_signature_candidate", [])
    }
    baseline_false_checks = {
        name
        for name, passed in dict(baseline_manifest.get("quality_checks") or {}).items()
        if not passed
    }
    coverage_check = "every_target_has_at_least_one_novel_methylated_signature_candidate"
    baseline_artifacts = dict(baseline_manifest.get("candidate_artifacts") or {})
    if not (
        baseline_manifest.get("protocol")
        == "temperature_0.5_source_scoped_hybrid_v8_reannotation_of_preserved_v6_pool"
        and baseline_manifest.get("model_sha256") == sha256_file(model_path)
        and float(baseline_manifest.get("temperature", -1.0)) == TEMPERATURE
        and float(baseline_manifest.get("methyl_threshold", -1.0)) == THRESHOLD
        and baseline_manifest.get("strict_threshold_operator") == ">"
        and int(baseline_manifest.get("scoring_batch_size", -1)) == 8
        and baseline_manifest.get("summary_score_label") == "v8"
        and baseline_manifest.get("expert_scope")
        == "residue-source-scoped-hybrid"
        and baseline_manifest.get("model_expert_qc_protocol") == V8_EXPERT_PROTOCOL
        and dict(
            baseline_manifest.get("cyclic_representation_heldout_audit") or {}
        ).get("sha256")
        == sha256_file(representation_path)
        and baseline_manifest.get("reannotator_program_sha256")
        == sha256_file(SEARCH_PATH.with_name("10_reannotate_v6_pool_serine_only_v7.py"))
        and baseline_manifest.get("generator_program_sha256")
        == sha256_file(
            REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
        )
        and baseline_manifest.get("common_program_sha256")
        == sha256_file(REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py")
        and baseline_manifest.get("model_utils_program_sha256")
        == sha256_file(REPO_ROOT / "model_utils.py")
        and baseline_manifest.get("nmethyl_config_program_sha256")
        == sha256_file(REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py")
        and baseline_manifest.get("plan_sha256") == sha256_file(plan_path)
        and baseline_manifest.get("native_jsonl_sha256") == sha256_file(native_path)
        and baseline_manifest.get("historical_design_csv_sha256")
        == sha256_file(historical_path)
        and baseline_manifest.get("prior_handoff_csv_sha256")
        == sha256_file(prior_path)
        and artifact_map_matches_exact_paths(
            baseline_artifacts,
            {
                "all": baseline / "all_candidates.csv",
                "unique": baseline / "unique_candidates.csv",
                "eligible": baseline / "methylated_new_candidates.csv",
                "target_manifest": baseline / "target_manifest.csv",
                "target_summary": baseline / "generation_summary_by_target.csv",
            },
        )
        and (
            (
                not baseline_missing
                and baseline_manifest.get("quality_gate") == "PASS"
                and not baseline_false_checks
            )
            or (
                baseline_missing
                and baseline_manifest.get("quality_gate") == "FAIL"
                and baseline_missing <= {"3WNE", "3ZGC"}
                and bool(baseline_manifest.get("directed_recovery_eligible"))
                and baseline_false_checks == {coverage_check}
            )
        )
    ):
        raise RuntimeError("V8 baseline is not an intact PASS/recoverable overlay source")
    search_config = dict(search_manifest.get("config") or {})
    search_input_hashes = dict(search_config.get("input_hashes") or {})
    expected_search_input_hashes = {
        "model": sha256_file(model_path),
        "model_manifest": sha256_file(model_manifest_path),
        "representation_audit": sha256_file(representation_path),
        "baseline_manifest": sha256_file(baseline / "generation_manifest.json"),
        "baseline_all": sha256_file(baseline / "all_candidates.csv"),
        "baseline_unique": sha256_file(baseline / "unique_candidates.csv"),
        "baseline_eligible": sha256_file(baseline / "methylated_new_candidates.csv"),
        "plan": sha256_file(plan_path),
        "native": sha256_file(native_path),
        "historical": sha256_file(historical_path),
        "prior": sha256_file(prior_path),
        "search_program": sha256_file(SEARCH_PATH),
        "reannotator_program": sha256_file(SEARCH_PATH.with_name("10_reannotate_v6_pool_serine_only_v7.py")),
        "generator_program": sha256_file(
            REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
        ),
        "common_program": sha256_file(
            REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"
        ),
        "model_utils_program": sha256_file(REPO_ROOT / "model_utils.py"),
        "nmethyl_config_program": sha256_file(
            REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"
        ),
    }
    search_artifacts = dict(search_manifest.get("artifacts") or {})
    expected_search_artifact_keys = {
        "controls",
        "plausibility",
        "directed_candidates",
        "trace",
    }
    if baseline_missing:
        expected_search_artifact_keys.add("search_ledgers")
    if "3ZGC" in baseline_missing:
        expected_search_artifact_keys.add("checkpoints")
    if not (
        search_manifest.get("config_sha256") == stable_json_sha256(search_config)
        and search_input_hashes == expected_search_input_hashes
        and set(
            str(value).upper()
            for value in search_manifest.get("missing_targets_before_search", [])
        )
        == baseline_missing
        and set(search_artifacts) == expected_search_artifact_keys
        and artifact_matches_exact_path(
            search_artifacts.get("controls"),
            search_dir / "mandatory_length_6_7_controls.csv",
        )
        and artifact_matches_exact_path(
            search_artifacts.get("plausibility"),
            search_dir / "qualified_candidate_plausibility_and_novelty.csv",
        )
        and artifact_matches_exact_path(
            search_artifacts.get("directed_candidates"),
            search_dir / "directed_candidates.csv",
        )
        and artifact_matches_exact_path(
            search_artifacts.get("trace"),
            search_dir / "search_trace_by_round.csv",
        )
        and artifacts_are_hash_pinned_under(search_artifacts, search_dir)
    ):
        raise RuntimeError("V8 search configuration, inputs, or evidence artifacts are stale")

    baseline_hashes_before = {
        name: sha256_file(baseline / filename)
        for name, filename in {
            "all": "all_candidates.csv",
            "unique": "unique_candidates.csv",
            "eligible": "methylated_new_candidates.csv",
            "target_manifest": "target_manifest.csv",
            "summary": "generation_summary_by_target.csv",
            "manifest": "generation_manifest.json",
        }.items()
    }
    baseline_raw = read_csv(baseline / "all_candidates.csv")
    baseline_unique = read_csv(baseline / "unique_candidates.csv")
    baseline_eligible = read_csv(baseline / "methylated_new_candidates.csv")
    baseline_target_manifest = read_csv(baseline / "target_manifest.csv")
    directed_rows = read_csv(search_dir / "directed_candidates.csv")
    controls = read_csv(search_dir / "mandatory_length_6_7_controls.csv")
    plausibility_rows = read_csv(
        search_dir / "qualified_candidate_plausibility_and_novelty.csv"
    )
    plausibility_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    duplicate_plausibility_rows: List[str] = []
    for evidence_row in plausibility_rows:
        evidence_key = (
            str(evidence_row.get("target_name", "")).upper(),
            str(evidence_row.get("sequence", "")).upper(),
        )
        if not all(evidence_key) or evidence_key in plausibility_by_key:
            duplicate_plausibility_rows.append(":".join(evidence_key))
        plausibility_by_key[evidence_key] = dict(evidence_row)
    plan = read_json(plan_path)
    target_names = [str(row["target_name"]).upper() for row in plan["targets"]]
    plan_by_target = {
        str(row["target_name"]).upper(): row for row in plan["targets"]
    }
    missing_before = {
        str(value).upper()
        for value in search_manifest.get("missing_targets_before_search", [])
    }
    if len(baseline_raw) != EXPECTED_BASELINE_ROWS or len(target_names) != EXPECTED_TARGETS:
        raise RuntimeError("Baseline row count or target count changed")
    if any(str(row["target_name"]).upper() not in missing_before for row in directed_rows):
        raise RuntimeError("Directed candidate was created for an unaffected target")
    frozen_search_contract = {
        "3wne_radius": 2,
        "3zgc_rounds": 6,
        "3zgc_beam_width": 512,
        "3zgc_offspring_per_round": 4096,
        "methyl_batch_size": 64,
        "base_plausibility_batch_size": 32,
        "maximum_released_candidates_per_target": 200,
        "probability_persistence_decimal_places": 8,
    }
    if any(
        int(search_config.get(name, -1)) != expected
        for name, expected in frozen_search_contract.items()
    ) or not (
        float(search_config.get("temperature", -1.0)) == TEMPERATURE
        and float(search_config.get("threshold", -1.0)) == THRESHOLD
        and search_config.get("strict_operator") == ">"
        and search_config.get("full_budget_no_early_stop") is True
        and search_config.get("cublas_workspace_config")
        == CUBLAS_WORKSPACE_CONFIG
        and search_config.get("deterministic_algorithms_enabled") is True
        and search_config.get("cudnn_deterministic") is True
        and search_config.get("cudnn_benchmark") is False
        and str(search_config.get("torch_version")) == str(torch.__version__)
        and str(search_config.get("torch_cuda_version")) == str(torch.version.cuda)
        and str(search_config.get("python_version")) == platform.python_version()
        and str(search_config.get("platform_system")) == platform.system()
        and str(search_config.get("platform_machine")) == platform.machine()
        and str(search_config.get("numpy_version")) == str(np.__version__)
        and search_config.get("cudnn_version")
        == (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None
        )
        and bool(search_config.get("cuda_available"))
        == bool(torch.cuda.is_available())
        and search_config.get("cuda_device_name")
        == (
            str(torch.cuda.get_device_name(0))
            if torch.cuda.is_available()
            else None
        )
        and search_config.get("cuda_device_capability")
        == (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        )
    ):
        raise RuntimeError("Directed search did not use the frozen V8 numerical budget")

    ledger_inventory = dict(search_artifacts.get("search_ledgers") or {})
    checkpoint_inventory = dict(search_artifacts.get("checkpoints") or {})
    expected_ledgers: set[str] = set()
    expected_checkpoints: set[str] = set()
    if "3WNE" in missing_before:
        expected_ledgers.add("3wne_exact_search_all.csv.gz")
    if "3ZGC" in missing_before:
        expected_ledgers.add("3zgc_round_00_initial.csv.gz")
        expected_ledgers.update(
            f"3zgc_round_{round_index:02d}.csv.gz" for round_index in range(1, 7)
        )
        expected_checkpoints.update(
            f"3zgc_round_{round_index:02d}.json.gz" for round_index in range(1, 7)
        )
    ledger_inventory_paths_are_exact = (
        set(ledger_inventory) == expected_ledgers
        and all(
            artifact_matches_exact_path(
                ledger_inventory[name], search_dir / name
            )
            for name in expected_ledgers
        )
    )
    checkpoint_inventory_paths_are_exact = (
        set(checkpoint_inventory) == expected_checkpoints
        and all(
            artifact_matches_exact_path(
                checkpoint_inventory[name], search_dir / "checkpoints" / name
            )
            for name in expected_checkpoints
        )
    )
    ledger_sequences: Dict[str, set[str]] = {
        target: set() for target in missing_before
    }
    ledger_row_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    duplicate_ledger_rows: List[str] = []
    for name in sorted(expected_ledgers):
        for row in read_gzip_csv(search_dir / name):
            target = str(row.get("target_name", "")).upper()
            sequence = str(row.get("sequence", "")).upper()
            if target not in ledger_sequences or not sequence:
                duplicate_ledger_rows.append(f"malformed:{name}")
                continue
            if sequence in ledger_sequences[target]:
                duplicate_ledger_rows.append(f"duplicate:{target}:{sequence}")
            ledger_sequences[target].add(sequence)
            ledger_row_by_key[(target, sequence)] = dict(row)
    evaluated_counts_manifest = {
        str(target).upper(): int(value)
        for target, value in dict(
            search_manifest.get("evaluated_sequence_counts") or {}
        ).items()
    }
    evaluated_hashes_manifest = {
        str(target).upper(): str(value)
        for target, value in dict(
            search_manifest.get("evaluated_sequence_sha256_by_target") or {}
        ).items()
    }
    ledger_evidence_complete = (
        ledger_inventory_paths_are_exact
        and checkpoint_inventory_paths_are_exact
        and not duplicate_ledger_rows
        and set(ledger_sequences) == missing_before
        and all(
            len(ledger_sequences[target]) == evaluated_counts_manifest.get(target)
            and hashlib.sha256(
                ("\n".join(sorted(ledger_sequences[target])) + "\n").encode("ascii")
            ).hexdigest()
            == evaluated_hashes_manifest.get(target)
            for target in missing_before
        )
    )
    trace_rows = read_csv(search_dir / "search_trace_by_round.csv")
    trace_by_key = {
        (str(row.get("target_name", "")).upper(), str(row.get("stage", ""))): row
        for row in trace_rows
    }
    expected_trace_stages = set()
    if "3WNE" in missing_before:
        expected_trace_stages.add(("3WNE", "exact_radius_2"))
    if "3ZGC" in missing_before:
        expected_trace_stages.add(("3ZGC", "beam_initial_anchors"))
        expected_trace_stages.update(
            ("3ZGC", f"beam_round_{round_index:02d}")
            for round_index in range(1, 7)
        )
    trace_is_complete = set(trace_by_key) == expected_trace_stages
    trace_values_are_exact = len(trace_by_key) == len(trace_rows)
    if "3WNE" in missing_before:
        wne_rows = read_gzip_csv(search_dir / "3wne_exact_search_all.csv.gz")
        wne_trace = trace_by_key.get(("3WNE", "exact_radius_2"), {})
        try:
            trace_values_are_exact = trace_values_are_exact and (
                int(wne_trace.get("generated_unique", -1)) == len(wne_rows)
                and int(wne_trace.get("newly_scored", -1)) == len(wne_rows)
                and int(wne_trace.get("strict_probability_hits", -1))
                == sum(
                    int(row.get("passes_strict_probability", 0))
                    for row in wne_rows
                )
                and abs(
                    float(wne_trace.get("maximum_probability", "nan"))
                    - max(float(row["maximum_probability"]) for row in wne_rows)
                )
                <= 1e-12
            )
        except (KeyError, TypeError, ValueError):
            trace_values_are_exact = False
    control_identities_are_exact = {
        (
            str(row.get("target_name", "")).upper(),
            str(row.get("control_type", "")),
            str(row.get("natural_sequence", "")).upper(),
            str(row.get("selected_chain", "")),
        )
        for row in controls
    } == {
        ("3WNE", "withdrawn_historical", "GRKWNC", "C"),
        ("3WNE", "native", "PKIDNG", "C"),
        ("3ZGC", "withdrawn_historical", "REGGQNR", "C"),
        ("3ZGC", "native", "GDEETGE", "C"),
    }
    directed_rows_come_from_strict_ledger_hits = True
    for row in directed_rows:
        key = (str(row.get("target_name", "")).upper(), natural_sequence(row))
        ledger_row = ledger_row_by_key.get(key)
        if not (
            ledger_row
            and int(ledger_row.get("passes_strict_probability", 0)) == 1
            and str(row.get("search_stage", ""))
            == str(ledger_row.get("search_stage", ""))
            and abs(
                float(row.get("search_maximum_probability", float("nan")))
                - float(ledger_row.get("maximum_probability", float("nan")))
            )
            <= 1e-12
        ):
            directed_rows_come_from_strict_ledger_hits = False
            break

    search_module = load_module("source_scoped_v8_search_for_audit", SEARCH_PATH)
    v7_auditor = load_module("source_scoped_v8_position_auditor", V7_AUDITOR_PATH)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU final audit requires --allow-cpu")
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
    native_rows = read_jsonl(native_path)
    native_names = [record_name(row, index) for index, row in enumerate(native_rows)]
    if (
        len(native_rows) != EXPECTED_TARGETS
        or len(set(native_names)) != EXPECTED_TARGETS
        or any(not name for name in native_names)
        or set(native_names) != set(target_names)
    ):
        raise RuntimeError(
            "Native final-audit input must contain exactly one record per planned target"
        )
    native_index = dict(zip(native_names, native_rows))
    selected_chains = {
        str(row["target_name"]).upper(): str(row["selected_chain"])
        for row in baseline_target_manifest
    }
    model = load_v28_model(str(model_path), device)
    model.eval()
    reannotator = search_module.load_module(
        "source_scoped_v8_reannotator_for_final_audit",
        search_module.REANNOTATOR_PATH,
    )
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
    # Search ledgers were produced at the frozen batch size of 64.  Recompute
    # every stored score at that same shape before it can authorize a beam or
    # fixed-budget provenance claim.  Candidate release remains independently
    # checked below with a separate batch-one scorer.
    ledger_rescorer = search_module.MethylScorer(
        model,
        device,
        native_index,
        selected_chains,
        64,
        torch,
        common,
        reannotator,
    )
    rescorer = search_module.MethylScorer(
        model,
        device,
        native_index,
        selected_chains,
        1,
        torch,
        common,
        reannotator,
    )
    generator = search_module.load_module(
        "source_scoped_v8_generator_for_final_audit",
        search_module.GENERATOR_PATH,
    )
    target_records, _ = generator.prepare_target_records(
        native_rows,
        selected_chains,
        sorted(search_module.ALLOWED_RECOVERY_TARGETS),
    )
    base_rescorer = search_module.BasePlausibilityScorer(
        model,
        device,
        target_records,
        32,
        torch,
        functional,
        common,
    )
    search_provenance_errors: List[str] = []
    if "3WNE" in missing_before:
        try:
            ranked_wne = search_module.top_ranked_sequences(
                baseline_unique, "3WNE"
            )
            wne_anchors = [
                (ranked_wne[0], "current_v8_baseline_top"),
                (
                    search_module.HISTORICAL_CONTROLS["3WNE"]["sequence"],
                    "withdrawn_historical_control",
                ),
                (search_module.NATIVE_CONTROLS["3WNE"], "native_control"),
            ]
            expected_wne = search_module.wne_search_provenance(wne_anchors, 2)
            if set(expected_wne) != ledger_sequences["3WNE"]:
                raise RuntimeError("3WNE ledger is not the exact radius-2 union")
            for sequence, expected in expected_wne.items():
                search_module.validate_search_ledger_row(
                    ledger_row_by_key[("3WNE", sequence)],
                    "3WNE",
                    sequence,
                    "exact_radius_2",
                    expected,
                )
            search_module.validate_ledger_scores_against_model(
                read_gzip_csv(search_dir / "3wne_exact_search_all.csv.gz"),
                "3WNE",
                "exact_radius_2",
                ledger_rescorer.score_minimal,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            search_provenance_errors.append(str(exc))
    if "3ZGC" in missing_before:
        try:
            ranked_zgc = search_module.top_ranked_sequences(
                baseline_unique, "3ZGC"
            )
            initial_zgc = [
                search_module.HISTORICAL_CONTROLS["3ZGC"]["sequence"],
                search_module.NATIVE_CONTROLS["3ZGC"],
                ranked_zgc[0],
            ]
            zgc_anchors = search_module.select_diverse_sequences(
                ranked_zgc[:128], 34, initial=initial_zgc
            )
            expected_initial_zgc = search_module.zgc_initial_anchor_provenance(
                zgc_anchors, ranked_zgc[0]
            )
            ordered_checkpoints = [
                search_dir / "checkpoints" / name
                for name in sorted(expected_checkpoints)
            ]
            (
                completed_round,
                replayed_seen,
                _replayed_beam,
                _replayed_qualified,
                _replayed_trace,
            ) = search_module.reconstruct_and_validate_zgc_resume(
                search_dir,
                ordered_checkpoints,
                str(search_manifest["config_sha256"]),
                512,
                expected_initial_zgc,
                4096,
                np,
                ledger_rescorer.score_minimal,
            )
            if completed_round != 6 or replayed_seen != ledger_sequences["3ZGC"]:
                raise RuntimeError(
                    "3ZGC fixed-budget provenance replay does not match the ledger union"
                )
            replayed_zgc = {
                (str(row.get("target_name", "")).upper(), str(row.get("stage", ""))): row
                for row in _replayed_trace
                if str(row.get("target_name", "")).upper() == "3ZGC"
            }
            observed_zgc = {
                key: row for key, row in trace_by_key.items() if key[0] == "3ZGC"
            }
            if set(observed_zgc) != set(replayed_zgc):
                raise RuntimeError("3ZGC trace file does not match replayed stages")
            for key, expected in replayed_zgc.items():
                observed = observed_zgc[key]
                if not (
                    int(observed.get("generated_unique", -1))
                    == int(expected.get("generated_unique", -2))
                    and int(observed.get("newly_scored", -1))
                    == int(expected.get("newly_scored", -2))
                    and int(observed.get("strict_probability_hits", -1))
                    == int(expected.get("strict_probability_hits", -2))
                    and abs(
                        float(observed.get("maximum_probability", "nan"))
                        - float(expected.get("maximum_probability", "nan"))
                    )
                    <= 1e-12
                ):
                    raise RuntimeError(f"3ZGC trace file value mismatch: {key[1]}")
        except (KeyError, RuntimeError, ValueError) as exc:
            search_provenance_errors.append(str(exc))
    base_plausibility_recompute_errors: List[str] = []
    strict_ledger_keys = {
        key
        for key, row in ledger_row_by_key.items()
        if int(row.get("passes_strict_probability", 0)) == 1
    }
    if set(plausibility_by_key) != strict_ledger_keys:
        base_plausibility_recompute_errors.append(
            "Plausibility evidence is not the exact strict-ledger-hit set"
        )
    independent_base_by_key: Dict[Tuple[str, str], float] = {}
    independent_floor_by_target: Dict[str, float] = {}
    for target in sorted(missing_before):
        try:
            pool_sequences = sorted(
                {
                    natural_sequence(row)
                    for row in baseline_unique
                    if str(row.get("target_name", "")).upper() == target
                }
            )
            pool_scores = base_rescorer.score(target, pool_sequences)
            floor = search_module.nearest_rank_percentile(
                list(pool_scores.values()), search_module.BASE_PERCENTILE
            )
            independent_floor_by_target[target] = floor
            evidence_sequences = sorted(
                sequence
                for evidence_target, sequence in plausibility_by_key
                if evidence_target == target
            )
            evidence_scores = base_rescorer.score(target, evidence_sequences)
            for sequence, score in evidence_scores.items():
                independent_base_by_key[(target, sequence)] = float(score)
            for sequence in evidence_sequences:
                evidence = plausibility_by_key[(target, sequence)]
                observed_score = float(
                    evidence.get("base_log_probability_mean_all_orders", "nan")
                )
                observed_floor = float(
                    evidence.get("base_plausibility_floor_1pct", "nan")
                )
                expected_score = independent_base_by_key[(target, sequence)]
                expected_pass = int(expected_score >= floor)
                if not (
                    all(
                        math.isfinite(value)
                        for value in (
                            observed_score,
                            observed_floor,
                            expected_score,
                            floor,
                        )
                    )
                    and abs(observed_score - expected_score) <= 1e-12
                    and abs(observed_floor - floor) <= 1e-12
                    and int(evidence.get("passes_base_plausibility", -1))
                    == expected_pass
                ):
                    base_plausibility_recompute_errors.append(
                        f"{target}:{sequence} base plausibility is not independently reproduced"
                    )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            base_plausibility_recompute_errors.append(f"{target}: {exc}")
    rescore_errors: List[str] = []
    baseline_annotation_errors: List[str] = []
    directed_annotation_errors: List[str] = []
    directed_plausibility_errors: List[str] = []
    for row in baseline_eligible:
        baseline_annotation_errors.extend(validate_annotation_row(row))
        baseline_annotation_errors.extend(validate_eligible_candidate_row(row, 1))
    for row in directed_rows:
        directed_annotation_errors.extend(validate_annotation_row(row))
        directed_annotation_errors.extend(validate_eligible_candidate_row(row, 0))
        row_id = str(row.get("candidate_id", ""))
        target = str(row["target_name"]).upper()
        sequence = natural_sequence(row)
        evidence = plausibility_by_key.get((target, sequence))
        if evidence is None:
            directed_plausibility_errors.append(
                f"{row_id}: absent from plausibility/novelty evidence"
            )
        else:
            try:
                evidence_search = float(evidence["search_maximum_probability"])
                evidence_full = float(
                    evidence["qualified_full_maximum_probability"]
                )
                evidence_full_difference = float(
                    evidence["qualified_full_rescore_absolute_difference"]
                )
                evidence_base = float(
                    evidence["base_log_probability_mean_all_orders"]
                )
                evidence_floor = float(evidence["base_plausibility_floor_1pct"])
                row_search = float(row["search_maximum_probability"])
                row_full = float(row["qualified_full_maximum_probability"])
                row_full_difference = float(
                    row["qualified_full_rescore_absolute_difference"]
                )
                row_base = float(row["base_log_probability_mean_all_orders"])
                row_floor = float(row["base_plausibility_floor_1pct"])
                evidence_values = (
                    evidence_search,
                    evidence_full,
                    evidence_full_difference,
                    evidence_base,
                    evidence_floor,
                    row_search,
                    row_full,
                    row_full_difference,
                    row_base,
                    row_floor,
                )
                evidence_matches = (
                    all(math.isfinite(value) for value in evidence_values)
                    and strict_rounded_pass(evidence_search)
                    and strict_rounded_pass(evidence_full)
                    and int(evidence.get("passes_base_plausibility", 0)) == 1
                    and int(evidence.get("pre_rescore_release_eligible", 0)) == 1
                    and not str(evidence.get("duplicate_reason", ""))
                    and str(evidence.get("search_stage", ""))
                    == str(row.get("search_stage", ""))
                    and abs(evidence_search - row_search) <= 1e-12
                    and abs(evidence_full - row_full) <= 1e-12
                    and abs(evidence_full_difference - row_full_difference) <= 1e-12
                    and abs(evidence_base - row_base) <= 1e-12
                    and abs(evidence_floor - row_floor) <= 1e-12
                    and evidence_base >= evidence_floor
                    and row_base >= row_floor
                    and abs(
                        evidence_base
                        - independent_base_by_key.get((target, sequence), float("nan"))
                    )
                    <= 1e-12
                    and abs(
                        evidence_floor
                        - independent_floor_by_target.get(target, float("nan"))
                    )
                    <= 1e-12
                    and abs(
                        abs(evidence_full - evidence_search)
                        - evidence_full_difference
                    )
                    <= 1e-12
                    and evidence_full_difference <= search_module.RESCORE_TOLERANCE
                )
            except (KeyError, TypeError, ValueError):
                evidence_matches = False
            if not evidence_matches:
                directed_plausibility_errors.append(
                    f"{row_id}: plausibility/novelty evidence mismatch"
                )
        if not (
            target in missing_before
            and str(row.get("selected_chain", "")) == selected_chains[target]
            and str(row.get("design_natural_seq", "")).upper() == sequence
            and str(row.get("native_seq", "")).upper()
            == search_module.NATIVE_CONTROLS[target]
            and int(row.get("native_length", -1)) == len(sequence)
            and int(row.get("design_length", -1)) == len(sequence)
            and float(row.get("temperature", "nan")) == TEMPERATURE
            and float(row.get("methyl_threshold", "nan")) == THRESHOLD
            and str(row.get("strict_threshold_operator", "")) == ">"
            and str(row.get("candidate_origin", ""))
            == "DETERMINISTIC_DIRECTED_SEARCH"
            and str(row.get("control_or_candidate", ""))
            == "NOVEL_RECOVERY_CANDIDATE"
            and str(row.get("forward_cyclic_identity", ""))
            == search_module.forward_cyclic_identity(sequence)
            and str(row.get("base_plausibility_context_policy", ""))
            == "native_complex_longest_receptor_visible_all_peptide_decoder_orders_mean"
            and str(row.get("sampling_context_policy", ""))
            == "DETERMINISTIC_DIRECTED_SEARCH_NO_AUTOREGRESSIVE_SAMPLING"
            and str(row.get("sampling_path_annotation_status", ""))
            == "NOT_APPLICABLE_DIRECTED_SEARCH"
            and not str(row.get("sampling_path_methyl_probabilities", ""))
            and not str(row.get("decoding_order_absolute", ""))
            and not str(row.get("base_log_probability_mean", ""))
            and math.isfinite(float(row.get("base_log_probability_mean_all_orders", "nan")))
            and int(row.get("occurrence_count", 0)) == 1
            and int(row.get("passes_methylation_hard_gate", 0)) == 1
            and int(row.get("eligible_for_new_permeability_screen", -1)) == 0
            and int(row.get("permeability_screen_authorized_in_this_release", -1))
            == 0
            and str(row.get("permeability_eligibility_status", ""))
            == "DEFERRED_PENDING_GLOBAL_AND_CYCLIC_RMSD_LT_3A"
            and int(row.get("eligible_for_manual_structure_review", 0)) == 1
            and all(
                int(row.get(field, 1)) == 0
                for field in (
                    "seen_in_historical_4115_exact",
                    "seen_in_historical_4115_naturalized",
                    "seen_in_historical_4115",
                    "seen_in_prior_1333_exact",
                    "seen_in_prior_1333_naturalized",
                    "seen_in_prior_1333",
                )
            )
        ):
            directed_annotation_errors.append(
                f"{row.get('candidate_id', '')}: directed provenance/eligibility contract mismatch"
            )
        recomputed = rescorer.score_full(target, [sequence])[sequence]
        if str(recomputed["design_seq"]) != str(row["design_seq"]):
            rescore_errors.append(f"{row['candidate_id']}: independent design mismatch")
        persisted = [float(value) for value in json.loads(str(row["methyl_probabilities"]))]
        observed = [
            float(value) for value in json.loads(str(recomputed["methyl_probabilities"]))
        ]
        deltas = [abs(left - right) for left, right in zip(persisted, observed)]
        if (
            len(persisted) != len(observed)
            or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in [*persisted, *observed]
            )
            or not all(math.isfinite(delta) and delta <= 2e-6 for delta in deltas)
        ):
            rescore_errors.append(f"{row['candidate_id']}: independent probability mismatch")
        try:
            actionable_maximum = search_module.actionable_probability_max(
                sequence, observed
            )
            stored_batch_one = float(row["batch_one_maximum_probability"])
            stored_search = float(row["search_maximum_probability"])
            stored_difference = float(row["batch_rescore_absolute_difference"])
            probability_summary_matches = (
                all(
                    math.isfinite(value)
                    for value in (
                        actionable_maximum,
                        stored_batch_one,
                        stored_search,
                        stored_difference,
                    )
                )
                and abs(stored_batch_one - actionable_maximum) <= 1e-12
                and abs(abs(stored_batch_one - stored_search) - stored_difference)
                <= 1e-12
                and stored_difference <= search_module.RESCORE_TOLERANCE
                and search_module.strict_rounded_pass(stored_batch_one)
            )
        except (KeyError, TypeError, ValueError):
            probability_summary_matches = False
        if not probability_summary_matches:
            rescore_errors.append(
                f"{row['candidate_id']}: independent maximum/difference mismatch"
            )

    baseline_augmented = []
    for source in baseline_eligible:
        row = dict(source)
        row["candidate_origin"] = "PRESERVED_V6_POOL_REANNOTATED_V8"
        row["eligible_for_manual_structure_review"] = 1
        row["source_eligible_for_new_permeability_screen"] = row.get(
            "eligible_for_new_permeability_screen", ""
        )
        row["eligible_for_new_permeability_screen"] = 0
        row["permeability_screen_authorized_in_this_release"] = 0
        row["permeability_eligibility_status"] = (
            "DEFERRED_PENDING_GLOBAL_AND_CYCLIC_RMSD_LT_3A"
        )
        baseline_augmented.append(row)
    final_rows = baseline_augmented + [dict(row) for row in directed_rows]
    final_rows.sort(
        key=lambda row: (
            str(row["target_name"]),
            str(row["design_natural_seq"]),
            str(row["design_seq"]),
            str(row["candidate_id"]),
        )
    )
    candidate_ids = [str(row.get("candidate_id", "")) for row in final_rows]
    final_keys = [
        (str(row["target_name"]).upper(), str(row["design_seq"])) for row in final_rows
    ]
    counts = Counter(target for target, _sequence in final_keys)
    uncovered = sorted(target for target in target_names if counts[target] < 1)

    historical_rows = read_csv(historical_path)
    prior_rows = read_csv(prior_path)
    historical_exact, historical_natural = exact_and_natural_keys(historical_rows)
    prior_exact, prior_natural = exact_and_natural_keys(prior_rows)
    final_exact, final_natural = exact_and_natural_keys(final_rows)
    baseline_pool_natural = exact_and_natural_keys(baseline_unique)[1]
    directed_natural = exact_and_natural_keys(directed_rows)[1]
    control_natural = {
        (str(row["target_name"]).upper(), str(row["natural_sequence"]).upper())
        for row in controls
    }
    directed_cyclic_keys = [
        (
            str(row["target_name"]).upper(),
            search_module.forward_cyclic_identity(natural_sequence(row)),
        )
        for row in directed_rows
    ]
    external_cyclic_keys = {
        (
            str(row.get("target_name", "")).upper(),
            search_module.forward_cyclic_identity(natural_sequence(row)),
        )
        for row in [*historical_rows, *prior_rows, *baseline_unique]
        if str(row.get("target_name", "")) and natural_sequence(row)
    }
    external_cyclic_keys.update(
        (
            str(row["target_name"]).upper(),
            search_module.forward_cyclic_identity(str(row["natural_sequence"])),
        )
        for row in controls
    )
    directed_forward_cyclic_novel = (
        len(directed_cyclic_keys) == len(set(directed_cyclic_keys))
        and not (set(directed_cyclic_keys) & external_cyclic_keys)
    )

    unaffected = set(target_names) - missing_before
    baseline_unaffected = [
        row for row in baseline_eligible if str(row["target_name"]).upper() in unaffected
    ]
    final_unaffected_original_fields = [
        project_preserved_source_fields(row, baseline_eligible[0])
        for row in final_rows
        if str(row["target_name"]).upper() in unaffected
    ] if baseline_eligible else []
    unaffected_unchanged = (
        canonical_rows_sha256(baseline_unaffected)
        == canonical_rows_sha256(final_unaffected_original_fields)
    )

    concentration_rows, av_alignment, concentration_checks = concentration_audit(
        final_rows,
        baseline_eligible,
        native_rows,
        selected_chains,
        v7_auditor,
    )
    baseline_av = [
        row for row in baseline_eligible if str(row["target_name"]).upper() in AV_FAMILY
    ]

    summary_rows: List[Dict[str, Any]] = []
    for target in target_names:
        baseline_count = sum(
            str(row["target_name"]).upper() == target for row in baseline_eligible
        )
        directed_count = sum(
            str(row["target_name"]).upper() == target for row in directed_rows
        )
        quota = int(plan_by_target[target]["structure_quota"])
        summary_rows.append(
            {
                "target_name": target,
                "baseline_novel_methylated_candidates": baseline_count,
                "directed_recovery_candidates": directed_count,
                "final_signature_candidates": baseline_count + directed_count,
                "has_signature_candidate": int(baseline_count + directed_count > 0),
                "planned_structure_quota": quota,
                "meets_planned_structure_quota": int(
                    baseline_count + directed_count >= quota
                ),
                "structure_status": "NOT_PREDICTED",
            }
        )

    merged_all_rows = []
    for source in baseline_raw:
        row = dict(source)
        row["candidate_origin"] = "PRESERVED_V6_POOL_REANNOTATED_V8"
        row["source_eligible_for_new_permeability_screen"] = row.get(
            "eligible_for_new_permeability_screen", ""
        )
        row["eligible_for_new_permeability_screen"] = 0
        row["permeability_screen_authorized_in_this_release"] = 0
        row["permeability_eligibility_status"] = (
            "DEFERRED_PENDING_GLOBAL_AND_CYCLIC_RMSD_LT_3A"
        )
        merged_all_rows.append(row)
    merged_all_rows.extend(dict(row) for row in directed_rows)
    merged_unique_rows = []
    for source in baseline_unique:
        row = dict(source)
        row["candidate_origin"] = "PRESERVED_V6_POOL_REANNOTATED_V8"
        row["source_eligible_for_new_permeability_screen"] = row.get(
            "eligible_for_new_permeability_screen", ""
        )
        row["eligible_for_new_permeability_screen"] = 0
        row["permeability_screen_authorized_in_this_release"] = 0
        row["permeability_eligibility_status"] = (
            "DEFERRED_PENDING_GLOBAL_AND_CYCLIC_RMSD_LT_3A"
        )
        merged_unique_rows.append(row)
    merged_unique_rows.extend(dict(row) for row in directed_rows)
    final_target_manifest_rows = [
        {
            "target_name": target,
            "selected_chain": selected_chains[target],
            "baseline_raw_rows_retained": sum(
                str(row["target_name"]).upper() == target for row in baseline_raw
            ),
            "baseline_unique_rows_retained": sum(
                str(row["target_name"]).upper() == target for row in baseline_unique
            ),
            "directed_rows_appended": sum(
                str(row["target_name"]).upper() == target for row in directed_rows
            ),
            "final_signature_candidates": counts[target],
            "target_release_status": (
                "CANDIDATE_FOUND_PENDING_MANUAL_AND_STRUCTURE_REVIEW"
                if counts[target]
                else "BLOCKED_NO_SIGNATURE_CANDIDATE"
            ),
            "formal_abstention": 0,
            "structure_status": "NOT_PREDICTED",
        }
        for target in target_names
    ]
    projected_merged_raw = [
        project_preserved_source_fields(row, baseline_raw[0])
        for row in merged_all_rows[: len(baseline_raw)]
    ] if baseline_raw else []
    projected_merged_unique = [
        project_preserved_source_fields(row, baseline_unique[0])
        for row in merged_unique_rows[: len(baseline_unique)]
    ] if baseline_unique else []

    pass_1_checks = {
        "model_representation_baseline_and_search_are_hash_pinned": True,
        "baseline_has_exactly_31500_rows": len(baseline_raw) == EXPECTED_BASELINE_ROWS,
        "merged_overlay_preserves_every_baseline_raw_and_unique_field": (
            canonical_rows_sha256(projected_merged_raw)
            == canonical_rows_sha256(baseline_raw)
            and canonical_rows_sha256(projected_merged_unique)
            == canonical_rows_sha256(baseline_unique)
        ),
        "baseline_eligible_rows_pass_annotation_contract": not baseline_annotation_errors,
        "complete_search_ledger_union_matches_manifest": ledger_evidence_complete,
        "search_ledgers_replay_from_frozen_anchors_rng_and_beam": (
            not search_provenance_errors
        ),
        "every_directed_candidate_is_a_matching_strict_search_ledger_hit": (
            directed_rows_come_from_strict_ledger_hits
        ),
        "fixed_search_trace_is_complete": (
            trace_is_complete and trace_values_are_exact
        ),
        "mandatory_length_6_7_control_identities_are_exact": control_identities_are_exact,
        "directed_release_counts_match_search_manifest": (
            int(search_manifest.get("released_candidates", -1)) == len(directed_rows)
            and {
                str(target).upper(): int(value)
                for target, value in dict(
                    search_manifest.get("released_candidate_counts") or {}
                ).items()
            }
            == dict(sorted(Counter(str(row["target_name"]).upper() for row in directed_rows).items()))
        ),
        "directed_artifact_hash_matches_search_manifest": (
            dict(search_manifest.get("artifacts") or {})
            .get("directed_candidates", {})
            .get("sha256")
            == sha256_file(search_dir / "directed_candidates.csv")
        ),
        "directed_rows_pass_annotation_contract": not directed_annotation_errors,
        "directed_rows_match_plausibility_and_novelty_evidence": (
            not duplicate_plausibility_rows
            and not directed_plausibility_errors
            and not base_plausibility_recompute_errors
        ),
        "directed_rows_independently_rescore": not rescore_errors,
        "candidate_ids_and_target_design_keys_are_unique": (
            all(candidate_ids)
            and len(candidate_ids) == len(set(candidate_ids))
            and len(final_keys) == len(set(final_keys))
        ),
        "all_unaffected_target_rows_are_unchanged": unaffected_unchanged,
        "3av_baseline_subset_is_unchanged": (
            canonical_rows_sha256(baseline_av)
            == canonical_rows_sha256(
                [
                    project_preserved_source_fields(row, baseline_eligible[0])
                    for row in final_rows
                    if str(row["target_name"]).upper() in AV_FAMILY
                ]
            )
            if baseline_eligible
            else False
        ),
    }
    pass_2_checks = {
        **concentration_checks,
        "directed_rows_have_no_fabricated_sampling_order": all(
            not str(row.get("decoding_order_absolute", "")) for row in directed_rows
        ),
        "position_diagnostics_include_recovered_targets": all(
            any(str(row["target_name"]) == target for row in concentration_rows)
            for target in missing_before
        ),
    }
    pass_3_checks = {
        "all_17_targets_have_at_least_one_signature_candidate": not uncovered,
        "all_candidates_are_novel_against_historical_and_prior": (
            not (final_exact & historical_exact)
            and not (final_natural & historical_natural)
            and not (final_exact & prior_exact)
            and not (final_natural & prior_natural)
        ),
        "directed_candidates_are_not_current_pool_or_controls": (
            not (directed_natural & baseline_pool_natural)
            and not (directed_natural & control_natural)
        ),
        "directed_candidates_are_forward_cyclic_novel": directed_forward_cyclic_novel,
        "formal_abstention_is_absent": (
            not baseline_manifest.get("targets_formally_abstained")
            and not search_manifest.get("missing_targets_after_search")
        ),
        "structure_handoff_is_not_created": not any(
            path.exists()
            for path in (
                V8_ROOT / "handoff",
                V8_ROOT / "serine_qc_source_scoped_hybrid_v8_shangge_handoff.zip",
            )
        ),
        "permeability_remains_deferred": (
            not any(
                (out_dir / name).exists()
                for name in ("permeability_input.csv", "permeability_input_manifest.csv")
            )
            and all(
                int(row.get("eligible_for_new_permeability_screen", -1)) == 0
                and int(
                    row.get("permeability_screen_authorized_in_this_release", -1)
                )
                == 0
                and str(row.get("permeability_eligibility_status", ""))
                == "DEFERRED_PENDING_GLOBAL_AND_CYCLIC_RMSD_LT_3A"
                for row in [*final_rows, *merged_all_rows, *merged_unique_rows]
            )
        ),
    }
    pass_1 = "PASS" if all(pass_1_checks.values()) else "FAIL"
    pass_2 = "PASS" if all(pass_2_checks.values()) else "FAIL"
    pass_3 = "PASS" if all(pass_3_checks.values()) else "FAIL"
    quality_gate = "PASS" if pass_1 == pass_2 == pass_3 == "PASS" else "FAIL"

    final_all_path = out_dir / "all_candidates.csv"
    final_unique_path = out_dir / "unique_candidates.csv"
    final_candidate_path = out_dir / "methylated_new_candidates.csv"
    final_target_manifest_path = out_dir / "target_manifest.csv"
    final_summary_path = out_dir / "generation_summary_by_target.csv"
    atomic_write_csv(final_all_path, merged_all_rows, union_fields(merged_all_rows))
    atomic_write_csv(
        final_unique_path, merged_unique_rows, union_fields(merged_unique_rows)
    )
    atomic_write_csv(final_candidate_path, final_rows, union_fields(final_rows))
    atomic_write_csv(
        final_target_manifest_path,
        final_target_manifest_rows,
        list(final_target_manifest_rows[0]),
    )
    atomic_write_csv(final_summary_path, summary_rows, list(summary_rows[0]))
    atomic_write_csv(
        audit_out / "three_pass_concentration_by_target.csv",
        concentration_rows,
        list(concentration_rows[0]),
    )
    atomic_write_json(audit_out / "av_family_physical_position_support.json", av_alignment)

    final_manifest = {
        "quality_gate": quality_gate,
        "release_status": (
            "READY_FOR_MANUAL_SCIENTIFIC_REVIEW_NO_STRUCTURE_HANDOFF"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_SEND_TO_SHANGGE"
        ),
        "protocol": V8_FINAL_PROTOCOL,
        "finalizer_program_sha256": sha256_file(SCRIPT_PATH),
        "position_auditor_program_sha256": sha256_file(V7_AUDITOR_PATH),
        "model_sha256": sha256_file(model_path),
        "baseline_artifact_sha256": baseline_hashes_before,
        "search_manifest_sha256": sha256_file(
            search_dir / "directed_search_manifest.json"
        ),
        "temperature": TEMPERATURE,
        "methyl_threshold": THRESHOLD,
        "strict_threshold_operator": ">",
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
        "baseline_raw_rows": len(baseline_raw),
        "directed_rows_appended_to_all_and_unique": len(directed_rows),
        "recovered_all_rows": len(merged_all_rows),
        "recovered_unique_rows": len(merged_unique_rows),
        "baseline_eligible_candidates": len(baseline_eligible),
        "directed_recovery_candidates": len(directed_rows),
        "final_signature_candidates": len(final_rows),
        "targets_with_signature_candidate": EXPECTED_TARGETS - len(uncovered),
        "targets_without_signature_candidate": uncovered,
        "targets_formally_abstained": [],
        "targets_below_planned_structure_quota_diagnostic": [
            row["target_name"]
            for row in summary_rows
            if not int(row["meets_planned_structure_quota"])
        ],
        "structure_status": (
            "NOT_PREDICTED; every candidate still requires global-complex and "
            "complete cyclic-peptide CA RMSD <3 A"
        ),
        "structure_handoff_status": "NOT_CREATED_PENDING_MANUAL_REVIEW",
        "permeability_status": "DEFERRED_UNTIL_RETURNED_STRUCTURES_PASS_BOTH_RMSD_GATES",
        "artifacts": {
            "all_candidates": {
                "path": str(final_all_path),
                "sha256": sha256_file(final_all_path),
            },
            "unique_candidates": {
                "path": str(final_unique_path),
                "sha256": sha256_file(final_unique_path),
            },
            "final_candidates": {
                "path": str(final_candidate_path),
                "sha256": sha256_file(final_candidate_path),
            },
            "target_summary": {
                "path": str(final_summary_path),
                "sha256": sha256_file(final_summary_path),
            },
            "target_manifest": {
                "path": str(final_target_manifest_path),
                "sha256": sha256_file(final_target_manifest_path),
            },
        },
    }
    atomic_write_json(out_dir / "generation_manifest.json", final_manifest)

    audit_report = {
        "quality_gate": quality_gate,
        "release_status": final_manifest["release_status"],
        "protocol": V8_AUDIT_PROTOCOL,
        "finalizer_program_sha256": sha256_file(SCRIPT_PATH),
        "position_auditor_program_sha256": sha256_file(V7_AUDITOR_PATH),
        "test_reuse_limitation": model_manifest.get("test_reuse_limitation"),
        "development_status": model_manifest.get("development_status"),
        "pass_1_integrity_and_rescore": {
            "quality_gate": pass_1,
            "checks": pass_1_checks,
            "baseline_annotation_errors": baseline_annotation_errors[:25],
            "directed_annotation_errors": directed_annotation_errors[:25],
            "directed_plausibility_errors": directed_plausibility_errors[:25],
            "base_plausibility_recompute_errors": (
                base_plausibility_recompute_errors[:25]
            ),
            "duplicate_plausibility_rows": duplicate_plausibility_rows[:25],
            "independent_rescore_errors": rescore_errors[:25],
            "search_ledger_errors": duplicate_ledger_rows[:25],
            "search_provenance_errors": search_provenance_errors[:25],
        },
        "pass_2_position_and_representation": {
            "quality_gate": pass_2,
            "checks": pass_2_checks,
            "concentration": concentration_rows,
            "av_family_physical_alignment": av_alignment,
        },
        "pass_3_novelty_coverage_workflow": {
            "quality_gate": pass_3,
            "checks": pass_3_checks,
            "uncovered_targets": uncovered,
            "candidate_count_by_target": dict(sorted(counts.items())),
        },
        "model_metrics_are_not_modified_by_search": model_manifest["v8_test"],
        "artifacts": {
            "final_manifest": {
                "path": str(out_dir / "generation_manifest.json"),
                "sha256": sha256_file(out_dir / "generation_manifest.json"),
            },
            "final_candidates": final_manifest["artifacts"]["final_candidates"],
            "final_all_candidates": final_manifest["artifacts"]["all_candidates"],
            "final_unique_candidates": final_manifest["artifacts"]["unique_candidates"],
            "final_target_manifest": final_manifest["artifacts"]["target_manifest"],
            "final_target_summary": final_manifest["artifacts"]["target_summary"],
            "search_manifest": {
                "path": str(search_dir / "directed_search_manifest.json"),
                "sha256": sha256_file(search_dir / "directed_search_manifest.json"),
            },
            "position_concentration": {
                "path": str(audit_out / "three_pass_concentration_by_target.csv"),
                "sha256": sha256_file(
                    audit_out / "three_pass_concentration_by_target.csv"
                ),
            },
            "av_family_physical_support": {
                "path": str(audit_out / "av_family_physical_position_support.json"),
                "sha256": sha256_file(
                    audit_out / "av_family_physical_position_support.json"
                ),
            },
        },
    }
    atomic_write_json(audit_out / "three_pass_generation_audit.json", audit_report)

    baseline_hashes_after = {
        name: sha256_file(baseline / filename)
        for name, filename in {
            "all": "all_candidates.csv",
            "unique": "unique_candidates.csv",
            "eligible": "methylated_new_candidates.csv",
            "target_manifest": "target_manifest.csv",
            "summary": "generation_summary_by_target.csv",
            "manifest": "generation_manifest.json",
        }.items()
    }
    if baseline_hashes_after != baseline_hashes_before:
        raise RuntimeError("Immutable V8 baseline changed during final audit")

    print("===== V8 RECOVERY OVERLAY THREE-PASS AUDIT COMPLETE =====", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    print(f"Target coverage: {EXPECTED_TARGETS - len(uncovered)}/{EXPECTED_TARGETS}", flush=True)
    print(f"Final candidates: {len(final_rows)}", flush=True)
    print("Structure handoff: NOT CREATED", flush=True)
    if quality_gate != "PASS":
        failed = [
            name
            for name, value in (("pass_1", pass_1), ("pass_2", pass_2), ("pass_3", pass_3))
            if value != "PASS"
        ]
        raise RuntimeError("V8 final three-pass audit failed: " + ", ".join(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-manifest", default=str(DEFAULT_MODEL_MANIFEST))
    parser.add_argument("--representation-audit", default=str(DEFAULT_REPRESENTATION))
    parser.add_argument("--baseline-run-dir", default=str(DEFAULT_BASELINE))
    parser.add_argument("--search-dir", default=str(DEFAULT_SEARCH))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--historical-designs-csv", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--prior-handoff-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--audit-out-dir", default=str(DEFAULT_AUDIT_OUT))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
