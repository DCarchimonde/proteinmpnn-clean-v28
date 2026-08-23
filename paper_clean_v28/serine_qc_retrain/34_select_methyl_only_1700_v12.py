#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select exactly 17 x 100 V11 methylated sequences before structure work.

No ProteinMPNN base floor and no predicted/observed RMSD participate in this
selection.  The only model release decision is the fail-closed full-cyclic V11
representation-minimum methylation gate.  Basic identity, length, novelty, and
forward-cyclic de-duplication checks prevent malformed or repeated handoff rows.
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
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
V8_SEARCH_PATH = SCRIPT_PATH.with_name("14_directed_recovery_search_v8.py")
V11_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "cyclic_native_v11_1700_monomer"
DEFAULT_GENERATION = V11_ROOT / "generation"
DEFAULT_3ZGC_DIR = V11_ROOT / "v12_methyl_only" / "3zgc_directed_search"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_v12_methyl_only_1700.json")
DEFAULT_MODEL = V11_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
DEFAULT_AUDIT = V11_ROOT / "representation_audit" / "cyclic_representation_audit.json"
DEFAULT_HISTORICAL = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "all_designs.csv"
)
DEFAULT_PRIOR = REPO_ROOT / "v9_inputs" / "methylated_new_candidates.csv"
DEFAULT_OUT = V11_ROOT / "v12_methyl_only" / "selection_17x100"

THRESHOLD = 0.6
TEMPERATURE = 0.5
QUOTA = 100
TARGETS = (
    "1SFI", "3AV9", "3AVA", "3AVB", "3AVF", "3AVG", "3AVH",
    "3AVI", "3AVJ", "3AVK", "3AVM", "3AVN", "3P8F", "3WNE",
    "3ZGC", "4K1E", "4KEL",
)
NATURAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
METHYLATABLE_AA = NATURAL_AA - {"P"}
PROTOCOL = "v12_methylation_only_exact_17_x_100_selector_v1"

DETAIL_NAME = "1700_详细审计.csv"
CONCISE_NAME = "1700_给尚哥_极简.csv"
FASTA_NAME = "1700_给尚哥_结构输入.fasta"
SUMMARY_NAME = "selection_summary_by_target.csv"
PROBLEMS_NAME = "candidate_validation_problems.csv"
MANIFEST_NAME = "v12_1700_methyl_only_release_manifest.json"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
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


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(fields or union_fields(rows) or ["status"])
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def strict_pass(value: float, threshold: float = THRESHOLD) -> bool:
    numeric = float(value)
    return (
        math.isfinite(numeric)
        and 0.0 <= numeric <= 1.0
        and round(numeric, 8) > float(threshold)
    )


def parse_vector(value: Any, field: str, length: int) -> List[float]:
    try:
        parsed = value if isinstance(value, list) else json.loads(str(value))
        vector = [float(item) for item in parsed]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field}_malformed") from exc
    if len(vector) != length:
        raise ValueError(f"{field}_length_mismatch")
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in vector):
        raise ValueError(f"{field}_nonfinite_or_out_of_range")
    return vector


def canonical_rotation(sequence: str) -> str:
    if not sequence:
        raise ValueError("empty_sequence")
    return min(sequence[index:] + sequence[:index] for index in range(len(sequence)))


def actionable_max(sequence: str, values: Sequence[float]) -> Tuple[float, int]:
    candidates = [
        (float(value), index)
        for index, (token, value) in enumerate(zip(sequence, values))
        if token in METHYLATABLE_AA
    ]
    if not candidates:
        return 0.0, 0
    value, index = max(candidates, key=lambda pair: (pair[0], -pair[1]))
    return value, index + 1


def validate_candidate(
    source: Mapping[str, Any],
    expected_target: str,
    stable_gate: Any,
) -> Tuple[Dict[str, Any] | None, List[str]]:
    row = dict(source)
    errors: List[str] = []
    target = str(row.get("target_name", "")).strip().upper()
    design = str(row.get("design_seq", "")).strip()
    natural = str(row.get("design_natural_seq", design)).strip().upper()
    candidate_id = str(row.get("candidate_id", "")).strip()
    if target != expected_target:
        errors.append("target_mismatch")
    if not candidate_id:
        errors.append("missing_candidate_id")
    if not natural or not set(natural) <= NATURAL_AA:
        errors.append("invalid_natural_sequence")
    if not design or design.upper() != natural:
        errors.append("design_natural_sequence_mismatch")
    if design and not any(token.islower() for token in design):
        errors.append("no_lowercase_methyl_token")
    try:
        native = str(row.get("native_seq", "")).strip().upper()
        if (
            not native
            or not set(native) <= NATURAL_AA
            or int(row.get("native_length", -1)) != len(native)
            or int(row.get("design_length", -1)) != len(natural)
            or len(native) != len(natural)
            or int(row.get("length_match", 0)) != 1
            or int(row.get("valid_token_gate", 0)) != 1
        ):
            errors.append("native_design_length_or_token_gate_failed")
        if int(row.get("passes_methylation_hard_gate", 0)) != 1:
            errors.append("persisted_methylation_hard_gate_not_pass")
    except (TypeError, ValueError):
        errors.append("native_design_length_or_token_gate_malformed")
    try:
        if float(row.get("methyl_threshold", "nan")) != THRESHOLD:
            errors.append("methyl_threshold_changed")
        if float(row.get("temperature", "nan")) != TEMPERATURE:
            errors.append("temperature_changed")
    except (TypeError, ValueError):
        errors.append("invalid_temperature_or_threshold")
    if str(row.get("strict_threshold_operator", ">")) != ">":
        errors.append("threshold_operator_changed")
    if natural:
        try:
            minimum = parse_vector(
                row.get("methyl_probability_representation_min", ""),
                "representation_min",
                len(natural),
            )
            mean = parse_vector(
                row.get("methyl_probabilities", ""),
                "representation_mean",
                len(natural),
            )
            maximum = parse_vector(
                row.get("methyl_probability_representation_max", ""),
                "representation_max",
                len(natural),
            )
            span = parse_vector(
                row.get("methyl_probability_representation_span", ""),
                "representation_span",
                len(natural),
            )
        except ValueError as exc:
            errors.append(str(exc))
            minimum, mean, maximum, span = [], [], [], []
        if minimum:
            floor_maximum, floor_position = actionable_max(natural, minimum)
            mean_maximum, _mean_position = actionable_max(natural, mean)
            expected_design = "".join(
                token.lower()
                if token in METHYLATABLE_AA and strict_pass(value)
                else token
                for token, value in zip(natural, minimum)
            )
            expected_positions = [
                index for index, token in enumerate(expected_design, start=1)
                if token.islower()
            ]
            try:
                saved_positions = [
                    int(value)
                    for value in json.loads(str(row.get("methyl_positions_1based", "")))
                ]
                disagreement = [
                    int(value)
                    for value in json.loads(
                        str(
                            row.get(
                                "representation_threshold_disagreement_positions_1based",
                                "",
                            )
                        )
                    )
                ]
            except (TypeError, ValueError, json.JSONDecodeError):
                saved_positions, disagreement = [], [-1]
                errors.append("methyl_position_or_disagreement_vector_malformed")
            if not strict_pass(floor_maximum):
                errors.append("no_representation_minimum_site_strictly_above_0_6")
            if design != expected_design or saved_positions != expected_positions:
                errors.append("lowercase_pattern_not_derived_from_representation_minimum")
            if disagreement or int(row.get("representation_threshold_disagreement_count", -1)) != 0:
                errors.append("cyclic_start_threshold_disagreement")
            if any(
                lower > center + 2e-6
                or center > upper + 2e-6
                or abs((upper - lower) - width) > 2e-6
                for lower, center, upper, width in zip(minimum, mean, maximum, span)
            ):
                errors.append("representation_probability_summary_inconsistent")
            row.update(
                {
                    "_release_floor": floor_maximum,
                    "_release_position": floor_position,
                    "_ranking_mean": mean_maximum,
                    "_span_max": max(span),
                    "_natural_cyclic_key": canonical_rotation(natural),
                }
            )
    if not errors and not stable_gate(row, natural):
        errors.append("independent_fail_closed_cyclic_gate_failed")
    if errors:
        return None, errors
    row.update(
        {
            "target_name": target,
            "design_seq": design,
            "design_natural_seq": natural,
            "prestructure_base_gate_used": 0,
            "prestructure_rmsd_available": 0,
            "prestructure_rmsd_rank_used": 0,
            "rmsd_status": "NOT_AVAILABLE_UNTIL_SHANGGE_RETURNS_STRUCTURES",
        }
    )
    return row, []


def quality_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(row["_release_floor"]),
        -float(row["_ranking_mean"]),
        float(row["_span_max"]),
        str(row.get("candidate_id", "")),
    )


def deduplicate_and_select(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in sorted(rows, key=quality_key):
        best.setdefault(str(row["_natural_cyclic_key"]), row)
    return sorted(best.values(), key=quality_key)[:QUOTA]


def identity(left: str, right: str) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def exclusion_keys(paths: Sequence[Path]) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]]]:
    exact: set[Tuple[str, str]] = set()
    cyclic: set[Tuple[str, str]] = set()
    for path in paths:
        for row in read_csv(path):
            target = str(row.get("target_name", "")).strip().upper()
            sequence = str(
                row.get("design_natural_seq")
                or row.get("design_seq")
                or row.get("sequence")
                or ""
            ).strip().upper()
            if target in TARGETS and sequence and set(sequence) <= NATURAL_AA:
                exact.add((target, sequence))
                cyclic.add((target, canonical_rotation(sequence)))
    return exact, cyclic


def clean_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def verify_views(detail: Path, concise: Path, fasta: Path) -> Dict[str, bool]:
    detailed = read_csv(detail)
    short = read_csv(concise)
    fasta_lines = [
        line.strip() for line in fasta.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    rows_match = len(detailed) == len(short) == len(TARGETS) * QUOTA
    if rows_match:
        rows_match = all(
            str(left.get(field, "")) == str(right.get(field, ""))
            for left, right in zip(detailed, short)
            for field in (
                "final_release_id", "candidate_id", "target_name", "design_seq",
                "design_natural_seq", "methyl_positions_1based",
            )
        )
    fasta_match = len(fasta_lines) == 2 * len(TARGETS) * QUOTA
    if fasta_match:
        for index, row in enumerate(detailed):
            expected_header = (
                f">{row['final_release_id']}|{row['target_name']}|"
                f"candidate={row['candidate_id']}|marked={row['design_seq']}|"
                f"methyl_positions={row['methyl_positions_1based']}"
            )
            if (
                fasta_lines[2 * index] != expected_header
                or fasta_lines[2 * index + 1] != row["design_natural_seq"]
            ):
                fasta_match = False
                break
    return {
        "reopened_detailed_and_concise_views_match": rows_match,
        "reopened_fasta_matches_detailed_rows": fasta_match,
    }


def run(args: argparse.Namespace) -> None:
    generation_dir = Path(args.generation_dir).resolve()
    candidates_path = generation_dir / "methylated_new_candidates.csv"
    generation_manifest_path = generation_dir / "generation_manifest.json"
    zgc_dir = Path(args.zgc_dir).resolve()
    zgc_path = zgc_dir / "3zgc_exact_100_methylated.csv"
    zgc_manifest_path = zgc_dir / "3zgc_methyl_only_search_manifest.json"
    plan_path = Path(args.plan).resolve()
    model_path = Path(args.model).resolve()
    audit_path = Path(args.representation_audit).resolve()
    historical_path = Path(args.historical_csv).resolve()
    prior_path = Path(args.prior_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    required = (
        candidates_path, generation_manifest_path, zgc_path, zgc_manifest_path,
        plan_path, model_path, audit_path, historical_path, prior_path,
        V8_SEARCH_PATH,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = read_json(plan_path)
    if not (
        plan.get("protocol") == "v12_prestructure_methylation_only_exact_17_target_1700_release"
        and float(plan.get("temperature", -1.0)) == TEMPERATURE
        and float(plan.get("methyl_threshold", -1.0)) == THRESHOLD
        and int(plan.get("final_release_quota_per_target", -1)) == QUOTA
        and tuple(plan.get("targets", [])) == TARGETS
        and plan.get("prestructure_base_score_policy")
        == "not_a_release_gate_and_not_used_for_selection"
        and plan.get("prestructure_rmsd_policy")
        == "not_available_and_not_predicted_before_shangge_returns_structures"
    ):
        raise RuntimeError("V12 methyl-only plan changed")
    model_hash = sha256_file(model_path)
    audit = read_json(audit_path)
    audit_checks = dict(audit.get("quality_checks") or {})
    if not (
        audit.get("quality_gate") == "PASS"
        and audit.get("model_sha256") == model_hash
        and audit_checks
        and all(value is True for value in audit_checks.values())
    ):
        raise RuntimeError("V12 selector requires the exact PASS V11 audit/model")
    generation = read_json(generation_manifest_path)
    generation_false = {
        name
        for name, passed in dict(generation.get("quality_checks") or {}).items()
        if not bool(passed)
    }
    allowed_false = {
        "every_target_meets_pre_structure_candidate_quota",
        "every_target_meets_final_release_diversity_reserve",
        "no_single_position_exceeds_80_percent_of_sites",
        "no_single_residue_exceeds_80_percent_of_sites",
        "no_target_has_single_residue_above_80_percent_when_n_ge_30",
        "no_target_has_unsupported_single_position_above_80_percent_when_n_ge_30",
    }
    generation_candidate_record = dict(
        dict(generation.get("artifacts") or {}).get("methylated_new_candidates") or {}
    )
    if not (
        generation.get("model_sha256") == model_hash
        and generation_false
        and generation_false <= allowed_false
        and generation_candidate_record.get("sha256") == sha256_file(candidates_path)
    ):
        raise RuntimeError("V12 selector refuses an unrecognized V11 generation pool")
    zgc_manifest = read_json(zgc_manifest_path)
    if not (
        zgc_manifest.get("quality_gate") == "PASS"
        and zgc_manifest.get("release_status")
        == "AUTHORIZED_3ZGC_EXACT_100_METHYLATION_ONLY_PRESTRUCTURE_ROWS"
        and int(zgc_manifest.get("quota", -1)) == QUOTA
        and zgc_manifest.get("inputs", {}).get("model", {}).get("sha256") == model_hash
        and zgc_manifest.get("artifacts", {}).get("exact_100_release", {}).get("sha256")
        == sha256_file(zgc_path)
        and len(read_csv(zgc_path)) == QUOTA
    ):
        raise RuntimeError("V12 selector requires a PASS exact-100 3ZGC search")

    v8 = load_module("v12_selector_stable_gate", V8_SEARCH_PATH)
    source_rows = read_csv(candidates_path) + read_csv(zgc_path)
    historical_exact, historical_cyclic = exclusion_keys((historical_path, prior_path))
    valid_by_target: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    problems: List[Dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for row_number, source in enumerate(source_rows, start=2):
        target = str(source.get("target_name", "")).strip().upper()
        if target not in TARGETS:
            problems.append(
                {"row_number": row_number, "candidate_id": source.get("candidate_id", ""), "problem": "target_not_in_frozen_17"}
            )
            continue
        candidate_id = str(source.get("candidate_id", "")).strip()
        if candidate_id in candidate_ids:
            problems.append(
                {"row_number": row_number, "candidate_id": candidate_id, "target_name": target, "problem": "duplicate_candidate_id"}
            )
            continue
        candidate_ids.add(candidate_id)
        validated, errors = validate_candidate(source, target, v8.stable_cyclic_methyl_release_gate)
        if errors or validated is None:
            for error in errors:
                problems.append(
                    {"row_number": row_number, "candidate_id": candidate_id, "target_name": target, "problem": error}
                )
            continue
        natural = str(validated["design_natural_seq"])
        if (
            (target, natural) in historical_exact
            or (target, canonical_rotation(natural)) in historical_cyclic
        ):
            problems.append(
                {"row_number": row_number, "candidate_id": candidate_id, "target_name": target, "problem": "historical_or_prior_duplicate"}
            )
            continue
        valid_by_target[target].append(validated)

    selected: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    shortfalls: List[str] = []
    for target in TARGETS:
        target_selected = deduplicate_and_select(valid_by_target[target])
        if len(target_selected) != QUOTA:
            shortfalls.append(target)
        identities = [
            identity(str(left["design_natural_seq"]), str(right["design_natural_seq"]))
            for left, right in combinations(target_selected, 2)
        ]
        for rank, source in enumerate(target_selected, start=1):
            row = clean_row(source)
            row.update(
                {
                    "final_release_id": f"V12_{target}_{rank:03d}",
                    "target_release_rank": rank,
                    "selection_release_floor_probability": source["_release_floor"],
                    "selection_ranking_mean_probability": source["_ranking_mean"],
                    "selection_representation_span_max": source["_span_max"],
                    "selection_policy": plan["selection_ranking_policy"],
                    "selection_used_base_score": 0,
                    "selection_used_rmsd": 0,
                }
            )
            selected.append(row)
        position_counts = Counter(
            position
            for row in target_selected
            for position in json.loads(str(row["methyl_positions_1based"]))
        )
        residue_counts = Counter(
            str(row["design_natural_seq"])[position - 1]
            for row in target_selected
            for position in json.loads(str(row["methyl_positions_1based"]))
        )
        summary.append(
            {
                "target_name": target,
                "strict_valid_pool_rows": len(valid_by_target[target]),
                "forward_cyclic_unique_pool_rows": len(
                    {str(row["_natural_cyclic_key"]) for row in valid_by_target[target]}
                ),
                "selected_rows": len(target_selected),
                "quota": QUOTA,
                "quota_pass": int(len(target_selected) == QUOTA),
                "release_floor_probability_min": min(
                    (float(row["_release_floor"]) for row in target_selected),
                    default="",
                ),
                "release_floor_probability_max": max(
                    (float(row["_release_floor"]) for row in target_selected),
                    default="",
                ),
                "maximum_pairwise_natural_identity": max(identities, default=0.0),
                "mean_pairwise_natural_identity": (
                    sum(identities) / len(identities) if identities else 0.0
                ),
                "methyl_position_counts": json.dumps(position_counts, sort_keys=True),
                "methyl_parent_residue_counts": json.dumps(residue_counts, sort_keys=True),
                "base_score_role": "NOT_USED",
                "rmsd_role": "WAITING_FOR_RETURNED_STRUCTURES",
            }
        )

    expected_count = len(TARGETS) * QUOTA
    target_counts = Counter(str(row["target_name"]) for row in selected)
    release_checks = {
        "v11_model_and_full_grid_audit_are_pinned": True,
        "source_generation_failures_are_only_replaced_quota_or_diagnostic_checks": (
            bool(generation_false) and generation_false <= allowed_false
        ),
        "every_selected_row_passes_strict_representation_minimum_methyl_gate": (
            len(selected) == expected_count
            and all(int(row.get("passes_methylation_hard_gate", 0)) == 1 for row in selected)
        ),
        "every_target_has_exactly_100_rows": (
            not shortfalls
            and set(target_counts) == set(TARGETS)
            and all(target_counts[target] == QUOTA for target in TARGETS)
        ),
        "selected_ids_are_unique": (
            len({str(row["final_release_id"]) for row in selected}) == expected_count
            and len({str(row["candidate_id"]) for row in selected}) == expected_count
        ),
        "selected_natural_and_forward_cyclic_identities_are_unique_within_target": all(
            len(
                {
                    str(row["design_natural_seq"])
                    for row in selected if str(row["target_name"]) == target
                }
            ) == QUOTA
            and len(
                {
                    canonical_rotation(str(row["design_natural_seq"]))
                    for row in selected if str(row["target_name"]) == target
                }
            ) == QUOTA
            for target in TARGETS
        ),
        "no_prestructure_base_score_was_used": all(
            int(row["selection_used_base_score"]) == 0 for row in selected
        ),
        "no_prestructure_rmsd_or_rmsd_prediction_was_used": all(
            int(row["selection_used_rmsd"]) == 0
            and row["rmsd_status"] == "NOT_AVAILABLE_UNTIL_SHANGGE_RETURNS_STRUCTURES"
            for row in selected
        ),
    }
    quality_gate = "PASS" if all(release_checks.values()) else "FAIL"
    atomic_write_csv(out_dir / SUMMARY_NAME, summary)
    atomic_write_csv(out_dir / PROBLEMS_NAME, problems)

    if quality_gate == "PASS":
        detail_path = out_dir / DETAIL_NAME
        concise_path = out_dir / CONCISE_NAME
        fasta_path = out_dir / FASTA_NAME
        atomic_write_csv(detail_path, selected)
        concise_rows = [
            {
                "final_release_id": row["final_release_id"],
                "candidate_id": row["candidate_id"],
                "target_name": row["target_name"],
                "design_seq": row["design_seq"],
                "design_natural_seq": row["design_natural_seq"],
                "methyl_positions_1based": row["methyl_positions_1based"],
            }
            for row in selected
        ]
        atomic_write_csv(
            concise_path,
            concise_rows,
            (
                "final_release_id", "candidate_id", "target_name", "design_seq",
                "design_natural_seq", "methyl_positions_1based",
            ),
        )
        fasta_lines: List[str] = []
        for row in selected:
            fasta_lines.extend(
                [
                    (
                        f">{row['final_release_id']}|{row['target_name']}|"
                        f"candidate={row['candidate_id']}|marked={row['design_seq']}|"
                        f"methyl_positions={row['methyl_positions_1based']}"
                    ),
                    str(row["design_natural_seq"]),
                ]
            )
        atomic_write_text(fasta_path, "\n".join(fasta_lines) + "\n")
        release_checks.update(verify_views(detail_path, concise_path, fasta_path))
        quality_gate = "PASS" if all(release_checks.values()) else "FAIL"

    artifacts: Dict[str, Any] = {
        "summary": {"path": str(out_dir / SUMMARY_NAME), "sha256": sha256_file(out_dir / SUMMARY_NAME)},
        "problems": {"path": str(out_dir / PROBLEMS_NAME), "sha256": sha256_file(out_dir / PROBLEMS_NAME)},
    }
    if quality_gate == "PASS":
        for key, name in (
            ("detailed", DETAIL_NAME), ("shangge_concise", CONCISE_NAME),
            ("shangge_fasta", FASTA_NAME),
        ):
            artifacts[key] = {
                "path": str(out_dir / name),
                "sha256": sha256_file(out_dir / name),
            }
    manifest = {
        "quality_gate": quality_gate,
        "release_status": (
            "AUTHORIZED_EXACT_17_X_100_METHYLATION_ONLY_PRESTRUCTURE_HANDOFF"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_SEND_TO_SHANGGE"
        ),
        "protocol": PROTOCOL,
        "selected_rows": len(selected),
        "quota_per_target": QUOTA,
        "threshold": THRESHOLD,
        "temperature": TEMPERATURE,
        "prestructure_base_score_policy": "NOT_USED",
        "prestructure_rmsd_policy": "NOT_AVAILABLE_UNTIL_STRUCTURES_RETURN",
        "targets_below_quota": shortfalls,
        "quality_checks": release_checks,
        "inputs": {
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "model": {"path": str(model_path), "sha256": model_hash},
            "representation_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
            "generation_manifest": {"path": str(generation_manifest_path), "sha256": sha256_file(generation_manifest_path)},
            "generation_candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            "zgc_search_manifest": {"path": str(zgc_manifest_path), "sha256": sha256_file(zgc_manifest_path)},
            "zgc_candidates": {"path": str(zgc_path), "sha256": sha256_file(zgc_path)},
            "historical_csv": {"path": str(historical_path), "sha256": sha256_file(historical_path)},
            "prior_csv": {"path": str(prior_path), "sha256": sha256_file(prior_path)},
        },
        "artifacts": artifacts,
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
    }
    atomic_write_json(out_dir / MANIFEST_NAME, manifest)
    print("===== V12 17 x 100 METHYL-ONLY SELECTION COMPLETE =====", flush=True)
    print(f"Strict input rows: {len(source_rows):,}", flush=True)
    print(f"Selected rows: {len(selected):,} (17 x 100 required)", flush=True)
    print(f"Targets below quota: {shortfalls}", flush=True)
    print("Pre-structure base gate: NOT USED", flush=True)
    print("Pre-structure RMSD: NOT AVAILABLE", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in release_checks.items() if not passed]
        raise RuntimeError("V12 methyl-only selection blocked: " + ", ".join(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", default=str(DEFAULT_GENERATION))
    parser.add_argument("--zgc-dir", default=str(DEFAULT_3ZGC_DIR))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--representation-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--historical-csv", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--prior-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
