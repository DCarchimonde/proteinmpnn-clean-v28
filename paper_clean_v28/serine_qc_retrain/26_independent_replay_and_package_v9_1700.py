#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GPU replay gate for the final V9 17 x 100 structure handoff.

The selector is deliberately not the last authority.  This program loads the
selected 1,700 rows and re-runs both model-dependent decisions one sequence at
a time:

* peptide-only N-methyl annotation over every physical cyclic start and every
  cyclic decoder order at T=0.5; and
* receptor-visible natural-base scoring over the exact L x L physical-start by
  decoder-order grid.

Persisted evidence is accepted only when it is numerically close to the fresh
batch-size-one replay *and* every strict, eight-decimal threshold decision is
identical.  Final files are copied into the output directory only after every
row, hash, quota, and reopened-view check passes.  A failed replay never emits
files with the final handoff names.
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
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
GENERATOR_PATH = REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
SCORER_PATH = SCRIPT_PATH.with_name("24_score_uniform_cyclic_base_v9.py")
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_BEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "best_designs.csv"
)
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_stability_v9_1700.json")

THRESHOLD = 0.6
TEMPERATURE = 0.5
QUOTA = 100
EXPECTED_TOTAL = 1700
PROBABILITY_ATOL = 5.0e-6
BASE_SCORE_ATOL = 1.0e-5
SUMMARY_ATOL = 1.0e-5
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
METHYLATABLE_AA = set(NATURAL_AA) - {"P"}
FROZEN_TARGETS = (
    "1SFI", "3AV9", "3AVA", "3AVB", "3AVF", "3AVG", "3AVH", "3AVI",
    "3AVJ", "3AVK", "3AVM", "3AVN", "3P8F", "3WNE", "3ZGC", "4K1E",
    "4KEL",
)
EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_"
    "cyclic_stability_worst_start_v9"
)
AUDIT_PROTOCOL = "cyclic_stability_worst_start_heldout_gate_v9"
AUDIT_AUTHORIZATION = "CYCLIC_STABILITY_V9_VALIDATED_FOR_UNIFORM_REGENERATION"
ANNOTATION_MODE = (
    "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
)
RANKING_POLICY = "representation_mean"
RELEASE_POLICY = "representation_min_strict_gt_threshold_zero_disagreement"
CYCLIC_BASE_PROTOCOL = (
    "receptor_visible_all_physical_starts_all_decoder_orders_exact_v9"
)
CYCLIC_BASE_FLOOR_POLICY = (
    "per_target_bottom_1pct_current_pool_outlier_filter_"
    "not_independent_calibration_v9"
)
SELECTOR_PROTOCOL = "independent_v9_cyclic_stability_17x100_release_audit_v1"
REPLAY_PROTOCOL = "batch1_full_model_replay_v9_17x100_release_gate_v1"

FINAL_DETAIL = "1700_详细审计.csv"
FINAL_CONCISE = "1700_给尚哥_极简.csv"
FINAL_FASTA = "1700_给尚哥_结构输入.fasta"
REPLAY_CSV = "v9_1700_independent_replay.csv"
REPLAY_MANIFEST = "v9_1700_independent_replay_manifest.json"
BLOCKED_CSV = "BLOCKED_replay_rows.csv"
BLOCKED_JSON = "BLOCKED_replay_diagnostics.json"
KNOWN_OUTPUTS = (
    FINAL_DETAIL,
    FINAL_CONCISE,
    FINAL_FASTA,
    REPLAY_CSV,
    REPLAY_MANIFEST,
    BLOCKED_CSV,
    BLOCKED_JSON,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    observed: set[str] = set()
    for row in rows:
        for field in row:
            if field not in observed:
                observed.add(field)
                fields.append(field)
    return fields


def atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
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


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    existing = [out_dir / name for name in KNOWN_OUTPUTS if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Replay output already exists; use --overwrite only for this isolated "
            "directory: " + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def strict_rounded_pass(value: float, threshold: float = THRESHOLD) -> bool:
    return round(float(value), 8) > float(threshold)


def parse_json_list(value: Any, field: str) -> List[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a JSON list")
    return parsed


def finite_vector(value: Any, field: str, length: int | None = None) -> List[float]:
    parsed = parse_json_list(value, field)
    try:
        result = [float(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains a non-numeric value") from exc
    if length is not None and len(result) != length:
        raise ValueError(f"{field} length {len(result)} != {length}")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} contains a non-finite value")
    return result


def finite_matrix(value: Any, field: str, rows: int, columns: int) -> List[List[float]]:
    parsed = parse_json_list(value, field)
    try:
        matrix = [[float(item) for item in row] for row in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains a non-numeric value") from exc
    if len(matrix) != rows or any(len(row) != columns for row in matrix):
        raise ValueError(f"{field} is not {rows} x {columns}")
    if not all(math.isfinite(item) for row in matrix for item in row):
        raise ValueError(f"{field} contains a non-finite value")
    return matrix


def flatten(values: Sequence[Any]) -> List[float]:
    result: List[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.extend(flatten(value))
        else:
            result.append(float(value))
    return result


def compare_numeric_contract(
    field: str,
    persisted: Sequence[Any],
    replayed: Sequence[Any],
    atol: float,
    threshold: float | None = None,
) -> Tuple[List[str], float]:
    """Compare tensors and, when relevant, their exact rounded hard calls."""

    left = flatten(persisted)
    right = flatten(replayed)
    if len(left) != len(right):
        return [f"{field}_shape_mismatch"], float("inf")
    if not all(math.isfinite(value) for value in left + right):
        return [f"{field}_nonfinite"], float("inf")
    maximum = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
    errors: List[str] = []
    if maximum > atol:
        errors.append(f"{field}_numeric_mismatch")
    if threshold is not None and any(
        strict_rounded_pass(a, threshold) != strict_rounded_pass(b, threshold)
        for a, b in zip(left, right)
    ):
        errors.append(f"{field}_threshold_decision_mismatch")
    return errors, maximum


def marked_sequence_from_floor(
    natural_sequence: str, minima: Sequence[float], threshold: float = THRESHOLD
) -> str:
    if len(natural_sequence) != len(minima):
        raise ValueError("Sequence/probability length mismatch")
    return "".join(
        token.lower()
        if token in METHYLATABLE_AA and strict_rounded_pass(value, threshold)
        else token
        for token, value in zip(natural_sequence, minima)
    )


def threshold_disagreements(
    minima: Sequence[float], maxima: Sequence[float], threshold: float = THRESHOLD
) -> List[int]:
    if len(minima) != len(maxima):
        raise ValueError("Minimum/maximum length mismatch")
    return [
        index
        for index, (minimum, maximum) in enumerate(zip(minima, maxima), start=1)
        if not strict_rounded_pass(minimum, threshold)
        and strict_rounded_pass(maximum, threshold)
    ]


def methyl_positions(sequence: str) -> List[int]:
    return [index for index, token in enumerate(sequence, start=1) if token.islower()]


def exact_target_quota_checks(
    rows: Sequence[Mapping[str, Any]], expected_targets: Sequence[str]
) -> Dict[str, bool]:
    counts = Counter(str(row.get("target_name", "")).strip().upper() for row in rows)
    expected = set(expected_targets)
    return {
        "detailed_row_count_is_exactly_1700": len(rows) == EXPECTED_TOTAL,
        "detailed_target_set_is_exact_frozen_17": set(counts) == expected,
        "every_detailed_target_has_exactly_100_rows": (
            set(counts) == expected and all(counts[target] == QUOTA for target in expected)
        ),
    }


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: str | None = None
    sequence_parts: List[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence_parts)))
            header = line[1:]
            sequence_parts = []
        elif header is None:
            raise ValueError("FASTA sequence appears before its header")
        else:
            sequence_parts.append(line)
    if header is not None:
        records.append((header, "".join(sequence_parts)))
    return records


def verify_selector_views(
    detailed_rows: Sequence[Mapping[str, Any]], concise_path: Path, fasta_path: Path
) -> Dict[str, bool]:
    concise = read_csv(concise_path)
    same_concise = len(concise) == len(detailed_rows)
    concise_fields = (
        "final_release_id",
        "candidate_id",
        "target_name",
        "design_seq",
        "design_natural_seq",
        "methyl_positions_1based",
    )
    if same_concise:
        same_concise = all(
            all(str(short.get(field, "")) == str(detail.get(field, "")) for field in concise_fields)
            for detail, short in zip(detailed_rows, concise)
        )
    try:
        fasta = parse_fasta(fasta_path)
    except ValueError:
        fasta = []
    same_fasta = len(fasta) == len(detailed_rows)
    if same_fasta:
        for row, (header, natural) in zip(detailed_rows, fasta):
            expected_header = (
                f"{row['final_release_id']}|{row['target_name']}|"
                f"candidate={row['candidate_id']}|marked={row['design_seq']}|"
                f"methyl_positions={row['methyl_positions_1based']}"
            )
            if header != expected_header or natural != str(row["design_natural_seq"]):
                same_fasta = False
                break
    return {
        "selector_concise_view_exactly_matches_detailed": same_concise,
        "selector_fasta_view_exactly_matches_detailed": same_fasta,
    }


def manifest_artifact_hash(
    manifest: Mapping[str, Any], section: str, label: str
) -> str:
    payload = manifest.get(section, {})
    if not isinstance(payload, Mapping):
        return ""
    artifact = payload.get(label, {})
    if not isinstance(artifact, Mapping):
        return ""
    return str(artifact.get("sha256", ""))


def manifest_input_hash(manifest: Mapping[str, Any], label: str) -> str:
    inputs = manifest.get("inputs", {})
    if not isinstance(inputs, Mapping):
        return ""
    value = inputs.get(label, {})
    if not isinstance(value, Mapping):
        return ""
    return str(value.get("sha256", ""))


def validate_upstream_hash_contract(
    selector: Mapping[str, Any],
    scorer: Mapping[str, Any],
    audit: Mapping[str, Any],
    plan: Mapping[str, Any],
    selector_manifest_path: Path,
    detailed_path: Path,
    concise_path: Path,
    fasta_path: Path,
    model_path: Path,
    audit_path: Path,
    scorer_manifest_path: Path,
    plan_path: Path,
    native_path: Path,
    best_path: Path,
) -> Dict[str, bool]:
    del selector_manifest_path  # Its bytes are recorded in the final manifest.
    model_hash = sha256_file(model_path)
    audit_hash = sha256_file(audit_path)
    plan_hash = sha256_file(plan_path)
    native_hash = sha256_file(native_path)
    best_hash = sha256_file(best_path)
    selector_checks = selector.get("quality_checks", {})
    plan_targets = [
        str(item.get("target_name", "")).upper()
        for item in plan.get("targets", [])
        if isinstance(item, Mapping)
    ]
    scorer_inputs = scorer.get("inputs", {})
    selector_inputs = selector.get("inputs", {})
    return {
        "selector_manifest_is_authorized_pass": (
            selector.get("quality_gate") == "PASS"
            and selector.get("release_status") == "AUTHORIZED_EXACT_17_X_100_STRUCTURE_HANDOFF"
            and selector.get("protocol") == SELECTOR_PROTOCOL
            and int(selector.get("selected_rows", -1)) == EXPECTED_TOTAL
            and int(selector.get("quota_per_target", -1)) == QUOTA
            and float(selector.get("threshold", -1.0)) == THRESHOLD
            and float(selector.get("temperature", -1.0)) == TEMPERATURE
        ),
        "selector_quality_checks_are_all_true": (
            isinstance(selector_checks, Mapping)
            and bool(selector_checks)
            and all(value is True for value in selector_checks.values())
        ),
        "frozen_plan_has_exact_17_targets_and_protocol": (
            tuple(plan_targets) == FROZEN_TARGETS
            and int(plan.get("expected_target_count", -1)) == len(FROZEN_TARGETS)
            and float(plan.get("temperature", -1.0)) == TEMPERATURE
            and float(plan.get("methyl_threshold", -1.0)) == THRESHOLD
        ),
        "selector_detailed_hash_matches_manifest": (
            manifest_artifact_hash(selector, "release_artifacts", "detailed_audit")
            == sha256_file(detailed_path)
        ),
        "selector_concise_hash_matches_manifest": (
            manifest_artifact_hash(selector, "release_artifacts", "shangge_concise")
            == sha256_file(concise_path)
        ),
        "selector_fasta_hash_matches_manifest": (
            manifest_artifact_hash(selector, "release_artifacts", "shangge_fasta")
            == sha256_file(fasta_path)
        ),
        "selector_model_hash_matches_actual_model": (
            manifest_input_hash(selector, "model") == model_hash
        ),
        "selector_audit_hash_matches_actual_audit": (
            manifest_input_hash(selector, "heldout_audit") == audit_hash
        ),
        "selector_plan_hash_matches_actual_plan": (
            manifest_input_hash(selector, "plan") == plan_hash
        ),
        "selector_scorer_hash_matches_actual_scorer_manifest": (
            manifest_input_hash(selector, "cyclic_base_manifest")
            == sha256_file(scorer_manifest_path)
        ),
        "scorer_manifest_is_authorized_pass": (
            scorer.get("quality_gate") == "PASS"
            and scorer.get("protocol") == CYCLIC_BASE_PROTOCOL
            and scorer.get("floor_policy") == CYCLIC_BASE_FLOOR_POLICY
            and int(scorer.get("target_count", -1)) == len(FROZEN_TARGETS)
        ),
        "scorer_model_and_plan_hashes_match_actual_bytes": (
            scorer.get("model_sha256") == model_hash
            and scorer.get("plan_sha256") == plan_hash
        ),
        "scorer_native_and_best_hashes_match_actual_bytes": (
            manifest_input_hash(scorer, "native_jsonl") == native_hash
            and manifest_input_hash(scorer, "best_csv") == best_hash
        ),
        "scorer_audit_hash_matches_actual_audit": (
            manifest_input_hash(scorer, "representation_audit") == audit_hash
        ),
        "scorer_passing_pool_is_selector_candidate_pool": (
            manifest_artifact_hash(scorer, "artifacts", "passing_candidates")
            == manifest_input_hash(selector, "candidates")
        ),
        "scorer_and_selector_generation_manifest_hashes_match": (
            isinstance(scorer_inputs, Mapping)
            and isinstance(selector_inputs, Mapping)
            and manifest_input_hash(scorer, "generation_manifest")
            == manifest_input_hash(selector, "generation_manifest")
            and bool(manifest_input_hash(scorer, "generation_manifest"))
        ),
        "heldout_audit_authorizes_exact_model_and_plan": (
            audit.get("quality_gate") == "PASS"
            and audit.get("protocol") == AUDIT_PROTOCOL
            and audit.get("release_authorization") == AUDIT_AUTHORIZATION
            and audit.get("model_sha256") == model_hash
            and audit.get("plan_sha256") == plan_hash
            and audit.get("annotation_mode") == ANNOTATION_MODE
        ),
    }


def target_floors(scorer_manifest: Mapping[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for row in scorer_manifest.get("target_summary", []):
        if not isinstance(row, Mapping):
            continue
        target = str(row.get("target_name", "")).upper()
        try:
            floor = float(row.get("cyclic_base_floor", "nan"))
        except (TypeError, ValueError):
            continue
        if target and math.isfinite(floor):
            result[target] = floor
    return result


def json_round_vector(values: Sequence[float]) -> List[float]:
    return [round(float(value), 8) for value in values]


def json_round_matrix(values: Sequence[Sequence[float]]) -> List[List[float]]:
    return [json_round_vector(row) for row in values]


class MethylReplay:
    """Peptide-only full cyclic representation replay, one selected row at a time."""

    def __init__(
        self,
        model: Any,
        device: Any,
        target_records: Mapping[str, Mapping[str, Any]],
        torch_module: Any,
        featurize_records: Any,
        peptide_only_annotation_tensors: Any,
        cyclic_representation_probabilities: Any,
        natural_alphabet: str,
    ) -> None:
        self.model = model
        self.device = device
        self.target_records = target_records
        self.torch = torch_module
        self.featurize_records = featurize_records
        self.peptide_only_annotation_tensors = peptide_only_annotation_tensors
        self.cyclic_representation_probabilities = cyclic_representation_probabilities
        self.alphabet_index = {token: index for index, token in enumerate(natural_alphabet)}
        self.features: Dict[str, Tuple[Any, ...]] = {}

    def _features(self, target: str) -> Tuple[Any, ...]:
        if target not in self.features:
            packed = self.featurize_records(
                [self.target_records[target]],
                device=self.device,
                eval_chains="masked",
                max_peptide_len=30,
            )
            if packed is None:
                raise RuntimeError(f"Feature construction failed for {target}")
            tensors, _metas = packed
            self.features[target] = tuple(tensors[:6])
        return self.features[target]

    def score(self, target: str, natural: str) -> Dict[str, Any]:
        X, S_true, mask, chain_M, _residue_idx, _chain_encoding = self._features(target)
        selected = self.torch.nonzero(
            (chain_M[0] * mask[0]) > 0.0, as_tuple=False
        ).squeeze(-1)
        length = int(selected.numel())
        if len(natural) != length:
            raise RuntimeError(
                f"Candidate length mismatch for {target}: {len(natural)} != {length}"
            )
        sequence_tensor = self.torch.tensor(
            [self.alphabet_index[token] for token in natural],
            device=self.device,
            dtype=self.torch.long,
        )
        S_candidate = S_true.clone()
        S_candidate[0, selected] = sequence_tensor
        annotation = self.peptide_only_annotation_tensors(
            X, S_candidate, mask, chain_M
        )
        annotation_X, annotation_S, annotation_mask, annotation_chain_M, annotation_residue_idx, annotation_chain_encoding = annotation
        annotation_selected = self.torch.nonzero(
            (annotation_chain_M[0] * annotation_mask[0]) > 0.0,
            as_tuple=False,
        ).squeeze(-1)
        if int(annotation_selected.numel()) != length:
            raise RuntimeError(f"Peptide-only replay length changed for {target}")
        with self.torch.no_grad():
            result = self.cyclic_representation_probabilities(
                model=self.model,
                X=annotation_X,
                S_natural=annotation_S,
                mask=annotation_mask,
                chain_M=annotation_chain_M,
                residue_idx=annotation_residue_idx,
                chain_encoding_all=annotation_chain_encoding,
                temperature=TEMPERATURE,
            )

        def selected_vector(name: str) -> List[float]:
            return [
                float(value)
                for value in result[name][0, annotation_selected].detach().cpu().tolist()
            ]

        matrix_tensor = result["representation_probability_by_start"][
            0, :length, annotation_selected
        ]
        matrix = [
            [float(value) for value in row]
            for row in matrix_tensor.detach().cpu().tolist()
        ]
        means = selected_vector("mean")
        minima = selected_vector("representation_min")
        maxima = selected_vector("representation_max")
        spans = selected_vector("representation_span")
        representation_std = selected_vector("representation_std")
        order_std = selected_vector("decoder_order_std_mean")
        return {
            "mean": means,
            "min": minima,
            "max": maxima,
            "span": spans,
            "representation_std": representation_std,
            "order_std": order_std,
            "by_start": matrix,
            "marked_sequence": marked_sequence_from_floor(natural, minima),
            "disagreement_positions": threshold_disagreements(minima, maxima),
        }


def compare_scalar(
    field: str, persisted_value: Any, replayed_value: float, atol: float
) -> Tuple[List[str], float]:
    try:
        persisted = float(persisted_value)
    except (TypeError, ValueError):
        return [f"{field}_invalid"], float("inf")
    if not math.isfinite(persisted) or not math.isfinite(replayed_value):
        return [f"{field}_nonfinite"], float("inf")
    delta = abs(persisted - replayed_value)
    return ([f"{field}_numeric_mismatch"] if delta > atol else []), delta


def replay_and_compare_row(
    source: Mapping[str, Any],
    methyl_replay: MethylReplay,
    base_scorer: Any,
    frozen_floor: float,
) -> Dict[str, Any]:
    row = dict(source)
    target = str(row.get("target_name", "")).strip().upper()
    natural = str(row.get("design_natural_seq", "")).strip().upper()
    marked = str(row.get("design_seq", "")).strip()
    length = len(natural)
    errors: List[str] = []
    maxima_by_group: Dict[str, float] = {}
    if not natural or not set(natural) <= set(NATURAL_AA):
        raise ValueError(f"Invalid natural sequence for {target}: {natural!r}")
    if marked.upper() != natural or len(marked) != length:
        errors.append("marked_and_natural_sequence_mismatch")
    try:
        if float(row.get("temperature", "nan")) != TEMPERATURE:
            errors.append("temperature_not_0.5")
        if float(row.get("methyl_threshold", "nan")) != THRESHOLD:
            errors.append("methyl_threshold_not_0.6")
    except ValueError:
        errors.append("invalid_temperature_or_threshold")

    methyl = methyl_replay.score(target, natural)
    replay_mean = json_round_vector(methyl["mean"])
    replay_min = json_round_vector(methyl["min"])
    replay_max = json_round_vector(methyl["max"])
    replay_span = json_round_vector(methyl["span"])
    replay_rep_std = json_round_vector(methyl["representation_std"])
    replay_order_std = json_round_vector(methyl["order_std"])
    replay_matrix = json_round_matrix(methyl["by_start"])

    persisted_vectors = {
        "methyl_probabilities": finite_vector(
            row.get("methyl_probabilities", ""), "methyl_probabilities", length
        ),
        "methyl_probability_representation_min": finite_vector(
            row.get("methyl_probability_representation_min", ""),
            "methyl_probability_representation_min",
            length,
        ),
        "methyl_probability_representation_max": finite_vector(
            row.get("methyl_probability_representation_max", ""),
            "methyl_probability_representation_max",
            length,
        ),
        "methyl_probability_representation_span": finite_vector(
            row.get("methyl_probability_representation_span", ""),
            "methyl_probability_representation_span",
            length,
        ),
        "methyl_probability_representation_std": finite_vector(
            row.get("methyl_probability_representation_std", ""),
            "methyl_probability_representation_std",
            length,
        ),
        "methyl_probability_order_std": finite_vector(
            row.get("methyl_probability_order_std", ""),
            "methyl_probability_order_std",
            length,
        ),
    }
    replay_vectors = {
        "methyl_probabilities": replay_mean,
        "methyl_probability_representation_min": replay_min,
        "methyl_probability_representation_max": replay_max,
        "methyl_probability_representation_span": replay_span,
        "methyl_probability_representation_std": replay_rep_std,
        "methyl_probability_order_std": replay_order_std,
    }
    threshold_fields = {
        "methyl_probabilities",
        "methyl_probability_representation_min",
        "methyl_probability_representation_max",
    }
    for field, persisted in persisted_vectors.items():
        current_errors, maximum = compare_numeric_contract(
            field,
            persisted,
            replay_vectors[field],
            PROBABILITY_ATOL,
            THRESHOLD if field in threshold_fields else None,
        )
        errors.extend(current_errors)
        maxima_by_group[field] = maximum

    persisted_matrix = finite_matrix(
        row.get("methyl_probability_representation_by_start", ""),
        "methyl_probability_representation_by_start",
        length,
        length,
    )
    matrix_errors, matrix_delta = compare_numeric_contract(
        "methyl_probability_representation_by_start",
        persisted_matrix,
        replay_matrix,
        PROBABILITY_ATOL,
        THRESHOLD,
    )
    errors.extend(matrix_errors)
    maxima_by_group["methyl_probability_representation_by_start"] = matrix_delta
    replay_columns = list(zip(*replay_matrix))
    matrix_summary_vectors = {
        "mean": [sum(values) / len(values) for values in replay_columns],
        "min": [min(values) for values in replay_columns],
        "max": [max(values) for values in replay_columns],
        "span": [max(values) - min(values) for values in replay_columns],
        "representation_std": [
            math.sqrt(
                sum((value - sum(values) / len(values)) ** 2 for value in values)
                / len(values)
            )
            for values in replay_columns
        ],
    }
    for name, recomputed in matrix_summary_vectors.items():
        summary_errors, _summary_delta = compare_numeric_contract(
            f"replay_methyl_matrix_{name}_summary",
            replay_vectors[
                {
                    "mean": "methyl_probabilities",
                    "min": "methyl_probability_representation_min",
                    "max": "methyl_probability_representation_max",
                    "span": "methyl_probability_representation_span",
                    "representation_std": "methyl_probability_representation_std",
                }[name]
            ],
            recomputed,
            PROBABILITY_ATOL,
            THRESHOLD if name in {"mean", "min", "max"} else None,
        )
        errors.extend(summary_errors)

    replay_marked = str(methyl["marked_sequence"])
    replay_disagreements = list(methyl["disagreement_positions"])
    if replay_marked != marked:
        errors.append("replay_lowercase_pattern_mismatch")
    if not methyl_positions(replay_marked):
        errors.append("replay_contains_no_methylation")
    if replay_disagreements:
        errors.append("replay_has_cyclic_threshold_disagreement")
    try:
        persisted_disagreements = [
            int(value)
            for value in parse_json_list(
                row.get("representation_threshold_disagreement_positions_1based", ""),
                "representation_threshold_disagreement_positions_1based",
            )
        ]
        if persisted_disagreements != replay_disagreements:
            errors.append("persisted_replay_disagreement_positions_mismatch")
        if int(row.get("representation_threshold_disagreement_count", -1)) != len(
            replay_disagreements
        ):
            errors.append("persisted_replay_disagreement_count_mismatch")
        persisted_methyl_positions = [
            int(value)
            for value in parse_json_list(
                row.get("methyl_positions_1based", ""), "methyl_positions_1based"
            )
        ]
        if persisted_methyl_positions != methyl_positions(replay_marked):
            errors.append("persisted_replay_methyl_positions_mismatch")
    except (TypeError, ValueError):
        errors.append("invalid_persisted_positions")
    expected_annotation_metadata = {
        "annotation_mode": ANNOTATION_MODE,
        "annotation_context_policy": "peptide_chain_only_no_visible_receptor_chains",
        "annotation_ranking_probability_policy": RANKING_POLICY,
        "annotation_release_probability_policy": RELEASE_POLICY,
    }
    for field, expected in expected_annotation_metadata.items():
        if str(row.get(field, "")) != expected:
            errors.append(f"{field}_mismatch")
    try:
        if int(row.get("annotation_visible_receptor_chains", -1)) != 0:
            errors.append("annotation_visible_receptor_chains_not_zero")
        if int(row.get("annotation_representation_ensemble_size", -1)) != length:
            errors.append("annotation_representation_ensemble_size_not_L")
        if int(row.get("annotation_order_ensemble_size", -1)) != length:
            errors.append("annotation_order_ensemble_size_not_L")
        if int(row.get("annotation_decoder_order_ensemble_size", -1)) != length:
            errors.append("annotation_decoder_order_ensemble_size_not_L")
        if int(row.get("annotation_total_probability_ensemble_size", -1)) != length * length:
            errors.append("annotation_total_probability_ensemble_size_not_L_squared")
        if int(row.get("stable_cyclic_release_gate", 0)) != 1:
            errors.append("stable_cyclic_release_gate_not_one")
    except ValueError:
        errors.append("invalid_annotation_metadata")

    methyl_positions_zero = [position - 1 for position in methyl_positions(replay_marked)]
    scalar_expectations = {
        "methyl_probability_min": min(replay_mean),
        "methyl_probability_mean": sum(replay_mean) / length,
        "methyl_probability_max": max(replay_mean),
        "methyl_probability_order_std_max": max(replay_order_std),
        "methyl_probability_representation_std_max": max(replay_rep_std),
        "methyl_probability_representation_span_max": max(replay_span),
        "methyl_site_probability_min": min(replay_mean[index] for index in methyl_positions_zero),
        "methyl_site_probability_mean": (
            sum(replay_mean[index] for index in methyl_positions_zero)
            / len(methyl_positions_zero)
        ),
        "methyl_site_probability_max": max(replay_mean[index] for index in methyl_positions_zero),
        "methyl_site_representation_floor_min": min(
            replay_min[index] for index in methyl_positions_zero
        ),
        "methyl_site_representation_floor_mean": (
            sum(replay_min[index] for index in methyl_positions_zero)
            / len(methyl_positions_zero)
        ),
        "methyl_site_representation_floor_max": max(
            replay_min[index] for index in methyl_positions_zero
        ),
    } if methyl_positions_zero else {}
    for field, replayed in scalar_expectations.items():
        scalar_errors, delta = compare_scalar(
            field, row.get(field, ""), replayed, SUMMARY_ATOL
        )
        errors.extend(scalar_errors)
        maxima_by_group[field] = delta

    base = base_scorer.score(target, [natural])[natural]
    replay_base_matrix = finite_matrix(
        base["cyclic_base_log_probability_start_by_decoder_order"],
        "replay_cyclic_base_log_probability_start_by_decoder_order",
        length,
        length,
    )
    replay_base_by_start = finite_vector(
        base["cyclic_base_log_probability_by_start"],
        "replay_cyclic_base_log_probability_by_start",
        length,
    )
    persisted_base_matrix = finite_matrix(
        row.get("cyclic_base_log_probability_start_by_decoder_order", ""),
        "cyclic_base_log_probability_start_by_decoder_order",
        length,
        length,
    )
    persisted_base_by_start = finite_vector(
        row.get("cyclic_base_log_probability_by_start", ""),
        "cyclic_base_log_probability_by_start",
        length,
    )
    base_matrix_errors, base_matrix_delta = compare_numeric_contract(
        "cyclic_base_log_probability_start_by_decoder_order",
        persisted_base_matrix,
        replay_base_matrix,
        BASE_SCORE_ATOL,
    )
    base_start_errors, base_start_delta = compare_numeric_contract(
        "cyclic_base_log_probability_by_start",
        persisted_base_by_start,
        replay_base_by_start,
        BASE_SCORE_ATOL,
    )
    errors.extend(base_matrix_errors)
    errors.extend(base_start_errors)
    maxima_by_group["cyclic_base_matrix"] = base_matrix_delta
    maxima_by_group["cyclic_base_by_start"] = base_start_delta
    replay_base_by_start_from_matrix = [
        sum(values) / len(values) for values in replay_base_matrix
    ]
    base_internal_errors, _base_internal_delta = compare_numeric_contract(
        "replay_cyclic_base_matrix_start_summary",
        replay_base_by_start,
        replay_base_by_start_from_matrix,
        BASE_SCORE_ATOL,
    )
    errors.extend(base_internal_errors)
    base_scalar_fields = (
        "cyclic_base_log_probability_mean",
        "cyclic_base_log_probability_min",
        "cyclic_base_log_probability_max",
        "cyclic_base_log_probability_span",
        "cyclic_base_log_probability_std",
    )
    for field in base_scalar_fields:
        scalar_errors, delta = compare_scalar(
            field, row.get(field, ""), float(base[field]), SUMMARY_ATOL
        )
        errors.extend(scalar_errors)
        maxima_by_group[field] = delta
    try:
        persisted_floor = float(row.get("cyclic_base_floor", "nan"))
    except ValueError:
        persisted_floor = float("nan")
    if (
        not math.isfinite(persisted_floor)
        or abs(persisted_floor - frozen_floor) > 1.0e-8
    ):
        errors.append("cyclic_base_floor_manifest_mismatch")
    replay_base_gate = (
        math.isfinite(persisted_floor)
        and round(float(base["cyclic_base_log_probability_mean"]), 8) >= persisted_floor
    )
    try:
        if int(row.get("cyclic_base_gate_pass", 0)) != int(replay_base_gate):
            errors.append("cyclic_base_gate_replay_mismatch")
    except ValueError:
        errors.append("invalid_cyclic_base_gate_pass")
    if not replay_base_gate:
        errors.append("replay_cyclic_base_below_frozen_floor")
    if str(row.get("cyclic_base_score_protocol", "")) != CYCLIC_BASE_PROTOCOL:
        errors.append("cyclic_base_score_protocol_mismatch")
    if str(row.get("cyclic_base_floor_policy", "")) != CYCLIC_BASE_FLOOR_POLICY:
        errors.append("cyclic_base_floor_policy_mismatch")
    try:
        if int(row.get("cyclic_base_physical_start_count", -1)) != length:
            errors.append("cyclic_base_physical_start_count_not_L")
        if int(row.get("cyclic_base_decoder_order_count_per_start", -1)) != length:
            errors.append("cyclic_base_decoder_order_count_not_L")
        if int(row.get("cyclic_base_total_ensemble_size", -1)) != length * length:
            errors.append("cyclic_base_total_ensemble_size_not_L_squared")
    except ValueError:
        errors.append("invalid_cyclic_base_ensemble_size")

    errors = sorted(set(errors))
    return {
        "final_release_id": str(row.get("final_release_id", "")),
        "candidate_id": str(row.get("candidate_id", "")),
        "target_name": target,
        "design_seq_persisted": marked,
        "design_seq_replayed": replay_marked,
        "design_natural_seq": natural,
        "peptide_length": length,
        "replay_methyl_positions_1based": json.dumps(methyl_positions(replay_marked)),
        "replay_threshold_disagreement_positions_1based": json.dumps(replay_disagreements),
        "replay_methyl_probability_mean": json.dumps(replay_mean),
        "replay_methyl_probability_representation_min": json.dumps(replay_min),
        "replay_methyl_probability_representation_max": json.dumps(replay_max),
        "replay_methyl_probability_representation_span": json.dumps(replay_span),
        "replay_methyl_probability_representation_std": json.dumps(replay_rep_std),
        "replay_methyl_probability_order_std": json.dumps(replay_order_std),
        "replay_methyl_probability_representation_by_start": json.dumps(replay_matrix),
        "methyl_vector_max_abs_delta": max(
            maxima_by_group.get(field, 0.0) for field in persisted_vectors
        ),
        "methyl_matrix_max_abs_delta": matrix_delta,
        "replay_cyclic_base_log_probability_start_by_decoder_order": json.dumps(
            replay_base_matrix
        ),
        "replay_cyclic_base_log_probability_by_start": json.dumps(replay_base_by_start),
        "replay_cyclic_base_log_probability_mean": base[
            "cyclic_base_log_probability_mean"
        ],
        "replay_cyclic_base_log_probability_min": base[
            "cyclic_base_log_probability_min"
        ],
        "replay_cyclic_base_log_probability_max": base[
            "cyclic_base_log_probability_max"
        ],
        "replay_cyclic_base_log_probability_span": base[
            "cyclic_base_log_probability_span"
        ],
        "replay_cyclic_base_log_probability_std": base[
            "cyclic_base_log_probability_std"
        ],
        "replay_cyclic_base_floor": frozen_floor,
        "replay_cyclic_base_gate_pass": int(replay_base_gate),
        "cyclic_base_matrix_max_abs_delta": base_matrix_delta,
        "cyclic_base_summary_max_abs_delta": max(
            maxima_by_group.get(field, 0.0) for field in base_scalar_fields
        ),
        "row_replay_status": "PASS" if not errors else "FAIL",
        "row_replay_problems": ";".join(errors),
    }


def make_input_record(path: Path) -> Dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-manifest", required=True)
    parser.add_argument("--detailed-csv", required=True)
    parser.add_argument("--selector-concise")
    parser.add_argument("--selector-fasta")
    parser.add_argument("--model", required=True)
    parser.add_argument("--heldout-audit", required=True)
    parser.add_argument("--scorer-manifest", required=True)
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--best-csv", default=str(DEFAULT_BEST))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selector_manifest_path = Path(args.selector_manifest).resolve()
    detailed_path = Path(args.detailed_csv).resolve()
    concise_path = (
        Path(args.selector_concise).resolve()
        if args.selector_concise
        else detailed_path.with_name(FINAL_CONCISE)
    )
    fasta_path = (
        Path(args.selector_fasta).resolve()
        if args.selector_fasta
        else detailed_path.with_name(FINAL_FASTA)
    )
    model_path = Path(args.model).resolve()
    audit_path = Path(args.heldout_audit).resolve()
    scorer_manifest_path = Path(args.scorer_manifest).resolve()
    native_path = Path(args.native_jsonl).resolve()
    best_path = Path(args.best_csv).resolve()
    plan_path = Path(args.plan).resolve()
    out_dir = Path(args.out_dir).resolve()
    required_paths = (
        selector_manifest_path,
        detailed_path,
        concise_path,
        fasta_path,
        model_path,
        audit_path,
        scorer_manifest_path,
        native_path,
        best_path,
        plan_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(out_dir == path.parent and path.name in KNOWN_OUTPUTS for path in required_paths):
        raise ValueError("Replay output directory must be separate from all source artifacts")
    prepare_output(out_dir, args.overwrite)

    selector_manifest = read_json(selector_manifest_path)
    scorer_manifest = read_json(scorer_manifest_path)
    audit = read_json(audit_path)
    plan = read_json(plan_path)
    detailed_rows = read_csv(detailed_path)
    upstream_checks = validate_upstream_hash_contract(
        selector_manifest,
        scorer_manifest,
        audit,
        plan,
        selector_manifest_path,
        detailed_path,
        concise_path,
        fasta_path,
        model_path,
        audit_path,
        scorer_manifest_path,
        plan_path,
        native_path,
        best_path,
    )
    upstream_checks.update(exact_target_quota_checks(detailed_rows, FROZEN_TARGETS))
    upstream_checks.update(verify_selector_views(detailed_rows, concise_path, fasta_path))
    floors = target_floors(scorer_manifest)
    upstream_checks["scorer_manifest_has_one_finite_floor_for_each_frozen_target"] = (
        set(floors) == set(FROZEN_TARGETS)
        and all(math.isfinite(value) for value in floors.values())
    )
    final_ids = [str(row.get("final_release_id", "")) for row in detailed_rows]
    candidate_ids = [str(row.get("candidate_id", "")) for row in detailed_rows]
    upstream_checks["selected_identifiers_are_nonempty_and_unique"] = (
        all(final_ids)
        and all(candidate_ids)
        and len(set(final_ids)) == EXPECTED_TOTAL
        and len(set(candidate_ids)) == EXPECTED_TOTAL
    )
    if not all(upstream_checks.values()):
        failed = [name for name, passed in upstream_checks.items() if not passed]
        raise RuntimeError("Replay upstream contract failed: " + ", ".join(failed))

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("The independent replay requires PyTorch") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CPU replay requires explicit --allow-cpu")
    device = torch.device(args.device)

    checkpoint = torch.load(model_path, map_location="cpu")
    metadata = (
        dict(checkpoint.get("expert_head_qc_metadata", {}))
        if isinstance(checkpoint, Mapping)
        else {}
    )
    del checkpoint
    checkpoint_check = (
        metadata.get("protocol") == EXPERT_PROTOCOL
        and float(metadata.get("worst_start_bce_weight", 0.0)) > 0.0
        and float(metadata.get("representation_consistency_weight", 0.0)) > 0.0
        and bool(metadata.get("full_physical_start_by_full_decoder_order_grid"))
        and float(metadata.get("training_ensemble_temperature", -1.0)) == TEMPERATURE
        and "full_physical_start_x_full_decoder_order_grid"
        in str(metadata.get("training_objective", ""))
    )
    if not checkpoint_check:
        raise RuntimeError("Replay checkpoint is not a promoted V9 full-grid checkpoint")

    generator = load_module("v9_replay_generator", GENERATOR_PATH)
    scorer_module = load_module("v9_replay_exact_cyclic_base", SCORER_PATH)
    clean_dir = REPO_ROOT / "paper_clean_v28"
    if str(clean_dir) not in sys.path:
        sys.path.insert(0, str(clean_dir))
    from clean_v28_common import (  # pylint: disable=import-error,import-outside-toplevel
        NATURAL_AA_ALPHABET,
        complete_decoding_order,
        cyclic_representation_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
        peptide_only_annotation_tensors,
    )
    if str(NATURAL_AA_ALPHABET) != NATURAL_AA:
        raise RuntimeError("Natural amino-acid alphabet differs from the frozen V9 order")

    best_rows = generator.read_csv(best_path)
    selected_chains = generator.selected_chain_index(best_rows)
    native_rows = generator.read_jsonl(native_path)
    target_records, _target_manifest = generator.prepare_target_records(
        native_rows, selected_chains, list(FROZEN_TARGETS)
    )
    model = load_v28_model(str(model_path), device)
    model.eval()
    methyl_replay = MethylReplay(
        model,
        device,
        target_records,
        torch,
        featurize_records,
        peptide_only_annotation_tensors,
        cyclic_representation_known_sequence_methyl_probabilities,
        str(NATURAL_AA_ALPHABET),
    )
    base_scorer = scorer_module.ExactCyclicBaseScorer(
        model,
        device,
        target_records,
        torch,
        functional,
        featurize_records,
        complete_decoding_order,
        1,
    )

    replay_rows: List[Dict[str, Any]] = []
    print("===== V9 INDEPENDENT BATCH-1 REPLAY START =====", flush=True)
    print(f"Rows: {len(detailed_rows)}; device: {device}", flush=True)
    for index, source in enumerate(detailed_rows, start=1):
        replay_rows.append(
            replay_and_compare_row(
                source,
                methyl_replay,
                base_scorer,
                floors[str(source.get("target_name", "")).upper()],
            )
        )
        if index == 1 or index % 25 == 0 or index == len(detailed_rows):
            failed_so_far = sum(
                row["row_replay_status"] != "PASS" for row in replay_rows
            )
            print(
                f"Replayed {index}/{len(detailed_rows)}; failures={failed_so_far}",
                flush=True,
            )

    row_failures = [row for row in replay_rows if row["row_replay_status"] != "PASS"]
    replay_checks = {
        **upstream_checks,
        "checkpoint_is_promoted_v9_full_grid_model": checkpoint_check,
        "every_selected_row_was_replayed_batch_size_one": len(replay_rows) == EXPECTED_TOTAL,
        "every_methyl_and_cyclic_base_replay_row_passes": not row_failures,
        "replay_target_quota_remains_exact_17_x_100": all(
            exact_target_quota_checks(replay_rows, FROZEN_TARGETS).values()
        ),
    }
    quality_gate = "PASS" if all(replay_checks.values()) else "FAIL"
    inputs = {
        "selector_manifest": make_input_record(selector_manifest_path),
        "selector_detailed": make_input_record(detailed_path),
        "selector_concise": make_input_record(concise_path),
        "selector_fasta": make_input_record(fasta_path),
        "model": make_input_record(model_path),
        "heldout_audit": make_input_record(audit_path),
        "scorer_manifest": make_input_record(scorer_manifest_path),
        "plan": make_input_record(plan_path),
        "native_jsonl": make_input_record(native_path),
        "best_csv": make_input_record(best_path),
    }
    maximum_methyl_vector_delta = max(
        (float(row["methyl_vector_max_abs_delta"]) for row in replay_rows), default=0.0
    )
    maximum_methyl_matrix_delta = max(
        (float(row["methyl_matrix_max_abs_delta"]) for row in replay_rows), default=0.0
    )
    maximum_base_matrix_delta = max(
        (float(row["cyclic_base_matrix_max_abs_delta"]) for row in replay_rows), default=0.0
    )
    diagnostics = {
        "quality_gate": quality_gate,
        "release_status": (
            "AUTHORIZED_AFTER_INDEPENDENT_BATCH1_MODEL_REPLAY"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_SEND_TO_SHANGGE"
        ),
        "protocol": REPLAY_PROTOCOL,
        "batch_size": 1,
        "temperature": TEMPERATURE,
        "methyl_threshold": THRESHOLD,
        "probability_absolute_tolerance": PROBABILITY_ATOL,
        "base_score_absolute_tolerance": BASE_SCORE_ATOL,
        "threshold_contract": "strict_round8_gt_0.6_and_zero_min_max_disagreement",
        "quality_checks": replay_checks,
        "input_rows": len(detailed_rows),
        "replayed_rows": len(replay_rows),
        "failed_rows": len(row_failures),
        "maximum_methyl_vector_absolute_delta": maximum_methyl_vector_delta,
        "maximum_methyl_matrix_absolute_delta": maximum_methyl_matrix_delta,
        "maximum_cyclic_base_matrix_absolute_delta": maximum_base_matrix_delta,
        "failed_examples_first_100": row_failures[:100],
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "dependencies": {
            "generator_module": {
                "path": str(GENERATOR_PATH),
                "sha256": sha256_file(GENERATOR_PATH),
            },
            "cyclic_base_scorer_module": {
                "path": str(SCORER_PATH),
                "sha256": sha256_file(SCORER_PATH),
            },
        },
        "inputs": inputs,
    }
    if quality_gate != "PASS":
        # Diagnostic names cannot be mistaken for a structure handoff.
        atomic_write_csv(out_dir / BLOCKED_CSV, replay_rows, union_fields(replay_rows))
        atomic_write_json(out_dir / BLOCKED_JSON, diagnostics)
        raise RuntimeError(
            f"Independent replay blocked the handoff; failed rows: {len(row_failures)}"
        )

    atomic_copy(detailed_path, out_dir / FINAL_DETAIL)
    atomic_copy(concise_path, out_dir / FINAL_CONCISE)
    atomic_copy(fasta_path, out_dir / FINAL_FASTA)
    atomic_write_csv(out_dir / REPLAY_CSV, replay_rows, union_fields(replay_rows))
    copied_checks = {
        "copied_detailed_is_byte_identical": (
            sha256_file(out_dir / FINAL_DETAIL) == sha256_file(detailed_path)
        ),
        "copied_concise_is_byte_identical": (
            sha256_file(out_dir / FINAL_CONCISE) == sha256_file(concise_path)
        ),
        "copied_fasta_is_byte_identical": (
            sha256_file(out_dir / FINAL_FASTA) == sha256_file(fasta_path)
        ),
        "reopened_copied_views_still_match": all(
            verify_selector_views(
                read_csv(out_dir / FINAL_DETAIL),
                out_dir / FINAL_CONCISE,
                out_dir / FINAL_FASTA,
            ).values()
        ),
        "reopened_replay_csv_has_1700_pass_rows": (
            len(read_csv(out_dir / REPLAY_CSV)) == EXPECTED_TOTAL
            and all(
                row.get("row_replay_status") == "PASS"
                for row in read_csv(out_dir / REPLAY_CSV)
            )
        ),
    }
    if not all(copied_checks.values()):
        for name in (FINAL_DETAIL, FINAL_CONCISE, FINAL_FASTA, REPLAY_CSV):
            path = out_dir / name
            if path.exists():
                path.unlink()
        raise RuntimeError("Reopened replay handoff verification failed")
    diagnostics["quality_checks"].update(copied_checks)
    diagnostics["release_artifacts"] = {
        "detailed": make_input_record(out_dir / FINAL_DETAIL),
        "concise": make_input_record(out_dir / FINAL_CONCISE),
        "fasta": make_input_record(out_dir / FINAL_FASTA),
        "replay_csv": make_input_record(out_dir / REPLAY_CSV),
    }
    atomic_write_json(out_dir / REPLAY_MANIFEST, diagnostics)
    print("===== V9 INDEPENDENT BATCH-1 REPLAY PASS =====", flush=True)
    print(f"Final rows: {EXPECTED_TOTAL} (17 x 100)", flush=True)
    print(f"Final handoff directory: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
