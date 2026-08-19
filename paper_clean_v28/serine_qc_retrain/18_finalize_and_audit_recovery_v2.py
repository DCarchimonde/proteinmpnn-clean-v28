#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent three-pass final audit for cyclic-base V8 recovery V2.

The immutable 31,500-row baseline is copied into a separate overlay and every
V2 directed candidate is independently re-scored at batch size one.  The
cyclic-start ProteinMPNN floor is independently recomputed from all 3ZGC
baseline sequences using a fresh scorer instance.  Integrity/rescore,
physical-position/representation, and novelty/workflow passes must all pass.
No structure handoff or permeability input is created here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
SEARCH_V2_PATH = SCRIPT_PATH.with_name("17_cyclic_base_recovery_v2.py")
FRONTIER_V3_PATH = SCRIPT_PATH.with_name("20_full_frontier_recovery_v3.py")
LEGACY_SEARCH_PATH = SCRIPT_PATH.with_name("14_directed_recovery_search_v8.py")
LEGACY_FINALIZER_PATH = SCRIPT_PATH.with_name("15_finalize_and_audit_recovery_v8.py")
V7_AUDITOR_PATH = SCRIPT_PATH.with_name("11_triple_audit_serine_only_v7.py")
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"
DEFAULT_MODEL = V8_ROOT / "model" / "frankenstein_v28_source_scoped_hybrid_v8.pt"
DEFAULT_MODEL_MANIFEST = V8_ROOT / "model" / "expert_source_composition_manifest.json"
DEFAULT_REPRESENTATION = V8_ROOT / "representation_audit" / "cyclic_representation_audit.json"
DEFAULT_BASELINE = V8_ROOT / "generation_baseline"
DEFAULT_SEARCH = V8_ROOT / "directed_search_cyclic_base_v2"
DEFAULT_OUT = V8_ROOT / "generation_recovered_cyclic_base_v2"
DEFAULT_AUDIT_OUT = V8_ROOT / "triple_audit_recovered_cyclic_base_v2"
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

V2_FINAL_PROTOCOL = "immutable_baseline_plus_cyclic_base_recovery_overlay_v8_v2"
V2_AUDIT_PROTOCOL = "independent_three_pass_cyclic_base_recovery_v8_v2"
V3_SEARCH_PROTOCOL = "full_legacy_frontier_cyclic_base_recovery_v8_v3"
V3_FINAL_PROTOCOL = (
    "immutable_baseline_plus_full_frontier_recovery_overlay_v8_v3"
)
V3_AUDIT_PROTOCOL = "independent_three_pass_full_frontier_recovery_v8_v3"
EXPECTED_BASELINE_ROWS = 31_500
EXPECTED_TARGETS = 17
THRESHOLD = 0.6
RESCORE_TOLERANCE = 2e-6


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


def artifact(path: Path) -> Dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def artifact_leaves(value: Any) -> List[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if set(value) >= {"path", "sha256"}:
            return [value]
        leaves: List[Mapping[str, Any]] = []
        for child in value.values():
            leaves.extend(artifact_leaves(child))
        return leaves
    if isinstance(value, list):
        leaves = []
        for child in value:
            leaves.extend(artifact_leaves(child))
        return leaves
    return []


def validate_artifacts_under(value: Any, root: Path) -> None:
    leaves = artifact_leaves(value)
    if not leaves:
        raise RuntimeError("V2 manifest has no artifact leaves")
    for leaf in leaves:
        path = Path(str(leaf.get("path", ""))).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"V2 artifact escapes search directory: {path}") from exc
        if not path.is_file() or sha256_file(path) != str(leaf.get("sha256", "")):
            raise RuntimeError(f"V2 artifact is absent or stale: {path}")


def augment_baseline_row(source: Mapping[str, Any]) -> Dict[str, Any]:
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
    return row


def point_contract_errors(
    search_v2: Any, row: Mapping[str, Any], evidence: Mapping[str, Any]
) -> List[str]:
    errors: List[str] = []
    row_id = str(row.get("candidate_id", ""))
    sequence = str(row.get("design_natural_seq", "")).upper()
    try:
        persisted = [
            float(value) for value in json.loads(str(row["methyl_probabilities"]))
        ]
        evidence_vector = [
            float(value)
            for value in json.loads(str(evidence["physical_probability_vector"]))
        ]
        summary = search_v2.physical_argmax_summary(sequence, persisted)
        methyl_positions = [
            index
            for index, token in enumerate(str(row["design_seq"]), start=1)
            if token.islower()
        ]
        if not (
            len(persisted) == len(evidence_vector) == len(sequence)
            and all(
                abs(left - right) <= RESCORE_TOLERANCE
                for left, right in zip(persisted, evidence_vector)
            )
            and int(summary["physical_argmax_position_1based"])
            == int(evidence["physical_argmax_position_1based"])
            and str(summary["physical_argmax_residue"])
            == str(evidence["physical_argmax_residue"])
            and json.loads(str(evidence["predicted_methyl_positions_1based"]))
            == methyl_positions
            and int(evidence["annotation_representation_ensemble_size"])
            == len(sequence)
            and int(evidence["annotation_decoder_order_ensemble_size"])
            == len(sequence)
        ):
            errors.append(f"{row_id}: physical-position evidence mismatch")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        errors.append(f"{row_id}: malformed physical-position evidence")
    return errors


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("V8 V2 final audit requires NumPy and PyTorch") from exc
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    search_v2 = load_module("v8_search_v2_for_final_audit", SEARCH_V2_PATH)
    old = load_module("v8_legacy_search_for_v2_final_audit", LEGACY_SEARCH_PATH)
    legacy_final = load_module(
        "v8_legacy_finalizer_helpers_for_v2", LEGACY_FINALIZER_PATH
    )
    v7_auditor = load_module("v7_position_auditor_for_v2", V7_AUDITOR_PATH)

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
    immutable = (
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
        SEARCH_V2_PATH,
        LEGACY_SEARCH_PATH,
        LEGACY_FINALIZER_PATH,
        V7_AUDITOR_PATH,
    )
    for writable in (out_dir, audit_out):
        if any(old.paths_overlap(writable, path) for path in immutable):
            raise ValueError("V2 final output overlaps an immutable input")
    if old.paths_overlap(out_dir, audit_out):
        raise ValueError("V2 generation and audit output directories overlap")
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
        search_dir / "cyclic_base_recovery_manifest.json",
        search_dir / "directed_candidates.csv",
        search_dir / "cyclic_base_plausibility_and_position_evidence.csv",
        search_dir / "mandatory_length_6_7_controls.csv",
        plan_path,
        native_path,
        historical_path,
        prior_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (out_dir.exists() and any(out_dir.iterdir())) or (
        audit_out.exists() and any(audit_out.iterdir())
    ):
        if not args.overwrite:
            raise FileExistsError("V2 final output exists; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_out.mkdir(parents=True, exist_ok=True)

    baseline_manifest, baseline_unique, baseline_target_rows = old.validate_baseline(
        baseline,
        model_path,
        model_manifest_path,
        representation_path,
        plan_path,
        native_path,
        historical_path,
        prior_path,
    )
    search_manifest = search_v2.read_json(
        search_dir / "cyclic_base_recovery_manifest.json"
    )
    search_protocol = str(search_manifest.get("protocol", ""))
    if search_protocol == search_v2.V2_SEARCH_PROTOCOL:
        version_label = "V2"
        final_protocol = V2_FINAL_PROTOCOL
        audit_protocol = V2_AUDIT_PROTOCOL
        frontier_contract_ok = True
    elif search_protocol == V3_SEARCH_PROTOCOL:
        version_label = "V3 FULL FRONTIER"
        final_protocol = V3_FINAL_PROTOCOL
        audit_protocol = V3_AUDIT_PROTOCOL
        frontier_contract_ok = (
            FRONTIER_V3_PATH.is_file()
            and dict(search_manifest.get("config") or {}).get(
                "frontier_v3_program_sha256"
            )
            == sha256_file(FRONTIER_V3_PATH)
            and dict(search_manifest.get("config") or {}).get(
                "surrogate_release_authority"
            )
            == "NONE_ACQUISITION_ONLY"
            and dict(search_manifest.get("quality_checks") or {}).get(
                "surrogate_is_acquisition_only_and_never_a_release_gate"
            )
            is True
        )
    else:
        raise RuntimeError(f"Unsupported V8 recovery protocol: {search_protocol}")
    search_config = dict(search_manifest.get("config") or {})
    serine_provenance = dict(search_manifest.get("serine_provenance_gate") or {})
    if not (
        search_manifest.get("quality_gate") == "PASS"
        and frontier_contract_ok
        and search_manifest.get("config_sha256")
        == search_v2.stable_json_sha256(search_config)
        and search_manifest.get("model_sha256") == sha256_file(model_path)
        and search_manifest.get("baseline_manifest_sha256")
        == sha256_file(baseline / "generation_manifest.json")
        and int(search_manifest.get("released_candidates", 0)) > 0
        and search_manifest.get("missing_targets_after_search") == []
        and search_manifest.get("targets_formally_abstained") == []
        and search_config.get("v2_search_program_sha256")
        == sha256_file(SEARCH_V2_PATH)
        and search_config.get("legacy_search_program_sha256")
        == sha256_file(LEGACY_SEARCH_PATH)
        and float(search_config.get("threshold", -1.0)) == THRESHOLD
        and search_config.get("strict_operator") == ">"
        and search_config.get("base_policy") == search_v2.V2_BASE_POLICY
        and int(search_config.get("physical_representation_count", -1)) == 7
        and int(search_config.get("decoder_orders_per_representation", -1)) == 7
        and serine_provenance.get("quality_gate") == "PASS"
        and serine_provenance.get("serine_expert_source") == "v7_serine"
        and serine_provenance.get("non_ser_expert_source") == "v6_non_ser"
        and dict(serine_provenance.get("expert_source_by_residue") or {}).get("S")
        == "v7_serine"
    ):
        raise RuntimeError(
            f"V8 {version_label} search is failed, stale, or uses the wrong contract"
        )
    validate_artifacts_under(search_manifest.get("artifacts"), search_dir)

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
    baseline_raw = search_v2.read_csv(baseline / "all_candidates.csv")
    baseline_eligible = search_v2.read_csv(
        baseline / "methylated_new_candidates.csv"
    )
    directed_rows = search_v2.read_csv(search_dir / "directed_candidates.csv")
    evidence_rows = search_v2.read_csv(
        search_dir / "cyclic_base_plausibility_and_position_evidence.csv"
    )
    controls = search_v2.read_csv(search_dir / "mandatory_length_6_7_controls.csv")
    plan = search_v2.read_json(plan_path)
    target_names = [str(row["target_name"]).upper() for row in plan["targets"]]
    plan_by_target = {
        str(row["target_name"]).upper(): row for row in plan["targets"]
    }
    selected_chains = {
        str(row["target_name"]).upper(): str(row["selected_chain"])
        for row in baseline_target_rows
    }
    if len(baseline_raw) != EXPECTED_BASELINE_ROWS or len(target_names) != EXPECTED_TARGETS:
        raise RuntimeError("Immutable baseline row count or target count changed")
    if not directed_rows or any(
        str(row.get("target_name", "")).upper() != "3ZGC" for row in directed_rows
    ):
        raise RuntimeError("V2 final audit expects one or more 3ZGC directed candidates")
    if len(directed_rows) != int(search_manifest["released_candidates"]):
        raise RuntimeError("V2 directed candidate count does not match its manifest")

    evidence_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    duplicate_evidence: List[str] = []
    for evidence in evidence_rows:
        key = (
            str(evidence.get("target_name", "")).upper(),
            str(evidence.get("sequence", "")).upper(),
        )
        if not all(key) or key in evidence_by_key:
            duplicate_evidence.append(":".join(key))
        evidence_by_key[key] = evidence
    release_evidence_keys = {
        key
        for key, row in evidence_by_key.items()
        if int(row.get("release_eligible", 0)) == 1
    }
    directed_keys = {
        (
            str(row["target_name"]).upper(),
            str(row["design_natural_seq"]).upper(),
        )
        for row in directed_rows
    }

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but unavailable")
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
            raise RuntimeError("No CUDA device is available")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    reannotator = old.load_module(
        "v8_reannotator_for_v2_independent_final", old.REANNOTATOR_PATH
    )
    generator = old.load_module(
        "v8_generator_for_v2_independent_final", old.GENERATOR_PATH
    )
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
    methyl_rescorer = old.MethylScorer(
        model, device, native_index, selected_chains, 1, torch, common, reannotator
    )
    target_records, _ = generator.prepare_target_records(
        native_rows, selected_chains, sorted(old.ALLOWED_RECOVERY_TARGETS)
    )
    base_rescorer = search_v2.CyclicBasePlausibilityScorer(
        model,
        device,
        target_records,
        32,
        torch,
        functional,
        common,
        old.ProgressBar,
    )

    target_pool = sorted(
        {
            str(row["design_natural_seq"]).upper()
            for row in baseline_unique
            if str(row["target_name"]).upper() == "3ZGC"
        }
    )
    independent_pool_base = base_rescorer.score_detailed(
        "3ZGC", target_pool, "independent V2 final baseline cyclic floor"
    )
    independent_floor = old.nearest_rank_percentile(
        [
            float(row["cyclic_base_log_probability_mean"])
            for row in independent_pool_base.values()
        ],
        search_v2.BASE_PERCENTILE,
    )
    directed_sequences = sorted(
        str(row["design_natural_seq"]).upper() for row in directed_rows
    )
    independent_directed_base = base_rescorer.score_detailed(
        "3ZGC", directed_sequences, "independent V2 final candidate cyclic base"
    )
    annotation_errors: List[str] = []
    rescore_errors: List[str] = []
    base_errors: List[str] = []
    point_errors: List[str] = []
    for row in baseline_eligible:
        annotation_errors.extend(legacy_final.validate_annotation_row(row))
        annotation_errors.extend(legacy_final.validate_eligible_candidate_row(row, 1))
    progress = old.ProgressBar(
        "3ZGC independent V2 final batch-one replay",
        len(directed_rows),
        unit="candidate",
    )
    for row in directed_rows:
        progress.update(1)
        row_id = str(row.get("candidate_id", ""))
        sequence = str(row["design_natural_seq"]).upper()
        key = ("3ZGC", sequence)
        evidence = evidence_by_key.get(key)
        annotation_errors.extend(legacy_final.validate_annotation_row(row))
        annotation_errors.extend(legacy_final.validate_eligible_candidate_row(row, 0))
        if evidence is None:
            rescore_errors.append(f"{row_id}: missing V2 evidence row")
            continue
        point_errors.extend(point_contract_errors(search_v2, row, evidence))
        recomputed = methyl_rescorer.score_full(
            "3ZGC",
            [sequence],
            stage="independent V2 final batch-one replay",
            show_progress=False,
        )[sequence]
        persisted = [
            float(value) for value in json.loads(str(row["methyl_probabilities"]))
        ]
        observed = [
            float(value)
            for value in json.loads(str(recomputed["methyl_probabilities"]))
        ]
        observed_summary = search_v2.physical_argmax_summary(sequence, observed)
        observed_maximum = float(observed_summary["physical_argmax_probability"])
        if not (
            str(recomputed["design_seq"]) == str(row["design_seq"])
            and len(persisted) == len(observed) == len(sequence)
            and all(
                abs(left - right) <= RESCORE_TOLERANCE
                for left, right in zip(persisted, observed)
            )
            and old.strict_rounded_pass(observed_maximum)
            and abs(
                observed_maximum - float(row["batch_one_maximum_probability"])
            )
            <= RESCORE_TOLERANCE
        ):
            rescore_errors.append(f"{row_id}: independent methyl replay mismatch")
        independent_base = independent_directed_base[sequence]
        try:
            evidence_base = float(evidence["cyclic_base_log_probability_mean"])
            row_base = float(row["cyclic_base_log_probability_mean"])
            if not (
                abs(independent_floor - float(search_manifest["cyclic_base_floor_1pct"]))
                <= RESCORE_TOLERANCE
                and abs(
                    float(independent_base["cyclic_base_log_probability_mean"])
                    - evidence_base
                )
                <= RESCORE_TOLERANCE
                and abs(row_base - evidence_base) <= RESCORE_TOLERANCE
                and float(independent_base["cyclic_base_log_probability_mean"])
                >= independent_floor
                and int(independent_base["cyclic_base_total_ensemble_size"])
                == len(sequence) ** 2
                and independent_base["cyclic_base_context_policy"]
                == search_v2.V2_BASE_POLICY
            ):
                base_errors.append(f"{row_id}: independent cyclic-base replay mismatch")
        except (KeyError, TypeError, ValueError):
            base_errors.append(f"{row_id}: malformed cyclic-base evidence")
    progress.close()

    historical_rows = search_v2.read_csv(historical_path)
    prior_rows = search_v2.read_csv(prior_path)
    novelty = search_v2.exclusion_sets(
        old, historical_rows, prior_rows, baseline_unique, "3ZGC"
    )
    novelty_errors = [
        str(row["candidate_id"])
        for row in directed_rows
        if search_v2.duplicate_reason(
            old, str(row["design_natural_seq"]).upper(), novelty
        )
    ]
    directed_cyclic = [
        old.forward_cyclic_identity(str(row["design_natural_seq"]).upper())
        for row in directed_rows
    ]
    if len(directed_cyclic) != len(set(directed_cyclic)):
        novelty_errors.append("directed candidates contain forward-cyclic duplicates")

    baseline_augmented = [augment_baseline_row(row) for row in baseline_eligible]
    final_rows = [*baseline_augmented, *(dict(row) for row in directed_rows)]
    final_rows.sort(
        key=lambda row: (
            str(row["target_name"]),
            str(row["design_natural_seq"]),
            str(row["design_seq"]),
            str(row["candidate_id"]),
        )
    )
    merged_all = [augment_baseline_row(row) for row in baseline_raw]
    merged_all.extend(dict(row) for row in directed_rows)
    merged_unique = [augment_baseline_row(row) for row in baseline_unique]
    merged_unique.extend(dict(row) for row in directed_rows)
    candidate_ids = [str(row.get("candidate_id", "")) for row in final_rows]
    final_keys = [
        (str(row["target_name"]).upper(), str(row["design_seq"])) for row in final_rows
    ]
    counts = Counter(target for target, _sequence in final_keys)
    uncovered = sorted(target for target in target_names if counts[target] < 1)

    projected_raw = [
        legacy_final.project_preserved_source_fields(row, baseline_raw[0])
        for row in merged_all[: len(baseline_raw)]
    ]
    projected_unique = [
        legacy_final.project_preserved_source_fields(row, baseline_unique[0])
        for row in merged_unique[: len(baseline_unique)]
    ]
    baseline_preserved = (
        legacy_final.canonical_rows_sha256(projected_raw)
        == legacy_final.canonical_rows_sha256(baseline_raw)
        and legacy_final.canonical_rows_sha256(projected_unique)
        == legacy_final.canonical_rows_sha256(baseline_unique)
    )

    concentration_rows, av_alignment, concentration_checks = (
        legacy_final.concentration_audit(
            final_rows,
            baseline_eligible,
            native_rows,
            selected_chains,
            v7_auditor,
        )
    )
    summary_rows: List[Dict[str, Any]] = []
    target_manifest_rows: List[Dict[str, Any]] = []
    for target in target_names:
        baseline_count = sum(
            str(row["target_name"]).upper() == target for row in baseline_eligible
        )
        directed_count = sum(
            str(row["target_name"]).upper() == target for row in directed_rows
        )
        quota = int(plan_by_target[target]["structure_quota"])
        total = baseline_count + directed_count
        summary_rows.append(
            {
                "target_name": target,
                "baseline_novel_methylated_candidates": baseline_count,
                "directed_recovery_candidates": directed_count,
                "final_signature_candidates": total,
                "has_signature_candidate": int(total > 0),
                "planned_structure_quota": quota,
                "meets_planned_structure_quota": int(total >= quota),
                "structure_status": "NOT_PREDICTED",
            }
        )
        target_manifest_rows.append(
            {
                "target_name": target,
                "selected_chain": selected_chains[target],
                "baseline_raw_rows_retained": sum(
                    str(row["target_name"]).upper() == target for row in baseline_raw
                ),
                "baseline_unique_rows_retained": sum(
                    str(row["target_name"]).upper() == target for row in baseline_unique
                ),
                "directed_rows_appended": directed_count,
                "final_signature_candidates": total,
                "target_release_status": (
                    "CANDIDATE_FOUND_PENDING_MANUAL_AND_STRUCTURE_REVIEW"
                    if total
                    else "BLOCKED_NO_SIGNATURE_CANDIDATE"
                ),
                "formal_abstention": 0,
                "structure_status": "NOT_PREDICTED",
            }
        )

    controls_exact = {
        (
            str(row.get("target_name", "")).upper(),
            str(row.get("control_type", "")),
            str(row.get("natural_sequence", "")).upper(),
            int(row.get("length", -1)),
        )
        for row in controls
    } == {
        ("3WNE", "withdrawn_historical", "GRKWNC", 6),
        ("3WNE", "native", "PKIDNG", 6),
        ("3ZGC", "withdrawn_historical", "REGGQNR", 7),
        ("3ZGC", "native", "GDEETGE", 7),
    }
    pass_1_checks = {
        "model_representation_baseline_search_and_artifacts_are_hash_pinned": True,
        "baseline_has_exactly_31500_rows": len(baseline_raw) == EXPECTED_BASELINE_ROWS,
        "overlay_preserves_every_baseline_raw_and_unique_field": baseline_preserved,
        "search_release_evidence_set_is_exact": (
            not duplicate_evidence and release_evidence_keys == directed_keys
        ),
        "baseline_and_directed_annotation_contracts_pass": not annotation_errors,
        "directed_candidates_independently_batch_one_rescore": not rescore_errors,
        "cyclic_base_floor_and_candidates_independently_recompute": not base_errors,
        "candidate_ids_and_target_design_keys_are_unique": (
            all(candidate_ids)
            and len(candidate_ids) == len(set(candidate_ids))
            and len(final_keys) == len(set(final_keys))
        ),
        "mandatory_length_6_7_controls_are_exact": controls_exact,
    }
    pass_2_checks = {
        **concentration_checks,
        "directed_physical_probability_vectors_argmax_and_sites_match": not point_errors,
        "directed_rows_have_no_fabricated_sampling_order": all(
            not str(row.get("decoding_order_absolute", "")) for row in directed_rows
        ),
        "3zgc_position_diagnostics_are_present": any(
            str(row.get("target_name", "")).upper() == "3ZGC"
            for row in concentration_rows
        ),
    }
    pass_3_checks = {
        "all_17_targets_have_a_signature_candidate": not uncovered,
        "all_directed_candidates_are_exact_and_forward_cyclic_novel": (
            not novelty_errors
        ),
        "formal_abstention_is_absent": (
            not baseline_manifest.get("targets_formally_abstained")
            and not search_manifest.get("targets_formally_abstained")
        ),
        "structure_handoff_is_not_created": not any(
            path.exists()
            for path in (
                V8_ROOT / "handoff",
                V8_ROOT / "serine_qc_source_scoped_hybrid_v8_shangge_handoff.zip",
                V8_ROOT / "serine_qc_source_scoped_hybrid_v8_v2_shangge_handoff.zip",
            )
        ),
        "permeability_remains_deferred": all(
            int(row.get("eligible_for_new_permeability_screen", -1)) == 0
            and int(row.get("permeability_screen_authorized_in_this_release", -1))
            == 0
            and str(row.get("permeability_eligibility_status", ""))
            == "DEFERRED_PENDING_GLOBAL_AND_CYCLIC_RMSD_LT_3A"
            for row in [*final_rows, *merged_all, *merged_unique]
        ),
    }
    pass_1 = "PASS" if all(pass_1_checks.values()) else "FAIL"
    pass_2 = "PASS" if all(pass_2_checks.values()) else "FAIL"
    pass_3 = "PASS" if all(pass_3_checks.values()) else "FAIL"
    quality_gate = "PASS" if pass_1 == pass_2 == pass_3 == "PASS" else "FAIL"

    final_all_path = out_dir / "all_candidates.csv"
    final_unique_path = out_dir / "unique_candidates.csv"
    final_candidates_path = out_dir / "methylated_new_candidates.csv"
    target_manifest_path = out_dir / "target_manifest.csv"
    summary_path = out_dir / "generation_summary_by_target.csv"
    legacy_final.atomic_write_csv(
        final_all_path, merged_all, legacy_final.union_fields(merged_all)
    )
    legacy_final.atomic_write_csv(
        final_unique_path, merged_unique, legacy_final.union_fields(merged_unique)
    )
    legacy_final.atomic_write_csv(
        final_candidates_path, final_rows, legacy_final.union_fields(final_rows)
    )
    legacy_final.atomic_write_csv(
        target_manifest_path, target_manifest_rows, list(target_manifest_rows[0])
    )
    legacy_final.atomic_write_csv(summary_path, summary_rows, list(summary_rows[0]))
    concentration_path = audit_out / "three_pass_concentration_by_target.csv"
    av_path = audit_out / "av_family_physical_position_support.json"
    legacy_final.atomic_write_csv(
        concentration_path, concentration_rows, list(concentration_rows[0])
    )
    legacy_final.atomic_write_json(av_path, av_alignment)

    final_manifest = {
        "quality_gate": quality_gate,
        "release_status": (
            "READY_FOR_MANUAL_SCIENTIFIC_REVIEW_NO_STRUCTURE_HANDOFF"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_SEND_TO_SHANGGE"
        ),
        "protocol": final_protocol,
        "search_protocol": search_protocol,
        "finalizer_program_sha256": sha256_file(SCRIPT_PATH),
        "search_program_sha256": sha256_file(SEARCH_V2_PATH),
        "model_sha256": sha256_file(model_path),
        "baseline_artifact_sha256": baseline_hashes_before,
        "search_manifest_sha256": sha256_file(
            search_dir / "cyclic_base_recovery_manifest.json"
        ),
        "independent_cyclic_base_floor_1pct": independent_floor,
        "baseline_raw_rows": len(baseline_raw),
        "directed_rows_appended_to_all_and_unique": len(directed_rows),
        "recovered_all_rows": len(merged_all),
        "recovered_unique_rows": len(merged_unique),
        "baseline_eligible_candidates": len(baseline_eligible),
        "directed_recovery_candidates": len(directed_rows),
        "final_signature_candidates": len(final_rows),
        "targets_with_signature_candidate": EXPECTED_TARGETS - len(uncovered),
        "targets_without_signature_candidate": uncovered,
        "targets_formally_abstained": [],
        "structure_status": (
            "NOT_PREDICTED; every candidate requires global-complex and complete "
            "cyclic-peptide CA RMSD <3 A"
        ),
        "structure_handoff_status": "NOT_CREATED_PENDING_MANUAL_REVIEW",
        "permeability_status": "DEFERRED_UNTIL_RETURNED_STRUCTURES_PASS_BOTH_RMSD_GATES",
        "artifacts": {
            "all_candidates": artifact(final_all_path),
            "unique_candidates": artifact(final_unique_path),
            "final_candidates": artifact(final_candidates_path),
            "target_manifest": artifact(target_manifest_path),
            "target_summary": artifact(summary_path),
        },
    }
    final_manifest_path = out_dir / "generation_manifest.json"
    legacy_final.atomic_write_json(final_manifest_path, final_manifest)

    audit_report = {
        "quality_gate": quality_gate,
        "release_status": final_manifest["release_status"],
        "protocol": audit_protocol,
        "search_protocol": search_protocol,
        "finalizer_program_sha256": sha256_file(SCRIPT_PATH),
        "search_program_sha256": sha256_file(SEARCH_V2_PATH),
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "pass_1_integrity_and_independent_rescore": {
            "quality_gate": pass_1,
            "checks": pass_1_checks,
            "annotation_errors": annotation_errors[:25],
            "independent_methyl_rescore_errors": rescore_errors[:25],
            "independent_cyclic_base_errors": base_errors[:25],
            "duplicate_evidence": duplicate_evidence[:25],
        },
        "pass_2_physical_position_and_representation": {
            "quality_gate": pass_2,
            "checks": pass_2_checks,
            "physical_position_errors": point_errors[:25],
            "concentration": concentration_rows,
            "av_family_physical_alignment": av_alignment,
        },
        "pass_3_novelty_coverage_and_workflow": {
            "quality_gate": pass_3,
            "checks": pass_3_checks,
            "novelty_errors": novelty_errors[:25],
            "uncovered_targets": uncovered,
            "candidate_count_by_target": dict(sorted(counts.items())),
        },
        "artifacts": {
            "final_manifest": artifact(final_manifest_path),
            "final_candidates": artifact(final_candidates_path),
            "search_manifest": artifact(
                search_dir / "cyclic_base_recovery_manifest.json"
            ),
            "position_concentration": artifact(concentration_path),
            "av_family_physical_support": artifact(av_path),
        },
    }
    audit_report_path = audit_out / "three_pass_generation_audit.json"
    legacy_final.atomic_write_json(audit_report_path, audit_report)

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
        raise RuntimeError("Immutable V8 baseline changed during V2 final audit")
    print(
        f"===== V8 {version_label} INDEPENDENT THREE-PASS AUDIT COMPLETE =====",
        flush=True,
    )
    print(f"Quality gate: {quality_gate}", flush=True)
    print(f"Target coverage: {EXPECTED_TARGETS - len(uncovered)}/{EXPECTED_TARGETS}", flush=True)
    print(f"Final candidates: {len(final_rows)}", flush=True)
    print("Structure handoff: NOT CREATED", flush=True)
    if quality_gate != "PASS":
        failed = [
            name
            for name, value in (
                ("pass_1", pass_1),
                ("pass_2", pass_2),
                ("pass_3", pass_3),
            )
            if value != "PASS"
        ]
        raise RuntimeError(
            f"V8 {version_label} final audit failed: " + ", ".join(failed)
        )


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
