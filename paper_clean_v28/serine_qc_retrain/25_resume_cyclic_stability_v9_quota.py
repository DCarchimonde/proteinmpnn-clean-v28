#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V9-configured entry point for the audited adaptive quota top-up engine.

The implementation remains in the hash-audited V6 top-up module, but every
protocol, checkpoint, audit, stage, plan, and output default is rebound here to
the V9 cyclic-stability contract before argument parsing.  This avoids copying
the thousand-line recovery engine while preventing a V9 run from accepting a
V6 checkpoint or audit by accident.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
ENGINE_PATH = SCRIPT_PATH.with_name("08_resume_cyclic_representation_v6_quota.py")
V9_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "cyclic_stability_v9_1700"


def load_engine():
    spec = importlib.util.spec_from_file_location("v9_quota_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import quota engine: {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    engine = load_engine()
    engine.ENTRYPOINT_PATH = SCRIPT_PATH
    engine.DEFAULT_PLAN = SCRIPT_PATH.with_name(
        "target_plan_cyclic_stability_v9_1700.json"
    )
    engine.DEFAULT_MODEL = V9_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
    engine.DEFAULT_SOURCE_RUN = V9_ROOT / "generation"
    engine.DEFAULT_REPRESENTATION_AUDIT = (
        V9_ROOT / "representation_audit" / "cyclic_representation_audit.json"
    )
    engine.REQUIRED_EXPERT_PROTOCOL = (
        "canonical_clean_v28_all_expert_heads_corrected_labels_"
        "cyclic_stability_worst_start_v9"
    )
    engine.REQUIRED_TRAINING_REPRESENTATION_POLICY = (
        "all_physical_cyclic_starts_jointly_rotate_sequence_labels_and_"
        "backbone_coordinates_with_residue_index_reset"
    )
    engine.REQUIRED_TRAINING_ORDER_POLICY = (
        "complete_physical_cyclic_start_x_complete_L_decoder_order_grid_"
        "differentiably_meaned_per_start_then_mapped_to_physical_labels"
    )
    engine.REQUIRED_DEPLOYMENT_POLICY = (
        "all_cyclic_starts_and_all_decoder_orders_mapped_to_physical_"
        "residues_probability_mean_for_ranking_representation_min_for_release"
    )
    engine.REPRESENTATION_AUDIT_PROTOCOL = (
        "cyclic_stability_worst_start_heldout_gate_v9"
    )
    engine.REPRESENTATION_AUDIT_AUTHORIZATION = (
        "CYCLIC_STABILITY_V9_VALIDATED_FOR_UNIFORM_REGENERATION"
    )
    engine.ANNOTATION_MODE = (
        "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
    )
    engine.ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
    engine.RECOVERY_MODE = (
        "RETAIN_COMPLETE_V9_RUN_AND_ADAPTIVELY_SAMPLE_ONLY_STABLE_QUOTA_SHORTFALL_TARGETS"
    )
    engine.INITIAL_STAGE = "V9_INITIAL_FULL_REGENERATION"
    engine.TOPUP_STAGE = "V9_ADAPTIVE_STABLE_QUOTA_TOPUP"
    original_checkpoint_metadata = engine.checkpoint_metadata
    original_validate_representation_audit = engine.validate_representation_audit

    def v9_checkpoint_metadata(torch_module, model_path):
        metadata = original_checkpoint_metadata(torch_module, model_path)
        if not (
            float(metadata.get("worst_start_bce_weight", 0.0)) > 0.0
            and float(metadata.get("representation_consistency_weight", 0.0)) > 0.0
            and "full_decoder_order_grid" in str(
                metadata.get("training_objective", "")
            )
        ):
            raise RuntimeError(
                "V9 top-up requires positive worst-start/consistency weights and "
                "the full physical-start x decoder-order training objective"
            )
        return metadata

    engine.checkpoint_metadata = v9_checkpoint_metadata

    def v9_validate_representation_audit(
        report, report_path, model_sha256, plan_path
    ):
        original_validate_representation_audit(
            report, report_path, model_sha256, plan_path
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        checks = report.get("quality_checks", {})
        if not (
            isinstance(checks, dict)
            and bool(checks)
            and all(bool(value) for value in checks.values())
            and str(report.get("annotation_context_policy", ""))
            == engine.ANNOTATION_CONTEXT
            and float(report.get("temperature", -1.0))
            == float(plan["temperature"])
            and float(report.get("threshold", -1.0))
            == float(plan["methyl_threshold"])
        ):
            raise RuntimeError(
                "V9 top-up requires an all-PASS held-out audit at the exact "
                "plan temperature/threshold and peptide-only annotation context"
            )

    engine.validate_representation_audit = v9_validate_representation_audit
    engine.main()


if __name__ == "__main__":
    main()
