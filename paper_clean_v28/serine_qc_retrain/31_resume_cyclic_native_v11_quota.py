#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V11-configured entry point for the audited adaptive quota top-up engine."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
ENGINE_PATH = SCRIPT_PATH.with_name("08_resume_cyclic_representation_v6_quota.py")
V11_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "cyclic_native_v11_1700_monomer"


def load_engine():
    spec = importlib.util.spec_from_file_location("v11_quota_engine", ENGINE_PATH)
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
        "target_plan_v11_cyclic_native_rmsd_priority_1700.json"
    )
    engine.DEFAULT_MODEL = (
        V11_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
    )
    engine.DEFAULT_SOURCE_RUN = V11_ROOT / "generation"
    engine.DEFAULT_REPRESENTATION_AUDIT = (
        V11_ROOT / "representation_audit" / "cyclic_representation_audit.json"
    )
    engine.REQUIRED_EXPERT_PROTOCOL = (
        "canonical_clean_v28_all_expert_heads_cyclic_native_relative_positions_v11"
    )
    engine.REQUIRED_TRAINING_REPRESENTATION_POLICY = (
        "boundary_marginalized_cyclic_relative_positions_with_all_physical_starts_"
        "retained_as_an_explicit_equivariance_verification_grid"
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
        "cyclic_native_relative_positions_heldout_gate_v11"
    )
    engine.REPRESENTATION_AUDIT_AUTHORIZATION = (
        "CYCLIC_NATIVE_V11_VALIDATED_FOR_RMSD_PRIORITY_REGENERATION"
    )
    engine.ANNOTATION_MODE = (
        "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
    )
    engine.ANNOTATION_CONTEXT = "peptide_chain_only_no_visible_receptor_chains"
    engine.RECOVERY_MODE = (
        "RETAIN_COMPLETE_V11_RUN_AND_METHYLATION_FIRST_GUIDED_SAMPLE_ONLY_"
        "POOL_SHORTFALL_TARGETS_WITH_SOFT_DIVERSITY_DIAGNOSTICS"
    )
    engine.INITIAL_STAGE = "V11_INITIAL_FULL_REGENERATION"
    engine.TOPUP_STAGE = "V11_METHYLATION_FIRST_GUIDED_DEFICIT_TOPUP"
    engine.TOPUP_CANDIDATE_PREFIX = "t05v11guided"
    engine.RECOVERY_LABEL = "V11"
    # Cycle several fixed product-of-experts strengths instead of blindly
    # repeating the base-only sampler.  Every proposal is still independently
    # annotated over the complete cyclic grid and must pass the unchanged
    # representation-min >0.6 release gate.
    engine.METHYL_GUIDANCE_STRENGTHS = (1.0, 2.0, 4.0, 8.0)
    # Retain the former 25-row target as a transparent diagnostic and soft
    # selection preference. It is deliberately not a generation/release gate:
    # a target-specific methylation hotspot must not cause 60,000 futile draws.
    engine.FINAL_RELEASE_DIVERSITY_RESERVE_PER_TARGET = 25
    engine.FINAL_RELEASE_DIVERSITY_IS_HARD_GATE = False
    engine.ALLOWED_SOURCE_FAILED_CHECKS = {
        "every_target_meets_pre_structure_candidate_quota",
        "every_target_meets_final_release_diversity_reserve",
    }
    original_checkpoint_metadata = engine.checkpoint_metadata
    original_validate_representation_audit = engine.validate_representation_audit

    def v11_checkpoint_metadata(torch_module, model_path):
        metadata = original_checkpoint_metadata(torch_module, model_path)
        if not (
            bool(metadata.get("cyclic_relative_positions"))
            and str(metadata.get("model_architecture_protocol", ""))
            == "proteinmpnn_boundary_marginalized_cyclic_relative_positions_v11"
            and float(metadata.get("base_sequence_loss_weight", 0.0)) > 0.0
            and float(metadata.get("positional_anchor_weight", 0.0)) > 0.0
            and float(metadata.get("maximum_equivariance_span_tolerance", -1.0))
            > 0.0
            and float(metadata.get("maximum_equivariance_span_tolerance", -1.0))
            <= 1e-5
            and float(
                metadata.get(
                    "best_epoch_maximum_training_representation_span",
                    float("inf"),
                )
            )
            <= 1e-5
            and float(metadata.get("worst_start_bce_weight", 0.0)) > 0.0
            and float(metadata.get("representation_consistency_weight", 0.0)) > 0.0
            and "full_decoder_order_grid"
            in str(metadata.get("training_objective", ""))
        ):
            raise RuntimeError(
                "V11 top-up requires cyclic-native model metadata, the base-sequence "
                "objective, positional trust region, and full-grid expert objective"
            )
        return metadata

    engine.checkpoint_metadata = v11_checkpoint_metadata

    def v11_validate_representation_audit(
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
            and bool(report.get("cyclic_relative_positions"))
            and float(report.get("maximum_equivariance_span_tolerance", -1.0))
            == 1e-5
            and str(report.get("annotation_context_policy", ""))
            == engine.ANNOTATION_CONTEXT
            and float(report.get("temperature", -1.0))
            == float(plan["temperature"])
            and float(report.get("threshold", -1.0))
            == float(plan["methyl_threshold"])
        ):
            raise RuntimeError(
                "V11 top-up requires the all-PASS numerical-equivariance audit "
                "at the exact plan temperature and threshold"
            )

    engine.validate_representation_audit = v11_validate_representation_audit
    engine.main()


if __name__ == "__main__":
    main()
