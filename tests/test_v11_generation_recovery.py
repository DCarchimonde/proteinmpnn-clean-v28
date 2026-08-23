from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py"
TOPUP_ENGINE_PATH = (
    ROOT
    / "paper_clean_v28/serine_qc_retrain/08_resume_cyclic_representation_v6_quota.py"
)
REAUDITOR_PATH = (
    ROOT
    / "paper_clean_v28/serine_qc_retrain/32_reaudit_v11_serialized_gate.py"
)
V11_PLAN = (
    ROOT
    / "paper_clean_v28/serine_qc_retrain/target_plan_v11_cyclic_native_rmsd_priority_1700.json"
)
RUNNER = ROOT / "run_v11_cyclic_native_1700_and_monomer.sh"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load("v11_recovery_generator_tests", GENERATOR_PATH)
topup = load("v11_recovery_topup_tests", TOPUP_ENGINE_PATH)
reauditor = load("v11_recovery_reauditor_tests", REAUDITOR_PATH)


def write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_row(target: str, index: int):
    means = [0.9, 0.1, 0.1, 0.1]
    zeros = [0.0] * 4
    return {
        "candidate_id": f"v11_{index:02d}",
        "target_name": target,
        "temperature": 0.5,
        "methyl_threshold": 0.6,
        "seed": 101,
        "effective_seed": 10100000 + index,
        "draw_index_within_seed": 1,
        "design_seq": "aCDE",
        "design_natural_seq": "ACDE",
        "native_seq": "ACDE",
        "native_length": 4,
        "design_length": 4,
        "length_match": 1,
        "valid_token_gate": 1,
        "design_methyl_count": 1,
        "design_methyl_rate": 0.25,
        "methyl_positions_1based": "[1]",
        "base_log_probability_mean": -1.0,
        "methyl_probabilities": json.dumps(means),
        "methyl_probability_representation_min": json.dumps(means),
        "methyl_probability_representation_max": json.dumps(means),
        "methyl_probability_representation_span": json.dumps(zeros),
        "methyl_probability_representation_std": json.dumps(zeros),
        "methyl_probability_representation_by_start": json.dumps([means] * 4),
        "methyl_probability_order_std_max": 0.0,
        "methyl_probability_representation_std_max": 0.0,
        "methyl_probability_representation_span_max": 0.0,
        "representation_threshold_disagreement_count": 0,
        "annotation_mode": generator.CYCLIC_REPRESENTATION_ANNOTATION_MODE,
        "annotation_context_policy": generator.PEPTIDE_ONLY_ANNOTATION_CONTEXT,
        "annotation_visible_receptor_chains": 0,
        "annotation_decoder_order_ensemble_size": 4,
        "annotation_representation_ensemble_size": 4,
        "annotation_total_probability_ensemble_size": 16,
        "stable_cyclic_release_gate": 0,
    }


class V11QuotaNamingTests(unittest.TestCase):
    def test_internal_pool_and_final_structure_handoff_are_distinct(self):
        plan = generator.read_json(V11_PLAN)
        checked = generator.validate_plan(plan)
        self.assertEqual(checked["planned_preselection_candidate_pool"], 8500)
        self.assertEqual(checked["planned_structure_handoff"], 1700)

    def test_runner_reaudits_then_uses_guided_recovery(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("32_reaudit_v11_serialized_gate.py", source)
        self.assertLess(
            source.index("Re-auditing persisted V11 probabilities"),
            source.index("Methylation-first guided recovery"),
        )
        self.assertIn("Methylation-first guided recovery", source)
        wrapper = TOPUP_ENGINE_PATH.with_name(
            "31_resume_cyclic_native_v11_quota.py"
        ).read_text(encoding="utf-8")
        self.assertIn("METHYL_GUIDANCE_STRENGTHS", wrapper)
        self.assertIn("FINAL_RELEASE_DIVERSITY_RESERVE_PER_TARGET = 25", wrapper)
        self.assertIn("FINAL_RELEASE_DIVERSITY_IS_HARD_GATE = False", wrapper)
        self.assertIn("--concentration-gates diagnostic", source)

    def test_diversity_reserve_counts_candidate_rows_not_raw_site_fraction(self):
        rows = [
            {"target_name": "T", "design_seq": "AAAAAAaA"}
            for _ in range(80)
        ] + [
            {"target_name": "T", "design_seq": "AcAAAAAA"}
            for _ in range(25)
        ]
        report = topup.target_diversity_reserve(rows, "T", {7}, {"A"})
        self.assertEqual(report["alternate_position_rows"], 25)
        self.assertEqual(report["alternate_residue_rows"], 25)

    def test_balanced_tied_positions_do_not_trigger_impossible_reserve(self):
        rows = [
            {
                "target_name": "T",
                "design_seq": "aCDE" if index % 2 == 0 else "AcDE",
            }
            for index in range(100)
        ]
        report = topup.target_release_diversity_state(rows, "T", 25)
        self.assertEqual(report["dominant_positions_1based"], [1, 2])
        self.assertEqual(report["alternate_position_rows"], 0)
        self.assertTrue(report["position_reserve_ready"])

    def test_v11_soft_diversity_stops_as_soon_as_candidate_goal_is_met(self):
        diversity = {"release_diversity_reserve_ready": False}
        self.assertTrue(
            topup.target_recovery_ready(510, 510, diversity, 25, False)
        )
        self.assertFalse(
            topup.target_recovery_ready(510, 510, diversity, 25, True)
        )
        self.assertFalse(
            topup.target_recovery_ready(509, 510, diversity, 25, False)
        )


class V11SerializedGateReauditTests(unittest.TestCase):
    def test_reaudit_reuses_raw_rows_and_rebuilds_only_derived_views(self):
        targets = [
            "1SFI", "3AV9", "3AVA", "3AVB", "3AVF", "3AVG", "3AVH",
            "3AVI", "3AVJ", "3AVK", "3AVM", "3AVN", "3P8F", "3WNE",
            "3ZGC", "4K1E", "4KEL",
        ]
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            run_dir = root / "generation"
            run_dir.mkdir()
            model = root / "model.pt"
            audit = root / "audit.json"
            plan_path = root / "plan.json"
            historical = root / "historical.csv"
            prior = root / "prior.csv"
            policy = root / "policy.json"
            model.write_bytes(b"v11-model")
            plan = {
                "protocol": (
                    "temperature_0.5_cyclic_native_relative_positions_v11_"
                    "unit_reaudit"
                ),
                "temperature": 0.5,
                "methyl_threshold": 0.6,
                "sampling_context_policy": generator.SAMPLING_CONTEXT_POLICY,
                "annotation_context_policy": generator.PEPTIDE_ONLY_ANNOTATION_CONTEXT,
                "annotation_ranking_probability_policy": "representation_mean",
                "annotation_release_probability_policy": (
                    "representation_min_strict_gt_threshold_zero_disagreement"
                ),
                "seeds": [101],
                "expected_target_count": 17,
                "final_release_quota_per_target": 100,
                "initial_stable_pool_quota_per_target": 500,
                "frozen_targets": [],
                "targets": [
                    {
                        "target_name": target,
                        "sequences_per_seed": 1,
                        "structure_quota": 500,
                        "current_problem": "unit",
                    }
                    for target in targets
                ],
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            audit.write_text(json.dumps({"quality_gate": "PASS"}), encoding="utf-8")
            write_csv(historical, [{"target_name": "OLD", "design_seq": "PPPP"}])
            write_csv(
                prior,
                [
                    {"target_name": "OLD", "design_seq": f"SEQ{index}"}
                    for index in range(1333)
                ],
            )
            policy.write_text(
                json.dumps(
                    {
                        "protocol": (
                            "historical_joint_lt5_supported_position_concentration_v10"
                        ),
                        "maximum_share_without_exemption": 0.8,
                        "supported_positions_1based_by_target": {},
                    }
                ),
                encoding="utf-8",
            )
            rows = [stable_row(target, index) for index, target in enumerate(targets)]
            write_csv(run_dir / "all_candidates.csv", rows)
            write_csv(run_dir / "unique_candidates.csv", rows)
            write_csv(run_dir / "methylated_new_candidates.csv", [])
            write_csv(
                run_dir / "generation_summary_by_target.csv",
                [{"target_name": target, "new_methylated_for_permeability": 0} for target in targets],
            )
            write_csv(
                run_dir / "target_manifest.csv",
                [{"target_name": target, "selected_chain": "P"} for target in targets],
            )
            all_path = run_dir / "all_candidates.csv"
            plan_hash = sha256(plan_path)
            model_hash = sha256(model)
            audit_hash = sha256(audit)
            artifacts = {
                name: {
                    "path": str(run_dir / filename),
                    "sha256": sha256(run_dir / filename),
                }
                for name, filename in (
                    ("all_candidates", "all_candidates.csv"),
                    ("unique_candidates", "unique_candidates.csv"),
                    ("methylated_new_candidates", "methylated_new_candidates.csv"),
                    ("generation_summary_by_target", "generation_summary_by_target.csv"),
                    ("target_manifest", "target_manifest.csv"),
                )
            }
            manifest = {
                "quality_gate": "FAIL",
                "quality_checks": {
                    "every_target_meets_pre_structure_candidate_quota": False
                },
                "protocol": plan["protocol"],
                "temperature": 0.5,
                "methyl_threshold": 0.6,
                "model_expert_qc_protocol": reauditor.EXPECTED_EXPERT_PROTOCOL,
                "model_sha256": model_hash,
                "raw_candidates_generated": 17,
                "new_methylated_candidates_for_permeability": 0,
                "artifacts": artifacts,
                "cyclic_representation_heldout_audit": {
                    "quality_gate": "PASS",
                    "model_sha256": model_hash,
                    "plan_sha256": plan_hash,
                    "sha256": audit_hash,
                },
            }
            (run_dir / "generation_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            immutable_hash = sha256(all_path)
            args = SimpleNamespace(
                run_dir=str(run_dir),
                plan=str(plan_path),
                model=str(model),
                representation_audit=str(audit),
                historical_csv=str(historical),
                prior_csv=str(prior),
                position_concentration_policy=str(policy),
            )
            reauditor.run(args)
            rebuilt = reauditor.read_json(run_dir / "generation_manifest.json")
            self.assertEqual(sha256(all_path), immutable_hash)
            self.assertEqual(rebuilt["planned_structure_handoff"], 1700)
            self.assertEqual(rebuilt["planned_preselection_candidate_pool"], 8500)
            self.assertEqual(rebuilt["new_methylated_candidates_for_permeability"], 17)
            self.assertEqual(
                rebuilt["serialized_gate_reaudit"]["all_candidates_rows_rewritten"],
                0,
            )
            self.assertEqual(
                Path(
                    rebuilt["serialized_gate_reaudit"]["source_validation"]
                    ["source_manifest_path"]
                ).resolve(),
                (
                    run_dir
                    / "pre_serialized_gate_reaudit_backup/generation_manifest.json"
                ).resolve(),
            )
            self.assertTrue(
                (run_dir / "pre_serialized_gate_reaudit_backup/generation_manifest.json").is_file()
            )
            with self.assertRaises(SystemExit) as second:
                reauditor.run(args)
            self.assertEqual(second.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
