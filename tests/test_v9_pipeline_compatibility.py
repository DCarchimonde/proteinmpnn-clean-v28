from __future__ import annotations

import ast
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "paper_clean_v28/serine_qc_retrain/02_retrain_canonical_expert_heads.py"
AUDITOR = ROOT / "paper_clean_v28/serine_qc_retrain/07_audit_cyclic_representation_equivariance.py"
GENERATOR = ROOT / "paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py"
TOPUP = ROOT / "paper_clean_v28/serine_qc_retrain/25_resume_cyclic_stability_v9_quota.py"
PLAN = ROOT / "paper_clean_v28/serine_qc_retrain/target_plan_cyclic_stability_v9_1700.json"
RUNNER = ROOT / "run_cyclic_stability_v9_1700.sh"


class V9PipelineCompatibilityTests(unittest.TestCase):
    def test_full_grid_diagnostic_call_has_the_runtime_signature(self):
        tree = ast.parse(TRAINER.read_text(encoding="utf-8"), filename=str(TRAINER))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "train_all_expert_heads"
        )
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "expert_probability_loss"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 4)
        self.assertEqual(
            [keyword.arg for keyword in calls[0].keywords],
            ["active_base_indices"],
        )

    def test_audit_uses_the_same_rounded_strict_release_boundary(self):
        source = AUDITOR.read_text(encoding="utf-8")
        self.assertIn("def strict_rounded_probability_pass", source)
        self.assertIn("round(numeric, 8) > float(threshold)", source)
        self.assertIn("if float(args.temperature) != 0.5", source)
        self.assertIn("float(args.threshold) != 0.6", source)
        self.assertNotIn("minimum <= threshold < maximum", source)

    def test_v9_audit_is_pinned_to_plan_operating_point_and_all_checks(self):
        generator = GENERATOR.read_text(encoding="utf-8")
        topup = TOPUP.read_text(encoding="utf-8")
        for source in (generator, topup):
            self.assertIn('report.get("temperature", -1.0)', source.replace("representation_audit", "report"))
            self.assertIn('report.get("threshold", -1.0)', source.replace("representation_audit", "report"))
            self.assertIn("all(bool(value) for value in", source)
        self.assertIn("annotation_context_policy", generator)
        self.assertIn("engine.ANNOTATION_CONTEXT", topup)

    def test_plan_and_launcher_are_exactly_17_by_100_at_t05(self):
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.assertEqual(plan["temperature"], 0.5)
        self.assertEqual(plan["methyl_threshold"], 0.6)
        self.assertEqual(plan["expected_target_count"], 17)
        self.assertEqual(plan["final_release_quota_per_target"], 100)
        self.assertEqual(plan["initial_stable_pool_quota_per_target"], 120)
        self.assertEqual(len(plan["targets"]), 17)
        self.assertEqual(
            {row["structure_quota"] for row in plan["targets"]}, {120}
        )
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_launcher_uses_committed_pinned_inputs_and_reopens_final_artifacts(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("$REPO_ROOT/v9_inputs/train_serine_provenance_corrected.jsonl", source)
        self.assertIn("$REPO_ROOT/v9_inputs/test_serine_provenance_corrected.jsonl", source)
        self.assertIn("$REPO_ROOT/v9_inputs/methylated_new_candidates.csv", source)
        self.assertIn("verify_frozen_inputs", source)
        self.assertIn("--require-frozen-input-sha256", source)
        self.assertIn("final_handoff_passes", source)
        self.assertIn("v9_1700_independent_replay.csv", source)
        self.assertIn("set(target_counts.values()) == {100}", source)

    def test_internal_151_records_do_not_promote_checkpoint(self):
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn("internal_development_check_names", source)
        self.assertIn(
            'quality_gate = "PASS" if all(checkpoint_quality_checks.values()) else "FAIL"',
            source,
        )
        self.assertIn("internal_development_audit_not_blind_outer_test", source)
        self.assertIn("probability_label_aware_adversarial", source)


if __name__ == "__main__":
    unittest.main()
