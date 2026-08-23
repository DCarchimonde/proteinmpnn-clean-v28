import ast
import json
import random
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "paper_clean_v28" / "serine_qc_retrain" / "36_retrain_short_length_balanced_v13.py"
GENERATION = ROOT / "paper_clean_v28" / "serine_qc_retrain" / "37_generate_fixed_budget_methyl_yield_v13.py"
PLAN = ROOT / "paper_clean_v28" / "serine_qc_retrain" / "target_plan_v13_fixed_budget_methyl_yield_1700.json"
BASE_GENERATOR = ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
AUDITOR = ROOT / "paper_clean_v28" / "serine_qc_retrain" / "07_audit_cyclic_representation_equivariance.py"
RUNNER = ROOT / "run_v13_fixed_budget_methyl_yield_1700.sh"
LAUNCHER = ROOT / "launch_v13_autodl_safe.sh"
TRAIN_JSONL = ROOT / "v9_inputs" / "train_serine_provenance_corrected.jsonl"


def extracted_namespace(path, constants, functions):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets
        ):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {
        "Any": object,
        "Dict": dict,
        "List": list,
        "Mapping": dict,
        "Sequence": list,
        "Tuple": tuple,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "random": random,
        "math": __import__("math"),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class V13FixedBudgetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        cls.trainer_text = TRAINER.read_text(encoding="utf-8")
        cls.generation_text = GENERATION.read_text(encoding="utf-8")
        cls.base_generator_text = BASE_GENERATOR.read_text(encoding="utf-8")
        cls.auditor_text = AUDITOR.read_text(encoding="utf-8")
        cls.runner_text = RUNNER.read_text(encoding="utf-8")
        cls.launcher_text = LAUNCHER.read_text(encoding="utf-8")

    def test_plan_is_fixed_budget_and_forbids_quota_filling(self):
        self.assertEqual(self.plan["expected_target_count"], 17)
        self.assertEqual(len(self.plan["targets"]), 17)
        self.assertEqual(self.plan["final_independent_draws_per_target"], 250)
        self.assertEqual(self.plan["batch_size"], 8)
        self.assertEqual(
            self.plan["minimum_raw_strict_methyl_hit_rate_per_target"], 0.5
        )
        self.assertEqual(
            self.plan["minimum_forward_cyclic_unique_strict_hits_per_target"],
            100,
        )
        self.assertFalse(self.plan["calibration_rows_eligible_for_handoff"])
        self.assertIn("FORBIDDEN", self.plan["topup_policy"])
        self.assertEqual(self.plan["prestructure_base_score_policy"], "NOT_COMPUTED_OR_USED")
        self.assertIn("WAIT_FOR_SHANGGE", self.plan["prestructure_rmsd_policy"])

    def test_real_training_split_has_short_lengths_on_both_sides(self):
        namespace = extracted_namespace(
            TRAINER,
            {"V13_SPLIT_PROTOCOL", "V13_REPEAT_FACTORS", "V13_SHORT_LENGTHS"},
            {
                "peptide_length",
                "counts_by_length",
                "deterministic_length_stratified_split",
                "repeat_factor",
                "length_balanced_records",
            },
        )
        records = [
            json.loads(line)
            for line in TRAIN_JSONL.read_text(encoding="utf-8").splitlines()
            if line
        ]
        supported = set("ACDEFGHIKLMNQRSTVWY")

        def counts(rows):
            result = {
                token: {"natural_negative": 0, "methyl_positive": 0}
                for token in supported
            }
            for row in rows:
                sequence = row[f"seq_chain_{row['masked_list'][0]}"]
                for token in sequence:
                    base = token.upper()
                    if base in result:
                        field = "methyl_positive" if token.islower() else "natural_negative"
                        result[base][field] += 1
            return result

        def record_name(row, index):
            return str(row.get("name") or row.get("pdb") or index)

        development, validation, manifest = namespace[
            "deterministic_length_stratified_split"
        ](records, 0.2, 42, counts, sorted(supported), record_name)
        self.assertEqual(manifest["accepted_seed_offset"], 0)
        self.assertGreaterEqual(int(manifest["validation_records_by_length"]["6"]), 2)
        self.assertGreaterEqual(int(manifest["validation_records_by_length"]["7"]), 2)
        self.assertEqual(
            set(manifest["development_records_by_length"]),
            set(manifest["validation_records_by_length"]),
        )
        weighted, weighting = namespace["length_balanced_records"](development)
        self.assertGreater(len(weighted), len(development))
        self.assertEqual(weighting["repeat_factors_by_length"]["6"], 5)
        self.assertEqual(weighting["repeat_factors_by_length"]["7"], 4)

    def test_global_strength_optimizes_worst_target_not_average(self):
        namespace = extracted_namespace(
            GENERATION,
            set(),
            {"choose_global_strength"},
        )
        rows = []
        for strength, rates in ((1.0, [0.9] * 16 + [0.0]), (2.0, [0.55] * 17)):
            for index, rate in enumerate(rates):
                rows.append(
                    {
                        "target_name": f"T{index}",
                        "guidance_strength": strength,
                        "raw_strict_methyl_hit_rate": rate,
                        "forward_cyclic_unique_strict_hits": int(rate * 40),
                    }
                )
        selected, ranking = namespace["choose_global_strength"](rows, [1.0, 2.0])
        self.assertEqual(selected, 2.0)
        self.assertEqual(ranking[0]["minimum_target_hit_rate"], 0.55)

    def test_cyclic_identity_collapses_forward_rotations_only(self):
        namespace = extracted_namespace(
            GENERATION,
            set(),
            {"forward_cyclic_identity"},
        )
        identity = namespace["forward_cyclic_identity"]
        self.assertEqual(identity("sACD"), identity("ACDs"))
        self.assertNotEqual(identity("sACD"), identity("DCAs"))

    def test_adaptive_native_guidance_is_available_but_release_stays_strict(self):
        tree = ast.parse(self.base_generator_text)
        generate = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate_batch"
        )
        arguments = [argument.arg for argument in generate.args.args]
        self.assertIn("methyl_guidance_mode", arguments)
        self.assertIn("until_provisional_hit", self.base_generator_text)
        self.assertIn("representation_min_strict_gt_threshold_zero_disagreement", self.base_generator_text)

    def test_v13_pipeline_has_no_legacy_search_or_prestructure_ranker(self):
        self.assertNotIn("33_recover_3zgc", self.runner_text)
        self.assertNotIn("35_replay_methyl_only", self.runner_text)
        self.assertNotIn("24_score_uniform", self.runner_text)
        self.assertNotIn("27_calibrate_and_apply_rmsd", self.runner_text)
        self.assertIn("V13_OVERWRITE_FAILED_GENERATION", self.runner_text)
        self.assertIn("STOPPED_WITH_PRESERVED_EVIDENCE", self.launcher_text)

    def test_auditor_recognizes_v13_without_changing_architecture_protocol(self):
        self.assertIn("V13_EXPERT_PROTOCOL", self.auditor_text)
        self.assertIn("V13_AUTHORIZATION", self.auditor_text)
        self.assertIn("v11_base_noninferiority_inherited_bitwise", self.auditor_text)
        self.assertIn("V11_MODEL_ARCHITECTURE_PROTOCOL", self.trainer_text)

    def test_handoff_requires_batch_one_replay_and_exact_quota(self):
        self.assertIn("every_selected_row_passed_independent_batch_one_replay", self.generation_text)
        self.assertIn("selected_rows_are_exactly_17_x_100", self.generation_text)
        self.assertIn("1700_给尚哥_结构输入.fasta", self.generation_text)
        self.assertIn("WAIT_FOR_SHANGGE_STRUCTURES", self.generation_text)


if __name__ == "__main__":
    unittest.main()
