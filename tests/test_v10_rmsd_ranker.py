from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RANKER_PATH = (
    ROOT / "paper_clean_v28" / "serine_qc_retrain" / "rmsd_ranker_v10.py"
)
SELECTOR_PATH = (
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "23_select_and_audit_v9_1700.py"
)
DEVELOPMENT = ROOT / "v10_inputs" / "six_non3av_t05_joint_rmsd_476.csv"
PLAN = (
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "target_plan_v10_rmsd_priority_1700.json"
)
RUNNER = ROOT / "run_v10_rmsd_aware_1700_and_monomer.sh"
GENERATOR = ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
TOPUP_ENGINE = (
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "08_resume_cyclic_representation_v6_quota.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ranker = load(RANKER_PATH, "rmsd_ranker_v10_tests")
selector = load(SELECTOR_PATH, "selector_v10_overlay_tests")


def development_rows():
    with DEVELOPMENT.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class V10FrozenPlanTests(unittest.TestCase):
    def test_pool_is_large_enough_for_a_true_top_quartile_release(self):
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.assertEqual(plan["initial_stable_pool_quota_per_target"], 500)
        self.assertEqual(len(plan["targets"]), 17)
        self.assertEqual({row["structure_quota"] for row in plan["targets"]}, {500})
        digest = hashlib.sha256(PLAN.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "65a6b1def84f9271a82da740995f4fc67911a607abb31e2b5ff14d19b85118ac",
        )
        self.assertIn(digest, RUNNER.read_text(encoding="utf-8"))

    def test_generation_and_repeated_topup_are_hash_bound(self):
        generator_source = GENERATOR.read_text(encoding="utf-8")
        topup_source = TOPUP_ENGINE.read_text(encoding="utf-8")
        self.assertIn('"target_manifest": {', generator_source)
        self.assertIn('"target_manifest": source_paths["target_manifest"]', topup_source)
        self.assertIn('source_manifest.get("topup_program", {})', topup_source)
        self.assertIn('source_manifest.get("topup_engine", {})', topup_source)


class FeatureContractTests(unittest.TestCase):
    def test_feature_vector_is_finite_target_agnostic_and_recomputed(self):
        row = development_rows()[0]
        forward = ranker.sequence_features(row)
        changed_target = dict(row, target_name="A_TARGET_IDENTIFIER_MUST_NOT_BE_A_FEATURE")
        second = ranker.sequence_features(changed_target)
        self.assertEqual(forward.shape, (len(ranker.FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(forward)))
        np.testing.assert_array_equal(forward, second)

    def test_persisted_recovery_or_methyl_tampering_is_rejected(self):
        row = development_rows()[0]
        with self.assertRaisesRegex(ValueError, "natural_aa_recovery"):
            ranker.sequence_features(dict(row, natural_aa_recovery="0.999"))
        with self.assertRaisesRegex(ValueError, "design_methyl_count"):
            ranker.sequence_features(dict(row, design_methyl_count="99"))


class FrozenDevelopmentValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = development_rows()
        cls.lt5, cls.lt5_oof, _detail = ranker.cross_validation_summary(
            cls.rows, "joint_lt5"
        )
        cls.lt3, cls.lt3_oof, _detail3 = ranker.cross_validation_summary(
            cls.rows, "joint_lt3"
        )

    def test_primary_leave_target_out_gate_has_real_enrichment(self):
        self.assertEqual(self.lt5["rows"], 476)
        self.assertEqual(self.lt5["targets"], 6)
        self.assertEqual(self.lt5["positives"], 101)
        self.assertGreaterEqual(self.lt5["pooled_oof_auc"], 0.55)
        self.assertGreaterEqual(self.lt5["absolute_enrichment"], 0.02)
        self.assertGreaterEqual(self.lt5["relative_enrichment"], 1.10)
        self.assertTrue(np.all(np.isfinite(self.lt5_oof)))

    def test_lt3_is_retained_as_descriptive_secondary_endpoint(self):
        self.assertEqual(self.lt3["positives"], 16)
        self.assertTrue(np.all(np.isfinite(self.lt3_oof)))
        self.assertGreater(self.lt3["top_fraction_rate"], self.lt3["baseline_rate"])

    def test_historical_position_exemption_is_narrow_and_data_derived(self):
        support = ranker.historical_site_support(self.rows)
        self.assertEqual(
            support["3WNE"]["supported_high_concentration_positions_1based"],
            [2],
        )
        self.assertEqual(set(support), {"1SFI", "3P8F", "3WNE", "3ZGC", "4K1E", "4KEL"})
        self.assertNotIn("3AV9", support)


class SelectorOverlayTests(unittest.TestCase):
    @staticmethod
    def row(candidate_id: str, score: float, position: int = 2):
        natural = "ACDE"
        marked = list(natural)
        marked[position - 1] = marked[position - 1].lower()
        return {
            "candidate_id": candidate_id,
            "design_seq": "".join(marked),
            "design_natural_seq": natural + candidate_id[-1],
            "_natural_cyclic_key": candidate_id,
            "_methyl_positions": [position],
            "_primary_methyl_position": position,
            "ranking_mean_argmax_position_1based": position,
            "release_min_argmax_position_1based": position,
            "_base_score": -1.0 if score > 0.5 else 0.0,
            "_release_floor": 0.7,
            "_representation_span": 0.01,
            "_rmsd_lt5_score": score,
            "_rmsd_lt3_score": score / 2.0,
        }

    def test_rmsd_priority_precedes_base_score_when_collapse_policy_is_equal(self):
        high_risk_quality = self.row("candidateA", 0.9)
        low_risk_quality = self.row("candidateB", 0.1)
        ordered = selector.deduplicate_cyclic([low_risk_quality, high_risk_quality])
        self.assertEqual(ordered[0]["candidate_id"], "candidateA")

    def test_descriptive_lt3_score_never_breaks_an_lt5_tie(self):
        better_base = self.row("candidateA", 0.8)
        worse_base = self.row("candidateB", 0.8)
        better_base["_base_score"] = 0.0
        better_base["_rmsd_lt3_score"] = 0.01
        worse_base["_base_score"] = -2.0
        worse_base["_rmsd_lt3_score"] = 0.99
        ordered = selector.deduplicate_cyclic([worse_base, better_base])
        self.assertEqual(ordered[0]["candidate_id"], "candidateA")

    def test_supported_target_concentration_passes_but_unseen_3av_does_not(self):
        rows = [self.row(f"candidate-{index:03d}", 1.0 - index / 1000) for index in range(100)]
        supported = selector.target_summary("3WNE", rows, rows, 100, [2])
        unsupported = selector.target_summary("3AV9", rows, rows, 100, [])
        self.assertTrue(supported["position_concentration_pass"])
        self.assertFalse(unsupported["position_concentration_pass"])
        self.assertEqual(
            supported["position_concentration_policy"],
            "evidence_aware_historical_joint_lt5_support",
        )

    def test_strict_rounding_rejects_apparent_float_above_threshold(self):
        self.assertFalse(selector.strict_rounded_pass(0.600000004, 0.6))
        self.assertTrue(selector.strict_rounded_pass(0.600000006, 0.6))

    def test_ranker_overlay_is_bound_to_named_candidate_and_scored_files(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            candidate = temp / "base.csv"
            scored = temp / "scored.csv"
            manifest = temp / "manifest.json"
            base_row = {
                "target_name": "1SFI",
                "candidate_id": "candidate-1",
                "design_natural_seq": "ACDE",
            }
            with candidate.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(base_row))
                writer.writeheader()
                writer.writerow(base_row)
            scored_row = {
                **base_row,
                "rmsd_priority_protocol": selector.RMSD_RANKER_PROTOCOL,
                "rmsd_priority_primary_endpoint": "joint_global_and_cyclic_lt5A",
                "rmsd_priority_score_joint_lt5": "0.8",
                "rmsd_priority_score_joint_lt3_descriptive": "0.2",
                "rmsd_priority_rank_within_target": "1",
                "rmsd_priority_feature_vector": json.dumps([0.0] * 16),
                "rmsd_priority_warning": "not_observed_structure",
            }
            with scored.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(scored_row))
                writer.writeheader()
                writer.writerow(scored_row)
            support = {
                target: {"supported_high_concentration_positions_1based": [1]}
                for target in ("1SFI", "3P8F", "3WNE", "3ZGC", "4K1E", "4KEL")
            }
            payload = {
                "quality_gate": "PASS",
                "release_status": "AUTHORIZED_FOR_PRESTRUCTURE_PRIORITY_SELECTION",
                "protocol": selector.RMSD_RANKER_PROTOCOL,
                "quality_checks": {"development_gate": True},
                "historical_site_support": support,
                "inputs": {
                    "development_csv": {"sha256": selector.RMSD_DEVELOPMENT_SHA256},
                    "candidate_csv": {"sha256": selector.sha256_file(candidate)},
                },
                "artifacts": {
                    "scored_candidates": {"sha256": selector.sha256_file(scored)}
                },
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            overlay, positions, checks = selector.load_rmsd_priority_overlay(
                scored, manifest, candidate
            )
        self.assertTrue(all(checks.values()))
        self.assertEqual(len(overlay), 1)
        self.assertEqual(positions["3WNE"], [1])


if __name__ == "__main__":
    unittest.main()
