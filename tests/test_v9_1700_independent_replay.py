import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "26_independent_replay_and_package_v9_1700.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("v9_independent_replay", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = load_module()


class IndependentReplayPureContractTest(unittest.TestCase):
    def test_strict_round8_floor_controls_lowercase_and_never_lowercases_proline(self):
        minima = [0.600000004, 0.600000006, 0.99]
        self.assertEqual(
            replay.marked_sequence_from_floor("ASP", minima),
            "AsP",
        )
        self.assertFalse(replay.strict_rounded_pass(minima[0]))
        self.assertTrue(replay.strict_rounded_pass(minima[1]))

    def test_numeric_tolerance_cannot_hide_a_threshold_decision_change(self):
        errors, maximum = replay.compare_numeric_contract(
            "probability",
            [0.600000004],
            [0.600000006],
            replay.PROBABILITY_ATOL,
            replay.THRESHOLD,
        )
        self.assertLess(maximum, replay.PROBABILITY_ATOL)
        self.assertNotIn("probability_numeric_mismatch", errors)
        self.assertIn("probability_threshold_decision_mismatch", errors)

    def test_min_max_straddle_is_reported_in_physical_one_based_coordinates(self):
        self.assertEqual(
            replay.threshold_disagreements(
                [0.4, 0.8, 0.600000004],
                [0.5, 0.9, 0.600000006],
            ),
            [3],
        )

    def test_exact_17_by_100_quota_is_not_satisfied_by_1700_wrong_rows(self):
        good = [
            {"target_name": target}
            for target in replay.FROZEN_TARGETS
            for _ in range(replay.QUOTA)
        ]
        self.assertTrue(all(replay.exact_target_quota_checks(good, replay.FROZEN_TARGETS).values()))
        bad = list(good)
        bad[-1] = {"target_name": replay.FROZEN_TARGETS[0]}
        checks = replay.exact_target_quota_checks(bad, replay.FROZEN_TARGETS)
        self.assertTrue(checks["detailed_row_count_is_exactly_1700"])
        self.assertFalse(checks["every_detailed_target_has_exactly_100_rows"])

    def test_concise_and_fasta_views_must_match_detailed_exactly(self):
        row = {
            "final_release_id": "v9_1sfi_0001",
            "candidate_id": "candidate_1",
            "target_name": "1SFI",
            "design_seq": "aCDE",
            "design_natural_seq": "ACDE",
            "methyl_positions_1based": json.dumps([1]),
        }
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            concise = temp / replay.FINAL_CONCISE
            fasta = temp / replay.FINAL_FASTA
            with concise.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            fasta.write_text(
                ">v9_1sfi_0001|1SFI|candidate=candidate_1|marked=aCDE|"
                "methyl_positions=[1]\nACDE\n",
                encoding="utf-8",
            )
            self.assertTrue(all(replay.verify_selector_views([row], concise, fasta).values()))
            fasta.write_text(
                ">v9_1sfi_0001|1SFI|candidate=candidate_1|marked=aCDE|"
                "methyl_positions=[1]\nACDF\n",
                encoding="utf-8",
            )
            self.assertFalse(
                replay.verify_selector_views([row], concise, fasta)[
                    "selector_fasta_view_exactly_matches_detailed"
                ]
            )


if __name__ == "__main__":
    unittest.main()
