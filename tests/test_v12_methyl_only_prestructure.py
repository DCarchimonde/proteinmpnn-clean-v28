import importlib.util
import csv
import hashlib
import json
import pathlib
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "paper_clean_v28" / "serine_qc_retrain"


def load(name, filename):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


search = load("v12_search_test", "33_recover_3zgc_methyl_only_v12.py")
selector = load("v12_selector_test", "34_select_methyl_only_1700_v12.py")
replay = load("v12_replay_test", "35_replay_methyl_only_1700_v12.py")


class V12MethylOnlyPrestructureTests(unittest.TestCase):
    def candidate(self, sequence="AAAAAAA", floor=0.7, mean=0.75, candidate_id="c1"):
        minimum = [floor] + [0.1] * (len(sequence) - 1)
        means = [mean] + [0.12] * (len(sequence) - 1)
        maximum = [mean + 0.01] + [0.13] * (len(sequence) - 1)
        span = [upper - lower for lower, upper in zip(minimum, maximum)]
        std = [0.001] * len(sequence)
        design = sequence[0].lower() + sequence[1:]
        return {
            "candidate_id": candidate_id,
            "target_name": "3ZGC",
            "temperature": "0.5",
            "methyl_threshold": "0.6",
            "strict_threshold_operator": ">",
            "design_seq": design,
            "design_natural_seq": sequence,
            "native_seq": "GDEETGE"[: len(sequence)],
            "native_length": str(len(sequence)),
            "design_length": str(len(sequence)),
            "length_match": "1",
            "valid_token_gate": "1",
            "design_methyl_count": "1",
            "methyl_positions_1based": "[1]",
            "methyl_probabilities": json.dumps(means),
            "methyl_probability_representation_min": json.dumps(minimum),
            "methyl_probability_representation_max": json.dumps(maximum),
            "methyl_probability_representation_span": json.dumps(span),
            "methyl_probability_representation_std": json.dumps(std),
            "methyl_probability_representation_span_max": max(span),
            "representation_threshold_disagreement_positions_1based": "[]",
            "representation_threshold_disagreement_count": "0",
            "passes_methylation_hard_gate": "1",
        }

    def test_plan_is_exact_17x100_and_defers_rmsd(self):
        plan = json.loads(
            (SCRIPTS / "target_plan_v12_methyl_only_1700.json").read_text()
        )
        self.assertEqual(len(plan["targets"]), 17)
        self.assertEqual(plan["final_release_quota_per_target"], 100)
        self.assertEqual(
            plan["prestructure_base_score_policy"],
            "not_a_release_gate_and_not_used_for_selection",
        )
        self.assertIn("not_available", plan["prestructure_rmsd_policy"])
        self.assertIn("structures_return", plan["poststructure_rmsd_policy"])

    def test_seed_extraction_is_target_and_length_specific(self):
        rows = [
            {"target_name": "3ZGC", "design_seq": "aAAAAAA"},
            {"target_name": "3ZGC", "design_seq": "AAAA"},
            {"target_name": "3P8F", "design_seq": "BBBBBBB"},
        ]
        sequences, problems = search.seed_sequences_from_rows(rows, "fixture")
        self.assertEqual(sequences, {"AAAAAAA"})
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["problem"], "length_not_equal_to_3zgc_native")

    def test_selector_accepts_only_representation_minimum_pass(self):
        row = self.candidate()
        validated, errors = selector.validate_candidate(
            row, "3ZGC", lambda _row, _natural: True
        )
        self.assertFalse(errors)
        self.assertIsNotNone(validated)
        self.assertAlmostEqual(validated["_release_floor"], 0.7)

        failed = self.candidate(floor=0.59, mean=0.99, candidate_id="c2")
        failed["design_seq"] = failed["design_natural_seq"]
        failed["methyl_positions_1based"] = "[]"
        failed["design_methyl_count"] = "0"
        validated, errors = selector.validate_candidate(
            failed, "3ZGC", lambda _row, _natural: True
        )
        self.assertIsNone(validated)
        self.assertIn(
            "no_representation_minimum_site_strictly_above_0_6", errors
        )

    def test_release_selection_is_quality_ranked_and_cyclic_unique(self):
        first = self.candidate("AAAAAAC", 0.9, 0.91, "first")
        second = self.candidate("CAAAAAA", 0.8, 0.82, "rotation")
        third = self.candidate("AAAAAAD", 0.7, 0.75, "third")
        rows = []
        for source in (first, second, third):
            validated, errors = selector.validate_candidate(
                source, "3ZGC", lambda _row, _natural: True
            )
            self.assertFalse(errors)
            rows.append(validated)
        selected = selector.deduplicate_and_select(rows)
        self.assertEqual([row["candidate_id"] for row in selected], ["first", "third"])

    def test_search_release_excludes_prior_cyclic_identity(self):
        rows = [
            {
                "design_natural_seq": "AAAAAAC",
                "release_floor_maximum_probability": 0.9,
                "ranking_mean_maximum_probability": 0.91,
                "methyl_probability_representation_span_max": 0.01,
            },
            {
                "design_natural_seq": "AAAAAAD",
                "release_floor_maximum_probability": 0.8,
                "ranking_mean_maximum_probability": 0.81,
                "methyl_probability_representation_span_max": 0.01,
            },
        ]
        selected, rejected = search.select_novel_release_rows(
            rows,
            set(),
            {search.canonical_rotation("CAAAAAA")},
            quota=100,
        )
        self.assertEqual([row["design_natural_seq"] for row in selected], ["AAAAAAD"])
        self.assertEqual(len(rejected), 1)

    def test_replay_vector_comparison_has_fixed_tolerance(self):
        errors, delta = replay.compare_vectors(
            "x", "[0.1, 0.2]", "[0.100001, 0.2]", 2
        )
        self.assertFalse(errors)
        self.assertLessEqual(delta, replay.ATOL)
        errors, _delta = replay.compare_vectors(
            "x", "[0.1, 0.2]", "[0.11, 0.2]", 2
        )
        self.assertEqual(errors, ["x_mismatch"])

    def test_runner_contains_no_base_or_rmsd_stage(self):
        source = (ROOT / "run_v12_methyl_only_1700.sh").read_text()
        self.assertNotIn("24_score_uniform_cyclic_base", source)
        self.assertNotIn("27_calibrate_and_apply_rmsd_ranker", source)
        self.assertIn("RMSD ranking: NOT AVAILABLE", source)
        self.assertIn("17 x 100", source)

    def test_selector_end_to_end_writes_exact_1700_views(self):
        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        def write_csv(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            fields = list(rows[0])
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        alphabet = "ACDEFGHIKLMNQRSTVWY"  # no Pro; first site remains methylatable
        sequences = []
        cyclic = set()
        number = 0
        while len(sequences) < 100:
            value = number
            tail = []
            for _ in range(6):
                tail.append(alphabet[value % len(alphabet)])
                value //= len(alphabet)
            sequence = "A" + "".join(tail)
            key = selector.canonical_rotation(sequence)
            if key not in cyclic:
                cyclic.add(key)
                sequences.append(sequence)
            number += 1

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            generation = root / "generation"
            zgc = root / "zgc"
            output = root / "output"
            model = root / "model.pt"
            audit = root / "audit.json"
            historical = root / "historical.csv"
            prior = root / "prior.csv"
            plan = SCRIPTS / "target_plan_v12_methyl_only_1700.json"
            model.write_bytes(b"fixture-model")
            audit.write_text(
                json.dumps(
                    {
                        "quality_gate": "PASS",
                        "model_sha256": digest(model),
                        "quality_checks": {"fixture": True},
                    }
                )
            )
            empty_fields = "target_name,design_natural_seq,design_seq\n"
            historical.write_text(empty_fields)
            prior.write_text(empty_fields)

            rows = []
            zgc_rows = []
            for target in selector.TARGETS:
                for index, sequence in enumerate(sequences):
                    row = self.candidate(
                        sequence,
                        floor=0.7 + index / 10000,
                        mean=0.75 + index / 10000,
                        candidate_id=f"fixture_{target}_{index:03d}",
                    )
                    row["target_name"] = target
                    if target == "3ZGC":
                        zgc_rows.append(row)
                    else:
                        rows.append(row)
            candidates = generation / "methylated_new_candidates.csv"
            write_csv(candidates, rows)
            generation_manifest = generation / "generation_manifest.json"
            generation_manifest.write_text(
                json.dumps(
                    {
                        "model_sha256": digest(model),
                        "quality_checks": {
                            "every_target_meets_pre_structure_candidate_quota": False,
                            "all_other_checks": True,
                        },
                        "artifacts": {
                            "methylated_new_candidates": {
                                "sha256": digest(candidates)
                            }
                        },
                    }
                )
            )
            zgc_candidates = zgc / "3zgc_exact_100_methylated.csv"
            write_csv(zgc_candidates, zgc_rows)
            (zgc / "3zgc_methyl_only_search_manifest.json").write_text(
                json.dumps(
                    {
                        "quality_gate": "PASS",
                        "release_status": "AUTHORIZED_3ZGC_EXACT_100_METHYLATION_ONLY_PRESTRUCTURE_ROWS",
                        "quota": 100,
                        "inputs": {"model": {"sha256": digest(model)}},
                        "artifacts": {
                            "exact_100_release": {"sha256": digest(zgc_candidates)}
                        },
                    }
                )
            )
            args = types.SimpleNamespace(
                generation_dir=str(generation),
                zgc_dir=str(zgc),
                plan=str(plan),
                model=str(model),
                representation_audit=str(audit),
                historical_csv=str(historical),
                prior_csv=str(prior),
                out_dir=str(output),
            )
            selector.run(args)
            detailed = selector.read_csv(output / selector.DETAIL_NAME)
            self.assertEqual(len(detailed), 1700)
            self.assertTrue(selector.verify_views(
                output / selector.DETAIL_NAME,
                output / selector.CONCISE_NAME,
                output / selector.FASTA_NAME,
            )["reopened_fasta_matches_detailed_rows"])
            manifest = selector.read_json(output / selector.MANIFEST_NAME)
            self.assertEqual(manifest["quality_gate"], "PASS")
            self.assertEqual(manifest["prestructure_base_score_policy"], "NOT_USED")
            self.assertEqual(
                manifest["prestructure_rmsd_policy"],
                "NOT_AVAILABLE_UNTIL_STRUCTURES_RETURN",
            )


if __name__ == "__main__":
    unittest.main()
