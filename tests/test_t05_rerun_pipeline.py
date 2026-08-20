from __future__ import annotations

import importlib.util
import csv
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_module(
    "t05_generator",
    ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py",
)
selector = load_module(
    "t05_selector",
    ROOT / "paper_clean_v28" / "rerun_t05" / "02_select_after_permeability.py",
)


class WindowsLauncherRegressionTests(unittest.TestCase):
    def test_python_probe_avoids_windows_powershell_dash_c_quoting(self):
        launcher = (ROOT / "run_t05_rerun.ps1").read_text(encoding="utf-8")
        self.assertNotIn("& $Candidate.Path -c $ProbeCode", launcher)
        self.assertIn("& $Candidate.Path $ProbePath", launcher)
        self.assertIn("Remove-Item -LiteralPath $TemporaryPath", launcher)

    def test_embedded_python_probe_programs_compile(self):
        launcher = (ROOT / "run_t05_rerun.ps1").read_text(encoding="utf-8")
        programs = re.findall(r"\$ProbeCode = '([^'\r\n]+)'", launcher)
        self.assertEqual(len(programs), 2)
        for program in programs:
            compile(program, "<PowerShell Python probe>", "exec")


class GenerationProtocolTests(unittest.TestCase):
    def test_frozen_plan_counts(self):
        plan = generator.read_json(
            ROOT / "paper_clean_v28" / "rerun_t05" / "target_plan.json"
        )
        checked = generator.validate_plan(plan)
        self.assertEqual(checked["expected_raw_candidates"], 13_500)
        self.assertEqual(checked["planned_structure_handoff"], 185)
        self.assertEqual(len(checked["target_names"]), 13)
        self.assertEqual(plan["frozen_targets"], ["1SFI", "3WNE", "4K1E", "4KEL"])

    def test_methylation_and_naturalization_helpers(self):
        self.assertEqual(generator.naturalize("AcDe"), "ACDE")
        self.assertEqual(generator.methyl_positions_1based("AcDe"), [2, 4])
        self.assertAlmostEqual(generator.sequence_recovery("ACDF", "AcDE"), 0.75)

    def test_unique_aggregation_applies_methyl_and_old_pool_gates(self):
        base = {
            "candidate_id": "new",
            "target_name": "T1",
            "design_seq": "AcDE",
            "design_methyl_count": 1,
            "methyl_threshold": 0.6,
            "methyl_probabilities": "[0.1, 0.9, 0.1, 0.1]",
            "methyl_probability_representation_min": "[0.1, 0.8, 0.1, 0.1]",
            "methyl_probability_representation_max": "[0.1, 0.95, 0.1, 0.1]",
            "methyl_probability_representation_span": "[0.0, 0.15, 0.0, 0.0]",
            "methyl_probability_representation_std": "[0.0, 0.06373774, 0.0, 0.0]",
            "methyl_probability_representation_by_start": (
                "[[0.1, 0.8, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], "
                "[0.1, 0.95, 0.1, 0.1], [0.1, 0.95, 0.1, 0.1]]"
            ),
            "annotation_mode": (
                "peptide_only_all_cyclic_starts_and_decoder_orders_mapped_to_physical_residues"
            ),
            "annotation_decoder_order_ensemble_size": 4,
            "annotation_representation_ensemble_size": 4,
            "annotation_total_probability_ensemble_size": 16,
            "representation_threshold_disagreement_count": 0,
            "base_log_probability_mean": -1.0,
            "seed": 101,
            "draw_index_within_seed": 1,
        }
        repeated = dict(base, candidate_id="repeat", seed=202, base_log_probability_mean=-2.0)
        old = dict(base, candidate_id="old", design_seq="ACdE")
        fresh = dict(base, candidate_id="fresh", design_seq="AcDF")
        prior = dict(base, candidate_id="prior", design_seq="AcDG")
        rows = generator.aggregate_unique_candidates(
            [base, repeated, old, fresh, prior],
            {("T1", "ACdE")},
            {("T1", "ACDE")},
            {("T1", "AcDG")},
            {("T1", "ACDG")},
        )
        by_sequence = {row["design_seq"]: row for row in rows}
        self.assertEqual(by_sequence["AcDE"]["occurrence_count"], 2)
        self.assertEqual(by_sequence["AcDE"]["eligible_for_new_permeability_screen"], 0)
        self.assertEqual(by_sequence["AcDE"]["seen_in_historical_4115_naturalized"], 1)
        self.assertEqual(by_sequence["ACdE"]["eligible_for_new_permeability_screen"], 0)
        self.assertEqual(by_sequence["AcDF"]["eligible_for_new_permeability_screen"], 1)
        self.assertEqual(by_sequence["AcDG"]["seen_in_prior_1333"], 1)
        self.assertEqual(by_sequence["AcDG"]["eligible_for_new_permeability_screen"], 0)

    def test_quota_resume_canonicalization_never_rewrites_preferred_source(self):
        source = {
            "candidate_id": "z_existing_source",
            "target_name": "T1",
            "design_seq": "AcDE",
            "design_natural_seq": "ACDE",
            "methyl_probabilities": "[0.1, 0.9, 0.1, 0.1]",
        }
        lexicographically_earlier_topup = {
            "candidate_id": "a_new_topup",
            "target_name": "T1",
            "design_seq": "ACdE",
            "design_natural_seq": "ACDE",
            "methyl_probabilities": "[0.1, 0.1, 0.9, 0.1]",
        }
        rows = [source, lexicographically_earlier_topup]
        result = generator.canonicalize_repeated_natural_annotations(
            rows, preferred_candidate_ids={"z_existing_source"}
        )
        self.assertEqual(result["rows_rewritten_to_canonical_payload"], 1)
        self.assertEqual(source["design_seq"], "AcDE")
        self.assertEqual(
            lexicographically_earlier_topup["design_seq"], source["design_seq"]
        )
        self.assertEqual(
            lexicographically_earlier_topup["methyl_probabilities"],
            source["methyl_probabilities"],
        )


class PermeabilitySelectionTests(unittest.TestCase):
    def test_prediction_id_normalization(self):
        self.assertEqual(
            selector.normalize_prediction_id(r"E:\out\3AV9_4_AcDE_model.pdb"),
            "3av9_4_acde",
        )

    def test_strict_improvement_rejects_equal_prediction(self):
        candidates = [
            {"target_name": "T1", "permeability_id": "T1_0_AcDE_model"},
            {"target_name": "T1", "permeability_id": "T1_1_ACdE_model"},
        ]
        manifest = [
            {"id": "T1_0_AcDE_model", "record_type": "candidate", "target_name": "T1"},
            {"id": "T1_1_ACdE_model", "record_type": "candidate", "target_name": "T1"},
            {"id": "T1_9000000_ACDE_model", "record_type": "native", "target_name": "T1"},
        ]
        predictions = {
            "t1_0_acde": 0.5,
            "t1_1_acde": 0.6,
            "t1_9000000_acde": 0.5,
        }
        attached, native, missing, unexpected = selector.attach_predictions(
            candidates, manifest, predictions, {key: 1 for key in predictions}
        )
        self.assertEqual(native, {"T1": 0.5})
        self.assertFalse(missing)
        self.assertFalse(unexpected)
        self.assertEqual(attached[0]["permeability_improved_vs_native"], "NO")
        self.assertEqual(attached[1]["permeability_improved_vs_native"], "YES")

    def test_naturalized_duplicates_collapse_to_best_permeability_variant(self):
        rows = [
            {
                "design_seq": "AcDE",
                "design_natural_seq": "ACDE",
                "permeability_delta_log10_vs_native": 0.1,
                "base_log_probability_mean": -0.5,
                "natural_aa_recovery": 0.5,
            },
            {
                "design_seq": "ACdE",
                "design_natural_seq": "ACDE",
                "permeability_delta_log10_vs_native": 0.3,
                "base_log_probability_mean": -1.0,
                "natural_aa_recovery": 0.5,
            },
        ]
        collapsed = selector.collapse_naturalized_variants(rows)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["design_seq"], "ACdE")
        self.assertEqual(collapsed[0]["naturalized_variant_count"], 2)

    def test_structure_proxy_and_diversity_precede_relaxed_fill(self):
        def row(sequence, log_probability, gain):
            return {
                "design_seq": sequence,
                "design_natural_seq": sequence.upper(),
                "base_log_probability_mean": log_probability,
                "permeability_delta_log10_vs_native": gain,
                "natural_aa_recovery": 0.5,
            }

        rows = [
            row("AAAAAAAAAA", -0.1, 0.1),
            row("AAAAAAAACC", -0.2, 0.2),  # exactly 80% identity: same cluster
            row("AAAAAACCCC", -0.3, 0.3),  # 60% identity: strict diversity pass
        ]
        selected = selector.select_diverse(rows, quota=3, identity_ceiling=0.8)
        self.assertEqual(selected[0]["design_seq"], "AAAAAAAAAA")
        self.assertEqual(selected[1]["design_seq"], "AAAAAACCCC")
        self.assertEqual(selected[1]["diversity_gate"], "STRICT_PASS")
        self.assertEqual(selected[2]["design_seq"], "AAAAAAAACC")
        self.assertEqual(selected[2]["diversity_gate"], "RELAXED_FILL")

    def test_end_to_end_partial_quota_handoff_is_written_and_audited(self):
        plan = generator.read_json(
            ROOT / "paper_clean_v28" / "rerun_t05" / "target_plan.json"
        )
        rerun_targets = [item["target_name"] for item in plan["targets"]]
        all_targets = rerun_targets + plan["frozen_targets"]

        def write_csv(path, rows):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            candidates = []
            input_manifest = []
            predictions = []
            target_manifest = []
            for index, target in enumerate(rerun_targets):
                candidate_id = f"{target.lower()}_{index}_AcDE_model"
                candidates.append(
                    {
                        "candidate_id": f"candidate_{target}",
                        "target_name": target,
                        "design_seq": "AcDE",
                        "design_natural_seq": "ACDE",
                        "methyl_positions_1based": "[2]",
                        "design_methyl_count": 1,
                        "base_log_probability_mean": -0.5,
                        "natural_aa_recovery": 0.5,
                        "permeability_id": candidate_id,
                        "seed": 101,
                        "current_problem": "test",
                    }
                )
                input_manifest.append(
                    {
                        "id": candidate_id,
                        "record_type": "candidate",
                        "target_name": target,
                    }
                )
                predictions.append({"id": candidate_id, "permeability_pred": 2.0})
                target_manifest.append(
                    {
                        "target_name": target,
                        "selected_chain": "P",
                        "structure_receptor_chains": "A",
                        "receptor_sequences_json": json.dumps({"A": "ACDEFG"}),
                        "native_peptide_seq": "ACDE",
                    }
                )
            for index, target in enumerate(all_targets):
                native_id = f"{target.lower()}_{9_000_000 + index}_ACDE_model"
                input_manifest.append(
                    {"id": native_id, "record_type": "native", "target_name": target}
                )
                predictions.append({"id": native_id, "permeability_pred": 1.0})

            candidates_path = temp / "candidates.csv"
            input_manifest_path = temp / "input_manifest.csv"
            predictions_path = temp / "predictions.csv"
            target_manifest_path = temp / "target_manifest.csv"
            out_dir = temp / "selected"
            write_csv(candidates_path, candidates)
            write_csv(input_manifest_path, input_manifest)
            write_csv(predictions_path, predictions)
            write_csv(target_manifest_path, target_manifest)

            selector.run(
                SimpleNamespace(
                    run_dir=str(temp),
                    plan=str(ROOT / "paper_clean_v28" / "rerun_t05" / "target_plan.json"),
                    candidates_csv=str(candidates_path),
                    input_manifest_csv=str(input_manifest_path),
                    target_manifest_csv=str(target_manifest_path),
                    permeability_csv=str(predictions_path),
                    out_dir=str(out_dir),
                    allow_partial_predictions=False,
                )
            )
            tasks = selector.read_csv(out_dir / "structure_tasks_for_shangge.csv")
            report = selector.read_json(out_dir / "selection_manifest.json")
            self.assertEqual(len(tasks), 13)
            self.assertEqual(report["quality_gate"], "NEEDS_MORE_CANDIDATES")
            self.assertEqual(len(report["shortfall_targets"]), 13)
            self.assertTrue((out_dir / "structure_tasks_for_shangge.fasta").is_file())
            self.assertTrue(
                (out_dir / "structure_inputs_for_shangge" / "3AV9_T0_5_rerun_01.fasta").is_file()
            )


if __name__ == "__main__":
    unittest.main()
