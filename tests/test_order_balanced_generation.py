from __future__ import annotations

import importlib.util
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load_module(
    "order_balanced_generator",
    ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py",
)
triple_auditor = load_module(
    "order_balanced_triple_auditor",
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "04_triple_audit_generation.py",
)


def audit_row(
    target: str,
    natural_sequence: str,
    methyl_position: int,
    residue: str,
    probability: float = 0.9,
):
    sequence = list(natural_sequence)
    sequence[methyl_position - 1] = residue.lower()
    design = "".join(sequence)
    probabilities = [0.1] * len(sequence)
    probabilities[methyl_position - 1] = probability
    return {
        "target_name": target,
        "design_seq": design,
        "design_natural_seq": design.upper(),
        "methyl_probabilities": json.dumps(probabilities),
        "methyl_probability_order_std_max": 0.05,
        "annotation_mode": "cyclic_order_ensemble_known_natural_sequence",
    }


class ResultAnomalyGateTests(unittest.TestCase):
    def test_invalid_872_style_single_point_concentration_is_blocked(self):
        residues = "ACDEFGHIKLMNQRSTVWY"
        rows = []
        for index in range(120):
            residue = residues[index % len(residues)]
            natural = "AAAAAA" + residue + "A"
            rows.append(audit_row("3AVB", natural, 7, residue))
        report = generator.audit_annotation_stability(rows, rows)
        self.assertEqual(report["quality_gate"], "FAIL")
        self.assertEqual(report["eligible_site_position_counts"], {7: 120})
        self.assertFalse(
            report["quality_checks"][
                "no_single_position_exceeds_80_percent_of_sites"
            ]
        )

    def test_old_all_serine_signature_is_blocked(self):
        rows = []
        for index in range(120):
            position = index % 8 + 1
            rows.append(audit_row("3AVA", "SSSSSSSS", position, "S"))
        report = generator.audit_annotation_stability(rows, rows)
        self.assertEqual(report["quality_gate"], "FAIL")
        self.assertFalse(
            report["quality_checks"][
                "no_single_residue_exceeds_80_percent_of_sites"
            ]
        )

    def test_same_natural_sequence_with_different_annotation_is_blocked(self):
        first = audit_row("3AVA", "ACDEFGHI", 2, "C")
        second = audit_row("3AVA", "ACDEFGHI", 3, "D")
        report = generator.audit_annotation_stability([first, second], [first, second])
        self.assertEqual(report["quality_gate"], "FAIL")
        self.assertEqual(report["raw_inconsistent_annotation_groups"], 1)
        self.assertEqual(report["raw_probability_disagreement_groups"], 1)

    def test_balanced_deterministic_annotations_pass(self):
        residues = "ACDEFGHIKLMNQRSTVWY"
        rows = []
        for index in range(160):
            target = f"T{index:03d}"
            position = index % 8 + 1
            residue = residues[index % len(residues)]
            natural = list("ACDEFGHI")
            natural[position - 1] = residue
            rows.append(audit_row(target, "".join(natural), position, residue))
        report = generator.audit_annotation_stability(rows, rows)
        self.assertEqual(report["quality_gate"], "PASS")
        self.assertTrue(all(report["quality_checks"].values()))


class IndependentTripleAuditIntegrationTests(unittest.TestCase):
    def test_complete_synthetic_v3_run_passes_all_three_independent_audits(self):
        natural_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        methylatable = natural_alphabet.replace("P", "")

        def write_csv(path: Path, rows):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        def encode(index: int):
            left = natural_alphabet[(index // len(natural_alphabet)) % len(natural_alphabet)]
            right = natural_alphabet[index % len(natural_alphabet)]
            return list((left + right) * 4)

        rows = []
        for target in ("T1", "T2"):
            for index in range(80):
                position = index % 8 + 1
                residue = methylatable[index % len(methylatable)]
                natural_tokens = encode(index)
                natural_tokens[position - 1] = residue
                natural = "".join(natural_tokens)
                design_tokens = list(natural)
                design_tokens[position - 1] = residue.lower()
                design = "".join(design_tokens)
                probabilities = [0.1] * 8
                probabilities[position - 1] = 0.9
                shift = (index // 8) % 8
                order = list(range(8))[shift:] + list(range(8))[:shift]
                rows.append(
                    {
                        "candidate_id": f"{target}_{index:03d}",
                        "target_name": target,
                        "design_seq": design,
                        "design_natural_seq": natural,
                        "native_length": 8,
                        "design_length": 8,
                        "design_methyl_count": 1,
                        "methyl_positions_1based": json.dumps([position]),
                        "methyl_probabilities": json.dumps(probabilities),
                        "methyl_probability_order_std": json.dumps([0.02] * 8),
                        "methyl_probability_order_std_max": 0.02,
                        "annotation_mode": (
                            "cyclic_order_ensemble_known_natural_sequence"
                        ),
                        "annotation_order_ensemble_size": 8,
                        "decoding_order_absolute": json.dumps(order),
                        "occurrence_count": 1,
                        "eligible_for_new_permeability_screen": 1,
                    }
                )
        embedded_audit = generator.audit_annotation_stability(rows, rows)
        self.assertEqual(embedded_audit["quality_gate"], "PASS")

        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            run_dir = temporary / "generation"
            audit_out = temporary / "audit"
            run_dir.mkdir()
            plan_path = temporary / "plan.json"
            prior_path = temporary / "prior.csv"
            historical_path = temporary / "historical.csv"
            plan = {
                "protocol": (
                    "temperature_0.5_all_expert_qc_order_balanced_"
                    "structure_failure_recovery_v3"
                ),
                "temperature": 0.5,
                "methyl_threshold": 0.6,
                "seeds": [101],
                "frozen_targets": ["FROZEN"],
                "targets": [
                    {
                        "target_name": "T1",
                        "sequences_per_seed": 80,
                        "structure_quota": 10,
                    },
                    {
                        "target_name": "T2",
                        "sequences_per_seed": 80,
                        "structure_quota": 10,
                    },
                ],
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            write_csv(
                prior_path,
                [
                    {
                        "candidate_id": f"prior_{index}",
                        "target_name": "OLD",
                        "design_seq": f"OLD{index}",
                        "design_natural_seq": f"OLD{index}",
                    }
                    for index in range(1_333)
                ],
            )
            write_csv(
                historical_path,
                [
                    {
                        "target_name": "OLD",
                        "design_seq": "ACDEFGHI",
                    }
                ],
            )
            write_csv(run_dir / "all_candidates.csv", rows)
            write_csv(run_dir / "unique_candidates.csv", rows)
            write_csv(run_dir / "methylated_new_candidates.csv", rows)
            write_csv(
                run_dir / "target_manifest.csv",
                [
                    {"target_name": "T1", "selected_chain": "P"},
                    {"target_name": "T2", "selected_chain": "P"},
                ],
            )
            write_csv(
                run_dir / "generation_summary_by_target.csv",
                [
                    {
                        "target_name": "T1",
                        "new_methylated_for_permeability": 80,
                    },
                    {
                        "target_name": "T2",
                        "new_methylated_for_permeability": 80,
                    },
                ],
            )
            manifest = {
                "quality_gate": "PASS",
                "model_expert_qc_protocol": (
                    "canonical_clean_v28_all_expert_heads_corrected_labels_"
                    "order_balanced_v3"
                ),
                "raw_candidates_generated": 160,
                "unique_candidates": 160,
                "new_methylated_candidates_for_permeability": 160,
                "historical_design_csv": str(historical_path),
                "permeability_status": "DEFERRED_UNTIL_STRUCTURE_RETURNS",
                "annotation_stability_audit": embedded_audit,
            }
            (run_dir / "generation_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            report = triple_auditor.audit(
                run_dir,
                plan_path,
                prior_path,
                audit_out,
            )

        self.assertEqual(report["quality_gate"], "PASS")
        self.assertEqual(report["pass_1_integrity"]["quality_gate"], "PASS")
        self.assertEqual(report["pass_2_result_annotation"]["quality_gate"], "PASS")
        self.assertEqual(
            report["pass_3_novelty_coverage_workflow"]["quality_gate"], "PASS"
        )


class OrderProtocolSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = (
            ROOT / "paper_clean_v28" / "clean_v28_common.py"
        ).read_text(encoding="utf-8")
        cls.trainer = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "02_retrain_canonical_expert_heads.py"
        ).read_text(encoding="utf-8")
        cls.generator = (
            ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
        ).read_text(encoding="utf-8")
        cls.selector = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "03_select_structure_first_handoff.py"
        ).read_text(encoding="utf-8")
        cls.independent_audit = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "04_triple_audit_generation.py"
        ).read_text(encoding="utf-8")

    def test_model_accepts_and_validates_explicit_full_order(self):
        self.assertIn("decoding_order=None", self.common)
        self.assertIn("Every decoding_order row must be a full permutation", self.common)
        self.assertIn("complete_decoding_order", self.common)

    def test_expert_retrain_is_cyclic_and_cannot_promote_before_full_coverage(self):
        self.assertIn("cyclic_designed_decoding_order", self.trainer)
        self.assertIn("shift=epoch - 1", self.trainer)
        self.assertIn("MINIMUM_ORDER_COVERAGE_EPOCHS = 30", self.trainer)
        self.assertIn(
            "selected_epoch_has_complete_cyclic_order_coverage", self.trainer
        )
        self.assertIn("corrected_labels_order_balanced_v3", self.trainer)

    def test_generation_shares_outer_order_with_causal_mask_then_rescores(self):
        order_creation = self.generator.index("full_orders = complete_order_fn")
        model_order_use = self.generator.index("decoding_order=full_orders")
        final_ensemble = self.generator.index("ensemble_probability_fn(")
        self.assertLess(order_creation, model_order_use)
        self.assertLess(model_order_use, final_ensemble)
        self.assertIn("REQUIRED_ORDER_BALANCED_EXPERT_PROTOCOL", self.generator)
        self.assertIn("Generation annotation/coverage quality gate failed", self.generator)

    def test_handoff_requires_passed_generation_and_independent_audit_has_three_passes(self):
        self.assertIn('generation_manifest.get("quality_gate", "")', self.selector)
        self.assertIn("order-balanced v3 expert checkpoint", self.selector)
        self.assertIn("independent three-pass", self.selector)
        self.assertIn("pass_1_integrity", self.independent_audit)
        self.assertIn("pass_2_result_annotation", self.independent_audit)
        self.assertIn("pass_3_novelty_coverage_workflow", self.independent_audit)


if __name__ == "__main__":
    unittest.main()
