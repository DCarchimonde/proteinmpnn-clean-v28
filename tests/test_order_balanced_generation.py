from __future__ import annotations

import importlib.util
import csv
import hashlib
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
structural_support = load_module(
    "serine_structural_support",
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "structural_support.py",
)
quota_topup = load_module(
    "serine_v5_quota_topup",
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "06_top_up_quota_and_finalize_v5.py",
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
    zeros = [0.0] * len(sequence)
    return {
        "target_name": target,
        "design_seq": design,
        "design_natural_seq": design.upper(),
        "methyl_probabilities": json.dumps(probabilities),
        "methyl_threshold": 0.6,
        "methyl_probability_representation_min": json.dumps(probabilities),
        "methyl_probability_representation_max": json.dumps(probabilities),
        "methyl_probability_representation_span": json.dumps(zeros),
        "methyl_probability_representation_std": json.dumps(zeros),
        "methyl_probability_representation_by_start": json.dumps([probabilities]),
        "representation_threshold_disagreement_positions_1based": "[]",
        "representation_threshold_disagreement_count": 0,
        "methyl_probability_order_std_max": 0.05,
        "annotation_mode": (
            "peptide_only_cyclic_order_ensemble_known_natural_sequence"
        ),
        "annotation_context_policy": (
            "peptide_chain_only_no_visible_receptor_chains"
        ),
        "annotation_visible_receptor_chains": 0,
        "annotation_decoder_order_ensemble_size": len(sequence),
        "annotation_representation_ensemble_size": 1,
        "annotation_total_probability_ensemble_size": len(sequence),
    }


class ResultAnomalyGateTests(unittest.TestCase):
    @staticmethod
    def evidence_policy():
        return {
            "protocol": "historical_joint_lt5_supported_position_concentration_v10",
            "maximum_share_without_exemption": 0.8,
            "supported_positions_1based_by_target": {"3WNE": [2]},
        }

    @staticmethod
    def balanced_background_rows():
        residues = "ACDEFGHIKLMNQRSTVWY"
        rows = []
        for target_index, target in enumerate(("T1", "T2", "T3", "T4")):
            for index in range(120):
                position = index % 4 + 1
                residue = residues[(index + target_index) % len(residues)]
                digest = hashlib.sha256(
                    f"{target}:{index}".encode("ascii")
                ).digest()
                natural = [residues[value % len(residues)] for value in digest[:12]]
                natural[position - 1] = residue
                rows.append(audit_row(target, "".join(natural), position, residue))
        return rows

    def test_v10_historical_position_support_can_exempt_3wne_only(self):
        residues = "ACDEFGHIKLMNQRSTVWY"
        concentrated = [
            audit_row(
                "3WNE",
                "A" + residues[index % len(residues)] + "DEFGHI",
                2,
                residues[index % len(residues)],
            )
            for index in range(120)
        ]
        report = generator.audit_annotation_stability(
            [*concentrated, *self.balanced_background_rows()],
            [*concentrated, *self.balanced_background_rows()],
            self.evidence_policy(),
        )
        target = next(
            row for row in report["per_target_concentration"]
            if row["target_name"] == "3WNE"
        )
        self.assertEqual(report["quality_gate"], "PASS")
        self.assertTrue(target["position_concentration_exemption_applied"])
        self.assertEqual(
            target["historically_supported_dominant_positions_1based"], [2]
        )

    def test_v10_pool_records_unlabelled_3av_collapse_for_final_selection(self):
        residues = "ACDEFGHIKLMNQRSTVWY"
        concentrated = []
        for index in range(120):
            residue = residues[index % len(residues)]
            natural = "AAAAAA" + residue + "A"
            concentrated.append(audit_row("3AV9", natural, 7, residue))
        rows = [*concentrated, *self.balanced_background_rows()]
        report = generator.audit_annotation_stability(
            rows, rows, self.evidence_policy()
        )
        target = next(
            row for row in report["per_target_concentration"]
            if row["target_name"] == "3AV9"
        )
        self.assertEqual(report["quality_gate"], "PASS")
        self.assertTrue(
            report["target_local_concentration_is_final_selection_gate_not_pool_gate"]
        )
        self.assertFalse(target["position_concentration_exemption_applied"])
        self.assertFalse(target["position_gate_pass"])

    def test_single_point_concentration_is_a_hard_release_block(self):
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
            report["concentration_diagnostics"][
                "no_single_position_exceeds_80_percent_of_sites"
            ]
        )
        self.assertFalse(
            report["concentration_diagnostics"]
            ["no_target_has_single_position_above_80_percent_when_n_ge_30"]
        )
        self.assertIn("HARD_BLOCK", report["concentration_gate_policy"])

    def test_threshold_straddling_candidate_is_rejected(self):
        row = audit_row("T1", "ACDEFGHI", 2, "C", probability=0.9)
        row.update(
            {
                "candidate_id": "straddle",
                "methyl_threshold": 0.6,
                "methyl_probability_representation_min": json.dumps(
                    [0.1, 0.59, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                ),
                "methyl_probability_representation_max": json.dumps(
                    [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                ),
                "representation_threshold_disagreement_count": 1,
            }
        )
        self.assertFalse(generator.stable_cyclic_release_gate(row))
        report = generator.audit_annotation_stability([row], [row])
        self.assertEqual(report["quality_gate"], "FAIL")
        self.assertFalse(
            report["quality_checks"]
            ["every_eligible_candidate_is_stable_across_all_cyclic_starts"]
        )

    def test_serialized_reduction_noise_does_not_reject_stable_candidate(self):
        row = audit_row("T1", "ACDE", 2, "C", probability=0.62030825)
        by_start = [
            [0.1, 0.62030824, 0.1, 0.1],
            [0.1, 0.62030831, 0.1, 0.1],
            [0.1, 0.62030862, 0.1, 0.1],
            [0.1, 0.62030872, 0.1, 0.1],
        ]
        row.update(
            {
                "methyl_probability_representation_min": json.dumps(
                    [0.1, 0.62030824, 0.1, 0.1]
                ),
                "methyl_probability_representation_max": json.dumps(
                    [0.1, 0.62030872, 0.1, 0.1]
                ),
                "methyl_probability_representation_span": json.dumps(
                    [0.0, 0.00000048, 0.0, 0.0]
                ),
                "methyl_probability_representation_by_start": json.dumps(by_start),
                "annotation_mode": generator.CYCLIC_REPRESENTATION_ANNOTATION_MODE,
                "annotation_representation_ensemble_size": 4,
                "annotation_decoder_order_ensemble_size": 4,
                "annotation_total_probability_ensemble_size": 16,
            }
        )
        self.assertTrue(generator.stable_cyclic_release_gate(row))
        corrupted = dict(row)
        corrupted["methyl_probabilities"] = json.dumps(
            [0.1, 0.620305, 0.1, 0.1]
        )
        self.assertFalse(generator.stable_cyclic_release_gate(corrupted))

    def test_old_global_all_serine_signature_remains_blocked(self):
        rows = []
        for index in range(120):
            position = index % 8 + 1
            rows.append(audit_row(f"T{index:03d}", "SSSSSSSS", position, "S"))
        report = generator.audit_annotation_stability(rows, rows)
        self.assertEqual(report["quality_gate"], "FAIL")
        self.assertFalse(
            report["quality_checks"][
                "no_single_residue_exceeds_80_percent_of_sites"
            ]
        )

    def test_one_all_serine_target_is_preserved_as_final_selection_diagnostic(self):
        rows = []
        for index in range(40):
            position = index % 8 + 1
            digest = hashlib.sha256(f"BAD:{index}".encode("ascii")).digest()
            natural = [
                "ACDEFGHIKLMNQRSTVWY"[value % 19] for value in digest[:8]
            ]
            natural[position - 1] = "S"
            rows.append(audit_row("BAD", "".join(natural), position, "S"))
        residues = "ACDEFGHIKLMNQRSTVWY"
        for index in range(120):
            position = index % 8 + 1
            residue = residues[index % len(residues)]
            natural = list("ACDEFGHI")
            natural[position - 1] = residue
            rows.append(
                audit_row(f"MIXED{index:03d}", "".join(natural), position, residue)
            )
        report = generator.audit_annotation_stability(rows, rows)
        self.assertTrue(
            report["quality_checks"]["no_single_residue_exceeds_80_percent_of_sites"]
        )
        self.assertFalse(
            report["concentration_diagnostics"][
                "no_target_has_single_residue_above_80_percent_when_n_ge_30"
            ]
        )
        self.assertEqual(report["quality_gate"], "PASS")

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

    def test_gpu_tail_noise_is_removed_by_canonical_payload_persistence(self):
        first = audit_row("3AVA", "ACDEFGHI", 2, "C")
        second = audit_row("3AVA", "ACDEFGHI", 2, "C")
        first["candidate_id"] = "a"
        second["candidate_id"] = "b"
        second_probabilities = json.loads(second["methyl_probabilities"])
        second_probabilities[0] += 0.0000015
        second["methyl_probabilities"] = json.dumps(second_probabilities)
        generator.canonicalize_repeated_natural_annotations([first, second])
        report = generator.audit_annotation_stability([first, second], [first])
        self.assertEqual(report["quality_gate"], "PASS")
        self.assertEqual(report["raw_probability_disagreement_groups"], 0)


class StructuralConcentrationSupportTests(unittest.TestCase):
    @staticmethod
    def record(name, sequence, coordinates):
        return {
            "name": name,
            "seq": sequence,
            "seq_chain_P": sequence,
            "CA_chain_P": coordinates,
            "masked_list": ["P"],
            "visible_list": [],
        }

    def test_forward_cyclic_heldout_positive_supports_dominant_position(self):
        reference_coordinates = [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.0],
            [2.0, 1.0, 0.3],
            [1.6, 2.2, 0.8],
            [0.4, 2.8, 1.4],
            [-0.8, 2.0, 1.1],
            [-1.2, 0.7, 0.5],
            [-0.4, -0.4, 0.2],
        ]
        shift = 5
        target_coordinates = [
            reference_coordinates[(index - shift) % 8] for index in range(8)
        ]
        negative_coordinates = [
            [float(index) * 2.0, float(index % 3), float(index % 2)]
            for index in range(8)
        ]
        # A frozen quota can legitimately be below the old n>=30 diagnostic
        # threshold. Structural evidence must still cover that target.
        eligible = [audit_row("T1", "AAAAAASA", 7, "S") for _ in range(21)]
        report = structural_support.audit_dominant_position_structural_support(
            eligible_rows=eligible,
            native_rows=[self.record("T1", "AAAAAAAA", target_coordinates)],
            target_manifest_rows=[{"target_name": "T1", "selected_chain": "P"}],
            train_records=[self.record("TRAIN", "AAAAAAAA", negative_coordinates)],
            test_records=[
                self.record("POSITIVE", "AsAAAAAA", reference_coordinates),
                self.record("NEGATIVE", "AAAAAAAA", negative_coordinates),
            ],
        )
        self.assertEqual(report["quality_gate"], "PASS")
        self.assertEqual(report["minimum_sites"], 1)
        self.assertEqual(report["concentrated_target_count"], 1)
        evidence = report["concentrated_targets"][0]
        self.assertEqual(evidence["dominant_position_1based"], 7)
        self.assertEqual(
            evidence["heldout_test_nearest_methyl_positive"][
                "reference_position_1based"
            ],
            2,
        )
        self.assertTrue(evidence["structural_support_pass"])


class AdaptiveQuotaSourceValidationTests(unittest.TestCase):
    @staticmethod
    def manifest(checks):
        return {
            "protocol": quota_topup.V4_PROTOCOL,
            "model_expert_qc_protocol": quota_topup.REQUIRED_EXPERT_PROTOCOL,
            "annotation_mode": quota_topup.ANNOTATION_MODE,
            "annotation_context_policy": quota_topup.ANNOTATION_CONTEXT,
            "annotation_visible_receptor_chains": 0,
            "train_deployment_context_match": True,
            "raw_candidates_generated": 2,
            "quality_gate": "FAIL",
            "quality_checks": checks,
        }

    def test_only_documented_v4_failures_can_enter_v5(self):
        manifest = self.manifest(
            {
                "repeated_final_natural_sequences_have_identical_annotations": True,
                "no_single_position_exceeds_80_percent_of_sites": False,
                "every_target_meets_pre_structure_candidate_quota": False,
            }
        )
        result = quota_topup.validate_v4_source(manifest, [{}, {}])
        self.assertTrue(result["source_false_checks_allowed_for_v5"])

    def test_unrelated_v4_integrity_failure_is_never_bypassed(self):
        manifest = self.manifest({"raw_rows_are_valid": False})
        with self.assertRaisesRegex(RuntimeError, "not allowed to bypass"):
            quota_topup.validate_v4_source(manifest, [{}, {}])


class IndependentTripleAuditIntegrationTests(unittest.TestCase):
    def test_complete_synthetic_v4_run_passes_all_three_independent_audits(self):
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
                zeros = [0.0] * 8
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
                        "methyl_threshold": 0.6,
                        "methyl_probability_representation_min": json.dumps(probabilities),
                        "methyl_probability_representation_max": json.dumps(probabilities),
                        "methyl_probability_representation_span": json.dumps(zeros),
                        "methyl_probability_representation_std": json.dumps(zeros),
                        "methyl_probability_representation_by_start": json.dumps(
                            [probabilities]
                        ),
                        "representation_threshold_disagreement_positions_1based": "[]",
                        "representation_threshold_disagreement_count": 0,
                        "methyl_probability_order_std": json.dumps([0.02] * 8),
                        "methyl_probability_order_std_max": 0.02,
                        "annotation_mode": (
                            "peptide_only_cyclic_order_ensemble_known_natural_sequence"
                        ),
                        "annotation_context_policy": (
                            "peptide_chain_only_no_visible_receptor_chains"
                        ),
                        "annotation_visible_receptor_chains": 0,
                        "annotation_order_ensemble_size": 8,
                        "annotation_decoder_order_ensemble_size": 8,
                        "annotation_representation_ensemble_size": 1,
                        "annotation_total_probability_ensemble_size": 8,
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
                "annotation_mode": (
                    "peptide_only_cyclic_order_ensemble_known_natural_sequence"
                ),
                "annotation_context_policy": (
                    "peptide_chain_only_no_visible_receptor_chains"
                ),
                "annotation_visible_receptor_chains": 0,
                "train_deployment_context_match": True,
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
        cls.rescorer = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "05_rescore_existing_generation_peptide_only.py"
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
        self.assertIn("peptide_only_tensors_fn(X, S_context, mask, chain_M)", self.generator)
        self.assertIn("peptide_chain_only_no_visible_receptor_chains", self.generator)

    def test_v4_recovery_reuses_natural_sequences_and_scores_each_unique_key_once(self):
        self.assertIn(
            "RESCORE_EXISTING_V3_NATURAL_SEQUENCES_NO_RETRAIN_NO_RESAMPLING",
            self.rescorer,
        )
        self.assertIn("by_target[str(row[\"target_name\"]).upper()].add", self.rescorer)
        self.assertIn("annotation_by_key[(target, natural_sequence)]", self.rescorer)
        self.assertIn('"visible_list": []', self.rescorer)
        self.assertNotIn("train_all_expert_heads", self.rescorer)

    def test_handoff_requires_passed_generation_and_independent_audit_has_three_passes(self):
        self.assertIn('generation_manifest.get("quality_gate", "")', self.selector)
        self.assertIn("order-balanced v3 expert checkpoint", self.selector)
        self.assertIn("independent three-pass", self.selector)
        self.assertIn("pass_1_integrity", self.independent_audit)
        self.assertIn("pass_2_result_annotation", self.independent_audit)
        self.assertIn("pass_3_novelty_coverage_workflow", self.independent_audit)
        self.assertNotIn('"generation_manifest_passed"', self.independent_audit)


if __name__ == "__main__":
    unittest.main()
