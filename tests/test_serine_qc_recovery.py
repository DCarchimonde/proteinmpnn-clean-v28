from __future__ import annotations

import csv
import importlib.util
import json
import re
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
    "serine_qc_t05_generator",
    ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py",
)
selector = load_module(
    "serine_qc_structure_selector",
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "03_select_structure_first_handoff.py",
)
provenance = load_module(
    "serine_qc_provenance",
    ROOT / "paper_clean_v28" / "serine_qc_retrain" / "provenance.py",
)

from nmethyl.utils import nmethyl_config  # noqa: E402


class ResidueProvenanceTests(unittest.TestCase):
    def test_alphabet_and_maps_keep_checkpoint_dimensions_frozen(self):
        self.assertEqual(len(nmethyl_config.EXTENDED_AA_ALPHABET), 40)
        self.assertEqual(nmethyl_config.EXTENDED_AA_ALPHABET[-1], "X")
        self.assertNotIn("p", nmethyl_config.METHYL_AA_ALPHABET)
        self.assertFalse(
            set(nmethyl_config.NATURAL_RESIDUE_MAP)
            & set(nmethyl_config.NMETHYL_RESIDUE_MAP)
        )
        self.assertEqual(nmethyl_config.ALL_RESIDUE_MAP["SER"], "S")

    def test_ser_is_resolved_by_record_type_and_cn_atom(self):
        resolve = nmethyl_config.residue_token_from_pdb
        self.assertEqual(resolve("ATOM", "SER", ["N", "CA", "C"]), "S")
        self.assertEqual(resolve("HETATM", "5JP", ["N", "CA", "CN"]), "s")
        self.assertEqual(resolve("HETATM", "SER", ["N", "CA", "CN"]), "s")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve("HETATM", "SER", ["N", "CA"])
        with self.assertRaisesRegex(ValueError, "missing the expected CN"):
            resolve("HETATM", "5JP", ["N", "CA"])

    def test_minimal_pdb_rebuild_changes_only_atom_ser_label(self):
        pdb_text = "\n".join(
            [
                "ATOM      1  N   SER A   1       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  SER A   1       1.000   0.000   0.000  1.00 20.00           C",
                "HETATM    3  N   5JP A   2       2.000   0.000   0.000  1.00 20.00           N",
                "HETATM    4  CN  5JP A   2       3.000   0.000   0.000  1.00 20.00           C",
                "HETATM    5  N   SER A   3       4.000   0.000   0.000  1.00 20.00           N",
                "HETATM    6  CN  SER A   3       5.000   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        ) + "\n"
        source_row = {
            "name": "fixture",
            "seq": "sss",
            "seq_chain_A": "sss",
            "masked_list": ["A"],
            "visible_list": [],
            "coords_chain_A": {"N_chain_A": [[0.0, 0.0, 0.0]]},
            "untouched_metadata": "same",
        }
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            raw_dir = temporary / "raw"
            raw_dir.mkdir()
            (raw_dir / "fixture.pdb").write_text(pdb_text, encoding="utf-8")
            input_path = temporary / "train.jsonl"
            input_path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
            output_path = temporary / "corrected.jsonl"

            summary, audit = provenance.rebuild_split(
                "train",
                input_path,
                raw_dir,
                output_path,
                allow_unpinned_input=True,
            )
            corrected = provenance.read_jsonl(output_path)[0]

        self.assertEqual(corrected["seq"], "Sss")
        self.assertEqual(corrected["seq_chain_A"], "Sss")
        self.assertEqual(corrected["coords_chain_A"], source_row["coords_chain_A"])
        self.assertEqual(corrected["untouched_metadata"], "same")
        self.assertEqual(summary["s_to_S"], 1)
        self.assertEqual(summary["natural_S"], 1)
        self.assertEqual(summary["methyl_s"], 2)
        self.assertEqual(sum(int(row["changed"]) for row in audit), 1)

    def test_pinned_dataset_counts_are_explicit(self):
        self.assertEqual(
            provenance.EXPECTED_INPUTS["train"],
            {
                "rows": 600,
                "semantic_sha256": "0d6cd9ff4fb9bb385521c780967e01114d5fbb9caa66d550988c7df87da2d1da",
                "s_to_S": 242,
                "natural_S": 242,
                "methyl_s": 50,
                "natural_P": 307,
                "methyl_p": 0,
            },
        )
        self.assertEqual(provenance.EXPECTED_INPUTS["test"]["rows"], 151)
        self.assertEqual(provenance.EXPECTED_INPUTS["test"]["s_to_S"], 62)
        self.assertEqual(provenance.EXPECTED_INPUTS["test"]["methyl_s"], 12)


class FrozenRecoveryPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_path = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "target_plan_structure_failures.json"
        )
        cls.plan = json.loads(cls.plan_path.read_text(encoding="utf-8"))

    def test_only_ten_failed_targets_are_regenerated(self):
        checked = generator.validate_plan(self.plan)
        self.assertEqual(checked["expected_target_count"], 10)
        self.assertEqual(checked["expected_raw_candidates"], 11_500)
        self.assertEqual(checked["planned_structure_handoff"], 150)
        self.assertEqual(len(self.plan["frozen_targets"]), 7)
        self.assertFalse(set(checked["target_names"]) & set(self.plan["frozen_targets"]))

    def test_frozen_evidence_passes_joint_gate_and_has_no_lowercase_ser(self):
        for target in self.plan["frozen_targets"]:
            evidence = self.plan["frozen_target_evidence"][target]
            self.assertLess(float(evidence["global_rmsd"]), 3.0)
            self.assertLess(float(evidence["cyclic_rmsd"]), 3.0)
            self.assertNotIn("s", evidence["design_seq"])
            self.assertTrue(evidence["selected_chain"])

    def test_each_rerun_target_failed_at_least_one_frozen_gate(self):
        for target in self.plan["targets"]:
            self.assertTrue(
                float(target["prior_global_rmsd"]) >= 3.0
                or float(target["prior_cyclic_rmsd"]) >= 3.0
            )


class CompleteExpertRetrainIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_text = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "02_retrain_canonical_expert_heads.py"
        ).read_text(encoding="utf-8")

    def test_train_and_test_naturalize_labels_before_model_forward(self):
        self.assertGreaterEqual(
            self.trainer_text.count("S_forward = naturalize_tensor_for_input(S_label)"),
            2,
        )
        leaked_forwards = re.findall(
            r"model\(\s*X,\s*S_label,", self.trainer_text, flags=re.MULTILINE
        )
        self.assertEqual(leaked_forwards, [])

    def test_all_canonical_expert_heads_are_trainable_and_only_experts_change(self):
        self.assertIn("parameters = list(model.experts.parameters())", self.trainer_text)
        self.assertIn("parameter.requires_grad_(False)", self.trainer_text)
        self.assertIn("set(changed_keys) != ALL_EXPERT_STATE_KEYS", self.trainer_text)
        self.assertIn("deterministic_train_validation_split", self.trainer_text)
        self.assertIn("validation_balanced_bce", self.trainer_text)
        self.assertNotIn("strict=False", self.trainer_text)
        self.assertNotIn("splice", self.trainer_text.lower())

    def test_padding_is_made_safe_before_20_expert_gather(self):
        common_text = (
            ROOT / "paper_clean_v28" / "clean_v28_common.py"
        ).read_text(encoding="utf-8")
        self.assertIn("safe_base", common_text)
        self.assertIn("safe_base >= N_NATURAL", common_text)
        self.assertIn("invalid_base & selected_mask", common_text)
        self.assertIn("safe_base.unsqueeze(-1)", common_text)

    def test_promotion_metrics_match_t05_generation_temperature(self):
        self.assertIn("temperature=deployment_temperature", self.trainer_text)
        self.assertIn(
            "cyclic_known_sequence_methyl_probabilities", self.trainer_text
        )
        self.assertIn('default=0.5', self.trainer_text)
        self.assertIn("probability_methyl_deployment_scaled", self.trainer_text)

    def test_training_context_is_hard_gated_to_one_peptide_and_zero_receptors(self):
        self.assertIn("def require_peptide_only_training_context(", self.trainer_text)
        self.assertIn('record.get("visible_list", [])', self.trainer_text)
        self.assertIn(
            '"train_and_test_are_peptide_only_with_zero_visible_receptors"',
            self.trainer_text,
        )
        self.assertIn(
            '"peptide_chain_only_no_visible_receptor_chains"',
            self.trainer_text,
        )

    def test_trainer_evaluates_test_partition_only_after_training(self):
        training_position = self.trainer_text.rindex("train_all_expert_heads(")
        final_test_position = self.trainer_text.rindex("corrected_summary, corrected_per_residue")
        self.assertGreater(final_test_position, training_position)
        test_evaluations = re.findall(
            r"evaluate\(\s*model,\s*test_records,", self.trainer_text, flags=re.MULTILINE
        )
        self.assertEqual(len(test_evaluations), 1)
        self.assertNotIn("baseline_test", self.trainer_text)
        self.assertNotIn("baseline_frankenstein_v28", self.trainer_text)

    def test_checkpoint_is_promoted_only_after_quality_gate_passes(self):
        gate_position = self.trainer_text.index('quality_gate = "PASS"')
        promotion_position = self.trainer_text.index(
            "os.replace(candidate_checkpoint_path, checkpoint_path)"
        )
        self.assertGreater(promotion_position, gate_position)
        self.assertIn('"checkpoint_ready_for_generation": quality_gate == "PASS"', self.trainer_text)


class StructureFirstHandoffTests(unittest.TestCase):
    def test_windows_preflight_programs_are_quote_safe_and_compile(self):
        launcher = (ROOT / "run_serine_qc_recovery.ps1").read_text(encoding="utf-8")
        programs = re.findall(r"\$ProbeCode = '([^'\r\n]+)'", launcher)
        self.assertEqual(len(programs), 2)
        for program in programs:
            compile(program, "<Ser QC PowerShell Python probe>", "exec")
        self.assertNotIn("& $ResolvedPython -c", launcher)
        self.assertIn("& $PythonPath $ProgramPath", launcher)
        self.assertIn("Remove-Item -LiteralPath $TemporaryPath", launcher)

    def test_launchers_enforce_stage_order_and_defer_permeability(self):
        for filename in ("run_serine_qc_recovery.ps1", "run_serine_qc_recovery.sh"):
            text = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("--defer-permeability-until-structure", text)
            self.assertNotIn("02_select_after_permeability.py", text)
            self.assertIn("04_triple_audit_generation.py", text)
            self.assertIn("HOLD FOR MANUAL REVIEW", text)
            positions = [
                text.index("01_rebuild_provenance_labels.py"),
                text.index("02_retrain_canonical_expert_heads.py"),
                text.index("03_revalidate_frozen_structures.py"),
                text.rindex("01_generate_t05_multiseed.py"),
                text.index("04_triple_audit_generation.py"),
                text.index("03_select_structure_first_handoff.py"),
            ]
            self.assertEqual(positions, sorted(positions))

    def test_resume_launcher_reuses_v3_checkpoint_and_rows_without_retraining(self):
        launcher = (
            ROOT / "resume_serine_qc_peptide_only_v4.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("05_rescore_existing_generation_peptide_only.py", launcher)
        self.assertIn("serine_qc_order_balanced_v3", launcher)
        self.assertIn("serine_qc_peptide_only_v4", launcher)
        self.assertIn("no retraining; no base-sequence resampling", launcher)
        self.assertNotIn("02_retrain_canonical_expert_heads.py", launcher)
        self.assertNotIn("01_rebuild_provenance_labels.py", launcher)
        programs = re.findall(r"\$ProbeCode = '([^'\r\n]+)'", launcher)
        self.assertEqual(len(programs), 2)
        for program in programs:
            compile(program, "<Ser V4 PowerShell Python probe>", "exec")
        self.assertLess(
            launcher.index("03_revalidate_frozen_structures.py"),
            launcher.index("05_rescore_existing_generation_peptide_only.py"),
        )
        self.assertLess(
            launcher.index("05_rescore_existing_generation_peptide_only.py"),
            launcher.index("04_triple_audit_generation.py"),
        )

    def test_v5_launcher_reuses_v4_then_tops_up_before_structural_audit(self):
        launcher = (
            ROOT / "resume_serine_qc_structural_support_v5.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("06_top_up_quota_and_finalize_v5.py", launcher)
        self.assertIn("structural_position_support.json", launcher)
        self.assertIn("serine_qc_structural_support_v5", launcher)
        self.assertIn("[switch]$ReviewOnly", launcher)
        self.assertIn("no retraining, rescoring, or sampling", launcher)
        self.assertIn('"v4_generation_manifest.json"', launcher)
        self.assertIn('"v5_generation_manifest.json"', launcher)
        self.assertIn('"v5_target_manifest.csv"', launcher)
        self.assertIn('"review_bundle_manifest.json"', launcher)
        self.assertIn("serine_qc_structural_support_v5_shangge_handoff.zip", launcher)
        self.assertIn('"review_evidence"', launcher)
        self.assertNotIn("02_retrain_canonical_expert_heads.py", launcher)
        self.assertLess(
            launcher.index("$V5Topup"),
            launcher.index("$Auditor"),
        )
        self.assertLess(
            launcher.rindex("& $ResolvedPython @TopupArguments"),
            launcher.rindex("& $ResolvedPython $Auditor"),
        )
        programs = re.findall(r"\$ProbeCode = '([^'\r\n]+)'", launcher)
        self.assertEqual(len(programs), 2)
        for program in programs:
            compile(program, "<Ser V5 PowerShell Python probe>", "exec")

    def test_generator_has_no_deferred_permeability_write_path(self):
        text = (
            ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--defer-permeability-until-structure", text)
        self.assertGreaterEqual(
            text.count("if not args.defer_permeability_until_structure:"), 2
        )
        self.assertIn('"DEFERRED_UNTIL_STRUCTURE_RETURNS"', text)

    def test_generator_never_feeds_lowercase_annotation_back_to_model(self):
        text = (
            ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
        ).read_text(encoding="utf-8")
        self.assertIn("S_context,\n                mask,", text)
        self.assertIn("S_context[row_indices, positions] = sampled_base", text)
        self.assertIn("S_output = S_context.clone()", text)
        self.assertIn("S_output[:, masked_positions] = final_output_tokens", text)
        self.assertNotIn("S_context[row_indices, positions] = final_token", text)
        self.assertNotIn("S_context[row_indices, positions] = final_output_tokens", text)

    def test_generation_and_bridge_remove_receptor_for_final_expert_annotation(self):
        common = (ROOT / "paper_clean_v28" / "clean_v28_common.py").read_text(
            encoding="utf-8"
        )
        generator_text = (
            ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
        ).read_text(encoding="utf-8")
        bridge = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "03_revalidate_frozen_structures.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def peptide_only_annotation_tensors(", common)
        self.assertIn("chain_encoding = torch.zeros_like(residue_idx)", common)
        self.assertIn("peptide_only_annotation_tensors", generator_text)
        self.assertIn('record["visible_list"] = []', bridge)
        self.assertIn("annotation_visible_receptor_chains", bridge)

    def test_prior_exact_sequences_are_identified_for_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "prior.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["candidate_id", "target_name", "design_seq", "design_natural_seq"],
                )
                writer.writeheader()
                for index in range(1_333):
                    writer.writerow(
                        {
                            "candidate_id": f"old-{index}",
                            "target_name": "3AVA",
                            "design_seq": "ACsD",
                            "design_natural_seq": "ACSD",
                        }
                    )
            natural_index, exact_keys, manifest = selector.prior_handoff_index(path)

        self.assertIn(("3AVA", "ACsD"), exact_keys)
        self.assertIn(("3AVA", "ACSD"), natural_index)
        self.assertEqual(manifest["unique_exact_target_sequence_keys"], 1)

    def test_launchers_require_prior_1333_before_generation(self):
        powershell = (ROOT / "run_serine_qc_recovery.ps1").read_text(encoding="utf-8")
        shell = (ROOT / "run_serine_qc_recovery.sh").read_text(encoding="utf-8")
        self.assertIn("$Plan, $PriorHandoff", powershell)
        self.assertIn('"--prior_designs_csv", $PriorHandoff', powershell)
        self.assertIn('[[ ! -f "$PRIOR_HANDOFF" ]]', shell)
        self.assertIn('--prior_designs_csv "$PRIOR_HANDOFF"', shell)
        self.assertLess(
            powershell.index("--validate-prior-designs-only"),
            powershell.index("$PinnedSource = Prepare-PinnedSourceRepo"),
        )
        self.assertLess(
            shell.index("--validate-prior-designs-only"),
            shell.index('"$TASK_PYTHON" paper_clean_v28/serine_qc_retrain/01_rebuild'),
        )

    def test_selector_excludes_prior_naturalized_sequences_before_ranking(self):
        text = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "03_select_structure_first_handoff.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '(target, str(row["design_natural_seq"]).upper()) not in prior_index',
            text,
        )
        self.assertIn("prior_handoff_naturalized_sequences_excluded", text)

    def test_frozen_structures_are_bridged_without_regeneration(self):
        text = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "03_revalidate_frozen_structures.py"
        ).read_text(encoding="utf-8")
        self.assertIn("KEEP_EXISTING_PDB_NO_HIGHFOLD_RERUN", text)
        self.assertIn("PRE_QC_GENERATION_AUDITED_BY_FINAL_CHECKPOINT", text)
        self.assertIn("retained_result_design_seq", text)
        self.assertIn("final_model_suggested_design_seq", text)
        self.assertIn("DO_NOT_SUBSTITUTE_MODEL_SUGGESTION", text)

    def test_structure_selector_ranks_and_collapses_without_permeability(self):
        base = {
            "target_name": "3AVA",
            "candidate_id": "candidate-1",
            "design_seq": "ACsD",
            "design_natural_seq": "ACSD",
            "methyl_positions_1based": "[3]",
            "methyl_probabilities": "[0.1, 0.2, 0.9, 0.1]",
            "base_log_probability_mean": "-0.5",
            "seeds_observed": "101;202",
            "occurrence_count": "2",
            "natural_aa_recovery": "0.25",
        }
        first = selector.add_methyl_site_statistics(base)
        alternate = selector.add_methyl_site_statistics(
            dict(
                base,
                candidate_id="candidate-2",
                design_seq="AcsD",
                methyl_positions_1based="[2, 3]",
                methyl_probabilities="[0.1, 0.8, 0.9, 0.1]",
                base_log_probability_mean="-1.0",
            )
        )
        collapsed = selector.collapse_naturalized_variants([alternate, first])
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["candidate_id"], "candidate-1")
        self.assertEqual(collapsed[0]["naturalized_variant_count"], 2)
        self.assertNotIn("permeability_pred", collapsed[0])


if __name__ == "__main__":
    unittest.main()
