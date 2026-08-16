from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


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
v6_quota_resumer = load_module(
    "serine_qc_v6_quota_resumer",
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "08_resume_cyclic_representation_v6_quota.py",
)
v6_exhaustion_finalizer = load_module(
    "serine_qc_v6_exhaustion_finalizer",
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "09_finalize_cyclic_representation_v6_exhaustion.py",
)
triple_auditor = load_module(
    "serine_qc_v6_triple_auditor",
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "04_triple_audit_generation.py",
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
        cls.v6_plan_path = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "target_plan_cyclic_representation_v6.json"
        )
        cls.v6_plan = json.loads(cls.v6_plan_path.read_text(encoding="utf-8"))

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

    def test_legacy_seven_row_set_is_not_a_valid_methylated_compound_set(self):
        evidence = self.plan["frozen_target_evidence"]["3AV9"]
        self.assertEqual(evidence["design_seq"], evidence["design_seq"].upper())
        bridge_text = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "03_revalidate_frozen_structures.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"all_frozen_compounds_contain_at_least_one_methyl_token"',
            bridge_text,
        )
        self.assertIn(
            '"WITHDRAW_NON_METHYLATED_ROW_NOT_A_VALID_COMPOUND"',
            bridge_text,
        )

    def test_v6_regenerates_all_17_targets_without_grandfathering_old_annotations(self):
        checked = generator.validate_plan(self.v6_plan)
        self.assertEqual(checked["expected_target_count"], 17)
        self.assertEqual(checked["expected_raw_candidates"], 19_500)
        self.assertEqual(checked["planned_structure_handoff"], 245)
        self.assertEqual(self.v6_plan["frozen_targets"], [])
        self.assertEqual(len(self.v6_plan["withdrawn_historical_structure_controls"]), 7)
        withdrawn = {
            row["target_name"]: row
            for row in self.v6_plan["withdrawn_historical_structure_controls"]
        }
        self.assertIn("no methyl token", withdrawn["3AV9"]["reason"])
        self.assertEqual(set(checked["target_names"]), {
            "1SFI", "3AV9", "3AVA", "3AVB", "3AVF", "3AVG", "3AVH",
            "3AVI", "3AVJ", "3AVK", "3AVM", "3AVN", "3P8F", "3WNE",
            "3ZGC", "4K1E", "4KEL",
        })

    def test_v6_launcher_retrains_before_audit_and_never_creates_handoff(self):
        launcher = (
            ROOT / "run_serine_qc_cyclic_representation_v6.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("--cyclic-representation-augmentation", launcher)
        self.assertIn("$ParentCheckpoint", launcher)
        self.assertIn('"frankenstein_v28.pt"', launcher)
        self.assertIn("all 17; no pre-QC methyl annotation is grandfathered", launcher)
        self.assertIn("Shang-ge handoff:     NOT CREATED", launcher)
        self.assertNotIn("03_select_structure_first_handoff.py", launcher)
        self.assertLess(
            launcher.index("02_retrain_canonical_expert_heads.py"),
            launcher.index("07_audit_cyclic_representation_equivariance.py"),
        )
        self.assertLess(
            launcher.rindex("& $ResolvedPython @TrainingArguments"),
            launcher.rindex("& $ResolvedPython @AuditArguments"),
        )
        self.assertLess(
            launcher.rindex("& $ResolvedPython @AuditArguments"),
            launcher.rindex("& $ResolvedPython @GenerationArguments"),
        )

    def test_v6_training_jointly_rotates_labels_coordinates_and_resets_indices(self):
        trainer = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "02_retrain_canonical_expert_heads.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def expand_all_cyclic_training_representations(", trainer)
        self.assertIn("row_X[positions] = torch.roll", trainer)
        self.assertIn("row_S[positions] = torch.roll", trainer)
        self.assertIn("row_residue_idx[positions] = canonical_residue_idx", trainer)
        self.assertIn("CYCLIC_REPRESENTATION_PROTOCOL", trainer)
        self.assertIn("cyclic_representation_augmentation=args.", trainer)

    def test_v6_quota_resume_reuses_completed_artifacts_and_never_forces_restart(self):
        launcher = (
            ROOT / "run_serine_qc_cyclic_representation_v6.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("[switch]$ResumeQuota", launcher)
        self.assertIn("Do not combine -ResumeQuota with -Force", launcher)
        self.assertIn("08_resume_cyclic_representation_v6_quota.py", launcher)
        self.assertIn("09_finalize_cyclic_representation_v6_exhaustion.py", launcher)
        self.assertIn("& $ResolvedPython @ResumeArguments", launcher)
        self.assertIn("GPU step:   skipped; preserved V6 coverage state", launcher)
        self.assertNotIn("KMP_DUPLICATE_LIB_OK", launcher)
        finalize_call = launcher.index("& $ResolvedPython @FinalizeArguments")
        resume_call = launcher.index("& $ResolvedPython @ResumeArguments")
        self.assertLess(finalize_call, resume_call)
        self.assertLess(
            resume_call,
            launcher.index("& $ResolvedPython $TripleAuditor"),
        )

    def test_v6_quota_resume_retains_rows_and_uses_representation_ensemble(self):
        resumer = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "08_resume_cyclic_representation_v6_quota.py"
        ).read_text(encoding="utf-8")
        self.assertIn("source_ids <= set(candidate_ids)", resumer)
        self.assertIn("every_pre_resume_v6_payload_field_is_retained", resumer)
        self.assertIn(
            "cyclic_representation_known_sequence_methyl_probabilities", resumer
        )
        self.assertIn('TOPUP_STAGE = "V6_ADAPTIVE_QUOTA_TOPUP"', resumer)
        self.assertIn('threshold = float(plan["methyl_threshold"])', resumer)
        self.assertNotIn("threshold = 0.5", resumer)
        self.assertIn("remaining_target_budget = max(", resumer)
        self.assertIn("maximum_draws_per_target_total", resumer)
        self.assertEqual(
            v6_quota_resumer.false_checks(
                {"ok": True, "quota": False, "also_ok": 1}
            ),
            ["quota"],
        )

    def test_v6_fixed_budget_zero_yield_becomes_explicit_model_abstention(self):
        class ResumerFixture:
            INITIAL_STAGE = v6_quota_resumer.INITIAL_STAGE
            TOPUP_STAGE = v6_quota_resumer.TOPUP_STAGE

        initial_rows = [
            {
                "target_name": "3ZGC",
                "source_recovery_stage": ResumerFixture.INITIAL_STAGE,
                "seed": seed,
                "design_methyl_count": 0,
                "methyl_probabilities": "[]",
                "methyl_threshold": "0.6",
            }
            for seed in (101, 202)
        ]
        reserve_seeds = [
            606,
            707,
            808,
            909,
            1111,
            1212,
            1313,
            1414,
            1515,
            1616,
            1717,
            1818,
        ]
        topup_rows = [
            {
                "target_name": "3ZGC",
                "source_recovery_stage": ResumerFixture.TOPUP_STAGE,
                "seed": seed,
                "design_methyl_count": 0,
                "methyl_probabilities": "[]",
                "methyl_threshold": "0.6",
            }
            for seed in reserve_seeds
            for _ in range(1_000)
        ]
        manifest = {
            "methyl_threshold": 0.6,
            "adaptive_topup_budget": {
                "maximum_draws_per_target_total": 12_000,
                "draws_per_reserve_seed": 1_000,
            },
        }
        target_plan = {"sequences_per_seed": 1, "structure_quota": 10}
        unique_rows = [
            {
                "target_name": "3ZGC",
                "design_methyl_count": 0,
                "passes_methylation_hard_gate": 0,
            }
        ]

        approved = v6_exhaustion_finalizer.evaluate_exhausted_target(
            "3ZGC",
            target_plan,
            [101, 202],
            initial_rows + topup_rows,
            unique_rows,
            [],
            manifest,
            ResumerFixture,
        )
        self.assertTrue(approved["formal_abstention_approved"])
        self.assertEqual(approved["adaptive_topup_raw_draws"], 12_000)
        self.assertEqual(approved["novel_v6_methylated_candidates"], 0)
        self.assertIn("DO_NOT_LOWER_THRESHOLD", approved["release_action"])
        self.assertIn("DO_NOT_CREATE_STRUCTURE_TASK", approved["release_action"])

        under_budget = v6_exhaustion_finalizer.evaluate_exhausted_target(
            "3ZGC",
            target_plan,
            [101, 202],
            initial_rows + topup_rows[:-1],
            unique_rows,
            [],
            manifest,
            ResumerFixture,
        )
        self.assertFalse(under_budget["formal_abstention_approved"])
        self.assertFalse(
            under_budget["checks"]["at_least_12000_adaptive_rows_are_present"]
        )

    def test_v6_triple_audit_recomputes_formal_abstention_independently(self):
        auditor = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "04_triple_audit_generation.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "every_target_meets_structure_quota_or_has_verified_fixed_budget_abstention",
            auditor,
        )
        self.assertIn("fixed_12000_draw_topup_budget_is_present", auditor)
        self.assertIn("formal_abstention_errors", auditor)
        self.assertIn("candidate_artifact_sha256_unchanged_by_abstention", auditor)
        self.assertNotIn("KMP_DUPLICATE_LIB_OK", auditor)

    def test_v6_exhaustion_finalizer_is_idempotent_and_never_rewrites_candidates(self):
        def write_csv(path, rows, fields=None):
            fields = list(fields or rows[0])
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        annotation_mode = v6_quota_resumer.ANNOTATION_MODE
        annotation_context = v6_quota_resumer.ANNOTATION_CONTEXT
        initial_seeds = [101, 202]
        reserve_seeds = [
            606,
            707,
            808,
            909,
            1111,
            1212,
            1313,
            1414,
            1515,
            1616,
            1717,
            1818,
        ]

        def candidate(candidate_id, seed, draw_index, stage):
            return {
                "candidate_id": candidate_id,
                "target_name": "3ZGC",
                "design_seq": "GDEETGE",
                "design_natural_seq": "GDEETGE",
                "native_length": 7,
                "design_length": 7,
                "length_match": 1,
                "valid_token_gate": 1,
                "seed": seed,
                "draw_index_within_seed": draw_index,
                "base_log_probability_mean": -1.0,
                "design_methyl_count": 0,
                "design_methyl_rate": 0.0,
                "methyl_positions_1based": "[]",
                "methyl_probabilities": json.dumps([0.1] * 7),
                "methyl_probability_order_std": json.dumps([0.0] * 7),
                "methyl_probability_order_std_max": 0.0,
                "decoding_order_absolute": json.dumps(list(range(7))),
                "methyl_threshold": 0.6,
                "annotation_mode": annotation_mode,
                "annotation_context_policy": annotation_context,
                "annotation_visible_receptor_chains": 0,
                "annotation_order_ensemble_size": 7,
                "annotation_decoder_order_ensemble_size": 7,
                "annotation_representation_ensemble_size": 7,
                "source_recovery_stage": stage,
            }

        raw_rows = [
            candidate(
                f"initial_{seed}",
                seed,
                1,
                v6_quota_resumer.INITIAL_STAGE,
            )
            for seed in initial_seeds
        ]
        raw_rows.extend(
            candidate(
                f"topup_{seed}_{draw_index:04d}",
                seed,
                draw_index,
                v6_quota_resumer.TOPUP_STAGE,
            )
            for seed in reserve_seeds
            for draw_index in range(1, 1_001)
        )

        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            run_dir = temporary / "generation"
            run_dir.mkdir()
            model_path = temporary / "model.pt"
            model_path.write_bytes(b"hash-pinned-test-checkpoint")
            model_sha = v6_quota_resumer.sha256_file(model_path)
            plan_path = temporary / "plan.json"
            plan = {
                "protocol": "synthetic_cyclic_representation_v6",
                "temperature": 0.5,
                "methyl_threshold": 0.6,
                "seeds": initial_seeds,
                "expected_target_count": 1,
                "frozen_targets": [],
                "targets": [
                    {
                        "target_name": "3ZGC",
                        "sequences_per_seed": 1,
                        "structure_quota": 10,
                        "current_problem": "synthetic_zero_yield",
                    }
                ],
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            audit_path = temporary / "representation_audit.json"
            representation_audit = {
                "quality_gate": "PASS",
                "protocol": v6_quota_resumer.REPRESENTATION_AUDIT_PROTOCOL,
                "release_authorization": (
                    v6_quota_resumer.REPRESENTATION_AUDIT_AUTHORIZATION
                ),
                "model_sha256": model_sha,
                "plan_sha256": v6_quota_resumer.sha256_file(plan_path),
                "annotation_mode": annotation_mode,
            }
            audit_path.write_text(
                json.dumps(representation_audit), encoding="utf-8"
            )
            old_path = temporary / "historical.csv"
            write_csv(
                old_path,
                [{"target_name": "OLD", "design_seq": "AAAAAAA"}],
            )
            prior_path = temporary / "prior.csv"
            write_csv(
                prior_path,
                [
                    {
                        "candidate_id": f"prior_{index}",
                        "target_name": "OLD",
                        "design_seq": f"OLD{index}",
                    }
                    for index in range(1_333)
                ],
            )

            old_exact, old_natural = generator.old_design_keys(old_path)
            _, prior_exact, prior_natural = generator.validate_prior_handoff(
                prior_path
            )
            unique_rows = generator.aggregate_unique_candidates(
                raw_rows,
                old_exact,
                old_natural,
                prior_exact,
                prior_natural,
            )
            eligible_rows = [
                row
                for row in unique_rows
                if int(row["eligible_for_new_permeability_screen"])
            ]
            self.assertEqual(eligible_rows, [])
            write_csv(run_dir / "all_candidates.csv", raw_rows)
            write_csv(run_dir / "unique_candidates.csv", unique_rows)
            write_csv(
                run_dir / "methylated_new_candidates.csv",
                eligible_rows,
                unique_rows[0].keys(),
            )
            write_csv(
                run_dir / "generation_summary_by_target.csv",
                [
                    {
                        "target_name": "3ZGC",
                        "new_methylated_for_permeability": 0,
                        "planned_structure_quota": 10,
                        "enough_candidates_before_permeability": 0,
                    }
                ],
            )
            write_csv(
                run_dir / "target_manifest.csv",
                [{"target_name": "3ZGC", "selected_chain": "C"}],
            )
            embedded_audit = generator.audit_annotation_stability(raw_rows, [])
            self.assertEqual(embedded_audit["quality_gate"], "PASS")
            backup_dir = run_dir / "pre_quota_resume_backup"
            backup_dir.mkdir()
            write_csv(backup_dir / "all_candidates.csv", raw_rows[:2])
            (backup_dir / "generation_manifest.json").write_text(
                json.dumps({"quality_gate": "FAIL", "synthetic": True}),
                encoding="utf-8",
            )
            manifest = {
                "quality_gate": "FAIL",
                "quality_checks": {
                    "source_integrity": True,
                    "every_target_meets_pre_structure_candidate_quota": False,
                },
                "protocol": plan["protocol"],
                "model_sha256": model_sha,
                "model_expert_qc_protocol": (
                    v6_quota_resumer.REQUIRED_EXPERT_PROTOCOL
                ),
                "methyl_threshold": 0.6,
                "annotation_mode": annotation_mode,
                "annotation_context_policy": annotation_context,
                "annotation_visible_receptor_chains": 0,
                "train_deployment_context_match": True,
                "cyclic_representation_ensemble_enabled": True,
                "cyclic_representation_heldout_audit": {
                    **representation_audit,
                    "sha256": v6_quota_resumer.sha256_file(audit_path),
                },
                "raw_candidates_generated": len(raw_rows),
                "unique_candidates": len(unique_rows),
                "new_methylated_candidates_for_permeability": 0,
                "recovery_mode": v6_quota_resumer.RECOVERY_MODE,
                "source_v6_raw_candidates_retained": 2,
                "adaptive_topup_raw_candidates": 12_000,
                "adaptive_topup_budget": {
                    "reserve_seeds": reserve_seeds,
                    "draws_per_reserve_seed": 1_000,
                    "maximum_draws_per_target_per_resume": 12_000,
                },
                "raw_candidates_expected": len(raw_rows),
                "targets_below_pre_permeability_quota": ["3ZGC"],
                "annotation_stability_audit": embedded_audit,
                "historical_design_csv": str(old_path),
                "source_v6_initial_backup_manifest_sha256": (
                    v6_quota_resumer.sha256_file(
                        backup_dir / "generation_manifest.json"
                    )
                ),
                "source_v6_initial_backup_all_candidates_sha256": (
                    v6_quota_resumer.sha256_file(
                        backup_dir / "all_candidates.csv"
                    )
                ),
                "permeability_status": "DEFERRED_UNTIL_STRUCTURE_RETURNS",
            }
            (run_dir / "generation_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            args = SimpleNamespace(
                plan=str(plan_path),
                model_path=str(model_path),
                run_dir=str(run_dir),
                representation_audit_json=str(audit_path),
                old_designs_csv=str(old_path),
                prior_designs_csv=str(prior_path),
            )
            candidate_paths = [
                run_dir / "all_candidates.csv",
                run_dir / "unique_candidates.csv",
                run_dir / "methylated_new_candidates.csv",
            ]
            hashes_before = [
                v6_quota_resumer.sha256_file(path) for path in candidate_paths
            ]

            self.assertEqual(v6_exhaustion_finalizer.run(args), 0)
            self.assertEqual(v6_exhaustion_finalizer.run(args), 0)

            hashes_after = [
                v6_quota_resumer.sha256_file(path) for path in candidate_paths
            ]
            finalized = json.loads(
                (run_dir / "generation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(hashes_after, hashes_before)
            self.assertEqual(finalized["quality_gate"], "PASS")
            self.assertEqual(finalized["targets_formally_abstained"], ["3ZGC"])
            self.assertEqual(
                finalized["unresolved_targets_below_pre_permeability_quota"],
                [],
            )
            self.assertEqual(finalized["effective_structure_target_count"], 0)
            self.assertEqual(finalized["effective_planned_structure_handoff"], 0)

            independent_report = triple_auditor.audit(
                run_dir,
                plan_path,
                prior_path,
                temporary / "triple_audit",
            )
            self.assertEqual(independent_report["quality_gate"], "PASS")
            self.assertEqual(
                independent_report["release_status"],
                "READY_FOR_MANUAL_SCIENTIFIC_REVIEW_WITH_FORMAL_TARGET_ABSTENTION",
            )
            pass_3 = independent_report[
                "pass_3_novelty_coverage_workflow"
            ]
            self.assertEqual(
                pass_3["independently_verified_formal_target_abstentions"],
                ["3ZGC"],
            )
            self.assertEqual(pass_3["unresolved_quota_shortfalls"], [])

            eligible_path = run_dir / "methylated_new_candidates.csv"
            eligible_path.write_bytes(eligible_path.read_bytes() + b"\n")
            tampered_report = triple_auditor.audit(
                run_dir,
                plan_path,
                prior_path,
                temporary / "triple_audit_after_hash_tamper",
            )
            self.assertEqual(tampered_report["quality_gate"], "FAIL")
            self.assertFalse(
                tampered_report["pass_1_integrity"]["checks"][
                    "formal_target_abstention_metadata_and_candidate_hashes_are_valid"
                ]
            )

    def test_v6_quota_resume_validates_complete_source_rows_before_sampling(self):
        validated = {
            "target_names": ["A", "B"],
            "seeds": [101, 202],
            "targets": [
                {"target_name": "A", "sequences_per_seed": 1},
                {"target_name": "B", "sequences_per_seed": 1},
            ],
        }
        plan = {"methyl_threshold": 0.6}

        def row(candidate_id, target):
            return {
                "candidate_id": candidate_id,
                "target_name": target,
                "length_match": "1",
                "valid_token_gate": "1",
                "annotation_mode": v6_quota_resumer.ANNOTATION_MODE,
                "annotation_context_policy": v6_quota_resumer.ANNOTATION_CONTEXT,
                "annotation_visible_receptor_chains": "0",
                "annotation_representation_ensemble_size": "3",
                "design_length": "3",
                "methyl_threshold": "0.6",
            }

        initial_rows = [
            row("a101", "A"),
            row("a202", "A"),
            row("b101", "B"),
            row("b202", "B"),
        ]
        result = v6_quota_resumer.validate_source_rows(
            initial_rows, {}, plan, validated
        )
        self.assertTrue(result["source_candidate_ids_unique"])
        self.assertEqual(result["source_initial_rows_by_target"], {"A": 2, "B": 2})

        duplicated = [dict(item) for item in initial_rows]
        duplicated[-1]["candidate_id"] = duplicated[0]["candidate_id"]
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            v6_quota_resumer.validate_source_rows(duplicated, {}, plan, validated)

        mixed_policy = [dict(item) for item in initial_rows]
        mixed_policy[0]["methyl_threshold"] = "0.5"
        with self.assertRaisesRegex(RuntimeError, "mixed-policy"):
            v6_quota_resumer.validate_source_rows(
                mixed_policy, {}, plan, validated
            )

        resumed_rows = [dict(item) for item in initial_rows]
        for item in resumed_rows:
            item["source_recovery_stage"] = v6_quota_resumer.INITIAL_STAGE
        topup = row("a606", "A")
        topup["source_recovery_stage"] = v6_quota_resumer.TOPUP_STAGE
        resumed_rows.append(topup)
        resumed_manifest = {
            "recovery_mode": v6_quota_resumer.RECOVERY_MODE,
            "source_v6_raw_candidates_retained": 4,
            "adaptive_topup_raw_candidates": 1,
        }
        resumed_result = v6_quota_resumer.validate_source_rows(
            resumed_rows, resumed_manifest, plan, validated
        )
        self.assertEqual(
            resumed_result["source_stage_counts"],
            {
                v6_quota_resumer.INITIAL_STAGE: 4,
                v6_quota_resumer.TOPUP_STAGE: 1,
            },
        )

    def test_triple_audit_accounts_for_initial_v6_and_topup_rows(self):
        auditor = (
            ROOT
            / "paper_clean_v28"
            / "serine_qc_retrain"
            / "04_triple_audit_generation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("adaptive_v6_source_plus_topup_accounting", auditor)
        self.assertIn(
            "adaptive_v6_initial_backup_is_hash_pinned_and_fully_retained",
            auditor,
        )
        self.assertIn('recovery_stage_counts["V6_INITIAL_FULL_REGENERATION"]', auditor)
        self.assertIn('recovery_stage_counts["V6_ADAPTIVE_QUOTA_TOPUP"]', auditor)
        self.assertIn("== plan_raw", auditor)


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
        self.assertIn("function Compress-PortableArchive", launcher)
        self.assertIn("V5 IS WITHDRAWN", launcher)
        self.assertIn("V5 structure handoff is permanently blocked", launcher)
        self.assertIn("AcknowledgeWithdrawnV5Diagnostic", launcher)
        self.assertIn("path.relative_to(source).as_posix()", launcher)
        self.assertIn("archive.testzip()", launcher)
        self.assertNotRegex(launcher, r"(?m)^\s*Compress-Archive\b")
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
        portable_program = re.search(
            r"\$PortableArchiveProgram = @'\r?\n(.*?)\r?\n'@",
            launcher,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(portable_program)
        compile(
            portable_program.group(1),
            "<Ser V5 portable ZIP packager>",
            "exec",
        )

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
