from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "paper_clean_v28/serine_qc_retrain/02_retrain_canonical_expert_heads.py"
AUDITOR = ROOT / "paper_clean_v28/serine_qc_retrain/07_audit_cyclic_representation_equivariance.py"
GENERATOR = ROOT / "paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py"
TOPUP = ROOT / "paper_clean_v28/serine_qc_retrain/25_resume_cyclic_stability_v9_quota.py"
BASE_SCORER = ROOT / "paper_clean_v28/serine_qc_retrain/24_score_uniform_cyclic_base_v9.py"
TOPUP_ENGINE = ROOT / "paper_clean_v28/serine_qc_retrain/08_resume_cyclic_representation_v6_quota.py"
PLAN = ROOT / "paper_clean_v28/serine_qc_retrain/target_plan_cyclic_stability_v9_1700.json"
RUNNER = ROOT / "run_cyclic_stability_v9_1700.sh"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_scorer = load(BASE_SCORER, "base_scorer_cache_contract_tests")
topup_engine = load(TOPUP_ENGINE, "topup_resume_contract_tests")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    def test_partial_base_checkpoints_bind_program_and_direct_dependencies(self):
        source = BASE_SCORER.read_text(encoding="utf-8")
        for field in (
            "scorer_program_sha256",
            "generator_dependency_sha256",
            "clean_v28_common_dependency_sha256",
            "model_utils_dependency_sha256",
            "nmethyl_config_dependency_sha256",
        ):
            self.assertIn(field, source)
        self.assertIn("checkpoint_program_hashes.items()", source)

    def test_base_manifest_input_records_make_model_and_plan_cache_reusable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                name: root / name
                for name in (
                    "candidate.csv",
                    "baseline.csv",
                    "model.pt",
                    "generation.json",
                    "audit.json",
                    "native.jsonl",
                    "best.csv",
                    "plan.json",
                )
            }
            for name, path in paths.items():
                path.write_bytes((name + "\n").encode("utf-8"))
            records = base_scorer.scoring_input_records(
                candidate_path=paths["candidate.csv"],
                baseline_path=paths["baseline.csv"],
                model_path=paths["model.pt"],
                generation_manifest_path=paths["generation.json"],
                audit_path=paths["audit.json"],
                native_path=paths["native.jsonl"],
                best_path=paths["best.csv"],
                plan_path=paths["plan.json"],
            )
            # This is the exact path/hash shape consumed by runner manifest_passes.
            path_records = {
                str(Path(record["path"]).resolve()): record["sha256"]
                for record in records.values()
            }
            for label, filename in (("model", "model.pt"), ("plan", "plan.json")):
                path = paths[filename]
                self.assertEqual(records[label]["sha256"], sha256(path))
                self.assertEqual(path_records[str(path.resolve())], sha256(path))

    def test_partial_base_checkpoint_reuse_binds_runtime_and_numeric_contract(self):
        sequences = ["A", "C"]
        runtime = {
            "python_version": "3.10.0",
            "torch_version": "2.1.0",
            "torch_cuda_version": "12.1",
            "cudnn_version": 8900,
            "device": "cuda",
            "gpu": {"name": "GPU-A", "capability": [8, 0]},
            "deterministic": {"cudnn_deterministic": False},
        }
        parameters = {"batch_size": 32, "device_argument": "cuda"}
        programs = {"scorer_program_sha256": "a" * 64}
        checkpoint = {
            "protocol": base_scorer.SCORE_PROTOCOL,
            "target_name": "1SFI",
            "config_sha256": "b" * 64,
            **programs,
            "runtime_contract": runtime,
            "scoring_parameters": parameters,
            "sequence_set_sha256": "c" * 64,
            "sequence_count": len(sequences),
            "scores": {
                sequence: {
                    "cyclic_base_score_protocol": base_scorer.SCORE_PROTOCOL,
                    "cyclic_base_log_probability_start_by_decoder_order": json.dumps(
                        [[float(-index - 1)]]
                    ),
                    "cyclic_base_log_probability_by_start": json.dumps(
                        [float(-index - 1)]
                    ),
                    "cyclic_base_log_probability_mean": float(-index - 1),
                    "cyclic_base_log_probability_min": float(-index - 1),
                    "cyclic_base_log_probability_max": float(-index - 1),
                    "cyclic_base_log_probability_span": 0.0,
                    "cyclic_base_log_probability_std": 0.0,
                    "cyclic_base_physical_start_count": 1,
                    "cyclic_base_decoder_order_count_per_start": 1,
                    "cyclic_base_total_ensemble_size": 1,
                }
                for index, sequence in enumerate(sequences)
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "1sfi.json.gz"
            base_scorer.atomic_write_gzip_json(path, checkpoint)
            reopened = base_scorer.read_gzip_json(path)
        common = {
            "target": "1SFI",
            "config_sha256": "b" * 64,
            "program_hashes": programs,
            "scoring_parameters": parameters,
            "sequence_set_sha256": "c" * 64,
            "sequences": sequences,
        }
        self.assertTrue(
            base_scorer.target_checkpoint_is_reusable(
                reopened, runtime_contract=runtime, **common
            )
        )
        changed_runtime = json.loads(json.dumps(runtime))
        changed_runtime["gpu"]["name"] = "GPU-B"
        self.assertFalse(
            base_scorer.target_checkpoint_is_reusable(
                reopened, runtime_contract=changed_runtime, **common
            )
        )
        malformed = json.loads(json.dumps(reopened))
        malformed["scores"]["A"]["cyclic_base_log_probability_mean"] = "nan"
        self.assertFalse(
            base_scorer.target_checkpoint_is_reusable(
                malformed, runtime_contract=runtime, **common
            )
        )
        changed_parameters = dict(parameters, batch_size=16)
        self.assertFalse(
            base_scorer.target_checkpoint_is_reusable(
                reopened,
                runtime_contract=runtime,
                **dict(common, scoring_parameters=changed_parameters),
            )
        )

    def test_repeated_topup_rejects_runtime_batch_or_missing_contract(self):
        rows = [{"source_recovery_stage": topup_engine.TOPUP_STAGE}]
        runtime = {
            "python_version": "3.10.0",
            "torch_version": "2.1.0",
            "device": "cuda",
            "gpu": {"name": "GPU-A", "uuid": "physical-gpu-1"},
            "deterministic": {"cublas_workspace_config": ":4096:8"},
        }
        numerical = {
            "batch_size": 16,
            "draws_per_reserve_seed": 1000,
            "reserve_seeds": [606, 707],
        }
        manifest = {
            "topup_runtime_contract": runtime,
            "topup_numerical_contract": numerical,
        }
        topup_engine.validate_existing_topup_resume_contract(
            manifest, rows, runtime, numerical
        )
        uuid_only_change = json.loads(json.dumps(runtime))
        uuid_only_change["gpu"]["uuid"] = "physical-gpu-2"
        topup_engine.validate_existing_topup_resume_contract(
            manifest, rows, uuid_only_change, numerical
        )
        with self.assertRaisesRegex(RuntimeError, "runtime changed"):
            topup_engine.validate_existing_topup_resume_contract(
                manifest,
                rows,
                dict(runtime, torch_version="2.2.0"),
                numerical,
            )
        cublas_change = json.loads(json.dumps(runtime))
        cublas_change["deterministic"]["cublas_workspace_config"] = None
        with self.assertRaisesRegex(RuntimeError, "runtime changed"):
            topup_engine.validate_existing_topup_resume_contract(
                manifest,
                rows,
                cublas_change,
                numerical,
            )
        with self.assertRaisesRegex(RuntimeError, "batch or numerical"):
            topup_engine.validate_existing_topup_resume_contract(
                manifest,
                rows,
                runtime,
                dict(numerical, batch_size=8),
            )
        with self.assertRaisesRegex(RuntimeError, "lack a pinned"):
            topup_engine.validate_existing_topup_resume_contract(
                {}, rows, runtime, numerical
            )

    def test_first_topup_rejects_initial_generation_runtime_or_batch_change(self):
        runtime = {
            "python_version": "3.10.0",
            "torch_version": "2.1.0",
            "device": "cuda",
            "gpu": {"name": "GPU-A", "uuid": "physical-gpu-1"},
            "deterministic": {"cublas_workspace_config": ":4096:8"},
        }
        current = {
            "protocol": "v9-test",
            "temperature": 0.5,
            "methyl_threshold": 0.6,
            "batch_size": 16,
            "device_argument": "cuda",
            "resolved_device": "cuda",
            "allow_cpu": False,
            "initial_generation_seeds": [101, 202],
            "sampling_context": topup_engine.SAMPLING_CONTEXT,
            "annotation_context": topup_engine.ANNOTATION_CONTEXT,
        }
        initial = {
            "protocol": "v9-test",
            "temperature": 0.5,
            "methyl_threshold": 0.6,
            "batch_size": 16,
            "device_argument": "cuda",
            "resolved_device": "cuda",
            "allow_cpu": False,
            "initial_seeds": [101, 202],
            "cyclic_representation_ensemble": True,
            "sampling_context": topup_engine.SAMPLING_CONTEXT,
            "annotation_context": topup_engine.ANNOTATION_CONTEXT,
            "effective_seed_policy": (
                "base_seed_x_100000_plus_stable_target_offset"
            ),
        }
        manifest = {
            "initial_generation_runtime_contract": runtime,
            "initial_generation_numerical_contract": initial,
        }
        topup_engine.validate_initial_generation_contract(manifest, runtime, current)
        uuid_only_change = json.loads(json.dumps(runtime))
        uuid_only_change["gpu"]["uuid"] = "physical-gpu-2"
        topup_engine.validate_initial_generation_contract(
            manifest, uuid_only_change, current
        )
        with self.assertRaisesRegex(RuntimeError, "runtime differs"):
            topup_engine.validate_initial_generation_contract(
                manifest,
                dict(runtime, torch_version="2.2.0"),
                current,
            )
        cublas_change = json.loads(json.dumps(runtime))
        cublas_change["deterministic"]["cublas_workspace_config"] = None
        with self.assertRaisesRegex(RuntimeError, "runtime differs"):
            topup_engine.validate_initial_generation_contract(
                manifest,
                cublas_change,
                current,
            )
        with self.assertRaisesRegex(RuntimeError, "batch or numerical"):
            topup_engine.validate_initial_generation_contract(
                manifest,
                runtime,
                dict(current, batch_size=8),
            )
        with self.assertRaisesRegex(RuntimeError, "lacks a pinned"):
            topup_engine.validate_initial_generation_contract({}, runtime, current)

        generator_source = GENERATOR.read_text(encoding="utf-8")
        self.assertIn('"initial_generation_runtime_contract"', generator_source)
        self.assertIn('"initial_generation_numerical_contract"', generator_source)


if __name__ == "__main__":
    unittest.main()
