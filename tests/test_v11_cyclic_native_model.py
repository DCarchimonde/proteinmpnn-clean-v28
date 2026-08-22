from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_UTILS = ROOT / "model_utils.py"
COMMON = ROOT / "paper_clean_v28" / "clean_v28_common.py"
TRAINER = (
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "02_retrain_canonical_expert_heads.py"
)
AUDITOR = TRAINER.with_name("07_audit_cyclic_representation_equivariance.py")
RUNNER = ROOT / "run_v11_cyclic_native_1700_and_monomer.sh"
TOPUP = TRAINER.with_name("31_resume_cyclic_native_v11_quota.py")
PLAN = TRAINER.with_name("target_plan_v11_cyclic_native_rmsd_priority_1700.json")
MONOMER_FINALIZER = TRAINER.with_name("28_finalize_monomer_v10.py")


def load_monomer_finalizer():
    module_dir = str(MONOMER_FINALIZER.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(
        "monomer_v11_finalizer_contract", MONOMER_FINALIZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load monomer finalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TORCH_PROGRAM = textwrap.dedent(
    r"""
    import sys
    from pathlib import Path

    import torch

    root = Path(sys.argv[1]).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from model_utils import PositionalEncodings, ProteinFeatures

    encoding = PositionalEncodings(1, max_relative_feature=4)
    with torch.no_grad():
        encoding.linear.weight.copy_(torch.arange(10, dtype=torch.float32).view(1, 10))
        encoding.linear.bias.zero_()
    observed = encoding.forward_cyclic(
        torch.tensor([[[1]]]),
        torch.tensor([[[5]]]),
    )
    expected = 0.8 * 5.0 + 0.2 * 0.0
    assert torch.allclose(observed, torch.tensor([[[[expected]]]]), atol=1e-7)

    length = 7
    mask = torch.ones((1, length), dtype=torch.float32)
    chain = torch.zeros((1, length), dtype=torch.long)
    cyclic = torch.ones_like(mask)
    directed, cycle_length, pair = ProteinFeatures._cyclic_pair_offsets(
        mask, chain, cyclic
    )
    expected_directed = torch.remainder(
        torch.arange(length).view(length, 1)
        - torch.arange(length).view(1, length),
        length,
    )
    assert torch.equal(directed[0], expected_directed)
    assert torch.equal(cycle_length[0], torch.full((length, length), length))
    assert bool(pair.all())

    torch.manual_seed(11)
    features = ProteinFeatures(
        edge_features=8,
        node_features=8,
        num_positional_embeddings=4,
        num_rbf=2,
        top_k=length,
        augment_eps=0.0,
    ).eval()
    X = torch.randn((1, length, 4, 3), dtype=torch.float32)
    residue_idx = torch.arange(length).unsqueeze(0)

    def dense_edges(X_value, use_cyclic):
        edge, index = features(
            X_value,
            mask,
            residue_idx,
            chain,
            cyclic_mask=cyclic if use_cyclic else None,
        )
        dense = torch.zeros((1, length, length, edge.shape[-1]))
        dense.scatter_(
            2,
            index.unsqueeze(-1).expand(-1, -1, -1, edge.shape[-1]),
            edge,
        )
        return dense

    canonical = dense_edges(X, True)
    legacy = dense_edges(X, False)
    saw_legacy_boundary_change = False
    for shift in range(1, length):
        shifted_X = torch.roll(X, shifts=-shift, dims=1)
        shifted = dense_edges(shifted_X, True)
        mapped = torch.roll(shifted, shifts=(shift, shift), dims=(1, 2))
        assert torch.allclose(canonical, mapped, atol=2e-5, rtol=1e-5), shift

        shifted_legacy = dense_edges(shifted_X, False)
        mapped_legacy = torch.roll(
            shifted_legacy, shifts=(shift, shift), dims=(1, 2)
        )
        saw_legacy_boundary_change = saw_legacy_boundary_change or not torch.allclose(
            legacy, mapped_legacy, atol=2e-5, rtol=1e-5
        )
    assert saw_legacy_boundary_change
    """
)


class V11SourceContractTests(unittest.TestCase):
    def test_model_uses_boundary_marginalization_only_for_designed_chain(self):
        source = MODEL_UTILS.read_text(encoding="utf-8")
        common = COMMON.read_text(encoding="utf-8")
        self.assertIn("def forward_cyclic", source)
        self.assertIn("negative_weight", source)
        self.assertIn("pair_is_cyclic", source)
        self.assertIn("cyclic_mask = (", common)
        self.assertIn("chain_M * mask", common)
        self.assertIn("V11_MODEL_ARCHITECTURE_PROTOCOL", common)

    def test_v11_training_changes_model_and_objective_not_epoch_count(self):
        source = TRAINER.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAINER))
        functions = {
            node.name: ast.get_source_segment(source, node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        training = functions["train_all_expert_heads"]
        self.assertIn("enable_v11_cyclic_positional_parameters", training)
        self.assertIn("base_sequence_loss", training)
        self.assertIn("positional_anchor_loss", training)
        self.assertIn("full_grid_base_sequence_metrics", source)
        self.assertIn(
            "v11_base_cross_entropy_is_noninferior_to_legacy_parent", source
        )
        self.assertIn("v11_base_accuracy_is_noninferior_to_legacy_parent", source)
        self.assertIn("V11_POSITIONAL_STATE_KEYS", source)
        self.assertIn("--cyclic-native-model-v11", source)
        self.assertEqual(source.count("MINIMUM_ORDER_COVERAGE_EPOCHS = 30"), 1)

    def test_v11_audit_has_numerical_and_hard_call_gates(self):
        source = AUDITOR.read_text(encoding="utf-8")
        self.assertIn("V11_MAXIMUM_EQUIVARIANCE_SPAN = 1e-5", source)
        self.assertIn("v11_heldout_known_sequence_maximum_span_le_1e_5", source)
        self.assertIn("v11_heldout_end_to_end_maximum_span_le_1e_5", source)
        self.assertIn("v11_every_native_known_sequence_maximum_span_le_1e_5", source)
        self.assertIn("v11_every_native_end_to_end_maximum_span_le_1e_5", source)
        self.assertIn(
            "heldout_hard_calls_have_zero_cyclic_start_threshold_disagreement",
            source,
        )

    def test_v11_runner_is_isolated_and_passes_new_flags(self):
        runner = RUNNER.read_text(encoding="utf-8")
        topup = TOPUP.read_text(encoding="utf-8")
        self.assertIn("cyclic_native_v11_1700_monomer", runner)
        self.assertIn(PLAN.name, runner)
        self.assertIn("--cyclic-native-model-v11", runner)
        self.assertIn("--base-sequence-loss-weight", runner)
        self.assertIn("--maximum-base-ce-increase", runner)
        self.assertIn("--maximum-base-accuracy-drop", runner)
        self.assertIn("--required-expert-protocol", runner)
        self.assertIn("--release-protocol v11", runner)
        self.assertIn("--model-training-manifest", runner)
        self.assertIn("monomer_final_v11", runner)
        self.assertIn("monomer_eval_passes", runner)
        self.assertNotIn("V10_OUTPUT_ROOT", runner)
        self.assertIn(
            "canonical_clean_v28_all_expert_heads_cyclic_native_relative_positions_v11",
            topup,
        )
        self.assertIn("cyclic_native_relative_positions_heldout_gate_v11", topup)
        self.assertIn(
            "temperature_0.5_cyclic_native_relative_positions_v11_",
            PLAN.read_text(encoding="utf-8"),
        )
        plan_sha256 = hashlib.sha256(PLAN.read_bytes()).hexdigest()
        self.assertIn(plan_sha256, runner)

    def test_v11_monomer_finalizer_replaces_only_the_obsolete_v10_gate(self):
        source = MONOMER_FINALIZER.read_text(encoding="utf-8")
        self.assertIn(
            "corrected_monomer_cyclic_native_and_base_noninferiority_audit_v11",
            source,
        )
        self.assertIn("v11_training_manifest_checks", source)
        self.assertIn(
            "v11_deterministic_monomer_base_recovery_is_noninferior_to_parent_within_0_02",
            source,
        )
        self.assertIn(
            "base_head_predictions_match_deterministic_parent_for_all_1505_positions",
            source,
        )


class V11MonomerManifestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finalizer = load_monomer_finalizer()

    @staticmethod
    def valid_manifest():
        changed = [
            f"experts.{index}.{suffix}"
            for index in range(20)
            for suffix in ("weight", "bias")
        ] + [
            "features.embeddings.linear.bias",
            "features.embeddings.linear.weight",
        ]
        return {
            "quality_gate": "PASS",
            "checkpoint_ready_for_generation": True,
            "checkpoint_artifact_sha256": "a" * 64,
            "protocol": (
                "canonical_clean_v28_all_expert_heads_"
                "cyclic_native_relative_positions_v11"
            ),
            "model_architecture_protocol": (
                "proteinmpnn_boundary_marginalized_"
                "cyclic_relative_positions_v11"
            ),
            "cyclic_relative_positions": True,
            "maximum_base_cross_entropy_increase": 0.05,
            "maximum_base_accuracy_drop": 0.02,
            "maximum_equivariance_span_tolerance": 1e-5,
            "expected_changed_state_keys": changed,
            "changed_state_keys": changed,
            "changed_non_expert_keys": [
                "features.embeddings.linear.bias",
                "features.embeddings.linear.weight",
            ],
            "base_sequence_noninferiority_validation": {
                "legacy_parent": {"cross_entropy": 2.33, "accuracy": 0.306},
                "cyclic_native_parent_before_adaptation": {
                    "cross_entropy": 2.29,
                    "accuracy": 0.312,
                },
                "selected_v11": {"cross_entropy": 2.28, "accuracy": 0.313},
            },
            "training": {
                "selection": {
                    "best_epoch_maximum_training_representation_span": 0.0
                }
            },
            "quality_checks": {"all_release_checks": True},
            "artifacts": {"promoted_checkpoint": {"sha256": "a" * 64}},
        }

    def test_valid_v11_training_manifest_replays_all_scientific_gates(self):
        checks = self.finalizer.v11_training_manifest_checks(
            self.valid_manifest(), "a" * 64
        )
        self.assertTrue(all(checks.values()), checks)

    def test_v11_training_manifest_rejects_base_regression_or_wrong_model(self):
        payload = self.valid_manifest()
        payload["base_sequence_noninferiority_validation"]["selected_v11"] = {
            "cross_entropy": 3.0,
            "accuracy": 0.1,
        }
        checks = self.finalizer.v11_training_manifest_checks(payload, "b" * 64)
        self.assertFalse(
            checks[
                "v11_training_manifest_base_noninferiority_is_numerically_replayed"
            ]
        )
        self.assertFalse(
            checks["v11_training_manifest_is_bound_to_exact_promoted_model"]
        )


class V11TorchEquivarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if probe.returncode != 0 and "No module named 'torch'" in probe.stderr:
            raise unittest.SkipTest("torch is not installed in this test environment")
        if probe.returncode != 0:
            raise RuntimeError(probe.stdout + probe.stderr)

    def test_cyclic_feature_path_is_rotation_equivariant(self):
        completed = subprocess.run(
            [sys.executable, "-c", TORCH_PROGRAM, str(ROOT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
