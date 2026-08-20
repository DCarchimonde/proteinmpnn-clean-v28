from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = (
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "02_retrain_canonical_expert_heads.py"
)


TORCH_FULL_GRID_PROGRAM = textwrap.dedent(
    r"""
    import importlib.util
    import math
    import sys
    from pathlib import Path

    import torch

    root = Path(sys.argv[1]).resolve()
    path = root / "paper_clean_v28" / "serine_qc_retrain" / "02_retrain_canonical_expert_heads.py"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("v9_full_grid_trainer", path)
    assert spec and spec.loader
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.2))
            self.calls = []

        def forward(
            self,
            X,
            S,
            mask,
            chain_M,
            residue_idx,
            chain_encoding_all,
            decoding_order=None,
        ):
            assert decoding_order is not None
            self.calls.append(decoding_order.detach().clone())
            marker = decoding_order[:, 0].to(dtype=torch.float32).unsqueeze(-1)
            scalar = self.bias + marker
            logits = scalar.unsqueeze(-1).expand(S.shape[0], S.shape[1], 20)
            return torch.zeros_like(logits), logits

    length = 3
    rows = length
    X = torch.zeros((rows, length, 4, 3), dtype=torch.float32)
    S = torch.zeros((rows, length), dtype=torch.long)
    mask = torch.ones((rows, length), dtype=torch.float32)
    chain_M = torch.ones_like(mask)
    residue_idx = torch.arange(length).unsqueeze(0).expand(rows, -1).clone()
    chain_encoding = torch.zeros_like(residue_idx)
    valid = torch.ones((rows, length), dtype=torch.bool)
    model = FakeModel()

    mean = trainer.differentiable_full_decoder_order_mean_probabilities(
        model,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding,
        valid,
        (length,),
        temperature=0.5,
    )
    expected = sum(
        torch.sigmoid(torch.tensor((0.2 + shift) / 0.5))
        for shift in range(length)
    ) / length
    assert len(model.calls) == length, len(model.calls)
    assert torch.allclose(mean, torch.ones_like(mean) * expected, atol=1e-7)
    mean.sum().backward()
    assert model.bias.grad is not None
    assert torch.isfinite(model.bias.grad)
    assert float(model.bias.grad) > 0.0

    ser = trainer.NATURAL_AA_ALPHABET.index("S")
    ala = trainer.NATURAL_AA_ALPHABET.index("A")
    methyl_ser = trainer.EXTENDED_AA_ALPHABET.index("s")
    canonical_labels = torch.tensor([methyl_ser, ala, ser], dtype=torch.long)
    labels = torch.stack(
        [torch.roll(canonical_labels, shifts=-shift) for shift in range(length)]
    )
    physical_probabilities = [
        torch.tensor([0.8, 0.7, 0.2]),
        torch.tensor([0.6, 0.9, 0.3]),
        torch.tensor([0.7, 0.8, 0.4]),
    ]
    serialized_probabilities = torch.stack(
        [
            torch.roll(value, shifts=-shift)
            for shift, value in enumerate(physical_probabilities)
        ]
    ).requires_grad_()
    worst, consistency, maximum_span, coverage = (
        trainer.cyclic_worst_start_and_consistency_loss(
            serialized_probabilities,
            labels,
            valid,
            (length,),
            {ser: 1.0, ala: 1.0},
            active_base_indices=(ser, ala),
        )
    )
    assert float(worst) > 0.0
    assert float(consistency) > 0.0
    assert math.isclose(maximum_span, 0.2, abs_tol=1e-6)
    assert coverage[ser] == (1, 1), coverage
    assert coverage[ala] == (1, 0), coverage
    (worst + 0.25 * consistency).backward()
    assert serialized_probabilities.grad is not None
    assert torch.isfinite(serialized_probabilities.grad).all()

    try:
        trainer.expanded_cyclic_group_slices(valid, (2,))
    except RuntimeError as error:
        assert "boundaries" in str(error)
    else:
        raise AssertionError("malformed group boundary was accepted")

    malformed_valid = valid.clone()
    malformed_valid[1, -1] = False
    try:
        trainer.expanded_cyclic_group_slices(malformed_valid, (length,))
    except RuntimeError as error:
        assert "valid-position mask" in str(error) or "valid positions" in str(error)
    else:
        raise AssertionError("malformed valid mask was accepted")
    """
)


class FullGridSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TRAINER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(TRAINER))
        cls.functions = {
            node.name: ast.get_source_segment(cls.source, node)
            for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_cyclic_optimizer_uses_full_decoder_mean_before_worst_start(self):
        training = self.functions["train_all_expert_heads"]
        full_grid_call = "differentiable_full_decoder_order_mean_probabilities("
        worst_call = "cyclic_worst_start_and_consistency_loss("
        self.assertLess(training.index(full_grid_call), training.index(worst_call))
        self.assertIn("order_mean_probability,", training)
        self.assertIn("temperature=ensemble_temperature", training)
        cyclic_loss = training[training.index("if cyclic_representation_augmentation:") :]
        objective = cyclic_loss[
            cyclic_loss.index("loss = (") : cyclic_loss.index("loss.backward()")
        ]
        self.assertIn("worst_start_loss", objective)
        self.assertIn("consistency_loss", objective)
        self.assertNotIn("+ mean_bce_loss", objective)

    def test_full_grid_mean_is_differentiable_and_complete(self):
        helper = self.functions[
            "differentiable_full_decoder_order_mean_probabilities"
        ]
        self.assertIn("for decoder_shift in range(max(row_lengths))", helper)
        self.assertIn("probability_sum = probability_sum + probabilities", helper)
        self.assertIn("Full-grid decoder-order coverage is incomplete", helper)
        self.assertNotIn("detach()", helper)

        worst = self.functions["cyclic_worst_start_and_consistency_loss"]
        self.assertIn(
            "torch.where(reference_labels > 0.5, minimum, maximum)", worst
        )
        self.assertIn("span_by_base[base_index]).square().mean()", worst)

    def test_consistency_is_strictly_positive_and_temperature_is_threaded(self):
        main = self.functions["main"]
        validation = self.functions["validation_balanced_bce"]
        training = self.functions["train_all_expert_heads"]
        self.assertIn("args.representation_consistency_weight <= 0.0", main)
        self.assertIn("representation_consistency_weight <= 0.0", validation)
        self.assertIn("representation_consistency_weight <= 0.0", training)
        self.assertGreaterEqual(
            main.count("ensemble_temperature=args.deployment_temperature"),
            2,
        )
        self.assertIn('"training_ensemble_temperature"', self.source)

    def test_group_and_valid_masks_fail_closed(self):
        mask_guard = self.functions["require_complete_cyclic_training_mask"]
        group_guard = self.functions["expanded_cyclic_group_slices"]
        self.assertIn("torch.equal(complete_valid, selected)", mask_guard)
        self.assertIn("sum(lengths) != int(valid.shape[0])", group_guard)
        self.assertIn("torch.equal(row_positions, positions)", group_guard)
        self.assertIn("full_physical_start_x_full_decoder_order_grid", self.source)
        self.assertIn(
            '"cyclic_training_full_grid_matches_deployment_temperature"',
            self.source,
        )


class FullGridTorchTests(unittest.TestCase):
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
            raise RuntimeError("PyTorch import failed:\n" + probe.stdout + probe.stderr)

    def test_full_grid_values_gradients_and_fail_closed_guards(self):
        completed = subprocess.run(
            [sys.executable, "-c", TORCH_FULL_GRID_PROGRAM, str(ROOT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                "isolated full-grid cyclic training test failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
