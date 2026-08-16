from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


TORCH_TEST_PROGRAM = textwrap.dedent(
    r"""
    import sys
    from pathlib import Path

    import torch

    root = Path(sys.argv[1]).resolve()
    case = sys.argv[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from paper_clean_v28.clean_v28_common import (
        cyclic_representation_known_sequence_methyl_probabilities,
    )


    def tensors(length=8):
        X = torch.zeros((1, length, 4, 3), dtype=torch.float32)
        for position in range(length):
            X[0, position, :, 0] = float(position)
        S = torch.zeros((1, length), dtype=torch.long)
        mask = torch.ones((1, length), dtype=torch.float32)
        chain_M = torch.ones((1, length), dtype=torch.float32)
        residue_idx = torch.arange(length, dtype=torch.long).unsqueeze(0)
        chain_encoding = torch.zeros((1, length), dtype=torch.long)
        return X, S, mask, chain_M, residue_idx, chain_encoding


    if case == "tensor_index":
        class TensorIndexModel(torch.nn.Module):
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
                batch, length = S.shape
                expert = torch.full((batch, length, 20), -8.0, device=S.device)
                expert[:, 6, :] = 8.0
                return torch.zeros((batch, length, 20), device=S.device), expert

        result = cyclic_representation_known_sequence_methyl_probabilities(
            TensorIndexModel(), *tensors(), temperature=1.0
        )
        mean = result["mean"][0]
        assert torch.allclose(mean, mean[0].expand_as(mean), atol=1e-6)
        assert torch.equal(
            result["representation_count"][0], torch.full((8,), 8.0)
        )
    elif case == "physical_coordinate":
        class PhysicalCoordinateModel(torch.nn.Module):
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
                physical_id = X[:, :, 0, 0]
                logit = torch.where(
                    physical_id.eq(4.0),
                    torch.full_like(physical_id, 8.0),
                    torch.full_like(physical_id, -8.0),
                )
                expert = logit.unsqueeze(-1).expand(-1, -1, 20)
                return torch.zeros_like(expert), expert

        result = cyclic_representation_known_sequence_methyl_probabilities(
            PhysicalCoordinateModel(), *tensors(), temperature=1.0
        )
        mean = result["mean"][0]
        assert int(torch.argmax(mean).item()) == 4
        assert float(mean[4]) > 0.99
        assert float(torch.max(torch.cat([mean[:4], mean[5:]]))) < 0.01
    else:
        raise AssertionError(f"unknown test case: {case}")
    """
)


class CyclicRepresentationEnsembleTests(unittest.TestCase):
    """Run Torch checks in clean processes so Windows loads one OpenMP runtime.

    The full test suite imports NumPy/other compiled packages before it reaches
    this module. Importing PyTorch into that already-populated Windows process
    can load both ``libomp.dll`` and ``libiomp5md.dll`` and abort before unittest
    can report anything. Production V6 programs already run in fresh Python
    processes; these tests now mirror that safe boundary instead of setting the
    unsafe ``KMP_DUPLICATE_LIB_OK`` workaround.
    """

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
            raise RuntimeError(
                "PyTorch failed in a clean test process:\n"
                + probe.stdout
                + probe.stderr
            )

    def run_torch_case(self, case: str) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", TORCH_TEST_PROGRAM, str(ROOT), case],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                f"isolated Torch case {case!r} failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ),
        )

    def test_fixed_tensor_index_bias_is_spread_evenly_after_mapping_back(self):
        self.run_torch_case("tensor_index")

    def test_geometry_linked_site_maps_to_the_same_physical_residue(self):
        self.run_torch_case("physical_coordinate")


if __name__ == "__main__":
    unittest.main()
