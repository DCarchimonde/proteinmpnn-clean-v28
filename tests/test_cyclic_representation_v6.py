from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
except ImportError:  # pragma: no cover - exercised in lightweight CI images
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in this test environment")
class CyclicRepresentationEnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from paper_clean_v28.clean_v28_common import (
            cyclic_representation_known_sequence_methyl_probabilities,
        )

        cls.score = staticmethod(
            cyclic_representation_known_sequence_methyl_probabilities
        )

    @staticmethod
    def tensors(length: int = 8):
        X = torch.zeros((1, length, 4, 3), dtype=torch.float32)
        for position in range(length):
            X[0, position, :, 0] = float(position)
        S = torch.zeros((1, length), dtype=torch.long)
        mask = torch.ones((1, length), dtype=torch.float32)
        chain_M = torch.ones((1, length), dtype=torch.float32)
        residue_idx = torch.arange(length, dtype=torch.long).unsqueeze(0)
        chain_encoding = torch.zeros((1, length), dtype=torch.long)
        return X, S, mask, chain_M, residue_idx, chain_encoding

    def test_fixed_tensor_index_bias_is_spread_evenly_after_mapping_back(self):
        class TensorIndexModel(torch.nn.Module):
            def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, decoding_order=None):
                batch, length = S.shape
                expert = torch.full((batch, length, 20), -8.0, device=S.device)
                expert[:, 6, :] = 8.0
                return torch.zeros((batch, length, 20), device=S.device), expert

        result = self.score(TensorIndexModel(), *self.tensors(), temperature=1.0)
        mean = result["mean"][0]
        self.assertTrue(torch.allclose(mean, mean[0].expand_as(mean), atol=1e-6))
        self.assertTrue(torch.equal(result["representation_count"][0], torch.full((8,), 8.0)))

    def test_geometry_linked_site_maps_to_the_same_physical_residue(self):
        class PhysicalCoordinateModel(torch.nn.Module):
            def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, decoding_order=None):
                physical_id = X[:, :, 0, 0]
                logit = torch.where(
                    physical_id.eq(4.0),
                    torch.full_like(physical_id, 8.0),
                    torch.full_like(physical_id, -8.0),
                )
                expert = logit.unsqueeze(-1).expand(-1, -1, 20)
                return torch.zeros_like(expert), expert

        result = self.score(PhysicalCoordinateModel(), *self.tensors(), temperature=1.0)
        mean = result["mean"][0]
        self.assertEqual(int(torch.argmax(mean).item()), 4)
        self.assertGreater(float(mean[4]), 0.99)
        self.assertLess(float(torch.max(torch.cat([mean[:4], mean[5:]]))), 0.01)


if __name__ == "__main__":
    unittest.main()
