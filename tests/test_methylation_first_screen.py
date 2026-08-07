from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("21_screen_methylation_first.py")


def load_module():
    spec = importlib.util.spec_from_file_location("methylation_first", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = load_module()


class ComplexScreenTests(unittest.TestCase):
    def make_complex(self):
        rows = []
        for target in ["T01", "T02"]:
            rows.extend(
                [
                    {
                        "target_name": target,
                        "temperature": 0.5,
                        "design_seq": "ACDE",
                        "pdb_file": f"{target}_nonmethyl.pdb",
                        "global_complex_ca_rmsd": 0.1,
                        "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift": 0.2,
                        "global_complex_ca_rmsd_status": "ok",
                        "cyclic_peptide_ca_rmsd_status": "ok",
                        "complete_final_chain_ca_pairing_gate": 1,
                        "decoded_design_seq_matches_design_naturalized": 1,
                    },
                    {
                        "target_name": target,
                        "temperature": 0.5,
                        "design_seq": "AcDE",
                        "pdb_file": f"{target}_methyl_best.pdb",
                        "global_complex_ca_rmsd": 1.0,
                        "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift": 2.0,
                        "global_complex_ca_rmsd_status": "ok",
                        "cyclic_peptide_ca_rmsd_status": "ok",
                        "complete_final_chain_ca_pairing_gate": 1,
                        "decoded_design_seq_matches_design_naturalized": 1,
                    },
                    {
                        "target_name": target,
                        "temperature": 0.5,
                        "design_seq": "ACdE",
                        "pdb_file": f"{target}_methyl_worse.pdb",
                        "global_complex_ca_rmsd": 0.8,
                        "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift": 4.0,
                        "global_complex_ca_rmsd_status": "ok",
                        "cyclic_peptide_ca_rmsd_status": "ok",
                        "complete_final_chain_ca_pairing_gate": 1,
                        "decoded_design_seq_matches_design_naturalized": 1,
                    },
                ]
            )
        return pd.DataFrame(rows)

    def test_methylation_gate_precedes_rmsd_ranking(self):
        candidates, best, strict = screen.screen_complex(
            self.make_complex(), expected_targets=2
        )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(best), 2)
        self.assertEqual(len(strict), 2)
        self.assertTrue(best["pdb_file"].str.endswith("_methyl_best.pdb").all())
        self.assertTrue(best["design_methyl_count"].eq(1).all())


class MonomerScreenTests(unittest.TestCase):
    def make_monomer(self):
        return pd.DataFrame(
            [
                {
                    "sample_name": "keep",
                    "e2e_design_sequence": "AcDE",
                    "e2e_methyl_count": 1,
                    "naturalized_ca_rmsd_best_forward_cyclic_shift": 2.0,
                    "permeability_delta_e2e_minus_reference": 0.1,
                    "rosetta_score_per_residue_delta_e2e_minus_reference": -1.0,
                },
                {
                    "sample_name": "methyl_not_priority",
                    "e2e_design_sequence": "ACdE",
                    "e2e_methyl_count": 1,
                    "naturalized_ca_rmsd_best_forward_cyclic_shift": 4.0,
                    "permeability_delta_e2e_minus_reference": 0.1,
                    "rosetta_score_per_residue_delta_e2e_minus_reference": -1.0,
                },
                {
                    "sample_name": "exclude",
                    "e2e_design_sequence": "ACDE",
                    "e2e_methyl_count": 0,
                    "naturalized_ca_rmsd_best_forward_cyclic_shift": 1.0,
                    "permeability_delta_e2e_minus_reference": 1.0,
                    "rosetta_score_per_residue_delta_e2e_minus_reference": -10.0,
                },
            ]
        )

    def test_monomer_filter_and_priority(self):
        selected, priority, excluded = screen.screen_monomer(self.make_monomer())
        self.assertEqual(set(selected["sample_name"]), {"keep", "methyl_not_priority"})
        self.assertEqual(priority["sample_name"].tolist(), ["keep"])
        self.assertEqual(excluded["sample_name"].tolist(), ["exclude"])

    def test_reported_methyl_count_must_match_lowercase_sequence(self):
        frame = self.make_monomer()
        frame.loc[0, "e2e_methyl_count"] = 0
        with self.assertRaisesRegex(ValueError, "disagrees"):
            screen.screen_monomer(frame)


if __name__ == "__main__":
    unittest.main()
