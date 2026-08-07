from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MONOMER_STRUCTURE_PATH = (
    ROOT
    / "paper_clean_v28"
    / "structure_metrics"
    / "18_compute_monomer_structure_metrics.py"
)
FINAL_PATH = (
    ROOT
    / "paper_clean_v28"
    / "structure_metrics"
    / "20_finalize_temperature05_best17_and_monomer.py"
)
CONTROLLER_PATH = (
    ROOT
    / "paper_clean_v28"
    / "structure_metrics"
    / "run_temperature05_best17_all.ps1"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


monomer = load(MONOMER_STRUCTURE_PATH, "monomer_structure_metrics")
finalize = load(FINAL_PATH, "final_complex_monomer")


def rotation_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def make_residues(ca_coords: np.ndarray):
    rows = []
    for ca in ca_coords:
        rows.append(
            {
                "atoms": {
                    "N": ca + np.array([-0.3, 0.1, 0.0]),
                    "CA": ca,
                    "C": ca + np.array([0.4, -0.1, 0.2]),
                    "O": ca + np.array([0.6, -0.2, 0.4]),
                },
                "bfactors": {"CA": 90.0},
                "resname": "ALA",
            }
        )
    return rows


class MonomerMetricTests(unittest.TestCase):
    def setUp(self):
        self.reference_ca = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.3, 0.2, 0.4],
                [1.9, 1.7, -0.1],
                [0.4, 2.4, 0.8],
                [-1.0, 1.3, 0.2],
                [-0.7, -0.4, 1.1],
            ]
        )
        self.shift = 2
        shifted = np.array(
            [
                self.reference_ca[(index + self.shift) % len(self.reference_ca)]
                for index in range(len(self.reference_ca))
            ]
        )
        rotation = rotation_z(0.73)
        translation = np.array([5.0, -2.0, 1.7])
        self.design_ca = shifted @ rotation + translation
        self.reference = {"residues": make_residues(self.reference_ca)}
        # Apply the same transform to every backbone atom, not just CA.
        design_rows = []
        reference_shifted = [
            self.reference["residues"][
                (index + self.shift) % len(self.reference_ca)
            ]
            for index in range(len(self.reference_ca))
        ]
        for residue in reference_shifted:
            design_rows.append(
                {
                    "atoms": {
                        atom: np.asarray(coord) @ rotation + translation
                        for atom, coord in residue["atoms"].items()
                    },
                    "bfactors": {"CA": 91.0},
                    "resname": "ALA",
                }
            )
        self.design = {"residues": design_rows}

    def test_filename_variant_mapping(self):
        name = "Me_1021AAAsresult_proc0006_0083_4_WHWWCIVLLIL_model.pdb"
        match = monomer.PDB_NAME_RE.match(name)
        self.assertIsNotNone(match)
        self.assertEqual(match.group("sample"), "Me_1021AAAsresult_proc0006_0083")
        self.assertEqual(int(match.group("variant")), 4)
        self.assertEqual(match.group("sequence"), "WHWWCIVLLIL")

    def test_best_forward_cyclic_ca_fit_recovers_known_shift(self):
        metrics, residue_rows = monomer.cyclic_structure_metrics(
            self.reference,
            self.design,
            "aBCdEF",
        )
        self.assertEqual(metrics["best_forward_cyclic_shift"], self.shift)
        self.assertLess(metrics["ca_rmsd_best_forward_cyclic_shift"], 1e-6)
        self.assertLess(
            metrics["backbone_rmsd_after_ca_fit_best_forward_cyclic_shift"],
            1e-6,
        )
        self.assertGreater(metrics["ca_rmsd_fixed_order"], 0.1)
        self.assertEqual(metrics["reverse_order_allowed"], 0)
        self.assertEqual(len(residue_rows), 6)
        self.assertEqual(sum(row["is_e2e_methylated"] for row in residue_rows), 2)

    def test_tm_metric_tests_all_forward_shifts(self):
        old = monomer.tm_align

        def fake_tm_align(coords1, coords2, seq1, seq2):
            rotation, translation = monomer.kabsch_fit(coords1, coords2)
            aligned = monomer.apply_transform(coords1, rotation, translation)
            rmsd = monomer.rms_from_distances(
                np.linalg.norm(aligned - coords2, axis=1)
            )
            score = 1.0 / (1.0 + rmsd)
            return SimpleNamespace(
                tm_norm_chain1=score,
                tm_norm_chain2=score,
                rmsd=rmsd,
            )

        monomer.tm_align = fake_tm_align
        try:
            metrics = monomer.cyclic_tm_metrics(
                self.reference,
                self.design,
                "ABCDEF",
                "GHIJKL",
            )
        finally:
            monomer.tm_align = old
        self.assertEqual(metrics["tm_best_forward_cyclic_shift"], self.shift)
        self.assertAlmostEqual(
            metrics["tm_score_symmetric_best_forward_cyclic_shift"], 1.0, places=6
        )
        self.assertAlmostEqual(
            metrics["diversity_1_minus_tm_best_forward_cyclic_shift"],
            0.0,
            places=6,
        )


class FinalQualityTests(unittest.TestCase):
    def test_quality_gate_accepts_complete_core_and_warns_on_missing_monomer_permeability(self):
        complex_frame = pd.DataFrame(
            {
                "target_name": [f"T{i:02d}" for i in range(17)],
                "temperature": [0.5] * 17,
                "global_complex_ca_rmsd": [1.0] * 17,
                "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift": [
                    2.0
                ]
                * 17,
                "permeability_pred": [0.1] * 17,
                "rosetta_complex_score_per_residue": [1.0] * 17,
                "rosetta_cross_interface_energy_fixed_pose": [-1.0] * 17,
            }
        )
        monomer_frame = pd.DataFrame(
            {
                "sample_name": [f"M{i:03d}" for i in range(151)],
                "naturalized_structure_status": ["ok"] * 151,
                "naturalized_tm_status": ["ok"] * 151,
                "paired_energy_status": ["ok"] * 151,
                "variant_2_present": [1] * 151,
                "variant_4_present": [1] * 151,
                "reference_permeability_pred": [np.nan] * 151,
                "e2e_permeability_pred": [np.nan] * 151,
            }
        )
        checks = finalize.build_quality_checks(complex_frame, monomer_frame)
        required_failures = checks[
            checks["required"].eq(1) & checks["status"].eq("FAIL")
        ]
        warnings = checks[checks["status"].eq("WARN")]
        self.assertEqual(len(required_failures), 0)
        self.assertEqual(len(warnings), 2)

    def test_controller_keeps_step8_resume_and_adds_monomer_stages(self):
        text = CONTROLLER_PATH.read_text(encoding="utf-8")
        self.assertIn("[ValidateRange(1, 12)][int]$StartStep = 1", text)
        self.assertIn('[string]$TmCondaEnv = "tmdiv"', text)
        self.assertIn("if ($StartStep -le 8)", text)
        self.assertIn("18_compute_monomer_structure_metrics.py", text)
        self.assertIn("19_compute_monomer_pyrosetta_energy.py", text)
        self.assertIn("20_finalize_temperature05_best17_and_monomer.py", text)
        self.assertIn(
            "-n $TmCondaEnv python $MonomerStructureScript",
            text,
        )
        self.assertNotIn(
            "-n $WindowsCondaEnv python $MonomerStructureScript",
            text,
        )
        self.assertIn("import numpy, pandas, openpyxl", text)
        self.assertIn("import numpy, pandas, tmtools", text)
        self.assertIn("Expected tmtools==0.3.0", text)
        self.assertIn("$MonomerEnergyBaseCommand --limit 1", text)
        self.assertIn("Assert-ExactFileCount", text)
        self.assertIn("ExpectedCount 560", text)
        self.assertIn("QUALITY GATE:\\s*PASS", text)
        self.assertIn("Convert-WindowsPathToWslMountPath", text)
        self.assertNotIn("wslpath -a", text)
        self.assertNotIn('$BashCommand = @"', text)
        self.assertEqual(text.count(') -join "; "'), 3)
        self.assertEqual(text.count('"set -euo pipefail"'), 3)

    def test_monomer_tm_dependency_message_names_the_tmdiv_environment(self):
        text = MONOMER_STRUCTURE_PATH.read_text(encoding="utf-8")
        self.assertIn("tmdiv environment", text)
        self.assertNotIn("Activate the wain environment", text)


if __name__ == "__main__":
    unittest.main()
