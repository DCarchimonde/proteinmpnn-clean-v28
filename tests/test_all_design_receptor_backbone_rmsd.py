import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "paper_clean_v28"
    / "structure_metrics"
    / "11_compute_all_design_receptor_backbone_rmsd.py"
)
SPEC = importlib.util.spec_from_file_location("all_design_rmsd", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


AA1_TO_3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "S": "SER",
}


def make_residues(sequence, ca_points):
    offsets = {
        "N": np.array([-0.45, 0.15, 0.05]),
        "CA": np.array([0.0, 0.0, 0.0]),
        "C": np.array([0.52, -0.12, 0.10]),
        "O": np.array([0.83, 0.21, -0.08]),
    }
    residues = []
    for index, (aa, ca) in enumerate(zip(sequence, ca_points), start=1):
        residues.append(
            {
                "resname": aa,
                "resseq": str(index),
                "icode": "",
                "atoms": {name: np.asarray(ca) + offset for name, offset in offsets.items()},
            }
        )
    return residues


def transform_residues(residues, rotation, translation, pre_shift=None):
    shift = np.zeros(3) if pre_shift is None else np.asarray(pre_shift, dtype=float)
    result = []
    for residue in residues:
        result.append(
            {
                "resname": AA1_TO_3.get(residue["resname"], residue["resname"]),
                "resseq": residue["resseq"],
                "icode": residue["icode"],
                "atoms": {
                    name: (np.asarray(xyz) + shift) @ rotation + translation
                    for name, xyz in residue["atoms"].items()
                },
            }
        )
    return result


def pdb_text(chains):
    lines = []
    serial = 1
    for chain_id, residues in chains.items():
        for residue in residues:
            for atom_name in MODULE.BACKBONE_ATOMS:
                x, y, z = residue["atoms"][atom_name]
                element = atom_name[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {residue['resname']:>3s} {chain_id:1s}"
                    f"{int(residue['resseq']):4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00 90.00          {element:>2s}  "
                )
                serial += 1
        lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


class ReceptorBackboneRmsdTests(unittest.TestCase):
    def test_kabsch_self_test(self):
        self.assertLess(MODULE.kabsch_self_test(), 1e-8)

    def test_semiglobal_alignment_skips_terminal_tag(self):
        result = MODULE.semi_global_sequence_pairs("HHHHHACDEFG", "ACDEFG")
        self.assertEqual(result["pairs"], [(5, 0), (6, 1), (7, 2), (8, 3), (9, 4), (10, 5)])
        self.assertEqual(result["identity"], 1.0)
        self.assertEqual(result["native_coverage"], 1.0)

    def test_peptide_translation_survives_receptor_fit(self):
        receptor_ca = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.7, 0.4, 0.3],
                [3.0, -0.7, 0.8],
                [4.4, 0.9, -0.2],
                [5.8, -0.3, 1.0],
                [7.1, 0.6, 0.2],
            ]
        )
        peptide_ca = np.array([[2.0, 3.0, 1.0], [3.2, 3.5, 0.4], [4.3, 2.8, 1.3]])
        native_receptor = make_residues("ACDEFG", receptor_ca)
        native_peptide = make_residues("GAS", peptide_ca)

        theta = 0.61
        forward_rotation = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        forward_translation = np.array([11.0, -6.0, 2.5])
        predicted_receptor = transform_residues(
            native_receptor, forward_rotation, forward_translation
        )
        predicted_peptide = transform_residues(
            native_peptide,
            forward_rotation,
            forward_translation,
            pre_shift=np.array([5.0, 0.0, 0.0]),
        )

        mapping = MODULE.receptor_mapping_candidates(
            predicted_chains={"A": predicted_receptor, "C": predicted_peptide},
            native_chains={"A": native_receptor, "C": native_peptide},
            predicted_peptide_chain="C",
            native_peptide_chains=["C"],
            design_len=3,
            min_identity=0.8,
            min_native_coverage=0.8,
        )[0]
        mobile, target = MODULE.index_paired_atoms(
            predicted_peptide, native_peptide, MODULE.BACKBONE_ATOMS
        )
        fixed_receptor_peptide = MODULE.rmsd(
            MODULE.apply_transform(np.asarray(mobile), mapping["rotation"], mapping["translation"]),
            np.asarray(target),
        )
        peptide_r, peptide_t = MODULE.kabsch_fit(np.asarray(mobile), np.asarray(target))
        self_superposed = MODULE.rmsd(
            MODULE.apply_transform(np.asarray(mobile), peptide_r, peptide_t), np.asarray(target)
        )

        self.assertEqual(mapping["n_backbone_atom_pairs"], 6 * 4)
        self.assertAlmostEqual(mapping["receptor_fit_rmsd"], 0.0, places=10)
        self.assertAlmostEqual(fixed_receptor_peptide, 5.0, places=10)
        self.assertAlmostEqual(self_superposed, 0.0, places=10)

    def test_end_to_end_pdb_metric_does_not_refit_peptide(self):
        receptor_ca = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.7, 0.4, 0.3],
                [3.0, -0.7, 0.8],
                [4.4, 0.9, -0.2],
                [5.8, -0.3, 1.0],
                [7.1, 0.6, 0.2],
            ]
        )
        peptide_ca = np.array([[2.0, 3.0, 1.0], [3.2, 3.5, 0.4], [4.3, 2.8, 1.3]])
        native_receptor = make_residues("ACDEFG", receptor_ca)
        native_peptide = make_residues("GAS", peptide_ca)

        theta = 0.61
        forward_rotation = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        forward_translation = np.array([11.0, -6.0, 2.5])
        predicted = {
            "A": transform_residues(native_receptor, forward_rotation, forward_translation),
            "C": transform_residues(
                native_peptide,
                forward_rotation,
                forward_translation,
                pre_shift=np.array([5.0, 0.0, 0.0]),
            ),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory) / "pdb_highfold4_t001"
            folder.mkdir()
            pdb_path = folder / "TST_1_GAS_model.pdb"
            pdb_path.write_text(pdb_text(predicted), encoding="utf-8")
            result = MODULE.evaluate_pdb(
                pdb_path=pdb_path,
                design_rows=[
                    {
                        "_row_index": 0,
                        "record_index": "0",
                        "selected_chains": "C",
                        "native_seq": "GAS",
                    }
                ],
                native_by_target={"TST": {"A": native_receptor, "C": native_peptide}},
                threshold=3.0,
                min_receptor_residue_pairs=3,
                min_receptor_identity=0.8,
                min_receptor_native_coverage=0.8,
                symmetry_receptor_fit_tolerance=0.25,
            )

        self.assertEqual(result["rmsd_status"], "ok")
        self.assertEqual(result["n_receptor_backbone_atom_pairs"], 6 * 4)
        self.assertEqual(result["n_peptide_backbone_atom_pairs"], 3 * 4)
        self.assertEqual(result["peptide_refit_performed"], 0)
        self.assertEqual(result["receptor_outlier_rejection_performed"], 0)
        self.assertAlmostEqual(
            float(result["peptide_backbone_rmsd_after_receptor_backbone_fit"]),
            5.0,
            places=2,
        )
        self.assertEqual(result["passes_peptide_backbone_rmsd_lt_threshold"], 0)

    def test_duplicate_native_peptide_chain_labels_do_not_inflate_rmsd(self):
        receptor_a_ca = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.7, 0.4, 0.3],
                [3.0, -0.7, 0.8],
                [4.4, 0.9, -0.2],
                [5.8, -0.3, 1.0],
                [7.1, 0.6, 0.2],
            ]
        )
        receptor_b_ca = receptor_a_ca + np.array([19.0, 4.0, -2.0])
        peptide_x_ca = np.array([[2.0, 3.0, 1.0], [3.2, 3.5, 0.4], [4.3, 2.8, 1.3]])
        peptide_y_ca = peptide_x_ca + np.array([19.0, 4.0, -2.0])

        native_a = make_residues("ACDEFG", receptor_a_ca)
        native_b = make_residues("ACDEFG", receptor_b_ca)
        native_x = make_residues("GAS", peptide_x_ca)
        native_y = make_residues("GAS", peptide_y_ca)

        theta = 0.37
        forward_rotation = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        forward_translation = np.array([-8.0, 5.0, 1.2])
        predicted = {
            # Deliberately swap the arbitrary receptor chain labels.
            "A": transform_residues(native_b, forward_rotation, forward_translation),
            "B": transform_residues(native_a, forward_rotation, forward_translation),
            # Predict the copy corresponding to native peptide Y, while the
            # design table anchors the native peptide as X.
            "C": transform_residues(native_y, forward_rotation, forward_translation),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory) / "pdb_highfold4_t001"
            folder.mkdir()
            pdb_path = folder / "TST2_1_GAS_model.pdb"
            pdb_path.write_text(pdb_text(predicted), encoding="utf-8")
            result = MODULE.evaluate_pdb(
                pdb_path=pdb_path,
                design_rows=[
                    {
                        "_row_index": 0,
                        "record_index": "0",
                        "selected_chains": "X",
                        "native_seq": "GAS",
                    }
                ],
                native_by_target={
                    "TST2": {"A": native_a, "B": native_b, "X": native_x, "Y": native_y}
                },
                threshold=3.0,
                min_receptor_residue_pairs=3,
                min_receptor_identity=0.8,
                min_receptor_native_coverage=0.8,
                symmetry_receptor_fit_tolerance=0.25,
            )

        self.assertEqual(result["rmsd_status"], "ok")
        self.assertEqual(result["equivalent_native_peptide_chains"], "X;Y")
        self.assertEqual(result["n_sequence_equivalent_receptor_mappings"], 2)
        self.assertGreaterEqual(result["n_receptor_fit_equivalent_mappings"], 1)
        self.assertAlmostEqual(
            float(result["peptide_backbone_rmsd_after_receptor_backbone_fit"]),
            0.0,
            places=2,
        )
        self.assertEqual(result["passes_peptide_backbone_rmsd_lt_threshold"], 1)


if __name__ == "__main__":
    unittest.main()
