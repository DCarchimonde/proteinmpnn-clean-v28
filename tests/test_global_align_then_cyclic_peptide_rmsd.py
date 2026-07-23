import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRIC_SCRIPT = (
    ROOT
    / "paper_clean_v28"
    / "structure_metrics"
    / "13_compute_global_and_cyclic_peptide_ca_rmsd.py"
)
REVIEW_SCRIPT = (
    ROOT
    / "paper_clean_v28"
    / "structure_metrics"
    / "15_export_best85_pymol_pair_review.py"
)


def load_module(name, path, pymol_cmd):
    previous = sys.modules.get("pymol")
    sys.modules["pymol"] = types.SimpleNamespace(cmd=pymol_cmd)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("pymol", None)
        else:
            sys.modules["pymol"] = previous


def pdb_ca_line(serial, record_type, resname, chain, resseq, xyz):
    x, y, z = xyz
    return (
        f"{record_type:<6s}{serial:5d} {'CA':^4s} {resname:>3s} "
        f"{chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C"
    )


class Atom:
    def __init__(self, coord):
        self.coord = coord


class Model:
    def __init__(self, coordinates):
        self.atom = [Atom(coord) for coord in coordinates]


class MetricFakeCmd:
    def __init__(self):
        self.align_calls = []
        self.predicted_peptide = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        self.native_final_peptide = [(0, 1, 0), (1, 1, 0), (2, 1, 0)]

    def delete(self, _name):
        return None

    def read_pdbstr(self, _text, _name):
        return None

    def load(self, _path, _name):
        return None

    def sort(self, _name):
        return None

    def align(self, mobile, target, **kwargs):
        self.align_calls.append((mobile, target, kwargs))
        return (1.5, 5, 0, 1.5, 5, 25.0, 5)

    def iterate(self, _selection, _expression, space):
        space["atom_info"].update(
            {
                ("batch_pred", 1): {"chain": "A", "resi": "1", "resn": "ALA"},
                ("batch_native", 1): {"chain": "A", "resi": "1", "resn": "ALA"},
                ("batch_pred", 2): {"chain": "C", "resi": "1", "resn": "MAA"},
                ("batch_native", 2): {"chain": "Y", "resi": "1", "resn": "ALA"},
            }
        )

    def get_raw_alignment(self, _name):
        return [
            [("batch_pred", 1), ("batch_native", 1)],
            [("batch_pred", 2), ("batch_native", 2)],
        ]

    def count_atoms(self, selection):
        if selection == "batch_pred and name CA":
            return 5
        if selection == "batch_native and name CA":
            return 8
        if "batch_pred and chain C" in selection:
            return 3
        if "batch_native and chain Y" in selection:
            return 3
        return 0

    def get_model(self, selection):
        if "batch_pred and chain C" in selection:
            return Model(self.predicted_peptide)
        if "batch_native and chain Y" in selection:
            return Model(self.native_final_peptide)
        return Model([])


class GlobalAlignThenPeptideTests(unittest.TestCase):
    def test_complete_peptide_rmsd_uses_only_one_whole_complex_align(self):
        fake_cmd = MetricFakeCmd()
        module = load_module("global_then_peptide_metric", METRIC_SCRIPT, fake_cmd)
        module.cmd = fake_cmd

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folder = (
                root
                / "raw_external"
                / "pdb_highfold_temperature"
                / "pdb_highfold4_t03"
            )
            folder.mkdir(parents=True)
            pdb_path = folder / "TST_1_aCr_model.pdb"
            pdb_path.write_text(
                "\n".join(
                    [
                        pdb_ca_line(1, "ATOM", "ALA", "A", 1, (0, 0, 0)),
                        pdb_ca_line(2, "ATOM", "GLY", "A", 2, (1, 0, 0)),
                        pdb_ca_line(3, "HETATM", "MAA", "C", 1, (0, 0, 0)),
                        pdb_ca_line(4, "ATOM", "CYS", "C", 2, (1, 0, 0)),
                        pdb_ca_line(5, "HETATM", "MMO", "C", 3, (2, 0, 0)),
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            native_record = {
                "name": "TST",
                "seq_chain_A": "AG",
                "CA_chain_A": [[0, 0, 0], [1, 0, 0]],
                "seq_chain_X": "ACR",
                "CA_chain_X": [[10, 0, 0], [11, 0, 0], [12, 0, 0]],
                "seq_chain_Y": "ACR",
                "CA_chain_Y": [[0, 1, 0], [1, 1, 0], [2, 1, 0]],
            }
            result = module.evaluate_pdb(
                pdb_path,
                native_record,
                {
                    "target_name": "TST",
                    "temperature": "0.3",
                    "design_seq": "aCr",
                    "design_length": 3,
                    # Deliberately wrong legacy labels: they must not override
                    # the final-chain rule.
                    "predicted_peptide_chain": "B",
                    "native_peptide_chain": "X",
                },
                threshold=3.0,
                repo_root=root,
            )

        self.assertEqual(result["global_complex_ca_rmsd_status"], "ok")
        self.assertEqual(result["predicted_peptide_chain"], "C")
        self.assertEqual(result["native_peptide_chain"], "Y")
        self.assertEqual(
            result["metadata_predicted_chain_matches_final_chain"],
            0,
        )
        self.assertEqual(result["metadata_native_chain_matches_final_chain"], 0)
        self.assertEqual(result["whole_complex_align_call_count"], 1)
        self.assertEqual(result["cyclic_peptide_second_fit_performed"], 0)
        self.assertEqual(result["n_complete_positional_peptide_ca_pairs"], 3)
        self.assertEqual(result["complete_final_chain_ca_pairing_gate"], 1)
        self.assertAlmostEqual(
            float(
                result[
                    "cyclic_peptide_ca_rmsd_after_global_complex_alignment"
                ]
            ),
            1.0,
            places=6,
        )
        self.assertEqual(len(fake_cmd.align_calls), 1)
        mobile, target, kwargs = fake_cmd.align_calls[0]
        self.assertEqual(mobile, "batch_pred and name CA")
        self.assertEqual(target, "batch_native and name CA")
        self.assertEqual(kwargs["cycles"], 0)
        self.assertEqual(kwargs["transform"], 1)

    def test_review_pml_contains_one_global_align_and_no_peptide_align(self):
        module = load_module(
            "global_then_peptide_review",
            REVIEW_SCRIPT,
            types.SimpleNamespace(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pml = root / "pair.pml"
            module.write_pair_pml(
                pml,
                root / "predicted.pdb",
                root / "native.pdb",
                "C",
                "Y",
                "best85_pair_001",
            )
            text = pml.read_text(encoding="utf-8")

            align_lines = [
                line for line in text.splitlines() if line.startswith("align ")
            ]
            self.assertEqual(len(align_lines), 1)
            self.assertIn(
                "best85_predicted_complex and name CA",
                align_lines[0],
            )
            self.assertIn(
                "best85_native_complex and name CA",
                align_lines[0],
            )
            self.assertNotIn("chain C", align_lines[0])
            self.assertNotIn("chain Y", align_lines[0])

            pair_dir = root / "001_TST"
            pair_dir.mkdir()
            module.write_navigator(
                root,
                [
                    {
                        "review_index": 1,
                        "pair_folder": str(pair_dir),
                        "target_name": "TST",
                        "temperature": "0.3",
                        "global_complex_ca_rmsd_reproduced": "1.500000",
                        "cyclic_peptide_ca_rmsd_after_global_align_reproduced": (
                            "1.000000"
                        ),
                        "n_complete_positional_peptide_ca_pairs": 3,
                        "n_native_peptide_ca": 3,
                    }
                ],
            )
            master = (root / "OPEN_BEST85_REVIEW.pml").read_text(
                encoding="utf-8"
            )
            self.assertTrue(master.startswith("run "))
            self.assertNotIn('"', master)


if __name__ == "__main__":
    unittest.main()
