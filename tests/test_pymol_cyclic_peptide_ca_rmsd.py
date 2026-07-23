import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "paper_clean_v28"
    / "structure_metrics"
    / "14_compute_pymol_independent_cyclic_peptide_ca_rmsd.py"
)
SPEC = importlib.util.spec_from_file_location("cyclic_peptide_rmsd", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


NATURAL_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
NMETHYL_MAP = {"MAA": "a", "NCY": "c", "MMO": "r"}


class FakeCmd:
    def __init__(self):
        self.align_kwargs = None

    def delete(self, _name):
        return None

    def read_pdbstr(self, _text, _name):
        return None

    def sort(self, _name):
        return None

    def align(self, mobile, target, **kwargs):
        self.align_kwargs = {"mobile": mobile, "target": target, **kwargs}
        return (1.25, 3, 0, 1.25, 3, 9.0, 3)

    def get_raw_alignment(self, _name):
        return [1, 2, 3]


def pdb_ca_line(serial, record_type, resname, chain, resseq, xyz):
    x, y, z = xyz
    return (
        f"{record_type:<6s}{serial:5d} {'CA':^4s} {resname:>3s} "
        f"{chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C"
    )


class CyclicPeptideRmsdTests(unittest.TestCase):
    def test_naturalizes_upper_and_lower_project_tokens(self):
        self.assertEqual(
            MODULE.naturalize_design_sequence(
                "aCr",
                NATURAL_MAP,
                NMETHYL_MAP,
            ),
            "ACR",
        )

    def test_parent_mapping_allows_only_unk_design_fallback(self):
        residues = [
            {"record_type": "HETATM", "resname": "MAA"},
            {"record_type": "ATOM", "resname": "CYS"},
            {"record_type": "HETATM", "resname": "UNK"},
        ]
        audit = MODULE.audit_predicted_parent_mapping(
            residues,
            "ACR",
            NATURAL_MAP,
            NMETHYL_MAP,
        )
        self.assertEqual(audit["residue_parent_mapping_gate"], "PASS")
        self.assertEqual(audit["n_nmethyl_residues_mapped"], 1)
        self.assertEqual(audit["n_parent_inferred_from_design_for_unk"], 1)

    def test_end_to_end_uses_final_chain_same_align_parameters_and_joint_class(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folder = root / "raw_external" / "pdb_highfold_temperature" / "pdb_highfold4_t03"
            folder.mkdir(parents=True)
            pdb_path = folder / "TST_1_aCr_model.pdb"
            lines = [
                pdb_ca_line(1, "ATOM", "ALA", "A", 1, (0.0, 0.0, 0.0)),
                pdb_ca_line(2, "HETATM", "MAA", "B", 1, (1.0, 0.0, 0.0)),
                pdb_ca_line(3, "ATOM", "CYS", "B", 2, (2.0, 0.0, 0.0)),
                pdb_ca_line(4, "HETATM", "UNK", "B", 3, (3.0, 0.0, 0.0)),
                "END",
            ]
            pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            native_record = {
                "name": "TST",
                "seq_chain_A": "A",
                "CA_chain_A": [[0.0, 0.0, 0.0]],
                "seq_chain_Z": "GAS",
                "CA_chain_Z": [
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                ],
            }
            fake_cmd = FakeCmd()
            result = MODULE.evaluate_row(
                {
                    "target_name": "TST",
                    "temperature": "0.3",
                    "design_seq": "aCr",
                    "pdb_file": pdb_path.name,
                    "pdb_path": str(pdb_path.relative_to(root)),
                    "global_complex_ca_rmsd": "2.000000",
                    "global_complex_ca_rmsd_status": "ok",
                    "passes_global_complex_ca_rmsd_lt_threshold": "1",
                },
                native_record,
                NATURAL_MAP,
                NMETHYL_MAP,
                3.0,
                root,
                cmd_api=fake_cmd,
            )

        self.assertEqual(result["cyclic_peptide_ca_rmsd_status"], "ok")
        self.assertEqual(result["predicted_final_chain"], "B")
        self.assertEqual(result["native_final_chain"], "Z")
        self.assertEqual(result["n_predicted_cyclic_peptide_ca"], 3)
        self.assertEqual(result["n_native_cyclic_peptide_ca"], 3)
        self.assertEqual(result["full_cyclic_peptide_ca_alignment_coverage"], 1)
        self.assertEqual(result["passes_cyclic_peptide_ca_rmsd_lt_threshold"], 1)
        self.assertEqual(
            result[
                "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold"
            ],
            1,
        )
        self.assertEqual(fake_cmd.align_kwargs["cycles"], 0)
        self.assertEqual(fake_cmd.align_kwargs["matrix"], "BLOSUM62")
        self.assertEqual(
            fake_cmd.align_kwargs["mobile"],
            "cyclic_peptide_pred and name CA",
        )

    def test_joint_manifest_prefers_highest_confidence_among_joint_pass(self):
        designs = {
            ("TST", "0.3", "ACR"): [
                {
                    "target_name": "TST",
                    "temperature": "0.3",
                    "design_seq": "ACR",
                    "_all_design_row_index": 0,
                }
            ]
        }
        rows = [
            {
                "target_name": "TST",
                "temperature": "0.3",
                "design_seq": "ACR",
                "pdb_file": "low_conf.pdb",
                "pdb_ca_bfactor_mean": "70",
                "global_complex_ca_rmsd_status": "ok",
                "cyclic_peptide_ca_rmsd_status": "ok",
                "global_complex_ca_rmsd": "0.5",
                "cyclic_peptide_ca_rmsd": "0.5",
                "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold": "1",
            },
            {
                "target_name": "TST",
                "temperature": "0.3",
                "design_seq": "ACR",
                "pdb_file": "high_conf.pdb",
                "pdb_ca_bfactor_mean": "90",
                "global_complex_ca_rmsd_status": "ok",
                "cyclic_peptide_ca_rmsd_status": "ok",
                "global_complex_ca_rmsd": "2.0",
                "cyclic_peptide_ca_rmsd": "2.0",
                "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold": "1",
            },
        ]
        confidence, manifest = MODULE.build_unique_design_tables(
            designs,
            rows,
            3.0,
        )
        self.assertEqual(confidence[0]["pdb_file"], "high_conf.pdb")
        self.assertEqual(manifest[0]["pdb_file"], "high_conf.pdb")


if __name__ == "__main__":
    unittest.main()
