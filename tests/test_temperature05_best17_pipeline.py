from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        "optional Windows structure-postprocessing test requires pandas"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
PREP_PATH = (
    ROOT
    / "paper_clean_v28"
    / "structure_metrics"
    / "16_prepare_temperature05_best17.py"
)
FINAL_PATH = (
    ROOT
    / "paper_clean_v28"
    / "structure_metrics"
    / "17_finalize_temperature05_best17.py"
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


prep = load(PREP_PATH, "temperature05_prepare")
finalize = load(FINAL_PATH, "temperature05_finalize")


def make_source(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for target_index in range(17):
        target = f"T{target_index:02d}"
        for temp in [0.01, 0.1, 0.2, 0.3, 0.5]:
            pdb = tmp_path / f"{target.lower()}_1_ACDE_model.pdb"
            pdb.write_text("", encoding="utf-8")
            rows.append(
                {
                    "target_name": target,
                    "temperature": temp,
                    "design_seq": "ACDE",
                    "pdb_file": pdb.name,
                    "pdb_path": str(pdb),
                    "global_complex_ca_rmsd": 1.0,
                    "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift": 2.0,
                    "global_complex_ca_rmsd_status": "ok",
                    "cyclic_peptide_ca_rmsd_status": "ok",
                    "complete_final_chain_ca_pairing_gate": 1,
                    "decoded_design_seq_matches_design_naturalized": 1,
                    "rmsd_rank_within_group": 1,
                }
            )
    return pd.DataFrame(rows)


class Temperature05PipelineTests(unittest.TestCase):
    def test_temperature05_selection_has_17_unique_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            selected = prep.validate_and_select(make_source(tmp_path), tmp_path)
            self.assertEqual(len(selected), 17)
            self.assertEqual(selected["target_name"].nunique(), 17)
            self.assertTrue(np.allclose(selected["temperature"], 0.5))
            self.assertTrue(
                selected["pdb_path"].map(Path).map(lambda p: p.is_absolute()).all()
            )

    def test_temperature05_selection_rejects_failed_pairing_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = make_source(tmp_path)
            idx = source.index[source["temperature"].eq(0.5)][0]
            source.loc[idx, "complete_final_chain_ca_pairing_gate"] = 0
            with self.assertRaisesRegex(
                ValueError, "complete_final_chain_ca_pairing_gate"
            ):
                prep.validate_and_select(source, tmp_path)

    def test_final_table_marks_non_estimable_metrics_as_missing(self):
        merged = pd.DataFrame(
            {
                "target_name": ["1SFI"],
                "temperature": [0.5],
                "pdb_file": ["x.pdb"],
                "pdb_path": ["x.pdb"],
                "design_seq": ["ACDE"],
                "design_natural_seq": ["ACDE"],
                "design_length": [4],
                "native_seq": ["ACDE"],
                "global_complex_ca_rmsd": [1.0],
                "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift": [
                    2.0
                ],
            }
        )
        table = finalize.make_final_table(merged)
        self.assertTrue(
            pd.isna(table.loc[0, "within_target_tm_diversity_1_minus_tm"])
        )
        self.assertTrue(pd.isna(table.loc[0, "energy_success_vs_native"]))
        self.assertEqual(
            table.loc[0, "within_target_tm_diversity_status"],
            "not_estimable_one_temperature_one_structure_per_target",
        )

    def test_controller_supports_safe_step8_resume_and_direct_wsl_mount_conversion(self):
        text = CONTROLLER_PATH.read_text(encoding="utf-8")
        self.assertIn("[ValidateRange(1, 12)][int]$StartStep = 1", text)
        self.assertIn("if ($StartStep -le 8)", text)
        self.assertIn("Convert-WindowsPathToWslMountPath", text)
        self.assertIn('return "/mnt/$Drive/$Rest"', text)
        self.assertNotIn("wslpath -a $Workspace", text)
        self.assertIn("Cannot continue: missing $Purpose file", text)
        self.assertNotIn('$BashCommand = @"', text)
        self.assertEqual(text.count(') -join "; "'), 3)


if __name__ == "__main__":
    unittest.main()
