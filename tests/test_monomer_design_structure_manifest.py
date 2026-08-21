from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_clean_v28" / "07_prepare_monomer_design_structure_manifest.py"


def load_script():
    spec = importlib.util.spec_from_file_location("monomer_design_manifest", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = load_script()


FIELDS = [
    "input_mode",
    "sample_name",
    "selected_chains",
    "position_in_model",
    "target_token",
    "true_base_token",
    "pred_base_token",
    "prob_methyl_known_sequence",
    "prob_methyl_end_to_end",
]


def write_position_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def position_row(**overrides: str) -> dict[str, str]:
    row = {
        "input_mode": "strict_naturalized_input",
        "sample_name": "sample_1",
        "selected_chains": "A",
        "position_in_model": "1",
        "target_token": "A",
        "true_base_token": "A",
        "pred_base_token": "G",
        "prob_methyl_known_sequence": "0.600000004",
        "prob_methyl_end_to_end": "0.600000006",
    }
    row.update(overrides)
    return row


class MonomerDesignManifestTests(unittest.TestCase):
    def test_default_threshold_uses_strict_rounded_comparison_and_records_policy(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            position_csv = temp / "positions.csv"
            out_csv = temp / "manifest.csv"
            write_position_csv(
                position_csv,
                [
                    position_row(),
                    position_row(
                        position_in_model="2",
                        target_token="C",
                        true_base_token="C",
                        pred_base_token="T",
                        prob_methyl_known_sequence="0.600000006",
                        prob_methyl_end_to_end="0.6",
                    ),
                ],
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--position_csv",
                    str(position_csv),
                    "--out_csv",
                    str(out_csv),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with out_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["threshold"], "0.6")
        self.assertEqual(row["threshold_operator"], ">")
        self.assertEqual(row["rounding_policy"], "round(prob,8)")
        self.assertEqual(row["known_base_design_sequence"], "Ac")
        self.assertEqual(row["known_base_methyl_positions_1based"], "2")
        self.assertEqual(row["known_base_methyl_count"], "1")
        self.assertEqual(row["e2e_design_sequence"], "gT")
        self.assertEqual(row["e2e_methyl_positions_1based"], "1")
        self.assertEqual(row["e2e_methyl_count"], "1")

    def test_probability_must_be_finite_and_within_unit_interval(self):
        for value in ["nan", "inf", "-0.00000001", "1.00000001", "bad"]:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "methylation probability"
            ):
                manifest.validate_probability(value)

    def test_cli_rejects_invalid_probability_in_either_probability_column(self):
        for field in ["prob_methyl_known_sequence", "prob_methyl_end_to_end"]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_name:
                temp = Path(temp_name)
                position_csv = temp / f"bad_{field}.csv"
                out_csv = temp / f"bad_{field}_manifest.csv"
                write_position_csv(position_csv, [position_row(**{field: "nan"})])
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--position_csv",
                        str(position_csv),
                        "--out_csv",
                        str(out_csv),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "methylation probability must be finite and within [0, 1]",
                    result.stderr,
                )


if __name__ == "__main__":
    unittest.main()
