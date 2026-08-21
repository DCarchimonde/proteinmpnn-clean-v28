from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from nmethyl.utils.nmethyl_config import (
    NATURAL_RESIDUE_MAP,
    NMETHYL_RESIDUE_MAP,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "paper_clean_v28" / "structure_metrics" / "22_audit_monomer_v10_pdb_reuse.py"
STRUCTURE_PATH = ROOT / "paper_clean_v28" / "structure_metrics" / "18_compute_monomer_structure_metrics.py"
CONTROLLER_PATH = ROOT / "paper_clean_v28" / "structure_metrics" / "run_temperature05_best17_all.ps1"
WINDOWS_ENTRY_PATH = ROOT / "run_v10_windows_structure_recalculation.ps1"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load(AUDIT_PATH, "monomer_v10_pdb_reuse_audit")
structure = load(STRUCTURE_PATH, "monomer_v10_structure_metrics")
NATURAL_TOKEN_TO_RESNAME = {
    token: resname for resname, token in NATURAL_RESIDUE_MAP.items()
}
METHYL_TOKEN_TO_RESNAME = {
    token: resname for resname, token in NMETHYL_RESIDUE_MAP.items()
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path) -> list[dict[str, str]]:
    rows = [
        {
            "sample_name": "M_001",
            "reference_natural_sequence": "ACD",
            "e2e_natural_sequence_for_structure_prediction": "WYA",
            "e2e_stable_methyl_design": "wYA",
        },
        {
            "sample_name": "M_002",
            "reference_natural_sequence": "EFG",
            "e2e_natural_sequence_for_structure_prediction": "RST",
            "e2e_stable_methyl_design": "RsT",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_autodl_monomer_manifest(
    design_manifest: Path,
    *,
    protocol: str | None = None,
    quality_gate: str = "PASS",
    quality_checks: dict[str, bool] | None = None,
    design_sha256: str | None = None,
) -> Path:
    path = design_manifest.with_name("monomer_v10_manifest.json")
    payload = {
        "protocol": protocol or audit.EXPECTED_AUTODL_MONOMER_PROTOCOL,
        "quality_gate": quality_gate,
        "quality_checks": quality_checks or {"fixture_complete": True},
        "artifacts": {
            "design_manifest": {
                # This is intentionally an AutoDL path. Windows validates bytes,
                # not the no-longer-applicable absolute source path.
                "path": "/root/autodl-tmp/run/monomer_final/monomer_v10_design_manifest_151.csv",
                "sha256": design_sha256 or sha256(design_manifest),
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_audit(
    design_manifest: Path,
    pdb_dir: Path,
    out_dir: Path,
    *,
    expected_samples: int = 2,
    expected_total_pdbs: int = 6,
):
    upstream = design_manifest.with_name("monomer_v10_manifest.json")
    if not upstream.is_file():
        upstream = write_autodl_monomer_manifest(design_manifest)
    return audit.audit_reuse(
        design_manifest,
        pdb_dir,
        out_dir,
        autodl_monomer_manifest=upstream,
        autodl_monomer_manifest_sha256=sha256(upstream),
        expected_samples=expected_samples,
        expected_total_pdbs=expected_total_pdbs,
    )


def atom_line(
    serial: int,
    residue: int,
    atom: str,
    resname: str,
    record_name: str,
) -> str:
    element = atom[0]
    return (
        f"{record_name:<6s}{serial:5d} {atom:^4s} {resname:>3s} A{residue:4d}    "
        f"{float(residue):8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"  1.00 90.00          {element:>2s}\n"
    )


def write_pdb(
    path: Path,
    sequence: str,
    *,
    missing_ca_position: int | None = None,
    omit_cn_positions: set[int] | None = None,
) -> None:
    lines = []
    serial = 1
    omit_cn_positions = set(omit_cn_positions or set())
    for position, token in enumerate(sequence, start=1):
        if token.islower():
            resname = METHYL_TOKEN_TO_RESNAME[token]
            record_name = "HETATM"
            atom_names = ["N", "CA", "C", "O", "CN"]
        else:
            resname = NATURAL_TOKEN_TO_RESNAME[token]
            record_name = "ATOM"
            atom_names = ["N", "CA", "C", "O"]
        for atom in atom_names:
            if atom == "CA" and position == missing_ca_position:
                continue
            if atom == "CN" and position in omit_cn_positions:
                continue
            lines.append(atom_line(serial, position, atom, resname, record_name))
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_inventory(
    pdb_dir: Path,
    rows: list[dict[str, str]],
    *,
    second_variant3: str = "RSt",
    bad_variant4: bool = False,
    bad_variant4_content: bool = False,
    missing_variant2_ca: bool = False,
    missing_variant3_cn: bool = False,
) -> None:
    pdb_dir.mkdir()
    for index, row in enumerate(rows):
        sample = row["sample_name"]
        variant2 = row["reference_natural_sequence"]
        variant4 = row["e2e_natural_sequence_for_structure_prediction"]
        variant4_content = variant4
        variant3 = row["e2e_stable_methyl_design"]
        if index == 1:
            variant3 = second_variant3
            if bad_variant4:
                variant4 = "RAT"
                variant4_content = variant4
        if index == 0 and bad_variant4_content:
            variant4_content = "WFA"
        write_pdb(
            pdb_dir / f"{sample}_2_{variant2}_model.pdb",
            variant2,
            missing_ca_position=2 if missing_variant2_ca and index == 0 else None,
        )
        write_pdb(
            pdb_dir / f"{sample}_3_{variant3}_model.pdb",
            variant3,
            omit_cn_positions=(
                {1} if missing_variant3_cn and index == 0 else set()
            ),
        )
        write_pdb(pdb_dir / f"{sample}_4_{variant4}_model.pdb", variant4_content)


class MonomerV10WindowsPdbReuseTests(unittest.TestCase):
    def test_primary_reuse_passes_while_variant3_is_authorized_only_per_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            manifest = tmp_path / "monomer_v10_design_manifest_151.csv"
            rows = write_manifest(manifest)
            pdb_dir = tmp_path / "pdb_monomer_hf4"
            # M_002 uses a different lowercase position in the old variant-3 file.
            write_inventory(pdb_dir, rows, second_variant3="RSt")
            before = {path.name: sha256(path) for path in pdb_dir.glob("*.pdb")}
            out_dir = tmp_path / "audit"

            report = run_audit(
                manifest,
                pdb_dir,
                out_dir,
                expected_samples=2,
                expected_total_pdbs=6,
            )

            self.assertEqual(report["quality_gate"], "PASS")
            self.assertIs(report["naturalized_reuse_authorized"], True)
            self.assertEqual(
                report["reuse_authorization"]["variant2_reference_naturalized"]["authorized_count"],
                2,
            )
            self.assertEqual(
                report["reuse_authorization"]["variant4_v10_e2e_naturalized"]["authorized_count"],
                2,
            )
            self.assertEqual(
                report["reuse_authorization"]["variant3_explicit_methyl"]["authorized_count"],
                1,
            )
            after = {path.name: sha256(path) for path in pdb_dir.glob("*.pdb")}
            self.assertEqual(after, before)

            with (out_dir / "monomer_v10_pdb_reuse_audit.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                audit_rows = {row["sample_name"]: row for row in csv.DictReader(handle)}
            self.assertEqual(audit_rows["M_001"]["variant3_status"], "AUTHORIZED")
            self.assertEqual(
                audit_rows["M_002"]["variant3_status"],
                "NOT_AUTHORIZED_FILENAME_SEQUENCE_MISMATCH",
            )
            self.assertEqual(audit_rows["M_002"]["row_quality_gate"], "PASS")
            parsed_variant2 = structure.parse_pdb(
                pdb_dir / "M_001_2_ACD_model.pdb", "ACD"
            )
            parsed_variant3 = structure.parse_pdb(
                pdb_dir / "M_001_3_wYA_model.pdb", "wYA"
            )
            self.assertEqual(parsed_variant2["pdb_residue_sequence"], "ACD")
            self.assertEqual(parsed_variant3["pdb_residue_sequence"], "wYA")

            old_expected_samples = structure.EXPECTED_SAMPLES
            structure.EXPECTED_SAMPLES = 2
            try:
                authorization, loaded_report = structure.load_v10_reuse_authorization(
                    out_dir / "monomer_v10_pdb_reuse_audit.json",
                    manifest,
                    pdb_dir,
                )
            finally:
                structure.EXPECTED_SAMPLES = old_expected_samples
            self.assertEqual(loaded_report["quality_gate"], "PASS")
            self.assertTrue(
                structure.audit_flag(authorization["M_001"]["variant3_reuse_authorized"])
            )
            self.assertFalse(
                structure.audit_flag(authorization["M_002"]["variant3_reuse_authorized"])
            )

    def test_primary_reuse_fails_on_filename_or_complete_ca_contract_violation(self):
        cases = [
            (True, False, "variant4_v10_naturalized_authorized_for_all_samples"),
            (False, True, "variant2_reference_naturalized_authorized_for_all_samples"),
        ]
        for bad_variant4, missing_variant2_ca, failed_check in cases:
            with self.subTest(
                bad_variant4=bad_variant4,
                missing_variant2_ca=missing_variant2_ca,
            ):
                with tempfile.TemporaryDirectory() as temporary:
                    tmp_path = Path(temporary)
                    manifest = tmp_path / "manifest.csv"
                    rows = write_manifest(manifest)
                    pdb_dir = tmp_path / "pdb_monomer_hf4"
                    write_inventory(
                        pdb_dir,
                        rows,
                        second_variant3="RsT",
                        bad_variant4=bad_variant4,
                        missing_variant2_ca=missing_variant2_ca,
                    )
                    report = run_audit(
                        manifest,
                        pdb_dir,
                        tmp_path / "audit",
                        expected_samples=2,
                        expected_total_pdbs=6,
                    )
                    self.assertEqual(report["quality_gate"], "FAIL")
                    self.assertIs(report["naturalized_reuse_authorized"], False)
                    self.assertIs(report["quality_checks"][failed_check], False)

    def test_correct_filename_but_wrong_pdb_residue_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            manifest = tmp_path / "manifest.csv"
            rows = write_manifest(manifest)
            pdb_dir = tmp_path / "pdb_monomer_hf4"
            write_inventory(
                pdb_dir,
                rows,
                second_variant3="RsT",
                bad_variant4_content=True,
            )
            out_dir = tmp_path / "audit"
            report = run_audit(
                manifest,
                pdb_dir,
                out_dir,
                expected_samples=2,
                expected_total_pdbs=6,
            )
            # The read-only audit independently decodes the PDB residue names.
            self.assertEqual(report["quality_gate"], "FAIL")
            self.assertFalse(report["naturalized_reuse_authorized"])
            self.assertFalse(
                report["quality_checks"][
                    "variant4_v10_naturalized_authorized_for_all_samples"
                ]
            )

            # The structure stage must repeat that check rather than trusting the
            # audit CSV or the correct-looking filename.
            with self.assertRaisesRegex(
                ValueError,
                r"PDB residue sequence mismatch: observed=WFA, expected=WYA",
            ):
                structure.parse_pdb(
                    pdb_dir / "M_001_4_WYA_model.pdb",
                    "WYA",
                )

    def test_variant3_marked_filename_without_cn_is_not_authorized(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            manifest = tmp_path / "manifest.csv"
            rows = write_manifest(manifest)
            pdb_dir = tmp_path / "pdb_monomer_hf4"
            write_inventory(
                pdb_dir,
                rows,
                second_variant3="RsT",
                missing_variant3_cn=True,
            )
            out_dir = tmp_path / "audit"
            report = run_audit(
                manifest,
                pdb_dir,
                out_dir,
                expected_samples=2,
                expected_total_pdbs=6,
            )

            self.assertEqual(report["quality_gate"], "PASS")
            self.assertTrue(report["naturalized_reuse_authorized"])
            self.assertEqual(
                report["reuse_authorization"]["variant2_reference_naturalized"][
                    "authorized_count"
                ],
                2,
            )
            self.assertEqual(
                report["reuse_authorization"]["variant4_v10_e2e_naturalized"][
                    "authorized_count"
                ],
                2,
            )
            self.assertEqual(
                report["reuse_authorization"]["variant3_explicit_methyl"][
                    "authorized_count"
                ],
                1,
            )
            with (out_dir / "monomer_v10_pdb_reuse_audit.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                audit_rows = {row["sample_name"]: row for row in csv.DictReader(handle)}
            missing_cn = audit_rows["M_001"]
            self.assertEqual(missing_cn["variant3_filename_sequence_match"], "1")
            self.assertEqual(missing_cn["variant3_pdb_decoded_marked_sequence"], "?YA")
            self.assertEqual(missing_cn["variant3_reuse_authorized"], "0")
            self.assertEqual(
                missing_cn["variant3_status"],
                "NOT_AUTHORIZED_PDB_CONTENT_SEQUENCE_MISMATCH",
            )
            self.assertIn(
                "missing the expected CN atom",
                missing_cn["variant3_residue_decode_errors_json"],
            )
            self.assertEqual(
                missing_cn["primary_naturalized_pair_reuse_authorized"], "1"
            )
            self.assertEqual(missing_cn["row_quality_gate"], "PASS")

    def test_structure_stage_rejects_inventory_changed_after_pass_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            manifest = tmp_path / "manifest.csv"
            rows = write_manifest(manifest)
            pdb_dir = tmp_path / "pdb_monomer_hf4"
            write_inventory(pdb_dir, rows, second_variant3="RSt")
            out_dir = tmp_path / "audit"
            report = run_audit(
                manifest,
                pdb_dir,
                out_dir,
                expected_samples=2,
                expected_total_pdbs=6,
            )
            self.assertEqual(report["quality_gate"], "PASS")
            changed = next(pdb_dir.glob("*.pdb"))
            changed.write_text(
                changed.read_text(encoding="utf-8") + "REMARK changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "inventory changed"):
                structure.load_v10_reuse_authorization(
                    out_dir / "monomer_v10_pdb_reuse_audit.json",
                    manifest,
                    pdb_dir,
                )

    def test_upstream_manifest_contract_rejects_bare_or_tampered_design_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            manifest = tmp_path / "manifest.csv"
            rows = write_manifest(manifest)
            pdb_dir = tmp_path / "pdb_monomer_hf4"
            write_inventory(pdb_dir, rows, second_variant3="RsT")
            upstream = manifest.with_name("monomer_v10_manifest.json")

            with self.assertRaisesRegex(
                FileNotFoundError, "Missing AutoDL monomer manifest"
            ):
                audit.audit_reuse(
                    manifest,
                    pdb_dir,
                    tmp_path / "bare_audit",
                    autodl_monomer_manifest=upstream,
                    autodl_monomer_manifest_sha256="0" * 64,
                    expected_samples=2,
                    expected_total_pdbs=6,
                )

            upstream = write_autodl_monomer_manifest(manifest)
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace("WYA,wYA", "WFA,wFA"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "does not match the AutoDL monomer manifest"
            ):
                audit.audit_reuse(
                    manifest,
                    pdb_dir,
                    tmp_path / "tampered_audit",
                    autodl_monomer_manifest=upstream,
                    autodl_monomer_manifest_sha256=sha256(upstream),
                    expected_samples=2,
                    expected_total_pdbs=6,
                )

    def test_upstream_manifest_requires_protocol_gate_and_all_checks_pass(self):
        cases = [
            (
                {"protocol": "obsolete_protocol"},
                "protocol mismatch",
            ),
            (
                {"quality_gate": "FAIL"},
                "quality_gate is not PASS",
            ),
            (
                {"quality_checks": {"fixture_complete": True, "frozen": False}},
                "non-PASS quality_checks: frozen",
            ),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as temporary:
                    tmp_path = Path(temporary)
                    manifest = tmp_path / "manifest.csv"
                    rows = write_manifest(manifest)
                    pdb_dir = tmp_path / "pdb_monomer_hf4"
                    write_inventory(pdb_dir, rows, second_variant3="RsT")
                    upstream = write_autodl_monomer_manifest(
                        manifest,
                        **overrides,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        audit.audit_reuse(
                            manifest,
                            pdb_dir,
                            tmp_path / "audit",
                            autodl_monomer_manifest=upstream,
                            autodl_monomer_manifest_sha256=sha256(upstream),
                            expected_samples=2,
                            expected_total_pdbs=6,
                        )

    def test_structure_rejects_tampered_csv_even_with_fake_pass_audit_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            manifest = tmp_path / "manifest.csv"
            rows = write_manifest(manifest)
            pdb_dir = tmp_path / "pdb_monomer_hf4"
            write_inventory(pdb_dir, rows, second_variant3="RsT")
            out_dir = tmp_path / "audit"
            report = run_audit(manifest, pdb_dir, out_dir)
            self.assertEqual(report["quality_gate"], "PASS")

            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace("WYA,wYA", "WFA,wFA"),
                encoding="utf-8",
            )
            audit_json = out_dir / "monomer_v10_pdb_reuse_audit.json"
            forged_report = json.loads(audit_json.read_text(encoding="utf-8"))
            # Simulate a hand-edited PASS audit that updates only the naked CSV
            # hash. The independently retained AutoDL manifest still pins the
            # original bytes and must make the structure stage stop.
            forged_report["inputs"]["design_manifest"]["sha256"] = sha256(manifest)
            audit_json.write_text(
                json.dumps(forged_report, indent=2) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "does not match the current AutoDL monomer manifest"
            ):
                structure.load_v10_reuse_authorization(
                    audit_json,
                    manifest,
                    pdb_dir,
                )

    def test_windows_entry_keeps_autodl_and_windows_paths_explicit_and_gated(self):
        entry = WINDOWS_ENTRY_PATH.read_text(encoding="utf-8")
        controller = CONTROLLER_PATH.read_text(encoding="utf-8")
        structure_source = STRUCTURE_PATH.read_text(encoding="utf-8")

        self.assertIn('"E:\\ProteinMPNN_work\\proteinmpnn-clean-v28"', entry)
        self.assertIn("AutoDL does not contain these old PDB files", entry)
        self.assertIn("monomer_v10_design_manifest_151.csv", entry)
        self.assertIn("monomer_v10_manifest.json", entry)
        self.assertIn("$AutoDlManifestPayload.quality_checks", entry)
        self.assertIn("$RecordedDesignSha256", entry)
        self.assertIn("--autodl-monomer-manifest", entry)
        self.assertIn("--autodl-monomer-manifest-sha256", entry)
        self.assertIn("22_audit_monomer_v10_pdb_reuse.py", entry)
        self.assertLess(entry.index("READ-ONLY PDB REUSE AUDIT"), entry.index("& $Controller"))
        self.assertIn('$AuditReport.quality_gate -ne "PASS"', entry)
        self.assertIn("-MonomerOnly", entry)
        self.assertIn("-StartStep 10", entry)
        self.assertIn("-PdbReuseAuditJson $AuditJson", entry)

        self.assertIn("[string]$MonomerDesignManifest", controller)
        self.assertIn("[string]$PdbReuseAuditJson", controller)
        self.assertIn("[switch]$MonomerOnly", controller)
        self.assertIn('"--pdb_reuse_audit_json", $PdbReuseAuditJson', controller)
        self.assertIn("load_v10_reuse_authorization", structure_source)
        self.assertIn("not_reused_v10_marked_sequence_not_authorized", structure_source)
        self.assertIn(
            "V10 monomer manifest requires --pdb_reuse_audit_json",
            structure_source,
        )


if __name__ == "__main__":
    unittest.main()
