#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only authorization audit for reusing the old monomer HighFold PDBs.

This is the Windows/local boundary for the V10 monomer handoff.  It never
renames, deletes, copies, or rewrites a PDB.  Naturalized variants 2 and 4 are
authorized only when the complete 151-sample filename/chain/CA contract is
verified.  Explicit-methyl variant 3 remains a per-sample authorization: a
file is reusable only when its case-sensitive marked sequence is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nmethyl.utils.nmethyl_config import residue_token_from_pdb

EXPECTED_SAMPLES = 151
EXPECTED_TOTAL_PDBS = 560
PROTOCOL = "monomer_v10_windows_read_only_pdb_reuse_audit_v3_upstream_pinned"
EXPECTED_AUTODL_MONOMER_PROTOCOL = (
    "corrected_monomer_cyclic_stability_and_base_freeze_audit_v10"
)
EXPECTED_AUTODL_MANIFEST_NAME = "monomer_v10_manifest.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
MARKED_SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]+$")
PDB_NAME_RE = re.compile(
    r"^(?P<sample>.+)_(?P<variant>[1-4])_(?P<sequence>[^_]+)_model\.pdb$"
)
DEFAULT_MANIFEST = (
    ROOT
    / "paper_clean_v28_outputs"
    / "rmsd_aware_v10_1700_monomer"
    / "monomer_final"
    / "monomer_v10_design_manifest_151.csv"
)
DEFAULT_AUTODL_MONOMER_MANIFEST = DEFAULT_MANIFEST.with_name(
    EXPECTED_AUTODL_MANIFEST_NAME
)
DEFAULT_PDB_DIR = (
    ROOT
    / "raw_external"
    / "pdb_permeability_v20260624"
    / "pdb_monomer"
    / "pdb_monomer_hf4"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "paper_clean_v28_outputs"
    / "rmsd_aware_v10_1700_monomer"
    / "windows_structure_recalculation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of old monomer HighFold PDB reuse against the "
            "downloaded V10 151-row design manifest."
        )
    )
    parser.add_argument("--design-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--autodl-monomer-manifest",
        default=str(DEFAULT_AUTODL_MONOMER_MANIFEST),
        help=(
            "Downloaded AutoDL monomer_v10_manifest.json beside the design CSV. "
            "Its protocol, gates, and design-manifest artifact hash are mandatory."
        ),
    )
    parser.add_argument(
        "--autodl-monomer-manifest-sha256",
        required=True,
        help=(
            "SHA-256 computed from the downloaded AutoDL monomer manifest by the "
            "Windows launcher; prevents an implicit path-only handoff."
        ),
    )
    parser.add_argument("--pdb-dir", default=str(DEFAULT_PDB_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"AutoDL monomer manifest must contain a JSON object: {path}")
    return payload


def require_sha256(value: object, label: str) -> str:
    normalized = str(value).strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{label} is not a lowercase/uppercase 64-hex SHA-256")
    return normalized


def validate_autodl_monomer_manifest(
    autodl_manifest: Path,
    expected_manifest_sha256: str,
    design_manifest: Path,
) -> Dict[str, Any]:
    """Validate the AutoDL-to-Windows monomer handoff without trusting source paths."""
    autodl_manifest = autodl_manifest.resolve()
    design_manifest = design_manifest.resolve()
    if autodl_manifest.name != EXPECTED_AUTODL_MANIFEST_NAME:
        raise ValueError(
            "AutoDL monomer manifest must be named "
            f"{EXPECTED_AUTODL_MANIFEST_NAME}: {autodl_manifest}"
        )
    if autodl_manifest.parent != design_manifest.parent:
        raise ValueError(
            "AutoDL monomer manifest and V10 design CSV must be in the same "
            "downloaded monomer_final directory"
        )
    if not autodl_manifest.is_file():
        raise FileNotFoundError(
            "Missing AutoDL monomer manifest beside the V10 design CSV: "
            f"{autodl_manifest}"
        )

    expected_sha256 = require_sha256(
        expected_manifest_sha256, "expected AutoDL monomer manifest SHA-256"
    )
    actual_sha256 = sha256_file(autodl_manifest)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "AutoDL monomer manifest SHA-256 differs from the Windows launcher "
            f"value: expected={expected_sha256}, observed={actual_sha256}"
        )

    payload = read_json_object(autodl_manifest)
    if payload.get("protocol") != EXPECTED_AUTODL_MONOMER_PROTOCOL:
        raise ValueError(
            "AutoDL monomer manifest protocol mismatch: "
            f"expected={EXPECTED_AUTODL_MONOMER_PROTOCOL}, "
            f"observed={payload.get('protocol')!r}"
        )
    if payload.get("quality_gate") != "PASS":
        raise ValueError("AutoDL monomer manifest quality_gate is not PASS")
    quality_checks = payload.get("quality_checks")
    if not isinstance(quality_checks, Mapping) or not quality_checks:
        raise ValueError("AutoDL monomer manifest quality_checks is missing or empty")
    failed_checks = sorted(
        str(name) for name, passed in quality_checks.items() if passed is not True
    )
    if failed_checks:
        raise ValueError(
            "AutoDL monomer manifest has non-PASS quality_checks: "
            + ", ".join(failed_checks)
        )

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("AutoDL monomer manifest artifacts object is missing")
    design_record = artifacts.get("design_manifest")
    if not isinstance(design_record, Mapping):
        raise ValueError("AutoDL monomer manifest artifacts.design_manifest is missing")
    recorded_design_sha256 = require_sha256(
        design_record.get("sha256"),
        "AutoDL monomer manifest artifacts.design_manifest SHA-256",
    )
    observed_design_sha256 = sha256_file(design_manifest)
    if observed_design_sha256 != recorded_design_sha256:
        raise ValueError(
            "Downloaded V10 design CSV does not match the AutoDL monomer manifest "
            "design-manifest artifact hash: "
            f"expected={recorded_design_sha256}, observed={observed_design_sha256}"
        )

    return {
        "path": str(autodl_manifest),
        "sha256": actual_sha256,
        "expected_sha256_from_windows_launcher": expected_sha256,
        "protocol": str(payload["protocol"]),
        "quality_gate": str(payload["quality_gate"]),
        "quality_check_count": len(quality_checks),
        "all_quality_checks_pass": True,
        "design_manifest_artifact": {
            "path_as_recorded_on_autodl": str(design_record.get("path", "")),
            "sha256": recorded_design_sha256,
        },
    }


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=union_fields(rows))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def pdb_chain_inventory(path: Path) -> List[Dict[str, Any]]:
    """Count residues and CA coverage per chain without modifying the PDB."""
    residues: Dict[str, Dict[Tuple[str, str], Dict[str, set[str]]]] = defaultdict(dict)
    ca_residues: Dict[str, set[Tuple[str, str]]] = defaultdict(set)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A", "1"}:
                continue
            chain = line[21].strip() or "_"
            key = (line[22:26].strip(), line[26].strip())
            record_name = line[:6].strip().upper()
            resname = line[17:20].strip().upper()
            atom_name = line[12:16].strip().upper()
            residue = residues[chain].setdefault(
                key,
                {
                    "record_names": set(),
                    "resnames": set(),
                    "atom_names": set(),
                },
            )
            residue["record_names"].add(record_name)
            residue["resnames"].add(resname)
            residue["atom_names"].add(atom_name)
            if atom_name == "CA":
                ca_residues[chain].add(key)
    result: List[Dict[str, Any]] = []
    for chain in sorted(residues):
        decoded: List[str] = []
        unknown: List[int] = []
        decode_errors: List[Dict[str, Any]] = []
        for position, residue in enumerate(residues[chain].values(), start=1):
            record_names = residue["record_names"]
            resnames = residue["resnames"]
            if len(record_names) != 1 or len(resnames) != 1:
                token = "?"
                error = (
                    "ambiguous record/residue identity: "
                    f"record_names={sorted(record_names)}, "
                    f"resnames={sorted(resnames)}"
                )
            else:
                try:
                    token = residue_token_from_pdb(
                        next(iter(record_names)),
                        next(iter(resnames)),
                        residue["atom_names"],
                    )
                    error = ""
                except ValueError as exc:
                    token = "?"
                    error = str(exc)
            decoded.append(token)
            if token == "?":
                unknown.append(position)
                decode_errors.append(
                    {
                        "position_1based": position,
                        "error": error,
                    }
                )
        marked = "".join(decoded)
        result.append(
            {
                "chain": chain,
                "residue_count": len(residues[chain]),
                "ca_count": len(ca_residues.get(chain, set())),
                "decoded_marked_sequence": marked,
                "decoded_natural_sequence": marked.upper(),
                "unknown_residue_positions_1based": unknown,
                "residue_decode_errors": decode_errors,
            }
        )
    return result


def chain_contract(path: Path, expected_length: int) -> Dict[str, Any]:
    chains = pdb_chain_inventory(path)
    exact = [
        row
        for row in chains
        if row["residue_count"] == expected_length
        and row["ca_count"] == expected_length
    ]
    if len(exact) == 1:
        selected = exact[0]
        status = "PASS"
    else:
        chain_a = next((row for row in chains if row["chain"] == "A"), None)
        selected = chain_a or (chains[0] if len(chains) == 1 else None)
        status = (
            "FAIL_NO_COMPLETE_EXPECTED_LENGTH_CHAIN"
            if not exact
            else "FAIL_AMBIGUOUS_MULTIPLE_COMPLETE_CHAINS"
        )
    return {
        "pdb_chain_count": len(chains),
        "pdb_chain_summary_json": json.dumps(
            chains, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "selected_chain": selected["chain"] if selected else "",
        "observed_residue_count": selected["residue_count"] if selected else "",
        "observed_ca_count": selected["ca_count"] if selected else "",
        "chain_length_match": int(
            bool(selected) and selected["residue_count"] == expected_length
        ),
        "complete_ca_coverage": int(
            bool(selected) and selected["ca_count"] == expected_length
        ),
        "single_complete_expected_length_chain": int(len(exact) == 1),
        "decoded_marked_sequence": (
            selected["decoded_marked_sequence"] if selected else ""
        ),
        "decoded_natural_sequence": (
            selected["decoded_natural_sequence"] if selected else ""
        ),
        "unknown_residue_positions_1based_json": json.dumps(
            selected["unknown_residue_positions_1based"] if selected else []
        ),
        "residue_decode_errors_json": json.dumps(
            selected["residue_decode_errors"] if selected else [],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "chain_contract_status": status,
    }


def scan_inventory(
    pdb_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], List[Dict[str, Any]]]]:
    inventory: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(pdb_dir.glob("*.pdb"), key=lambda item: item.name.lower()):
        match = PDB_NAME_RE.fullmatch(path.name)
        row: Dict[str, Any] = {
            "path": path,
            "pdb_file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "filename_parse_status": "FAIL",
            "sample_name": "",
            "variant": "",
            "sequence_from_filename": "",
        }
        if match:
            row.update(
                {
                    "filename_parse_status": "PASS",
                    "sample_name": match.group("sample"),
                    "variant": int(match.group("variant")),
                    "sequence_from_filename": match.group("sequence"),
                }
            )
            grouped[(row["sample_name"], row["variant"])].append(row)
        inventory.append(row)
    return inventory, grouped


def inventory_sha256(inventory: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(inventory, key=lambda item: str(item["pdb_file"]).lower()):
        record = (
            f"{row['pdb_file']}\0{row['size_bytes']}\0{row['sha256']}\n"
        ).encode("utf-8")
        digest.update(record)
    return digest.hexdigest()


def audit_variant(
    records: Sequence[Mapping[str, Any]],
    expected_sequence: str,
    expected_length: int,
    prefix: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        f"{prefix}_expected_sequence": expected_sequence,
        f"{prefix}_pdb_count": len(records),
        f"{prefix}_pdb_file": "",
        f"{prefix}_pdb_sha256": "",
        f"{prefix}_filename_sequence": "",
        f"{prefix}_filename_sequence_match": 0,
        f"{prefix}_chain_length_match": 0,
        f"{prefix}_complete_ca_coverage": 0,
        f"{prefix}_pdb_decoded_marked_sequence": "",
        f"{prefix}_pdb_sequence_match": 0,
        f"{prefix}_reuse_authorized": 0,
    }
    if len(records) != 1:
        result[f"{prefix}_status"] = (
            "NOT_AUTHORIZED_MISSING" if not records else "NOT_AUTHORIZED_DUPLICATE"
        )
        result[f"{prefix}_pdb_file"] = ";".join(
            str(row["pdb_file"]) for row in records
        )
        return result

    record = records[0]
    observed_sequence = str(record["sequence_from_filename"])
    sequence_match = observed_sequence == expected_sequence
    contract = chain_contract(Path(record["path"]), expected_length)
    result.update(
        {
            f"{prefix}_pdb_file": record["pdb_file"],
            f"{prefix}_pdb_sha256": record["sha256"],
            f"{prefix}_filename_sequence": observed_sequence,
            f"{prefix}_filename_sequence_match": int(sequence_match),
            **{f"{prefix}_{key}": value for key, value in contract.items()},
        }
    )
    pdb_sequence = str(contract["decoded_marked_sequence"])
    pdb_sequence_match = pdb_sequence == expected_sequence
    result[f"{prefix}_pdb_decoded_marked_sequence"] = pdb_sequence
    result[f"{prefix}_pdb_sequence_match"] = int(pdb_sequence_match)
    authorized = (
        sequence_match
        and pdb_sequence_match
        and contract["chain_contract_status"] == "PASS"
    )
    result[f"{prefix}_reuse_authorized"] = int(authorized)
    if authorized:
        result[f"{prefix}_status"] = "AUTHORIZED"
    elif not sequence_match:
        result[f"{prefix}_status"] = "NOT_AUTHORIZED_FILENAME_SEQUENCE_MISMATCH"
    elif not pdb_sequence_match:
        result[f"{prefix}_status"] = "NOT_AUTHORIZED_PDB_CONTENT_SEQUENCE_MISMATCH"
    else:
        result[f"{prefix}_status"] = "NOT_AUTHORIZED_PDB_CHAIN_OR_CA_MISMATCH"
    return result


def valid_natural_sequence(value: str) -> bool:
    return bool(CANONICAL_SEQUENCE_RE.fullmatch(value))


def valid_marked_sequence(value: str) -> bool:
    return bool(MARKED_SEQUENCE_RE.fullmatch(value))


def audit_reuse(
    design_manifest: Path,
    pdb_dir: Path,
    out_dir: Path,
    *,
    autodl_monomer_manifest: Path,
    autodl_monomer_manifest_sha256: str,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_total_pdbs: int = EXPECTED_TOTAL_PDBS,
) -> Dict[str, Any]:
    design_manifest = design_manifest.resolve()
    autodl_monomer_manifest = autodl_monomer_manifest.resolve()
    pdb_dir = pdb_dir.resolve()
    out_dir = out_dir.resolve()
    if not design_manifest.is_file():
        raise FileNotFoundError(
            "Missing V10 design manifest. Download the AutoDL output into the "
            f"Windows repository before running this audit: {design_manifest}"
        )
    upstream_manifest_record = validate_autodl_monomer_manifest(
        autodl_monomer_manifest,
        autodl_monomer_manifest_sha256,
        design_manifest,
    )
    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"Missing local pdb_monomer_hf4 directory: {pdb_dir}")

    manifest_rows = read_csv(design_manifest)
    required_columns = {
        "sample_name",
        "reference_natural_sequence",
        "e2e_natural_sequence_for_structure_prediction",
        "e2e_stable_methyl_design",
    }
    columns = set(manifest_rows[0]) if manifest_rows else set()
    missing_columns = sorted(required_columns - columns)
    sample_counts = Counter(str(row.get("sample_name", "")) for row in manifest_rows)
    inventory, grouped = scan_inventory(pdb_dir)

    audit_rows: List[Dict[str, Any]] = []
    for source in sorted(manifest_rows, key=lambda row: str(row.get("sample_name", ""))):
        sample = str(source.get("sample_name", ""))
        reference = str(source.get("reference_natural_sequence", ""))
        natural_design = str(
            source.get("e2e_natural_sequence_for_structure_prediction", "")
        )
        marked_design = str(source.get("e2e_stable_methyl_design", ""))
        sequence_contract = (
            valid_natural_sequence(reference)
            and valid_natural_sequence(natural_design)
            and valid_marked_sequence(marked_design)
            and marked_design.upper() == natural_design
            and len(reference) == len(natural_design) == len(marked_design)
        )
        expected_length = len(natural_design)
        row: Dict[str, Any] = {
            "sample_name": sample,
            "manifest_duplicate_count": sample_counts[sample],
            "reference_natural_sequence": reference,
            "e2e_natural_sequence": natural_design,
            "e2e_marked_sequence": marked_design,
            "sequence_length": expected_length,
            "manifest_sequence_contract_valid": int(sequence_contract),
        }
        row.update(
            audit_variant(grouped.get((sample, 2), []), reference, expected_length, "variant2")
        )
        row.update(
            audit_variant(
                grouped.get((sample, 4), []),
                natural_design,
                expected_length,
                "variant4",
            )
        )
        row.update(
            audit_variant(
                grouped.get((sample, 3), []),
                marked_design,
                expected_length,
                "variant3",
            )
        )
        primary_authorized = (
            sequence_contract
            and sample_counts[sample] == 1
            and bool(row["variant2_reuse_authorized"])
            and bool(row["variant4_reuse_authorized"])
        )
        row["primary_naturalized_pair_reuse_authorized"] = int(primary_authorized)
        row["row_quality_gate"] = "PASS" if primary_authorized else "FAIL"
        audit_rows.append(row)

    parsed_inventory = [
        row for row in inventory if row["filename_parse_status"] == "PASS"
    ]
    inventory_sample_names = {str(row["sample_name"]) for row in parsed_inventory}
    manifest_sample_names = set(sample_counts)
    variant_file_counts = Counter(int(row["variant"]) for row in parsed_inventory)
    variant2_authorized = sum(
        int(row["variant2_reuse_authorized"]) for row in audit_rows
    )
    variant4_authorized = sum(
        int(row["variant4_reuse_authorized"]) for row in audit_rows
    )
    variant3_authorized = sum(
        int(row["variant3_reuse_authorized"]) for row in audit_rows
    )
    primary_authorized = sum(
        int(row["primary_naturalized_pair_reuse_authorized"])
        for row in audit_rows
    )

    quality_checks = {
        "autodl_monomer_manifest_protocol_quality_and_hash_contract_passed": True,
        "autodl_monomer_manifest_and_design_csv_are_colocated": True,
        "design_csv_matches_autodl_manifest_artifact_sha256": True,
        "manifest_has_required_v10_columns": not missing_columns,
        "manifest_has_exact_expected_rows": len(manifest_rows) == expected_samples,
        "manifest_sample_names_are_nonempty_and_unique": (
            len(sample_counts) == expected_samples
            and "" not in sample_counts
            and set(sample_counts.values()) == {1}
        ),
        "all_manifest_sequence_contracts_are_valid": bool(audit_rows)
        and all(bool(row["manifest_sequence_contract_valid"]) for row in audit_rows),
        "pdb_directory_has_exact_expected_total": len(inventory)
        == expected_total_pdbs,
        "all_pdb_filenames_are_parseable": len(parsed_inventory) == len(inventory),
        "pdb_inventory_sample_set_matches_manifest": inventory_sample_names
        == manifest_sample_names,
        "variant2_has_exact_expected_file_count": variant_file_counts[2]
        == expected_samples,
        "variant4_has_exact_expected_file_count": variant_file_counts[4]
        == expected_samples,
        "variant2_reference_naturalized_authorized_for_all_samples": variant2_authorized
        == expected_samples,
        "variant4_v10_naturalized_authorized_for_all_samples": variant4_authorized
        == expected_samples,
        "primary_naturalized_pair_authorized_for_all_samples": primary_authorized
        == expected_samples,
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    naturalized_authorized = quality_gate == "PASS"

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "monomer_v10_pdb_reuse_audit.csv"
    json_path = out_dir / "monomer_v10_pdb_reuse_audit.json"
    atomic_write_csv(csv_path, audit_rows)
    report: Dict[str, Any] = {
        "protocol": PROTOCOL,
        "quality_gate": quality_gate,
        "release_status": (
            "AUTHORIZED_REUSE_VARIANT2_VARIANT4_FOR_151_MONOMERS"
            if naturalized_authorized
            else "BLOCKED_PDB_REUSE_AUDIT_FAILED"
        ),
        "naturalized_reuse_authorized": naturalized_authorized,
        "pdb_access_policy": "READ_ONLY_NO_DELETE_NO_RENAME_NO_REWRITE",
        "machine_boundary": {
            "v10_manifest_origin": (
                "AutoDL monomer_v10_manifest.json plus its hash-pinned design CSV "
                "downloaded into this Windows repository"
            ),
            "pdb_origin": "existing Windows-local pdb_monomer_hf4 directory",
            "audit_execution": "Windows local",
        },
        "expected_samples": expected_samples,
        "expected_total_pdbs": expected_total_pdbs,
        "manifest_row_count": len(manifest_rows),
        "missing_manifest_columns": missing_columns,
        "pdb_count": len(inventory),
        "variant_file_counts": {
            str(key): int(value) for key, value in sorted(variant_file_counts.items())
        },
        "reuse_authorization": {
            "variant2_reference_naturalized": {
                "policy": "all_samples_or_block",
                "authorized_count": variant2_authorized,
                "required_count": expected_samples,
                "authorized": variant2_authorized == expected_samples,
            },
            "variant4_v10_e2e_naturalized": {
                "policy": "all_samples_or_block",
                "authorized_count": variant4_authorized,
                "required_count": expected_samples,
                "authorized": variant4_authorized == expected_samples,
            },
            "variant3_explicit_methyl": {
                "policy": (
                    "per_sample_case_sensitive_marked_sequence_and_"
                    "pdb_chemistry_with_cn_match_only"
                ),
                "authorized_count": variant3_authorized,
                "manifest_sample_count": expected_samples,
                "global_authorization_is_never_implied": True,
            },
        },
        "quality_checks": quality_checks,
        "inputs": {
            "autodl_monomer_manifest": upstream_manifest_record,
            "design_manifest": {
                "path": str(design_manifest),
                "sha256": sha256_file(design_manifest),
            },
            "pdb_inventory": {
                "directory": str(pdb_dir),
                "pdb_count": len(inventory),
                "sha256": inventory_sha256(inventory),
            },
        },
        "artifacts": {
            "audit_csv": {
                "path": str(csv_path),
                "sha256": sha256_file(csv_path),
                "row_count": len(audit_rows),
            }
        },
    }
    atomic_write_json(json_path, report)
    return report


def main() -> int:
    args = parse_args()
    report = audit_reuse(
        Path(args.design_manifest),
        Path(args.pdb_dir),
        Path(args.out_dir),
        autodl_monomer_manifest=Path(args.autodl_monomer_manifest),
        autodl_monomer_manifest_sha256=args.autodl_monomer_manifest_sha256,
    )
    print("===== V10 WINDOWS MONOMER PDB REUSE AUDIT =====")
    print(f"Quality gate: {report['quality_gate']}")
    print(
        "Variant 2 authorized: "
        f"{report['reuse_authorization']['variant2_reference_naturalized']['authorized_count']}"
        f"/{report['expected_samples']}"
    )
    print(
        "Variant 4 authorized: "
        f"{report['reuse_authorization']['variant4_v10_e2e_naturalized']['authorized_count']}"
        f"/{report['expected_samples']}"
    )
    print(
        "Variant 3 per-sample authorized: "
        f"{report['reuse_authorization']['variant3_explicit_methyl']['authorized_count']}"
        f"/{report['expected_samples']}"
    )
    if report["quality_gate"] != "PASS":
        failed = [
            name for name, passed in report["quality_checks"].items() if not passed
        ]
        print("Blocked checks: " + ", ".join(failed))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
