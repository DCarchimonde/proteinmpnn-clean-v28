#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild the frozen train/test labels from their source PDB records.

The historical tokenizer merged ``ATOM`` and ``HETATM`` records and then used
only ``residue_name``.  Because ``SER`` appeared in both maps, ordinary
``ATOM-SER`` was emitted as lowercase ``s``.  This module reconstructs the
labels from record type, component name, and the N-methyl carbon atom ``CN``.
Coordinates and every non-sequence field are preserved byte-for-value at the
JSON object level.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from nmethyl.utils.nmethyl_config import (
    EXTENDED_AA_ALPHABET,
    NATURAL_RESIDUE_MAP,
    NMETHYL_RESIDUE_MAP,
    residue_token_from_pdb,
)


SOURCE_COMMIT = "28dff152d83623dfb322480413b7dc889f8537a4"
EXPECTED_INPUTS = {
    "train": {
        "rows": 600,
        "semantic_sha256": "0d6cd9ff4fb9bb385521c780967e01114d5fbb9caa66d550988c7df87da2d1da",
        "s_to_S": 242,
        "natural_S": 242,
        "methyl_s": 50,
        "natural_P": 307,
        "methyl_p": 0,
    },
    "test": {
        "rows": 151,
        "semantic_sha256": "913e2f2081486e533eb86f49178720ae5af3814ac8044e4c395246e239c69d82",
        "s_to_S": 62,
        "natural_S": 62,
        "methyl_s": 12,
        "natural_P": 83,
        "methyl_p": 0,
    },
}

LEGACY_NMETHYL_RESIDUE_MAP = dict(NMETHYL_RESIDUE_MAP)
LEGACY_NMETHYL_RESIDUE_MAP["SER"] = "s"
LEGACY_ALL_RESIDUE_MAP = {
    **NATURAL_RESIDUE_MAP,
    **LEGACY_NMETHYL_RESIDUE_MAP,
}


@dataclass(frozen=True)
class PDBResidue:
    chain_id: str
    residue_number: int
    insertion_code: str
    record_name: str
    residue_name: str
    atom_names: Tuple[str, ...]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def semantic_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_pdb_residues(path: Path) -> List[PDBResidue]:
    """Parse residues in the same chain/residue order as the old preprocessor."""

    grouped: MutableMapping[Tuple[str, int, str], Dict[str, set[str]]] = OrderedDict()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record_name = line[:6].strip().upper()
            if record_name not in {"ATOM", "HETATM"}:
                continue
            altloc = line[16:17]
            if altloc not in {" ", "A"}:
                continue
            residue_name = line[17:20].strip().upper()
            if residue_name not in NATURAL_RESIDUE_MAP and residue_name not in LEGACY_NMETHYL_RESIDUE_MAP:
                continue
            residue_number_text = line[22:26].strip()
            if not residue_number_text:
                raise ValueError(f"Missing residue number in {path}: {line.rstrip()}")
            key = (
                line[21:22].strip(),
                int(residue_number_text),
                line[26:27].strip(),
            )
            item = grouped.setdefault(
                key,
                {"record_names": set(), "residue_names": set(), "atom_names": set()},
            )
            item["record_names"].add(record_name)
            item["residue_names"].add(residue_name)
            item["atom_names"].add(line[12:16].strip().upper())

    residues: List[PDBResidue] = []
    for (chain_id, residue_number, insertion_code), item in sorted(
        grouped.items(), key=lambda pair: (pair[0][0], pair[0][1], pair[0][2])
    ):
        if len(item["record_names"]) != 1 or len(item["residue_names"]) != 1:
            raise ValueError(
                f"Mixed record/residue identity in {path} at "
                f"{chain_id}:{residue_number}{insertion_code}"
            )
        residues.append(
            PDBResidue(
                chain_id=chain_id,
                residue_number=residue_number,
                insertion_code=insertion_code,
                record_name=next(iter(item["record_names"])),
                residue_name=next(iter(item["residue_names"])),
                atom_names=tuple(sorted(item["atom_names"])),
            )
        )
    return residues


def legacy_token(residue: PDBResidue) -> str:
    try:
        return LEGACY_ALL_RESIDUE_MAP[residue.residue_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported legacy residue {residue.residue_name}") from exc


def corrected_token(residue: PDBResidue) -> str:
    return residue_token_from_pdb(
        residue.record_name,
        residue.residue_name,
        residue.atom_names,
    )


def _sequence_chain_ids(row: Mapping[str, Any]) -> List[str]:
    return sorted(
        key.removeprefix("seq_chain_")
        for key in row
        if key.startswith("seq_chain_")
    )


def rebuild_split(
    split: str,
    input_jsonl: Path,
    raw_pdb_dir: Path,
    output_jsonl: Path,
    allow_unpinned_input: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if split not in EXPECTED_INPUTS:
        raise ValueError(f"Unknown split: {split}")
    expected = EXPECTED_INPUTS[split]
    rows = read_jsonl(input_jsonl)
    semantic_hash = semantic_jsonl_sha256(rows)
    if not allow_unpinned_input:
        if len(rows) != expected["rows"]:
            raise RuntimeError(
                f"{split} row count changed: expected {expected['rows']}, observed {len(rows)}"
            )
        if semantic_hash != expected["semantic_sha256"]:
            raise RuntimeError(
                f"{split} semantic SHA256 mismatch: expected "
                f"{expected['semantic_sha256']}, observed {semantic_hash}"
            )

    corrected_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    token_counts: Counter[str] = Counter()
    change_counts: Counter[Tuple[str, str, str, str, int]] = Counter()
    source_component_counts: Counter[Tuple[str, str, int]] = Counter()
    seen_names: set[str] = set()

    for row_index, source_row in enumerate(rows):
        name = str(source_row.get("name", "")).strip()
        if not name or name in seen_names:
            raise RuntimeError(f"Missing or duplicate {split} record name: {name!r}")
        seen_names.add(name)
        pdb_path = raw_pdb_dir / f"{name}.pdb"
        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)

        residues = parse_pdb_residues(pdb_path)
        by_chain: MutableMapping[str, List[PDBResidue]] = OrderedDict()
        for residue in residues:
            by_chain.setdefault(residue.chain_id, []).append(residue)

        row_chain_ids = _sequence_chain_ids(source_row)
        if row_chain_ids != sorted(by_chain):
            raise RuntimeError(
                f"Chain mismatch for {name}: JSON={row_chain_ids}, PDB={sorted(by_chain)}"
            )

        output_row = copy.deepcopy(source_row)
        corrected_full_parts: List[str] = []
        legacy_full_parts: List[str] = []
        for chain_id in row_chain_ids:
            original_sequence = str(source_row[f"seq_chain_{chain_id}"])
            chain_residues = by_chain[chain_id]
            observed_legacy = "".join(legacy_token(residue) for residue in chain_residues)
            if observed_legacy != original_sequence:
                raise RuntimeError(
                    f"Legacy reconstruction mismatch for {name} chain {chain_id}: "
                    f"JSON={original_sequence}, PDB={observed_legacy}"
                )

            corrected_sequence_chars: List[str] = []
            for position, (original, residue) in enumerate(
                zip(original_sequence, chain_residues), start=1
            ):
                corrected = corrected_token(residue)
                if corrected not in EXTENDED_AA_ALPHABET:
                    raise RuntimeError(f"Resolved unsupported token {corrected!r}")
                corrected_sequence_chars.append(corrected)
                token_counts[corrected] += 1
                has_cn = int("CN" in residue.atom_names)
                source_component_counts[
                    (residue.record_name, residue.residue_name, has_cn)
                ] += 1
                changed = int(corrected != original)
                if changed:
                    change_counts[
                        (original, corrected, residue.record_name, residue.residue_name, has_cn)
                    ] += 1
                audit_rows.append(
                    {
                        "split": split,
                        "row_index": row_index,
                        "sample_name": name,
                        "chain_id": chain_id,
                        "position_1based": position,
                        "pdb_residue_number": residue.residue_number,
                        "pdb_insertion_code": residue.insertion_code,
                        "record_name": residue.record_name,
                        "residue_name": residue.residue_name,
                        "has_cn_atom": has_cn,
                        "legacy_token": original,
                        "corrected_token": corrected,
                        "changed": changed,
                        "correction_reason": (
                            "ATOM_SER_RESTORED_TO_NATURAL"
                            if changed
                            else "SOURCE_PROVENANCE_CONFIRMED"
                        ),
                    }
                )
            corrected_sequence = "".join(corrected_sequence_chars)
            output_row[f"seq_chain_{chain_id}"] = corrected_sequence
            corrected_full_parts.append(corrected_sequence)
            legacy_full_parts.append(original_sequence)

        legacy_full = "".join(legacy_full_parts)
        if str(source_row.get("seq", "")) != legacy_full:
            raise RuntimeError(f"Full-sequence reconstruction mismatch for {name}")
        output_row["seq"] = "".join(corrected_full_parts)
        corrected_rows.append(output_row)

    changed_s_to_S = sum(
        count
        for (old, new, _record, _residue, _cn), count in change_counts.items()
        if old == "s" and new == "S"
    )
    unexpected_changes = {
        str(key): value
        for key, value in change_counts.items()
        if not (key[0] == "s" and key[1] == "S" and key[2] == "ATOM" and key[3] == "SER" and key[4] == 0)
    }
    if unexpected_changes:
        raise RuntimeError(f"Unexpected label changes: {unexpected_changes}")

    if not allow_unpinned_input:
        observed_expected = {
            "s_to_S": changed_s_to_S,
            "natural_S": token_counts["S"],
            "methyl_s": token_counts["s"],
            "natural_P": token_counts["P"],
            "methyl_p": token_counts["p"],
        }
        expected_subset = {key: expected[key] for key in observed_expected}
        if observed_expected != expected_subset:
            raise RuntimeError(
                f"{split} provenance counts changed: expected {expected_subset}, "
                f"observed {observed_expected}"
            )

    atomic_write_jsonl(output_jsonl, corrected_rows)
    output_rows = read_jsonl(output_jsonl)
    if output_rows != corrected_rows:
        raise RuntimeError(f"Round-trip verification failed: {output_jsonl}")

    summary = {
        "split": split,
        "quality_gate": "PASS",
        "input_path": str(input_jsonl.resolve()),
        "input_file_sha256": file_sha256(input_jsonl),
        "input_semantic_sha256": semantic_hash,
        "output_path": str(output_jsonl.resolve()),
        "output_file_sha256": file_sha256(output_jsonl),
        "output_semantic_sha256": semantic_jsonl_sha256(corrected_rows),
        "rows": len(corrected_rows),
        "positions": sum(token_counts.values()),
        "rows_changed": len({row["sample_name"] for row in audit_rows if row["changed"]}),
        "s_to_S": changed_s_to_S,
        "natural_S": token_counts["S"],
        "methyl_s": token_counts["s"],
        "natural_P": token_counts["P"],
        "methyl_p": token_counts["p"],
        "token_counts": dict(sorted(token_counts.items())),
        "source_component_counts": [
            {
                "record_name": key[0],
                "residue_name": key[1],
                "has_cn_atom": key[2],
                "count": count,
            }
            for key, count in sorted(source_component_counts.items())
        ],
        "allowed_label_change": "s -> S only for ATOM-SER without CN",
        "coordinates_modified": False,
        "non_sequence_fields_modified": False,
    }
    return summary, audit_rows


AUDIT_FIELDS = [
    "split",
    "row_index",
    "sample_name",
    "chain_id",
    "position_1based",
    "pdb_residue_number",
    "pdb_insertion_code",
    "record_name",
    "residue_name",
    "has_cn_atom",
    "legacy_token",
    "corrected_token",
    "changed",
    "correction_reason",
]
