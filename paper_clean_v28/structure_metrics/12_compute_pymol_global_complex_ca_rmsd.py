#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Batch the exact PyMOL whole-complex C-alpha ``align(..., cycles=0)`` metric.

This audit deliberately does *not* replace the receptor-frame peptide RMSD
audit.  It reproduces the whole-complex CA operation demonstrated manually in
PyMOL and writes peptide/receptor alignment coverage next to every RMSD.

The program runs two cohorts in one invocation:

1. the existing 85 selected structures from ``complex_rmsd_metrics.csv``;
2. every PDB under the five HighFold temperature directories.

The known 4KEL manual result is used only as an implementation checkpoint.  It
never changes, filters, or ranks a computed value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from pymol import cmd
except ImportError as exc:  # pragma: no cover - exercised in the user's PyMOL env
    raise SystemExit(
        "PyMOL's Python module is required. Run this script with PyMOL, for example:\n"
        "  pymol -cq paper_clean_v28/structure_metrics/"
        "12_compute_pymol_global_complex_ca_rmsd.py"
    ) from exc


TEMP_MAP = {
    "pdb_highfold4_t001": "0.01",
    "pdb_highfold4_t01": "0.1",
    "pdb_highfold4_t02": "0.2",
    "pdb_highfold4_t03": "0.3",
    "pdb_highfold4_t05": "0.5",
}

AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V", "MSE": "M", "NCY": "C", "GNC": "Q", "MMO": "R",
    "UNK": "X",
}

PYMOL_OBJECTS = ("batch_pred", "batch_native", "batch_global_ca_alignment")

CHECKPOINT = {
    "pdb_file": "4kel_13_rcrrrGNrQGQCGR_model.pdb",
    "temperature": "0.3",
    "global_complex_ca_rmsd": 1.8244132995605469,
    "n_global_aligned_ca_pairs": 228,
    "n_matched_receptor_ca_pairs": 223,
    "n_matched_peptide_ca_pairs": 5,
    "global_align_raw_score": 1202.0,
}


def norm_temp(value: object) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def naturalize(sequence: object) -> str:
    return str(sequence or "").strip().upper()


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return None if math.isnan(result) else result
    except (TypeError, ValueError):
        return None


def fmt(value: object, digits: int = 6) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = list(fieldnames or [])
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_repo_path(repo_root: Path, value: object) -> Path:
    """Resolve Windows or POSIX repository-relative paths on either OS."""
    raw = str(value or "").strip()
    if not raw:
        return Path("")
    direct = Path(raw)
    if direct.is_absolute() or direct.exists():
        return direct
    normalized = raw.replace("\\", "/")
    return repo_root.joinpath(*PurePosixPath(normalized).parts)


def relative_path_text(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("/", os.sep)
    except (OSError, ValueError):
        return str(path)


def path_key(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./").casefold()


def parse_temperature(path: Path) -> Tuple[str, str]:
    normalized = str(path).replace("\\", "/").lower()
    for folder, temperature in TEMP_MAP.items():
        if folder in normalized:
            return temperature, folder
    return "", ""


def parse_pdb_filename(path: Path) -> dict:
    match = re.match(r"^([A-Za-z0-9]+)_(\d+)_(.+)_model\.pdb$", path.name)
    if not match:
        return {
            "target_name": "",
            "file_index": "",
            "design_seq": "",
            "filename_parse_ok": 0,
        }
    return {
        "target_name": match.group(1).upper(),
        "file_index": match.group(2),
        "design_seq": match.group(3),
        "filename_parse_ok": 1,
    }


def parse_chain_list(value: object) -> List[str]:
    return [part for part in re.split(r"[;,|\s]+", str(value or "").strip()) if part]


def design_key(row: Mapping[str, object]) -> Tuple[str, str, str]:
    return (
        str(row.get("target_name", "")).upper(),
        norm_temp(row.get("temperature")),
        str(row.get("design_seq", "")),
    )


def load_native_records(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            target = str(record.get("name", "")).upper()
            if target:
                records[target] = record
    return records


def native_sequences(record: Mapping[str, object]) -> Dict[str, str]:
    return {
        key[len("seq_chain_") :]: str(value)
        for key, value in record.items()
        if key.startswith("seq_chain_")
    }


def native_record_to_pdbstr(record: Mapping[str, object]) -> str:
    """Build the same coordinate-only native object used in the manual check."""
    lines: List[str] = []
    serial = 1
    for key, sequence_value in record.items():
        if not key.startswith("seq_chain_"):
            continue
        chain = key[len("seq_chain_") :]
        sequence = str(sequence_value)
        for residue_number, aa in enumerate(sequence, start=1):
            resname = AA1_TO_3.get(aa.upper(), "UNK")
            for atom_name in ("N", "CA", "C", "O"):
                coordinates = record.get(f"{atom_name}_chain_{chain}")
                if coordinates is None:
                    nested = record.get(f"coords_chain_{chain}", {})
                    coordinates = nested.get(
                        f"{atom_name}_chain_{chain}",
                        nested.get(atom_name, []),
                    )
                if residue_number - 1 >= len(coordinates or []):
                    continue
                xyz = coordinates[residue_number - 1]
                if xyz is None or len(xyz) != 3:
                    continue
                try:
                    x, y, z = map(float, xyz)
                except (TypeError, ValueError):
                    continue
                element = atom_name[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} "
                    f"{chain[:1]:1s}{residue_number:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {element:>2s}"
                )
                serial += 1
            # Kept deliberately identical to the native object used for the
            # confirmed 4KEL PyMOL operation.
            lines.append("TER")
    return "\n".join(lines) + "\nEND\n"


def parse_predicted_ca_metadata(path: Path) -> dict:
    """Read CA residue order, sequences, and mean CA B-factor from a PDB."""
    residues: Dict[str, List[dict]] = defaultdict(list)
    seen = set()
    b_factors: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            resname = line[17:20].strip().upper()
            residues[chain].append(
                {
                    "resseq": resseq,
                    "icode": icode,
                    "resname": resname,
                    "aa": AA3_TO_1.get(resname, resname if len(resname) == 1 else "X"),
                }
            )
            try:
                b_factors.append(float(line[60:66]))
            except ValueError:
                pass
    return {
        "chains": dict(residues),
        "chain_sequences": {
            chain: "".join(item["aa"] for item in items)
            for chain, items in residues.items()
        },
        "pdb_ca_bfactor_mean": mean(b_factors) if b_factors else None,
    }


def choose_predicted_peptide_chain(
    chain_sequences: Mapping[str, str],
    design_seq: str,
    design_length: int,
) -> str:
    expected = naturalize(design_seq)
    candidates = []
    for chain, sequence in chain_sequences.items():
        if design_length and len(sequence) != design_length:
            continue
        overlap = min(len(sequence), len(expected))
        matches = sum(sequence[index] == expected[index] for index in range(overlap))
        identity = matches / overlap if overlap else 0.0
        candidates.append((identity, matches, -abs(len(sequence) - design_length), chain))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    return candidates[0][3]


def choose_native_peptide_chains(
    record: Mapping[str, object],
    selected_chains: Sequence[str],
    native_seq: str,
    design_length: int,
) -> List[str]:
    sequences = native_sequences(record)
    anchors = [
        chain
        for chain in selected_chains
        if chain in sequences and (not design_length or len(sequences[chain]) == design_length)
    ]
    anchor_sequence = sequences[anchors[0]] if anchors else ""
    if not anchor_sequence:
        expected = naturalize(native_seq)
        exact = [
            chain
            for chain, sequence in sequences.items()
            if sequence.upper() == expected and (not design_length or len(sequence) == design_length)
        ]
        if exact:
            anchor_sequence = sequences[exact[0]]
    if anchor_sequence:
        return sorted(
            chain for chain, sequence in sequences.items() if sequence == anchor_sequence
        )
    return sorted(
        chain
        for chain, sequence in sequences.items()
        if design_length and len(sequence) == design_length
    )


def count_selection(selection: str) -> int:
    return int(cmd.count_atoms(selection))


def cleanup_pymol_objects() -> None:
    for name in PYMOL_OBJECTS:
        try:
            cmd.delete(name)
        except Exception:
            pass


def evaluate_pdb(
    pdb_path: Path,
    native_record: Mapping[str, object],
    metadata: Mapping[str, object],
    threshold: float,
    repo_root: Path,
) -> dict:
    parsed = parse_pdb_filename(pdb_path)
    temperature, temperature_folder = parse_temperature(pdb_path)
    target = str(metadata.get("target_name") or parsed["target_name"]).upper()
    design_seq = str(metadata.get("design_seq") or parsed["design_seq"])
    design_length_value = (
        metadata.get("design_length")
        or metadata.get("design_len")
        or len(design_seq)
    )
    try:
        design_length = int(float(design_length_value))
    except (TypeError, ValueError):
        design_length = len(design_seq)

    base = {
        "target_name": target,
        "temperature": norm_temp(metadata.get("temperature") or temperature),
        "temperature_folder": temperature_folder,
        "design_seq": design_seq,
        "design_natural_seq": naturalize(design_seq),
        "design_length": design_length,
        "native_seq": metadata.get("native_seq", ""),
        "pdb_file": pdb_path.name,
        "pdb_path": relative_path_text(pdb_path, repo_root),
        "pymol_align_selection_mobile": "batch_pred and name CA",
        "pymol_align_selection_target": "batch_native and name CA",
        "pymol_align_cycles": 0,
        "pymol_align_cutoff": 2.0,
        "pymol_align_gap": -10.0,
        "pymol_align_extend": -0.5,
        "pymol_align_max_gap": 50,
        "pymol_align_matrix": "BLOSUM62",
        "pymol_align_max_skip": 0,
        "global_complex_ca_rmsd_threshold_angstrom": threshold,
        "global_complex_ca_rmsd_status": "",
    }

    try:
        pred_meta = parse_predicted_ca_metadata(pdb_path)
        predicted_peptide_chain = str(
            metadata.get("predicted_peptide_chain", "")
        ).strip()
        if not predicted_peptide_chain:
            predicted_peptide_chain = choose_predicted_peptide_chain(
                pred_meta["chain_sequences"], design_seq, design_length
            )

        equivalent_native = parse_chain_list(
            metadata.get("equivalent_native_peptide_chains", "")
        )
        native_peptide_chain = str(
            metadata.get("native_peptide_chain_used")
            or metadata.get("native_peptide_chain")
            or ""
        ).strip()
        selected_native = parse_chain_list(metadata.get("selected_chains", ""))
        if native_peptide_chain:
            selected_native = [native_peptide_chain] + selected_native
        native_peptide_chains = equivalent_native or choose_native_peptide_chains(
            native_record,
            selected_native,
            str(metadata.get("native_seq", "")),
            design_length,
        )
        if native_peptide_chain and native_peptide_chain not in native_peptide_chains:
            native_peptide_chains.insert(0, native_peptide_chain)
        native_peptide_chains = list(dict.fromkeys(native_peptide_chains))

        base["predicted_peptide_chain"] = predicted_peptide_chain
        base["native_peptide_chains_considered"] = ";".join(native_peptide_chains)
        base["pdb_ca_bfactor_mean"] = fmt(pred_meta["pdb_ca_bfactor_mean"])
        base["predicted_chain_ca_counts"] = ";".join(
            f"{chain}:{len(items)}"
            for chain, items in sorted(pred_meta["chains"].items())
        )

        if not predicted_peptide_chain:
            raise ValueError("predicted peptide chain could not be inferred")
        if not native_peptide_chains:
            raise ValueError("native peptide chain could not be inferred")

        cleanup_pymol_objects()
        cmd.read_pdbstr(native_record_to_pdbstr(native_record), "batch_native")
        cmd.load(str(pdb_path), "batch_pred")
        cmd.sort("batch_native")
        cmd.sort("batch_pred")

        result = cmd.align(
            "batch_pred and name CA",
            "batch_native and name CA",
            cutoff=2.0,
            cycles=0,
            gap=-10.0,
            extend=-0.5,
            max_gap=50,
            object="batch_global_ca_alignment",
            matrix="BLOSUM62",
            mobile_state=0,
            target_state=0,
            quiet=1,
            max_skip=0,
            transform=1,
            reset=0,
        )
        if len(result) != 7:
            raise RuntimeError(f"Unexpected PyMOL align result: {result!r}")

        (
            rms_after,
            n_after,
            n_cycles,
            rms_before,
            n_before,
            raw_score,
            n_residues,
        ) = result

        atom_info: Dict[Tuple[str, int], dict] = {}
        cmd.iterate(
            "(batch_pred or batch_native) and name CA",
            "atom_info[(model,index)] = {'chain': chain, 'resi': resi, 'resn': resn}",
            space={"atom_info": atom_info},
        )

        chain_pairs: Counter = Counter()
        receptor_pairs = 0
        peptide_pairs = 0
        mixed_pairs = 0
        native_peptide_pair_counts: Counter = Counter()
        raw_columns = cmd.get_raw_alignment("batch_global_ca_alignment")
        for column in raw_columns:
            pred_atoms = [item for item in column if item[0] == "batch_pred"]
            native_atoms = [item for item in column if item[0] == "batch_native"]
            if len(pred_atoms) != 1 or len(native_atoms) != 1:
                mixed_pairs += 1
                continue
            pred_chain = atom_info.get(tuple(pred_atoms[0]), {}).get("chain", "?")
            native_chain = atom_info.get(tuple(native_atoms[0]), {}).get("chain", "?")
            chain_pairs[(pred_chain, native_chain)] += 1
            pred_is_peptide = pred_chain == predicted_peptide_chain
            native_is_peptide = native_chain in set(native_peptide_chains)
            if pred_is_peptide and native_is_peptide:
                peptide_pairs += 1
                native_peptide_pair_counts[native_chain] += 1
            elif not pred_is_peptide and not native_is_peptide:
                receptor_pairs += 1
            else:
                mixed_pairs += 1

        native_peptide_chain_used = ""
        if native_peptide_pair_counts:
            native_peptide_chain_used = sorted(
                native_peptide_pair_counts,
                key=lambda chain: (-native_peptide_pair_counts[chain], chain),
            )[0]
        elif native_peptide_chain:
            native_peptide_chain_used = native_peptide_chain
        else:
            native_peptide_chain_used = native_peptide_chains[0]

        pred_peptide_ca = count_selection(
            f"batch_pred and chain {predicted_peptide_chain} and name CA"
        )
        native_peptide_ca = count_selection(
            f"batch_native and chain {native_peptide_chain_used} and name CA"
        )
        pred_total_ca = count_selection("batch_pred and name CA")
        native_total_ca = count_selection("batch_native and name CA")
        pred_receptor_ca = pred_total_ca - pred_peptide_ca
        native_receptor_ca = native_total_ca - sum(
            count_selection(f"batch_native and chain {chain} and name CA")
            for chain in native_peptide_chains
        )

        base.update(
            {
                "native_peptide_chain_used_by_global_alignment": native_peptide_chain_used,
                "global_complex_ca_rmsd": fmt(rms_after),
                "n_global_aligned_ca_pairs": int(n_after),
                "global_align_cycles_performed": int(n_cycles),
                "global_complex_ca_rmsd_before_rejection": fmt(rms_before),
                "n_global_aligned_ca_pairs_before_rejection": int(n_before),
                "global_align_raw_score": fmt(raw_score),
                "global_align_residues": int(n_residues),
                "n_raw_alignment_columns": len(raw_columns),
                "n_predicted_total_ca": pred_total_ca,
                "n_native_total_ca": native_total_ca,
                "n_predicted_receptor_ca": pred_receptor_ca,
                "n_native_receptor_ca": native_receptor_ca,
                "n_predicted_peptide_ca": pred_peptide_ca,
                "n_native_peptide_ca": native_peptide_ca,
                "n_matched_receptor_ca_pairs": receptor_pairs,
                "n_matched_peptide_ca_pairs": peptide_pairs,
                "n_mixed_or_unclassified_ca_pairs": mixed_pairs,
                "matched_receptor_ca_coverage_vs_predicted": fmt(
                    receptor_pairs / pred_receptor_ca if pred_receptor_ca else None,
                    6,
                ),
                "matched_receptor_ca_coverage_vs_native": fmt(
                    receptor_pairs / native_receptor_ca if native_receptor_ca else None,
                    6,
                ),
                "matched_peptide_ca_coverage_vs_predicted": fmt(
                    peptide_pairs / pred_peptide_ca if pred_peptide_ca else None,
                    6,
                ),
                "matched_peptide_ca_coverage_vs_native": fmt(
                    peptide_pairs / native_peptide_ca if native_peptide_ca else None,
                    6,
                ),
                "full_peptide_ca_alignment_coverage": int(
                    peptide_pairs > 0
                    and peptide_pairs == pred_peptide_ca
                    and peptide_pairs == native_peptide_ca
                ),
                "aligned_chain_pair_counts": ";".join(
                    f"{pred}->{native}:{count}"
                    for (pred, native), count in sorted(chain_pairs.items())
                ),
                "passes_global_complex_ca_rmsd_lt_threshold": int(
                    float(rms_after) < threshold
                ),
                "global_complex_ca_rmsd_status": "ok",
            }
        )
    except Exception as exc:
        base["global_complex_ca_rmsd_status"] = "failed"
        base["global_complex_ca_rmsd_error"] = repr(exc)
    finally:
        cleanup_pymol_objects()
    return base


def unique_design_index(rows: Sequence[dict]) -> Dict[Tuple[str, str, str], List[dict]]:
    output: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        enriched = dict(row)
        enriched["_all_design_row_index"] = index
        output[design_key(row)].append(enriched)
    return dict(output)


def audit_index(rows: Sequence[dict]) -> dict:
    by_path = {}
    by_file_key = {}
    for row in rows:
        if row.get("pdb_path"):
            by_path[path_key(row["pdb_path"])] = row
        key = design_key(row) + (str(row.get("pdb_file", "")),)
        by_file_key[key] = row
    return {"by_path": by_path, "by_file_key": by_file_key}


def find_audit_row(
    index: Mapping[str, Mapping[object, dict]],
    pdb_path: Path,
    repo_root: Path,
    key: Tuple[str, str, str],
) -> dict:
    relative = relative_path_text(pdb_path, repo_root)
    row = index.get("by_path", {}).get(path_key(relative))
    if row:
        return row
    return index.get("by_file_key", {}).get(key + (pdb_path.name,), {})


def merge_chain_metadata(primary: Mapping[str, object], fallback: Mapping[str, object]) -> dict:
    output = dict(fallback)
    for key, value in primary.items():
        if value not in (None, ""):
            output[key] = value
    return output


def evaluate_best85(
    best_rows: Sequence[dict],
    native_records: Mapping[str, dict],
    threshold: float,
    repo_root: Path,
) -> List[dict]:
    output = []
    for index, source in enumerate(best_rows, start=1):
        target = str(source.get("target_name", "")).upper()
        pdb_path = resolve_repo_path(repo_root, source.get("pdb_path", ""))
        if not pdb_path.is_file():
            row = dict(source)
            row.update(
                {
                    "best85_row_index": index - 1,
                    "global_complex_ca_rmsd_status": "missing_pdb",
                    "global_complex_ca_rmsd_error": f"PDB not found: {pdb_path}",
                }
            )
        elif target not in native_records:
            row = dict(source)
            row.update(
                {
                    "best85_row_index": index - 1,
                    "global_complex_ca_rmsd_status": "missing_native_target",
                }
            )
        else:
            metrics = evaluate_pdb(
                pdb_path,
                native_records[target],
                source,
                threshold,
                repo_root,
            )
            row = dict(source)
            row.update(metrics)
            row["best85_row_index"] = index - 1
        output.append(row)
        if index % 10 == 0 or index == len(best_rows):
            print(f"[best85] processed: {index}/{len(best_rows)}", flush=True)
    return output


def evaluate_all_pdbs(
    pdb_files: Sequence[Path],
    designs: Mapping[Tuple[str, str, str], Sequence[dict]],
    strict_audit: Mapping[str, Mapping[object, dict]],
    native_records: Mapping[str, dict],
    threshold: float,
    repo_root: Path,
) -> List[dict]:
    output = []
    for index, pdb_path in enumerate(pdb_files, start=1):
        parsed = parse_pdb_filename(pdb_path)
        temperature, _folder = parse_temperature(pdb_path)
        key = (parsed["target_name"], temperature, parsed["design_seq"])
        design_rows = list(designs.get(key, []))
        metadata = dict(design_rows[0]) if design_rows else {
            "target_name": key[0],
            "temperature": key[1],
            "design_seq": key[2],
            "design_length": len(key[2]),
        }
        prior_audit = find_audit_row(strict_audit, pdb_path, repo_root, key)
        metadata = merge_chain_metadata(prior_audit, metadata)
        metadata["matched_all_design_rows"] = len(design_rows)

        if key[0] not in native_records:
            row = {
                **metadata,
                "pdb_file": pdb_path.name,
                "pdb_path": relative_path_text(pdb_path, repo_root),
                "global_complex_ca_rmsd_status": "missing_native_target",
            }
        else:
            row = evaluate_pdb(
                pdb_path,
                native_records[key[0]],
                metadata,
                threshold,
                repo_root,
            )
            row["matched_all_design_rows"] = len(design_rows)
        output.append(row)
        if index % 100 == 0 or index == len(pdb_files):
            print(f"[all PDBs] processed: {index}/{len(pdb_files)}", flush=True)
    return output


def representative_sort_confidence(row: Mapping[str, object]) -> tuple:
    confidence = safe_float(row.get("pdb_ca_bfactor_mean"))
    return (-(confidence if confidence is not None else -math.inf), str(row.get("pdb_file", "")))


def representative_sort_best_rmsd(row: Mapping[str, object]) -> tuple:
    value = safe_float(row.get("global_complex_ca_rmsd"))
    confidence = safe_float(row.get("pdb_ca_bfactor_mean"))
    return (
        value if value is not None else math.inf,
        -(confidence if confidence is not None else -math.inf),
        str(row.get("pdb_file", "")),
    )


def build_design_tables(
    designs: Mapping[Tuple[str, str, str], Sequence[dict]],
    pdb_rows: Sequence[dict],
    threshold: float,
) -> Tuple[List[dict], List[dict]]:
    by_key: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in pdb_rows:
        by_key[design_key(row)].append(row)

    confidence_rows = []
    best_rmsd_rows = []
    for key, raw_rows in sorted(designs.items(), key=lambda item: item[0]):
        candidates = by_key.get(key, [])
        valid = [
            row for row in candidates
            if row.get("global_complex_ca_rmsd_status") == "ok"
        ]
        base = dict(raw_rows[0])
        base.update(
            {
                "n_raw_design_rows_for_key": len(raw_rows),
                "raw_all_design_row_indices": ";".join(
                    str(row["_all_design_row_index"]) for row in raw_rows
                ),
                "n_pdb_for_key": len(candidates),
                "n_global_complex_ca_rmsd_ok_pdb_for_key": len(valid),
                "n_pdb_lt_threshold_for_key": sum(
                    str(row.get("passes_global_complex_ca_rmsd_lt_threshold", "")) == "1"
                    for row in valid
                ),
                "any_pdb_global_complex_ca_rmsd_lt_threshold": int(
                    any(
                        str(row.get("passes_global_complex_ca_rmsd_lt_threshold", "")) == "1"
                        for row in valid
                    )
                ),
                "global_complex_ca_rmsd_threshold_angstrom": threshold,
            }
        )

        if valid:
            confidence = sorted(valid, key=representative_sort_confidence)[0]
            exploratory = sorted(valid, key=representative_sort_best_rmsd)[0]
            confidence_row = {**base, **confidence}
            confidence_row["unique_design_representative_rule"] = (
                "highest_mean_ca_bfactor_among_global_rmsd_ok_pdbs"
            )
            exploratory_row = {**base, **exploratory}
            exploratory_row["unique_design_representative_rule"] = (
                "lowest_global_complex_ca_rmsd_exploratory_candidate_discovery"
            )
        elif candidates:
            fallback = sorted(candidates, key=lambda row: str(row.get("pdb_file", "")))[0]
            confidence_row = {**base, **fallback}
            confidence_row["unique_design_representative_rule"] = "first_failed_pdb_no_ok_result"
            exploratory_row = dict(confidence_row)
        else:
            confidence_row = dict(base)
            confidence_row.update(
                {
                    "global_complex_ca_rmsd_status": "missing_pdb",
                    "passes_global_complex_ca_rmsd_lt_threshold": "",
                    "unique_design_representative_rule": "missing_pdb",
                }
            )
            exploratory_row = dict(confidence_row)

        confidence_rows.append(confidence_row)
        best_rmsd_rows.append(exploratory_row)
    return confidence_rows, best_rmsd_rows


def summarize(rows: Sequence[dict], keys: Sequence[str]) -> List[dict]:
    groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output = []
    for values, items in sorted(groups.items()):
        ok = [row for row in items if row.get("global_complex_ca_rmsd_status") == "ok"]
        passed = [
            row for row in ok
            if str(row.get("passes_global_complex_ca_rmsd_lt_threshold", "")) == "1"
        ]
        rmsd_values = [safe_float(row.get("global_complex_ca_rmsd")) for row in ok]
        rmsd_values = [value for value in rmsd_values if value is not None]
        peptide_coverage = [
            safe_float(row.get("matched_peptide_ca_coverage_vs_native")) for row in ok
        ]
        peptide_coverage = [value for value in peptide_coverage if value is not None]
        result = {key: value for key, value in zip(keys, values)}
        result.update(
            {
                "n_rows": len(items),
                "n_global_complex_ca_rmsd_ok": len(ok),
                "n_failed_or_missing": len(items) - len(ok),
                "n_global_complex_ca_rmsd_lt_threshold": len(passed),
                "fraction_lt_threshold_among_ok": fmt(
                    len(passed) / len(ok) if ok else None, 8
                ),
                "mean_global_complex_ca_rmsd": fmt(
                    mean(rmsd_values) if rmsd_values else None
                ),
                "median_global_complex_ca_rmsd": fmt(
                    median(rmsd_values) if rmsd_values else None
                ),
                "min_global_complex_ca_rmsd": fmt(
                    min(rmsd_values) if rmsd_values else None
                ),
                "max_global_complex_ca_rmsd": fmt(
                    max(rmsd_values) if rmsd_values else None
                ),
                "median_matched_peptide_ca_coverage_vs_native": fmt(
                    median(peptide_coverage) if peptide_coverage else None
                ),
                "n_full_peptide_ca_alignment_coverage": sum(
                    str(row.get("full_peptide_ca_alignment_coverage", "")) == "1"
                    for row in ok
                ),
            }
        )
        output.append(result)
    return output


def cohort_report_lines(
    title: str,
    rows: Sequence[dict],
    threshold: float,
    expected_count: int,
) -> List[str]:
    ok = [row for row in rows if row.get("global_complex_ca_rmsd_status") == "ok"]
    passed = [
        row for row in ok
        if str(row.get("passes_global_complex_ca_rmsd_lt_threshold", "")) == "1"
    ]
    coverage = [
        safe_float(row.get("matched_peptide_ca_coverage_vs_native")) for row in ok
    ]
    coverage = [value for value in coverage if value is not None]
    status_counts = Counter(
        str(row.get("global_complex_ca_rmsd_status", "")) for row in rows
    )
    lines = [
        title,
        "",
        "Exact requested metric:",
        "  PyMOL align on complete predicted/native complexes, selection = name CA",
        "  cycles=0 (no structural outlier rejection)",
        "  default BLOSUM62 sequence alignment parameters fixed explicitly",
        f"  strict threshold = global complex CA RMSD < {threshold:.3f} Angstrom",
        "",
        f"rows: {len(rows)}",
        f"expected rows: {expected_count}",
        f"count gate: {'PASS' if not expected_count or len(rows) == expected_count else 'FAIL'}",
        f"RMSD OK: {len(ok)}",
        f"RMSD failed/missing: {len(rows) - len(ok)}",
        f"RMSD < {threshold:.3f}: {len(passed)}",
        (
            f"fraction < {threshold:.3f} among RMSD-OK: {len(passed) / len(ok):.8f}"
            if ok else f"fraction < {threshold:.3f} among RMSD-OK: NA"
        ),
        f"full peptide CA alignment coverage: "
        f"{sum(str(row.get('full_peptide_ca_alignment_coverage', '')) == '1' for row in ok)}"
        f"/{len(ok)}",
        (
            f"median peptide CA alignment coverage vs native: {median(coverage):.6f}"
            if coverage else "median peptide CA alignment coverage vs native: NA"
        ),
        "",
        "Status counts:",
    ]
    lines.extend(f"  {status}: {count}" for status, count in sorted(status_counts.items()))
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "  This is the requested whole-complex CA metric.",
            "  It can be dominated by receptor residues and can align only a subset of",
            "  peptide residues when designed and native peptide sequences differ.",
            "  Peptide/receptor pair counts and coverage must therefore travel with RMSD.",
            "  The receptor-frame peptide RMSD audit remains a separate complementary result.",
        ]
    )
    return lines


def write_checkpoint_report(
    path: Path,
    rows: Sequence[dict],
    tolerance: float,
) -> bool:
    matches = [
        row
        for row in rows
        if row.get("pdb_file") == CHECKPOINT["pdb_file"]
        and norm_temp(row.get("temperature")) == CHECKPOINT["temperature"]
    ]
    lines = ["===== PYMOL MANUAL 4KEL CHECKPOINT =====", ""]
    if len(matches) != 1:
        lines.extend(
            [
                "status: FAIL",
                f"expected exactly one row for {CHECKPOINT['pdb_file']}, found {len(matches)}",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return False

    row = matches[0]
    checks = []
    for field in (
        "global_complex_ca_rmsd",
        "global_align_raw_score",
    ):
        observed = safe_float(row.get(field))
        expected = float(CHECKPOINT[field])
        checks.append((field, observed is not None and abs(observed - expected) <= tolerance))
    for field in (
        "n_global_aligned_ca_pairs",
        "n_matched_receptor_ca_pairs",
        "n_matched_peptide_ca_pairs",
    ):
        observed = safe_float(row.get(field))
        expected = int(CHECKPOINT[field])
        checks.append((field, observed is not None and int(observed) == expected))

    passed = all(value for _field, value in checks)
    lines.append(f"status: {'PASS' if passed else 'FAIL'}")
    lines.append(f"numeric tolerance: {tolerance}")
    lines.append("")
    for field, field_passed in checks:
        lines.append(
            f"{field}: observed={row.get(field, '')}, expected={CHECKPOINT[field]}, "
            f"check={'PASS' if field_passed else 'FAIL'}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed


def parser_with_defaults(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch exact PyMOL whole-complex CA RMSD for best85 and all PDBs."
    )
    parser.add_argument(
        "--best85",
        default=str(repo_root / "paper_clean_v28_outputs/structure_metrics/complex_rmsd_metrics.csv"),
    )
    parser.add_argument(
        "--all_designs",
        default=str(
            repo_root
            / "paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv"
        ),
    )
    parser.add_argument("--native_jsonl", default=str(repo_root / "17_complexes_native.jsonl"))
    parser.add_argument(
        "--pdb_root",
        default=str(repo_root / "raw_external/pdb_highfold_temperature"),
    )
    parser.add_argument(
        "--strict_audit",
        default=str(
            repo_root
            / "paper_clean_v28_outputs/structure_metrics/rmsd_recheck_all_designs/"
            "all_design_receptor_backbone_rmsd_by_pdb.csv"
        ),
        help="Optional prior audit used only for peptide-chain labels.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(
            repo_root
            / "paper_clean_v28_outputs/structure_metrics/global_complex_ca_rmsd"
        ),
    )
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--expected_best85", type=int, default=85)
    parser.add_argument("--expected_all_pdbs", type=int, default=4108)
    parser.add_argument("--expected_unique_designs", type=int, default=4015)
    parser.add_argument("--checkpoint_tolerance", type=float, default=0.005)
    return parser


def main() -> None:
    # PyMOL recommends disabling undo for automation that repeatedly loads and
    # transforms many structures; otherwise the undo history can retain a large
    # amount of coordinate data during this 4,108-PDB run.
    cmd.set("suspend_undo", 1)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    parser = parser_with_defaults(repo_root)
    # PyMOL itself may leave flags in sys.argv. Only this script's known flags matter.
    args, _unknown = parser.parse_known_args(sys.argv[1:])

    best85_path = Path(args.best85)
    all_designs_path = Path(args.all_designs)
    native_path = Path(args.native_jsonl)
    pdb_root = Path(args.pdb_root)
    strict_audit_path = Path(args.strict_audit)
    out_dir = Path(args.out_dir)

    for required in (best85_path, all_designs_path, native_path, pdb_root):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive")

    started = time.time()
    best_source = read_csv(best85_path)
    all_design_rows = read_csv(all_designs_path)
    designs = unique_design_index(all_design_rows)
    native_records = load_native_records(native_path)
    pdb_files = sorted(pdb_root.rglob("*.pdb"))
    strict_rows = read_csv(strict_audit_path) if strict_audit_path.is_file() else []
    strict_index = audit_index(strict_rows)

    if args.expected_best85 and len(best_source) != args.expected_best85:
        raise RuntimeError(
            f"best85 count gate failed: observed {len(best_source)}, expected {args.expected_best85}"
        )
    if args.expected_all_pdbs and len(pdb_files) != args.expected_all_pdbs:
        raise RuntimeError(
            f"all-PDB count gate failed: observed {len(pdb_files)}, expected {args.expected_all_pdbs}"
        )
    if args.expected_unique_designs and len(designs) != args.expected_unique_designs:
        raise RuntimeError(
            "unique-design count gate failed: "
            f"observed {len(designs)}, expected {args.expected_unique_designs}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print("===== PYMOL GLOBAL COMPLEX CA RMSD BATCH =====", flush=True)
    print("best85 rows:", len(best_source), flush=True)
    print("all PDB files:", len(pdb_files), flush=True)
    print("unique designs:", len(designs), flush=True)
    print("strict chain-label audit available:", bool(strict_rows), flush=True)

    print("\nPhase 1/2: best85", flush=True)
    best85_rows = evaluate_best85(
        best_source,
        native_records,
        args.threshold,
        repo_root,
    )
    best85_pass = [
        row for row in best85_rows
        if row.get("global_complex_ca_rmsd_status") == "ok"
        and str(row.get("passes_global_complex_ca_rmsd_lt_threshold", "")) == "1"
    ]
    write_csv(out_dir / "global_complex_ca_rmsd_best85.csv", best85_rows)
    write_csv(out_dir / "global_complex_ca_rmsd_best85_lt3.csv", best85_pass)
    write_csv(
        out_dir / "global_complex_ca_rmsd_best85_summary_by_target.csv",
        summarize(best85_rows, ("target_name",)),
    )
    write_csv(
        out_dir / "global_complex_ca_rmsd_best85_summary_by_temperature.csv",
        summarize(best85_rows, ("temperature",)),
    )
    (out_dir / "global_complex_ca_rmsd_best85_report.txt").write_text(
        "\n".join(
            cohort_report_lines(
                "===== BEST85 PYMOL GLOBAL COMPLEX CA RMSD =====",
                best85_rows,
                args.threshold,
                args.expected_best85,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"best85 result ready: OK={sum(row.get('global_complex_ca_rmsd_status') == 'ok' for row in best85_rows)}, "
        f"RMSD<{args.threshold:g}={len(best85_pass)}",
        flush=True,
    )

    print("\nPhase 2/2: all PDBs", flush=True)
    all_pdb_rows = evaluate_all_pdbs(
        pdb_files,
        designs,
        strict_index,
        native_records,
        args.threshold,
        repo_root,
    )
    all_pdb_pass = [
        row for row in all_pdb_rows
        if row.get("global_complex_ca_rmsd_status") == "ok"
        and str(row.get("passes_global_complex_ca_rmsd_lt_threshold", "")) == "1"
    ]
    problem_rows = [
        row for row in all_pdb_rows
        if row.get("global_complex_ca_rmsd_status") != "ok"
    ]
    confidence_designs, exploratory_designs = build_design_tables(
        designs, all_pdb_rows, args.threshold
    )
    confidence_pass = [
        row for row in confidence_designs
        if row.get("global_complex_ca_rmsd_status") == "ok"
        and str(row.get("passes_global_complex_ca_rmsd_lt_threshold", "")) == "1"
    ]
    downstream_candidates = [
        row for row in exploratory_designs
        if row.get("global_complex_ca_rmsd_status") == "ok"
        and str(row.get("passes_global_complex_ca_rmsd_lt_threshold", "")) == "1"
    ]

    write_csv(out_dir / "global_complex_ca_rmsd_all_pdbs.csv", all_pdb_rows)
    write_csv(out_dir / "global_complex_ca_rmsd_all_pdbs_lt3.csv", all_pdb_pass)
    write_csv(out_dir / "global_complex_ca_rmsd_problem_rows.csv", problem_rows)
    write_csv(
        out_dir / "global_complex_ca_rmsd_all_designs_confidence_representative.csv",
        confidence_designs,
    )
    write_csv(
        out_dir / "global_complex_ca_rmsd_all_designs_confidence_representative_lt3.csv",
        confidence_pass,
    )
    write_csv(
        out_dir / "global_complex_ca_rmsd_all_designs_best_rmsd_exploratory.csv",
        exploratory_designs,
    )
    write_csv(
        out_dir / "global_complex_ca_rmsd_lt3_candidates_for_downstream.csv",
        downstream_candidates,
    )
    write_csv(
        out_dir / "global_complex_ca_rmsd_all_pdbs_summary_by_target_temperature.csv",
        summarize(all_pdb_rows, ("target_name", "temperature")),
    )
    write_csv(
        out_dir / "global_complex_ca_rmsd_all_designs_summary_by_target_temperature.csv",
        summarize(confidence_designs, ("target_name", "temperature")),
    )

    checkpoint_ok = write_checkpoint_report(
        out_dir / "pymol_manual_checkpoint_4kel.txt",
        all_pdb_rows,
        args.checkpoint_tolerance,
    )
    report_lines = cohort_report_lines(
        "===== ALL-PDB PYMOL GLOBAL COMPLEX CA RMSD =====",
        all_pdb_rows,
        args.threshold,
        args.expected_all_pdbs,
    )
    report_lines.extend(
        [
            "",
            "Unique-design accounting:",
            f"  raw all_design rows: {len(all_design_rows)}",
            f"  unique exact design keys: {len(designs)}",
            f"  confidence representatives RMSD OK: "
            f"{sum(row.get('global_complex_ca_rmsd_status') == 'ok' for row in confidence_designs)}",
            f"  confidence representatives RMSD < {args.threshold:.3f}: {len(confidence_pass)}",
            f"  unique designs with any PDB RMSD < {args.threshold:.3f} "
            f"(exploratory): {len(downstream_candidates)}",
            "",
            "Selection rules:",
            "  Primary unique-design table: highest mean CA B-factor among RMSD-OK PDBs.",
            "  Downstream discovery manifest: lowest global RMSD PDB per design, only when < threshold.",
            "  The exploratory minimum must not be reported as an unbiased population success rate.",
            "",
            f"4KEL manual checkpoint: {'PASS' if checkpoint_ok else 'FAIL'}",
        ]
    )
    (out_dir / "global_complex_ca_rmsd_all_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    try:
        pymol_version = cmd.get_version()
    except Exception:
        pymol_version = "unknown"
    write_json(
        out_dir / "global_complex_ca_rmsd_run_metadata.json",
        {
            "metric": "PyMOL whole-complex CA align",
            "cycles": 0,
            "cutoff": 2.0,
            "gap": -10.0,
            "extend": -0.5,
            "max_gap": 50,
            "matrix": "BLOSUM62",
            "max_skip": 0,
            "threshold_angstrom_strict_lt": args.threshold,
            "pymol_version": pymol_version,
            "best85_rows": len(best85_rows),
            "all_pdb_rows": len(all_pdb_rows),
            "unique_designs": len(designs),
            "strict_chain_label_audit_used": bool(strict_rows),
            "manual_4kel_checkpoint_pass": checkpoint_ok,
            "elapsed_seconds": time.time() - started,
        },
    )

    print("\n===== COMPLETE =====", flush=True)
    print(
        f"best85: {len(best85_pass)}/{sum(row.get('global_complex_ca_rmsd_status') == 'ok' for row in best85_rows)} "
        f"RMSD < {args.threshold:g}",
        flush=True,
    )
    print(
        f"all PDBs: {len(all_pdb_pass)}/{sum(row.get('global_complex_ca_rmsd_status') == 'ok' for row in all_pdb_rows)} "
        f"RMSD < {args.threshold:g}",
        flush=True,
    )
    print(
        f"unique confidence representatives: {len(confidence_pass)}/"
        f"{sum(row.get('global_complex_ca_rmsd_status') == 'ok' for row in confidence_designs)} "
        f"RMSD < {args.threshold:g}",
        flush=True,
    )
    print(
        f"unique downstream candidates (any PDB, exploratory): {len(downstream_candidates)}",
        flush=True,
    )
    print("4KEL manual checkpoint:", "PASS" if checkpoint_ok else "FAIL", flush=True)
    print("output directory:", out_dir, flush=True)

    if not checkpoint_ok:
        raise RuntimeError(
            "The batch implementation did not reproduce the confirmed 4KEL PyMOL result. "
            "Do not use these outputs downstream; inspect pymol_manual_checkpoint_4kel.txt."
        )


if __name__ in {"__main__", "pymol"}:
    main()
