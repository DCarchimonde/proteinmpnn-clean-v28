#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Batch independent PyMOL C-alpha RMSD for the final cyclic-peptide chain.

This script is the companion to
``12_compute_pymol_global_complex_ca_rmsd.py``.  It does not change or
recalculate the confirmed whole-complex result.  Instead, it:

1. reads the already validated best85 and all-PDB global-RMSD tables;
2. treats the final chain in each predicted PDB and native JSONL record as the
   cyclic peptide, as specified by the project;
3. naturalizes N-methyl residue identities from
   ``nmethyl/utils/nmethyl_config.py`` without changing coordinates;
4. independently applies the same PyMOL CA ``align(..., cycles=0)`` operation
   to the two cyclic-peptide chains;
5. keeps the same strict ``RMSD < 3 Angstrom`` classification; and
6. writes global, peptide, and joint-pass tables for best85, all PDBs, and
   unique designs.

The peptide calculation is a self-superposed chain RMSD. It measures cyclic
peptide conformation after the peptide itself is fitted; it is deliberately
different from the no-second-fit, whole-complex-frame peptide-position audit
in ``13_compute_global_and_cyclic_peptide_ca_rmsd.py``. Sequence-alignment pair
counts and coverage are therefore written beside every RMSD and never silently
used as an extra pass criterion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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
except ImportError:  # Allows pure helper tests outside the user's PyMOL env.
    cmd = None


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

PYMOL_OBJECTS = (
    "cyclic_peptide_pred",
    "cyclic_peptide_native",
    "cyclic_peptide_ca_alignment",
)

CHECKPOINT = {
    "pdb_file": "4kel_13_rcrrrGNrQGQCGR_model.pdb",
    "temperature": "0.3",
    "global_complex_ca_rmsd": 1.8244132995605469,
    "predicted_final_chain": "B",
    "native_final_chain": "B",
    "design_natural_seq": "RCRRRGNRQGQCGR",
    "n_predicted_cyclic_peptide_ca": 14,
    "n_native_cyclic_peptide_ca": 14,
}


def norm_temp(value: object) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return None if math.isnan(number) else number
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
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_repo_path(repo_root: Path, value: object) -> Path:
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


def parse_temperature(path: Path) -> str:
    normalized = str(path).replace("\\", "/").lower()
    for folder, temperature in TEMP_MAP.items():
        if folder in normalized:
            return temperature
    return ""


def parse_pdb_filename(path: Path) -> dict:
    match = re.match(r"^([A-Za-z0-9]+)_(\d+)_(.+)_model\.pdb$", path.name)
    if not match:
        return {"target_name": "", "design_seq": "", "file_index": ""}
    return {
        "target_name": match.group(1).upper(),
        "file_index": match.group(2),
        "design_seq": match.group(3),
    }


def design_key(row: Mapping[str, object]) -> Tuple[str, str, str]:
    return (
        str(row.get("target_name", "")).upper(),
        norm_temp(row.get("temperature")),
        str(row.get("design_seq", "")),
    )


def load_native_records(path: Path) -> Dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            target = str(record.get("name", "")).upper()
            if target:
                records[target] = record
    return records


def load_residue_maps(path: Path) -> Tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"N-methyl residue configuration not found: {path}")
    spec = importlib.util.spec_from_file_location("project_nmethyl_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load residue configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    natural_map = {
        str(key).upper(): str(value)
        for key, value in dict(module.NATURAL_RESIDUE_MAP).items()
    }
    nmethyl_map = {
        str(key).upper(): str(value)
        for key, value in dict(module.NMETHYL_RESIDUE_MAP).items()
    }

    problems = []
    if len(natural_map) != 20:
        problems.append(f"expected 20 natural entries, observed {len(natural_map)}")
    if len(nmethyl_map) != 20:
        problems.append(f"expected 20 N-methyl entries, observed {len(nmethyl_map)}")
    if set(natural_map.values()) != set(AA1_TO_3):
        problems.append("natural-map values do not equal the 20 standard one-letter codes")
    invalid_nmethyl = {
        key: value
        for key, value in nmethyl_map.items()
        if len(value) != 1 or not value.islower() or value.upper() not in AA1_TO_3
    }
    if invalid_nmethyl:
        problems.append(f"invalid N-methyl parent tokens: {invalid_nmethyl}")
    if problems:
        raise RuntimeError("Residue-map quality gate failed: " + "; ".join(problems))
    return natural_map, nmethyl_map


def naturalize_design_sequence(
    sequence: object,
    natural_map: Mapping[str, str],
    nmethyl_map: Mapping[str, str],
) -> str:
    text = str(sequence or "").strip()
    natural_tokens = set(natural_map.values())
    methyl_tokens = set(nmethyl_map.values())
    output = []
    invalid = []
    for index, token in enumerate(text, start=1):
        if token in natural_tokens:
            output.append(token)
        elif token in methyl_tokens:
            output.append(token.upper())
        else:
            invalid.append(f"{index}:{token!r}")
    if invalid:
        raise ValueError(
            "Design sequence contains tokens absent from nmethyl_config.py: "
            + ",".join(invalid)
        )
    return "".join(output)


def parse_predicted_ca_residues(path: Path) -> Tuple[List[str], Dict[str, List[dict]]]:
    chain_order: List[str] = []
    residues: Dict[str, List[dict]] = defaultdict(list)
    seen = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16].strip() not in ("", "A"):
                continue
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            if chain not in residues:
                chain_order.append(chain)
            try:
                xyz = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue
            residues[chain].append(
                {
                    "record_type": line[:6].strip().upper(),
                    "resname": line[17:20].strip().upper(),
                    "resseq": resseq,
                    "icode": icode,
                    "xyz": xyz,
                }
            )
    return chain_order, dict(residues)


def native_chain_order(record: Mapping[str, object]) -> List[str]:
    return [
        key[len("seq_chain_") :]
        for key in record
        if key.startswith("seq_chain_")
    ]


def native_chain_ca(
    record: Mapping[str, object],
    chain: str,
) -> Tuple[str, List[dict]]:
    sequence = str(record.get(f"seq_chain_{chain}", ""))
    coordinates = record.get(f"CA_chain_{chain}")
    if coordinates is None:
        nested = record.get(f"coords_chain_{chain}", {})
        coordinates = nested.get(f"CA_chain_{chain}", nested.get("CA", []))
    residues = []
    for index, aa in enumerate(sequence):
        if index >= len(coordinates or []):
            break
        xyz = coordinates[index]
        if xyz is None or len(xyz) != 3:
            continue
        try:
            point = tuple(float(value) for value in xyz)
        except (TypeError, ValueError):
            continue
        residues.append(
            {
                "record_type": "ATOM",
                "resname": AA1_TO_3.get(aa.upper(), "UNK"),
                "resseq": str(index + 1),
                "icode": "",
                "xyz": point,
            }
        )
    return sequence.upper(), residues


def mapped_parent(
    residue: Mapping[str, object],
    natural_map: Mapping[str, str],
    nmethyl_map: Mapping[str, str],
) -> Tuple[str, str]:
    resname = str(residue.get("resname", "")).upper()
    record_type = str(residue.get("record_type", "")).upper()
    if record_type == "HETATM" and resname in nmethyl_map:
        return nmethyl_map[resname].upper(), "nmethyl_map"
    if resname in natural_map:
        return natural_map[resname].upper(), "natural_map"
    if resname in nmethyl_map:
        return nmethyl_map[resname].upper(), "nmethyl_map_nonhetatm"
    return "", "unmapped"


def audit_predicted_parent_mapping(
    residues: Sequence[Mapping[str, object]],
    design_natural: str,
    natural_map: Mapping[str, str],
    nmethyl_map: Mapping[str, str],
) -> dict:
    if len(residues) != len(design_natural):
        raise ValueError(
            "Predicted final-chain CA count does not match design length: "
            f"{len(residues)} vs {len(design_natural)}"
        )
    counts = Counter()
    mismatch_details = []
    unmapped_non_unk = []
    unmapped_codes = Counter()
    for index, (residue, expected) in enumerate(
        zip(residues, design_natural),
        start=1,
    ):
        observed, source = mapped_parent(residue, natural_map, nmethyl_map)
        resname = str(residue.get("resname", "")).upper()
        if observed:
            counts[source] += 1
            if observed == expected:
                counts["explicit_parent_match"] += 1
            else:
                mismatch_details.append(
                    f"{index}:{resname}->{observed},expected={expected}"
                )
        else:
            unmapped_codes[resname] += 1
            counts["inferred_from_design"] += 1
            if resname != "UNK":
                unmapped_non_unk.append(f"{index}:{resname}")

    if mismatch_details:
        raise ValueError(
            "Mapped PDB residue parents contradict design_seq: "
            + ";".join(mismatch_details)
        )
    if unmapped_non_unk:
        raise ValueError(
            "PDB contains non-UNK residue codes absent from nmethyl_config.py: "
            + ";".join(unmapped_non_unk)
        )
    return {
        "n_parent_explicitly_mapped": counts["explicit_parent_match"],
        "n_nmethyl_residues_mapped": (
            counts["nmethyl_map"] + counts["nmethyl_map_nonhetatm"]
        ),
        "n_nmethyl_residues_encoded_as_nonhetatm": counts["nmethyl_map_nonhetatm"],
        "n_parent_inferred_from_design_for_unk": counts["inferred_from_design"],
        "unmapped_pdb_resname_counts": ";".join(
            f"{key}:{value}" for key, value in sorted(unmapped_codes.items())
        ),
        "n_parent_mapping_mismatches": 0,
        "residue_parent_mapping_gate": "PASS",
    }


def ca_object_pdbstr(
    sequence: str,
    residues: Sequence[Mapping[str, object]],
    chain: str = "P",
) -> str:
    if len(sequence) != len(residues):
        raise ValueError(
            f"CA object length mismatch: sequence={len(sequence)}, residues={len(residues)}"
        )
    lines = []
    for serial, (aa, residue) in enumerate(zip(sequence, residues), start=1):
        if aa not in AA1_TO_3:
            raise ValueError(f"Non-natural parent token at position {serial}: {aa!r}")
        x, y, z = residue["xyz"]
        resname = AA1_TO_3[aa]
        lines.append(
            f"ATOM  {serial:5d} {'CA':^4s} {resname:>3s} "
            f"{chain:1s}{serial:4d}    "
            f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
            f"  1.00  0.00           C"
        )
    lines.extend(("TER", "END"))
    return "\n".join(lines) + "\n"


def cleanup_pymol_objects(cmd_api) -> None:
    if cmd_api is None:
        return
    for name in PYMOL_OBJECTS:
        try:
            cmd_api.delete(name)
        except Exception:
            pass


def binary_pass(row: Mapping[str, object], field: str) -> Optional[int]:
    value = row.get(field)
    if value in (1, "1", True, "True", "true"):
        return 1
    if value in (0, "0", False, "False", "false"):
        return 0
    return None


def evaluate_row(
    global_row: Mapping[str, object],
    native_record: Mapping[str, object],
    natural_map: Mapping[str, str],
    nmethyl_map: Mapping[str, str],
    threshold: float,
    repo_root: Path,
    cmd_api=None,
) -> dict:
    cmd_api = cmd if cmd_api is None else cmd_api
    row = dict(global_row)
    row.update(
        {
            "cyclic_peptide_chain_rule": "final_chain_in_file_or_json_key_order",
            "cyclic_peptide_fit_mode": "independent_peptide_pymol_align",
            "cyclic_peptide_align_selection_mobile": (
                "cyclic_peptide_pred and name CA"
            ),
            "cyclic_peptide_align_selection_target": (
                "cyclic_peptide_native and name CA"
            ),
            "cyclic_peptide_align_cycles": 0,
            "cyclic_peptide_align_cutoff": 2.0,
            "cyclic_peptide_align_gap": -10.0,
            "cyclic_peptide_align_extend": -0.5,
            "cyclic_peptide_align_max_gap": 50,
            "cyclic_peptide_align_matrix": "BLOSUM62",
            "cyclic_peptide_align_max_skip": 0,
            "cyclic_peptide_ca_rmsd_threshold_angstrom": threshold,
            "cyclic_peptide_ca_rmsd_status": "",
            "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold": "",
        }
    )

    try:
        if cmd_api is None:
            raise RuntimeError(
                "PyMOL module unavailable. Run with: pymol -cq -r "
                "paper_clean_v28/structure_metrics/"
                "14_compute_pymol_independent_cyclic_peptide_ca_rmsd.py"
            )
        pdb_path = resolve_repo_path(repo_root, row.get("pdb_path", ""))
        if not pdb_path.is_file():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")
        parsed = parse_pdb_filename(pdb_path)
        design_seq = str(row.get("design_seq") or parsed["design_seq"])
        design_natural = naturalize_design_sequence(
            design_seq,
            natural_map,
            nmethyl_map,
        )

        predicted_order, predicted_chains = parse_predicted_ca_residues(pdb_path)
        if not predicted_order:
            raise ValueError("No predicted CA-containing chains found")
        predicted_final_chain = predicted_order[-1]
        predicted_residues = predicted_chains[predicted_final_chain]

        native_order = native_chain_order(native_record)
        if not native_order:
            raise ValueError("No native seq_chain_* entries found")
        native_final_chain = native_order[-1]
        native_sequence, native_residues = native_chain_ca(
            native_record,
            native_final_chain,
        )

        parent_audit = audit_predicted_parent_mapping(
            predicted_residues,
            design_natural,
            natural_map,
            nmethyl_map,
        )
        if len(native_residues) != len(native_sequence):
            raise ValueError(
                "Native final-chain CA count does not match native sequence length: "
                f"{len(native_residues)} vs {len(native_sequence)}"
            )

        prior_predicted_chain = str(row.get("predicted_peptide_chain", "")).strip()
        prior_native_chain = str(
            row.get("native_peptide_chain")
            or row.get("native_peptide_chain_used")
            or ""
        ).strip()
        row.update(parent_audit)
        row.update(
            {
                "design_natural_seq": design_natural,
                "predicted_chain_order": ";".join(predicted_order),
                "native_chain_order": ";".join(native_order),
                "predicted_final_chain": predicted_final_chain,
                "native_final_chain": native_final_chain,
                "prior_predicted_peptide_chain": prior_predicted_chain,
                "prior_native_peptide_chain": prior_native_chain,
                "final_chain_matches_prior_predicted_peptide_chain": (
                    int(predicted_final_chain == prior_predicted_chain)
                    if prior_predicted_chain else ""
                ),
                "final_chain_matches_prior_native_peptide_chain": (
                    int(native_final_chain == prior_native_chain)
                    if prior_native_chain else ""
                ),
                "n_predicted_cyclic_peptide_ca": len(predicted_residues),
                "n_native_cyclic_peptide_ca": len(native_residues),
                "native_cyclic_peptide_seq": native_sequence,
                "cyclic_peptide_length_match": int(
                    len(predicted_residues) == len(native_residues)
                ),
            }
        )

        predicted_pdbstr = ca_object_pdbstr(design_natural, predicted_residues)
        native_pdbstr = ca_object_pdbstr(native_sequence, native_residues)
        cleanup_pymol_objects(cmd_api)
        cmd_api.read_pdbstr(predicted_pdbstr, "cyclic_peptide_pred")
        cmd_api.read_pdbstr(native_pdbstr, "cyclic_peptide_native")
        cmd_api.sort("cyclic_peptide_pred")
        cmd_api.sort("cyclic_peptide_native")

        result = cmd_api.align(
            "cyclic_peptide_pred and name CA",
            "cyclic_peptide_native and name CA",
            cutoff=2.0,
            cycles=0,
            gap=-10.0,
            extend=-0.5,
            max_gap=50,
            object="cyclic_peptide_ca_alignment",
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
        try:
            raw_columns = cmd_api.get_raw_alignment("cyclic_peptide_ca_alignment")
            n_raw_columns = len(raw_columns)
        except Exception:
            n_raw_columns = int(n_after)

        pred_count = len(predicted_residues)
        native_count = len(native_residues)
        peptide_pass = int(float(rms_after) < threshold)
        global_pass = binary_pass(
            row,
            "passes_global_complex_ca_rmsd_lt_threshold",
        )
        if global_pass is None:
            global_rmsd = safe_float(row.get("global_complex_ca_rmsd"))
            global_pass = (
                int(global_rmsd < threshold)
                if global_rmsd is not None
                and row.get("global_complex_ca_rmsd_status") == "ok"
                else None
            )
        row.update(
            {
                "cyclic_peptide_ca_rmsd": fmt(rms_after),
                "n_cyclic_peptide_aligned_ca_pairs": int(n_after),
                "cyclic_peptide_align_cycles_performed": int(n_cycles),
                "cyclic_peptide_ca_rmsd_before_rejection": fmt(rms_before),
                "n_cyclic_peptide_aligned_ca_pairs_before_rejection": int(n_before),
                "cyclic_peptide_align_raw_score": fmt(raw_score),
                "cyclic_peptide_align_residues": int(n_residues),
                "n_cyclic_peptide_raw_alignment_columns": n_raw_columns,
                "cyclic_peptide_alignment_coverage_vs_predicted": fmt(
                    int(n_after) / pred_count if pred_count else None
                ),
                "cyclic_peptide_alignment_coverage_vs_native": fmt(
                    int(n_after) / native_count if native_count else None
                ),
                "full_cyclic_peptide_ca_alignment_coverage": int(
                    int(n_after) > 0
                    and int(n_after) == pred_count
                    and int(n_after) == native_count
                ),
                "passes_cyclic_peptide_ca_rmsd_lt_threshold": peptide_pass,
                "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold": (
                    int(global_pass == 1 and peptide_pass == 1)
                    if global_pass is not None else ""
                ),
                "cyclic_peptide_ca_rmsd_status": "ok",
            }
        )
    except Exception as exc:
        row["cyclic_peptide_ca_rmsd_status"] = "failed"
        row["cyclic_peptide_ca_rmsd_error"] = repr(exc)
    finally:
        cleanup_pymol_objects(cmd_api)
    return row


def evaluate_cohort(
    global_rows: Sequence[dict],
    native_records: Mapping[str, dict],
    natural_map: Mapping[str, str],
    nmethyl_map: Mapping[str, str],
    threshold: float,
    repo_root: Path,
    label: str,
) -> List[dict]:
    output = []
    for index, source in enumerate(global_rows, start=1):
        target = str(source.get("target_name", "")).upper()
        if target not in native_records:
            row = dict(source)
            row["cyclic_peptide_ca_rmsd_status"] = "missing_native_target"
            row["cyclic_peptide_ca_rmsd_error"] = target
        else:
            row = evaluate_row(
                source,
                native_records[target],
                natural_map,
                nmethyl_map,
                threshold,
                repo_root,
            )
        output.append(row)
        interval = 10 if len(global_rows) <= 100 else 100
        if index % interval == 0 or index == len(global_rows):
            print(f"[{label}] processed: {index}/{len(global_rows)}", flush=True)
    return output


def unique_design_index(rows: Sequence[dict]) -> Dict[Tuple[str, str, str], List[dict]]:
    output: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        enriched = dict(row)
        enriched["_all_design_row_index"] = index
        output[design_key(row)].append(enriched)
    return dict(output)


def confidence_sort(row: Mapping[str, object]) -> tuple:
    confidence = safe_float(row.get("pdb_ca_bfactor_mean"))
    return (
        -(confidence if confidence is not None else -math.inf),
        str(row.get("pdb_file", "")),
    )


def joint_candidate_sort(row: Mapping[str, object]) -> tuple:
    confidence = safe_float(row.get("pdb_ca_bfactor_mean"))
    global_rmsd = safe_float(row.get("global_complex_ca_rmsd"))
    peptide_rmsd = safe_float(row.get("cyclic_peptide_ca_rmsd"))
    maximum = max(global_rmsd, peptide_rmsd)
    return (
        -(confidence if confidence is not None else -math.inf),
        maximum,
        global_rmsd + peptide_rmsd,
        str(row.get("pdb_file", "")),
    )


def build_unique_design_tables(
    designs: Mapping[Tuple[str, str, str], Sequence[dict]],
    pdb_rows: Sequence[dict],
    threshold: float,
) -> Tuple[List[dict], List[dict]]:
    by_key: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in pdb_rows:
        by_key[design_key(row)].append(row)

    confidence_rows = []
    joint_manifest = []
    for key, raw_rows in sorted(designs.items(), key=lambda item: item[0]):
        candidates = by_key.get(key, [])
        global_ok = [
            row
            for row in candidates
            if row.get("global_complex_ca_rmsd_status") == "ok"
        ]
        peptide_ok = [
            row
            for row in candidates
            if row.get("cyclic_peptide_ca_rmsd_status") == "ok"
        ]
        joint_pass = [
            row
            for row in candidates
            if binary_pass(
                row,
                "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold",
            )
            == 1
        ]
        base = dict(raw_rows[0])
        base.update(
            {
                "n_raw_design_rows_for_key": len(raw_rows),
                "raw_all_design_row_indices": ";".join(
                    str(row["_all_design_row_index"]) for row in raw_rows
                ),
                "n_pdb_for_key": len(candidates),
                "n_global_rmsd_ok_pdb_for_key": len(global_ok),
                "n_cyclic_peptide_rmsd_ok_pdb_for_key": len(peptide_ok),
                "n_joint_lt_threshold_pdb_for_key": len(joint_pass),
                "any_pdb_joint_global_and_cyclic_peptide_lt_threshold": int(
                    bool(joint_pass)
                ),
                "rmsd_threshold_angstrom_strict_lt": threshold,
            }
        )

        if global_ok:
            representative = sorted(global_ok, key=confidence_sort)[0]
            confidence_row = {**base, **representative}
            confidence_row["unique_design_representative_rule"] = (
                "highest_mean_ca_bfactor_among_global_rmsd_ok_pdbs"
            )
        elif candidates:
            confidence_row = {**base, **candidates[0]}
            confidence_row["unique_design_representative_rule"] = (
                "first_pdb_no_global_rmsd_ok_result"
            )
        else:
            confidence_row = dict(base)
            confidence_row.update(
                {
                    "global_complex_ca_rmsd_status": "missing_pdb",
                    "cyclic_peptide_ca_rmsd_status": "missing_pdb",
                    "unique_design_representative_rule": "missing_pdb",
                }
            )
        confidence_rows.append(confidence_row)

        if joint_pass:
            selected = sorted(joint_pass, key=joint_candidate_sort)[0]
            manifest_row = {**base, **selected}
            manifest_row["unique_design_representative_rule"] = (
                "highest_mean_ca_bfactor_among_joint_lt3_pdbs_then_lower_rmsd"
            )
            joint_manifest.append(manifest_row)
    return confidence_rows, joint_manifest


def summarize(rows: Sequence[dict], keys: Sequence[str]) -> List[dict]:
    groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output = []
    for values, items in sorted(groups.items()):
        ok = [
            row
            for row in items
            if row.get("cyclic_peptide_ca_rmsd_status") == "ok"
        ]
        rmsds = [
            safe_float(row.get("cyclic_peptide_ca_rmsd"))
            for row in ok
        ]
        rmsds = [value for value in rmsds if value is not None]
        result = {key: value for key, value in zip(keys, values)}
        result.update(
            {
                "n_rows": len(items),
                "n_cyclic_peptide_rmsd_ok": len(ok),
                "n_cyclic_peptide_rmsd_failed": len(items) - len(ok),
                "n_cyclic_peptide_rmsd_lt_threshold": sum(
                    binary_pass(
                        row,
                        "passes_cyclic_peptide_ca_rmsd_lt_threshold",
                    )
                    == 1
                    for row in items
                ),
                "n_joint_global_and_cyclic_peptide_lt_threshold": sum(
                    binary_pass(
                        row,
                        "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold",
                    )
                    == 1
                    for row in items
                ),
                "mean_cyclic_peptide_ca_rmsd": fmt(
                    mean(rmsds) if rmsds else None
                ),
                "median_cyclic_peptide_ca_rmsd": fmt(
                    median(rmsds) if rmsds else None
                ),
                "n_full_cyclic_peptide_ca_alignment_coverage": sum(
                    binary_pass(
                        row,
                        "full_cyclic_peptide_ca_alignment_coverage",
                    )
                    == 1
                    for row in items
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
    peptide_ok = [
        row
        for row in rows
        if row.get("cyclic_peptide_ca_rmsd_status") == "ok"
    ]
    peptide_pass = [
        row
        for row in peptide_ok
        if binary_pass(row, "passes_cyclic_peptide_ca_rmsd_lt_threshold") == 1
    ]
    global_pass = [
        row
        for row in rows
        if binary_pass(row, "passes_global_complex_ca_rmsd_lt_threshold") == 1
    ]
    joint_pass = [
        row
        for row in rows
        if binary_pass(
            row,
            "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold",
        )
        == 1
    ]
    coverage = [
        safe_float(row.get("cyclic_peptide_alignment_coverage_vs_native"))
        for row in peptide_ok
    ]
    coverage = [value for value in coverage if value is not None]
    statuses = Counter(
        str(row.get("cyclic_peptide_ca_rmsd_status", ""))
        for row in rows
    )
    lines = [
        title,
        "",
        "Exact requested cyclic-peptide metric:",
        "  final chain of predicted PDB versus final seq_chain_* of native JSONL",
        "  N-methyl identities naturalized from nmethyl/utils/nmethyl_config.py",
        "  PyMOL align on cyclic-peptide CA atoms only",
        "  cycles=0 (no structural outlier rejection)",
        "  BLOSUM62 parameters identical to the confirmed whole-complex run",
        "  peptide is independently fitted (self-superposed chain RMSD)",
        f"  strict classification = RMSD < {threshold:.3f} Angstrom",
        "  alignment coverage is reported but does not change the requested class",
        "",
        f"rows: {len(rows)}",
        f"expected rows: {expected_count}",
        f"count gate: {'PASS' if len(rows) == expected_count else 'FAIL'}",
        f"cyclic-peptide RMSD OK: {len(peptide_ok)}",
        f"cyclic-peptide RMSD failed/missing: {len(rows) - len(peptide_ok)}",
        f"global RMSD < {threshold:.3f}: {len(global_pass)}",
        f"cyclic-peptide RMSD < {threshold:.3f}: {len(peptide_pass)}",
        f"joint global AND cyclic-peptide RMSD < {threshold:.3f}: {len(joint_pass)}",
        (
            f"cyclic-peptide pass fraction among RMSD-OK: "
            f"{len(peptide_pass) / len(peptide_ok):.8f}"
            if peptide_ok
            else "cyclic-peptide pass fraction among RMSD-OK: NA"
        ),
        (
            f"full cyclic-peptide CA alignment coverage: "
            f"{sum(binary_pass(row, 'full_cyclic_peptide_ca_alignment_coverage') == 1 for row in peptide_ok)}"
            f"/{len(peptide_ok)}"
        ),
        (
            f"median cyclic-peptide CA alignment coverage vs native: "
            f"{median(coverage):.6f}"
            if coverage
            else "median cyclic-peptide CA alignment coverage vs native: NA"
        ),
        f"residue-parent mapping gate PASS: "
        f"{sum(row.get('residue_parent_mapping_gate') == 'PASS' for row in rows)}"
        f"/{len(rows)}",
        "",
        "Status counts:",
    ]
    lines.extend(f"  {status}: {count}" for status, count in sorted(statuses.items()))
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "  This is the requested peptide-only, self-superposed CA RMSD.",
            "  It measures peptide conformation after an independent peptide fit.",
            "  It does not measure the peptide pose in the receptor coordinate frame.",
            "  The earlier receptor-frame peptide RMSD remains a separate metric.",
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
    lines = ["===== 4KEL CYCLIC-PEPTIDE DATA/IMPLEMENTATION CHECKPOINT =====", ""]
    if len(matches) != 1:
        lines.extend(
            [
                "status: FAIL",
                f"expected one checkpoint row, observed {len(matches)}",
            ]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return False

    row = matches[0]
    checks = []
    observed_global = safe_float(row.get("global_complex_ca_rmsd"))
    checks.append(
        (
            "global_complex_ca_rmsd",
            observed_global is not None
            and abs(observed_global - CHECKPOINT["global_complex_ca_rmsd"]) <= tolerance,
            row.get("global_complex_ca_rmsd", ""),
            CHECKPOINT["global_complex_ca_rmsd"],
        )
    )
    for field in (
        "predicted_final_chain",
        "native_final_chain",
        "design_natural_seq",
        "n_predicted_cyclic_peptide_ca",
        "n_native_cyclic_peptide_ca",
    ):
        observed = row.get(field, "")
        expected = CHECKPOINT[field]
        checks.append((field, str(observed) == str(expected), observed, expected))
    checks.extend(
        [
            (
                "residue_parent_mapping_gate",
                row.get("residue_parent_mapping_gate") == "PASS",
                row.get("residue_parent_mapping_gate", ""),
                "PASS",
            ),
            (
                "cyclic_peptide_ca_rmsd_status",
                row.get("cyclic_peptide_ca_rmsd_status") == "ok",
                row.get("cyclic_peptide_ca_rmsd_status", ""),
                "ok",
            ),
            (
                "positive_alignment_pair_count",
                (safe_float(row.get("n_cyclic_peptide_aligned_ca_pairs")) or 0) > 0,
                row.get("n_cyclic_peptide_aligned_ca_pairs", ""),
                ">0",
            ),
        ]
    )
    passed = all(item[1] for item in checks)
    lines.append(f"status: {'PASS' if passed else 'FAIL'}")
    lines.append(f"global numeric tolerance: {tolerance}")
    lines.append("")
    for field, field_passed, observed, expected in checks:
        lines.append(
            f"{field}: observed={observed}, expected={expected}, "
            f"check={'PASS' if field_passed else 'FAIL'}"
        )
    lines.extend(
        [
            "",
            f"cyclic_peptide_ca_rmsd: {row.get('cyclic_peptide_ca_rmsd', '')}",
            "note: no peptide numeric value is hard-coded; this first batch run "
            "establishes the auditable value.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed


def parser_with_defaults(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch final-chain cyclic-peptide PyMOL CA RMSD for best85 and all PDBs."
        )
    )
    global_dir = (
        repo_root
        / "paper_clean_v28_outputs/structure_metrics/global_complex_ca_rmsd"
    )
    parser.add_argument(
        "--global_best85",
        default=str(global_dir / "global_complex_ca_rmsd_best85.csv"),
    )
    parser.add_argument(
        "--global_all_pdbs",
        default=str(global_dir / "global_complex_ca_rmsd_all_pdbs.csv"),
    )
    parser.add_argument(
        "--all_designs",
        default=str(
            repo_root
            / "paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv"
        ),
    )
    parser.add_argument(
        "--native_jsonl",
        default=str(repo_root / "17_complexes_native.jsonl"),
    )
    parser.add_argument(
        "--nmethyl_config",
        default=str(repo_root / "nmethyl/utils/nmethyl_config.py"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(
            repo_root
            / "paper_clean_v28_outputs/structure_metrics/"
            "independent_cyclic_peptide_ca_rmsd"
        ),
    )
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--expected_best85", type=int, default=85)
    parser.add_argument("--expected_all_pdbs", type=int, default=4108)
    parser.add_argument("--expected_unique_designs", type=int, default=4015)
    parser.add_argument("--checkpoint_tolerance", type=float, default=0.005)
    return parser


def main() -> None:
    script_value = globals().get("__script__") or globals().get("__file__")
    if script_value:
        script_path = Path(str(script_value)).resolve()
        repo_root = script_path.parents[2]
    else:
        repo_root = Path.cwd().resolve()
    print("resolved repository root:", repo_root, flush=True)

    parser = parser_with_defaults(repo_root)
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive")

    global_best85_path = Path(args.global_best85)
    global_all_path = Path(args.global_all_pdbs)
    all_designs_path = Path(args.all_designs)
    native_path = Path(args.native_jsonl)
    config_path = Path(args.nmethyl_config)
    out_dir = Path(args.out_dir)
    for required in (
        global_best85_path,
        global_all_path,
        all_designs_path,
        native_path,
        config_path,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")

    started = time.time()
    best_global_rows = read_csv(global_best85_path)
    all_global_rows = read_csv(global_all_path)
    all_design_rows = read_csv(all_designs_path)
    designs = unique_design_index(all_design_rows)
    native_records = load_native_records(native_path)
    natural_map, nmethyl_map = load_residue_maps(config_path)

    gates = (
        ("best85", len(best_global_rows), args.expected_best85),
        ("all PDBs", len(all_global_rows), args.expected_all_pdbs),
        ("unique designs", len(designs), args.expected_unique_designs),
    )
    for label, observed, expected in gates:
        if expected and observed != expected:
            raise RuntimeError(
                f"{label} count gate failed: observed {observed}, expected {expected}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "===== INDEPENDENT PYMOL CYCLIC-PEPTIDE CA RMSD BATCH =====",
        flush=True,
    )
    print("best85 rows:", len(best_global_rows), flush=True)
    print("all PDB rows:", len(all_global_rows), flush=True)
    print("unique designs:", len(designs), flush=True)
    print("natural residue-map entries:", len(natural_map), flush=True)
    print("N-methyl residue-map entries:", len(nmethyl_map), flush=True)

    print("\nPhase 1/2: best85", flush=True)
    best_rows = evaluate_cohort(
        best_global_rows,
        native_records,
        natural_map,
        nmethyl_map,
        args.threshold,
        repo_root,
        "best85",
    )
    best_peptide_pass = [
        row
        for row in best_rows
        if binary_pass(row, "passes_cyclic_peptide_ca_rmsd_lt_threshold") == 1
    ]
    best_joint_pass = [
        row
        for row in best_rows
        if binary_pass(
            row,
            "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold",
        )
        == 1
    ]
    write_csv(out_dir / "cyclic_peptide_ca_rmsd_best85.csv", best_rows)
    write_csv(
        out_dir / "cyclic_peptide_ca_rmsd_best85_lt3.csv",
        best_peptide_pass,
    )
    write_csv(
        out_dir / "global_and_cyclic_peptide_joint_lt3_best85.csv",
        best_joint_pass,
    )
    write_csv(
        out_dir / "cyclic_peptide_ca_rmsd_best85_summary_by_target.csv",
        summarize(best_rows, ("target_name",)),
    )
    (out_dir / "cyclic_peptide_ca_rmsd_best85_report.txt").write_text(
        "\n".join(
            cohort_report_lines(
                "===== BEST85 CYCLIC-PEPTIDE CA RMSD =====",
                best_rows,
                args.threshold,
                args.expected_best85,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"best85 ready: peptide<{args.threshold:g}={len(best_peptide_pass)}, "
        f"joint<{args.threshold:g}={len(best_joint_pass)}",
        flush=True,
    )

    print("\nPhase 2/2: all PDBs", flush=True)
    all_rows = evaluate_cohort(
        all_global_rows,
        native_records,
        natural_map,
        nmethyl_map,
        args.threshold,
        repo_root,
        "all PDBs",
    )
    all_peptide_pass = [
        row
        for row in all_rows
        if binary_pass(row, "passes_cyclic_peptide_ca_rmsd_lt_threshold") == 1
    ]
    all_joint_pass = [
        row
        for row in all_rows
        if binary_pass(
            row,
            "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold",
        )
        == 1
    ]
    problem_rows = [
        row
        for row in all_rows
        if row.get("cyclic_peptide_ca_rmsd_status") != "ok"
    ]
    confidence_designs, joint_manifest = build_unique_design_tables(
        designs,
        all_rows,
        args.threshold,
    )
    confidence_joint_pass = [
        row
        for row in confidence_designs
        if binary_pass(
            row,
            "passes_joint_global_and_cyclic_peptide_ca_rmsd_lt_threshold",
        )
        == 1
    ]

    write_csv(out_dir / "cyclic_peptide_ca_rmsd_all_pdbs.csv", all_rows)
    write_csv(
        out_dir / "cyclic_peptide_ca_rmsd_all_pdbs_lt3.csv",
        all_peptide_pass,
    )
    write_csv(
        out_dir / "global_and_cyclic_peptide_joint_lt3_all_pdbs.csv",
        all_joint_pass,
    )
    write_csv(out_dir / "cyclic_peptide_ca_rmsd_problem_rows.csv", problem_rows)
    write_csv(
        out_dir / "cyclic_peptide_ca_rmsd_all_designs_confidence_representative.csv",
        confidence_designs,
    )
    write_csv(
        out_dir
        / "cyclic_peptide_ca_rmsd_all_designs_confidence_representative_joint_lt3.csv",
        confidence_joint_pass,
    )
    write_csv(
        out_dir
        / "global_and_cyclic_peptide_joint_lt3_candidates_for_downstream.csv",
        joint_manifest,
    )
    write_csv(
        out_dir / "cyclic_peptide_ca_rmsd_all_pdbs_summary_by_target_temperature.csv",
        summarize(all_rows, ("target_name", "temperature")),
    )

    checkpoint_ok = write_checkpoint_report(
        out_dir / "pymol_cyclic_peptide_checkpoint_4kel.txt",
        all_rows,
        args.checkpoint_tolerance,
    )
    report_lines = cohort_report_lines(
        "===== ALL-PDB CYCLIC-PEPTIDE CA RMSD =====",
        all_rows,
        args.threshold,
        args.expected_all_pdbs,
    )
    report_lines.extend(
        [
            "",
            "Unique-design accounting:",
            f"  raw all_design rows: {len(all_design_rows)}",
            f"  unique exact design keys: {len(designs)}",
            f"  confidence representatives with a PDB/global RMSD: "
            f"{sum(row.get('global_complex_ca_rmsd_status') == 'ok' for row in confidence_designs)}",
            f"  confidence representatives joint < {args.threshold:.3f}: "
            f"{len(confidence_joint_pass)}",
            f"  unique designs with any joint-pass PDB: {len(joint_manifest)}",
            "",
            "Selection rules:",
            "  Primary table keeps the original highest-mean-CA-B-factor representative.",
            "  Downstream joint manifest chooses the highest-confidence PDB among",
            "  PDBs passing both strict RMSD thresholds, then lower RMSD as a tie-break.",
            "  Any-PDB selection is exploratory and is not an unbiased success rate.",
            "",
            f"4KEL data/implementation checkpoint: "
            f"{'PASS' if checkpoint_ok else 'FAIL'}",
        ]
    )
    (out_dir / "cyclic_peptide_ca_rmsd_all_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    try:
        pymol_version = cmd.get_version() if cmd is not None else "unavailable"
    except Exception:
        pymol_version = "unknown"
    write_json(
        out_dir / "cyclic_peptide_ca_rmsd_run_metadata.json",
        {
            "metric": "PyMOL final-chain independent cyclic-peptide CA self-align",
            "cycles": 0,
            "cutoff": 2.0,
            "gap": -10.0,
            "extend": -0.5,
            "max_gap": 50,
            "matrix": "BLOSUM62",
            "max_skip": 0,
            "threshold_angstrom_strict_lt": args.threshold,
            "chain_rule": "final chain in PDB/JSONL key order",
            "residue_map_path": relative_path_text(config_path, repo_root),
            "residue_map_sha256": config_sha256,
            "natural_map_entries": len(natural_map),
            "nmethyl_map_entries": len(nmethyl_map),
            "pymol_version": pymol_version,
            "best85_rows": len(best_rows),
            "all_pdb_rows": len(all_rows),
            "unique_designs": len(designs),
            "manual_global_4kel_value_preserved": checkpoint_ok,
            "elapsed_seconds": time.time() - started,
        },
    )

    print("\n===== COMPLETE =====", flush=True)
    print(
        f"best85 cyclic peptide: {len(best_peptide_pass)}/"
        f"{sum(row.get('cyclic_peptide_ca_rmsd_status') == 'ok' for row in best_rows)} "
        f"RMSD < {args.threshold:g}",
        flush=True,
    )
    print(
        f"best85 joint global+peptide: {len(best_joint_pass)}/{len(best_rows)}",
        flush=True,
    )
    print(
        f"all PDB cyclic peptide: {len(all_peptide_pass)}/"
        f"{sum(row.get('cyclic_peptide_ca_rmsd_status') == 'ok' for row in all_rows)} "
        f"RMSD < {args.threshold:g}",
        flush=True,
    )
    print(
        f"all PDB joint global+peptide: {len(all_joint_pass)}/{len(all_rows)}",
        flush=True,
    )
    print(
        f"unique confidence representatives joint: "
        f"{len(confidence_joint_pass)}/"
        f"{sum(row.get('global_complex_ca_rmsd_status') == 'ok' for row in confidence_designs)}",
        flush=True,
    )
    print(
        f"unique downstream joint candidates (any PDB, exploratory): "
        f"{len(joint_manifest)}",
        flush=True,
    )
    print(
        "4KEL data/implementation checkpoint:",
        "PASS" if checkpoint_ok else "FAIL",
        flush=True,
    )
    print("output directory:", out_dir, flush=True)

    if not checkpoint_ok:
        raise RuntimeError(
            "The cyclic-peptide run failed the 4KEL data/implementation checkpoint. "
            "Do not use the joint manifest downstream; inspect "
            "pymol_cyclic_peptide_checkpoint_4kel.txt."
        )


if __name__ in {"__main__", "pymol"}:
    main()
