#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Recompute receptor-frame peptide RMSD for every available HighFold PDB.

This is a deliberately separate audit from ``04_compute_complex_rmsd.py``.
It does not overwrite the historical best85 outputs.

Primary metric
--------------
1. Match predicted and native receptor residues by sequence.
2. Fit the predicted receptor to the native receptor with all available
   backbone heavy atoms (N, CA, C and O), without atom-pair rejection.
3. Apply that one receptor-derived rigid transform to the predicted peptide.
4. Compute peptide backbone RMSD (N, CA, C and O) in the fixed receptor frame.

The peptide is never fitted to the native peptide.  Therefore a misplaced
peptide remains misplaced and cannot obtain an artificially small RMSD merely
by peptide self-superposition.

For complexes with duplicate, sequence-identical native peptide copies and/or
sequence- and receptor-fit-equivalent receptor-chain permutations, the
reported value is the minimum over those chain-label-equivalent comparisons.
No non-equivalent native peptide sequence is considered.

Inputs are the 4,115-row clean design table, the 17 native complexes and the
local (gitignored) HighFold PDB directory.  Outputs include PDB-level results,
one independently selected representative per unique design, the strict
``RMSD < threshold`` subset, problem rows and coverage/success summaries.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


BACKBONE_ATOMS: Tuple[str, ...] = ("N", "CA", "C", "O")

TEMP_MAP = {
    "pdb_highfold4_t001": "0.01",
    "pdb_highfold4_t01": "0.1",
    "pdb_highfold4_t02": "0.2",
    "pdb_highfold4_t03": "0.3",
    "pdb_highfold4_t05": "0.5",
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
    # HighFold modified-residue fallbacks already used by this project.
    "NCY": "C", "GNC": "Q", "MMO": "R", "UNK": "X",
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
    output_fields: List[str] = list(fieldnames or [])
    seen = set(output_fields)
    for row in rows:
        for key in row:
            if key not in seen:
                output_fields.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in output_fields})


def rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"RMSD coordinate shape mismatch: {mobile.shape} vs {target.shape}")
    if len(mobile) == 0:
        raise ValueError("RMSD requires at least one atom pair")
    delta = mobile - target
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def kabsch_fit(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return R, t for row-vector coordinates: ``mobile @ R + t``."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or len(mobile) < 3:
        raise ValueError("Kabsch fit requires at least three paired 3D points")

    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_zero = mobile - mobile_center
    target_zero = target - target_center

    covariance = mobile_zero.T @ target_zero
    u, _singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    translation = target_center - mobile_center @ rotation
    return rotation, translation


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ rotation + translation


def kabsch_self_test() -> float:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.3, 0.2, -0.5], [-0.4, 2.1, 0.8], [0.6, -1.2, 1.7]],
        dtype=float,
    )
    theta = 0.73
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    moved = points @ rotation + np.array([7.1, -3.4, 2.2])
    recovered_r, recovered_t = kabsch_fit(moved, points)
    return rmsd(apply_transform(moved, recovered_r, recovered_t), points)


def parse_temperature(path: Path) -> Tuple[str, str]:
    normalized = str(path).replace("\\", "/").lower()
    for folder, temperature in TEMP_MAP.items():
        if folder in normalized:
            return temperature, folder

    # Defensive fallback for a future directory named with an explicit T value.
    match = re.search(r"(?:^|[/_])t(0(?:\.\d+)?)(?:[/_]|$)", normalized)
    if match:
        return norm_temp(match.group(1)), match.group(0).strip("/_")
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


def residue_to_aa(residue: Mapping[str, object]) -> str:
    name = str(residue.get("resname", "")).strip().upper()
    if len(name) == 1:
        return name
    return AA3_TO_1.get(name, "X")


def chain_sequence(residues: Sequence[Mapping[str, object]]) -> str:
    return "".join(residue_to_aa(residue) for residue in residues)


def parse_pdb_structure(path: Path) -> Tuple[Dict[str, List[dict]], Optional[float]]:
    """Parse N/CA/C/O from ATOM and HETATM records.

    Blank and ``A`` alternate locations are accepted.  A blank location wins
    over ``A`` if both occur.  The returned confidence proxy is the mean CA
    B-factor over the complete complex.
    """
    chains: Dict[str, List[dict]] = defaultdict(list)
    residue_index: Dict[Tuple[str, str, str], int] = {}
    ca_bfactors: List[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in BACKBONE_ATOMS:
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue

            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode)

            try:
                coordinate = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                )
            except ValueError:
                continue

            if key not in residue_index:
                residue_index[key] = len(chains[chain_id])
                chains[chain_id].append(
                    {
                        "resname": line[17:20].strip(),
                        "resseq": resseq,
                        "icode": icode,
                        "atoms": {},
                        "atom_altloc": {},
                    }
                )

            residue = chains[chain_id][residue_index[key]]
            previous_altloc = residue["atom_altloc"].get(atom_name)
            if previous_altloc == "" and altloc == "A":
                continue
            residue["atoms"][atom_name] = coordinate
            residue["atom_altloc"][atom_name] = altloc

            if atom_name == "CA":
                try:
                    ca_bfactors.append(float(line[60:66]))
                except ValueError:
                    pass

    confidence = mean(ca_bfactors) if ca_bfactors else None
    return dict(chains), confidence


def load_native_structures(path: Path) -> Dict[str, Dict[str, List[dict]]]:
    result: Dict[str, Dict[str, List[dict]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            target = str(obj.get("name", "")).upper()
            if not target:
                continue

            chains: Dict[str, List[dict]] = {}
            for key, sequence in obj.items():
                if not key.startswith("seq_chain_"):
                    continue
                chain_id = key[len("seq_chain_") :]
                residues = [
                    {"resname": aa, "resseq": str(index + 1), "icode": "", "atoms": {}}
                    for index, aa in enumerate(str(sequence))
                ]
                for atom_name in BACKBONE_ATOMS:
                    values = obj.get(f"{atom_name}_chain_{chain_id}")
                    if values is None:
                        nested = obj.get(f"coords_chain_{chain_id}", {})
                        values = nested.get(f"{atom_name}_chain_{chain_id}", nested.get(atom_name, []))
                    for index, xyz in enumerate(values or []):
                        if index >= len(residues) or xyz is None or len(xyz) != 3:
                            continue
                        try:
                            coordinate = np.asarray(xyz, dtype=float)
                        except (TypeError, ValueError):
                            continue
                        if np.all(np.isfinite(coordinate)):
                            residues[index]["atoms"][atom_name] = coordinate
                chains[chain_id] = residues
            result[target] = chains
    return result


@lru_cache(maxsize=None)
def semi_global_sequence_pairs(predicted: str, native: str) -> dict:
    """End-gap-free sequence alignment returning residue index pairs.

    Free terminal gaps handle His-tags/linkers without index-shifting the
    receptor.  Internal gaps remain penalized.  The routine is intentionally
    dependency-free and deterministic.
    """
    predicted = str(predicted)
    native = str(native)
    n, m = len(predicted), len(native)
    if n == 0 or m == 0:
        return {"pairs": [], "matches": 0, "identity": 0.0, "native_coverage": 0.0}

    match_score, mismatch_score, gap_score = 2, -1, -2
    score = np.zeros((n + 1, m + 1), dtype=int)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 diagonal, 2 up, 3 left
    trace[1:, 0] = 2
    trace[0, 1:] = 3

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = score[i - 1, j - 1] + (
                match_score if predicted[i - 1] == native[j - 1] else mismatch_score
            )
            up = score[i - 1, j] + gap_score
            left = score[i, j - 1] + gap_score
            best = max(diagonal, up, left)
            score[i, j] = best
            trace[i, j] = 1 if diagonal == best else (2 if up == best else 3)

    endpoints = [(int(score[n, j]), n, j) for j in range(1, m + 1)]
    endpoints.extend((int(score[i, m]), i, m) for i in range(1, n + 1))
    _best_score, i, j = max(endpoints, key=lambda item: (item[0], item[1] + item[2]))

    pairs: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        direction = int(trace[i, j])
        if direction == 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 2:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    matches = sum(predicted[pi] == native[ni] for pi, ni in pairs)
    return {
        "pairs": pairs,
        "matches": int(matches),
        "identity": matches / len(pairs) if pairs else 0.0,
        "native_coverage": len({ni for _pi, ni in pairs}) / m,
    }


def paired_atoms(
    predicted_residues: Sequence[Mapping[str, object]],
    native_residues: Sequence[Mapping[str, object]],
    residue_pairs: Iterable[Tuple[int, int]],
    atom_names: Sequence[str],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    mobile: List[np.ndarray] = []
    target: List[np.ndarray] = []
    for predicted_index, native_index in residue_pairs:
        if predicted_index >= len(predicted_residues) or native_index >= len(native_residues):
            continue
        predicted_atoms = predicted_residues[predicted_index]["atoms"]
        native_atoms = native_residues[native_index]["atoms"]
        for atom_name in atom_names:
            if atom_name in predicted_atoms and atom_name in native_atoms:
                mobile.append(predicted_atoms[atom_name])
                target.append(native_atoms[atom_name])
    return mobile, target


def index_paired_atoms(
    predicted_residues: Sequence[Mapping[str, object]],
    native_residues: Sequence[Mapping[str, object]],
    atom_names: Sequence[str],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    count = min(len(predicted_residues), len(native_residues))
    return paired_atoms(predicted_residues, native_residues, ((i, i) for i in range(count)), atom_names)


def parse_selected_chains(value: object) -> List[str]:
    return [part for part in re.split(r"[;,|\s]+", str(value or "").strip()) if part]


def choose_predicted_peptide_chain(chains: Mapping[str, Sequence[dict]], design_seq: str) -> dict:
    design_natural = naturalize(design_seq)
    candidates = []
    for chain_id, residues in chains.items():
        if len(residues) != len(design_natural):
            continue
        sequence = chain_sequence(residues)
        matches = sum(a == b for a, b in zip(sequence, design_natural))
        identity = matches / len(design_natural) if design_natural else 0.0
        candidates.append((identity, matches, chain_id, sequence))

    if not candidates:
        return {"chain": "", "status": "no_exact_length_peptide_chain", "identity": 0.0}
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    identity, _matches, chain_id, sequence = candidates[0]
    return {
        "chain": chain_id,
        "status": "unique_exact_length" if len(candidates) == 1 else "best_identity_among_exact_length",
        "identity": identity,
        "parsed_sequence": sequence,
        "n_candidates": len(candidates),
    }


def native_peptide_candidates(
    native_chains: Mapping[str, Sequence[dict]],
    selected_chains: Sequence[str],
    native_seq: str,
    design_len: int,
) -> List[str]:
    """Return only sequence-identical native peptide copies.

    The selected chain from ``all_designs.csv`` anchors the identity class.
    Other short chains are accepted only when their native sequence is exactly
    identical, making the minimum invariant to arbitrary duplicate chain IDs.
    """
    anchor = ""
    for chain_id in selected_chains:
        if chain_id in native_chains and len(native_chains[chain_id]) == design_len:
            anchor = chain_sequence(native_chains[chain_id])
            break
    if not anchor:
        expected = naturalize(native_seq)
        matches = [
            chain_id
            for chain_id, residues in native_chains.items()
            if len(residues) == design_len and chain_sequence(residues) == expected
        ]
        if matches:
            anchor = expected
    if not anchor:
        return []
    return sorted(
        chain_id
        for chain_id, residues in native_chains.items()
        if len(residues) == design_len and chain_sequence(residues) == anchor
    )


def receptor_mapping_candidates(
    predicted_chains: Mapping[str, Sequence[dict]],
    native_chains: Mapping[str, Sequence[dict]],
    predicted_peptide_chain: str,
    native_peptide_chains: Sequence[str],
    design_len: int,
    min_identity: float,
    min_native_coverage: float,
) -> List[dict]:
    predicted_receptors = sorted(
        chain_id
        for chain_id, residues in predicted_chains.items()
        if chain_id != predicted_peptide_chain and len(residues) > design_len + 2
    )
    native_receptors = sorted(
        chain_id
        for chain_id, residues in native_chains.items()
        if chain_id not in set(native_peptide_chains) and len(residues) > design_len + 2
    )
    if not predicted_receptors or not native_receptors:
        return []

    mappings: List[List[Tuple[str, str]]] = []
    if len(predicted_receptors) <= len(native_receptors):
        for native_perm in itertools.permutations(native_receptors, len(predicted_receptors)):
            mappings.append(list(zip(predicted_receptors, native_perm)))
    else:
        for predicted_perm in itertools.permutations(predicted_receptors, len(native_receptors)):
            mappings.append(list(zip(predicted_perm, native_receptors)))

    results = []
    for mapping in mappings:
        mobile: List[np.ndarray] = []
        target: List[np.ndarray] = []
        total_residue_pairs = 0
        total_matches = 0
        details = []
        compatible = True

        for predicted_chain, native_chain in mapping:
            predicted_residues = predicted_chains[predicted_chain]
            native_residues = native_chains[native_chain]
            alignment = semi_global_sequence_pairs(
                chain_sequence(predicted_residues),
                chain_sequence(native_residues),
            )
            if (
                alignment["identity"] < min_identity
                or alignment["native_coverage"] < min_native_coverage
            ):
                compatible = False
                break
            chain_mobile, chain_target = paired_atoms(
                predicted_residues,
                native_residues,
                alignment["pairs"],
                BACKBONE_ATOMS,
            )
            mobile.extend(chain_mobile)
            target.extend(chain_target)
            total_residue_pairs += len(alignment["pairs"])
            total_matches += alignment["matches"]
            details.append(
                f"{predicted_chain}->{native_chain}"
                f"(ident={alignment['identity']:.4f},native_cov={alignment['native_coverage']:.4f},"
                f"res_pairs={len(alignment['pairs'])},bb_pairs={len(chain_mobile)})"
            )

        if not compatible or len(mobile) < 3:
            continue
        rotation, translation = kabsch_fit(np.asarray(mobile), np.asarray(target))
        fit_rmsd = rmsd(apply_transform(np.asarray(mobile), rotation, translation), np.asarray(target))
        results.append(
            {
                "mapping": mapping,
                "mapping_text": ";".join(f"{a}->{b}" for a, b in mapping),
                "detail": ";".join(details),
                "rotation": rotation,
                "translation": translation,
                "receptor_mobile": mobile,
                "receptor_target": target,
                "n_residue_pairs": total_residue_pairs,
                "n_backbone_atom_pairs": len(mobile),
                "sequence_matches": total_matches,
                "receptor_fit_rmsd": fit_rmsd,
            }
        )
    return results


def evaluate_pdb(
    pdb_path: Path,
    design_rows: Sequence[dict],
    native_by_target: Mapping[str, Mapping[str, Sequence[dict]]],
    threshold: float,
    min_receptor_residue_pairs: int,
    min_receptor_identity: float,
    min_receptor_native_coverage: float,
    symmetry_receptor_fit_tolerance: float,
) -> dict:
    filename = parse_pdb_filename(pdb_path)
    temperature, temperature_folder = parse_temperature(pdb_path)
    target_name = filename["target_name"]
    design_seq = filename["design_seq"]
    base = {
        "pdb_path": str(pdb_path),
        "pdb_file": pdb_path.name,
        "temperature_folder": temperature_folder,
        "temperature": temperature,
        "target_name": target_name,
        "file_index": filename["file_index"],
        "design_seq": design_seq,
        "design_natural_seq": naturalize(design_seq),
        "filename_parse_ok": filename["filename_parse_ok"],
        "matched_all_design_rows": len(design_rows),
        "all_design_row_indices": ";".join(str(row.get("_row_index", "")) for row in design_rows),
        "rmsd_threshold_angstrom": threshold,
        "fit_atom_names": ";".join(BACKBONE_ATOMS),
        "peptide_atom_names": ";".join(BACKBONE_ATOMS),
        "peptide_refit_performed": 0,
        "receptor_outlier_rejection_performed": 0,
    }

    if not filename["filename_parse_ok"]:
        base["rmsd_status"] = "filename_parse_failed"
        return base
    if not temperature:
        base["rmsd_status"] = "temperature_parse_failed"
        return base
    if not design_rows:
        base["rmsd_status"] = "no_exact_all_design_match"
        return base

    metadata = design_rows[0]
    native_seq = metadata.get("native_seq", "")
    selected_chains = parse_selected_chains(metadata.get("selected_chains", ""))
    design_len = len(naturalize(design_seq))
    base.update(
        {
            "native_seq": native_seq,
            "selected_native_chain_anchor": ";".join(selected_chains),
            "design_len": design_len,
            "raw_design_record_indices": ";".join(row.get("record_index", "") for row in design_rows),
        }
    )

    native_chains = native_by_target.get(target_name)
    if not native_chains:
        base["rmsd_status"] = "native_target_not_found"
        return base

    try:
        predicted_chains, ca_bfactor_mean = parse_pdb_structure(pdb_path)
    except Exception as error:  # preserve a per-file audit row rather than stopping 4,000 files
        base["rmsd_status"] = f"pdb_parse_failed:{type(error).__name__}"
        return base
    base["pdb_ca_bfactor_mean"] = fmt(ca_bfactor_mean, 4)
    base["predicted_chain_lengths"] = ";".join(
        f"{chain}:{len(residues)}" for chain, residues in sorted(predicted_chains.items())
    )

    predicted_peptide = choose_predicted_peptide_chain(predicted_chains, design_seq)
    base["predicted_peptide_chain"] = predicted_peptide.get("chain", "")
    base["predicted_peptide_chain_method"] = predicted_peptide.get("status", "")
    base["predicted_peptide_identity_to_design"] = fmt(predicted_peptide.get("identity"), 4)
    if not predicted_peptide.get("chain"):
        base["rmsd_status"] = predicted_peptide["status"]
        return base

    equivalent_native_peptides = native_peptide_candidates(
        native_chains,
        selected_chains,
        native_seq,
        design_len,
    )
    base["equivalent_native_peptide_chains"] = ";".join(equivalent_native_peptides)
    base["n_equivalent_native_peptide_chains"] = len(equivalent_native_peptides)
    if not equivalent_native_peptides:
        base["rmsd_status"] = "native_peptide_chain_not_resolved"
        return base

    mapping_results = receptor_mapping_candidates(
        predicted_chains=predicted_chains,
        native_chains=native_chains,
        predicted_peptide_chain=predicted_peptide["chain"],
        native_peptide_chains=equivalent_native_peptides,
        design_len=design_len,
        min_identity=min_receptor_identity,
        min_native_coverage=min_receptor_native_coverage,
    )
    mapping_results = [
        result
        for result in mapping_results
        if result["n_residue_pairs"] >= min_receptor_residue_pairs
    ]
    base["n_valid_receptor_chain_mappings"] = len(mapping_results)
    if not mapping_results:
        base["rmsd_status"] = "no_sequence_compatible_receptor_mapping"
        return base

    # Keep only the best sequence-score class.  Receptor-chain permutations in
    # that class are chain-label equivalent; peptide RMSD may resolve which
    # native copy corresponds to the predicted binding site.
    best_sequence_score = max(
        (result["sequence_matches"], result["n_residue_pairs"])
        for result in mapping_results
    )
    mapping_results = [
        result
        for result in mapping_results
        if (result["sequence_matches"], result["n_residue_pairs"]) == best_sequence_score
    ]
    base["n_sequence_equivalent_receptor_mappings"] = len(mapping_results)

    best_receptor_fit_rmsd = min(result["receptor_fit_rmsd"] for result in mapping_results)
    mapping_results = [
        result
        for result in mapping_results
        if result["receptor_fit_rmsd"]
        <= best_receptor_fit_rmsd + symmetry_receptor_fit_tolerance
    ]
    base["symmetry_receptor_fit_tolerance_angstrom"] = symmetry_receptor_fit_tolerance
    base["n_receptor_fit_equivalent_mappings"] = len(mapping_results)

    predicted_peptide_residues = predicted_chains[predicted_peptide["chain"]]
    comparisons = []
    for mapping in mapping_results:
        for native_peptide_chain in equivalent_native_peptides:
            native_peptide_residues = native_chains[native_peptide_chain]
            peptide_mobile, peptide_target = index_paired_atoms(
                predicted_peptide_residues,
                native_peptide_residues,
                BACKBONE_ATOMS,
            )
            peptide_ca_mobile, peptide_ca_target = index_paired_atoms(
                predicted_peptide_residues,
                native_peptide_residues,
                ("CA",),
            )
            if not peptide_mobile or not peptide_ca_mobile:
                continue

            transformed_backbone = apply_transform(
                np.asarray(peptide_mobile), mapping["rotation"], mapping["translation"]
            )
            transformed_ca = apply_transform(
                np.asarray(peptide_ca_mobile), mapping["rotation"], mapping["translation"]
            )
            comparisons.append(
                {
                    "mapping": mapping,
                    "native_peptide_chain": native_peptide_chain,
                    "n_peptide_backbone_pairs": len(peptide_mobile),
                    "n_peptide_ca_pairs": len(peptide_ca_mobile),
                    "peptide_backbone_rmsd": rmsd(transformed_backbone, np.asarray(peptide_target)),
                    "peptide_ca_rmsd": rmsd(transformed_ca, np.asarray(peptide_ca_target)),
                }
            )

    base["n_chain_label_equivalent_comparisons"] = len(comparisons)
    if not comparisons:
        base["rmsd_status"] = "no_complete_peptide_atom_comparison"
        return base

    base["chain_label_equivalent_comparison_summary"] = "|".join(
        (
            f"map={item['mapping']['mapping_text']},native_pep={item['native_peptide_chain']},"
            f"receptor_bb={item['mapping']['receptor_fit_rmsd']:.6f},"
            f"peptide_bb={item['peptide_backbone_rmsd']:.6f},"
            f"peptide_ca={item['peptide_ca_rmsd']:.6f}"
        )
        for item in sorted(
            comparisons,
            key=lambda comparison: (
                comparison["mapping"]["mapping_text"],
                comparison["native_peptide_chain"],
            ),
        )
    )

    # Primary selection is explicitly the fixed-receptor peptide BACKBONE RMSD.
    best = min(
        comparisons,
        key=lambda item: (
            item["peptide_backbone_rmsd"],
            item["peptide_ca_rmsd"],
            item["mapping"]["receptor_fit_rmsd"],
            item["native_peptide_chain"],
            item["mapping"]["mapping_text"],
        ),
    )
    mapping = best["mapping"]
    expected_peptide_backbone_pairs = design_len * len(BACKBONE_ATOMS)
    base.update(
        {
            "native_peptide_chain_used": best["native_peptide_chain"],
            "receptor_chain_mapping_used": mapping["mapping_text"],
            "receptor_alignment_detail": mapping["detail"],
            "n_receptor_residue_pairs": mapping["n_residue_pairs"],
            "n_receptor_backbone_atom_pairs": mapping["n_backbone_atom_pairs"],
            "receptor_backbone_fit_rmsd": fmt(mapping["receptor_fit_rmsd"]),
            "n_peptide_backbone_atom_pairs": best["n_peptide_backbone_pairs"],
            "expected_peptide_backbone_atom_pairs": expected_peptide_backbone_pairs,
            "peptide_backbone_atom_pair_completeness": fmt(
                best["n_peptide_backbone_pairs"] / expected_peptide_backbone_pairs
                if expected_peptide_backbone_pairs
                else None,
                4,
            ),
            "n_peptide_ca_pairs": best["n_peptide_ca_pairs"],
            "peptide_backbone_rmsd_after_receptor_backbone_fit": fmt(
                best["peptide_backbone_rmsd"]
            ),
            "peptide_ca_rmsd_after_receptor_backbone_fit": fmt(best["peptide_ca_rmsd"]),
            "passes_peptide_backbone_rmsd_lt_threshold": int(
                best["peptide_backbone_rmsd"] < threshold
            ),
            "rmsd_status": "ok",
        }
    )
    return base


def unique_design_index(rows: Sequence[dict]) -> Dict[Tuple[str, str, str], List[dict]]:
    result: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        enriched = dict(row)
        enriched["_row_index"] = index
        key = (
            str(row.get("target_name", "")).upper(),
            norm_temp(row.get("temperature")),
            str(row.get("design_seq", "")),
        )
        result[key].append(enriched)
    return dict(result)


def design_key_from_pdb(path: Path) -> Tuple[str, str, str]:
    parsed = parse_pdb_filename(path)
    temperature, _folder = parse_temperature(path)
    return parsed["target_name"], temperature, parsed["design_seq"]


def choose_design_representatives(
    design_index: Mapping[Tuple[str, str, str], Sequence[dict]],
    pdb_rows: Sequence[dict],
    threshold: float,
) -> List[dict]:
    by_key: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in pdb_rows:
        key = (row.get("target_name", ""), row.get("temperature", ""), row.get("design_seq", ""))
        if key in design_index:
            by_key[key].append(row)

    output = []
    for key, raw_rows in sorted(design_index.items(), key=lambda item: item[0]):
        metadata = raw_rows[0]
        pdb_candidates = by_key.get(key, [])
        valid_candidates = [row for row in pdb_candidates if row.get("rmsd_status") == "ok"]
        pool = valid_candidates if valid_candidates else pdb_candidates

        representative = None
        if pool:
            representative = sorted(
                pool,
                key=lambda row: (
                    -(safe_float(row.get("pdb_ca_bfactor_mean")) or -math.inf),
                    str(row.get("pdb_file", "")),
                ),
            )[0]

        result = {
            "target_name": key[0],
            "temperature": key[1],
            "design_seq": key[2],
            "design_natural_seq": naturalize(key[2]),
            "native_seq": metadata.get("native_seq", ""),
            "selected_native_chain_anchor": metadata.get("selected_chains", ""),
            "n_raw_design_rows_for_key": len(raw_rows),
            "raw_all_design_row_indices": ";".join(str(row["_row_index"]) for row in raw_rows),
            "n_pdb_for_key": len(pdb_candidates),
            "n_rmsd_ok_pdb_for_key": len(valid_candidates),
            "representative_rule": (
                "highest_mean_ca_bfactor_among_rmsd_ok_pdbs_else_all_pdbs"
            ),
            "rmsd_threshold_angstrom": threshold,
        }
        if representative is None:
            result["rmsd_status"] = "missing_pdb"
            result["passes_peptide_backbone_rmsd_lt_threshold"] = ""
        else:
            for field in (
                "pdb_file",
                "pdb_path",
                "pdb_ca_bfactor_mean",
                "rmsd_status",
                "predicted_peptide_chain",
                "native_peptide_chain_used",
                "equivalent_native_peptide_chains",
                "n_sequence_equivalent_receptor_mappings",
                "n_receptor_fit_equivalent_mappings",
                "n_chain_label_equivalent_comparisons",
                "chain_label_equivalent_comparison_summary",
                "receptor_chain_mapping_used",
                "n_receptor_residue_pairs",
                "n_receptor_backbone_atom_pairs",
                "receptor_backbone_fit_rmsd",
                "n_peptide_backbone_atom_pairs",
                "expected_peptide_backbone_atom_pairs",
                "peptide_backbone_atom_pair_completeness",
                "n_peptide_ca_pairs",
                "peptide_backbone_rmsd_after_receptor_backbone_fit",
                "peptide_ca_rmsd_after_receptor_backbone_fit",
                "passes_peptide_backbone_rmsd_lt_threshold",
                "peptide_refit_performed",
                "receptor_outlier_rejection_performed",
            ):
                result[field] = representative.get(field, "")
        output.append(result)
    return output


def summarize_designs(rows: Sequence[dict], keys: Sequence[str]) -> List[dict]:
    grouped: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)

    output = []
    for group_values, items in sorted(grouped.items()):
        valid = [row for row in items if row.get("rmsd_status") == "ok"]
        passed = [
            row
            for row in valid
            if str(row.get("passes_peptide_backbone_rmsd_lt_threshold", "")) == "1"
        ]
        values = [
            safe_float(row.get("peptide_backbone_rmsd_after_receptor_backbone_fit"))
            for row in valid
        ]
        values = [value for value in values if value is not None]
        result = {key: value for key, value in zip(keys, group_values)}
        result.update(
            {
                "n_unique_designs": len(items),
                "n_with_pdb": sum(int(row.get("n_pdb_for_key", 0)) > 0 for row in items),
                "n_rmsd_ok": len(valid),
                "n_rmsd_failed_or_missing": len(items) - len(valid),
                "n_rmsd_lt_threshold": len(passed),
                "fraction_lt_threshold_among_rmsd_ok": fmt(
                    len(passed) / len(valid) if valid else None,
                    8,
                ),
                "mean_peptide_backbone_rmsd": fmt(mean(values) if values else None),
                "median_peptide_backbone_rmsd": fmt(median(values) if values else None),
                "min_peptide_backbone_rmsd": fmt(min(values) if values else None),
                "max_peptide_backbone_rmsd": fmt(max(values) if values else None),
            }
        )
        output.append(result)
    return output


def write_report(
    path: Path,
    design_rows: Sequence[dict],
    design_index: Mapping[Tuple[str, str, str], Sequence[dict]],
    pdb_files: Sequence[Path],
    pdb_rows: Sequence[dict],
    representative_rows: Sequence[dict],
    threshold: float,
    self_test_rmsd: float,
) -> None:
    matched_pdb = [row for row in pdb_rows if int(row.get("matched_all_design_rows", 0)) > 0]
    valid_pdb = [row for row in pdb_rows if row.get("rmsd_status") == "ok"]
    valid_designs = [row for row in representative_rows if row.get("rmsd_status") == "ok"]
    passed_designs = [
        row
        for row in valid_designs
        if str(row.get("passes_peptide_backbone_rmsd_lt_threshold", "")) == "1"
    ]
    with_pdb = [row for row in representative_rows if int(row.get("n_pdb_for_key", 0)) > 0]
    status_counts = Counter(str(row.get("rmsd_status", "")) for row in pdb_rows)

    lines = [
        "===== ALL-DESIGN RECEPTOR-BACKBONE RMSD AUDIT =====",
        "",
        "Primary metric:",
        "  peptide N/CA/C/O RMSD after a receptor-only N/CA/C/O rigid fit",
        "  peptide self-superposition: NEVER performed",
        "  receptor atom-pair rejection: NEVER performed (PyMOL cycles=0 analogue)",
        "  symmetry handling: minimum only across sequence-identical native peptide copies",
        "                     and sequence- plus receptor-fit-equivalent chain mappings",
        f"  strict threshold: RMSD < {threshold:.3f} Angstrom",
        "",
        f"Kabsch self-test RMSD: {self_test_rmsd:.12e}",
        "",
        f"raw design rows: {len(design_rows)}",
        f"unique exact design keys: {len(design_index)}",
        f"PDB files discovered: {len(pdb_files)}",
        f"PDB files exactly matched to all_designs: {len(matched_pdb)}",
        f"PDB-level RMSD OK: {len(valid_pdb)}",
        f"unique designs with >=1 PDB: {len(with_pdb)}",
        f"unique designs without PDB: {len(representative_rows) - len(with_pdb)}",
        f"unique-design representative RMSD OK: {len(valid_designs)}",
        f"unique-design RMSD < {threshold:.3f}: {len(passed_designs)}",
        (
            f"fraction < {threshold:.3f} among RMSD-OK unique designs: "
            f"{len(passed_designs) / len(valid_designs):.8f}"
            if valid_designs
            else f"fraction < {threshold:.3f} among RMSD-OK unique designs: NA"
        ),
        (
            f"coverage-normalized lower bound among all unique designs: "
            f"{len(passed_designs) / len(representative_rows):.8f}"
            if representative_rows
            else "coverage-normalized lower bound among all unique designs: NA"
        ),
        "",
        "PDB-level RMSD status counts:",
    ]
    lines.extend(f"  {status}: {count}" for status, count in sorted(status_counts.items()))
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "  Missing/failed structures are excluded from the evaluated success fraction.",
            "  The all-design denominator value is only a coverage-normalized lower bound.",
            "  RMSD is a structural-retention filter; it is not a permeability measurement.",
            "  Permeability and other metrics must be computed only after this RMSD gate.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute all-design peptide RMSD after receptor-backbone alignment."
    )
    parser.add_argument(
        "--pdb_root",
        default="raw_external/pdb_highfold_temperature",
        help="Root containing the five HighFold temperature directories.",
    )
    parser.add_argument(
        "--all_designs",
        default="paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv",
    )
    parser.add_argument("--native_jsonl", default="17_complexes_native.jsonl")
    parser.add_argument(
        "--out_dir",
        default="paper_clean_v28_outputs/structure_metrics/rmsd_recheck_all_designs",
    )
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--min_receptor_residue_pairs", type=int, default=20)
    parser.add_argument("--min_receptor_identity", type=float, default=0.80)
    parser.add_argument("--min_receptor_native_coverage", type=float, default=0.80)
    parser.add_argument(
        "--symmetry_receptor_fit_tolerance",
        type=float,
        default=0.25,
        help=(
            "Only sequence-equivalent receptor mappings within this many Angstrom of the "
            "best receptor-only fit may enter the symmetry-aware peptide comparison."
        ),
    )
    args = parser.parse_args()

    pdb_root = Path(args.pdb_root)
    all_designs_path = Path(args.all_designs)
    native_path = Path(args.native_jsonl)
    out_dir = Path(args.out_dir)

    for required in (pdb_root, all_designs_path, native_path):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive")
    if args.symmetry_receptor_fit_tolerance < 0:
        raise ValueError("--symmetry_receptor_fit_tolerance must be non-negative")

    self_test_rmsd = kabsch_self_test()
    if self_test_rmsd > 1e-8:
        raise RuntimeError(f"Kabsch self-test failed: RMSD={self_test_rmsd}")

    design_rows = read_csv(all_designs_path)
    design_index = unique_design_index(design_rows)
    native_by_target = load_native_structures(native_path)
    pdb_files = sorted(pdb_root.rglob("*.pdb"))

    pdb_rows = []
    for index, pdb_path in enumerate(pdb_files, start=1):
        key = design_key_from_pdb(pdb_path)
        row = evaluate_pdb(
            pdb_path=pdb_path,
            design_rows=design_index.get(key, []),
            native_by_target=native_by_target,
            threshold=args.threshold,
            min_receptor_residue_pairs=args.min_receptor_residue_pairs,
            min_receptor_identity=args.min_receptor_identity,
            min_receptor_native_coverage=args.min_receptor_native_coverage,
            symmetry_receptor_fit_tolerance=args.symmetry_receptor_fit_tolerance,
        )
        pdb_rows.append(row)
        if index % 250 == 0 or index == len(pdb_files):
            print(f"Processed PDBs: {index}/{len(pdb_files)}")

    representative_rows = choose_design_representatives(design_index, pdb_rows, args.threshold)
    selected_rows = [
        row
        for row in representative_rows
        if row.get("rmsd_status") == "ok"
        and str(row.get("passes_peptide_backbone_rmsd_lt_threshold", "")) == "1"
    ]
    problem_rows = [row for row in pdb_rows if row.get("rmsd_status") != "ok"]

    out_dir.mkdir(parents=True, exist_ok=True)
    by_pdb_path = out_dir / "all_design_receptor_backbone_rmsd_by_pdb.csv"
    by_design_path = out_dir / "all_design_receptor_backbone_rmsd_by_design.csv"
    selected_path = out_dir / "all_design_receptor_backbone_rmsd_lt3.csv"
    problems_path = out_dir / "all_design_receptor_backbone_rmsd_problem_rows.csv"
    summary_target_temp_path = out_dir / "all_design_receptor_backbone_rmsd_summary_by_target_temperature.csv"
    report_path = out_dir / "all_design_receptor_backbone_rmsd_report.txt"

    write_csv(by_pdb_path, pdb_rows)
    write_csv(by_design_path, representative_rows)
    write_csv(
        selected_path,
        selected_rows,
        fieldnames=list(representative_rows[0]) if representative_rows else None,
    )
    write_csv(
        problems_path,
        problem_rows,
        fieldnames=list(pdb_rows[0]) if pdb_rows else None,
    )
    write_csv(summary_target_temp_path, summarize_designs(representative_rows, ("target_name", "temperature")))
    write_report(
        report_path,
        design_rows,
        design_index,
        pdb_files,
        pdb_rows,
        representative_rows,
        args.threshold,
        self_test_rmsd,
    )

    valid_designs = [row for row in representative_rows if row.get("rmsd_status") == "ok"]
    print("完成：全量设计 receptor-backbone RMSD 审计")
    print("PDB files:", len(pdb_files))
    print("unique design keys:", len(design_index))
    print("unique design RMSD OK:", len(valid_designs))
    print(f"unique design RMSD < {args.threshold:g}:", len(selected_rows))
    if valid_designs:
        print("evaluated pass fraction:", f"{len(selected_rows) / len(valid_designs):.8f}")
    print("输出目录:", out_dir)


if __name__ == "__main__":
    main()
