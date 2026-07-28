#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recompute structure, confidence, TM, and permeability metrics for monomers.

The external HighFold directory contains four named variants per sample:

1. reference sequence with explicit methylation annotation/residue types;
2. naturalized reference sequence;
3. end-to-end design with explicit methylation annotation/residue types;
4. naturalized end-to-end design.

Variants 2 and 4 are the complete, primary 151-sample panel.  Variants 1 and 3
are retained as an explicit-methylation sensitivity subset when both exist.

The primary structural comparison is a single-chain CA self-superposition.  All
forward cyclic register shifts are tested, reverse order is never allowed, and
the shift with minimum complete-chain CA RMSD is retained.  Backbone and
methylation-position deviations use the same CA-derived transform and shift.

This comparison is between two HighFold predictions (reference sequence versus
end-to-end design).  It is a conformational-change/designability analysis, not
an experimental native-structure validation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tmtools import tm_align
except Exception:  # pragma: no cover - dependency is checked in main
    tm_align = None


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

DEFAULT_DESIGN_MANIFEST = (
    ROOT / "paper_clean_v28_outputs" / "monomer_design_structure_manifest.csv"
)
DEFAULT_REFERENCE_MANIFEST = (
    ROOT / "paper_clean_v28_outputs" / "monomer_structure_manifest.csv"
)
DEFAULT_PDB_DIR = (
    ROOT
    / "raw_external"
    / "pdb_permeability_v20260624"
    / "pdb_monomer"
    / "pdb_monomer_hf4"
)
DEFAULT_PERMEABILITY_ROOT = (
    ROOT / "raw_external" / "pdb_permeability_v20260624"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "paper_clean_v28_outputs"
    / "temperature_0.5_best17"
    / "monomer"
)

EXPECTED_SAMPLES = 151
EXPECTED_PDBS = 560
PRIMARY_VARIANTS = (2, 4)
VARIANT_ROLE = {
    1: "reference_explicit_methyl",
    2: "reference_naturalized",
    3: "e2e_explicit_methyl",
    4: "e2e_naturalized",
}
BACKBONE_ATOMS = ("N", "CA", "C")
PDB_NAME_RE = re.compile(
    r"^(?P<sample>.+)_(?P<variant>[1-4])_(?P<sequence>[^_]+)_model\.pdb$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the complete 151-sample monomer structure panel."
    )
    parser.add_argument("--design_manifest", default=str(DEFAULT_DESIGN_MANIFEST))
    parser.add_argument("--reference_manifest", default=str(DEFAULT_REFERENCE_MANIFEST))
    parser.add_argument("--pdb_dir", default=str(DEFAULT_PDB_DIR))
    parser.add_argument("--permeability_root", default=str(DEFAULT_PERMEABILITY_ROOT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def fmt(value: object, digits: int = 6):
    if not finite(value):
        return np.nan
    return round(float(value), digits)


def naturalize_sequence(value: object) -> str:
    return re.sub(r"[^A-Za-z]", "", str(value or "")).upper()


def rms_from_distances(values: Iterable[float]) -> float:
    array = np.asarray([x for x in values if finite(x)], dtype=float)
    if array.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(array * array)))


def kabsch_fit(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return R, t for row-vector coordinates: mobile @ R + t ~= target."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(
            f"Kabsch coordinate shape mismatch: {mobile.shape} vs {target.shape}"
        )
    if len(mobile) < 3:
        raise ValueError("Kabsch requires at least three coordinate pairs")

    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_zero = mobile - mobile_center
    target_zero = target - target_center
    covariance = mobile_zero.T @ target_zero
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def apply_transform(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
    return np.asarray(coords, dtype=float) @ rotation + translation


def normalize_comment_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def first_numeric_comment(
    comments: Mapping[str, object], candidates: Sequence[str]
) -> float:
    normalized = {
        normalize_comment_key(key): value for key, value in comments.items()
    }
    for candidate in candidates:
        key = normalize_comment_key(candidate)
        if key in normalized and finite(normalized[key]):
            return float(normalized[key])
    for key, value in normalized.items():
        if any(normalize_comment_key(candidate) in key for candidate in candidates):
            if finite(value):
                return float(value)
    return math.nan


def parse_pdb(
    path: Path,
    expected_sequence: str,
) -> Dict[str, object]:
    """Read a single matching chain and retain N/CA/C/O coordinates."""
    comments: Dict[str, object] = {}
    chains: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    residue_index: Dict[Tuple[str, str, str], int] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("COMMENT"):
                match = re.match(r"^COMMENT\s+([^:]+):\s*(.+?)\s*$", line)
                if match:
                    key, raw = match.group(1).strip(), match.group(2).strip()
                    comments[key] = float(raw) if finite(raw) else raw
                continue
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 66:
                continue

            altloc = line[16].strip()
            if altloc not in {"", "A", "1"}:
                continue
            atom = line[12:16].strip()
            if atom not in {"N", "CA", "C", "O"}:
                continue
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            try:
                coord = np.asarray(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                )
                bfactor = float(line[60:66])
            except Exception:
                continue

            if key not in residue_index:
                residue_index[key] = len(chains[chain])
                chains[chain].append(
                    {
                        "resname": line[17:20].strip().upper(),
                        "resseq": resseq,
                        "icode": icode,
                        "atoms": {},
                        "bfactors": {},
                    }
                )
            residue = chains[chain][residue_index[key]]
            residue["atoms"].setdefault(atom, coord)
            residue["bfactors"].setdefault(atom, bfactor)

    expected_length = len(expected_sequence)
    candidates = [
        chain for chain, residues in chains.items() if len(residues) == expected_length
    ]
    if not candidates:
        counts = {chain: len(residues) for chain, residues in chains.items()}
        raise ValueError(
            f"no PDB chain has expected length {expected_length}; observed={counts}"
        )
    chain = "A" if "A" in candidates else sorted(candidates)[0]
    residues = chains[chain]

    missing_ca = [i + 1 for i, residue in enumerate(residues) if "CA" not in residue["atoms"]]
    missing_backbone = [
        i + 1
        for i, residue in enumerate(residues)
        if not all(atom in residue["atoms"] for atom in BACKBONE_ATOMS)
    ]
    if missing_ca:
        raise ValueError(f"missing CA atoms at positions {missing_ca}")
    if missing_backbone:
        raise ValueError(f"missing N/CA/C atoms at positions {missing_backbone}")

    ca_bfactors = [
        float(residue["bfactors"]["CA"])
        for residue in residues
        if finite(residue["bfactors"].get("CA"))
    ]
    return {
        "path": str(path.resolve()),
        "file": path.name,
        "chain": chain,
        "residues": residues,
        "length": len(residues),
        "ca_bfactor_mean": float(mean(ca_bfactors)) if ca_bfactors else math.nan,
        "ca_bfactor_min": min(ca_bfactors) if ca_bfactors else math.nan,
        "ca_bfactor_max": max(ca_bfactors) if ca_bfactors else math.nan,
        "comment_plddt": first_numeric_comment(
            comments, ["plddt", "mean plddt", "global plddt"]
        ),
        "comment_ptm": first_numeric_comment(comments, ["ptm", "pTM"]),
        "comment_iptm": first_numeric_comment(comments, ["iptm", "ipTM"]),
        "comment_inter_pae": first_numeric_comment(
            comments, ["inter pae", "inter_pae", "interpae"]
        ),
        "comments_json": json.dumps(comments, sort_keys=True, ensure_ascii=False),
    }


def coordinate_array(residues: Sequence[Mapping[str, object]], atom: str) -> np.ndarray:
    return np.asarray([residue["atoms"][atom] for residue in residues], dtype=float)


def cyclic_structure_metrics(
    reference: Mapping[str, object],
    design: Mapping[str, object],
    design_sequence: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Best complete-chain forward-cyclic CA superposition and residue audit."""
    reference_residues = list(reference["residues"])
    design_residues = list(design["residues"])
    length = len(design_sequence)
    if len(reference_residues) != length or len(design_residues) != length:
        raise ValueError(
            "sequence/structure length mismatch: "
            f"sequence={length}, reference={len(reference_residues)}, "
            f"design={len(design_residues)}"
        )

    design_ca = coordinate_array(design_residues, "CA")
    shift_results: List[Dict[str, object]] = []
    for shift in range(length):
        target_order = [(index + shift) % length for index in range(length)]
        reference_ca = np.asarray(
            [reference_residues[index]["atoms"]["CA"] for index in target_order],
            dtype=float,
        )
        rotation, translation = kabsch_fit(design_ca, reference_ca)
        aligned_ca = apply_transform(design_ca, rotation, translation)
        ca_distances = np.linalg.norm(aligned_ca - reference_ca, axis=1)

        backbone_distances: List[float] = []
        for design_index, reference_index in enumerate(target_order):
            for atom in BACKBONE_ATOMS:
                mobile = design_residues[design_index]["atoms"][atom]
                target = reference_residues[reference_index]["atoms"][atom]
                aligned = apply_transform(
                    np.asarray([mobile]), rotation, translation
                )[0]
                backbone_distances.append(float(np.linalg.norm(aligned - target)))

        shift_results.append(
            {
                "shift": shift,
                "ca_rmsd": rms_from_distances(ca_distances),
                "backbone_rmsd_after_ca_fit": rms_from_distances(
                    backbone_distances
                ),
                "rotation": rotation,
                "translation": translation,
                "target_order": target_order,
                "ca_distances": ca_distances,
            }
        )

    best = min(shift_results, key=lambda item: (item["ca_rmsd"], item["shift"]))
    fixed = shift_results[0]
    rotation = best["rotation"]
    translation = best["translation"]
    target_order = best["target_order"]

    residue_rows: List[Dict[str, object]] = []
    methyl_ca: List[float] = []
    nonmethyl_ca: List[float] = []
    methyl_backbone: List[float] = []
    nonmethyl_backbone: List[float] = []

    for design_index, reference_index in enumerate(target_order):
        is_methylated = int(design_sequence[design_index].islower())
        design_residue = design_residues[design_index]
        reference_residue = reference_residues[reference_index]

        mobile_ca = design_residue["atoms"]["CA"]
        target_ca = reference_residue["atoms"]["CA"]
        aligned_ca = apply_transform(
            np.asarray([mobile_ca]), rotation, translation
        )[0]
        ca_distance = float(np.linalg.norm(aligned_ca - target_ca))
        backbone_distances: List[float] = []
        for atom in BACKBONE_ATOMS:
            mobile = design_residue["atoms"][atom]
            target = reference_residue["atoms"][atom]
            aligned = apply_transform(np.asarray([mobile]), rotation, translation)[0]
            backbone_distances.append(float(np.linalg.norm(aligned - target)))

        residue_rows.append(
            {
                "design_position_1based": design_index + 1,
                "reference_position_1based": reference_index + 1,
                "design_token": design_sequence[design_index],
                "is_e2e_methylated": is_methylated,
                "ca_distance_after_best_cyclic_ca_fit": fmt(ca_distance),
                "backbone_residue_rmsd_after_best_cyclic_ca_fit": fmt(
                    rms_from_distances(backbone_distances)
                ),
            }
        )
        if is_methylated:
            methyl_ca.append(ca_distance)
            methyl_backbone.extend(backbone_distances)
        else:
            nonmethyl_ca.append(ca_distance)
            nonmethyl_backbone.extend(backbone_distances)

    metrics = {
        "ca_rmsd_fixed_order": fmt(fixed["ca_rmsd"]),
        "ca_rmsd_best_forward_cyclic_shift": fmt(best["ca_rmsd"]),
        "best_forward_cyclic_shift": int(best["shift"]),
        "ca_rmsd_improvement_from_forward_shift": fmt(
            fixed["ca_rmsd"] - best["ca_rmsd"]
        ),
        "backbone_rmsd_after_ca_fit_best_forward_cyclic_shift": fmt(
            best["backbone_rmsd_after_ca_fit"]
        ),
        "n_forward_cyclic_shifts_tested": length,
        "forward_cyclic_shift_ca_rmsds": ";".join(
            f"{item['shift']}:{item['ca_rmsd']:.6f}" for item in shift_results
        ),
        "reverse_order_allowed": 0,
        "n_e2e_methyl_positions": sum(
            1 for token in design_sequence if token.islower()
        ),
        "n_e2e_nonmethyl_positions": sum(
            1 for token in design_sequence if not token.islower()
        ),
        "e2e_methyl_ca_rmsd_after_best_cyclic_ca_fit": fmt(
            rms_from_distances(methyl_ca)
        ),
        "e2e_nonmethyl_ca_rmsd_after_best_cyclic_ca_fit": fmt(
            rms_from_distances(nonmethyl_ca)
        ),
        "e2e_methyl_backbone_rmsd_after_best_cyclic_ca_fit": fmt(
            rms_from_distances(methyl_backbone)
        ),
        "e2e_nonmethyl_backbone_rmsd_after_best_cyclic_ca_fit": fmt(
            rms_from_distances(nonmethyl_backbone)
        ),
    }
    return metrics, residue_rows


def cyclic_tm_metrics(
    reference: Mapping[str, object],
    design: Mapping[str, object],
    reference_sequence: str,
    design_sequence: str,
) -> Dict[str, object]:
    if tm_align is None:
        raise RuntimeError(
            "tmtools is required for monomer TM-score. "
            "Use the same wain environment that ran the earlier complex "
            "structural-diversity workflow (tmtools==0.3.0)."
        )

    reference_ca_all = coordinate_array(reference["residues"], "CA")
    design_ca = coordinate_array(design["residues"], "CA")
    length = len(design_sequence)
    results: List[Dict[str, float]] = []
    for shift in range(length):
        order = [(index + shift) % length for index in range(length)]
        reference_ca = reference_ca_all[order]
        shifted_reference_sequence = "".join(reference_sequence[index] for index in order)
        aligned = tm_align(
            design_ca,
            reference_ca,
            naturalize_sequence(design_sequence),
            shifted_reference_sequence,
        )
        tm1 = float(aligned.tm_norm_chain1)
        tm2 = float(aligned.tm_norm_chain2)
        symmetric = (tm1 + tm2) / 2.0
        results.append(
            {
                "shift": shift,
                "tm_norm_design": tm1,
                "tm_norm_reference": tm2,
                "tm_symmetric": symmetric,
                "tmalign_rmsd": float(aligned.rmsd),
            }
        )
    best = max(results, key=lambda item: (item["tm_symmetric"], -item["shift"]))
    fixed = results[0]
    return {
        "tm_score_symmetric_fixed_order": fmt(fixed["tm_symmetric"]),
        "tm_score_symmetric_best_forward_cyclic_shift": fmt(
            best["tm_symmetric"]
        ),
        "tm_best_forward_cyclic_shift": int(best["shift"]),
        "diversity_1_minus_tm_best_forward_cyclic_shift": fmt(
            1.0 - best["tm_symmetric"]
        ),
        "tmalign_rmsd_best_forward_cyclic_shift": fmt(best["tmalign_rmsd"]),
        "tm_score_norm_e2e": fmt(best["tm_norm_design"]),
        "tm_score_norm_reference": fmt(best["tm_norm_reference"]),
        "tm_forward_shift_scores": ";".join(
            f"{item['shift']}:{item['tm_symmetric']:.6f}" for item in results
        ),
        "tm_status": "ok",
    }


def parse_inventory(pdb_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Dict]]]:
    inventory_rows: List[Dict[str, object]] = []
    grouped: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for path in sorted(pdb_dir.glob("*.pdb"), key=lambda item: item.name.lower()):
        match = PDB_NAME_RE.match(path.name)
        row: Dict[str, object] = {
            "pdb_file": path.name,
            "pdb_path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "filename_parse_status": "failed",
        }
        if match:
            sample = match.group("sample")
            variant = int(match.group("variant"))
            sequence = match.group("sequence")
            row.update(
                {
                    "sample_name": sample,
                    "variant": variant,
                    "variant_role": VARIANT_ROLE[variant],
                    "sequence_from_filename": sequence,
                    "filename_parse_status": "ok",
                }
            )
            if variant in grouped[sample]:
                raise ValueError(
                    f"duplicate monomer PDB variant for {sample}, variant {variant}"
                )
            grouped[sample][variant] = {
                "path": path,
                "sequence": sequence,
                "role": VARIANT_ROLE[variant],
            }
        inventory_rows.append(row)
    return pd.DataFrame(inventory_rows), grouped


def find_column(columns: Sequence[object], aliases: Sequence[str]) -> Optional[str]:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): str(column)
        for column in columns
    }
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key in normalized:
            return normalized[key]
    return None


def sequence_from_external_id(value: object) -> str:
    text = str(value or "")
    match = re.search(r"_[1-4]_([^_]+?)(?:_model)?$", text)
    if match:
        return naturalize_sequence(match.group(1))
    parts = text.rsplit("_", 1)
    return naturalize_sequence(parts[-1]) if parts else ""


def discover_permeability(permeability_root: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Discover monomer permeability CSVs by schema, then index by sequence."""
    notes: List[str] = []
    if not permeability_root.exists():
        notes.append(f"permeability root does not exist: {permeability_root}")
        return pd.DataFrame(), notes

    all_csvs = sorted(permeability_root.rglob("*.csv"))
    monomer_csvs = [
        path
        for path in all_csvs
        if "monomer" in str(path).lower() and "complex" not in str(path).lower()
    ]
    candidates = monomer_csvs or [
        path for path in all_csvs if "complex" not in str(path).lower()
    ]

    rows: List[Dict[str, object]] = []
    accepted_files = 0
    for path in candidates:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            notes.append(f"skipped unreadable CSV {path}: {type(exc).__name__}")
            continue
        pred_col = find_column(
            frame.columns,
            ["permeability_pred", "permeability_prediction", "permeability", "pred"],
        )
        if pred_col is None:
            continue
        seq_col = find_column(
            frame.columns,
            ["fasta", "sequence", "seq", "peptide", "natural_sequence"],
        )
        id_col = find_column(
            frame.columns,
            ["id", "name", "sample_id", "pdb_file", "filename"],
        )
        accepted_files += 1
        for external_index, record in frame.iterrows():
            sequence = (
                naturalize_sequence(record.get(seq_col, "")) if seq_col else ""
            )
            if not sequence and id_col:
                sequence = sequence_from_external_id(record.get(id_col, ""))
            prediction = pd.to_numeric(
                pd.Series([record.get(pred_col)]), errors="coerce"
            ).iloc[0]
            if not sequence or pd.isna(prediction):
                continue
            rows.append(
                {
                    "sequence_key": sequence,
                    "permeability_pred": float(prediction),
                    "permeability_source_file": str(path.resolve()),
                    "permeability_external_id": (
                        str(record.get(id_col, "")) if id_col else str(external_index)
                    ),
                }
            )
    notes.append(
        f"permeability CSV candidates={len(candidates)}, accepted_by_schema={accepted_files}"
    )
    if not rows:
        return pd.DataFrame(), notes
    raw = pd.DataFrame(rows).drop_duplicates(
        ["sequence_key", "permeability_pred", "permeability_source_file", "permeability_external_id"]
    )
    return raw, notes


def permeability_index(raw: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    if raw.empty:
        return {}
    result: Dict[str, Dict[str, object]] = {}
    for sequence, group in raw.groupby("sequence_key", sort=True):
        values = pd.to_numeric(group["permeability_pred"], errors="coerce").dropna()
        result[str(sequence)] = {
            "permeability_pred": float(values.max()),
            "permeability_pred_mean": float(values.mean()),
            "permeability_pred_median": float(values.median()),
            "permeability_match_count": int(len(values)),
            "permeability_source_files": ";".join(
                sorted(set(group["permeability_source_file"].astype(str)))
            ),
        }
    return result


def add_prefixed(target: Dict[str, object], prefix: str, values: Mapping[str, object]):
    for key, value in values.items():
        if key in {"residues", "path", "file"}:
            continue
        target[f"{prefix}_{key}"] = value


def aggregate_summary(frame: pd.DataFrame, label: str) -> Dict[str, object]:
    def number(column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    primary_rmsd = number("naturalized_ca_rmsd_best_forward_cyclic_shift")
    tm_score = number("naturalized_tm_score_symmetric_best_forward_cyclic_shift")
    diversity = number(
        "naturalized_diversity_1_minus_tm_best_forward_cyclic_shift"
    )
    e2e_plddt = number("e2e_naturalized_ca_bfactor_mean")
    permeability = number("e2e_permeability_pred")
    return {
        "group": label,
        "n_samples": len(frame),
        "n_structure_ok": int(
            frame.get("naturalized_structure_status", pd.Series(dtype=str))
            .astype(str)
            .eq("ok")
            .sum()
        ),
        "n_tm_ok": int(
            frame.get("naturalized_tm_status", pd.Series(dtype=str))
            .astype(str)
            .eq("ok")
            .sum()
        ),
        "n_explicit_methyl_pair_available": int(
            frame.get("explicit_methyl_structure_status", pd.Series(dtype=str))
            .astype(str)
            .eq("ok")
            .sum()
        ),
        "median_naturalized_ca_rmsd": primary_rmsd.median(),
        "mean_naturalized_ca_rmsd": primary_rmsd.mean(),
        "naturalized_ca_rmsd_lt_2": int(primary_rmsd.lt(2).sum()),
        "naturalized_ca_rmsd_lt_3": int(primary_rmsd.lt(3).sum()),
        "naturalized_ca_rmsd_lt_5": int(primary_rmsd.lt(5).sum()),
        "median_tm_score": tm_score.median(),
        "median_diversity_1_minus_tm": diversity.median(),
        "median_e2e_ca_bfactor_plddt": e2e_plddt.median(),
        "e2e_permeability_available": int(permeability.notna().sum()),
        "median_e2e_permeability": permeability.median(),
    }


def write_report(
    path: Path,
    detail: pd.DataFrame,
    inventory: pd.DataFrame,
    problems: pd.DataFrame,
    warnings: List[str],
    quality_pass: bool,
) -> None:
    def detail_numeric(column: str) -> pd.Series:
        if column not in detail:
            return pd.Series(np.nan, index=detail.index, dtype=float)
        return pd.to_numeric(detail[column], errors="coerce")

    def detail_status(column: str) -> pd.Series:
        if column not in detail:
            return pd.Series("", index=detail.index, dtype=str)
        return detail[column].astype(str)

    variant_counts = (
        pd.to_numeric(inventory.get("variant"), errors="coerce")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    rmsd = detail_numeric("naturalized_ca_rmsd_best_forward_cyclic_shift")
    tm_score = detail_numeric(
        "naturalized_tm_score_symmetric_best_forward_cyclic_shift"
    )
    permeability = detail_numeric("e2e_permeability_pred")
    structure_status = detail_status("naturalized_structure_status")
    explicit_status = detail_status("explicit_methyl_structure_status")
    lines = [
        "===== MONOMER STRUCTURE / CONFIDENCE / TM / PERMEABILITY REPORT =====",
        f"Expected monomer samples: {EXPECTED_SAMPLES}",
        f"Observed monomer samples: {len(detail)}",
        f"Expected PDB files: {EXPECTED_PDBS}",
        f"Observed PDB files: {len(inventory)}",
        f"PDB variant counts: {variant_counts}",
        "Variant mapping: 1=reference explicit methyl, 2=reference naturalized, "
        "3=e2e explicit methyl, 4=e2e naturalized",
        (
            "Primary naturalized pairs complete: "
            f"{int(structure_status.eq('ok').sum())}/{len(detail)}"
        ),
        (
            "Explicit-methyl sensitivity pairs available: "
            f"{int(explicit_status.eq('ok').sum())}/{len(detail)}"
        ),
        "",
        "===== PRIMARY NATURALIZED STRUCTURE RESULTS =====",
        f"Median best-forward-cyclic CA RMSD (A): {rmsd.median():.6f}",
        f"CA RMSD < 2 A: {int(rmsd.lt(2).sum())}/{len(detail)}",
        f"CA RMSD < 3 A: {int(rmsd.lt(3).sum())}/{len(detail)}",
        f"CA RMSD < 5 A: {int(rmsd.lt(5).sum())}/{len(detail)}",
        f"TM-score available: {int(tm_score.notna().sum())}/{len(detail)}",
        f"Median symmetric TM-score: {tm_score.median():.6f}",
        f"Median diversity (1-TM): {(1.0 - tm_score).median():.6f}",
        (
            "E2E permeability available: "
            f"{int(permeability.notna().sum())}/{len(detail)}"
        ),
        "",
        "===== INTERPRETATION NOTES =====",
        "- Primary RMSD/TM metrics compare HighFold reference-sequence and e2e-design predictions.",
        "- They are conformational-change/designability metrics, not experimental native validation.",
        "- Every forward cyclic register is tested; reverse residue order is never allowed.",
        "- Methyl/non-methyl deviations are grouped by lowercase positions in e2e_design_sequence.",
        "- CA B-factor is retained as the complete pLDDT fallback; COMMENT pLDDT/pTM remain separate.",
        "- ipTM/inter-PAE and cross-interface energy are not applicable to single-chain monomers.",
        "- All-atom RMSD is not claimed because reference and design sequences differ.",
        "",
        f"QUALITY GATE: {'PASS' if quality_pass else 'FAIL'}",
        f"PROBLEMS: {len(problems)}",
        f"WARNINGS: {len(warnings)}",
    ]
    if warnings:
        lines.extend(["", "Warning details:"] + [f"- {item}" for item in warnings])
    if len(problems):
        lines.extend(
            ["", "Problem details:"]
            + [
                f"- {row.get('sample_name', '')}: {row.get('problem', '')}"
                for _, row in problems.iterrows()
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    design_manifest_path = Path(args.design_manifest).resolve()
    reference_manifest_path = Path(args.reference_manifest).resolve()
    pdb_dir = Path(args.pdb_dir).resolve()
    permeability_root = Path(args.permeability_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, label in [
        (design_manifest_path, "design manifest"),
        (reference_manifest_path, "reference manifest"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"missing monomer PDB directory: {pdb_dir}")
    if tm_align is None:
        raise RuntimeError(
            "tmtools is not importable. Activate the wain environment used for "
            "the earlier TM-diversity run, or install tmtools==0.3.0 there."
        )

    design_manifest = pd.read_csv(design_manifest_path)
    reference_manifest = pd.read_csv(reference_manifest_path)
    required_design = {
        "sample_name",
        "reference_original_sequence",
        "reference_natural_sequence",
        "e2e_design_sequence",
        "e2e_sequence_for_structure_prediction",
        "sequence_length",
    }
    missing = sorted(required_design - set(design_manifest.columns))
    if missing:
        raise ValueError(f"design manifest missing columns: {missing}")
    if len(design_manifest) != EXPECTED_SAMPLES:
        raise ValueError(
            f"expected {EXPECTED_SAMPLES} design-manifest rows, "
            f"observed {len(design_manifest)}"
        )
    if design_manifest["sample_name"].nunique() != EXPECTED_SAMPLES:
        raise ValueError("design manifest sample_name is not unique")

    reference_meta = reference_manifest[
        [column for column in ["target_name", "dataset_type", "record_index"] if column in reference_manifest]
    ].copy()
    reference_meta = reference_meta.rename(columns={"target_name": "sample_name"})
    manifest = design_manifest.merge(
        reference_meta, how="left", on="sample_name", validate="one_to_one"
    )
    if "dataset_type" not in manifest:
        manifest["dataset_type"] = ""

    inventory, grouped = parse_inventory(pdb_dir)
    if len(inventory) != EXPECTED_PDBS:
        raise ValueError(
            f"expected {EXPECTED_PDBS} monomer PDBs, observed {len(inventory)}"
        )
    if int(inventory["filename_parse_status"].eq("ok").sum()) != EXPECTED_PDBS:
        bad = inventory[inventory["filename_parse_status"] != "ok"]["pdb_file"].tolist()
        raise ValueError(f"unparseable monomer PDB filenames: {bad[:20]}")
    if len(grouped) != EXPECTED_SAMPLES:
        raise ValueError(
            f"expected {EXPECTED_SAMPLES} PDB sample groups, observed {len(grouped)}"
        )
    manifest_names = set(manifest["sample_name"].astype(str))
    grouped_names = set(grouped)
    if manifest_names != grouped_names:
        raise ValueError(
            "manifest/PDB sample mismatch: "
            f"missing={sorted(manifest_names-grouped_names)}, "
            f"extra={sorted(grouped_names-manifest_names)}"
        )

    permeability_raw, permeability_notes = discover_permeability(permeability_root)
    perm_by_sequence = permeability_index(permeability_raw)
    warnings: List[str] = [
        note
        for note in permeability_notes
        if note.startswith("skipped unreadable")
        or "does not exist" in note
    ]
    if not perm_by_sequence:
        warnings.append(
            "No monomer permeability rows were discovered; permeability columns remain NA."
        )

    detail_rows: List[Dict[str, object]] = []
    residue_rows: List[Dict[str, object]] = []
    problem_rows: List[Dict[str, object]] = []
    inventory_updates: Dict[str, Dict[str, object]] = {}

    for _, source in manifest.sort_values("sample_name").iterrows():
        sample = str(source["sample_name"])
        variants = grouped[sample]
        expected_sequences = {
            1: str(source["reference_original_sequence"]),
            2: str(source["reference_natural_sequence"]),
            3: str(source["e2e_design_sequence"]),
            4: str(source["e2e_sequence_for_structure_prediction"]),
        }
        row: Dict[str, object] = {
            "sample_name": sample,
            "dataset_type": source.get("dataset_type", ""),
            "record_index": source.get("record_index", ""),
            "reference_original_sequence": expected_sequences[1],
            "reference_natural_sequence": expected_sequences[2],
            "e2e_design_sequence": expected_sequences[3],
            "e2e_natural_sequence": expected_sequences[4],
            "sequence_length": int(source["sequence_length"]),
            "reference_methyl_count": sum(
                token.islower() for token in expected_sequences[1]
            ),
            "e2e_methyl_count": sum(
                token.islower() for token in expected_sequences[3]
            ),
            "natural_aa_recovery_fixed_order": sum(
                a == b
                for a, b in zip(expected_sequences[2], expected_sequences[4])
            )
            / len(expected_sequences[2]),
            "variant_1_present": int(1 in variants),
            "variant_2_present": int(2 in variants),
            "variant_3_present": int(3 in variants),
            "variant_4_present": int(4 in variants),
        }
        try:
            if len(expected_sequences[2]) != len(expected_sequences[4]):
                raise ValueError("reference/e2e naturalized sequence length mismatch")
            if len(expected_sequences[2]) != int(source["sequence_length"]):
                raise ValueError("manifest sequence_length mismatch")

            parsed: Dict[int, Dict[str, object]] = {}
            for variant, item in variants.items():
                observed_sequence = str(item["sequence"])
                if observed_sequence != expected_sequences[variant]:
                    raise ValueError(
                        f"variant {variant} filename sequence mismatch: "
                        f"observed={observed_sequence}, "
                        f"expected={expected_sequences[variant]}"
                    )
                parsed[variant] = parse_pdb(
                    Path(item["path"]), expected_sequences[variant]
                )
                prefix = VARIANT_ROLE[variant]
                row[f"{prefix}_pdb_file"] = parsed[variant]["file"]
                row[f"{prefix}_pdb_path"] = parsed[variant]["path"]
                add_prefixed(row, prefix, parsed[variant])
                inventory_updates[parsed[variant]["file"]] = {
                    "manifest_sequence": expected_sequences[variant],
                    "sequence_match_manifest": 1,
                    "parsed_chain": parsed[variant]["chain"],
                    "parsed_residue_count": parsed[variant]["length"],
                    "ca_bfactor_mean": parsed[variant]["ca_bfactor_mean"],
                    "pdb_parse_status": "ok",
                }

            for variant in PRIMARY_VARIANTS:
                if variant not in parsed:
                    raise ValueError(
                        f"required naturalized PDB variant {variant} is missing"
                    )

            structural, residue_detail = cyclic_structure_metrics(
                reference=parsed[2],
                design=parsed[4],
                design_sequence=expected_sequences[3],
            )
            add_prefixed(row, "naturalized", structural)
            tm_metrics = cyclic_tm_metrics(
                reference=parsed[2],
                design=parsed[4],
                reference_sequence=expected_sequences[2],
                design_sequence=expected_sequences[4],
            )
            add_prefixed(row, "naturalized", tm_metrics)
            row["naturalized_structure_status"] = "ok"
            for residue in residue_detail:
                residue_rows.append(
                    {
                        "sample_name": sample,
                        "dataset_type": source.get("dataset_type", ""),
                        "comparison_scope": "naturalized_reference_vs_e2e",
                        **residue,
                    }
                )

            if 1 in parsed and 3 in parsed:
                explicit, _explicit_residue = cyclic_structure_metrics(
                    reference=parsed[1],
                    design=parsed[3],
                    design_sequence=expected_sequences[3],
                )
                add_prefixed(row, "explicit_methyl", explicit)
                row["explicit_methyl_structure_status"] = "ok"
            else:
                row["explicit_methyl_structure_status"] = (
                    "not_available_missing_variant_1_or_3"
                )

            for role, sequence in [
                ("reference", expected_sequences[2]),
                ("e2e", expected_sequences[4]),
            ]:
                perm = perm_by_sequence.get(naturalize_sequence(sequence))
                if perm:
                    add_prefixed(row, f"{role}_permeability", perm)
                    row[f"{role}_permeability_status"] = "matched_exact_sequence"
                else:
                    row[f"{role}_permeability_status"] = "missing"
            ref_perm = row.get("reference_permeability_permeability_pred", np.nan)
            e2e_perm = row.get("e2e_permeability_permeability_pred", np.nan)
            # Friendlier aliases retained for the final workbook.
            row["reference_permeability_pred"] = ref_perm
            row["e2e_permeability_pred"] = e2e_perm
            if finite(ref_perm) and finite(e2e_perm):
                row["permeability_delta_e2e_minus_reference"] = (
                    float(e2e_perm) - float(ref_perm)
                )
                row["e2e_permeability_gt_reference"] = int(
                    float(e2e_perm) > float(ref_perm)
                )
            else:
                row["permeability_delta_e2e_minus_reference"] = np.nan
                row["e2e_permeability_gt_reference"] = np.nan

        except Exception as exc:
            row["naturalized_structure_status"] = "failed"
            row["naturalized_tm_status"] = row.get("naturalized_tm_status", "failed")
            row["structure_error"] = f"{type(exc).__name__}: {exc}"
            problem_rows.append(
                {
                    "sample_name": sample,
                    "problem": row["structure_error"],
                }
            )
        detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    residue_detail = pd.DataFrame(residue_rows)
    problems = pd.DataFrame(problem_rows, columns=["sample_name", "problem"])

    # Preserve a complete, auditable output schema even if every sample fails
    # before a derived metric is created.  The quality gate will still fail and
    # the original sample-level errors remain in the problem table/report.
    for column, default in {
        "naturalized_structure_status": "failed",
        "naturalized_tm_status": "failed",
        "explicit_methyl_structure_status": "not_available",
        "naturalized_ca_rmsd_best_forward_cyclic_shift": np.nan,
        "naturalized_tm_score_symmetric_best_forward_cyclic_shift": np.nan,
        "naturalized_diversity_1_minus_tm_best_forward_cyclic_shift": np.nan,
        "reference_permeability_pred": np.nan,
        "e2e_permeability_pred": np.nan,
    }.items():
        if column not in detail:
            detail[column] = default

    reference_permeability_available = int(
        pd.to_numeric(detail["reference_permeability_pred"], errors="coerce")
        .notna()
        .sum()
    )
    e2e_permeability_available = int(
        pd.to_numeric(detail["e2e_permeability_pred"], errors="coerce")
        .notna()
        .sum()
    )
    if perm_by_sequence and reference_permeability_available != EXPECTED_SAMPLES:
        warnings.append(
            "Reference monomer permeability is incomplete: "
            f"{reference_permeability_available}/{EXPECTED_SAMPLES}."
        )
    if perm_by_sequence and e2e_permeability_available != EXPECTED_SAMPLES:
        warnings.append(
            "E2E monomer permeability is incomplete: "
            f"{e2e_permeability_available}/{EXPECTED_SAMPLES}."
        )

    inventory = inventory.copy()
    for index, item in inventory.iterrows():
        update = inventory_updates.get(str(item["pdb_file"]), {})
        for key, value in update.items():
            inventory.loc[index, key] = value

    summary_rows = [aggregate_summary(detail, "ALL")]
    for dataset_type, group in detail.groupby("dataset_type", dropna=False, sort=True):
        summary_rows.append(aggregate_summary(group, str(dataset_type)))
    summary = pd.DataFrame(summary_rows)

    core_status = detail["naturalized_structure_status"].astype(str).eq("ok")
    tm_status = detail["naturalized_tm_status"].astype(str).eq("ok")
    core_quality_pass = (
        len(detail) == EXPECTED_SAMPLES
        and detail["sample_name"].nunique() == EXPECTED_SAMPLES
        and len(inventory) == EXPECTED_PDBS
        and int(detail["variant_2_present"].sum()) == EXPECTED_SAMPLES
        and int(detail["variant_4_present"].sum()) == EXPECTED_SAMPLES
        and bool(core_status.all())
        and bool(tm_status.all())
        and len(problems) == 0
    )

    inventory.to_csv(
        out_dir / "monomer_pdb_inventory_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail.to_csv(
        out_dir / "monomer_structure_metrics_by_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residue_detail.to_csv(
        out_dir / "monomer_structure_metrics_by_residue.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        out_dir / "monomer_structure_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    problems.to_csv(
        out_dir / "monomer_structure_metrics_problem_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not permeability_raw.empty:
        permeability_raw.to_csv(
            out_dir / "monomer_permeability_discovered_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )
    write_report(
        out_dir / "monomer_structure_metrics_report.txt",
        detail=detail,
        inventory=inventory,
        problems=problems,
        warnings=warnings,
        quality_pass=core_quality_pass,
    )
    (out_dir / "monomer_structure_output_manifest.json").write_text(
        json.dumps(
            {
                "quality_gate": "PASS" if core_quality_pass else "FAIL",
                "samples": len(detail),
                "pdb_files": len(inventory),
                "naturalized_pairs_ok": int(core_status.sum()),
                "tm_pairs_ok": int(tm_status.sum()),
                "explicit_methyl_pairs_available": int(
                    detail["explicit_methyl_structure_status"].eq("ok").sum()
                ),
                "reference_permeability_available": int(
                    reference_permeability_available
                ),
                "e2e_permeability_available": int(
                    e2e_permeability_available
                ),
                "permeability_discovery_notes": permeability_notes,
                "warnings": warnings,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== MONOMER STRUCTURE METRICS COMPLETE =====")
    print(f"samples: {len(detail)}")
    print(f"PDB files: {len(inventory)}")
    print(f"naturalized structure pairs OK: {int(core_status.sum())}/{len(detail)}")
    print(f"TM pairs OK: {int(tm_status.sum())}/{len(detail)}")
    print(
        "explicit-methyl pairs available: "
        f"{int(detail['explicit_methyl_structure_status'].eq('ok').sum())}/{len(detail)}"
    )
    print(f"quality gate: {'PASS' if core_quality_pass else 'FAIL'}")
    print(f"output directory: {out_dir}")
    return 0 if core_quality_pass else 1


if __name__ == "__main__":
    sys.exit(main())
