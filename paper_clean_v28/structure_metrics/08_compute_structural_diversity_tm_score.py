#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_compute_structural_diversity_tm_score.py

Compute peptide structural diversity for the complex best85 set using TM-align
through the Python package ``tmtools``.

Primary metric:
    structural_diversity = 1 - symmetric_TM_score

Analysis unit:
    Within each target, compare the five selected peptide structures (one per
    generation temperature) in all pairwise combinations. With 17 targets and
    five structures per target, the expected number of comparisons is:
        17 * C(5, 2) = 170 pairs.

Important interpretation:
- TM-align removes rigid-body rotation and translation, so this metric measures
  internal peptide-shape diversity, not diversity of binding placement in the
  receptor frame.
- TM-score is length dependent for very short peptides. Therefore, comparisons
  are made and summarized within target, where peptide length is fixed or nearly
  fixed, and pooled values should be interpreted descriptively.
- Lowercase design tokens are converted to uppercase natural amino-acid codes
  before TM-align; lowercase only records the methylation mark in this project.

Dependency:
    python -m pip install tmtools==0.3.0

Inputs:
    paper_clean_v28_outputs/structure_metrics/complex_rmsd_metrics.csv

Outputs:
    paper_clean_v28_outputs/structure_metrics/
        complex_structural_diversity_tm_pairwise.csv
        complex_structural_diversity_tm_by_target.csv
        complex_structural_diversity_tm_by_temperature_pair.csv
        complex_structural_diversity_tm_report.txt
        complex_structural_diversity_tm_problem_rows.csv
"""

from __future__ import annotations

import itertools
import math
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from tmtools import tm_align
except Exception as exc:  # pragma: no cover - user-facing dependency guard
    raise SystemExit(
        "tmtools is required. Install it in the active environment with:\n"
        "    python -m pip install tmtools==0.3.0\n"
        f"Original import error: {exc!r}"
    )


OUT_DIR = Path("paper_clean_v28_outputs/structure_metrics")
INPUT_PATH = OUT_DIR / "complex_rmsd_metrics.csv"
PAIRWISE_PATH = OUT_DIR / "complex_structural_diversity_tm_pairwise.csv"
TARGET_PATH = OUT_DIR / "complex_structural_diversity_tm_by_target.csv"
TEMP_PAIR_PATH = OUT_DIR / "complex_structural_diversity_tm_by_temperature_pair.csv"
REPORT_PATH = OUT_DIR / "complex_structural_diversity_tm_report.txt"
PROBLEM_PATH = OUT_DIR / "complex_structural_diversity_tm_problem_rows.csv"

EXPECTED_DESIGNS = 85
EXPECTED_TARGETS = 17
EXPECTED_PER_TARGET = 5
EXPECTED_PAIRS = EXPECTED_TARGETS * (EXPECTED_PER_TARGET * (EXPECTED_PER_TARGET - 1) // 2)


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
    # Project-specific/nonstandard residue names observed in HighFold outputs.
    "NCY": "C", "GNC": "Q", "MMO": "R", "UNK": "X",
}


def norm_temp(value) -> str:
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value).strip()


def temp_sort_key(value) -> Tuple[int, float | str]:
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def safe_float(value):
    try:
        x = float(value)
        return None if math.isnan(x) else x
    except Exception:
        return None


def read_peptide_ca(
    pdb_path: str | Path,
    chain_id: str,
    design_seq: str,
) -> Tuple[np.ndarray, str, List[Tuple[str, str, str]]]:
    """Read one CA coordinate per residue for a selected PDB chain."""
    path = Path(pdb_path)
    if not path.exists():
        raise FileNotFoundError(path)

    requested_chain = str(chain_id).strip()
    residues: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    order: List[Tuple[str, str, str]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            altloc = line[16].strip()
            if altloc not in {"", "A", "1"}:
                continue

            pdb_chain = line[21].strip() or "_"
            if pdb_chain != requested_chain:
                continue

            resname = line[17:20].strip().upper()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (pdb_chain, resseq, icode)

            if key in residues:
                continue

            try:
                xyz = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                )
            except Exception:
                continue

            residues[key] = {"coord": xyz, "resname": resname}
            order.append(key)

    if not order:
        raise ValueError(
            f"No CA atoms found for chain {requested_chain!r} in {path}"
        )

    coords = np.array([residues[key]["coord"] for key in order], dtype=float)
    pdb_seq = "".join(AA3_TO_1.get(str(residues[key]["resname"]), "X") for key in order)

    natural_design_seq = str(design_seq).upper().strip()
    if natural_design_seq and len(natural_design_seq) == len(coords):
        seq = natural_design_seq
    else:
        seq = pdb_seq

    if len(seq) != len(coords):
        raise ValueError(
            f"Sequence/coordinate length mismatch for {path.name} chain {requested_chain}: "
            f"seq={len(seq)}, coords={len(coords)}"
        )

    return coords, seq, order


def positional_sequence_identity(seq1: str, seq2: str):
    if not seq1 or not seq2 or len(seq1) != len(seq2):
        return np.nan
    return sum(a == b for a, b in zip(seq1, seq2)) / len(seq1)


def numeric_summary(values: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
        }
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "min": float(x.min()),
        "max": float(x.max()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    designs = pd.read_csv(INPUT_PATH)
    required_cols = [
        "target_name",
        "temperature",
        "design_seq",
        "pdb_path",
        "predicted_peptide_chain",
        "rmsd_status",
    ]
    missing_cols = [c for c in required_cols if c not in designs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    designs = designs.copy()
    designs["target_name"] = designs["target_name"].astype(str).str.upper().str.strip()
    designs["temperature_norm"] = designs["temperature"].map(norm_temp)
    designs["design_seq"] = designs["design_seq"].astype(str).str.strip()

    problems: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    cache: Dict[int, Tuple[np.ndarray, str, List[Tuple[str, str, str]]]] = {}

    # Preload all peptide chains so parse failures are reported once.
    for idx, row in designs.iterrows():
        if row.get("rmsd_status") != "ok":
            problems.append({
                "level": "PROBLEM",
                "problem_type": "input_rmsd_not_ok",
                "row_index": idx,
                "target_name": row.get("target_name", ""),
                "temperature": row.get("temperature_norm", ""),
                "design_seq": row.get("design_seq", ""),
                "detail": row.get("rmsd_status", ""),
            })
            continue

        try:
            cache[idx] = read_peptide_ca(
                row["pdb_path"],
                row["predicted_peptide_chain"],
                row["design_seq"],
            )
        except Exception as exc:
            problems.append({
                "level": "PROBLEM",
                "problem_type": "peptide_chain_parse_failed",
                "row_index": idx,
                "target_name": row.get("target_name", ""),
                "temperature": row.get("temperature_norm", ""),
                "design_seq": row.get("design_seq", ""),
                "detail": repr(exc),
            })

    for target, group in designs.groupby("target_name", sort=True):
        group = group.sort_values(
            "temperature_norm",
            key=lambda s: s.map(temp_sort_key),
        )

        if len(group) != EXPECTED_PER_TARGET:
            problems.append({
                "level": "PROBLEM",
                "problem_type": "unexpected_design_count_for_target",
                "row_index": "",
                "target_name": target,
                "temperature": "",
                "design_seq": "",
                "detail": f"expected={EXPECTED_PER_TARGET}, observed={len(group)}",
            })

        for (idx1, r1), (idx2, r2) in itertools.combinations(group.iterrows(), 2):
            base = {
                "target_name": target,
                "row_index_1": idx1,
                "row_index_2": idx2,
                "temperature_1": r1["temperature_norm"],
                "temperature_2": r2["temperature_norm"],
                "temperature_pair": f"{r1['temperature_norm']}__{r2['temperature_norm']}",
                "design_seq_1": r1["design_seq"],
                "design_seq_2": r2["design_seq"],
                "pdb_file_1": Path(str(r1["pdb_path"])).name,
                "pdb_file_2": Path(str(r2["pdb_path"])).name,
                "peptide_chain_1": r1["predicted_peptide_chain"],
                "peptide_chain_2": r2["predicted_peptide_chain"],
            }

            if idx1 not in cache or idx2 not in cache:
                base.update({
                    "status": "failed_missing_parsed_chain",
                    "detail": "one or both peptide chains were not parsed",
                })
                pair_rows.append(base)
                continue

            coords1, seq1, _ = cache[idx1]
            coords2, seq2, _ = cache[idx2]

            try:
                result = tm_align(coords1, coords2, seq1, seq2)
                tm1 = float(result.tm_norm_chain1)
                tm2 = float(result.tm_norm_chain2)
                symmetric_tm = (tm1 + tm2) / 2.0
                diversity = 1.0 - symmetric_tm
                pair_rmsd = float(result.rmsd)

                base.update({
                    "status": "ok",
                    "detail": "",
                    "peptide_length_1": len(seq1),
                    "peptide_length_2": len(seq2),
                    "sequence_1_used_by_tmalign": seq1,
                    "sequence_2_used_by_tmalign": seq2,
                    "positional_natural_sequence_identity": positional_sequence_identity(seq1, seq2),
                    "tm_score_norm_chain1": tm1,
                    "tm_score_norm_chain2": tm2,
                    "tm_score_symmetric_mean": symmetric_tm,
                    "diversity_1_minus_tm": diversity,
                    "tmalign_rmsd": pair_rmsd,
                })
                pair_rows.append(base)
            except Exception as exc:
                base.update({
                    "status": "failed_tmalign",
                    "detail": repr(exc),
                    "peptide_length_1": len(seq1),
                    "peptide_length_2": len(seq2),
                    "sequence_1_used_by_tmalign": seq1,
                    "sequence_2_used_by_tmalign": seq2,
                })
                pair_rows.append(base)

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(PAIRWISE_PATH, index=False, encoding="utf-8")

    ok_pairs = pair_df[pair_df.get("status") == "ok"].copy()

    target_rows = []
    for target, group in ok_pairs.groupby("target_name", sort=True):
        tm_stats = numeric_summary(group["tm_score_symmetric_mean"])
        div_stats = numeric_summary(group["diversity_1_minus_tm"])
        rmsd_stats = numeric_summary(group["tmalign_rmsd"])
        target_rows.append({
            "target_name": target,
            "n_pairs": len(group),
            "mean_tm_score": tm_stats["mean"],
            "median_tm_score": tm_stats["median"],
            "min_tm_score": tm_stats["min"],
            "max_tm_score": tm_stats["max"],
            "mean_diversity_1_minus_tm": div_stats["mean"],
            "median_diversity_1_minus_tm": div_stats["median"],
            "min_diversity_1_minus_tm": div_stats["min"],
            "max_diversity_1_minus_tm": div_stats["max"],
            "mean_tmalign_rmsd": rmsd_stats["mean"],
            "median_tmalign_rmsd": rmsd_stats["median"],
        })
    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(TARGET_PATH, index=False, encoding="utf-8")

    temp_rows = []
    for temp_pair, group in ok_pairs.groupby("temperature_pair", sort=True):
        tm_stats = numeric_summary(group["tm_score_symmetric_mean"])
        div_stats = numeric_summary(group["diversity_1_minus_tm"])
        rmsd_stats = numeric_summary(group["tmalign_rmsd"])
        temp_rows.append({
            "temperature_pair": temp_pair,
            "n_targets": group["target_name"].nunique(),
            "n_pairs": len(group),
            "mean_tm_score": tm_stats["mean"],
            "median_tm_score": tm_stats["median"],
            "mean_diversity_1_minus_tm": div_stats["mean"],
            "median_diversity_1_minus_tm": div_stats["median"],
            "mean_tmalign_rmsd": rmsd_stats["mean"],
            "median_tmalign_rmsd": rmsd_stats["median"],
        })
    temp_df = pd.DataFrame(temp_rows)
    temp_df.to_csv(TEMP_PAIR_PATH, index=False, encoding="utf-8")

    failed_pairs = pair_df[pair_df.get("status") != "ok"].copy()
    for _, row in failed_pairs.iterrows():
        problems.append({
            "level": "PROBLEM",
            "problem_type": row.get("status", "pair_failed"),
            "row_index": f"{row.get('row_index_1', '')};{row.get('row_index_2', '')}",
            "target_name": row.get("target_name", ""),
            "temperature": row.get("temperature_pair", ""),
            "design_seq": f"{row.get('design_seq_1', '')};{row.get('design_seq_2', '')}",
            "detail": row.get("detail", ""),
        })

    # Numeric sanity checks.
    if len(ok_pairs) > 0:
        bad_range = ok_pairs[
            (pd.to_numeric(ok_pairs["tm_score_symmetric_mean"], errors="coerce") < -1e-8)
            | (pd.to_numeric(ok_pairs["tm_score_symmetric_mean"], errors="coerce") > 1.0 + 1e-8)
            | (pd.to_numeric(ok_pairs["diversity_1_minus_tm"], errors="coerce") < -1e-8)
            | (pd.to_numeric(ok_pairs["diversity_1_minus_tm"], errors="coerce") > 1.0 + 1e-8)
        ]
        for _, row in bad_range.iterrows():
            problems.append({
                "level": "PROBLEM",
                "problem_type": "tm_score_out_of_range",
                "row_index": f"{row.get('row_index_1')};{row.get('row_index_2')}",
                "target_name": row.get("target_name", ""),
                "temperature": row.get("temperature_pair", ""),
                "design_seq": f"{row.get('design_seq_1')};{row.get('design_seq_2')}",
                "detail": (
                    f"tm={row.get('tm_score_symmetric_mean')}, "
                    f"diversity={row.get('diversity_1_minus_tm')}"
                ),
            })

    problem_df = pd.DataFrame(
        problems,
        columns=[
            "level",
            "problem_type",
            "row_index",
            "target_name",
            "temperature",
            "design_seq",
            "detail",
        ],
    )
    problem_df.to_csv(PROBLEM_PATH, index=False, encoding="utf-8")

    n_designs = len(designs)
    n_targets = designs["target_name"].nunique()
    n_parsed = len(cache)
    n_pairs = len(pair_df)
    n_ok_pairs = len(ok_pairs)
    n_failed_pairs = n_pairs - n_ok_pairs

    try:
        tmtools_version = metadata.version("tmtools")
    except Exception:
        tmtools_version = "unknown"

    global_tm = numeric_summary(ok_pairs["tm_score_symmetric_mean"]) if len(ok_pairs) else numeric_summary(pd.Series(dtype=float))
    global_div = numeric_summary(ok_pairs["diversity_1_minus_tm"]) if len(ok_pairs) else numeric_summary(pd.Series(dtype=float))
    global_rmsd = numeric_summary(ok_pairs["tmalign_rmsd"]) if len(ok_pairs) else numeric_summary(pd.Series(dtype=float))

    quality_pass = (
        n_designs == EXPECTED_DESIGNS
        and n_targets == EXPECTED_TARGETS
        and n_parsed == EXPECTED_DESIGNS
        and n_pairs == EXPECTED_PAIRS
        and n_ok_pairs == EXPECTED_PAIRS
        and len(problem_df) == 0
    )

    lines = [
        "===== STRUCTURAL DIVERSITY 1-TM-SCORE REPORT =====",
        f"tmtools version: {tmtools_version}",
        f"Expected best85 designs: {EXPECTED_DESIGNS}",
        f"Observed designs: {n_designs}",
        f"Expected targets: {EXPECTED_TARGETS}",
        f"Observed targets: {n_targets}",
        f"Parsed peptide chains: {n_parsed}/{n_designs}",
        f"Expected within-target pairs: {EXPECTED_PAIRS}",
        f"Observed pairs: {n_pairs}",
        f"TM-align OK pairs: {n_ok_pairs}",
        f"TM-align failed pairs: {n_failed_pairs}",
        "",
        "===== GLOBAL PAIRWISE RESULTS =====",
        f"Mean symmetric TM-score: {global_tm['mean']:.6f}" if not pd.isna(global_tm["mean"]) else "Mean symmetric TM-score: NaN",
        f"Median symmetric TM-score: {global_tm['median']:.6f}" if not pd.isna(global_tm["median"]) else "Median symmetric TM-score: NaN",
        f"Mean diversity (1-TM): {global_div['mean']:.6f}" if not pd.isna(global_div["mean"]) else "Mean diversity (1-TM): NaN",
        f"Median diversity (1-TM): {global_div['median']:.6f}" if not pd.isna(global_div["median"]) else "Median diversity (1-TM): NaN",
        f"Mean TM-align RMSD: {global_rmsd['mean']:.6f}" if not pd.isna(global_rmsd["mean"]) else "Mean TM-align RMSD: NaN",
        f"Median TM-align RMSD: {global_rmsd['median']:.6f}" if not pd.isna(global_rmsd["median"]) else "Median TM-align RMSD: NaN",
        "",
        "===== BY TARGET =====",
        target_df.to_string(index=False) if len(target_df) else "No target summaries.",
        "",
        "===== BY TEMPERATURE PAIR =====",
        temp_df.to_string(index=False) if len(temp_df) else "No temperature-pair summaries.",
        "",
        "===== INTERPRETATION NOTES =====",
        "- Primary diversity metric is 1 - mean(TM-score normalized by chain 1, TM-score normalized by chain 2).",
        "- TM-align removes rigid-body placement; this evaluates internal peptide-shape diversity, not receptor-frame pose diversity.",
        "- Lowercase methylation markers are naturalized to uppercase for TM-align sequence input.",
        "- Short-peptide TM-scores are length sensitive; within-target comparisons are the primary interpretation.",
        "- These 85 structures are best85 selections, so this is diversity of the selected panel, not all 4115 raw designs.",
        "",
        f"QUALITY GATE: {'PASS' if quality_pass else 'FAIL'}",
        f"PROBLEMS: {len(problem_df)}",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("完成：pairwise peptide TM-score structural diversity")
    print(f"designs: {n_designs}, targets: {n_targets}")
    print(f"pairs: {n_pairs}, OK: {n_ok_pairs}, failed: {n_failed_pairs}")
    if not pd.isna(global_div["mean"]):
        print(f"mean diversity (1-TM): {global_div['mean']:.6f}")
        print(f"median diversity (1-TM): {global_div['median']:.6f}")
    print(f"quality gate: {'PASS' if quality_pass else 'FAIL'}")
    print("outputs:")
    for path in [PAIRWISE_PATH, TARGET_PATH, TEMP_PAIR_PATH, REPORT_PATH, PROBLEM_PATH]:
        print(path.resolve())

    return 0 if quality_pass else 1


if __name__ == "__main__":
    sys.exit(main())
