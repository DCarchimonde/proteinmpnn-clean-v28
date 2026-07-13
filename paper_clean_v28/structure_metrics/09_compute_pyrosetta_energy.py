#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09_compute_pyrosetta_energy.py

Evaluate the 85 selected HighFold complex structures with the PyRosetta
ref2015 score function.

Primary descriptive outputs
---------------------------
1. Complex ref2015 total score and score per residue.
2. Receptor-only score after deleting the designed peptide chain.
3. Peptide-only score after deleting all receptor chains.
4. Unrelaxed interaction energy:

       E_interaction = E_complex - E_receptor - E_peptide

   More-negative values indicate a more favorable interaction in this fixed
   predicted conformation. This is a Rosetta energy difference in REU, not an
   experimental binding free energy and not kcal/mol.

Important limitations
---------------------
- No FastRelax or backbone minimization is performed. This deliberately avoids
  changing the HighFold-predicted structures before evaluation.
- Lowercase letters in design_seq are methylation annotations. The HighFold PDB
  structures generally contain ordinary natural-amino-acid residue records, so
  these scores do not explicitly model N-methyl chemistry.
- The flag interaction_energy_lt_0 is descriptive only. It must not be called a
  validated stability or experimental success label without an independently
  justified criterion.
- Failed structures are recorded as NaN with an error message. They are never
  replaced by zero.

Input
-----
paper_clean_v28_outputs/structure_metrics/complex_rmsd_metrics.csv

Full-run outputs
----------------
paper_clean_v28_outputs/structure_metrics/
    complex_pyrosetta_energy_best85.csv
    complex_pyrosetta_energy_summary_by_temperature.csv
    complex_pyrosetta_energy_summary_by_target.csv
    complex_pyrosetta_energy_report.txt
    complex_pyrosetta_energy_problem_rows.csv

A smoke test with --limit N writes files containing the suffix _smoke and does
not overwrite the full-run files.
"""

from __future__ import annotations

import argparse
import math
import platform
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory


OUT_DIR = Path("paper_clean_v28_outputs/structure_metrics")
INPUT_PATH = OUT_DIR / "complex_rmsd_metrics.csv"
EXPECTED_ROWS = 85


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute PyRosetta ref2015 energies for best85 complexes.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test only the first N rows and write *_smoke outputs.",
    )
    return p.parse_args()


def safe_float(x):
    try:
        if x is None or x == "":
            return math.nan
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except Exception:
        return math.nan


def resolve_pdb_path(text: str) -> Path:
    """Convert Windows-style relative paths stored in CSV to WSL/Linux paths."""
    raw = str(text or "").strip().replace("\\", "/")
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def residue_indices_for_pdb_chain(pose, chain_id: str) -> List[int]:
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        return []
    wanted = str(chain_id).strip()
    return [i for i in range(1, pose.total_residue() + 1) if str(pdb_info.chain(i)).strip() == wanted]


def is_contiguous(indices: List[int]) -> bool:
    if not indices:
        return False
    return indices == list(range(indices[0], indices[-1] + 1))


def extract_chain_sequence(pose, indices: Iterable[int]) -> str:
    chars = []
    for i in indices:
        try:
            chars.append(str(pose.residue(i).name1()))
        except Exception:
            chars.append("X")
    return "".join(chars)


def make_receptor_and_peptide_poses(pose, peptide_indices: List[int]):
    """
    Build receptor-only and peptide-only poses by deleting complete contiguous
    residue ranges. The designed peptide chain is required to be contiguous.
    """
    if not peptide_indices or not is_contiguous(peptide_indices):
        raise ValueError("peptide chain residue indices are absent or non-contiguous")

    pep_start = peptide_indices[0]
    pep_end = peptide_indices[-1]
    total = pose.total_residue()

    receptor_pose = pose.clone()
    receptor_pose.delete_residue_range_slow(pep_start, pep_end)

    peptide_pose = pose.clone()
    if pep_end < total:
        peptide_pose.delete_residue_range_slow(pep_end + 1, total)
    if pep_start > 1:
        peptide_pose.delete_residue_range_slow(1, pep_start - 1)

    return receptor_pose, peptide_pose


def score_one(row: pd.Series, scorefxn) -> Dict[str, object]:
    started = time.perf_counter()
    target = str(row.get("target_name", "")).upper().strip()
    temperature = row.get("temperature", "")
    design_seq = str(row.get("design_seq", "")).strip()
    peptide_chain = str(row.get("predicted_peptide_chain", "")).strip()
    pdb_path = resolve_pdb_path(row.get("pdb_path", ""))

    out: Dict[str, object] = {
        "row_index": row.get("row_index", ""),
        "target_name": target,
        "temperature": temperature,
        "design_seq": design_seq,
        "design_natural_seq": design_seq.upper(),
        "n_methylation_marks": sum(1 for x in design_seq if x.islower()),
        "predicted_peptide_chain": peptide_chain,
        "pdb_file": row.get("pdb_file", pdb_path.name),
        "pdb_path": str(row.get("pdb_path", "")),
        "resolved_pdb_path": str(pdb_path),
        "rmsd_status": row.get("rmsd_status", ""),
        "energy_status": "failed",
        "error_message": "",
    }

    try:
        if not pdb_path.exists():
            raise FileNotFoundError(str(pdb_path))
        if not peptide_chain:
            raise ValueError("predicted_peptide_chain is empty")

        pose = pose_from_pdb(str(pdb_path))
        n_total = int(pose.total_residue())
        if n_total == 0:
            raise ValueError("PyRosetta loaded a pose with zero residues")

        peptide_indices = residue_indices_for_pdb_chain(pose, peptide_chain)
        if not peptide_indices:
            available = sorted(
                {
                    str(pose.pdb_info().chain(i)).strip()
                    for i in range(1, pose.total_residue() + 1)
                }
            )
            raise ValueError(
                f"peptide chain {peptide_chain!r} not found after PyRosetta import; "
                f"available chains={available}"
            )
        if not is_contiguous(peptide_indices):
            raise ValueError(f"peptide chain {peptide_chain!r} is not contiguous in the pose")

        loaded_peptide_seq = extract_chain_sequence(pose, peptide_indices)
        receptor_pose, peptide_pose = make_receptor_and_peptide_poses(pose, peptide_indices)

        n_peptide = int(peptide_pose.total_residue())
        n_receptor = int(receptor_pose.total_residue())
        if n_peptide <= 0 or n_receptor <= 0:
            raise ValueError(
                f"invalid split sizes: total={n_total}, receptor={n_receptor}, peptide={n_peptide}"
            )
        if n_receptor + n_peptide != n_total:
            raise ValueError(
                f"split residue count mismatch: total={n_total}, receptor+peptide={n_receptor+n_peptide}"
            )

        complex_score = float(scorefxn(pose))
        receptor_score = float(scorefxn(receptor_pose))
        peptide_score = float(scorefxn(peptide_pose))
        interaction = complex_score - receptor_score - peptide_score

        out.update(
            {
                "n_total_residues_loaded": n_total,
                "n_receptor_residues_loaded": n_receptor,
                "n_peptide_residues_loaded": n_peptide,
                "peptide_sequence_loaded": loaded_peptide_seq,
                "peptide_sequence_matches_design_naturalized": int(
                    loaded_peptide_seq == design_seq.upper()
                ),
                "rosetta_complex_total_score": complex_score,
                "rosetta_complex_score_per_residue": complex_score / n_total,
                "rosetta_receptor_total_score": receptor_score,
                "rosetta_receptor_score_per_residue": receptor_score / n_receptor,
                "rosetta_peptide_total_score": peptide_score,
                "rosetta_peptide_score_per_residue": peptide_score / n_peptide,
                "rosetta_interaction_energy_unrelaxed": interaction,
                "rosetta_interaction_energy_per_peptide_residue": interaction / n_peptide,
                "interaction_energy_lt_0": int(interaction < 0.0),
                "energy_status": "ok",
                "error_message": "",
            }
        )
    except Exception as exc:
        out["error_message"] = f"{type(exc).__name__}: {exc}"
        out["error_trace_tail"] = " | ".join(
            traceback.format_exc(limit=2).strip().splitlines()[-3:]
        )

    out["elapsed_seconds"] = time.perf_counter() - started
    return out


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, group in df.groupby(group_col, dropna=False, sort=True):
        ok = group[group["energy_status"] == "ok"].copy()
        interaction = numeric_series(ok, "rosetta_interaction_energy_unrelaxed")
        interaction_per_pep = numeric_series(ok, "rosetta_interaction_energy_per_peptide_residue")
        total_per_res = numeric_series(ok, "rosetta_complex_score_per_residue")
        peptide_per_res = numeric_series(ok, "rosetta_peptide_score_per_residue")
        seq_match = numeric_series(ok, "peptide_sequence_matches_design_naturalized")

        rows.append(
            {
                group_col: key,
                "n_rows": len(group),
                "n_ok": len(ok),
                "n_failed": len(group) - len(ok),
                "n_sequence_match": int(seq_match.sum()) if len(seq_match) else 0,
                "mean_complex_score_per_residue": total_per_res.mean(),
                "median_complex_score_per_residue": total_per_res.median(),
                "mean_peptide_score_per_residue": peptide_per_res.mean(),
                "median_peptide_score_per_residue": peptide_per_res.median(),
                "mean_interaction_energy_unrelaxed": interaction.mean(),
                "median_interaction_energy_unrelaxed": interaction.median(),
                "mean_interaction_energy_per_peptide_residue": interaction_per_pep.mean(),
                "median_interaction_energy_per_peptide_residue": interaction_per_pep.median(),
                "n_interaction_energy_lt_0": int((interaction < 0).sum()),
                "fraction_interaction_energy_lt_0": float((interaction < 0).mean()) if len(interaction) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def version_text() -> str:
    try:
        return str(pyrosetta.version())
    except Exception:
        return "not_reported"


def write_report(
    df: pd.DataFrame,
    summary_temp: pd.DataFrame,
    report_path: Path,
    expected_rows: int,
    is_smoke: bool,
    init_options: str,
) -> None:
    ok = df[df["energy_status"] == "ok"].copy()
    failed = df[df["energy_status"] != "ok"].copy()
    interaction = numeric_series(ok, "rosetta_interaction_energy_unrelaxed")
    interaction_per_pep = numeric_series(ok, "rosetta_interaction_energy_per_peptide_residue")
    total_per_res = numeric_series(ok, "rosetta_complex_score_per_residue")
    seq_match = numeric_series(ok, "peptide_sequence_matches_design_naturalized")

    quality_pass = (
        len(df) == expected_rows
        and len(ok) == expected_rows
        and len(failed) == 0
        and numeric_series(ok, "rosetta_complex_total_score").notna().all()
        and interaction.notna().all()
    )

    lines = [
        "===== PYROSETTA REF2015 ENERGY REPORT =====",
        f"Run mode: {'SMOKE TEST' if is_smoke else 'FULL BEST85'}",
        f"Platform: {platform.platform()}",
        f"PyRosetta version: {version_text()}",
        f"Initialization options: {init_options}",
        "Score function: ref2015",
        f"Expected rows for this run: {expected_rows}",
        f"Observed rows: {len(df)}",
        f"Energy OK: {len(ok)}",
        f"Energy failed: {len(failed)}",
        f"Naturalized peptide sequence match: {int(seq_match.sum())}/{len(ok)}",
        "",
        "===== GLOBAL ENERGY RESULTS =====",
        f"Mean complex score per residue: {total_per_res.mean():.6f}" if len(total_per_res) else "Mean complex score per residue: NaN",
        f"Median complex score per residue: {total_per_res.median():.6f}" if len(total_per_res) else "Median complex score per residue: NaN",
        f"Mean unrelaxed interaction energy: {interaction.mean():.6f}" if len(interaction) else "Mean unrelaxed interaction energy: NaN",
        f"Median unrelaxed interaction energy: {interaction.median():.6f}" if len(interaction) else "Median unrelaxed interaction energy: NaN",
        f"Mean interaction energy per peptide residue: {interaction_per_pep.mean():.6f}" if len(interaction_per_pep) else "Mean interaction energy per peptide residue: NaN",
        f"Median interaction energy per peptide residue: {interaction_per_pep.median():.6f}" if len(interaction_per_pep) else "Median interaction energy per peptide residue: NaN",
        f"Interaction energy < 0: {int((interaction < 0).sum())}/{len(interaction)}" if len(interaction) else "Interaction energy < 0: 0/0",
        "",
        "===== SUMMARY BY TEMPERATURE =====",
        summary_temp.to_string(index=False),
        "",
        "===== INTERPRETATION NOTES =====",
        "- Energies are Rosetta Energy Units (REU), not kcal/mol.",
        "- Unrelaxed interaction energy is E_complex - E_receptor - E_peptide at fixed input coordinates.",
        "- More-negative interaction energy is descriptively more favorable.",
        "- interaction_energy_lt_0 is not a validated experimental stability/success label.",
        "- No FastRelax or backbone minimization was applied.",
        "- HighFold PDB residue records generally naturalize methylation-marked sequence positions; explicit N-methyl chemistry is not modeled.",
        "- Failed evaluations remain missing/NaN and are never replaced with zero.",
        "",
        f"QUALITY GATE: {'PASS' if quality_pass else 'FAIL'}",
        f"PROBLEMS: {len(failed)}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df_input = pd.read_csv(INPUT_PATH)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        df_input = df_input.head(args.limit).copy()

    suffix = "_smoke" if args.limit is not None else ""
    expected_rows = len(df_input) if args.limit is not None else EXPECTED_ROWS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / f"complex_pyrosetta_energy_best85{suffix}.csv"
    temp_path = OUT_DIR / f"complex_pyrosetta_energy_summary_by_temperature{suffix}.csv"
    target_path = OUT_DIR / f"complex_pyrosetta_energy_summary_by_target{suffix}.csv"
    report_path = OUT_DIR / f"complex_pyrosetta_energy_report{suffix}.txt"
    problem_path = OUT_DIR / f"complex_pyrosetta_energy_problem_rows{suffix}.csv"

    init_options = "-mute all -ignore_unrecognized_res -ignore_waters -load_PDB_components false -ex1 -ex2"
    pyrosetta.init(init_options)
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015")

    rows: List[Dict[str, object]] = []
    total = len(df_input)
    for pos, (_, row) in enumerate(df_input.iterrows(), start=1):
        result = score_one(row, scorefxn)
        rows.append(result)
        print(
            f"[{pos:02d}/{total:02d}] {result['target_name']} "
            f"T={result['temperature']} {result['design_seq']} -> {result['energy_status']} "
            f"({result['elapsed_seconds']:.2f}s)",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)

    summary_temp = summarize(df, "temperature")
    summary_target = summarize(df, "target_name")
    summary_temp.to_csv(temp_path, index=False)
    summary_target.to_csv(target_path, index=False)

    problem_cols = [
        "row_index",
        "target_name",
        "temperature",
        "design_seq",
        "pdb_file",
        "energy_status",
        "error_message",
        "error_trace_tail",
    ]
    problems = df[df["energy_status"] != "ok"].copy()
    for col in problem_cols:
        if col not in problems.columns:
            problems[col] = ""
    problems[problem_cols].to_csv(problem_path, index=False)

    write_report(
        df=df,
        summary_temp=summary_temp,
        report_path=report_path,
        expected_rows=expected_rows,
        is_smoke=args.limit is not None,
        init_options=init_options,
    )

    n_ok = int((df["energy_status"] == "ok").sum())
    print("完成：PyRosetta ref2015 energy evaluation")
    print(f"rows: {len(df)}, OK: {n_ok}, failed: {len(df)-n_ok}")
    print("outputs:")
    print(result_path.resolve())
    print(temp_path.resolve())
    print(target_path.resolve())
    print(report_path.resolve())
    print(problem_path.resolve())


if __name__ == "__main__":
    main()
