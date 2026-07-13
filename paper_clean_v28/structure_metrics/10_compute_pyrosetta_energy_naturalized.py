#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_compute_pyrosetta_energy_naturalized.py

Corrected PyRosetta energy evaluation for the complex best85 panel.

Why this script is needed
-------------------------
The HighFold PDB files contain project-specific/nonstandard peptide residue
records. Loading them with ``-ignore_unrecognized_res`` can silently omit those
residues. This script first creates a temporary naturalized PDB in which every
peptide residue is renamed position-by-position to the canonical amino acid in
``design_seq.upper()``. Extra atoms that do not belong to the corresponding
canonical amino acid are removed from the temporary file. The original PDB is
never modified.

The score function is ref2015 with rama_prepro, omega, and p_aa_pp weights set
to zero. These backbone statistical terms are incompatible with some cyclic
peptide topologies in the current structures. The modified weights are applied
uniformly to all 85 structures.

To avoid dangling-disulfide failures caused by deleting chains, receptor-only
and peptide-only subscores are obtained from residue subsets of the intact
pose. The reported fixed-pose cross-interface energy is:

    E_interface = E_complex - E_receptor_subset - E_peptide_subset

This is a Rosetta score in REU for the naturalized, fixed input conformation.
It is not an experimental binding free energy, not kcal/mol, and does not
explicitly model N-methyl chemistry.

Inputs
------
paper_clean_v28_outputs/structure_metrics/complex_rmsd_metrics.csv

Outputs
-------
paper_clean_v28_outputs/structure_metrics/
    complex_pyrosetta_energy_naturalized_best85.csv
    complex_pyrosetta_energy_naturalized_summary_by_temperature.csv
    complex_pyrosetta_energy_naturalized_summary_by_target.csv
    complex_pyrosetta_energy_naturalized_report.txt
    complex_pyrosetta_energy_naturalized_problem_rows.csv

Smoke-test selections write the same names with a ``_smoke`` suffix.
"""

from __future__ import annotations

import argparse
import math
import platform
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory, ScoreType
from pyrosetta.rosetta.utility import vector1_bool


OUT_DIR = Path("paper_clean_v28_outputs/structure_metrics")
INPUT_PATH = OUT_DIR / "complex_rmsd_metrics.csv"
EXPECTED_ROWS = 85

AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

# Heavy atoms retained when naturalizing a peptide residue. OXT is allowed at
# a terminus. Hydrogens are intentionally omitted because HighFold structures
# are scored after PyRosetta adds/handles its own standard residue topology.
AA_ALLOWED_ATOMS = {
    "A": {"N", "CA", "C", "O", "OXT", "CB"},
    "R": {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "N": {"N", "CA", "C", "O", "OXT", "CB", "CG", "OD1", "ND2"},
    "D": {"N", "CA", "C", "O", "OXT", "CB", "CG", "OD1", "OD2"},
    "C": {"N", "CA", "C", "O", "OXT", "CB", "SG"},
    "Q": {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "OE1", "NE2"},
    "E": {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "OE1", "OE2"},
    "G": {"N", "CA", "C", "O", "OXT"},
    "H": {"N", "CA", "C", "O", "OXT", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "I": {"N", "CA", "C", "O", "OXT", "CB", "CG1", "CG2", "CD1"},
    "L": {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD1", "CD2"},
    "K": {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "CE", "NZ"},
    "M": {"N", "CA", "C", "O", "OXT", "CB", "CG", "SD", "CE"},
    "F": {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "P": {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD"},
    "S": {"N", "CA", "C", "O", "OXT", "CB", "OG"},
    "T": {"N", "CA", "C", "O", "OXT", "CB", "OG1", "CG2"},
    "W": {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "Y": {"N", "CA", "C", "O", "OXT", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "V": {"N", "CA", "C", "O", "OXT", "CB", "CG1", "CG2"},
}



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Naturalized PyRosetta energy evaluation for best85 complexes.")
    p.add_argument("--limit", type=int, default=None, help="Use only the first N rows and write *_smoke outputs.")
    p.add_argument(
        "--rows",
        type=str,
        default=None,
        help="Comma-separated original row_index values, e.g. 0,14,60; writes *_smoke outputs.",
    )
    return p.parse_args()



def resolve_pdb_path(text: str) -> Path:
    raw = str(text or "").strip().replace("\\", "/")
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()



def pdb_residue_key(line: str) -> Tuple[str, str]:
    return line[22:26].strip(), line[26].strip()



def collect_peptide_residue_keys(lines: Sequence[str], chain_id: str) -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    seen = set()
    wanted = str(chain_id).strip()
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        if line[21].strip() != wanted:
            continue
        key = pdb_residue_key(line)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys



def naturalize_peptide_pdb(
    source_path: Path,
    output_path: Path,
    peptide_chain: str,
    design_seq: str,
) -> Dict[str, object]:
    """Write a temporary canonical-residue PDB without modifying the source."""
    sequence = str(design_seq).upper().strip()
    if not sequence or any(aa not in AA1_TO_3 for aa in sequence):
        raise ValueError(f"unsupported or empty design sequence: {design_seq!r}")

    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    residue_keys = collect_peptide_residue_keys(lines, peptide_chain)
    if len(residue_keys) != len(sequence):
        raise ValueError(
            f"raw peptide residue count mismatch before PyRosetta import: "
            f"pdb={len(residue_keys)}, design={len(sequence)}"
        )

    key_to_aa = {key: sequence[i] for i, key in enumerate(residue_keys)}
    original_resnames: Dict[Tuple[str, str], str] = {}
    replaced_residue_keys = set()
    removed_extra_atoms = 0
    output_lines: List[str] = []
    wanted = str(peptide_chain).strip()

    for line in lines:
        if line.startswith("CONECT"):
            # Custom methyl atoms can leave CONECT references that are invalid
            # after naturalization. Standard peptide connectivity, LINK, and
            # SSBOND records are retained/inferred separately.
            continue

        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27 and line[21].strip() == wanted:
            key = pdb_residue_key(line)
            aa = key_to_aa.get(key)
            if aa is None:
                continue
            atom_name = line[12:16].strip()
            if atom_name not in AA_ALLOWED_ATOMS[aa]:
                removed_extra_atoms += 1
                continue

            old_resname = line[17:20].strip()
            original_resnames.setdefault(key, old_resname)
            new_resname = AA1_TO_3[aa]
            if old_resname != new_resname or line.startswith("HETATM"):
                replaced_residue_keys.add(key)
            record = "ATOM  " if line.startswith("HETATM") else line[:6]
            line = record + line[6:17] + f"{new_resname:>3}" + line[20:]
            output_lines.append(line)
            continue

        # Update LINK residue names for peptide entries when present. This keeps
        # cyclic/backbone or other explicit links consistent with the temporary
        # canonical residue names.
        if line.startswith("LINK  ") and len(line) >= 57:
            chars = list(line)
            # First LINK partner.
            if line[21].strip() == wanted:
                key1 = (line[22:26].strip(), line[26].strip())
                aa1 = key_to_aa.get(key1)
                if aa1:
                    chars[17:20] = list(f"{AA1_TO_3[aa1]:>3}")
            # Second LINK partner.
            if line[51].strip() == wanted:
                key2 = (line[52:56].strip(), line[56].strip())
                aa2 = key_to_aa.get(key2)
                if aa2:
                    chars[47:50] = list(f"{AA1_TO_3[aa2]:>3}")
            output_lines.append("".join(chars))
            continue

        output_lines.append(line)

    output_path.write_text("".join(output_lines), encoding="utf-8")
    original_names_ordered = [original_resnames.get(k, "") for k in residue_keys]
    return {
        "n_raw_peptide_residues": len(residue_keys),
        "n_naturalized_residue_records": len(replaced_residue_keys),
        "n_removed_extra_peptide_atoms": removed_extra_atoms,
        "raw_peptide_resnames": ";".join(original_names_ordered),
    }



def residue_indices_for_pdb_chain(pose, chain_id: str) -> List[int]:
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        return []
    wanted = str(chain_id).strip()
    return [i for i in range(1, pose.total_residue() + 1) if str(pdb_info.chain(i)).strip() == wanted]



def extract_chain_sequence(pose, indices: Iterable[int]) -> str:
    chars: List[str] = []
    for i in indices:
        chars.append(str(pose.residue(i).name1()))
    return "".join(chars)



def make_subset(total_residue: int, selected_indices: Iterable[int]):
    selected = set(int(i) for i in selected_indices)
    subset = vector1_bool(total_residue)
    for i in range(1, total_residue + 1):
        subset[i] = i in selected
    return subset



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
        "design_length": len(design_seq),
        "n_methylation_marks": sum(1 for x in design_seq if x.islower()),
        "predicted_peptide_chain": peptide_chain,
        "pdb_file": row.get("pdb_file", pdb_path.name),
        "pdb_path": str(row.get("pdb_path", "")),
        "resolved_pdb_path": str(pdb_path),
        "rmsd_status": row.get("rmsd_status", ""),
        "energy_status": "failed",
        "error_stage": "precheck",
        "error_message": "",
    }

    try:
        if not pdb_path.exists():
            raise FileNotFoundError(str(pdb_path))
        if not peptide_chain:
            raise ValueError("predicted_peptide_chain is empty")

        with tempfile.TemporaryDirectory(prefix="pyrosetta_nat_") as tmpdir:
            naturalized_path = Path(tmpdir) / pdb_path.name
            out["error_stage"] = "naturalize_pdb"
            nat_info = naturalize_peptide_pdb(
                source_path=pdb_path,
                output_path=naturalized_path,
                peptide_chain=peptide_chain,
                design_seq=design_seq,
            )
            out.update(nat_info)

            out["error_stage"] = "pose_from_pdb"
            pose = pose_from_pdb(str(naturalized_path))
            n_total = int(pose.total_residue())
            if n_total <= 0:
                raise ValueError("PyRosetta loaded a pose with zero residues")

            peptide_indices = residue_indices_for_pdb_chain(pose, peptide_chain)
            if not peptide_indices:
                available = sorted(
                    {str(pose.pdb_info().chain(i)).strip() for i in range(1, n_total + 1)}
                )
                raise ValueError(
                    f"peptide chain {peptide_chain!r} not found after naturalized import; "
                    f"available chains={available}"
                )

            loaded_sequence = extract_chain_sequence(pose, peptide_indices)
            expected_sequence = design_seq.upper()
            if loaded_sequence != expected_sequence:
                raise ValueError(
                    f"naturalized peptide sequence mismatch: expected={expected_sequence}, "
                    f"loaded={loaded_sequence}"
                )

            receptor_indices = [i for i in range(1, n_total + 1) if i not in set(peptide_indices)]
            if not receptor_indices:
                raise ValueError("no receptor residues remain after identifying peptide chain")

            peptide_subset = make_subset(n_total, peptide_indices)
            receptor_subset = make_subset(n_total, receptor_indices)

            out["error_stage"] = "complex_score"
            complex_score = float(scorefxn(pose))

            out["error_stage"] = "subset_scores"
            receptor_subscore = float(scorefxn.get_sub_score(pose, receptor_subset))
            peptide_subscore = float(scorefxn.get_sub_score(pose, peptide_subset))
            interface_score = complex_score - receptor_subscore - peptide_subscore

            values = [complex_score, receptor_subscore, peptide_subscore, interface_score]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"non-finite Rosetta score(s): {values}")

            out.update(
                {
                    "n_total_residues_loaded": n_total,
                    "n_receptor_residues_loaded": len(receptor_indices),
                    "n_peptide_residues_loaded": len(peptide_indices),
                    "peptide_sequence_loaded": loaded_sequence,
                    "peptide_sequence_matches_design_naturalized": 1,
                    "rosetta_complex_total_score": complex_score,
                    "rosetta_complex_score_per_residue": complex_score / n_total,
                    "rosetta_receptor_subset_score": receptor_subscore,
                    "rosetta_receptor_subset_score_per_residue": receptor_subscore / len(receptor_indices),
                    "rosetta_peptide_subset_score": peptide_subscore,
                    "rosetta_peptide_subset_score_per_residue": peptide_subscore / len(peptide_indices),
                    "rosetta_cross_interface_energy_fixed_pose": interface_score,
                    "rosetta_cross_interface_energy_per_peptide_residue": interface_score / len(peptide_indices),
                    "cross_interface_energy_lt_0": int(interface_score < 0.0),
                    "energy_status": "ok",
                    "error_stage": "",
                    "error_message": "",
                }
            )

    except Exception as exc:
        out["error_message"] = f"{type(exc).__name__}: {exc}"
        out["error_trace_tail"] = " | ".join(
            traceback.format_exc(limit=3).strip().splitlines()[-4:]
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
        complex_per_res = numeric_series(ok, "rosetta_complex_score_per_residue")
        peptide_per_res = numeric_series(ok, "rosetta_peptide_subset_score_per_residue")
        interface = numeric_series(ok, "rosetta_cross_interface_energy_fixed_pose")
        interface_per_pep = numeric_series(ok, "rosetta_cross_interface_energy_per_peptide_residue")
        rows.append(
            {
                group_col: key,
                "n_rows": len(group),
                "n_ok": len(ok),
                "n_failed": len(group) - len(ok),
                "n_sequence_match": int(
                    numeric_series(ok, "peptide_sequence_matches_design_naturalized").sum()
                ) if len(ok) else 0,
                "mean_complex_score_per_residue": complex_per_res.mean(),
                "median_complex_score_per_residue": complex_per_res.median(),
                "mean_peptide_subset_score_per_residue": peptide_per_res.mean(),
                "median_peptide_subset_score_per_residue": peptide_per_res.median(),
                "mean_cross_interface_energy_fixed_pose": interface.mean(),
                "median_cross_interface_energy_fixed_pose": interface.median(),
                "mean_cross_interface_energy_per_peptide_residue": interface_per_pep.mean(),
                "median_cross_interface_energy_per_peptide_residue": interface_per_pep.median(),
                "n_cross_interface_energy_lt_0": int((interface < 0).sum()),
                "fraction_cross_interface_energy_lt_0": float((interface < 0).mean()) if len(interface) else math.nan,
            }
        )
    return pd.DataFrame(rows)



def version_text() -> str:
    try:
        text = str(pyrosetta.version())
        return text.splitlines()[-1] if text else "not_reported"
    except Exception:
        return "not_reported"



def write_report(
    df: pd.DataFrame,
    summary_temp: pd.DataFrame,
    report_path: Path,
    expected_rows: int,
    is_smoke: bool,
    init_options: str,
    disabled_weights: Dict[str, float],
) -> None:
    ok = df[df["energy_status"] == "ok"].copy()
    failed = df[df["energy_status"] != "ok"].copy()
    complex_per_res = numeric_series(ok, "rosetta_complex_score_per_residue")
    interface = numeric_series(ok, "rosetta_cross_interface_energy_fixed_pose")
    interface_per_pep = numeric_series(ok, "rosetta_cross_interface_energy_per_peptide_residue")
    seq_match = numeric_series(ok, "peptide_sequence_matches_design_naturalized")
    loaded_len = numeric_series(ok, "n_peptide_residues_loaded")
    design_len = numeric_series(ok, "design_length")

    quality_pass = (
        len(df) == expected_rows
        and len(ok) == expected_rows
        and len(failed) == 0
        and len(seq_match) == expected_rows
        and int(seq_match.sum()) == expected_rows
        and loaded_len.notna().all()
        and design_len.notna().all()
        and bool((loaded_len == design_len).all())
        and numeric_series(ok, "rosetta_complex_total_score").notna().all()
        and interface.notna().all()
    )

    lines = [
        "===== NATURALIZED PYROSETTA ENERGY REPORT =====",
        f"Run mode: {'SMOKE TEST' if is_smoke else 'FULL BEST85'}",
        f"Platform: {platform.platform()}",
        f"PyRosetta version: {version_text()}",
        f"Initialization options: {init_options}",
        "Score function: ref2015 with rama_prepro=0, omega=0, p_aa_pp=0",
        f"Original disabled weights: {disabled_weights}",
        "Interface method: intact-pose residue-subset decomposition",
        f"Expected rows for this run: {expected_rows}",
        f"Observed rows: {len(df)}",
        f"Energy OK: {len(ok)}",
        f"Energy failed: {len(failed)}",
        f"Naturalized peptide sequence match: {int(seq_match.sum())}/{len(ok)}",
        f"Peptide length match: {int((loaded_len == design_len).sum())}/{len(ok)}",
        "",
        "===== GLOBAL ENERGY RESULTS =====",
        f"Mean complex score per residue: {complex_per_res.mean():.6f}" if len(complex_per_res) else "Mean complex score per residue: NaN",
        f"Median complex score per residue: {complex_per_res.median():.6f}" if len(complex_per_res) else "Median complex score per residue: NaN",
        f"Mean cross-interface energy: {interface.mean():.6f}" if len(interface) else "Mean cross-interface energy: NaN",
        f"Median cross-interface energy: {interface.median():.6f}" if len(interface) else "Median cross-interface energy: NaN",
        f"Mean cross-interface energy per peptide residue: {interface_per_pep.mean():.6f}" if len(interface_per_pep) else "Mean cross-interface energy per peptide residue: NaN",
        f"Median cross-interface energy per peptide residue: {interface_per_pep.median():.6f}" if len(interface_per_pep) else "Median cross-interface energy per peptide residue: NaN",
        f"Cross-interface energy < 0: {int((interface < 0).sum())}/{len(interface)}" if len(interface) else "Cross-interface energy < 0: 0/0",
        "",
        "===== SUMMARY BY TEMPERATURE =====",
        summary_temp.to_string(index=False),
        "",
        "===== INTERPRETATION NOTES =====",
        "- Peptide residue records were naturalized position-by-position to design_seq.upper() in temporary files.",
        "- Original PDB files were not modified.",
        "- Extra peptide atoms outside the canonical natural-amino-acid atom set were removed from temporary files.",
        "- These scores therefore do not explicitly model N-methyl chemistry.",
        "- Energies are Rosetta Energy Units (REU), not kcal/mol.",
        "- Cross-interface energy is derived from residue subsets of the intact fixed pose; no chain deletion was used.",
        "- No FastRelax, minimization, or repacking was applied.",
        "- cross_interface_energy_lt_0 is descriptive and is not a validated experimental success/stability label.",
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
    is_smoke = False

    if args.rows:
        requested = [int(x.strip()) for x in args.rows.split(",") if x.strip()]
        df_input = df_input[df_input["row_index"].astype(int).isin(requested)].copy()
        found = set(df_input["row_index"].astype(int).tolist())
        missing = [x for x in requested if x not in found]
        if missing:
            raise ValueError(f"requested row_index values not found: {missing}")
        order = {value: pos for pos, value in enumerate(requested)}
        df_input["_order"] = df_input["row_index"].astype(int).map(order)
        df_input = df_input.sort_values("_order").drop(columns=["_order"])
        is_smoke = True
    elif args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        df_input = df_input.head(args.limit).copy()
        is_smoke = True

    suffix = "_smoke" if is_smoke else ""
    expected_rows = len(df_input) if is_smoke else EXPECTED_ROWS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / f"complex_pyrosetta_energy_naturalized_best85{suffix}.csv"
    temp_path = OUT_DIR / f"complex_pyrosetta_energy_naturalized_summary_by_temperature{suffix}.csv"
    target_path = OUT_DIR / f"complex_pyrosetta_energy_naturalized_summary_by_target{suffix}.csv"
    report_path = OUT_DIR / f"complex_pyrosetta_energy_naturalized_report{suffix}.txt"
    problem_path = OUT_DIR / f"complex_pyrosetta_energy_naturalized_problem_rows{suffix}.csv"

    init_options = "-mute all -ignore_unrecognized_res -ignore_waters -load_PDB_components false -ex1 -ex2"
    pyrosetta.init(init_options)
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015")

    disabled_weights: Dict[str, float] = {}
    for name, score_type in [
        ("rama_prepro", ScoreType.rama_prepro),
        ("omega", ScoreType.omega),
        ("p_aa_pp", ScoreType.p_aa_pp),
    ]:
        disabled_weights[name] = float(scorefxn.get_weight(score_type))
        scorefxn.set_weight(score_type, 0.0)

    rows: List[Dict[str, object]] = []
    total = len(df_input)
    for pos, (_, row) in enumerate(df_input.iterrows(), start=1):
        result = score_one(row, scorefxn)
        rows.append(result)
        print(
            f"[{pos:02d}/{total:02d}] {result['target_name']} "
            f"T={result['temperature']} {result['design_seq']} -> {result['energy_status']} "
            f"stage={result.get('error_stage', '') or 'complete'} "
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
        "row_index", "target_name", "temperature", "design_seq", "pdb_file",
        "energy_status", "error_stage", "error_message", "error_trace_tail",
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
        is_smoke=is_smoke,
        init_options=init_options,
        disabled_weights=disabled_weights,
    )

    n_ok = int((df["energy_status"] == "ok").sum())
    print("完成：naturalized PyRosetta energy evaluation")
    print(f"rows: {len(df)}, OK: {n_ok}, failed: {len(df)-n_ok}")
    print("outputs:")
    print(result_path.resolve())
    print(temp_path.resolve())
    print(target_path.resolve())
    print(report_path.resolve())
    print(problem_path.resolve())


if __name__ == "__main__":
    main()
