#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_compute_methylation_site_rmsd.py

Compute position-stratified structural deviations for best85 complex designs.
Lowercase residues in design_seq are treated as methylation-marked positions.

Important scientific scope:
- This script computes CA/backbone deviations at methylation-marked positions.
- It does NOT compute the geometry of an explicit N-methyl carbon atom.
- It does NOT claim all-atom RMSD for chemically modified residues.

Alignment definitions:
1. receptor-fit: predicted receptor is aligned to the native receptor, then peptide
   position-wise deviations are measured in the receptor frame;
2. peptide-self-fit: predicted peptide CA atoms are superposed to native peptide CA
   atoms, then methylated/non-methylated position deviations are measured.

Outputs:
- complex_methylation_site_rmsd_by_design.csv
- complex_methylation_site_rmsd_by_residue.csv
- complex_methylation_site_rmsd_summary_by_temperature.csv
- complex_methylation_site_rmsd_report.txt
- complex_methylation_site_rmsd_problem_rows.csv
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT_DIR = ROOT / "paper_clean_v28_outputs" / "structure_metrics"
AUDIT_PATH = OUT_DIR / "complex_chain_mapping_audit.csv"
NATIVE_PATH = ROOT / "17_complexes_native.jsonl"
HELPER_PATH = HERE / "04_compute_complex_rmsd.py"

BY_DESIGN_PATH = OUT_DIR / "complex_methylation_site_rmsd_by_design.csv"
BY_RESIDUE_PATH = OUT_DIR / "complex_methylation_site_rmsd_by_residue.csv"
SUMMARY_TEMP_PATH = OUT_DIR / "complex_methylation_site_rmsd_summary_by_temperature.csv"
REPORT_PATH = OUT_DIR / "complex_methylation_site_rmsd_report.txt"
PROBLEM_PATH = OUT_DIR / "complex_methylation_site_rmsd_problem_rows.csv"

EXPECTED_ROWS = 85
BACKBONE_ATOMS = ["N", "CA", "C"]


def load_helper_module():
    spec = importlib.util.spec_from_file_location("complex_rmsd_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_temp(x) -> str:
    try:
        return f"{float(x):.4f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x).strip()


def finite_xyz(x) -> bool:
    if x is None:
        return False
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return False
    return arr.shape == (3,) and np.isfinite(arr).all()


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def rms_from_distances(values: List[float]) -> Optional[float]:
    xs = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if xs.size == 0:
        return None
    return float(np.sqrt(np.mean(xs * xs)))


def mean_or_none(values: List[float]) -> Optional[float]:
    xs = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if xs.size == 0:
        return None
    return float(np.mean(xs))


def median_or_none(values: List[float]) -> Optional[float]:
    xs = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if xs.size == 0:
        return None
    return float(np.median(xs))


def fmt(x: Optional[float], ndigits: int = 6):
    if x is None:
        return ""
    try:
        if not np.isfinite(float(x)):
            return ""
    except Exception:
        return ""
    return round(float(x), ndigits)


def group_rms(df: pd.DataFrame, column: str, mask: pd.Series) -> Optional[float]:
    values = pd.to_numeric(df.loc[mask, column], errors="coerce").dropna().tolist()
    return rms_from_distances(values)


def group_mean(df: pd.DataFrame, column: str, mask: pd.Series) -> Optional[float]:
    values = pd.to_numeric(df.loc[mask, column], errors="coerce").dropna().tolist()
    return mean_or_none(values)


def summarize_temperature(residue_df: pd.DataFrame, design_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for temp in sorted(design_df["temperature"].astype(str).unique(), key=lambda x: float(x)):
        d = design_df[design_df["temperature"].astype(str) == str(temp)]
        r = residue_df[residue_df["temperature"].astype(str) == str(temp)]
        methyl = r["is_methylated"] == 1
        nonmethyl = r["is_methylated"] == 0

        rows.append({
            "temperature": temp,
            "n_designs": len(d),
            "n_designs_ok": int((d["site_rmsd_status"] == "ok").sum()),
            "n_methyl_positions": int(methyl.sum()),
            "n_nonmethyl_positions": int(nonmethyl.sum()),
            "n_methyl_ca_pairs": int(pd.to_numeric(r.loc[methyl, "ca_distance_after_receptor_fit"], errors="coerce").notna().sum()),
            "n_nonmethyl_ca_pairs": int(pd.to_numeric(r.loc[nonmethyl, "ca_distance_after_receptor_fit"], errors="coerce").notna().sum()),
            "pooled_methyl_ca_rmsd_after_receptor_fit": fmt(group_rms(r, "ca_distance_after_receptor_fit", methyl)),
            "pooled_nonmethyl_ca_rmsd_after_receptor_fit": fmt(group_rms(r, "ca_distance_after_receptor_fit", nonmethyl)),
            "pooled_methyl_backbone_rmsd_after_receptor_fit": fmt(group_rms(r, "backbone_residue_rmsd_after_receptor_fit", methyl)),
            "pooled_nonmethyl_backbone_rmsd_after_receptor_fit": fmt(group_rms(r, "backbone_residue_rmsd_after_receptor_fit", nonmethyl)),
            "pooled_methyl_ca_rmsd_after_peptide_self_fit": fmt(group_rms(r, "ca_distance_after_peptide_self_fit", methyl)),
            "pooled_nonmethyl_ca_rmsd_after_peptide_self_fit": fmt(group_rms(r, "ca_distance_after_peptide_self_fit", nonmethyl)),
            "mean_design_methyl_ca_rmsd_after_receptor_fit": fmt(pd.to_numeric(d["methyl_ca_rmsd_after_receptor_fit"], errors="coerce").mean()),
            "median_design_methyl_ca_rmsd_after_receptor_fit": fmt(pd.to_numeric(d["methyl_ca_rmsd_after_receptor_fit"], errors="coerce").median()),
            "mean_design_nonmethyl_ca_rmsd_after_receptor_fit": fmt(pd.to_numeric(d["nonmethyl_ca_rmsd_after_receptor_fit"], errors="coerce").mean()),
            "median_design_nonmethyl_ca_rmsd_after_receptor_fit": fmt(pd.to_numeric(d["nonmethyl_ca_rmsd_after_receptor_fit"], errors="coerce").median()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    helper = load_helper_module()

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(AUDIT_PATH)
    if not NATIVE_PATH.exists():
        raise FileNotFoundError(NATIVE_PATH)

    audit_df = pd.read_csv(AUDIT_PATH)
    native_by_target = helper.load_native_structures(NATIVE_PATH)

    design_rows: List[Dict] = []
    residue_rows: List[Dict] = []
    problems: List[Dict] = []

    for row_index, audit in audit_df.iterrows():
        target = str(audit.get("target_name", "")).upper().strip()
        temp = norm_temp(audit.get("temperature", ""))
        design_seq = str(audit.get("design_seq", "")).strip()
        pdb_path = str(audit.get("pdb_path", "")).strip()
        pred_pep_chain = str(audit.get("predicted_peptide_chain", "")).strip()
        native_pep_chain = str(audit.get("native_peptide_chain", "")).strip()
        chain_mapping_status = str(audit.get("chain_mapping_status", "")).strip()
        design_len = audit.get("design_len", len(design_seq))

        out = {
            "row_index": row_index,
            "target_name": target,
            "temperature": temp,
            "design_seq": design_seq,
            "design_len": len(design_seq),
            "pdb_file": audit.get("pdb_file", ""),
            "pdb_path": pdb_path,
            "predicted_peptide_chain": pred_pep_chain,
            "native_peptide_chain": native_pep_chain,
            "n_methyl_positions_total": sum(1 for aa in design_seq if aa.islower()),
            "n_nonmethyl_positions_total": sum(1 for aa in design_seq if not aa.islower()),
        }

        try:
            if chain_mapping_status != "ok":
                raise RuntimeError(f"chain_mapping_status={chain_mapping_status}")
            if not Path(pdb_path).exists():
                raise FileNotFoundError(pdb_path)

            pred_struct = helper.parse_pdb_structure(pdb_path)
            native_struct = native_by_target.get(target, {})
            if not native_struct:
                raise RuntimeError("native_target_not_found")
            if pred_pep_chain not in pred_struct:
                raise RuntimeError("predicted_peptide_chain_not_found")
            if native_pep_chain not in native_struct:
                raise RuntimeError("native_peptide_chain_not_found")

            mapping, fit_P, fit_Q, receptor_alignment_detail = helper.infer_best_receptor_mapping_and_pairs(
                pred_struct=pred_struct,
                native_struct=native_struct,
                pred_pep_chain=pred_pep_chain,
                native_pep_chain=native_pep_chain,
                design_len=design_len,
            )
            if len(fit_P) < 3:
                raise RuntimeError("insufficient_receptor_ca_fit_pairs")

            R_rec, t_rec = helper.kabsch_fit(np.asarray(fit_P), np.asarray(fit_Q))
            pred_pep = pred_struct[pred_pep_chain]
            native_pep = native_struct[native_pep_chain]
            n_pos = min(len(design_seq), len(pred_pep), len(native_pep))
            if n_pos == 0:
                raise RuntimeError("no_comparable_peptide_positions")

            all_pred_ca = []
            all_native_ca = []
            for pos in range(n_pos):
                pa = pred_pep[pos].get("atoms", {})
                qa = native_pep[pos].get("atoms", {})
                if finite_xyz(pa.get("CA")) and finite_xyz(qa.get("CA")):
                    all_pred_ca.append(pa["CA"])
                    all_native_ca.append(qa["CA"])

            R_pep = t_pep = None
            if len(all_pred_ca) >= 3:
                R_pep, t_pep = helper.kabsch_fit(np.asarray(all_pred_ca), np.asarray(all_native_ca))

            out["receptor_alignment_detail"] = receptor_alignment_detail
            out["n_positions_compared"] = n_pos
            out["predicted_peptide_length"] = len(pred_pep)
            out["native_peptide_length"] = len(native_pep)
            out["lengths_all_equal"] = int(len(design_seq) == len(pred_pep) == len(native_pep))

            methyl_ca_rec: List[float] = []
            nonmethyl_ca_rec: List[float] = []
            methyl_bb_rec_atoms: List[float] = []
            nonmethyl_bb_rec_atoms: List[float] = []
            methyl_ca_self: List[float] = []
            nonmethyl_ca_self: List[float] = []
            methyl_bb_self_atoms: List[float] = []
            nonmethyl_bb_self_atoms: List[float] = []

            for pos in range(n_pos):
                aa = design_seq[pos]
                is_methyl = int(aa.islower())
                pa = pred_pep[pos].get("atoms", {})
                qa = native_pep[pos].get("atoms", {})

                ca_rec = None
                ca_self = None
                if finite_xyz(pa.get("CA")) and finite_xyz(qa.get("CA")):
                    pred_ca_rec = helper.apply_transform(np.asarray([pa["CA"]]), R_rec, t_rec)[0]
                    ca_rec = distance(pred_ca_rec, qa["CA"])
                    if R_pep is not None and t_pep is not None:
                        pred_ca_self = helper.apply_transform(np.asarray([pa["CA"]]), R_pep, t_pep)[0]
                        ca_self = distance(pred_ca_self, qa["CA"])

                bb_rec_distances: List[float] = []
                bb_self_distances: List[float] = []
                common_bb_atoms = []
                for atom in BACKBONE_ATOMS:
                    if finite_xyz(pa.get(atom)) and finite_xyz(qa.get(atom)):
                        common_bb_atoms.append(atom)
                        pred_atom_rec = helper.apply_transform(np.asarray([pa[atom]]), R_rec, t_rec)[0]
                        bb_rec_distances.append(distance(pred_atom_rec, qa[atom]))
                        if R_pep is not None and t_pep is not None:
                            pred_atom_self = helper.apply_transform(np.asarray([pa[atom]]), R_pep, t_pep)[0]
                            bb_self_distances.append(distance(pred_atom_self, qa[atom]))

                bb_rec_rmsd = rms_from_distances(bb_rec_distances)
                bb_self_rmsd = rms_from_distances(bb_self_distances)

                residue_rows.append({
                    "row_index": row_index,
                    "target_name": target,
                    "temperature": temp,
                    "design_seq": design_seq,
                    "position_1based": pos + 1,
                    "design_token": aa,
                    "design_natural_aa": aa.upper(),
                    "is_methylated": is_methyl,
                    "predicted_resname": pred_pep[pos].get("resname", ""),
                    "native_resname": native_pep[pos].get("resname", ""),
                    "ca_pair_available": int(ca_rec is not None),
                    "n_common_backbone_atoms": len(common_bb_atoms),
                    "common_backbone_atoms": ";".join(common_bb_atoms),
                    "ca_distance_after_receptor_fit": fmt(ca_rec),
                    "backbone_residue_rmsd_after_receptor_fit": fmt(bb_rec_rmsd),
                    "ca_distance_after_peptide_self_fit": fmt(ca_self),
                    "backbone_residue_rmsd_after_peptide_self_fit": fmt(bb_self_rmsd),
                })

                if is_methyl:
                    if ca_rec is not None:
                        methyl_ca_rec.append(ca_rec)
                    methyl_bb_rec_atoms.extend(bb_rec_distances)
                    if ca_self is not None:
                        methyl_ca_self.append(ca_self)
                    methyl_bb_self_atoms.extend(bb_self_distances)
                else:
                    if ca_rec is not None:
                        nonmethyl_ca_rec.append(ca_rec)
                    nonmethyl_bb_rec_atoms.extend(bb_rec_distances)
                    if ca_self is not None:
                        nonmethyl_ca_self.append(ca_self)
                    nonmethyl_bb_self_atoms.extend(bb_self_distances)

            out.update({
                "n_methyl_ca_pairs": len(methyl_ca_rec),
                "n_nonmethyl_ca_pairs": len(nonmethyl_ca_rec),
                "n_methyl_backbone_atom_pairs": len(methyl_bb_rec_atoms),
                "n_nonmethyl_backbone_atom_pairs": len(nonmethyl_bb_rec_atoms),
                "methyl_ca_rmsd_after_receptor_fit": fmt(rms_from_distances(methyl_ca_rec)),
                "nonmethyl_ca_rmsd_after_receptor_fit": fmt(rms_from_distances(nonmethyl_ca_rec)),
                "methyl_backbone_rmsd_after_receptor_fit": fmt(rms_from_distances(methyl_bb_rec_atoms)),
                "nonmethyl_backbone_rmsd_after_receptor_fit": fmt(rms_from_distances(nonmethyl_bb_rec_atoms)),
                "methyl_ca_rmsd_after_peptide_self_fit": fmt(rms_from_distances(methyl_ca_self)),
                "nonmethyl_ca_rmsd_after_peptide_self_fit": fmt(rms_from_distances(nonmethyl_ca_self)),
                "methyl_backbone_rmsd_after_peptide_self_fit": fmt(rms_from_distances(methyl_bb_self_atoms)),
                "nonmethyl_backbone_rmsd_after_peptide_self_fit": fmt(rms_from_distances(nonmethyl_bb_self_atoms)),
                "site_rmsd_status": "ok",
            })

            if out["n_methyl_positions_total"] == 0:
                out["site_rmsd_note"] = "no_methylated_positions_in_design"
            elif out["n_methyl_ca_pairs"] == 0:
                out["site_rmsd_note"] = "methylated_positions_exist_but_no_ca_pairs"
            else:
                out["site_rmsd_note"] = ""

        except Exception as exc:
            out["site_rmsd_status"] = "failed"
            out["site_rmsd_note"] = repr(exc)
            problems.append({
                "row_index": row_index,
                "target_name": target,
                "temperature": temp,
                "design_seq": design_seq,
                "problem": repr(exc),
            })

        design_rows.append(out)

    design_df = pd.DataFrame(design_rows)
    residue_df = pd.DataFrame(residue_rows)
    design_df.to_csv(BY_DESIGN_PATH, index=False, encoding="utf-8")
    residue_df.to_csv(BY_RESIDUE_PATH, index=False, encoding="utf-8")

    summary_df = summarize_temperature(residue_df, design_df)
    summary_df.to_csv(SUMMARY_TEMP_PATH, index=False, encoding="utf-8")
    pd.DataFrame(problems, columns=["row_index", "target_name", "temperature", "design_seq", "problem"]).to_csv(
        PROBLEM_PATH, index=False, encoding="utf-8"
    )

    n_ok = int((design_df["site_rmsd_status"] == "ok").sum())
    n_failed = len(design_df) - n_ok
    n_designs_with_methyl = int((design_df["n_methyl_positions_total"] > 0).sum())
    n_methyl_positions = int(pd.to_numeric(design_df["n_methyl_positions_total"], errors="coerce").fillna(0).sum())
    n_nonmethyl_positions = int(pd.to_numeric(design_df["n_nonmethyl_positions_total"], errors="coerce").fillna(0).sum())
    n_length_mismatch = int((pd.to_numeric(design_df.get("lengths_all_equal"), errors="coerce") == 0).sum())

    methyl_mask = residue_df["is_methylated"] == 1
    nonmethyl_mask = residue_df["is_methylated"] == 0
    pooled_methyl_ca_rec = group_rms(residue_df, "ca_distance_after_receptor_fit", methyl_mask)
    pooled_nonmethyl_ca_rec = group_rms(residue_df, "ca_distance_after_receptor_fit", nonmethyl_mask)
    pooled_methyl_ca_self = group_rms(residue_df, "ca_distance_after_peptide_self_fit", methyl_mask)
    pooled_nonmethyl_ca_self = group_rms(residue_df, "ca_distance_after_peptide_self_fit", nonmethyl_mask)

    report = []
    report.append("===== METHYLATION-SITE RMSD QUALITY REPORT =====")
    report.append(f"Expected best85 rows: {EXPECTED_ROWS}")
    report.append(f"Observed design rows: {len(design_df)}")
    report.append(f"Status OK: {n_ok}")
    report.append(f"Status failed: {n_failed}")
    report.append(f"Designs with >=1 lowercase methylation position: {n_designs_with_methyl}/{len(design_df)}")
    report.append(f"Total methylation-marked positions: {n_methyl_positions}")
    report.append(f"Total non-methylated positions: {n_nonmethyl_positions}")
    report.append(f"Length-mismatch rows: {n_length_mismatch}")
    report.append("")
    report.append("===== POOLED POSITION-LEVEL RESULTS =====")
    report.append(f"Methyl-position CA RMSD after receptor fit: {fmt(pooled_methyl_ca_rec)}")
    report.append(f"Non-methyl-position CA RMSD after receptor fit: {fmt(pooled_nonmethyl_ca_rec)}")
    report.append(f"Methyl-position CA RMSD after peptide self-fit: {fmt(pooled_methyl_ca_self)}")
    report.append(f"Non-methyl-position CA RMSD after peptide self-fit: {fmt(pooled_nonmethyl_ca_self)}")
    report.append("")
    report.append("===== SUMMARY BY TEMPERATURE =====")
    report.append(summary_df.to_string(index=False))
    report.append("")
    report.append("===== INTERPRETATION NOTES =====")
    report.append("- Lowercase design tokens define methylation-marked positions.")
    report.append("- These are position-stratified CA/backbone deviations, not explicit N-methyl-atom RMSD.")
    report.append("- Receptor-fit metrics measure peptide placement in the binding frame.")
    report.append("- Peptide-self-fit metrics measure peptide internal shape after removing rigid-body placement.")
    report.append("- Methylated and non-methylated positions are not randomized groups; comparisons are descriptive.")
    report.append("")

    quality_pass = len(design_df) == EXPECTED_ROWS and n_failed == 0 and n_methyl_positions > 0
    report.append("QUALITY GATE: PASS" if quality_pass else "QUALITY GATE: FAIL")
    report.append(f"PROBLEMS: {len(problems)}")
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("完成：methylation-site RMSD evaluation")
    print(f"design rows: {len(design_df)}, OK: {n_ok}, failed: {n_failed}")
    print(f"methylation-marked positions: {n_methyl_positions}")
    print("quality gate:", "PASS" if quality_pass else "FAIL")
    print("outputs:")
    print(BY_DESIGN_PATH)
    print(BY_RESIDUE_PATH)
    print(SUMMARY_TEMP_PATH)
    print(REPORT_PATH)
    print(PROBLEM_PATH)
    return 0 if quality_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
