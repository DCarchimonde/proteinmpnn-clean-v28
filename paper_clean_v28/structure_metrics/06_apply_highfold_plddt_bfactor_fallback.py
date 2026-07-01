#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_apply_highfold_plddt_bfactor_fallback.py

Patch complex structure metric outputs so that global pLDDT can use the mean CA
B-factor as a fallback when the PDB COMMENT global pLDDT field is missing.

Why this is needed:
- Some HighFold/AlphaFold-style PDBs store residue-level pLDDT in the B-factor
  column but do not include a global COMMENT plddt header.
- ipTM and inter-PAE cannot be recovered from ordinary ATOM coordinates, so this
  script only fixes pLDDT availability.

Inputs:
    paper_clean_v28_outputs/structure_metrics/complex_rmsd_metrics.csv
    paper_clean_v28_outputs/structure_metrics/complex_best85_highfold_representative.csv

Outputs overwritten/created:
    complex_rmsd_metrics.csv
    complex_rmsd_summary_by_temperature.csv
    complex_rmsd_summary_by_target.csv
    complex_highfold_plddt_fallback_report.txt

New columns:
    highfold_plddt_comment
    highfold_plddt_bfactor_fallback
    highfold_plddt_source
    highfold_plddt_effective

For backward compatibility, highfold_plddt is replaced with highfold_plddt_effective.
"""

from pathlib import Path
import pandas as pd


OUT_DIR = Path("paper_clean_v28_outputs/structure_metrics")
RMSD_PATH = OUT_DIR / "complex_rmsd_metrics.csv"
REP_PATH = OUT_DIR / "complex_best85_highfold_representative.csv"
SUMMARY_TEMP_PATH = OUT_DIR / "complex_rmsd_summary_by_temperature.csv"
SUMMARY_TARGET_PATH = OUT_DIR / "complex_rmsd_summary_by_target.csv"
REPORT_PATH = OUT_DIR / "complex_highfold_plddt_fallback_report.txt"


def norm_temp(x):
    try:
        return f"{float(x):.4f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x).strip()


def make_key(df, target_col, temp_col, seq_col):
    return (
        df[target_col].astype(str).str.upper().str.strip()
        + "|"
        + df[temp_col].map(norm_temp).astype(str)
        + "|"
        + df[seq_col].astype(str).str.strip()
    )


def mean_numeric(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    return float(x.mean())


def median_numeric(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    return float(x.median())


def success_rate_lt(s, threshold):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    return float((x < threshold).mean())


def summarize(df, group_col):
    ok = df[df["rmsd_status"] == "ok"].copy()
    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        gok = ok[ok[group_col] == key]
        rows.append({
            group_col: key,
            "n_rows": len(g),
            "n_ok": len(gok),
            "n_missing_or_failed": len(g) - len(gok),
            "mean_peptide_ca_rmsd_after_receptor_fit": mean_numeric(gok["peptide_ca_rmsd_after_receptor_fit"]),
            "median_peptide_ca_rmsd_after_receptor_fit": median_numeric(gok["peptide_ca_rmsd_after_receptor_fit"]),
            "mean_peptide_backbone_rmsd_after_receptor_fit": mean_numeric(gok["peptide_backbone_rmsd_after_receptor_fit"]),
            "median_peptide_backbone_rmsd_after_receptor_fit": median_numeric(gok["peptide_backbone_rmsd_after_receptor_fit"]),
            "success_rate_ca_rmsd_lt_2": success_rate_lt(gok["peptide_ca_rmsd_after_receptor_fit"], 2.0),
            "success_rate_ca_rmsd_lt_5": success_rate_lt(gok["peptide_ca_rmsd_after_receptor_fit"], 5.0),
            "mean_highfold_plddt": mean_numeric(gok["highfold_plddt"]),
            "mean_highfold_plddt_comment_only": mean_numeric(gok["highfold_plddt_comment"]),
            "mean_highfold_plddt_bfactor_fallback": mean_numeric(gok["highfold_plddt_bfactor_fallback"]),
            "n_highfold_plddt_comment": int(pd.to_numeric(gok["highfold_plddt_comment"], errors="coerce").notna().sum()),
            "n_highfold_plddt_bfactor_fallback": int((gok["highfold_plddt_source"] == "ca_bfactor_mean_fallback").sum()),
            "mean_peptide_receptor_iptm": mean_numeric(gok["peptide_receptor_iptm_mean"]),
            "mean_peptide_receptor_inter_pae": mean_numeric(gok["peptide_receptor_inter_pae_mean"]),
        })
    return pd.DataFrame(rows).sort_values(group_col)


def main():
    if not RMSD_PATH.exists():
        raise FileNotFoundError(RMSD_PATH)
    if not REP_PATH.exists():
        raise FileNotFoundError(REP_PATH)

    rmsd = pd.read_csv(RMSD_PATH)
    rep = pd.read_csv(REP_PATH)

    rmsd["_merge_key"] = make_key(rmsd, "target_name", "temperature", "design_seq")
    rep["_merge_key"] = make_key(rep, "target_name", "temperature", "design_peptide_seq")

    rep_cols = rep[["_merge_key", "highfold_pdb_ca_bfactor_mean"]].copy()
    rep_cols = rep_cols.drop_duplicates("_merge_key", keep="first")
    rmsd = rmsd.merge(rep_cols, on="_merge_key", how="left", suffixes=("", "_from_rep"), validate="m:1")

    comment = pd.to_numeric(rmsd.get("highfold_plddt"), errors="coerce")
    bfac = pd.to_numeric(rmsd.get("highfold_pdb_ca_bfactor_mean_from_rep"), errors="coerce")

    rmsd["highfold_plddt_comment"] = comment
    rmsd["highfold_plddt_bfactor_fallback"] = bfac
    rmsd["highfold_plddt_effective"] = comment.where(comment.notna(), bfac)
    rmsd["highfold_plddt_source"] = "missing"
    rmsd.loc[comment.notna(), "highfold_plddt_source"] = "comment_global_plddt"
    rmsd.loc[comment.isna() & bfac.notna(), "highfold_plddt_source"] = "ca_bfactor_mean_fallback"

    # Backward compatible field used by existing validation/report scripts.
    rmsd["highfold_plddt"] = rmsd["highfold_plddt_effective"]

    rmsd = rmsd.drop(columns=[c for c in ["_merge_key", "highfold_pdb_ca_bfactor_mean_from_rep"] if c in rmsd.columns])
    rmsd.to_csv(RMSD_PATH, index=False, encoding="utf-8")

    summary_temp = summarize(rmsd, "temperature")
    summary_target = summarize(rmsd, "target_name")
    summary_temp.to_csv(SUMMARY_TEMP_PATH, index=False, encoding="utf-8")
    summary_target.to_csv(SUMMARY_TARGET_PATH, index=False, encoding="utf-8")

    ok = rmsd[rmsd["rmsd_status"] == "ok"].copy()
    n_ok = len(ok)
    n_comment = int(pd.to_numeric(ok["highfold_plddt_comment"], errors="coerce").notna().sum())
    n_fallback = int((ok["highfold_plddt_source"] == "ca_bfactor_mean_fallback").sum())
    n_effective = int(pd.to_numeric(ok["highfold_plddt_effective"], errors="coerce").notna().sum())
    n_missing = n_ok - n_effective

    lines = []
    lines.append("===== HIGHFOLD PLDDT B-FACTOR FALLBACK REPORT =====")
    lines.append(f"OK rows: {n_ok}")
    lines.append(f"COMMENT global pLDDT available: {n_comment}/{n_ok}")
    lines.append(f"CA B-factor fallback used: {n_fallback}/{n_ok}")
    lines.append(f"Effective pLDDT available: {n_effective}/{n_ok}")
    lines.append(f"Effective pLDDT missing: {n_missing}/{n_ok}")
    lines.append("")
    lines.append("pLDDT source counts:")
    lines.append(ok["highfold_plddt_source"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("Effective pLDDT by temperature:")
    lines.append(
        ok.groupby("temperature").agg(
            n_rows=("highfold_plddt_effective", "size"),
            n_effective_plddt=("highfold_plddt_effective", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            n_comment=("highfold_plddt_source", lambda s: int((s == "comment_global_plddt").sum())),
            n_bfactor_fallback=("highfold_plddt_source", lambda s: int((s == "ca_bfactor_mean_fallback").sum())),
            mean_highfold_plddt=("highfold_plddt_effective", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        ).to_string()
    )
    lines.append("")
    lines.append("Notes:")
    lines.append("- highfold_plddt now means effective pLDDT: COMMENT global pLDDT if present, otherwise mean CA B-factor.")
    lines.append("- ipTM and inter-PAE are not patched because they cannot be recovered from ordinary ATOM coordinates.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("完成：HighFold pLDDT fallback patch")
    print("report:", REPORT_PATH)
    print(f"effective pLDDT: {n_effective}/{n_ok}")
    print(f"comment: {n_comment}/{n_ok}, bfactor fallback: {n_fallback}/{n_ok}, missing: {n_missing}/{n_ok}")


if __name__ == "__main__":
    main()
