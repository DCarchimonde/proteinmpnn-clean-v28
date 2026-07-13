#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_apply_highfold_plddt_bfactor_fallback.py

Audit and attach the mean CA B-factor from each HighFold PDB as a complete
per-structure confidence proxy.

Scientific handling:
- COMMENT global pLDDT and mean CA B-factor are kept as separate metrics.
- They are not mixed into a single column because their numerical definitions can
  differ substantially for complexes.
- The mean CA B-factor is labelled as a pLDDT proxy unless the generating pipeline
  is independently confirmed to encode pLDDT in the PDB B-factor field.
- ipTM and inter-PAE cannot be recovered from ordinary ATOM coordinates.

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
    highfold_ca_bfactor_mean
    highfold_ca_bfactor_available
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
    return "" if len(x) == 0 else float(x.mean())


def median_numeric(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return "" if len(x) == 0 else float(x.median())


def success_rate_lt(s, threshold):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return "" if len(x) == 0 else float((x < threshold).mean())


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
            "mean_highfold_plddt_comment": mean_numeric(gok["highfold_plddt_comment"]),
            "n_highfold_plddt_comment": int(pd.to_numeric(gok["highfold_plddt_comment"], errors="coerce").notna().sum()),
            "mean_highfold_ca_bfactor_mean": mean_numeric(gok["highfold_ca_bfactor_mean"]),
            "n_highfold_ca_bfactor_mean": int(pd.to_numeric(gok["highfold_ca_bfactor_mean"], errors="coerce").notna().sum()),
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

    rmsd_key = make_key(rmsd, "target_name", "temperature", "design_seq")
    rep_key = make_key(rep, "target_name", "temperature", "design_peptide_seq")

    rep_bfactor = pd.to_numeric(rep["highfold_pdb_ca_bfactor_mean"], errors="coerce")
    bfactor_map = pd.Series(rep_bfactor.values, index=rep_key).groupby(level=0).first()

    if "highfold_plddt_comment" in rmsd.columns:
        comment = pd.to_numeric(rmsd["highfold_plddt_comment"], errors="coerce")
    else:
        comment = pd.to_numeric(rmsd["highfold_plddt"], errors="coerce")

    bfac = pd.to_numeric(rmsd_key.map(bfactor_map), errors="coerce")

    rmsd["highfold_plddt_comment"] = comment
    rmsd["highfold_ca_bfactor_mean"] = bfac
    rmsd["highfold_ca_bfactor_available"] = bfac.notna().astype(int)

    # Keep the historical highfold_plddt column as COMMENT global pLDDT only.
    # Do not overwrite it with a different quantity.
    rmsd["highfold_plddt"] = comment

    rmsd.to_csv(RMSD_PATH, index=False, encoding="utf-8")

    summary_temp = summarize(rmsd, "temperature")
    summary_target = summarize(rmsd, "target_name")
    summary_temp.to_csv(SUMMARY_TEMP_PATH, index=False, encoding="utf-8")
    summary_target.to_csv(SUMMARY_TARGET_PATH, index=False, encoding="utf-8")

    ok = rmsd[rmsd["rmsd_status"] == "ok"].copy()
    n_ok = len(ok)
    n_comment = int(pd.to_numeric(ok["highfold_plddt_comment"], errors="coerce").notna().sum())
    n_bfactor = int(pd.to_numeric(ok["highfold_ca_bfactor_mean"], errors="coerce").notna().sum())

    lines = []
    lines.append("===== HIGHFOLD PLDDT / CA B-FACTOR AUDIT =====")
    lines.append(f"OK rows: {n_ok}")
    lines.append(f"COMMENT global pLDDT available: {n_comment}/{n_ok}")
    lines.append(f"Mean CA B-factor available: {n_bfactor}/{n_ok}")
    lines.append(f"Mean CA B-factor missing: {n_ok - n_bfactor}/{n_ok}")
    lines.append("")
    lines.append("Mean CA B-factor by temperature:")
    lines.append(
        ok.groupby("temperature").agg(
            n_rows=("highfold_ca_bfactor_mean", "size"),
            n_ca_bfactor=("highfold_ca_bfactor_mean", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            mean_ca_bfactor=("highfold_ca_bfactor_mean", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            n_comment_plddt=("highfold_plddt_comment", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            mean_comment_plddt=("highfold_plddt_comment", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        ).to_string()
    )
    lines.append("")
    lines.append("Notes:")
    lines.append("- COMMENT global pLDDT and mean CA B-factor are reported separately and are not mixed.")
    lines.append("- Mean CA B-factor may be used as a per-residue pLDDT proxy only with an explicit label or after confirming the HighFold output convention.")
    lines.append("- ipTM and inter-PAE are not reconstructed from coordinates.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("完成：HighFold pLDDT / CA B-factor audit")
    print("report:", REPORT_PATH)
    print(f"COMMENT global pLDDT: {n_comment}/{n_ok}")
    print(f"mean CA B-factor: {n_bfactor}/{n_ok}")


if __name__ == "__main__":
    main()
