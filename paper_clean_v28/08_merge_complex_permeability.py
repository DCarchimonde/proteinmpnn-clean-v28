#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_merge_complex_permeability.py

Merge complex permeability predictions returned by the external permeability model
with clean generated FASTA metrics and complex structure RMSD metrics.

Expected local input directory, not committed to Git:
    raw_external/pdb_permeability_v20260624/permeability_complex/

Expected permeability CSV columns:
    id, fasta, methy_index, permeability_pred
with optional leading index columns such as Unnamed: 0.

The id is expected to look like:
    1sfi_0_RcGrGqcrQcrQGC_model
where:
    target_name = 1sfi
    record_index = 0
    design_seq = RcGrGqcrQcrQGC

Outputs:
    paper_clean_v28_outputs/permeability/complex_permeability_raw_merged.csv
    paper_clean_v28_outputs/permeability/complex_permeability_all_designs.csv
    paper_clean_v28_outputs/permeability/complex_permeability_best85.csv
    paper_clean_v28_outputs/permeability/complex_structure_permeability_merged.csv
    paper_clean_v28_outputs/permeability/complex_permeability_summary_by_temperature.csv
    paper_clean_v28_outputs/permeability/complex_permeability_summary_by_target.csv
    paper_clean_v28_outputs/permeability/complex_permeability_merge_report.txt
    paper_clean_v28_outputs/permeability/complex_permeability_unmatched_rows.csv
"""

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PERM_DIR = Path("raw_external/pdb_permeability_v20260624/permeability_complex")
DEFAULT_ALL_DESIGNS = Path("paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv")
DEFAULT_BEST85 = Path("paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv")
DEFAULT_RMSD = Path("paper_clean_v28_outputs/structure_metrics/complex_rmsd_metrics.csv")
DEFAULT_OUT_DIR = Path("paper_clean_v28_outputs/permeability")

TEMP_FROM_FILE = {
    "t001": 0.01,
    "t01": 0.10,
    "t02": 0.20,
    "t03": 0.30,
    "t05": 0.50,
}

NATURAL_AA = set("ACDEFGHIKLMNPQRSTVWYX")


def naturalize(seq):
    if seq is None or (isinstance(seq, float) and math.isnan(seq)):
        return ""
    seq = str(seq).strip().upper()
    return "".join(ch for ch in seq if ch in NATURAL_AA)


def norm_design_seq(seq):
    if seq is None or (isinstance(seq, float) and math.isnan(seq)):
        return ""
    return str(seq).strip()


def temp_key(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        s = str(x).strip()
        if s in TEMP_FROM_FILE:
            return f"{TEMP_FROM_FILE[s]:.2f}"
        return s


def target_key(x):
    return str(x).strip().upper()


def parse_temp_from_filename(path):
    name = Path(path).stem.lower()
    for token, temp in TEMP_FROM_FILE.items():
        if re.search(rf"(?:^|_){re.escape(token)}(?:$|_)", name):
            return temp
    # fallback: last token after underscore
    last = name.split("_")[-1]
    if last in TEMP_FROM_FILE:
        return TEMP_FROM_FILE[last]
    raise ValueError(f"Cannot infer temperature from file name: {path}")


def parse_permeability_id(pid):
    """Parse target, record index, design sequence from external permeability id."""
    s = str(pid).strip()
    s = re.sub(r"\.pdb$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_model$", "", s, flags=re.IGNORECASE)
    m = re.match(r"^(?P<target>[^_]+)_(?P<record_index>\d+)_(?P<design_seq>.+)$", s)
    if not m:
        return None, None, None
    return m.group("target"), int(m.group("record_index")), m.group("design_seq")


def read_permeability_folder(perm_dir):
    rows = []
    files = sorted(Path(perm_dir).glob("*.csv"))
    for path in files:
        temp = parse_temp_from_filename(path)
        df = pd.read_csv(path)
        # Drop anonymous index columns if present.
        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        required = {"id", "fasta", "methy_index", "permeability_pred"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for _, r in df.iterrows():
            target, record_index, design_seq = parse_permeability_id(r["id"])
            rows.append({
                "permeability_source_file": str(path),
                "permeability_id": str(r["id"]),
                "target_name": str(target).upper() if target is not None else "",
                "temperature": float(temp),
                "record_index": record_index,
                "design_seq_from_id": norm_design_seq(design_seq),
                "design_natural_seq_from_id": naturalize(design_seq),
                "permeability_fasta": naturalize(r["fasta"]),
                "permeability_methy_index": str(r["methy_index"]),
                "permeability_pred": pd.to_numeric(r["permeability_pred"], errors="coerce"),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["target_key"] = out["target_name"].map(target_key)
        out["temperature_key"] = out["temperature"].map(temp_key)
        out["record_index_key"] = out["record_index"].astype("Int64").astype(str)
        out["design_seq_key"] = out["design_seq_from_id"].map(norm_design_seq)
        out["design_natural_key"] = out["design_natural_seq_from_id"].map(naturalize)
        out["merge_key_full"] = (
            out["target_key"] + "|" + out["temperature_key"] + "|" +
            out["record_index_key"] + "|" + out["design_seq_key"]
        )
        out["merge_key_design"] = (
            out["target_key"] + "|" + out["temperature_key"] + "|" + out["design_seq_key"]
        )
        out["merge_key_design_natural"] = (
            out["target_key"] + "|" + out["temperature_key"] + "|" + out["design_natural_key"]
        )
        out["permeability_log10"] = np.log10(pd.to_numeric(out["permeability_pred"], errors="coerce"))
    return out


def add_design_keys(df):
    out = df.copy()
    out["target_key"] = out["target_name"].map(target_key)
    out["temperature_key"] = out["temperature"].map(temp_key)
    if "record_index" in out.columns:
        out["record_index_key"] = pd.to_numeric(out["record_index"], errors="coerce").astype("Int64").astype(str)
    else:
        out["record_index_key"] = ""
    out["design_seq_key"] = out["design_seq"].map(norm_design_seq)
    if "design_natural_seq" in out.columns:
        out["design_natural_key"] = out["design_natural_seq"].map(naturalize)
    else:
        out["design_natural_key"] = out["design_seq"].map(naturalize)
    out["merge_key_full"] = (
        out["target_key"] + "|" + out["temperature_key"] + "|" +
        out["record_index_key"] + "|" + out["design_seq_key"]
    )
    out["merge_key_design"] = out["target_key"] + "|" + out["temperature_key"] + "|" + out["design_seq_key"]
    out["merge_key_design_natural"] = out["target_key"] + "|" + out["temperature_key"] + "|" + out["design_natural_key"]
    return out


def add_structure_keys(df):
    out = df.copy()
    out["target_key"] = out["target_name"].map(target_key)
    out["temperature_key"] = out["temperature"].map(temp_key)
    out["design_seq_key"] = out["design_seq"].map(norm_design_seq)
    out["design_natural_key"] = out["design_seq"].map(naturalize)
    out["merge_key_design"] = out["target_key"] + "|" + out["temperature_key"] + "|" + out["design_seq_key"]
    out["merge_key_design_natural"] = out["target_key"] + "|" + out["temperature_key"] + "|" + out["design_natural_key"]
    return out


def prepare_perm_for_design_merge(perm):
    cols = [
        "merge_key_full",
        "permeability_id",
        "permeability_source_file",
        "permeability_fasta",
        "permeability_methy_index",
        "permeability_pred",
        "permeability_log10",
        "design_seq_from_id",
        "design_natural_seq_from_id",
    ]
    dup = perm[perm.duplicated("merge_key_full", keep=False)].copy()
    if not dup.empty:
        # Full key should be unique; keep first but report duplicates later.
        perm = perm.sort_values("permeability_pred", ascending=False).drop_duplicates("merge_key_full", keep="first")
    return perm[cols], dup


def prepare_perm_for_design_seq_merge(perm):
    cols = [
        "merge_key_design",
        "permeability_id",
        "permeability_source_file",
        "permeability_fasta",
        "permeability_methy_index",
        "permeability_pred",
        "permeability_log10",
        "design_seq_from_id",
        "design_natural_seq_from_id",
    ]
    tmp = perm.copy()
    # For structure/best85, one target/temp/design_seq may have multiple PDB predictions.
    # Use the maximum permeability_pred as conservative high-permeability representative,
    # and keep counts for transparency.
    agg = tmp.groupby("merge_key_design", dropna=False).agg(
        permeability_pred=("permeability_pred", "max"),
        permeability_log10=("permeability_log10", "max"),
        permeability_match_count=("permeability_id", "count"),
        permeability_id=("permeability_id", lambda s: ";".join(map(str, s.head(10)))),
        permeability_source_file=("permeability_source_file", lambda s: ";".join(sorted(set(map(str, s)))[:10])),
        permeability_fasta=("permeability_fasta", "first"),
        permeability_methy_index=("permeability_methy_index", "first"),
        design_seq_from_id=("design_seq_from_id", "first"),
        design_natural_seq_from_id=("design_natural_seq_from_id", "first"),
    ).reset_index()
    return agg


def summarize_by(df, group_col):
    d = df.copy()
    d["permeability_pred"] = pd.to_numeric(d["permeability_pred"], errors="coerce")
    d["permeability_log10"] = pd.to_numeric(d["permeability_log10"], errors="coerce")
    if "natural_aa_recovery" in d.columns:
        d["natural_aa_recovery"] = pd.to_numeric(d["natural_aa_recovery"], errors="coerce")
    if "peptide_ca_rmsd_after_receptor_fit" in d.columns:
        d["peptide_ca_rmsd_after_receptor_fit"] = pd.to_numeric(d["peptide_ca_rmsd_after_receptor_fit"], errors="coerce")

    agg = {
        "n_rows": ("permeability_pred", "size"),
        "n_with_permeability": ("permeability_pred", lambda s: int(s.notna().sum())),
        "mean_permeability_pred": ("permeability_pred", "mean"),
        "median_permeability_pred": ("permeability_pred", "median"),
        "max_permeability_pred": ("permeability_pred", "max"),
        "mean_permeability_log10": ("permeability_log10", "mean"),
        "median_permeability_log10": ("permeability_log10", "median"),
    }
    if "natural_aa_recovery" in d.columns:
        agg["mean_natural_aa_recovery"] = ("natural_aa_recovery", "mean")
        agg["best_natural_aa_recovery"] = ("natural_aa_recovery", "max")
    if "peptide_ca_rmsd_after_receptor_fit" in d.columns:
        agg["mean_peptide_ca_rmsd_after_receptor_fit"] = ("peptide_ca_rmsd_after_receptor_fit", "mean")
        agg["median_peptide_ca_rmsd_after_receptor_fit"] = ("peptide_ca_rmsd_after_receptor_fit", "median")
    out = d.groupby(group_col, dropna=False).agg(**agg).reset_index()
    return out


def write_report(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permeability_dir", default=str(DEFAULT_PERM_DIR))
    ap.add_argument("--all_designs_csv", default=str(DEFAULT_ALL_DESIGNS))
    ap.add_argument("--best85_csv", default=str(DEFAULT_BEST85))
    ap.add_argument("--rmsd_csv", default=str(DEFAULT_RMSD))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    perm_dir = Path(args.permeability_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("===== COMPLEX PERMEABILITY MERGE =====")
    report.append(f"permeability_dir: {perm_dir}")
    report.append(f"permeability_dir_exists: {perm_dir.exists()}")

    if not perm_dir.exists():
        raise FileNotFoundError(f"Missing permeability directory: {perm_dir}")

    perm = read_permeability_folder(perm_dir)
    report.append(f"permeability_rows: {len(perm)}")
    report.append(f"permeability_files: {len(list(perm_dir.glob('*.csv')))}")
    report.append("permeability_rows_by_temperature:")
    report.append(perm["temperature"].value_counts().sort_index().to_string())

    perm.to_csv(out_dir / "complex_permeability_raw_merged.csv", index=False, encoding="utf-8")

    all_designs = add_design_keys(pd.read_csv(args.all_designs_csv))
    best85 = add_design_keys(pd.read_csv(args.best85_csv))
    rmsd = add_structure_keys(pd.read_csv(args.rmsd_csv))

    perm_full, dup_full = prepare_perm_for_design_merge(perm)
    perm_design = prepare_perm_for_design_seq_merge(perm)

    all_merged = all_designs.merge(perm_full, on="merge_key_full", how="left", validate="m:1")
    best_merged = best85.merge(perm_full, on="merge_key_full", how="left", validate="m:1")
    struct_merged = rmsd.merge(perm_design, on="merge_key_design", how="left", validate="m:1")

    # Clean helper key columns are retained for auditability but placed near the end by pandas default.
    all_merged.to_csv(out_dir / "complex_permeability_all_designs.csv", index=False, encoding="utf-8")
    best_merged.to_csv(out_dir / "complex_permeability_best85.csv", index=False, encoding="utf-8")
    struct_merged.to_csv(out_dir / "complex_structure_permeability_merged.csv", index=False, encoding="utf-8")

    # Unmatched rows for audit.
    unmatched = []
    for label, df in [
        ("all_designs", all_merged),
        ("best85", best_merged),
        ("structure_rmsd", struct_merged),
    ]:
        miss = df[df["permeability_pred"].isna()].copy()
        if not miss.empty:
            keep = [c for c in ["target_name", "temperature", "record_index", "design_seq", "design_natural_seq", "merge_key_full", "merge_key_design"] if c in miss.columns]
            tmp = miss[keep].copy()
            tmp.insert(0, "table", label)
            unmatched.append(tmp)
    unmatched_df = pd.concat(unmatched, ignore_index=True) if unmatched else pd.DataFrame(columns=["table"])
    unmatched_df.to_csv(out_dir / "complex_permeability_unmatched_rows.csv", index=False, encoding="utf-8")

    summary_temp = summarize_by(best_merged, "temperature")
    summary_target = summarize_by(best_merged, "target_name")
    summary_temp.to_csv(out_dir / "complex_permeability_summary_by_temperature.csv", index=False, encoding="utf-8")
    summary_target.to_csv(out_dir / "complex_permeability_summary_by_target.csv", index=False, encoding="utf-8")

    # Optional joint structure/permeability summaries.
    summarize_by(struct_merged, "temperature").to_csv(
        out_dir / "complex_structure_permeability_summary_by_temperature.csv",
        index=False,
        encoding="utf-8",
    )
    summarize_by(struct_merged, "target_name").to_csv(
        out_dir / "complex_structure_permeability_summary_by_target.csv",
        index=False,
        encoding="utf-8",
    )

    report.append("")
    report.append("===== MERGE STATUS =====")
    for label, df in [("all_designs", all_merged), ("best85", best_merged), ("structure_rmsd", struct_merged)]:
        n = len(df)
        n_match = int(df["permeability_pred"].notna().sum())
        n_missing = n - n_match
        report.append(f"{label}: rows={n}, matched={n_match}, missing={n_missing}")
    report.append(f"duplicate_full_permeability_keys: {len(dup_full)}")

    report.append("")
    report.append("===== BEST85 PERMEABILITY SUMMARY BY TEMPERATURE =====")
    report.append(summary_temp.to_string(index=False))

    report.append("")
    report.append("===== NOTES =====")
    report.append("permeability_pred values are extremely small; use permeability_log10 for plots and correlations.")
    report.append("structure_rmsd merge uses target + temperature + exact design_seq. If a design has multiple PDB/permeability records, the maximum permeability_pred is used and permeability_match_count is recorded.")
    report.append("This script does not calculate energy-based Success/Stability.")

    write_report(out_dir / "complex_permeability_merge_report.txt", report)

    print("完成：complex permeability merge")
    print("report:", out_dir / "complex_permeability_merge_report.txt")
    print("all designs:", out_dir / "complex_permeability_all_designs.csv")
    print("best85:", out_dir / "complex_permeability_best85.csv")
    print("structure merged:", out_dir / "complex_structure_permeability_merged.csv")
    print("summary by temperature:", out_dir / "complex_permeability_summary_by_temperature.csv")
    print("summary by target:", out_dir / "complex_permeability_summary_by_target.csv")


if __name__ == "__main__":
    main()
