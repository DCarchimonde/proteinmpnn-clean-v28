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

Primary strict merge rule:
    target_name + temperature + exact methylated design_seq

Fallback merge rule:
    If strict match is missing, use target_name + exact methylated design_seq
    across temperatures. This is allowed because the permeability model is
    sequence-based, not generation-temperature-based. The fallback is explicitly
    labeled and audited.

Do not use the numeric part of the external permeability id as the original
all_designs record_index. The external id number is retained for audit only.

Outputs are written under:
    paper_clean_v28_outputs/permeability/
"""

import argparse
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
LOG10_FLOOR = 1e-300

PERM_FILL_COLS = [
    "permeability_pred",
    "permeability_pred_mean",
    "permeability_pred_median",
    "permeability_match_count",
    "permeability_zero_count",
    "permeability_positive_count",
    "permeability_id",
    "permeability_source_file",
    "permeability_external_index",
    "permeability_fasta",
    "permeability_methy_index",
    "design_seq_from_id",
    "design_natural_seq_from_id",
    "permeability_fasta_matches_id_natural_seq",
    "permeability_is_zero",
    "permeability_is_positive",
    "permeability_log10_positive_only",
    "permeability_log10_floor300",
    "permeability_log10",
    "permeability_temperatures_available",
]


def naturalize(seq):
    if seq is None or (isinstance(seq, float) and math.isnan(seq)):
        return ""
    seq = str(seq).strip().upper()
    return "".join(ch for ch in seq if ch in NATURAL_AA)


def norm_design_seq(seq):
    if seq is None or (isinstance(seq, float) and math.isnan(seq)):
        return ""
    return str(seq).strip()


def target_key(x):
    return str(x).strip().upper()


def temp_key(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        s = str(x).strip()
        if s in TEMP_FROM_FILE:
            return f"{TEMP_FROM_FILE[s]:.2f}"
        return s


def parse_temp_from_filename(path):
    name = Path(path).stem.lower()
    for token, temp in TEMP_FROM_FILE.items():
        if re.search(rf"(?:^|_){re.escape(token)}(?:$|_)", name):
            return temp
    last = name.split("_")[-1]
    if last in TEMP_FROM_FILE:
        return TEMP_FROM_FILE[last]
    raise ValueError(f"Cannot infer temperature from file name: {path}")


def parse_permeability_id(pid):
    s = str(pid).strip()
    s = re.sub(r"\.pdb$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_model$", "", s, flags=re.IGNORECASE)
    m = re.match(r"^(?P<target>[^_]+)_(?P<external_index>\d+)_(?P<design_seq>.+)$", s)
    if not m:
        return None, None, None
    return m.group("target"), int(m.group("external_index")), m.group("design_seq")


def compute_log10_positive_only(pred):
    pred = pd.to_numeric(pred, errors="coerce")
    out = pd.Series(np.nan, index=pred.index, dtype="float64")
    mask = pred.gt(0)
    out.loc[mask] = np.log10(pred.loc[mask])
    return out


def compute_log10_floor300(pred):
    pred = pd.to_numeric(pred, errors="coerce")
    safe = pred.copy()
    safe = safe.where(safe.gt(LOG10_FLOOR), LOG10_FLOOR)
    safe = safe.where(pred.notna(), np.nan)
    return np.log10(safe)


def add_log_columns(df):
    out = df.copy()
    pred = pd.to_numeric(out["permeability_pred"], errors="coerce")
    out["permeability_pred"] = pred
    out["permeability_is_zero"] = pred.eq(0)
    out["permeability_is_positive"] = pred.gt(0)
    out["permeability_log10_positive_only"] = compute_log10_positive_only(pred)
    out["permeability_log10_floor300"] = compute_log10_floor300(pred)
    out["permeability_log10"] = out["permeability_log10_positive_only"]
    return out


def add_common_keys(df, design_col="design_seq", natural_col=None):
    out = df.copy()
    out["target_key"] = out["target_name"].map(target_key)
    out["temperature_key"] = out["temperature"].map(temp_key)
    out["design_seq_key"] = out[design_col].map(norm_design_seq)
    if natural_col is not None and natural_col in out.columns:
        out["design_natural_key"] = out[natural_col].map(naturalize)
    else:
        out["design_natural_key"] = out[design_col].map(naturalize)
    out["merge_key_design"] = out["target_key"] + "|" + out["temperature_key"] + "|" + out["design_seq_key"]
    out["merge_key_target_design"] = out["target_key"] + "|" + out["design_seq_key"]
    out["merge_key_design_natural"] = out["target_key"] + "|" + out["temperature_key"] + "|" + out["design_natural_key"]
    out["merge_key_target_design_natural"] = out["target_key"] + "|" + out["design_natural_key"]
    return out


def read_permeability_folder(perm_dir):
    rows = []
    files = sorted(Path(perm_dir).glob("*.csv"))
    for path in files:
        temp = parse_temp_from_filename(path)
        df = pd.read_csv(path)
        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        required = {"id", "fasta", "methy_index", "permeability_pred"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for _, r in df.iterrows():
            target, external_index, design_seq = parse_permeability_id(r["id"])
            rows.append({
                "permeability_source_file": str(path),
                "permeability_id": str(r["id"]),
                "target_name": str(target).upper() if target is not None else "",
                "temperature": float(temp),
                "permeability_external_index": external_index,
                "design_seq_from_id": norm_design_seq(design_seq),
                "design_natural_seq_from_id": naturalize(design_seq),
                "permeability_fasta": naturalize(r["fasta"]),
                "permeability_methy_index": str(r["methy_index"]),
                "permeability_pred": r["permeability_pred"],
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = add_log_columns(out)
    out = add_common_keys(out, design_col="design_seq_from_id", natural_col="design_natural_seq_from_id")
    out["permeability_fasta_matches_id_natural_seq"] = out["permeability_fasta"].map(naturalize).eq(out["design_natural_key"])
    return out


def count_true(s):
    return int(pd.Series(s).astype("boolean").fillna(False).sum())


def join_unique(s):
    vals = []
    for x in s:
        if pd.isna(x):
            continue
        txt = str(x)
        if txt not in vals:
            vals.append(txt)
    return ";".join(vals[:20])


def aggregate_permeability(perm, key_col):
    g = perm.groupby(key_col, dropna=False)
    agg = g.agg(
        permeability_pred=("permeability_pred", "max"),
        permeability_pred_mean=("permeability_pred", "mean"),
        permeability_pred_median=("permeability_pred", "median"),
        permeability_match_count=("permeability_id", "count"),
        permeability_zero_count=("permeability_is_zero", count_true),
        permeability_positive_count=("permeability_is_positive", count_true),
        permeability_id=("permeability_id", join_unique),
        permeability_source_file=("permeability_source_file", join_unique),
        permeability_external_index=("permeability_external_index", join_unique),
        permeability_temperatures_available=("temperature", lambda s: ";".join(f"{float(x):.2f}" for x in sorted(set(pd.to_numeric(s, errors='coerce').dropna())))),
        permeability_fasta=("permeability_fasta", "first"),
        permeability_methy_index=("permeability_methy_index", "first"),
        design_seq_from_id=("design_seq_from_id", "first"),
        design_natural_seq_from_id=("design_natural_seq_from_id", "first"),
        permeability_fasta_matches_id_natural_seq=("permeability_fasta_matches_id_natural_seq", "all"),
    ).reset_index()
    agg = add_log_columns(agg)
    return agg


def merge_with_sequence_fallback(base_df, strict_perm, fallback_perm):
    merged = base_df.merge(strict_perm, on="merge_key_design", how="left", validate="m:1")
    merged["permeability_match_mode"] = np.where(
        merged["permeability_pred"].notna(),
        "strict_temperature",
        "missing",
    )

    fallback_cols = ["merge_key_target_design"] + [c for c in PERM_FILL_COLS if c in fallback_perm.columns]
    fb = fallback_perm[fallback_cols].copy()
    fb = fb.rename(columns={c: f"fallback_{c}" for c in fb.columns if c != "merge_key_target_design"})
    merged = merged.merge(fb, on="merge_key_target_design", how="left", validate="m:1")

    use_fb = merged["permeability_pred"].isna() & merged.get("fallback_permeability_pred").notna()
    for col in PERM_FILL_COLS:
        fb_col = f"fallback_{col}"
        if col in merged.columns and fb_col in merged.columns:
            merged.loc[use_fb, col] = merged.loc[use_fb, fb_col]
    merged.loc[use_fb, "permeability_match_mode"] = "sequence_fallback_across_temperature"
    merged.loc[merged["permeability_pred"].isna(), "permeability_match_mode"] = "missing"

    # Keep explicit fallback audit columns for traceability, but remove bulky duplicates.
    drop_cols = [c for c in merged.columns if c.startswith("fallback_")]
    merged = merged.drop(columns=drop_cols)
    return merged


def summarize_by(df, group_col):
    d = df.copy()
    d["permeability_pred"] = pd.to_numeric(d["permeability_pred"], errors="coerce")
    d["permeability_log10_positive_only"] = pd.to_numeric(d.get("permeability_log10_positive_only"), errors="coerce")
    d["permeability_log10_floor300"] = pd.to_numeric(d.get("permeability_log10_floor300"), errors="coerce")
    if "permeability_is_zero" not in d.columns:
        d["permeability_is_zero"] = False
    if "natural_aa_recovery" in d.columns:
        d["natural_aa_recovery"] = pd.to_numeric(d["natural_aa_recovery"], errors="coerce")
    if "peptide_ca_rmsd_after_receptor_fit" in d.columns:
        d["peptide_ca_rmsd_after_receptor_fit"] = pd.to_numeric(d["peptide_ca_rmsd_after_receptor_fit"], errors="coerce")

    agg = {
        "n_rows": ("permeability_pred", "size"),
        "n_with_permeability": ("permeability_pred", lambda s: int(s.notna().sum())),
        "n_strict_temperature_match": ("permeability_match_mode", lambda s: int((s == "strict_temperature").sum())),
        "n_sequence_fallback_match": ("permeability_match_mode", lambda s: int((s == "sequence_fallback_across_temperature").sum())),
        "n_missing_permeability": ("permeability_match_mode", lambda s: int((s == "missing").sum())),
        "n_zero_permeability": ("permeability_is_zero", count_true),
        "mean_permeability_pred": ("permeability_pred", "mean"),
        "median_permeability_pred": ("permeability_pred", "median"),
        "max_permeability_pred": ("permeability_pred", "max"),
        "mean_permeability_log10_positive_only": ("permeability_log10_positive_only", "mean"),
        "median_permeability_log10_positive_only": ("permeability_log10_positive_only", "median"),
        "mean_permeability_log10_floor300": ("permeability_log10_floor300", "mean"),
        "median_permeability_log10_floor300": ("permeability_log10_floor300", "median"),
    }
    if "natural_aa_recovery" in d.columns:
        agg["mean_natural_aa_recovery"] = ("natural_aa_recovery", "mean")
        agg["best_natural_aa_recovery"] = ("natural_aa_recovery", "max")
    if "peptide_ca_rmsd_after_receptor_fit" in d.columns:
        agg["mean_peptide_ca_rmsd_after_receptor_fit"] = ("peptide_ca_rmsd_after_receptor_fit", "mean")
        agg["median_peptide_ca_rmsd_after_receptor_fit"] = ("peptide_ca_rmsd_after_receptor_fit", "median")
    return d.groupby(group_col, dropna=False).agg(**agg).reset_index()


def unmatched_rows(label, df):
    miss = df[df["permeability_match_mode"] == "missing"].copy()
    if miss.empty:
        return pd.DataFrame()
    keep = [c for c in [
        "target_name", "temperature", "record_index", "design_seq", "design_natural_seq",
        "merge_key_design", "merge_key_target_design", "merge_key_design_natural",
        "merge_key_target_design_natural", "rmsd_status"
    ] if c in miss.columns]
    out = miss[keep].copy()
    out.insert(0, "table", label)
    return out


def write_report(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_match_status(report, label, df):
    n = len(df)
    strict = int((df["permeability_match_mode"] == "strict_temperature").sum())
    fallback = int((df["permeability_match_mode"] == "sequence_fallback_across_temperature").sum())
    missing = int((df["permeability_match_mode"] == "missing").sum())
    matched = strict + fallback
    report.append(f"{label}: rows={n}, matched={matched}, strict={strict}, sequence_fallback={fallback}, missing={missing}")


def append_unmatched_summary(report, unmatched_df):
    report.append("")
    report.append("===== REMAINING UNMATCHED SUMMARY AFTER SEQUENCE FALLBACK =====")
    if unmatched_df.empty:
        report.append("No unmatched rows.")
        return
    report.append("unmatched_by_table:")
    report.append(unmatched_df["table"].value_counts(dropna=False).to_string())
    if "temperature" in unmatched_df.columns:
        report.append("\nunmatched_by_table_temperature:")
        report.append(unmatched_df.groupby(["table", "temperature"]).size().to_string())
    if "target_name" in unmatched_df.columns:
        report.append("\nunmatched_by_table_target:")
        report.append(unmatched_df.groupby(["table", "target_name"]).size().to_string())
    report.append("\nremaining_unmatched_rows:")
    report.append(unmatched_df.to_string(index=False))


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
    report.append(f"permeability_zero_values: {int(perm['permeability_is_zero'].sum())}")
    report.append(f"permeability_positive_values: {int(perm['permeability_is_positive'].sum())}")
    report.append(f"permeability_fasta_mismatch_id_natural_seq: {int((~perm['permeability_fasta_matches_id_natural_seq']).sum())}")

    perm.to_csv(out_dir / "complex_permeability_raw_merged.csv", index=False, encoding="utf-8")
    perm_by_design = aggregate_permeability(perm, "merge_key_design")
    perm_by_target_design = aggregate_permeability(perm, "merge_key_target_design")
    perm_by_design.to_csv(out_dir / "complex_permeability_by_design.csv", index=False, encoding="utf-8")
    perm_by_target_design.to_csv(out_dir / "complex_permeability_by_target_design.csv", index=False, encoding="utf-8")

    all_designs = add_common_keys(pd.read_csv(args.all_designs_csv), design_col="design_seq", natural_col="design_natural_seq")
    best85 = add_common_keys(pd.read_csv(args.best85_csv), design_col="design_seq", natural_col="design_natural_seq")
    rmsd = add_common_keys(pd.read_csv(args.rmsd_csv), design_col="design_seq", natural_col=None)

    all_merged = merge_with_sequence_fallback(all_designs, perm_by_design, perm_by_target_design)
    best_merged = merge_with_sequence_fallback(best85, perm_by_design, perm_by_target_design)
    struct_merged = merge_with_sequence_fallback(rmsd, perm_by_design, perm_by_target_design)

    all_merged.to_csv(out_dir / "complex_permeability_all_designs.csv", index=False, encoding="utf-8")
    best_merged.to_csv(out_dir / "complex_permeability_best85.csv", index=False, encoding="utf-8")
    struct_merged.to_csv(out_dir / "complex_structure_permeability_merged.csv", index=False, encoding="utf-8")

    unmatched_parts = [
        unmatched_rows("all_designs", all_merged),
        unmatched_rows("best85", best_merged),
        unmatched_rows("structure_rmsd", struct_merged),
    ]
    unmatched_parts = [x for x in unmatched_parts if not x.empty]
    unmatched_df = pd.concat(unmatched_parts, ignore_index=True) if unmatched_parts else pd.DataFrame(columns=["table"])
    unmatched_df.to_csv(out_dir / "complex_permeability_unmatched_rows.csv", index=False, encoding="utf-8")

    fallback_rows = []
    for label, df in [("all_designs", all_merged), ("best85", best_merged), ("structure_rmsd", struct_merged)]:
        fb = df[df["permeability_match_mode"] == "sequence_fallback_across_temperature"].copy()
        if not fb.empty:
            keep = [c for c in [
                "target_name", "temperature", "record_index", "design_seq", "design_natural_seq",
                "permeability_pred", "permeability_id", "permeability_temperatures_available",
                "permeability_match_count", "merge_key_target_design"
            ] if c in fb.columns]
            fb = fb[keep]
            fb.insert(0, "table", label)
            fallback_rows.append(fb)
    fallback_df = pd.concat(fallback_rows, ignore_index=True) if fallback_rows else pd.DataFrame(columns=["table"])
    fallback_df.to_csv(out_dir / "complex_permeability_sequence_fallback_rows.csv", index=False, encoding="utf-8")

    summary_temp = summarize_by(best_merged, "temperature")
    summary_target = summarize_by(best_merged, "target_name")
    summary_temp.to_csv(out_dir / "complex_permeability_summary_by_temperature.csv", index=False, encoding="utf-8")
    summary_target.to_csv(out_dir / "complex_permeability_summary_by_target.csv", index=False, encoding="utf-8")

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
    report.append("===== MERGE STATUS WITH SEQUENCE FALLBACK =====")
    append_match_status(report, "all_designs", all_merged)
    append_match_status(report, "best85", best_merged)
    append_match_status(report, "structure_rmsd", struct_merged)
    report.append(f"unique_strict_permeability_design_keys: {perm_by_design['merge_key_design'].nunique()}")
    report.append(f"unique_target_design_fallback_keys: {perm_by_target_design['merge_key_target_design'].nunique()}")
    report.append(f"raw_permeability_rows: {len(perm)}")

    report.append("")
    report.append("===== SEQUENCE FALLBACK ROWS =====")
    if fallback_df.empty:
        report.append("No sequence fallback rows.")
    else:
        report.append(fallback_df.to_string(index=False))

    append_unmatched_summary(report, unmatched_df)

    report.append("")
    report.append("===== BEST85 PERMEABILITY SUMMARY BY TEMPERATURE =====")
    report.append(summary_temp.to_string(index=False))

    report.append("")
    report.append("===== NOTES =====")
    report.append("Strict match is target + temperature + exact methylated design_seq.")
    report.append("Fallback match is target + exact methylated design_seq across temperatures. It is labeled as sequence_fallback_across_temperature.")
    report.append("Permeability is sequence-based; generation temperature is metadata, so exact-sequence fallback is acceptable if disclosed.")
    report.append("permeability_log10_positive_only is computed only for positive predictions; zero predictions remain NaN there.")
    report.append("permeability_log10_floor300 maps zeros to 1e-300 before log10 for plotting if a finite lower bound is needed.")
    report.append("This script does not calculate energy-based Success/Stability.")

    write_report(out_dir / "complex_permeability_merge_report.txt", report)

    print("完成：complex permeability merge")
    print("report:", out_dir / "complex_permeability_merge_report.txt")
    print("all designs:", out_dir / "complex_permeability_all_designs.csv")
    print("best85:", out_dir / "complex_permeability_best85.csv")
    print("structure merged:", out_dir / "complex_structure_permeability_merged.csv")
    print("summary by temperature:", out_dir / "complex_permeability_summary_by_temperature.csv")
    print("summary by target:", out_dir / "complex_permeability_summary_by_target.csv")
    print("fallback rows:", out_dir / "complex_permeability_sequence_fallback_rows.csv")


if __name__ == "__main__":
    main()
