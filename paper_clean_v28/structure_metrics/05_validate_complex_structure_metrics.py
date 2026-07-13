#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_validate_complex_structure_metrics.py

一键质量闸门：
只有本脚本显示 QUALITY GATE: PASS，04/06 的结构结果才允许 commit/push。

检查内容：
1. Kabsch 数学自测；
2. 关键输入/输出文件是否存在；
3. chain mapping 是否符合预期；
4. RMSD 输出表是否 85 行；
5. 当前完整 best85 口径下，chain mapping 和 RMSD 应为 85/85 OK；
6. 必要 RMSD / HighFold 列是否存在；
7. receptor fit RMSD 是否在合理范围；
8. peptide self-superposed RMSD 是否不大于 receptor-fit peptide RMSD；
9. HighFold COMMENT global pLDDT、mean CA B-factor、ipTM、inter-PAE 分开审计。

科学口径：
- COMMENT global pLDDT 与 mean CA B-factor 不是同一数值，不能混成一列；
- mean CA B-factor 可作为 per-residue pLDDT proxy，但必须明确标注；
- ipTM / inter-PAE 不能从普通 ATOM 坐标恢复。
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("paper_clean_v28_outputs/structure_metrics")

AUDIT_PATH = OUT_DIR / "complex_chain_mapping_audit.csv"
RMSD_PATH = OUT_DIR / "complex_rmsd_metrics.csv"
SUMMARY_TEMP_PATH = OUT_DIR / "complex_rmsd_summary_by_temperature.csv"
SUMMARY_TARGET_PATH = OUT_DIR / "complex_rmsd_summary_by_target.csv"

REPORT_PATH = OUT_DIR / "complex_structure_quality_gate_report.txt"
PROBLEM_PATH = OUT_DIR / "complex_structure_quality_gate_problem_rows.csv"

EXPECTED_ROWS = 85
EXPECTED_OK = 85
EXPECTED_NOT_OK = 0


def rmsd(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


def kabsch_fit(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)

    P0 = P - Pc
    Q0 = Q - Qc

    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = Qc - Pc @ R
    return R, t


def kabsch_self_test():
    np.random.seed(1)
    P = np.random.randn(20, 3)
    theta = 0.7
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    t_true = np.array([5, -3, 2])
    Q = P @ R_true + t_true
    R, t = kabsch_fit(P, Q)
    return rmsd(P @ R + t, Q)


def add(lines, text):
    print(text)
    lines.append(str(text))


def numeric_missing(df, col):
    return int(pd.to_numeric(df[col], errors="coerce").isna().sum())


def main():
    lines = []
    problems = []
    warnings = []

    add(lines, "===== COMPLEX STRUCTURE QUALITY GATE =====")
    add(lines, f"Expected best85 rows = {EXPECTED_ROWS}")
    add(lines, f"Expected OK rows     = {EXPECTED_OK}")
    add(lines, f"Expected non-OK rows = {EXPECTED_NOT_OK}")

    err = kabsch_self_test()
    add(lines, f"Kabsch self-test RMSD = {err:.12g}")
    if err > 1e-6:
        problems.append({"problem_type": "kabsch_self_test_failed", "detail": f"err={err}"})

    required_files = [AUDIT_PATH, RMSD_PATH, SUMMARY_TEMP_PATH, SUMMARY_TARGET_PATH]
    for p in required_files:
        add(lines, f"exists {p}: {p.exists()}")
        if not p.exists():
            problems.append({"problem_type": "missing_file", "detail": str(p)})

    if problems:
        write_outputs(lines, problems, warnings)
        return 1

    audit = pd.read_csv(AUDIT_PATH)
    rmsd_df = pd.read_csv(RMSD_PATH)
    summary_temp = pd.read_csv(SUMMARY_TEMP_PATH)

    add(lines, "")
    add(lines, "===== BASIC SHAPE CHECK =====")
    add(lines, f"chain_mapping_audit shape = {audit.shape}")
    add(lines, f"rmsd_metrics shape = {rmsd_df.shape}")
    add(lines, f"summary_by_temperature shape = {summary_temp.shape}")

    if len(audit) != EXPECTED_ROWS:
        problems.append({"problem_type": "unexpected_chain_mapping_rows", "detail": f"expected {EXPECTED_ROWS}, got {len(audit)}"})
    if len(rmsd_df) != EXPECTED_ROWS:
        problems.append({"problem_type": "unexpected_rmsd_rows", "detail": f"expected {EXPECTED_ROWS}, got {len(rmsd_df)}"})

    add(lines, "")
    add(lines, "===== CHAIN MAPPING STATUS =====")
    if "chain_mapping_status" in audit.columns:
        vc = audit["chain_mapping_status"].value_counts(dropna=False)
        add(lines, vc.to_string())
        n_ok_mapping = int((audit["chain_mapping_status"] == "ok").sum())
        n_not_ok_mapping = len(audit) - n_ok_mapping
        if n_ok_mapping != EXPECTED_OK or n_not_ok_mapping != EXPECTED_NOT_OK:
            problems.append({
                "problem_type": "unexpected_chain_mapping_counts",
                "detail": f"ok={n_ok_mapping}, not_ok={n_not_ok_mapping}, expected {EXPECTED_OK}/{EXPECTED_NOT_OK}",
            })
    else:
        problems.append({"problem_type": "missing_column", "detail": "chain_mapping_status"})

    add(lines, "")
    add(lines, "===== RMSD STATUS =====")
    if "rmsd_status" not in rmsd_df.columns:
        problems.append({"problem_type": "missing_column", "detail": "rmsd_status"})
    else:
        vc = rmsd_df["rmsd_status"].value_counts(dropna=False)
        add(lines, vc.to_string())
        n_ok = int((rmsd_df["rmsd_status"] == "ok").sum())
        n_skip = len(rmsd_df) - n_ok
        if n_ok != EXPECTED_OK or n_skip != EXPECTED_NOT_OK:
            problems.append({
                "problem_type": "unexpected_rmsd_status_counts",
                "detail": f"ok={n_ok}, skip/fail={n_skip}, expected {EXPECTED_OK}/{EXPECTED_NOT_OK}",
            })

    required_cols = [
        "target_name", "temperature", "design_seq", "rmsd_status",
        "n_receptor_ca_fit_pairs", "n_peptide_ca_pairs",
        "receptor_ca_fit_rmsd", "peptide_ca_rmsd_after_receptor_fit",
        "peptide_backbone_rmsd_after_receptor_fit", "peptide_ca_rmsd_self_superposed",
        "highfold_plddt", "highfold_plddt_comment", "highfold_ca_bfactor_mean",
        "peptide_receptor_iptm_mean", "peptide_receptor_inter_pae_mean",
    ]

    add(lines, "")
    add(lines, "===== REQUIRED COLUMN CHECK =====")
    for col in required_cols:
        ok = col in rmsd_df.columns
        add(lines, f"{col}: {ok}")
        if not ok:
            problems.append({"problem_type": "missing_column", "detail": col})

    if problems:
        write_outputs(lines, problems, warnings)
        return 1

    ok_df = rmsd_df[rmsd_df["rmsd_status"] == "ok"].copy()

    numeric_cols = [
        "n_receptor_ca_fit_pairs", "n_peptide_ca_pairs",
        "receptor_ca_fit_rmsd", "peptide_ca_rmsd_after_receptor_fit",
        "peptide_backbone_rmsd_after_receptor_fit", "peptide_ca_rmsd_self_superposed",
    ]
    for col in numeric_cols:
        ok_df[col] = pd.to_numeric(ok_df[col], errors="coerce")

    add(lines, "")
    add(lines, "===== RMSD NUMERIC SUMMARY FOR OK ROWS =====")
    add(lines, ok_df[numeric_cols].describe().to_string())

    receptor_median = float(ok_df["receptor_ca_fit_rmsd"].median())
    receptor_max = float(ok_df["receptor_ca_fit_rmsd"].max())
    add(lines, "")
    add(lines, f"receptor_ca_fit_rmsd median = {receptor_median:.4f}")
    add(lines, f"receptor_ca_fit_rmsd max    = {receptor_max:.4f}")

    if receptor_median > 5.0:
        problems.append({"problem_type": "receptor_fit_rmsd_median_too_large", "detail": f"median={receptor_median:.4f}, threshold=5.0"})
    if receptor_max > 15.0:
        warnings.append({"warning_type": "receptor_fit_rmsd_max_large", "detail": f"max={receptor_max:.4f}, threshold=15.0"})

    bad_self = ok_df[
        ok_df["peptide_ca_rmsd_self_superposed"]
        > ok_df["peptide_ca_rmsd_after_receptor_fit"] + 1e-6
    ]
    for _, r in bad_self.iterrows():
        problems.append({
            "problem_type": "peptide_self_superposed_larger_than_receptor_fit",
            "detail": f"{r.get('target_name')} T{r.get('temperature')} {r.get('design_seq')}",
        })

    low_receptor_pairs = ok_df[ok_df["n_receptor_ca_fit_pairs"] < 20]
    for _, r in low_receptor_pairs.iterrows():
        problems.append({
            "problem_type": "too_few_receptor_ca_fit_pairs",
            "detail": f"{r.get('target_name')} T{r.get('temperature')} pairs={r.get('n_receptor_ca_fit_pairs')}",
        })

    low_peptide_pairs = ok_df[ok_df["n_peptide_ca_pairs"] < 3]
    for _, r in low_peptide_pairs.iterrows():
        problems.append({
            "problem_type": "too_few_peptide_ca_pairs",
            "detail": f"{r.get('target_name')} T{r.get('temperature')} pairs={r.get('n_peptide_ca_pairs')}",
        })

    add(lines, "")
    add(lines, "===== HIGHFOLD SCORE COMPLETENESS =====")

    comment_missing = numeric_missing(ok_df, "highfold_plddt_comment")
    ca_missing = numeric_missing(ok_df, "highfold_ca_bfactor_mean")
    iptm_missing = numeric_missing(ok_df, "peptide_receptor_iptm_mean")
    pae_missing = numeric_missing(ok_df, "peptide_receptor_inter_pae_mean")

    add(lines, f"COMMENT global pLDDT: missing among OK rows = {comment_missing}/{len(ok_df)}")
    add(lines, f"mean CA B-factor / pLDDT proxy: missing among OK rows = {ca_missing}/{len(ok_df)}")
    add(lines, f"peptide_receptor_iptm_mean: missing among OK rows = {iptm_missing}/{len(ok_df)}")
    add(lines, f"peptide_receptor_inter_pae_mean: missing among OK rows = {pae_missing}/{len(ok_df)}")

    if ca_missing > 0:
        warnings.append({"warning_type": "missing_highfold_ca_bfactor", "detail": f"mean CA B-factor missing {ca_missing}/{len(ok_df)}"})
    if iptm_missing > 0:
        warnings.append({"warning_type": "missing_highfold_score", "detail": f"peptide_receptor_iptm_mean missing {iptm_missing}/{len(ok_df)}"})
    if pae_missing > 0:
        warnings.append({"warning_type": "missing_highfold_score", "detail": f"peptide_receptor_inter_pae_mean missing {pae_missing}/{len(ok_df)}"})

    add(lines, "")
    add(lines, "HighFold score missing by temperature:")
    for label, col in [
        ("COMMENT global pLDDT", "highfold_plddt_comment"),
        ("mean CA B-factor / pLDDT proxy", "highfold_ca_bfactor_mean"),
        ("peptide_receptor_iptm_mean", "peptide_receptor_iptm_mean"),
        ("peptide_receptor_inter_pae_mean", "peptide_receptor_inter_pae_mean"),
    ]:
        numeric = pd.to_numeric(ok_df[col], errors="coerce")
        miss = ok_df.assign(_numeric=numeric).groupby("temperature")["_numeric"].apply(lambda s: int(s.isna().sum()))
        add(lines, f"\n{label}")
        add(lines, miss.to_string())

    add(lines, "")
    add(lines, "===== SUMMARY BY TEMPERATURE =====")
    add(lines, summary_temp.to_string(index=False))

    if len(problems) == 0:
        add(lines, "")
        add(lines, "QUALITY GATE: PASS")
        add(lines, f"WARNINGS: {len(warnings)}")
        write_outputs(lines, problems, warnings)
        return 0

    add(lines, "")
    add(lines, "QUALITY GATE: FAIL")
    add(lines, f"PROBLEMS: {len(problems)}")
    add(lines, f"WARNINGS: {len(warnings)}")
    write_outputs(lines, problems, warnings)
    return 1


def write_outputs(lines, problems, warnings):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for p in problems:
        rows.append({"level": "PROBLEM", "type": p.get("problem_type", ""), "detail": p.get("detail", "")})
    for w in warnings:
        rows.append({"level": "WARNING", "type": w.get("warning_type", ""), "detail": w.get("detail", "")})

    pd.DataFrame(rows, columns=["level", "type", "detail"]).to_csv(
        PROBLEM_PATH, index=False, encoding="utf-8"
    )

    print("")
    print("quality report:", REPORT_PATH)
    print("problem/warning rows:", PROBLEM_PATH)


if __name__ == "__main__":
    sys.exit(main())
