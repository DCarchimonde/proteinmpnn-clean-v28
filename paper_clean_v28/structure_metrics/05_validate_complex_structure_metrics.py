#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_validate_complex_structure_metrics.py

一键质量闸门：
只有本脚本显示 QUALITY GATE: PASS，04 的 RMSD 结果才允许 commit/push。

检查内容：
1. Kabsch 数学自测；
2. 关键输入/输出文件是否存在；
3. chain mapping 是否符合预期；
4. RMSD 输出表是否 85 行；
5. RMSD OK 是否为 81，缺失是否为 4；
6. 必要 RMSD 列是否存在；
7. receptor fit RMSD 是否在合理范围；
8. peptide self-superposed RMSD 是否不大于 receptor-fit peptide RMSD；
9. HighFold 分数缺失情况按温度汇总，作为 WARN。
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
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])
    t_true = np.array([5, -3, 2])

    Q = P @ R_true + t_true
    R, t = kabsch_fit(P, Q)
    err = rmsd(P @ R + t, Q)
    return err


def add(lines, text):
    print(text)
    lines.append(str(text))


def main():
    lines = []
    problems = []
    warnings = []

    add(lines, "===== COMPLEX STRUCTURE QUALITY GATE =====")

    # 1. Kabsch self-test
    err = kabsch_self_test()
    add(lines, f"Kabsch self-test RMSD = {err:.12g}")

    if err > 1e-6:
        problems.append({
            "problem_type": "kabsch_self_test_failed",
            "detail": f"err={err}",
        })

    # 2. File existence
    required_files = [
        AUDIT_PATH,
        RMSD_PATH,
        SUMMARY_TEMP_PATH,
        SUMMARY_TARGET_PATH,
    ]

    for p in required_files:
        add(lines, f"exists {p}: {p.exists()}")
        if not p.exists():
            problems.append({
                "problem_type": "missing_file",
                "detail": str(p),
            })

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

    if len(audit) != 85:
        problems.append({
            "problem_type": "unexpected_chain_mapping_rows",
            "detail": f"expected 85, got {len(audit)}",
        })

    if len(rmsd_df) != 85:
        problems.append({
            "problem_type": "unexpected_rmsd_rows",
            "detail": f"expected 85, got {len(rmsd_df)}",
        })

    # 3. Chain mapping status
    add(lines, "")
    add(lines, "===== CHAIN MAPPING STATUS =====")

    if "chain_mapping_status" in audit.columns:
        vc = audit["chain_mapping_status"].value_counts(dropna=False)
        add(lines, vc.to_string())

        n_ok_mapping = int((audit["chain_mapping_status"] == "ok").sum())
        n_not_ok_mapping = len(audit) - n_ok_mapping

        if n_ok_mapping != 81 or n_not_ok_mapping != 4:
            problems.append({
                "problem_type": "unexpected_chain_mapping_counts",
                "detail": f"ok={n_ok_mapping}, not_ok={n_not_ok_mapping}, expected 81/4",
            })
    else:
        problems.append({
            "problem_type": "missing_column",
            "detail": "chain_mapping_status",
        })

    # 4. RMSD status
    add(lines, "")
    add(lines, "===== RMSD STATUS =====")

    if "rmsd_status" not in rmsd_df.columns:
        problems.append({
            "problem_type": "missing_column",
            "detail": "rmsd_status",
        })
    else:
        vc = rmsd_df["rmsd_status"].value_counts(dropna=False)
        add(lines, vc.to_string())

        n_ok = int((rmsd_df["rmsd_status"] == "ok").sum())
        n_skip = len(rmsd_df) - n_ok

        if n_ok != 81 or n_skip != 4:
            problems.append({
                "problem_type": "unexpected_rmsd_status_counts",
                "detail": f"ok={n_ok}, skip/fail={n_skip}, expected 81/4",
            })

    # 5. Required metric columns
    required_cols = [
        "target_name",
        "temperature",
        "design_seq",
        "rmsd_status",
        "n_receptor_ca_fit_pairs",
        "n_peptide_ca_pairs",
        "receptor_ca_fit_rmsd",
        "peptide_ca_rmsd_after_receptor_fit",
        "peptide_backbone_rmsd_after_receptor_fit",
        "peptide_ca_rmsd_self_superposed",
        "highfold_plddt",
        "peptide_receptor_iptm_mean",
        "peptide_receptor_inter_pae_mean",
    ]

    add(lines, "")
    add(lines, "===== REQUIRED COLUMN CHECK =====")

    for col in required_cols:
        ok = col in rmsd_df.columns
        add(lines, f"{col}: {ok}")
        if not ok:
            problems.append({
                "problem_type": "missing_column",
                "detail": col,
            })

    if problems:
        write_outputs(lines, problems, warnings)
        return 1

    ok_df = rmsd_df[rmsd_df["rmsd_status"] == "ok"].copy()

    # 6. Numeric sanity
    numeric_cols = [
        "n_receptor_ca_fit_pairs",
        "n_peptide_ca_pairs",
        "receptor_ca_fit_rmsd",
        "peptide_ca_rmsd_after_receptor_fit",
        "peptide_backbone_rmsd_after_receptor_fit",
        "peptide_ca_rmsd_self_superposed",
    ]

    for col in numeric_cols:
        ok_df[col] = pd.to_numeric(ok_df[col], errors="coerce")

    add(lines, "")
    add(lines, "===== RMSD NUMERIC SUMMARY FOR OK ROWS =====")
    show_cols = [
        "n_receptor_ca_fit_pairs",
        "n_peptide_ca_pairs",
        "receptor_ca_fit_rmsd",
        "peptide_ca_rmsd_after_receptor_fit",
        "peptide_backbone_rmsd_after_receptor_fit",
        "peptide_ca_rmsd_self_superposed",
    ]
    add(lines, ok_df[show_cols].describe().to_string())

    # receptor fit RMSD should not be globally huge after correct Kabsch.
    receptor_median = float(ok_df["receptor_ca_fit_rmsd"].median())
    receptor_max = float(ok_df["receptor_ca_fit_rmsd"].max())

    add(lines, "")
    add(lines, f"receptor_ca_fit_rmsd median = {receptor_median:.4f}")
    add(lines, f"receptor_ca_fit_rmsd max    = {receptor_max:.4f}")

    if receptor_median > 5.0:
        problems.append({
            "problem_type": "receptor_fit_rmsd_median_too_large",
            "detail": f"median={receptor_median:.4f}, threshold=5.0",
        })

    if receptor_max > 15.0:
        warnings.append({
            "warning_type": "receptor_fit_rmsd_max_large",
            "detail": f"max={receptor_max:.4f}, threshold=15.0",
        })

    # peptide self-superposed should not be larger than receptor-fit peptide CA RMSD, allowing small tolerance.
    bad_self = ok_df[
        ok_df["peptide_ca_rmsd_self_superposed"]
        > ok_df["peptide_ca_rmsd_after_receptor_fit"] + 1e-6
    ]

    if len(bad_self) > 0:
        for _, r in bad_self.iterrows():
            problems.append({
                "problem_type": "peptide_self_superposed_larger_than_receptor_fit",
                "detail": f"{r.get('target_name')} T{r.get('temperature')} {r.get('design_seq')}",
            })

    # Pair counts
    low_receptor_pairs = ok_df[ok_df["n_receptor_ca_fit_pairs"] < 20]
    if len(low_receptor_pairs) > 0:
        for _, r in low_receptor_pairs.iterrows():
            problems.append({
                "problem_type": "too_few_receptor_ca_fit_pairs",
                "detail": f"{r.get('target_name')} T{r.get('temperature')} pairs={r.get('n_receptor_ca_fit_pairs')}",
            })

    low_peptide_pairs = ok_df[ok_df["n_peptide_ca_pairs"] < 3]
    if len(low_peptide_pairs) > 0:
        for _, r in low_peptide_pairs.iterrows():
            problems.append({
                "problem_type": "too_few_peptide_ca_pairs",
                "detail": f"{r.get('target_name')} T{r.get('temperature')} pairs={r.get('n_peptide_ca_pairs')}",
            })

    # 7. HighFold score completeness
    add(lines, "")
    add(lines, "===== HIGHFOLD SCORE COMPLETENESS =====")

    score_cols = [
        "highfold_plddt",
        "peptide_receptor_iptm_mean",
        "peptide_receptor_inter_pae_mean",
    ]

    for col in score_cols:
        ok_df[col] = pd.to_numeric(ok_df[col], errors="coerce")
        n_missing = int(ok_df[col].isna().sum())
        add(lines, f"{col}: missing among OK rows = {n_missing}/{len(ok_df)}")

        if n_missing > 0:
            warnings.append({
                "warning_type": "missing_highfold_score",
                "detail": f"{col}: missing {n_missing}/{len(ok_df)} among OK rows",
            })

    add(lines, "")
    add(lines, "HighFold score missing by temperature:")
    for col in score_cols:
        miss = ok_df.groupby("temperature")[col].apply(lambda s: int(s.isna().sum()))
        add(lines, f"\n{col}")
        add(lines, miss.to_string())

    # 8. Summary table sanity
    add(lines, "")
    add(lines, "===== SUMMARY BY TEMPERATURE =====")
    add(lines, summary_temp.to_string(index=False))

    if len(problems) == 0:
        add(lines, "")
        add(lines, "QUALITY GATE: PASS")
        if warnings:
            add(lines, f"WARNINGS: {len(warnings)}")
        else:
            add(lines, "WARNINGS: 0")
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

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    rows = []
    for p in problems:
        rows.append({
            "level": "PROBLEM",
            "type": p.get("problem_type", ""),
            "detail": p.get("detail", ""),
        })

    for w in warnings:
        rows.append({
            "level": "WARNING",
            "type": w.get("warning_type", ""),
            "detail": w.get("detail", ""),
        })

    pd.DataFrame(rows).to_csv(PROBLEM_PATH, index=False, encoding="utf-8")

    print("")
    print("quality report:", REPORT_PATH)
    print("problem/warning rows:", PROBLEM_PATH)


if __name__ == "__main__":
    sys.exit(main())
