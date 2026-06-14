#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_audit_best85_structure_coverage.py

作用：
1. 以 af3_manifest.csv 的 85 条 best design 为基准；
2. 检查每条 best design 是否在 HighFold PDB 提取结果中有对应结构；
3. 如果一个 best design 对应多个 PDB，保留全部匹配记录，并选择 pLDDT 最高的作为代表结构；
4. 输出 85 行审计表和代表结构表。

输入：
- paper_clean_v28_outputs/af3_manifest.csv
- paper_clean_v28_outputs/structure_metrics/complex_highfold_scores.csv

输出：
- paper_clean_v28_outputs/structure_metrics/complex_best85_structure_audit.csv
- paper_clean_v28_outputs/structure_metrics/complex_best85_highfold_representative.csv
"""

import csv
from pathlib import Path
from collections import defaultdict


def norm_temp(x):
    if x is None or x == "":
        return ""
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def to_float(x, default=-999999.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def get_first(row, names):
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return ""


def main():
    af3_path = Path("paper_clean_v28_outputs/af3_manifest.csv")
    score_path = Path("paper_clean_v28_outputs/structure_metrics/complex_highfold_scores.csv")
    out_dir = Path("paper_clean_v28_outputs/structure_metrics")

    af3_rows = read_csv(af3_path)
    score_rows = read_csv(score_path)

    score_index = defaultdict(list)

    for r in score_rows:
        target = get_first(r, ["target_name", "target"]).upper()
        temp = norm_temp(get_first(r, ["temperature"]))
        seq = get_first(r, ["design_seq_from_filename", "design_peptide_seq", "design_seq"])
        key = (target, temp, seq)
        score_index[key].append(r)

    audit_rows = []
    rep_rows = []

    for i, r in enumerate(af3_rows):
        target = get_first(r, ["target_name", "target"]).upper()
        temp = norm_temp(get_first(r, ["temperature"]))
        seq = get_first(r, ["design_peptide_seq", "design_seq", "sequence"])

        key = (target, temp, seq)
        matches = score_index.get(key, [])

        if len(matches) == 0:
            status = "missing_pdb"
            representative = {}
        elif len(matches) == 1:
            status = "single_pdb"
            representative = matches[0]
        else:
            status = "multiple_pdb_same_design_seq"
            representative = sorted(
                matches,
                key=lambda x: to_float(x.get("plddt")),
                reverse=True
            )[0]

        audit = dict(r)
        audit["af3_manifest_row_index"] = i
        audit["match_target_name"] = target
        audit["match_temperature"] = temp
        audit["match_design_seq"] = seq
        audit["matched_pdb_count"] = len(matches)
        audit["structure_match_status"] = status
        audit["matched_pdb_files"] = ";".join(m.get("pdb_file", "") for m in matches)
        audit["representative_rule"] = "highest_plddt_if_multiple"
        audit["representative_pdb_file"] = representative.get("pdb_file", "")
        audit["representative_pdb_path"] = representative.get("pdb_path", "")
        audit["representative_plddt"] = representative.get("plddt", "")
        audit["representative_iptm_A_B"] = representative.get("iptm_A_B", "")
        audit["representative_iptm_A_C"] = representative.get("iptm_A_C", "")
        audit["representative_iptm_B_C"] = representative.get("iptm_B_C", "")
        audit["representative_inter_pae_A_B"] = representative.get("inter_pae_A_B", "")
        audit["representative_inter_pae_A_C"] = representative.get("inter_pae_A_C", "")
        audit["representative_inter_pae_B_C"] = representative.get("inter_pae_B_C", "")
        audit_rows.append(audit)

        rep = dict(r)
        rep["af3_manifest_row_index"] = i
        rep["structure_match_status"] = status
        rep["matched_pdb_count"] = len(matches)
        rep["representative_rule"] = "highest_plddt_if_multiple"

        for k, v in representative.items():
            rep[f"highfold_{k}"] = v

        rep_rows.append(rep)

    write_csv(out_dir / "complex_best85_structure_audit.csv", audit_rows)
    write_csv(out_dir / "complex_best85_highfold_representative.csv", rep_rows)

    n_missing = sum(1 for r in audit_rows if r["structure_match_status"] == "missing_pdb")
    n_single = sum(1 for r in audit_rows if r["structure_match_status"] == "single_pdb")
    n_multi = sum(1 for r in audit_rows if r["structure_match_status"] == "multiple_pdb_same_design_seq")

    print("完成：best85 结构覆盖审计")
    print("af3_manifest rows:", len(af3_rows))
    print("single_pdb:", n_single)
    print("multiple_pdb_same_design_seq:", n_multi)
    print("missing_pdb:", n_missing)
    print("输出:")
    print(out_dir / "complex_best85_structure_audit.csv")
    print(out_dir / "complex_best85_highfold_representative.csv")


if __name__ == "__main__":
    main()
