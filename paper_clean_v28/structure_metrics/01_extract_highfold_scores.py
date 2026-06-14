#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_extract_highfold_scores.py

作用：
1. 从 raw_external/pdb_highfold_temperature/ 递归读取 HighFold 预测 PDB；
2. 从 PDB 开头 COMMENT 里提取 pLDDT、pTM、ipTM、inter-PAE 等分数；
3. 从文件名解析 target、temperature、design_seq；
4. 和 clean V28 的 all_designs.csv / af3_manifest.csv 做精确匹配审计；
5. 输出结果到 paper_clean_v28_outputs/structure_metrics/。

注意：
- 不需要 torch。
- 不会读取模型。
- 不会修改原始 PDB。
- raw_external/ 不应该上传 GitHub。
"""

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean


TEMP_MAP = {
    "pdb_highfold4_t001": "0.01",
    "pdb_highfold4_t01": "0.1",
    "pdb_highfold4_t02": "0.2",
    "pdb_highfold4_t03": "0.3",
    "pdb_highfold4_t05": "0.5",
}


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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_temp_from_path(path):
    s = str(path).replace("\\", "/")
    for folder_name, temp in TEMP_MAP.items():
        if folder_name in s:
            return temp, folder_name
    return "", ""


def parse_filename(path):
    name = Path(path).name
    m = re.match(r"^([A-Za-z0-9]+)_(\d+)_(.+)_model\.pdb$", name)

    if not m:
        return {
            "target_name": "",
            "file_index": "",
            "design_seq_from_filename": "",
            "filename_parse_ok": 0,
        }

    return {
        "target_name": m.group(1).upper(),
        "file_index": m.group(2),
        "design_seq_from_filename": m.group(3),
        "filename_parse_ok": 1,
    }


def parse_pdb(path):
    scores = {}
    residue_keys = set()
    chain_residue_keys = defaultdict(set)
    ca_bfactors = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("COMMENT"):
                m = re.match(r"^COMMENT\s+([^:]+):\s*(.+?)\s*$", line)
                if m:
                    key = m.group(1).strip()
                    value = m.group(2).strip()
                    try:
                        scores[key] = float(value)
                    except ValueError:
                        scores[key] = value

            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                chain_id = line[21].strip() or "_"
                resseq = line[22:26].strip()
                icode = line[26].strip()

                residue_key = (chain_id, resseq, icode)
                residue_keys.add(residue_key)
                chain_residue_keys[chain_id].add((resseq, icode))

                if atom_name == "CA":
                    try:
                        ca_bfactors.append(float(line[60:66]))
                    except Exception:
                        pass

    chain_counts = {
        chain: len(values)
        for chain, values in sorted(chain_residue_keys.items())
    }

    pdb_stats = {
        "pdb_total_residue_count": len(residue_keys),
        "pdb_chain_residue_counts": ";".join(
            f"{chain}:{count}" for chain, count in chain_counts.items()
        ),
        "pdb_ca_bfactor_mean": mean(ca_bfactors) if ca_bfactors else "",
    }

    return scores, pdb_stats


def build_design_index(all_designs_csv):
    rows = read_csv(all_designs_csv)
    index = defaultdict(list)

    for i, row in enumerate(rows):
        target = (row.get("target_name") or row.get("target") or "").upper()
        temp = norm_temp(row.get("temperature"))
        seq = row.get("design_seq") or row.get("design_peptide_seq") or ""
        key = (target, temp, seq)
        index[key].append((i, row))

    return rows, index


def build_best_key_set(af3_manifest_csv):
    if not af3_manifest_csv or not Path(af3_manifest_csv).exists():
        return set()

    rows = read_csv(af3_manifest_csv)
    keys = set()

    for row in rows:
        target = (row.get("target_name") or row.get("target") or "").upper()
        temp = norm_temp(row.get("temperature"))
        seq = row.get("design_peptide_seq") or row.get("design_seq") or ""
        keys.add((target, temp, seq))

    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_root",
        default="raw_external/pdb_highfold_temperature",
        help="HighFold PDB 解压目录",
    )
    parser.add_argument(
        "--all_designs",
        default="paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv",
    )
    parser.add_argument(
        "--af3_manifest",
        default="paper_clean_v28_outputs/af3_manifest.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="paper_clean_v28_outputs/structure_metrics",
    )
    args = parser.parse_args()

    pdb_root = Path(args.pdb_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdb_root.exists():
        raise FileNotFoundError(f"找不到 PDB 目录: {pdb_root}")

    pdb_files = sorted(pdb_root.rglob("*.pdb"))
    all_design_rows, design_index = build_design_index(args.all_designs)
    best_keys = build_best_key_set(args.af3_manifest)

    score_rows = []
    all_score_columns = set()
    pdb_match_key_counter = Counter()

    for pdb_file in pdb_files:
        temp, temp_folder = parse_temp_from_path(pdb_file)
        meta = parse_filename(pdb_file)
        scores, pdb_stats = parse_pdb(pdb_file)

        target = meta["target_name"]
        design_seq = meta["design_seq_from_filename"]
        match_key = (target, temp, design_seq)
        matches = design_index.get(match_key, [])

        pdb_match_key_counter[match_key] += 1

        row = {
            "pdb_path": str(pdb_file),
            "pdb_file": pdb_file.name,
            "temperature_folder": temp_folder,
            "temperature": temp,
            "target_name": target,
            "file_index": meta["file_index"],
            "design_seq_from_filename": design_seq,
            "design_natural_seq_from_filename": design_seq.upper(),
            "filename_parse_ok": meta["filename_parse_ok"],
            "matched_all_designs_count": len(matches),
            "matched_all_designs": 1 if matches else 0,
            "matched_all_designs_row_indices": ";".join(
                str(i) for i, _ in matches[:20]
            ),
            "is_best_design_in_af3_manifest": 1 if match_key in best_keys else 0,
        }

        row.update(pdb_stats)

        for key, value in scores.items():
            row[key] = value
            all_score_columns.add(key)

        score_rows.append(row)

    base_columns = [
        "pdb_path",
        "pdb_file",
        "temperature_folder",
        "temperature",
        "target_name",
        "file_index",
        "design_seq_from_filename",
        "design_natural_seq_from_filename",
        "filename_parse_ok",
        "matched_all_designs_count",
        "matched_all_designs",
        "matched_all_designs_row_indices",
        "is_best_design_in_af3_manifest",
        "pdb_total_residue_count",
        "pdb_chain_residue_counts",
        "pdb_ca_bfactor_mean",
    ]

    score_columns = sorted(all_score_columns)

    write_csv(
        out_dir / "complex_highfold_scores.csv",
        score_rows,
        base_columns + score_columns,
    )

    unmatched_pdb_rows = [
        row for row in score_rows
        if int(row["matched_all_designs"]) == 0
    ]

    write_csv(
        out_dir / "complex_unmatched_pdb.csv",
        unmatched_pdb_rows,
        base_columns + score_columns,
    )

    missing_design_rows = []

    for i, row in enumerate(all_design_rows):
        target = (row.get("target_name") or row.get("target") or "").upper()
        temp = norm_temp(row.get("temperature"))
        seq = row.get("design_seq") or row.get("design_peptide_seq") or ""
        key = (target, temp, seq)

        if pdb_match_key_counter[key] == 0:
            out = dict(row)
            out["all_designs_row_index"] = i
            out["match_key"] = "|".join(key)
            missing_design_rows.append(out)

    write_csv(
        out_dir / "complex_designs_missing_pdb.csv",
        missing_design_rows,
    )

    summary_counter = defaultdict(Counter)

    for row in score_rows:
        key = (row["temperature"], row["target_name"])
        summary_counter[key]["n_pdb"] += 1
        summary_counter[key]["n_pdb_matched_exact"] += int(row["matched_all_designs"])
        summary_counter[key]["n_pdb_best85"] += int(row["is_best_design_in_af3_manifest"])

    design_counter = defaultdict(Counter)

    for row in all_design_rows:
        target = (row.get("target_name") or row.get("target") or "").upper()
        temp = norm_temp(row.get("temperature"))
        seq = row.get("design_seq") or row.get("design_peptide_seq") or ""
        key = (temp, target)
        design_counter[key]["n_all_design_rows"] += 1

        if pdb_match_key_counter[(target, temp, seq)] > 0:
            design_counter[key]["n_all_design_rows_with_pdb"] += 1

    summary_rows = []

    for key in sorted(
        set(summary_counter.keys()) | set(design_counter.keys()),
        key=lambda x: (float(x[0]) if x[0] else 999, x[1]),
    ):
        temp, target = key
        s = summary_counter[key]
        d = design_counter[key]

        summary_rows.append({
            "temperature": temp,
            "target_name": target,
            "n_pdb": s["n_pdb"],
            "n_pdb_matched_exact": s["n_pdb_matched_exact"],
            "n_pdb_unmatched": s["n_pdb"] - s["n_pdb_matched_exact"],
            "n_pdb_best85": s["n_pdb_best85"],
            "n_all_design_rows": d["n_all_design_rows"],
            "n_all_design_rows_with_pdb": d["n_all_design_rows_with_pdb"],
            "n_all_design_rows_missing_pdb": d["n_all_design_rows"] - d["n_all_design_rows_with_pdb"],
        })

    write_csv(
        out_dir / "complex_structure_match_audit_summary.csv",
        summary_rows,
    )

    print("完成：HighFold PDB 分数提取和匹配审计")
    print("PDB 文件数:", len(pdb_files))
    print("all_designs 行数:", len(all_design_rows))
    print("PDB exact matched:", sum(1 for row in score_rows if int(row["matched_all_designs"]) > 0))
    print("PDB unmatched:", len(unmatched_pdb_rows))
    print("all_designs missing PDB:", len(missing_design_rows))
    print("输出目录:", out_dir)


if __name__ == "__main__":
    main()
