#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_prepare_monomer_structure_manifest.py

从单体测试集 jsonl 生成单体结构预测清单。

用途：
- 告诉师兄单体需要预测哪些序列。
- 如果结构预测平台不支持 N-甲基化残基，就用 naturalized_sequence。
- 如果支持修饰残基，则可参考 original_sequence 和 methyl_positions_1based。
"""

import os
import re
import csv
import json
import argparse
from collections import Counter


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_name(text):
    text = str(text)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in "_-":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def get_record_name(record, idx):
    for k in ["name", "title", "id", "pdb_id", "pdb", "target_name", "protein_name"]:
        v = record.get(k)
        if v:
            return str(v)
    return f"monomer_{idx:05d}"


def find_sequence_chains(record):
    """
    返回 [(chain_id, sequence), ...]
    优先读取 seq_chain_A 这种字段。
    如果没有，就尝试 seq / sequence。
    """
    out = []

    for k in sorted(record.keys()):
        m = re.match(r"seq_chain_(.+)$", k)
        if m:
            seq = record.get(k, "")
            if isinstance(seq, str) and seq:
                out.append((m.group(1), seq))

    if out:
        return out

    for k in ["seq", "sequence", "native_seq", "design_seq"]:
        seq = record.get(k)
        if isinstance(seq, str) and seq:
            return [("A", seq)]

    return []


def naturalize_sequence(seq):
    """
    小写字母代表 N-甲基化氨基酸。
    做结构预测时，如果平台不支持修饰残基，可以先转成普通大写氨基酸。
    """
    return "".join(ch.upper() if ch.islower() else ch for ch in seq)


def methyl_positions_1based(seq):
    return [str(i + 1) for i, ch in enumerate(seq) if ch.islower()]


def infer_dataset_type(record, name):
    text_parts = [name]
    for k, v in record.items():
        lk = k.lower()
        if any(x in lk for x in ["source", "dataset", "path", "file", "pdb", "name", "title"]):
            if isinstance(v, (str, int, float)):
                text_parts.append(str(v))
    text = " ".join(text_parts).lower()

    if "baker" in text:
        return "Baker33_true_structure"
    if "rosetta" in text:
        return "Rosetta_generated_structure"
    if "true" in text or "native" in text or "pdb" in text:
        return "monomer_with_reference_structure"
    return "unknown_monomer"


def find_reference_hint(record):
    hints = []
    for k, v in record.items():
        lk = k.lower()
        if lk.startswith("coords"):
            continue
        if any(x in lk for x in ["pdb", "cif", "structure", "path", "file"]):
            if isinstance(v, (str, int, float)):
                hints.append(f"{k}={v}")
    return "; ".join(hints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monomer_jsonl", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    records = read_jsonl(args.monomer_jsonl)
    rows = []
    warnings = []

    for i, r in enumerate(records):
        name = get_record_name(r, i)
        seq_chains = find_sequence_chains(r)
        dataset_type = infer_dataset_type(r, name)
        reference_hint = find_reference_hint(r)

        if not seq_chains:
            warnings.append({
                "record_index": i,
                "target_name": name,
                "warning": "没有找到 seq_chain_* 或 seq / sequence 字段",
            })
            continue

        for chain_id, seq in seq_chains:
            nat_seq = naturalize_sequence(seq)
            methyl_pos = methyl_positions_1based(seq)
            job_name = safe_name(f"{name}_chain_{chain_id}_monomer")

            rows.append({
                "suggested_job_name": job_name,
                "dataset_type": dataset_type,
                "record_index": i,
                "target_name": name,
                "chain_id": chain_id,
                "original_sequence": seq,
                "naturalized_sequence": nat_seq,
                "sequence_for_structure_prediction": nat_seq,
                "sequence_length": len(seq),
                "methyl_positions_1based": ",".join(methyl_pos),
                "methyl_count": len(methyl_pos),
                "reference_structure_hint": reference_hint,
                "note": "如果结构预测平台支持 N-甲基化残基，可参考 original_sequence 和 methyl_positions_1based；如果不支持，先用 naturalized_sequence 预测结构。",
            })

    write_csv(args.out_csv, rows)
    warn_csv = os.path.join(os.path.dirname(args.out_csv), "monomer_structure_manifest_warnings.csv")
    write_csv(warn_csv, warnings)

    print("单体结构预测清单已生成:", args.out_csv)
    print("任务数:", len(rows))
    print("警告数:", len(warnings))
    print("警告文件:", warn_csv)

    c = Counter(r["dataset_type"] for r in rows)
    print("dataset_type 统计:")
    for k, v in c.items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
