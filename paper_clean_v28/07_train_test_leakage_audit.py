#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_train_test_leakage_audit.py

Full exact-match train/test leakage audit when local train_set/train.jsonl is available.

This script intentionally does not require the training set to be committed to Git.
It reads local paths and writes audit outputs under paper_clean_v28_outputs/leakage_audit/.

Default paths:
- train: nmethyl_data/train_set/train.jsonl
- test:  nmethyl_data/test_set/test.jsonl
- complex native: 17_complexes_native.jsonl

Checks:
1. train/test file existence.
2. Duplicate names and duplicate naturalized chain sequences inside train/test.
3. Exact name overlap between train and test.
4. Exact naturalized chain-sequence overlap between train and test.
5. Exact naturalized concatenated-sequence overlap between train and test.
6. Exact naturalized sequence overlap between train and 17-complex native chains/short peptides.
7. Exact naturalized sequence overlap between test and 17-complex native chains/short peptides.

Important limitation:
This is exact-match leakage audit. It does not prove absence of homologous leakage.
Homology-level audit requires alignment/clustering, e.g. MMseqs/CD-HIT/BLAST-like search.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NATURAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def naturalize(seq):
    if seq is None:
        return ""
    seq = str(seq).strip().upper()
    return "".join(ch for ch in seq if ch in NATURAL_AA or ch == "X")


def read_jsonl(path):
    path = Path(path)
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["__line_no__"] = line_no
            rows.append(obj)
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def extract_records(rows, source_name, max_short_len=30):
    records = []
    for obj in rows:
        name = str(obj.get("name", f"line_{obj.get('__line_no__', '')}"))
        chain_items = []
        for key, val in obj.items():
            m = re.match(r"^seq_chain_(.+)$", key)
            if not m:
                continue
            chain_id = m.group(1)
            seq = naturalize(val)
            if not seq:
                continue
            chain_items.append((chain_id, seq))
            records.append({
                "source": source_name,
                "record_name": name,
                "chain_id": chain_id,
                "seq_type": "chain",
                "seq": seq,
                "length": len(seq),
            })
            if len(seq) <= max_short_len:
                records.append({
                    "source": source_name,
                    "record_name": name,
                    "chain_id": chain_id,
                    "seq_type": "short_chain",
                    "seq": seq,
                    "length": len(seq),
                })
        if chain_items:
            concat = naturalize("".join(seq for _, seq in sorted(chain_items)))
            records.append({
                "source": source_name,
                "record_name": name,
                "chain_id": "ALL_SORTED_CHAINS",
                "seq_type": "concat_all_chains",
                "seq": concat,
                "length": len(concat),
            })
    return records


def group(records, source=None, seq_types=None, key="seq"):
    out = defaultdict(list)
    for r in records:
        if source is not None and r["source"] != source:
            continue
        if seq_types is not None and r["seq_type"] not in seq_types:
            continue
        val = r.get(key, "")
        if val:
            out[val].append(r)
    return out


def duplicates(records, source, seq_type):
    g = group(records, source=source, seq_types={seq_type})
    return {k: v for k, v in g.items() if len(v) > 1}


def add_summary(summary, lines, check_name, status, count, note):
    summary.append({"check_name": check_name, "status": status, "count": count, "note": note})
    lines.append(f"{check_name}: {status} | count={count} | {note}")


def add_overlap(overlap_rows, check_name, source_a, source_b, group_a, group_b, key_type):
    overlap = sorted(set(group_a.keys()) & set(group_b.keys()))
    for key in overlap:
        a = group_a[key]
        b = group_b[key]
        overlap_rows.append({
            "check_name": check_name,
            "source_a": source_a,
            "source_b": source_b,
            "key_type": key_type,
            "key": key,
            "length": len(key),
            "count_a": len(a),
            "count_b": len(b),
            "examples_a": ";".join(f"{r['record_name']}:{r['chain_id']}:{r['seq_type']}" for r in a[:10]),
            "examples_b": ";".join(f"{r['record_name']}:{r['chain_id']}:{r['seq_type']}" for r in b[:10]),
        })
    return len(overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="nmethyl_data/train_set/train.jsonl")
    ap.add_argument("--test_jsonl", default="nmethyl_data/test_set/test.jsonl")
    ap.add_argument("--complex_jsonl", default="17_complexes_native.jsonl")
    ap.add_argument("--out_dir", default="paper_clean_v28_outputs/leakage_audit")
    ap.add_argument("--max_short_len", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_jsonl)
    test_path = Path(args.test_jsonl)
    complex_path = Path(args.complex_jsonl)

    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    complex_rows = read_jsonl(complex_path)

    train_records = extract_records(train_rows, "train_jsonl", args.max_short_len)
    test_records = extract_records(test_rows, "test_jsonl", args.max_short_len)
    complex_records = extract_records(complex_rows, "complex_native_jsonl", args.max_short_len)
    records = train_records + test_records + complex_records

    summary = []
    overlaps = []
    lines = ["===== TRAIN/TEST EXACT LEAKAGE AUDIT ====="]

    add_summary(summary, lines, "train_jsonl_exists", "PASS" if train_path.exists() else "FAIL", int(train_path.exists()), str(train_path))
    add_summary(summary, lines, "test_jsonl_exists", "PASS" if test_path.exists() else "FAIL", int(test_path.exists()), str(test_path))
    add_summary(summary, lines, "complex_jsonl_exists", "PASS" if complex_path.exists() else "FAIL", int(complex_path.exists()), str(complex_path))
    add_summary(summary, lines, "n_train_jsonl_rows", "INFO", len(train_rows), "JSONL rows in train file")
    add_summary(summary, lines, "n_test_jsonl_rows", "INFO", len(test_rows), "JSONL rows in test file")
    add_summary(summary, lines, "n_complex_jsonl_rows", "INFO", len(complex_rows), "JSONL rows in complex native file")

    if not train_path.exists() or not test_path.exists():
        lines.append("")
        lines.append("FAIL: train/test audit cannot run because train or test file is missing.")
    else:
        # name overlap
        train_names = Counter(str(x.get("name", f"line_{x.get('__line_no__', '')}")) for x in train_rows)
        test_names = Counter(str(x.get("name", f"line_{x.get('__line_no__', '')}")) for x in test_rows)
        dup_train_names = {k: v for k, v in train_names.items() if v > 1}
        dup_test_names = {k: v for k, v in test_names.items() if v > 1}
        name_overlap = sorted(set(train_names) & set(test_names))
        add_summary(summary, lines, "train_duplicate_names", "PASS" if not dup_train_names else "WARN", len(dup_train_names), "Duplicate names inside train set")
        add_summary(summary, lines, "test_duplicate_names", "PASS" if not dup_test_names else "WARN", len(dup_test_names), "Duplicate names inside test set")
        add_summary(summary, lines, "train_test_name_overlap", "PASS" if not name_overlap else "FAIL", len(name_overlap), "Exact record-name overlap between train and test")
        for name in name_overlap:
            overlaps.append({
                "check_name": "train_test_name_overlap",
                "source_a": "train_jsonl",
                "source_b": "test_jsonl",
                "key_type": "record_name",
                "key": name,
                "length": "",
                "count_a": train_names[name],
                "count_b": test_names[name],
                "examples_a": name,
                "examples_b": name,
            })

        # sequence duplicate and overlap checks
        for src, label in [("train_jsonl", "train"), ("test_jsonl", "test")]:
            for seq_type in ["chain", "concat_all_chains", "short_chain"]:
                d = duplicates(records, src, seq_type)
                add_summary(summary, lines, f"{label}_duplicate_{seq_type}_sequences", "PASS" if not d else "WARN", len(d), f"Duplicate {seq_type} naturalized sequences inside {label} set")

        train_chain = group(records, source="train_jsonl", seq_types={"chain"})
        test_chain = group(records, source="test_jsonl", seq_types={"chain"})
        train_concat = group(records, source="train_jsonl", seq_types={"concat_all_chains"})
        test_concat = group(records, source="test_jsonl", seq_types={"concat_all_chains"})
        train_short = group(records, source="train_jsonl", seq_types={"short_chain"})
        test_short = group(records, source="test_jsonl", seq_types={"short_chain"})
        complex_chain = group(records, source="complex_native_jsonl", seq_types={"chain", "concat_all_chains"})
        complex_short = group(records, source="complex_native_jsonl", seq_types={"short_chain"})

        checks = [
            ("train_test_chain_sequence_overlap", "train_jsonl", "test_jsonl", train_chain, test_chain, "exact_naturalized_chain_sequence", "FAIL"),
            ("train_test_concat_sequence_overlap", "train_jsonl", "test_jsonl", train_concat, test_concat, "exact_naturalized_concat_sequence", "FAIL"),
            ("train_test_short_chain_sequence_overlap", "train_jsonl", "test_jsonl", train_short, test_short, "exact_naturalized_short_chain_sequence", "WARN"),
            ("train_vs_complex_native_all_chains", "train_jsonl", "complex_native_jsonl", train_chain, complex_chain, "exact_naturalized_sequence", "WARN"),
            ("train_vs_complex_native_short_peptides", "train_jsonl", "complex_native_short_peptides", train_chain, complex_short, "exact_naturalized_sequence", "WARN"),
            ("test_vs_complex_native_all_chains", "test_jsonl", "complex_native_jsonl", test_chain, complex_chain, "exact_naturalized_sequence", "WARN"),
            ("test_vs_complex_native_short_peptides", "test_jsonl", "complex_native_short_peptides", test_chain, complex_short, "exact_naturalized_sequence", "WARN"),
        ]
        for check_name, source_a, source_b, ga, gb, key_type, bad_status in checks:
            n = add_overlap(overlaps, check_name, source_a, source_b, ga, gb, key_type)
            add_summary(summary, lines, check_name, "PASS" if n == 0 else bad_status, n, f"Exact naturalized overlap for {check_name}")

    lines.append("")
    lines.append("===== LIMITATION =====")
    lines.append("This audit checks exact record-name and exact naturalized-sequence overlaps only.")
    lines.append("It does not rule out homologous leakage; homology-level audit requires sequence alignment or clustering.")

    summary_path = out_dir / "train_test_leakage_audit_summary.csv"
    overlap_path = out_dir / "train_test_leakage_overlap_rows.csv"
    report_path = out_dir / "train_test_leakage_audit_report.txt"

    write_csv(summary_path, summary, ["check_name", "status", "count", "note"])
    write_csv(overlap_path, overlaps, ["check_name", "source_a", "source_b", "key_type", "key", "length", "count_a", "count_b", "examples_a", "examples_b"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("完成：train/test exact leakage audit")
    print("summary:", summary_path)
    print("overlaps:", overlap_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
