#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_available_data_leakage_audit.py

Available-data leakage audit for proteinmpnn-clean-v28.

This is NOT a full Rosetta train/test leakage audit, because the current clean
repository only contains nmethyl_data/test_set/test.jsonl and does not contain
training-set JSONL files. The script therefore performs only the checks that can
be performed with files currently present in the repository.

Checks:
1. Whether candidate train/valid files are present in this repository.
2. Internal duplicate names / sequences inside nmethyl_data/test_set/test.jsonl.
3. Exact naturalized sequence overlaps between monomer test set and 17 complex
   native sequences.
4. Exact naturalized sequence overlaps between monomer test set and generated
   design peptide sequences / best85 / af3_manifest.

Outputs:
- paper_clean_v28_outputs/leakage_audit/available_data_leakage_audit_report.txt
- paper_clean_v28_outputs/leakage_audit/available_data_leakage_audit_summary.csv
- paper_clean_v28_outputs/leakage_audit/available_data_leakage_overlap_rows.csv
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(".")
OUT_DIR = Path("paper_clean_v28_outputs/leakage_audit")

TEST_JSONL = Path("nmethyl_data/test_set/test.jsonl")
COMPLEX_JSONL = Path("17_complexes_native.jsonl")
GENERATED_ALL = Path("paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv")
GENERATED_BEST = Path("paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv")
AF3_MANIFEST = Path("paper_clean_v28_outputs/af3_manifest.csv")

NATURAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def naturalize(seq):
    """Map methyl lower-case letters to upper-case natural letters and remove non-AA separators."""
    if seq is None:
        return ""
    seq = str(seq).strip().upper()
    return "".join(ch for ch in seq if ch in NATURAL_AA or ch == "X")


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Failed to parse {path} line {line_no}: {e}") from e
            obj["__line_no__"] = line_no
            rows.append(obj)
    return rows


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def extract_jsonl_sequences(rows, source_name, max_short_len=30):
    """
    Return sequence records from ProteinMPNN-style jsonl rows.
    Each record has source, record_name, chain_id, seq_type, seq.
    """
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


def get_first(row, names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return ""


def extract_generated_sequences(path, source_name):
    rows = read_csv_rows(path)
    records = []
    for i, row in enumerate(rows):
        seq = get_first(row, [
            "design_peptide_natural_seq",
            "design_peptide_seq",
            "design_seq_naturalized",
            "design_seq",
            "sequence",
        ])
        seq = naturalize(seq)
        if not seq:
            continue
        target = get_first(row, ["target_name", "target", "suggested_job_name"])
        temp = get_first(row, ["temperature", "temp"])
        records.append({
            "source": source_name,
            "record_name": f"{target}|T{temp}|row{i}",
            "chain_id": "design_peptide",
            "seq_type": "design_peptide",
            "seq": seq,
            "length": len(seq),
        })
    return records


def find_candidate_train_files():
    candidates = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.as_posix())
        lower = rel.lower()
        if not lower.endswith((".jsonl", ".json", ".csv", ".txt")):
            continue
        if any(token in lower for token in ["train", "valid", "validation", "val", "split"]):
            if "paper_clean_v28_outputs" in lower and "leakage_audit" in lower:
                continue
            candidates.append(rel)
    return sorted(candidates)


def count_duplicates(records, source_filter=None, seq_type_filter=None, field="seq"):
    rows = [r for r in records]
    if source_filter is not None:
        rows = [r for r in rows if r["source"] == source_filter]
    if seq_type_filter is not None:
        rows = [r for r in rows if r["seq_type"] == seq_type_filter]
    c = Counter(r[field] for r in rows if r.get(field))
    return {k: v for k, v in c.items() if v > 1}


def group_by_seq(records, source=None, seq_types=None):
    out = defaultdict(list)
    for r in records:
        if source is not None and r["source"] != source:
            continue
        if seq_types is not None and r["seq_type"] not in seq_types:
            continue
        seq = r.get("seq", "")
        if seq:
            out[seq].append(r)
    return out


def add_overlap_rows(overlap_rows, source_a, source_b, group_a, group_b, key_type, check_name):
    overlap = sorted(set(group_a.keys()) & set(group_b.keys()))
    for seq in overlap:
        a_items = group_a[seq]
        b_items = group_b[seq]
        overlap_rows.append({
            "check_name": check_name,
            "source_a": source_a,
            "source_b": source_b,
            "key_type": key_type,
            "key": seq,
            "length": len(seq),
            "count_a": len(a_items),
            "count_b": len(b_items),
            "examples_a": ";".join(f"{r['record_name']}:{r['chain_id']}:{r['seq_type']}" for r in a_items[:5]),
            "examples_b": ";".join(f"{r['record_name']}:{r['chain_id']}:{r['seq_type']}" for r in b_items[:5]),
        })
    return len(overlap)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    overlap_rows = []
    report_lines = []

    def add_summary(check_name, status, count, note):
        summary.append({
            "check_name": check_name,
            "status": status,
            "count": count,
            "note": note,
        })
        report_lines.append(f"{check_name}: {status} | count={count} | {note}")

    report_lines.append("===== AVAILABLE-DATA LEAKAGE AUDIT =====")
    report_lines.append("This is not a full Rosetta train/test leakage audit because no train-set file is present in the current repository.")
    report_lines.append("")

    train_candidates = find_candidate_train_files()
    train_candidates_real = [p for p in train_candidates if "test_set/test.jsonl" not in p.replace("\\", "/")]
    if train_candidates_real:
        add_summary(
            "candidate_train_or_split_files_present",
            "WARN",
            len(train_candidates_real),
            "; ".join(train_candidates_real[:20]),
        )
    else:
        add_summary(
            "candidate_train_or_split_files_present",
            "INCOMPLETE",
            0,
            "No train/valid/split file found in repository; full train/test leakage audit cannot be completed.",
        )

    test_rows = read_jsonl(TEST_JSONL)
    complex_rows = read_jsonl(COMPLEX_JSONL)

    add_summary("test_jsonl_exists", "PASS" if TEST_JSONL.exists() else "FAIL", int(TEST_JSONL.exists()), str(TEST_JSONL))
    add_summary("complex_jsonl_exists", "PASS" if COMPLEX_JSONL.exists() else "FAIL", int(COMPLEX_JSONL.exists()), str(COMPLEX_JSONL))

    records = []
    records += extract_jsonl_sequences(test_rows, "monomer_test_jsonl")
    records += extract_jsonl_sequences(complex_rows, "complex_native_jsonl")
    records += extract_generated_sequences(GENERATED_ALL, "generated_all_designs")
    records += extract_generated_sequences(GENERATED_BEST, "generated_best_designs")
    records += extract_generated_sequences(AF3_MANIFEST, "af3_manifest")

    add_summary("n_monomer_test_records", "INFO", len(test_rows), "JSONL rows in nmethyl_data/test_set/test.jsonl")
    add_summary("n_complex_native_records", "INFO", len(complex_rows), "JSONL rows in 17_complexes_native.jsonl")
    add_summary("n_sequence_records_extracted", "INFO", len(records), "All extracted chain/design sequence records")

    # Internal duplicate checks.
    test_name_counter = Counter(str(r.get("name", f"line_{r.get('__line_no__', '')}")) for r in test_rows)
    dup_test_names = {k: v for k, v in test_name_counter.items() if v > 1}
    add_summary(
        "monomer_test_duplicate_names",
        "PASS" if not dup_test_names else "WARN",
        len(dup_test_names),
        "Duplicate test names" if dup_test_names else "No duplicate test names detected.",
    )

    test_chain_dups = count_duplicates(records, source_filter="monomer_test_jsonl", seq_type_filter="chain", field="seq")
    add_summary(
        "monomer_test_duplicate_chain_sequences",
        "PASS" if not test_chain_dups else "WARN",
        len(test_chain_dups),
        "Exact duplicate chain sequences inside monomer test set" if test_chain_dups else "No exact duplicate chain sequences detected inside monomer test set.",
    )

    # Cross-source exact sequence overlaps.
    test_chains = group_by_seq(records, source="monomer_test_jsonl", seq_types={"chain", "concat_all_chains"})
    complex_all = group_by_seq(records, source="complex_native_jsonl", seq_types={"chain", "concat_all_chains"})
    complex_short = group_by_seq(records, source="complex_native_jsonl", seq_types={"short_chain"})
    generated_all = group_by_seq(records, source="generated_all_designs", seq_types={"design_peptide"})
    generated_best = group_by_seq(records, source="generated_best_designs", seq_types={"design_peptide"})
    af3 = group_by_seq(records, source="af3_manifest", seq_types={"design_peptide"})

    n = add_overlap_rows(
        overlap_rows,
        "monomer_test_jsonl",
        "complex_native_jsonl",
        test_chains,
        complex_all,
        "exact_naturalized_sequence",
        "monomer_test_vs_complex_native_all_chains",
    )
    add_summary(
        "monomer_test_vs_complex_native_all_chains",
        "PASS" if n == 0 else "WARN",
        n,
        "Exact naturalized sequence overlaps between monomer test and complex native all/chain sequences.",
    )

    n = add_overlap_rows(
        overlap_rows,
        "monomer_test_jsonl",
        "complex_native_short_peptides",
        test_chains,
        complex_short,
        "exact_naturalized_sequence",
        "monomer_test_vs_complex_native_short_peptides",
    )
    add_summary(
        "monomer_test_vs_complex_native_short_peptides",
        "PASS" if n == 0 else "WARN",
        n,
        "Exact naturalized sequence overlaps between monomer test and complex native short peptides.",
    )

    for label, group in [
        ("generated_all_designs", generated_all),
        ("generated_best_designs", generated_best),
        ("af3_manifest", af3),
    ]:
        n = add_overlap_rows(
            overlap_rows,
            "monomer_test_jsonl",
            label,
            test_chains,
            group,
            "exact_naturalized_sequence",
            f"monomer_test_vs_{label}",
        )
        add_summary(
            f"monomer_test_vs_{label}",
            "PASS" if n == 0 else "WARN",
            n,
            f"Exact naturalized sequence overlaps between monomer test and {label} peptide sequences.",
        )

    # best85 / af3 consistency check.
    n = add_overlap_rows(
        overlap_rows,
        "generated_best_designs",
        "af3_manifest",
        generated_best,
        af3,
        "exact_naturalized_sequence",
        "generated_best_vs_af3_manifest",
    )
    add_summary(
        "generated_best_vs_af3_manifest_sequence_overlap",
        "INFO",
        n,
        "Exact naturalized sequence overlap count between best_designs and af3_manifest; expected to be nonzero/high because af3_manifest is derived from best85.",
    )

    report_lines.append("")
    report_lines.append("===== LIMITATION =====")
    report_lines.append("Full Rosetta train/test leakage audit cannot be completed from this repository alone because no training set file is present.")
    report_lines.append("To complete the full audit, provide the train/valid JSONL or a training data manifest used to train frankenstein_v28.pt.")
    report_lines.append("")
    report_lines.append("===== OUTPUTS =====")
    report_lines.append(str(OUT_DIR / "available_data_leakage_audit_summary.csv"))
    report_lines.append(str(OUT_DIR / "available_data_leakage_overlap_rows.csv"))

    write_csv(
        OUT_DIR / "available_data_leakage_audit_summary.csv",
        summary,
        ["check_name", "status", "count", "note"],
    )
    write_csv(
        OUT_DIR / "available_data_leakage_overlap_rows.csv",
        overlap_rows,
        [
            "check_name",
            "source_a",
            "source_b",
            "key_type",
            "key",
            "length",
            "count_a",
            "count_b",
            "examples_a",
            "examples_b",
        ],
    )

    report_path = OUT_DIR / "available_data_leakage_audit_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("完成：available-data leakage audit")
    print("summary:", OUT_DIR / "available_data_leakage_audit_summary.csv")
    print("overlaps:", OUT_DIR / "available_data_leakage_overlap_rows.csv")
    print("report:", report_path)
    print("")
    print("重要限制：当前仓库没有 train set 文件，因此完整 train/test leakage audit 仍然需要训练集或训练清单。")


if __name__ == "__main__":
    main()
