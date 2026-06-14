#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_audit_complex_chain_mapping.py

作用：
1. 读取 best85 代表结构表；
2. 读取 17_complexes_native.jsonl；
3. 解析 HighFold PDB 每条链的序列和长度；
4. 判断预测结构中哪条链是 design peptide；
5. 判断 native/reference 中哪条链是 native peptide；
6. 输出链映射审计表，为后续 RMSD 严格对齐做准备。

重要修正：
- HighFold PDB 中，N-甲基化/非标准残基可能写在 HETATM 里；
- 因此解析链长度和肽链时必须同时读取 ATOM 和 HETATM；
- 复合物 design sequence 本来可能和 native peptide sequence 不同，所以不能用序列完全一致作为严格条件；
- 更可靠的严格条件是 peptide chain residue count 与 design sequence 长度一致。
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict, Counter


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",

    # HighFold / modified residue fallback.
    # 这些非标准残基主要用于链长和位置对齐；如果无法确定天然氨基酸，就记为 X。
    "NCY": "C",
    "GNC": "Q",
    "MMO": "R",
    "UNK": "X",
}


def naturalize_seq(seq):
    return (seq or "").strip().upper()


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
            for k in row.keys():
                if k not in seen:
                    fieldnames.append(k)
                    seen.add(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def get_first(row, names):
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return ""


def seq_identity(a, b):
    a = naturalize_seq(a)
    b = naturalize_seq(b)
    if not a or not b:
        return 0.0

    n = min(len(a), len(b))
    if n == 0:
        return 0.0

    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / max(len(a), len(b))


def parse_pdb_chain_sequences(pdb_path):
    """
    用 CA 原子按 residue 顺序重建每条链序列。
    同时读取 ATOM 和 HETATM，因为非标准/甲基化残基常在 HETATM 中。
    """
    chains = defaultdict(list)
    resname_chains = defaultdict(list)
    seen = set()

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            resname = line[17:20].strip()
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode)

            if key in seen:
                continue

            seen.add(key)

            aa = AA3_TO_1.get(resname, "X")
            chains[chain_id].append(aa)
            resname_chains[chain_id].append(resname)

    seqs = {chain: "".join(aas) for chain, aas in sorted(chains.items())}
    resnames = {chain: ",".join(names) for chain, names in sorted(resname_chains.items())}
    return seqs, resnames


def load_native_jsonl(path):
    native_by_target = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            name = str(obj.get("name", "")).upper()

            if not name:
                continue

            chains = {}

            for k, v in obj.items():
                m = re.match(r"^seq_chain_(.+)$", k)
                if m:
                    chain_id = m.group(1)
                    chains[chain_id] = naturalize_seq(v)

            native_by_target[name] = chains

    return native_by_target


def select_predicted_peptide_chain(pred_chains, design_seq):
    design_upper = naturalize_seq(design_seq)
    design_len = len(design_upper)

    length_matches = [
        (chain, seq) for chain, seq in pred_chains.items()
        if len(seq) == design_len
    ]

    if len(length_matches) == 1:
        chain, seq = length_matches[0]
        return chain, "unique_length_match_to_design", seq_identity(seq, design_upper)

    if len(length_matches) > 1:
        scored = [
            (seq_identity(seq, design_upper), chain, seq)
            for chain, seq in length_matches
        ]
        scored.sort(reverse=True)
        ident, chain, seq = scored[0]
        return chain, "multiple_length_match_choose_best_identity", ident

    candidates = []

    for chain, seq in pred_chains.items():
        ident = seq_identity(seq, design_upper)
        length_diff = abs(len(seq) - design_len)
        score = ident - length_diff * 0.03
        candidates.append((score, ident, length_diff, chain, seq))

    if not candidates:
        return "", "no_predicted_chains", 0.0

    candidates.sort(reverse=True)
    score, ident, length_diff, chain, seq = candidates[0]
    return chain, "no_length_match_choose_best_identity", ident


def select_native_peptide_chain(native_chains, design_seq, pred_peptide_len):
    design_upper = naturalize_seq(design_seq)
    design_len = len(design_upper)

    target_len = pred_peptide_len if pred_peptide_len else design_len

    length_matches = [
        (chain, seq) for chain, seq in native_chains.items()
        if len(seq) == target_len
    ]

    if len(length_matches) == 1:
        chain, seq = length_matches[0]
        return chain, "unique_length_match_to_predicted_peptide", seq_identity(seq, design_upper)

    if len(length_matches) > 1:
        scored = [
            (seq_identity(seq, design_upper), chain, seq)
            for chain, seq in length_matches
        ]
        scored.sort(reverse=True)
        ident, chain, seq = scored[0]
        return chain, "multiple_length_match_choose_best_identity", ident

    scored = [
        (
            seq_identity(seq, design_upper) - abs(len(seq) - target_len) * 0.03,
            seq_identity(seq, design_upper),
            abs(len(seq) - target_len),
            chain,
            seq,
        )
        for chain, seq in native_chains.items()
    ]

    if not scored:
        return "", "no_native_chains", 0.0

    scored.sort(reverse=True)
    score, ident, length_diff, chain, seq = scored[0]
    return chain, "no_length_match_choose_best_identity", ident


def make_chain_summary(chains):
    return ";".join(
        f"{chain}:{len(seq)}:{seq}"
        for chain, seq in chains.items()
    )


def make_resname_summary(resnames):
    return ";".join(
        f"{chain}:{names}"
        for chain, names in resnames.items()
    )


def make_receptor_mapping(pred_chains, native_chains, pred_pep_chain, native_pep_chain):
    pred_receptors = {
        c: s for c, s in pred_chains.items()
        if c != pred_pep_chain
    }
    native_receptors = {
        c: s for c, s in native_chains.items()
        if c != native_pep_chain
    }

    mappings = []

    for pc, ps in pred_receptors.items():
        best = None

        for nc, ns in native_receptors.items():
            ident = seq_identity(ps, ns)
            length_diff = abs(len(ps) - len(ns))
            score = ident - length_diff * 0.02
            item = (score, ident, length_diff, nc, ns)

            if best is None or item > best:
                best = item

        if best is None:
            mappings.append(f"{pc}->NA")
        else:
            score, ident, length_diff, nc, ns = best
            mappings.append(f"{pc}->{nc}(ident={ident:.3f},len_diff={length_diff})")

    return ";".join(mappings)


def main():
    rep_path = Path("paper_clean_v28_outputs/structure_metrics/complex_best85_highfold_representative.csv")
    native_path = Path("17_complexes_native.jsonl")
    out_dir = Path("paper_clean_v28_outputs/structure_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_rows = read_csv(rep_path)
    native_by_target = load_native_jsonl(native_path)

    audit_rows = []

    for i, row in enumerate(rep_rows):
        target = get_first(row, [
            "target_name",
            "target",
            "match_target_name",
            "highfold_target_name",
        ]).upper()

        temp = norm_temp(get_first(row, [
            "temperature",
            "match_temperature",
            "highfold_temperature",
        ]))

        design_seq = get_first(row, [
            "design_peptide_seq",
            "design_seq",
            "match_design_seq",
            "highfold_design_seq_from_filename",
        ])

        status = get_first(row, ["structure_match_status"])
        pdb_path = get_first(row, [
            "highfold_pdb_path",
            "representative_pdb_path",
            "pdb_path",
        ])

        pdb_file = get_first(row, [
            "highfold_pdb_file",
            "representative_pdb_file",
            "pdb_file",
        ])

        pred_chains = {}
        pred_resnames = {}
        native_chains = native_by_target.get(target, {})

        pred_pep_chain = ""
        pred_pep_method = ""
        pred_pep_ident = 0.0
        native_pep_chain = ""
        native_pep_method = ""
        native_pep_ident = 0.0
        receptor_mapping = ""

        problem = []

        if status == "missing_pdb" or not pdb_path:
            problem.append("missing_pdb")
        else:
            p = Path(pdb_path)
            if not p.exists():
                problem.append("pdb_path_not_found")
            else:
                pred_chains, pred_resnames = parse_pdb_chain_sequences(p)

        if not native_chains:
            problem.append("native_target_not_found")

        if pred_chains and design_seq:
            pred_pep_chain, pred_pep_method, pred_pep_ident = select_predicted_peptide_chain(
                pred_chains,
                design_seq,
            )
        else:
            if not pred_chains:
                problem.append("no_predicted_chains")
            if not design_seq:
                problem.append("empty_design_seq")

        if native_chains and pred_pep_chain:
            pred_pep_len = len(pred_chains[pred_pep_chain])
            native_pep_chain, native_pep_method, native_pep_ident = select_native_peptide_chain(
                native_chains,
                design_seq,
                pred_pep_len,
            )
        else:
            if not native_chains:
                problem.append("no_native_chains")
            if not pred_pep_chain:
                problem.append("no_predicted_peptide_chain")

        if pred_chains and native_chains and pred_pep_chain and native_pep_chain:
            receptor_mapping = make_receptor_mapping(
                pred_chains,
                native_chains,
                pred_pep_chain,
                native_pep_chain,
            )

        design_len = len(naturalize_seq(design_seq))
        pred_pep_len = len(pred_chains.get(pred_pep_chain, ""))
        native_pep_len = len(native_chains.get(native_pep_chain, ""))

        if pred_pep_len != design_len and status != "missing_pdb":
            problem.append("predicted_peptide_length_not_equal_design_length")

        if native_pep_len != design_len and native_chains:
            problem.append("native_peptide_length_not_equal_design_length")

        if not problem:
            final_status = "ok"
        else:
            final_status = ";".join(sorted(set(problem)))

        audit_rows.append({
            "row_index": i,
            "target_name": target,
            "temperature": temp,
            "design_seq": design_seq,
            "design_seq_naturalized": naturalize_seq(design_seq),
            "design_len": design_len,
            "structure_match_status": status,
            "pdb_file": pdb_file,
            "pdb_path": pdb_path,
            "predicted_chain_summary": make_chain_summary(pred_chains),
            "predicted_chain_resname_summary": make_resname_summary(pred_resnames),
            "native_chain_summary": make_chain_summary(native_chains),
            "predicted_peptide_chain": pred_pep_chain,
            "predicted_peptide_len": pred_pep_len,
            "predicted_peptide_method": pred_pep_method,
            "predicted_peptide_identity_to_design": f"{pred_pep_ident:.4f}",
            "native_peptide_chain": native_pep_chain,
            "native_peptide_len": native_pep_len,
            "native_peptide_method": native_pep_method,
            "native_peptide_identity_to_design": f"{native_pep_ident:.4f}",
            "receptor_chain_mapping_candidate": receptor_mapping,
            "chain_mapping_status": final_status,
        })

    write_csv(
        out_dir / "complex_chain_mapping_audit.csv",
        audit_rows,
    )

    problem_rows = [
        r for r in audit_rows
        if r["chain_mapping_status"] != "ok"
    ]

    write_csv(
        out_dir / "complex_chain_mapping_problem_rows.csv",
        problem_rows,
    )

    c = Counter(r["chain_mapping_status"] for r in audit_rows)

    print("完成：复合物 best85 链映射审计")
    print("输入 best85 行数:", len(rep_rows))
    print("输出 audit 行数:", len(audit_rows))
    print("状态统计:")
    for k, v in c.items():
        print(f"  {k}: {v}")
    print("问题行数:", len(problem_rows))
    print("输出:")
    print(out_dir / "complex_chain_mapping_audit.csv")
    print(out_dir / "complex_chain_mapping_problem_rows.csv")


if __name__ == "__main__":
    main()
