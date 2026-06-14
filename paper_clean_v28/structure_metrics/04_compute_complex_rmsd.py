#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_compute_complex_rmsd.py

作用：
1. 读取 complex_chain_mapping_audit.csv；
2. 读取 17_complexes_native.jsonl 的 native/reference 坐标；
3. 读取 HighFold PDB 预测结构；
4. 用 receptor CA 原子做刚体对齐；
5. 计算 peptide RMSD / backbone RMSD / complex RMSD；
6. 合并 HighFold pLDDT、ipTM、inter-PAE 等结构置信度；
7. 输出复合物 best85 的结构指标表。

注意：
- 只计算 chain_mapping_status == ok 的行；
- missing_pdb 的 4 条会保留在输出表中，但 RMSD 为空；
- 设计肽和 native 肽序列可能不同，因此 all-atom RMSD 不作为主指标；
- 主指标推荐使用 CA RMSD 和 backbone RMSD。
"""

import csv
import json
import math
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

import numpy as np


BACKBONE_ATOMS = ["N", "CA", "C"]


def norm_temp(x):
    if x is None or x == "":
        return ""
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def safe_float(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def fmt(x, ndigits=4):
    if x is None:
        return ""
    try:
        if math.isnan(float(x)):
            return ""
    except Exception:
        pass
    return f"{float(x):.{ndigits}f}"


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


def xyz_ok(x):
    if x is None:
        return False
    if len(x) != 3:
        return False
    for v in x:
        if v is None:
            return False
        try:
            vf = float(v)
            if math.isnan(vf):
                return False
        except Exception:
            return False
    return True


def to_xyz(x):
    return np.array([float(x[0]), float(x[1]), float(x[2])], dtype=float)


def rmsd(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if len(P) == 0:
        return None
    diff = P - Q
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def kabsch_fit(P, Q):
    """
    返回 R, t，使得 P @ R + t 尽量贴近 Q。

    注意：
    本函数使用 row-vector 坐标约定，即点坐标形状为 (N, 3)，变换为:
        P_aligned = P @ R + t

    对应的 Kabsch 解应为:
        H = P0.T @ Q0
        U, S, Vt = svd(H)
        R = U @ Vt

    旧版 R = Vt.T @ U.T 对 row-vector 约定是反的，会导致 receptor fit RMSD 异常偏大。
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if len(P) < 3:
        raise ValueError("Kabsch 至少需要 3 个点")

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


def apply_transform(P, R, t):
    P = np.asarray(P, dtype=float)
    return P @ R + t


def parse_pdb_structure(pdb_path):
    """
    返回：
    chains[chain_id] = [
        {"resname": "...", "resseq": "...", "icode": "...", "atoms": {"CA": np.array([...])}}
    ]

    同时读取 ATOM 和 HETATM。
    """
    chains = defaultdict(list)
    index = {}

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom_name = line[12:16].strip()
            if atom_name not in {"N", "CA", "C", "O"}:
                continue

            resname = line[17:20].strip()
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode)

            try:
                coord = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                )
            except Exception:
                continue

            if key not in index:
                index[key] = len(chains[chain_id])
                chains[chain_id].append({
                    "resname": resname,
                    "resseq": resseq,
                    "icode": icode,
                    "atoms": {},
                })

            chains[chain_id][index[key]]["atoms"][atom_name] = coord

    return dict(chains)


def load_native_structures(jsonl_path):
    """
    读取 ProteinMPNN 风格 jsonl。

    支持两种格式：
    1. 顶层坐标格式：
       seq_chain_A
       N_chain_A
       CA_chain_A
       C_chain_A
       O_chain_A

    2. 嵌套坐标格式：
       coords_chain_A:
           N_chain_A
           CA_chain_A
           C_chain_A
           O_chain_A

    当前 17_complexes_native.jsonl 是第 1 种顶层格式。
    """
    native_by_target = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            target = str(obj.get("name", "")).upper()
            if not target:
                continue

            target_chains = {}

            for key, seq in obj.items():
                m = re.match(r"^seq_chain_(.+)$", key)
                if not m:
                    continue

                chain_id = m.group(1)
                seq = str(seq)

                residues = []
                for i, aa in enumerate(seq):
                    residues.append({
                        "resname": aa,
                        "resseq": str(i + 1),
                        "icode": "",
                        "atoms": {},
                    })

                # 优先读取顶层坐标字段：N_chain_A / CA_chain_A / C_chain_A / O_chain_A
                for atom in ["N", "CA", "C", "O"]:
                    arr = obj.get(f"{atom}_chain_{chain_id}", None)

                    # 兼容旧的嵌套 coords_chain_A 格式
                    if arr is None:
                        coords_obj = obj.get(f"coords_chain_{chain_id}", {})
                        if isinstance(coords_obj, dict):
                            arr = coords_obj.get(f"{atom}_chain_{chain_id}", None)
                            if arr is None:
                                arr = coords_obj.get(atom, None)

                    if arr is None:
                        continue

                    for i, xyz in enumerate(arr):
                        if i >= len(residues):
                            break
                        if xyz_ok(xyz):
                            residues[i]["atoms"][atom] = to_xyz(xyz)

                target_chains[chain_id] = residues

            native_by_target[target] = target_chains

    return native_by_target


def parse_receptor_mapping(mapping_text):
    """
    解析：
    A->A(ident=1.000,len_diff=0);C->B(...)
    """
    pairs = []
    if not mapping_text:
        return pairs

    for part in str(mapping_text).split(";"):
        part = part.strip()
        if not part or "->" not in part:
            continue

        left, right = part.split("->", 1)
        pred_chain = left.strip()
        native_chain = right.split("(", 1)[0].strip()

        if native_chain and native_chain != "NA":
            pairs.append((pred_chain, native_chain))

    return pairs


def collect_paired_atoms(pred_residues, native_residues, atoms):
    P = []
    Q = []

    n = min(len(pred_residues), len(native_residues))

    for i in range(n):
        pa = pred_residues[i]["atoms"]
        qa = native_residues[i]["atoms"]

        for atom in atoms:
            if atom in pa and atom in qa:
                P.append(pa[atom])
                Q.append(qa[atom])

    return P, Q


def mean_values(values):
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return mean(xs)


def get_pair_score(row, metric, a, b):
    """
    metric examples:
    iptm, inter_pae, ipae, ipsae
    """
    candidates = [
        f"highfold_{metric}_{a}_{b}",
        f"highfold_{metric}_{b}_{a}",
        f"{metric}_{a}_{b}",
        f"{metric}_{b}_{a}",
    ]

    for k in candidates:
        if k in row and row[k] not in ("", None):
            return row[k]

    return ""


def summarize(rows, group_key, out_path):
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(group_key, "")].append(r)

    out = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        ok_items = [r for r in items if r.get("rmsd_status") == "ok"]

        def vals(col):
            xs = []
            for r in ok_items:
                v = safe_float(r.get(col))
                if v is not None:
                    xs.append(v)
            return xs

        ca_vals = vals("peptide_ca_rmsd_after_receptor_fit")
        bb_vals = vals("peptide_backbone_rmsd_after_receptor_fit")
        plddt_vals = vals("highfold_plddt")
        iptm_vals = vals("peptide_receptor_iptm_mean")
        pae_vals = vals("peptide_receptor_inter_pae_mean")

        out.append({
            group_key: key,
            "n_rows": len(items),
            "n_ok": len(ok_items),
            "n_missing_or_failed": len(items) - len(ok_items),
            "mean_peptide_ca_rmsd_after_receptor_fit": fmt(mean(ca_vals), 4) if ca_vals else "",
            "median_peptide_ca_rmsd_after_receptor_fit": fmt(median(ca_vals), 4) if ca_vals else "",
            "mean_peptide_backbone_rmsd_after_receptor_fit": fmt(mean(bb_vals), 4) if bb_vals else "",
            "median_peptide_backbone_rmsd_after_receptor_fit": fmt(median(bb_vals), 4) if bb_vals else "",
            "success_rate_ca_rmsd_lt_2": fmt(sum(1 for x in ca_vals if x < 2.0) / len(ca_vals), 4) if ca_vals else "",
            "success_rate_ca_rmsd_lt_5": fmt(sum(1 for x in ca_vals if x < 5.0) / len(ca_vals), 4) if ca_vals else "",
            "mean_highfold_plddt": fmt(mean(plddt_vals), 4) if plddt_vals else "",
            "mean_peptide_receptor_iptm": fmt(mean(iptm_vals), 4) if iptm_vals else "",
            "mean_peptide_receptor_inter_pae": fmt(mean(pae_vals), 4) if pae_vals else "",
        })

    write_csv(out_path, out)



AA3_TO_1_FOR_ALIGN = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
    "NCY": "C", "GNC": "Q", "MMO": "R", "UNK": "X",
}


def residue_to_aa_for_align(residue):
    name = str(residue.get("resname", "")).strip().upper()
    if len(name) == 1:
        return name
    return AA3_TO_1_FOR_ALIGN.get(name, "X")


def residue_sequence_for_align(residues):
    return "".join(residue_to_aa_for_align(r) for r in residues)


def best_ungapped_sequence_pairs(pred_seq, native_seq):
    """
    找 predicted chain 和 native chain 的最佳无 gap offset 对齐。
    这个专门用来处理 predicted receptor 前面多 His-tag/linker 的情况。

    返回:
    - pred/native residue index pairs
    - identity
    - offset
    - n_overlap
    """
    pred_seq = str(pred_seq)
    native_seq = str(native_seq)

    if not pred_seq or not native_seq:
        return [], 0.0, 0, 0

    best = None

    # offset = pred_index - native_index
    for offset in range(-len(native_seq) + 1, len(pred_seq)):
        p_start = max(0, offset)
        n_start = max(0, -offset)
        n_overlap = min(len(pred_seq) - p_start, len(native_seq) - n_start)

        if n_overlap <= 0:
            continue

        matches = 0
        for k in range(n_overlap):
            if pred_seq[p_start + k] == native_seq[n_start + k]:
                matches += 1

        identity = matches / n_overlap

        # 优先：匹配数量多；其次：identity 高；再次：overlap 长
        score = (matches, identity, n_overlap)

        if best is None or score > best[0]:
            pairs = [(p_start + k, n_start + k) for k in range(n_overlap)]
            best = (score, pairs, identity, offset, n_overlap)

    if best is None:
        return [], 0.0, 0, 0

    score, pairs, identity, offset, n_overlap = best
    return pairs, identity, offset, n_overlap


def collect_aligned_atoms_by_pairs(pred_residues, native_residues, index_pairs, atoms):
    P = []
    Q = []

    for pi, ni in index_pairs:
        if pi >= len(pred_residues) or ni >= len(native_residues):
            continue

        pa = pred_residues[pi]["atoms"]
        qa = native_residues[ni]["atoms"]

        for atom in atoms:
            if atom in pa and atom in qa:
                P.append(pa[atom])
                Q.append(qa[atom])

    return P, Q


def infer_best_receptor_mapping_and_pairs(
    pred_struct,
    native_struct,
    pred_pep_chain,
    native_pep_chain,
    design_len,
):
    """
    自动选择 receptor chain mapping，并且跳过 predicted receptor 的 His-tag/linker。

    逻辑：
    1. predicted peptide chain 和 native peptide chain 不参与 receptor fit；
    2. 额外的 native 短 peptide chain 也排除；
    3. 对 predicted receptor chains 和 native receptor chains 做 one-to-one permutations；
    4. 每个 chain pair 用最佳 offset sequence alignment 配对残基；
    5. 选择 receptor CA fit RMSD 最小的一组 mapping。
    """
    import itertools

    try:
        design_len = int(float(design_len))
    except Exception:
        design_len = 0

    pred_receptors = []
    for c, residues in pred_struct.items():
        if str(c) == str(pred_pep_chain):
            continue
        if design_len and len(residues) <= design_len + 2:
            continue
        pred_receptors.append(c)

    native_receptors = []
    for c, residues in native_struct.items():
        if str(c) == str(native_pep_chain):
            continue
        if design_len and len(residues) <= design_len + 2:
            continue
        native_receptors.append(c)

    pred_receptors = sorted(pred_receptors)
    native_receptors = sorted(native_receptors)

    if not pred_receptors or not native_receptors:
        return [], [], [], "no_receptor_candidates"

    candidates = []

    if len(pred_receptors) <= len(native_receptors):
        for native_perm in itertools.permutations(native_receptors, len(pred_receptors)):
            candidates.append(list(zip(pred_receptors, native_perm)))
    else:
        for pred_perm in itertools.permutations(pred_receptors, len(native_receptors)):
            candidates.append(list(zip(pred_perm, native_receptors)))

    best = None

    for mapping in candidates:
        fit_P = []
        fit_Q = []
        details = []

        for pc, nc in mapping:
            pred_res = pred_struct.get(pc, [])
            native_res = native_struct.get(nc, [])

            pred_seq = residue_sequence_for_align(pred_res)
            native_seq = residue_sequence_for_align(native_res)

            index_pairs, ident, offset, n_overlap = best_ungapped_sequence_pairs(
                pred_seq,
                native_seq,
            )

            P, Q = collect_aligned_atoms_by_pairs(
                pred_res,
                native_res,
                index_pairs,
                ["CA"],
            )

            fit_P.extend(P)
            fit_Q.extend(Q)

            details.append(
                f"{pc}->{nc}(ident={ident:.3f},offset={offset},overlap={n_overlap},ca_pairs={len(P)})"
            )

        if len(fit_P) < 3:
            continue

        try:
            R_test, t_test = kabsch_fit(np.array(fit_P), np.array(fit_Q))
            aligned = apply_transform(np.array(fit_P), R_test, t_test)
            receptor_fit = rmsd(aligned, np.array(fit_Q))
        except Exception:
            continue

        score = receptor_fit

        if best is None or score < best[0]:
            best = (
                score,
                mapping,
                fit_P,
                fit_Q,
                ";".join(details),
            )

    if best is None:
        return [], [], [], "no_valid_receptor_mapping"

    score, mapping, fit_P, fit_Q, detail_text = best
    mapping_text = ";".join(f"{pc}->{nc}" for pc, nc in mapping)
    detail_text = f"best_receptor_fit={score:.4f};" + detail_text

    return mapping, fit_P, fit_Q, detail_text


def main():
    audit_path = Path("paper_clean_v28_outputs/structure_metrics/complex_chain_mapping_audit.csv")
    rep_path = Path("paper_clean_v28_outputs/structure_metrics/complex_best85_highfold_representative.csv")
    native_path = Path("17_complexes_native.jsonl")
    out_dir = Path("paper_clean_v28_outputs/structure_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = read_csv(audit_path)
    rep_rows = read_csv(rep_path)
    native_by_target = load_native_structures(native_path)

    final_rows = []

    for i, audit in enumerate(audit_rows):
        rep = rep_rows[i] if i < len(rep_rows) else {}

        target = audit.get("target_name", "").upper()
        temp = norm_temp(audit.get("temperature"))
        design_seq = audit.get("design_seq", "")
        design_len = audit.get("design_len", "")

        pdb_path = audit.get("pdb_path", "")
        pred_pep_chain = str(audit.get("predicted_peptide_chain", "")).strip()
        native_pep_chain = str(audit.get("native_peptide_chain", "")).strip()
        chain_mapping_status = audit.get("chain_mapping_status", "")

        out = {
            "row_index": audit.get("row_index", i),
            "target_name": target,
            "temperature": temp,
            "design_seq": design_seq,
            "design_len": design_len,
            "structure_match_status": audit.get("structure_match_status", ""),
            "chain_mapping_status": chain_mapping_status,
            "pdb_file": audit.get("pdb_file", ""),
            "pdb_path": pdb_path,
            "predicted_peptide_chain": pred_pep_chain,
            "native_peptide_chain": native_pep_chain,
            "receptor_chain_mapping_candidate": audit.get("receptor_chain_mapping_candidate", ""),
        }

        # 合并 HighFold 全局和链级指标
        out["highfold_plddt"] = rep.get("highfold_plddt", "")
        out["highfold_pdb_total_residue_count"] = rep.get("highfold_pdb_total_residue_count", "")
        out["highfold_pdb_chain_residue_counts"] = rep.get("highfold_pdb_chain_residue_counts", "")

        receptor_pairs = parse_receptor_mapping(audit.get("receptor_chain_mapping_candidate", ""))

        if pred_pep_chain:
            out["peptide_chain_plddt"] = rep.get(f"highfold_chain_{pred_pep_chain}_plddt", "")
            out["peptide_chain_ptm"] = rep.get(f"highfold_chain_{pred_pep_chain}_ptm", "")
            out["peptide_chain_iptm"] = rep.get(f"highfold_chain_{pred_pep_chain}_iptm", "")
            out["peptide_chain_pae"] = rep.get(f"highfold_chain_{pred_pep_chain}_pae", "")

        receptor_plddts = []
        peptide_receptor_iptms = []
        peptide_receptor_inter_paes = []
        peptide_receptor_ipaes = []

        for pred_rec_chain, native_rec_chain in receptor_pairs:
            receptor_plddts.append(rep.get(f"highfold_chain_{pred_rec_chain}_plddt", ""))
            peptide_receptor_iptms.append(get_pair_score(rep, "iptm", pred_pep_chain, pred_rec_chain))
            peptide_receptor_inter_paes.append(get_pair_score(rep, "inter_pae", pred_pep_chain, pred_rec_chain))
            peptide_receptor_ipaes.append(get_pair_score(rep, "ipae", pred_pep_chain, pred_rec_chain))

        out["receptor_chain_plddt_mean"] = fmt(mean_values(receptor_plddts), 4)
        out["peptide_receptor_iptm_mean"] = fmt(mean_values(peptide_receptor_iptms), 4)
        out["peptide_receptor_inter_pae_mean"] = fmt(mean_values(peptide_receptor_inter_paes), 4)
        out["peptide_receptor_ipae_mean"] = fmt(mean_values(peptide_receptor_ipaes), 4)

        if chain_mapping_status != "ok":
            out["rmsd_status"] = "skip_not_ok_chain_mapping"
            final_rows.append(out)
            continue

        try:
            pred_struct = parse_pdb_structure(pdb_path)
        except Exception as e:
            out["rmsd_status"] = f"failed_parse_pdb:{repr(e)}"
            final_rows.append(out)
            continue

        native_struct = native_by_target.get(target, {})

        if not native_struct:
            out["rmsd_status"] = "native_target_not_found"
            final_rows.append(out)
            continue

        # 1. receptor CA 对齐
        # 重要：HighFold predicted receptor 可能带 N-terminal His-tag/linker，
        # 所以不能按 residue index 直接硬配，必须先做 sequence offset alignment。
        robust_mapping, fit_P, fit_Q, receptor_alignment_detail = infer_best_receptor_mapping_and_pairs(
            pred_struct=pred_struct,
            native_struct=native_struct,
            pred_pep_chain=pred_pep_chain,
            native_pep_chain=native_pep_chain,
            design_len=design_len,
        )

        out["receptor_chain_mapping_used"] = ";".join(
            f"{pc}->{nc}" for pc, nc in robust_mapping
        )
        out["receptor_alignment_detail"] = receptor_alignment_detail
        out["n_receptor_ca_fit_pairs"] = len(fit_P)

        if len(fit_P) < 3:
            out["rmsd_status"] = "insufficient_receptor_ca_fit_pairs_after_sequence_alignment"
            final_rows.append(out)
            continue

        R, t = kabsch_fit(np.array(fit_P), np.array(fit_Q))
        fit_P_aligned = apply_transform(np.array(fit_P), R, t)
        out["receptor_ca_fit_rmsd"] = fmt(rmsd(fit_P_aligned, np.array(fit_Q)), 4)

        # 2. peptide RMSD after receptor fit
        if pred_pep_chain not in pred_struct:
            out["rmsd_status"] = "predicted_peptide_chain_not_found"
            final_rows.append(out)
            continue

        if native_pep_chain not in native_struct:
            out["rmsd_status"] = "native_peptide_chain_not_found"
            final_rows.append(out)
            continue

        pep_P_ca, pep_Q_ca = collect_paired_atoms(
            pred_struct[pred_pep_chain],
            native_struct[native_pep_chain],
            ["CA"],
        )

        pep_P_bb, pep_Q_bb = collect_paired_atoms(
            pred_struct[pred_pep_chain],
            native_struct[native_pep_chain],
            BACKBONE_ATOMS,
        )

        out["n_peptide_ca_pairs"] = len(pep_P_ca)
        out["n_peptide_backbone_atom_pairs"] = len(pep_P_bb)

        if len(pep_P_ca) >= 1:
            pep_P_ca_aligned = apply_transform(np.array(pep_P_ca), R, t)
            out["peptide_ca_rmsd_after_receptor_fit"] = fmt(
                rmsd(pep_P_ca_aligned, np.array(pep_Q_ca)),
                4,
            )

        if len(pep_P_bb) >= 1:
            pep_P_bb_aligned = apply_transform(np.array(pep_P_bb), R, t)
            out["peptide_backbone_rmsd_after_receptor_fit"] = fmt(
                rmsd(pep_P_bb_aligned, np.array(pep_Q_bb)),
                4,
            )

        # 3. peptide self-superposed RMSD
        if len(pep_P_ca) >= 3:
            R_pep, t_pep = kabsch_fit(np.array(pep_P_ca), np.array(pep_Q_ca))
            pep_P_ca_self = apply_transform(np.array(pep_P_ca), R_pep, t_pep)
            out["peptide_ca_rmsd_self_superposed"] = fmt(
                rmsd(pep_P_ca_self, np.array(pep_Q_ca)),
                4,
            )

            if len(pep_P_bb) >= 1:
                pep_P_bb_self = apply_transform(np.array(pep_P_bb), R_pep, t_pep)
                out["peptide_backbone_rmsd_self_superposed"] = fmt(
                    rmsd(pep_P_bb_self, np.array(pep_Q_bb)),
                    4,
                )

        # 4. complex CA RMSD after receptor fit
        complex_P_ca = list(fit_P) + list(pep_P_ca)
        complex_Q_ca = list(fit_Q) + list(pep_Q_ca)

        if len(complex_P_ca) >= 1:
            complex_P_ca_aligned = apply_transform(np.array(complex_P_ca), R, t)
            out["complex_ca_rmsd_after_receptor_fit"] = fmt(
                rmsd(complex_P_ca_aligned, np.array(complex_Q_ca)),
                4,
            )

        ca_val = safe_float(out.get("peptide_ca_rmsd_after_receptor_fit"))
        out["success_peptide_ca_rmsd_lt_2"] = 1 if ca_val is not None and ca_val < 2.0 else 0
        out["success_peptide_ca_rmsd_lt_5"] = 1 if ca_val is not None and ca_val < 5.0 else 0

        out["rmsd_status"] = "ok"
        final_rows.append(out)

    metric_path = out_dir / "complex_rmsd_metrics.csv"
    write_csv(metric_path, final_rows)

    summarize(
        final_rows,
        "temperature",
        out_dir / "complex_rmsd_summary_by_temperature.csv",
    )

    summarize(
        final_rows,
        "target_name",
        out_dir / "complex_rmsd_summary_by_target.csv",
    )

    ok_count = sum(1 for r in final_rows if r.get("rmsd_status") == "ok")
    skip_count = len(final_rows) - ok_count

    print("完成：复合物 RMSD 和结构指标计算")
    print("总行数:", len(final_rows))
    print("RMSD OK:", ok_count)
    print("跳过/失败:", skip_count)
    print("输出:")
    print(metric_path)
    print(out_dir / "complex_rmsd_summary_by_temperature.csv")
    print(out_dir / "complex_rmsd_summary_by_target.csv")


if __name__ == "__main__":
    main()
