#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compute cyclic-peptide CA RMSD with the same PyMOL align used globally.

This is a correction/extension of script 13.  Two distinct questions are kept
separate:

1. ``cyclic_peptide_self_aligned_pymol_ca_rmsd`` (primary here):
   independently apply the exact PyMOL ``align`` settings used for the global
   complex metric to the final predicted chain and the native peptide chain.
2. ``cyclic_peptide_ca_rmsd_after_global_complex_alignment`` (already produced
   by script 13):
   after the whole complex has been fitted, compare every peptide CA by
   residue position without a second peptide fit.  This is a stricter
   receptor-frame/ligand-position metric and is retained as a complement.

The project rule is enforced directly from each predicted PDB: the final chain
in file order is the cyclic peptide.  Prior audit labels cannot override it.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


ALIGNMENT_OBJECT = "batch_peptide_self_alignment"


def safe_float(value: object) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return None if math.isnan(number) else number
    except (TypeError, ValueError):
        return None


def fmt(value: object, digits: int = 6) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def path_key(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./").casefold()


def resolve_script_path() -> Path:
    value = globals().get("__script__") or globals().get("__file__")
    return Path(str(value)).resolve() if value else Path.cwd().resolve()


def load_support(script_path: Path):
    spec = importlib.util.spec_from_file_location("rmsd_script13_support", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import support script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_chain_fields(row: Mapping[str, object]) -> List[str]:
    ordered: List[str] = []
    scalar_fields = (
        "native_peptide_chain_used_by_complete_positional_rmsd",
        "native_peptide_chain_used_by_global_alignment",
        "native_peptide_chain_used",
        "native_peptide_chain",
    )
    for field in scalar_fields:
        value = str(row.get(field, "")).strip()
        if value and value not in ordered:
            ordered.append(value)
    for field in ("native_peptide_chains_considered", "equivalent_native_peptide_chains"):
        raw = str(row.get(field, "")).strip()
        for value in raw.replace(",", ";").replace("|", ";").split(";"):
            value = value.strip()
            if value and value not in ordered:
                ordered.append(value)
    return ordered


def native_chain_candidates(
    row: Mapping[str, object],
    native_record: Mapping[str, object],
    support,
    design_length: int,
) -> List[str]:
    sequences = support.native_sequences(native_record)
    candidates = [
        chain for chain in parse_chain_fields(row)
        if chain in sequences
    ]
    fallback = support.choose_native_peptide_chains(
        native_record,
        candidates,
        str(row.get("native_seq", "")),
        design_length,
    )
    for chain in fallback:
        if chain not in candidates:
            candidates.append(chain)
    if not candidates:
        candidates = [
            chain for chain, sequence in sequences.items()
            if not design_length or len(sequence) == design_length
        ]
    return candidates


def cleanup(cmd) -> None:
    for name in (
        "batch_pred",
        "batch_native",
        ALIGNMENT_OBJECT,
    ):
        try:
            cmd.delete(name)
        except Exception:
            pass


def evaluate_row(
    source: Mapping[str, object],
    native_records: Mapping[str, dict],
    support,
    cmd,
    repo_root: Path,
    threshold: float,
) -> dict:
    row = dict(source)
    row["cyclic_peptide_same_align_status"] = ""
    pdb_path = support.resolve_repo_path(repo_root, row.get("pdb_path", ""))
    target = str(row.get("target_name", "")).upper()

    try:
        if not pdb_path.is_file():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")
        if target not in native_records:
            raise KeyError(f"Native record not found for target: {target}")

        pred_meta = support.parse_predicted_ca_metadata(pdb_path)
        chain_sequences = pred_meta["chain_sequences"]
        if not chain_sequences:
            raise ValueError("No CA-containing chains found in predicted PDB")

        # Non-negotiable project rule: use the final chain observed in the PDB.
        predicted_chain = list(chain_sequences)[-1]
        design_seq = str(row.get("design_seq", ""))
        design_length_value = row.get("design_length") or len(design_seq)
        try:
            design_length = int(float(design_length_value))
        except (TypeError, ValueError):
            design_length = len(design_seq)

        candidates = native_chain_candidates(
            row,
            native_records[target],
            support,
            design_length,
        )
        if not candidates:
            raise ValueError("No native peptide-chain candidate could be identified")

        cleanup(cmd)
        cmd.read_pdbstr(
            support.native_record_to_pdbstr(native_records[target]),
            "batch_native",
        )
        cmd.load(str(pdb_path), "batch_pred")
        cmd.sort("batch_native")
        cmd.sort("batch_pred")

        predicted_selection = (
            f"batch_pred and chain {predicted_chain} and name CA"
        )
        predicted_count = int(cmd.count_atoms(predicted_selection))
        if predicted_count <= 0:
            raise ValueError(
                f"Final predicted chain {predicted_chain!r} contains no CA atoms"
            )

        results = []
        errors = []
        for native_chain in candidates:
            native_selection = (
                f"batch_native and chain {native_chain} and name CA"
            )
            native_count = int(cmd.count_atoms(native_selection))
            if native_count <= 0:
                errors.append(f"{native_chain}: no native CA")
                continue
            try:
                cmd.delete(ALIGNMENT_OBJECT)
                result = cmd.align(
                    predicted_selection,
                    native_selection,
                    cutoff=2.0,
                    cycles=0,
                    gap=-10.0,
                    extend=-0.5,
                    max_gap=50,
                    object=ALIGNMENT_OBJECT,
                    matrix="BLOSUM62",
                    mobile_state=0,
                    target_state=0,
                    quiet=1,
                    max_skip=0,
                    transform=0,
                    reset=0,
                )
                if len(result) != 7:
                    raise RuntimeError(f"Unexpected PyMOL align result: {result!r}")
                (
                    rms_after,
                    n_after,
                    n_cycles,
                    rms_before,
                    n_before,
                    raw_score,
                    n_residues,
                ) = result
                if int(n_after) <= 0:
                    raise ValueError("PyMOL align returned zero paired CA atoms")
                results.append(
                    {
                        "rmsd": float(rms_after),
                        "native_chain": native_chain,
                        "predicted_count": predicted_count,
                        "native_count": native_count,
                        "n_after": int(n_after),
                        "n_cycles": int(n_cycles),
                        "rms_before": float(rms_before),
                        "n_before": int(n_before),
                        "raw_score": float(raw_score),
                        "n_residues": int(n_residues),
                    }
                )
            except Exception as exc:
                errors.append(f"{native_chain}: {exc!r}")

        if not results:
            raise ValueError(
                "No native peptide chain produced a valid PyMOL alignment; "
                + " | ".join(errors)
            )

        # Equivalent native chains are symmetry alternatives. Use the best
        # valid chain, exactly as script 13 did for positional peptide RMSD.
        best = min(
            results,
            key=lambda item: (
                item["rmsd"],
                -item["n_after"],
                item["native_chain"],
            ),
        )
        peptide_pass = int(best["rmsd"] < threshold)
        global_pass = int(
            str(
                row.get(
                    "passes_global_complex_ca_rmsd_lt_threshold",
                    "",
                )
            )
            == "1"
        )

        decoded = chain_sequences.get(predicted_chain, "")
        row.update(
            {
                "cyclic_peptide_chain_rule": "final_chain_in_predicted_pdb_file_order",
                "cyclic_peptide_predicted_chain": predicted_chain,
                "cyclic_peptide_all_predicted_chains_in_file_order": ";".join(
                    chain_sequences
                ),
                "cyclic_peptide_native_chains_considered_for_same_align": ";".join(
                    candidates
                ),
                "cyclic_peptide_native_chain_used_for_same_align": best[
                    "native_chain"
                ],
                "cyclic_peptide_decoded_sequence_from_pdb": decoded,
                "cyclic_peptide_naturalized_sequence_matches_design": int(
                    support.naturalize(decoded)
                    == support.naturalize(design_seq)
                ),
                "cyclic_peptide_self_align_selection_mobile": predicted_selection,
                "cyclic_peptide_self_align_selection_target": (
                    f"batch_native and chain {best['native_chain']} and name CA"
                ),
                "cyclic_peptide_self_align_cycles": 0,
                "cyclic_peptide_self_align_cutoff": 2.0,
                "cyclic_peptide_self_align_gap": -10.0,
                "cyclic_peptide_self_align_extend": -0.5,
                "cyclic_peptide_self_align_max_gap": 50,
                "cyclic_peptide_self_align_matrix": "BLOSUM62",
                "cyclic_peptide_self_align_max_skip": 0,
                "cyclic_peptide_self_aligned_pymol_ca_rmsd": fmt(best["rmsd"]),
                "cyclic_peptide_self_aligned_pymol_ca_rmsd_before_rejection": fmt(
                    best["rms_before"]
                ),
                "n_cyclic_peptide_self_aligned_ca_pairs": best["n_after"],
                "n_cyclic_peptide_self_aligned_ca_pairs_before_rejection": best[
                    "n_before"
                ],
                "n_predicted_cyclic_peptide_ca": best["predicted_count"],
                "n_native_cyclic_peptide_ca": best["native_count"],
                "cyclic_peptide_self_align_coverage_vs_predicted": fmt(
                    best["n_after"] / best["predicted_count"]
                ),
                "cyclic_peptide_self_align_coverage_vs_native": fmt(
                    best["n_after"] / best["native_count"]
                ),
                "cyclic_peptide_self_align_full_ca_coverage": int(
                    best["n_after"] == best["predicted_count"]
                    and best["n_after"] == best["native_count"]
                ),
                "cyclic_peptide_self_align_raw_score": fmt(best["raw_score"]),
                "cyclic_peptide_self_align_residues": best["n_residues"],
                "cyclic_peptide_same_align_threshold_angstrom": threshold,
                "passes_cyclic_peptide_self_aligned_pymol_ca_rmsd_lt_threshold": (
                    peptide_pass
                ),
                "passes_joint_global_and_cyclic_peptide_same_align_lt_threshold": int(
                    global_pass and peptide_pass
                ),
                "cyclic_peptide_same_align_status": "ok",
                "cyclic_peptide_same_align_error": "",
            }
        )
    except Exception as exc:
        row["cyclic_peptide_same_align_status"] = "failed"
        row["cyclic_peptide_same_align_error"] = repr(exc)
    finally:
        cleanup(cmd)
    return row


def evaluate_rows(
    rows: Sequence[dict],
    label: str,
    native_records: Mapping[str, dict],
    support,
    cmd,
    repo_root: Path,
    threshold: float,
) -> List[dict]:
    output = []
    for index, row in enumerate(rows, start=1):
        output.append(
            evaluate_row(
                row,
                native_records,
                support,
                cmd,
                repo_root,
                threshold,
            )
        )
        if index % 100 == 0 or index == len(rows):
            print(f"[{label}] processed: {index}/{len(rows)}", flush=True)
    return output


def index_rows(rows: Sequence[dict]) -> dict:
    by_path = {}
    by_file = {}
    for row in rows:
        if row.get("pdb_path"):
            by_path[path_key(row["pdb_path"])] = row
        key = (
            str(row.get("target_name", "")).upper(),
            str(row.get("temperature", "")),
            str(row.get("design_seq", "")),
            str(row.get("pdb_file", "")),
        )
        by_file[key] = row
    return {"by_path": by_path, "by_file": by_file}


def enrich_representatives(
    representatives: Sequence[dict],
    evaluated_all: Sequence[dict],
) -> List[dict]:
    index = index_rows(evaluated_all)
    output = []
    for source in representatives:
        match = None
        if source.get("pdb_path"):
            match = index["by_path"].get(path_key(source["pdb_path"]))
        if match is None:
            key = (
                str(source.get("target_name", "")).upper(),
                str(source.get("temperature", "")),
                str(source.get("design_seq", "")),
                str(source.get("pdb_file", "")),
            )
            match = index["by_file"].get(key)
        row = dict(source)
        if match is not None:
            row.update(match)
        elif not row.get("cyclic_peptide_same_align_status"):
            row["cyclic_peptide_same_align_status"] = "missing_pdb_representative"
        output.append(row)
    return output


def design_key(row: Mapping[str, object]) -> Tuple[str, str, str]:
    return (
        str(row.get("target_name", "")).upper(),
        str(row.get("temperature", "")),
        str(row.get("design_seq", "")),
    )


def exploratory_candidates(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        if (
            row.get("cyclic_peptide_same_align_status") == "ok"
            and str(
                row.get(
                    "passes_joint_global_and_cyclic_peptide_same_align_lt_threshold",
                    "",
                )
            )
            == "1"
        ):
            groups[design_key(row)].append(row)

    output = []
    for _key, candidates in sorted(groups.items()):
        def rank(row: Mapping[str, object]) -> tuple:
            global_rmsd = safe_float(row.get("global_complex_ca_rmsd"))
            peptide_rmsd = safe_float(
                row.get("cyclic_peptide_self_aligned_pymol_ca_rmsd")
            )
            confidence = safe_float(row.get("pdb_ca_bfactor_mean"))
            g = global_rmsd if global_rmsd is not None else math.inf
            p = peptide_rmsd if peptide_rmsd is not None else math.inf
            c = confidence if confidence is not None else -math.inf
            return (max(g, p), g + p, -c, str(row.get("pdb_file", "")))

        chosen = dict(sorted(candidates, key=rank)[0])
        chosen["unique_design_representative_rule"] = (
            "lowest_max_global_and_self_aligned_peptide_rmsd_among_joint_pass_"
            "pdbs_exploratory"
        )
        chosen["n_joint_pass_pdb_for_design"] = len(candidates)
        output.append(chosen)
    return output


def metric_summary(rows: Sequence[dict], threshold: float) -> dict:
    ok = [
        row for row in rows
        if row.get("cyclic_peptide_same_align_status") == "ok"
    ]
    values = [
        safe_float(row.get("cyclic_peptide_self_aligned_pymol_ca_rmsd"))
        for row in ok
    ]
    values = [value for value in values if value is not None]
    peptide_pass = [
        row for row in ok
        if str(
            row.get(
                "passes_cyclic_peptide_self_aligned_pymol_ca_rmsd_lt_threshold",
                "",
            )
        )
        == "1"
    ]
    joint_pass = [
        row for row in ok
        if str(
            row.get(
                "passes_joint_global_and_cyclic_peptide_same_align_lt_threshold",
                "",
            )
        )
        == "1"
    ]
    return {
        "rows": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "peptide_pass": len(peptide_pass),
        "joint_pass": len(joint_pass),
        "median": median(values) if values else None,
        "mean": mean(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "full_coverage": sum(
            str(row.get("cyclic_peptide_self_align_full_ca_coverage", "")) == "1"
            for row in ok
        ),
        "naturalized_match": sum(
            str(
                row.get(
                    "cyclic_peptide_naturalized_sequence_matches_design",
                    "",
                )
            )
            == "1"
            for row in ok
        ),
        "threshold": threshold,
    }


def summary_lines(label: str, summary: Mapping[str, object]) -> List[str]:
    threshold = float(summary["threshold"])
    return [
        f"{label}:",
        f"  rows: {summary['rows']}",
        f"  RMSD OK: {summary['ok']}",
        f"  failed/missing: {summary['failed']}",
        (
            f"  cyclic-peptide self-aligned PyMOL CA RMSD < {threshold:.3f}: "
            f"{summary['peptide_pass']}/{summary['rows']}"
        ),
        (
            f"  joint global + cyclic-peptide pass: "
            f"{summary['joint_pass']}/{summary['rows']}"
        ),
        f"  mean peptide RMSD: {fmt(summary['mean'])}",
        f"  median peptide RMSD: {fmt(summary['median'])}",
        f"  minimum peptide RMSD: {fmt(summary['minimum'])}",
        f"  maximum peptide RMSD: {fmt(summary['maximum'])}",
        f"  full peptide CA sequence-alignment coverage: "
        f"{summary['full_coverage']}/{summary['ok']}",
        f"  naturalized design sequence matches: "
        f"{summary['naturalized_match']}/{summary['ok']}",
    ]


def write_4kel_audit(path: Path, rows: Sequence[dict]) -> None:
    matches = [
        row for row in rows
        if str(row.get("pdb_file", "")).casefold()
        == "4kel_13_rcrrrgnrqgqcgr_model.pdb"
        and str(row.get("temperature", "")) == "0.3"
    ]
    lines = ["===== 4KEL CYCLIC-PEPTIDE SAME-ALIGN AUDIT =====", ""]
    if len(matches) != 1:
        lines.extend(
            [
                "status: FAIL",
                f"expected exactly one 4KEL row, found {len(matches)}",
            ]
        )
    else:
        row = matches[0]
        selected_ok = (
            row.get("cyclic_peptide_same_align_status") == "ok"
            and row.get("cyclic_peptide_predicted_chain") == "B"
            and str(row.get("n_predicted_cyclic_peptide_ca", "")) == "14"
            and str(row.get("n_native_cyclic_peptide_ca", "")) == "14"
        )
        lines.extend(
            [
                f"status: {'PASS' if selected_ok else 'FAIL'}",
                "expected final predicted chain: B",
                f"observed final predicted chain: "
                f"{row.get('cyclic_peptide_predicted_chain', '')}",
                f"all predicted chains in file order: "
                f"{row.get('cyclic_peptide_all_predicted_chains_in_file_order', '')}",
                f"native peptide chain used: "
                f"{row.get('cyclic_peptide_native_chain_used_for_same_align', '')}",
                f"predicted peptide CA count: "
                f"{row.get('n_predicted_cyclic_peptide_ca', '')}",
                f"native peptide CA count: "
                f"{row.get('n_native_cyclic_peptide_ca', '')}",
                f"PyMOL-aligned peptide CA pairs: "
                f"{row.get('n_cyclic_peptide_self_aligned_ca_pairs', '')}",
                f"cyclic-peptide self-aligned PyMOL CA RMSD: "
                f"{row.get('cyclic_peptide_self_aligned_pymol_ca_rmsd', '')}",
                f"strict peptide RMSD after global-complex alignment: "
                f"{row.get('cyclic_peptide_ca_rmsd_after_global_complex_alignment', '')}",
                f"global-complex CA RMSD: "
                f"{row.get('global_complex_ca_rmsd', '')}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    prior_dir = (
        repo_root
        / "paper_clean_v28_outputs/structure_metrics/"
        "global_and_cyclic_peptide_ca_rmsd"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Apply the exact global PyMOL align algorithm independently to "
            "the final cyclic-peptide chain."
        )
    )
    parser.add_argument(
        "--support_script",
        default=str(
            repo_root
            / "paper_clean_v28/structure_metrics/"
            "13_compute_global_and_cyclic_peptide_ca_rmsd.py"
        ),
    )
    parser.add_argument(
        "--best85_csv",
        default=str(prior_dir / "global_complex_ca_rmsd_best85.csv"),
    )
    parser.add_argument(
        "--all_pdb_csv",
        default=str(prior_dir / "global_complex_ca_rmsd_all_pdbs.csv"),
    )
    parser.add_argument(
        "--confidence_representatives_csv",
        default=str(
            prior_dir
            / "global_complex_ca_rmsd_all_designs_confidence_representative.csv"
        ),
    )
    parser.add_argument(
        "--native_jsonl",
        default=str(repo_root / "17_complexes_native.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(
            repo_root
            / "paper_clean_v28_outputs/structure_metrics/"
            "cyclic_peptide_same_pymol_align"
        ),
    )
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--expected_best85", type=int, default=85)
    parser.add_argument("--expected_all_pdbs", type=int, default=4108)
    parser.add_argument("--expected_unique_designs", type=int, default=4015)
    return parser


def main() -> None:
    script_path = resolve_script_path()
    repo_root = script_path.parents[2]
    parser = build_parser(repo_root)
    args, _unknown = parser.parse_known_args(sys.argv[1:])

    support_script = Path(args.support_script)
    best85_path = Path(args.best85_csv)
    all_pdb_path = Path(args.all_pdb_csv)
    confidence_path = Path(args.confidence_representatives_csv)
    native_path = Path(args.native_jsonl)
    out_dir = Path(args.out_dir)

    for required in (
        support_script,
        best85_path,
        all_pdb_path,
        confidence_path,
        native_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive")

    support = load_support(support_script)
    cmd = support.cmd
    best_source = read_csv(best85_path)
    all_source = read_csv(all_pdb_path)
    confidence_source = read_csv(confidence_path)
    native_records = support.load_native_records(native_path)

    if len(best_source) != args.expected_best85:
        raise RuntimeError(
            f"best85 count gate failed: {len(best_source)} != {args.expected_best85}"
        )
    if len(all_source) != args.expected_all_pdbs:
        raise RuntimeError(
            f"all-PDB count gate failed: {len(all_source)} != "
            f"{args.expected_all_pdbs}"
        )
    if len(confidence_source) != args.expected_unique_designs:
        raise RuntimeError(
            f"unique-design count gate failed: {len(confidence_source)} != "
            f"{args.expected_unique_designs}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("===== CYCLIC PEPTIDE: SAME PYMOL ALIGN AS GLOBAL =====", flush=True)
    print("repository root:", repo_root, flush=True)
    print("final predicted chain is enforced from each PDB", flush=True)
    print("best85 rows:", len(best_source), flush=True)
    print("all PDB rows:", len(all_source), flush=True)

    best_rows = evaluate_rows(
        best_source,
        "best85",
        native_records,
        support,
        cmd,
        repo_root,
        args.threshold,
    )
    all_rows = evaluate_rows(
        all_source,
        "all PDBs",
        native_records,
        support,
        cmd,
        repo_root,
        args.threshold,
    )
    confidence_rows = enrich_representatives(confidence_source, all_rows)
    downstream = exploratory_candidates(all_rows)

    best_peptide_pass = [
        row for row in best_rows
        if str(
            row.get(
                "passes_cyclic_peptide_self_aligned_pymol_ca_rmsd_lt_threshold",
                "",
            )
        )
        == "1"
    ]
    best_joint_pass = [
        row for row in best_rows
        if str(
            row.get(
                "passes_joint_global_and_cyclic_peptide_same_align_lt_threshold",
                "",
            )
        )
        == "1"
    ]
    all_peptide_pass = [
        row for row in all_rows
        if str(
            row.get(
                "passes_cyclic_peptide_self_aligned_pymol_ca_rmsd_lt_threshold",
                "",
            )
        )
        == "1"
    ]
    all_joint_pass = [
        row for row in all_rows
        if str(
            row.get(
                "passes_joint_global_and_cyclic_peptide_same_align_lt_threshold",
                "",
            )
        )
        == "1"
    ]
    confidence_peptide_pass = [
        row for row in confidence_rows
        if str(
            row.get(
                "passes_cyclic_peptide_self_aligned_pymol_ca_rmsd_lt_threshold",
                "",
            )
        )
        == "1"
    ]
    confidence_joint_pass = [
        row for row in confidence_rows
        if str(
            row.get(
                "passes_joint_global_and_cyclic_peptide_same_align_lt_threshold",
                "",
            )
        )
        == "1"
    ]

    write_csv(out_dir / "cyclic_peptide_same_pymol_align_best85.csv", best_rows)
    write_csv(
        out_dir / "cyclic_peptide_same_pymol_align_best85_lt3.csv",
        best_peptide_pass,
    )
    write_csv(
        out_dir / "joint_global_and_cyclic_same_align_best85_lt3.csv",
        best_joint_pass,
    )
    write_csv(out_dir / "cyclic_peptide_same_pymol_align_all_pdbs.csv", all_rows)
    write_csv(
        out_dir / "cyclic_peptide_same_pymol_align_all_pdbs_lt3.csv",
        all_peptide_pass,
    )
    write_csv(
        out_dir / "joint_global_and_cyclic_same_align_all_pdbs_lt3.csv",
        all_joint_pass,
    )
    write_csv(
        out_dir
        / "cyclic_peptide_same_pymol_align_all_designs_confidence_representative.csv",
        confidence_rows,
    )
    write_csv(
        out_dir
        / "cyclic_peptide_same_pymol_align_all_designs_confidence_representative_lt3.csv",
        confidence_peptide_pass,
    )
    write_csv(
        out_dir
        / "joint_global_and_cyclic_same_align_all_designs_confidence_representative_lt3.csv",
        confidence_joint_pass,
    )
    write_csv(
        out_dir
        / "joint_global_and_cyclic_same_align_candidates_for_downstream_exploratory.csv",
        downstream,
    )
    write_csv(
        out_dir / "cyclic_peptide_same_pymol_align_problem_rows.csv",
        [
            row for row in all_rows
            if row.get("cyclic_peptide_same_align_status") != "ok"
        ],
    )

    best_summary = metric_summary(best_rows, args.threshold)
    all_summary = metric_summary(all_rows, args.threshold)
    confidence_summary = metric_summary(confidence_rows, args.threshold)
    report = [
        "===== CYCLIC-PEPTIDE CA RMSD: SAME PYMOL ALIGN AS GLOBAL =====",
        "",
        "Presence and pass/fail are different:",
        "  every successfully processed row has a cyclic peptide (the final PDB chain);",
        "  pass means only that its independently aligned CA RMSD is below threshold.",
        "",
        "Primary cyclic-peptide metric:",
        "  mobile = final predicted chain and name CA",
        "  target = native peptide chain and name CA",
        "  PyMOL align settings identical to the global metric",
        "  cycles=0; no outlier rejection",
        "  the peptide receives its own fit",
        f"  strict threshold: < {args.threshold:.3f} Angstrom",
        "",
        "Complement retained from script 13:",
        "  complete positional peptide CA RMSD after whole-complex alignment",
        "  no second peptide fit; this is a stricter ligand-position metric",
        "",
    ]
    report.extend(summary_lines("best85", best_summary))
    report.append("")
    report.extend(summary_lines("all PDBs", all_summary))
    report.append("")
    report.extend(summary_lines("confidence representatives", confidence_summary))
    report.extend(
        [
            "",
            f"unique exploratory downstream candidates with both metrics < "
            f"{args.threshold:.3f}: {len(downstream)}",
            "",
            "Interpretation boundary:",
            "  PyMOL align is sequence-matched and may use fewer than all peptide CA atoms.",
            "  Coverage columns must be reported with RMSD.",
            "  The retained complete positional receptor-frame peptide RMSD answers a",
            "  different, stricter question about binding position and must not be renamed",
            "  as the independently self-aligned peptide RMSD.",
            "",
            f"elapsed seconds: {time.time() - started:.3f}",
        ]
    )
    (out_dir / "cyclic_peptide_same_pymol_align_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    write_4kel_audit(out_dir / "pymol_4kel_cyclic_peptide_audit.txt", all_rows)

    print("\n===== COMPLETE =====", flush=True)
    print(
        f"best85 peptide pass: {best_summary['peptide_pass']}/{best_summary['rows']}; "
        f"joint: {best_summary['joint_pass']}/{best_summary['rows']}",
        flush=True,
    )
    print(
        f"all PDB peptide pass: {all_summary['peptide_pass']}/{all_summary['rows']}; "
        f"joint: {all_summary['joint_pass']}/{all_summary['rows']}",
        flush=True,
    )
    print(
        "confidence representatives peptide pass: "
        f"{confidence_summary['peptide_pass']}/{confidence_summary['rows']}; "
        f"joint: {confidence_summary['joint_pass']}/{confidence_summary['rows']}",
        flush=True,
    )
    print(
        "unique exploratory downstream candidates:",
        len(downstream),
        flush=True,
    )
    print("output directory:", out_dir, flush=True)


if __name__ in {"__main__", "pymol"}:
    main()
