#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select a small structure-first handoff for the frozen failed T=0.5 targets.

Permeability is deliberately absent: in the executed collaboration workflow it
is evaluated only after structures return.  Candidates are ranked by the
unchanged V28 base likelihood, corrected methyl-site confidence, repeated-seed
stability, and then native recovery.  Exact naturalized duplicates are
collapsed before a greedy diversity gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_retrain" / "generation"
DEFAULT_OUT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_retrain" / "handoff"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_structure_failures.json")
DEFAULT_PRIOR_HANDOFF = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "rerun_temperature_0.5_multiseed"
    / "methylated_new_candidates.csv"
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def optional_float(value: object, default: float = -math.inf) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def sequence_identity(left: object, right: object) -> float:
    left_text = str(left).upper()
    right_text = str(right).upper()
    if not left_text or len(left_text) != len(right_text):
        return 0.0
    return sum(a == b for a, b in zip(left_text, right_text)) / len(left_text)


def parse_json_list(value: object, field: str) -> List[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a JSON list")
    return parsed


def add_methyl_site_statistics(source: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(source)
    sequence = str(row.get("design_seq", ""))
    positions = [int(value) for value in parse_json_list(row.get("methyl_positions_1based", "[]"), "methyl_positions_1based")]
    probabilities = [float(value) for value in parse_json_list(row.get("methyl_probabilities", "[]"), "methyl_probabilities")]
    expected_positions = [index for index, token in enumerate(sequence, start=1) if token.islower()]
    if positions != expected_positions:
        raise RuntimeError(
            f"Methyl-position mismatch for {row.get('candidate_id')}: "
            f"column={positions}, sequence={expected_positions}"
        )
    if len(probabilities) != len(sequence):
        raise RuntimeError(
            f"Probability-length mismatch for {row.get('candidate_id')}: "
            f"{len(probabilities)} != {len(sequence)}"
        )
    if not positions:
        raise RuntimeError(f"Non-methylated row entered structure selector: {row.get('candidate_id')}")
    site_values = [probabilities[position - 1] for position in positions]
    row["methyl_site_probability_min"] = min(site_values)
    row["methyl_site_probability_mean"] = sum(site_values) / len(site_values)
    row["methyl_site_probability_max"] = max(site_values)
    row["seeds_observed_count"] = len(
        {value for value in str(row.get("seeds_observed", "")).split(";") if value}
    )
    return row


def candidate_sort_key(row: Mapping[str, Any]) -> Tuple[float, float, int, int, float, str]:
    return (
        -optional_float(row.get("base_log_probability_mean")),
        -optional_float(row.get("methyl_site_probability_min")),
        -int(row.get("seeds_observed_count", 0)),
        -int(row.get("occurrence_count", 0)),
        -optional_float(row.get("natural_aa_recovery")),
        str(row.get("design_seq", "")),
    )


def collapse_naturalized_variants(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["design_natural_seq"]).upper()].append(row)

    collapsed: List[Dict[str, Any]] = []
    for natural_sequence, variants in sorted(grouped.items()):
        ranked = sorted(variants, key=candidate_sort_key)
        output = dict(ranked[0])
        output["naturalized_variant_count"] = len(variants)
        output["alternate_methylated_sequences"] = ";".join(
            sorted({str(row["design_seq"]) for row in variants[1:]})
        )
        collapsed.append(output)
    return collapsed


def select_diverse(
    rows: Sequence[Mapping[str, Any]], quota: int, identity_ceiling: float
) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=candidate_sort_key)
    selected: List[Dict[str, Any]] = []
    skipped: List[Mapping[str, Any]] = []
    for source in ranked:
        identities = [
            sequence_identity(source["design_natural_seq"], earlier["design_natural_seq"])
            for earlier in selected
        ]
        maximum = max(identities) if identities else 0.0
        if not selected or maximum < identity_ceiling:
            row = dict(source)
            row["diversity_gate"] = "STRICT_PASS"
            row["max_identity_to_earlier_selected"] = maximum
            selected.append(row)
            if len(selected) == quota:
                break
        else:
            skipped.append(source)

    if len(selected) < quota:
        chosen = {str(row["design_seq"]) for row in selected}
        for source in skipped:
            if str(source["design_seq"]) in chosen:
                continue
            identities = [
                sequence_identity(source["design_natural_seq"], earlier["design_natural_seq"])
                for earlier in selected
            ]
            row = dict(source)
            row["diversity_gate"] = "RELAXED_FILL"
            row["max_identity_to_earlier_selected"] = max(identities) if identities else 0.0
            selected.append(row)
            chosen.add(str(row["design_seq"]))
            if len(selected) == quota:
                break

    for rank, row in enumerate(selected, start=1):
        row["structure_selection_rank"] = rank
    return selected


def prior_handoff_index(
    path: Path | None,
) -> Tuple[
    Dict[Tuple[str, str], List[Dict[str, str]]],
    set[Tuple[str, str]],
    Dict[str, Any],
]:
    if path is None or not path.is_file():
        return {}, set(), {"available": False, "path": str(path) if path else None}
    rows = read_csv(path)
    grouped: MutableMapping[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    exact_keys: set[Tuple[str, str]] = set()
    for row in rows:
        target = str(row.get("target_name", "")).upper()
        design_sequence = str(row.get("design_seq", ""))
        key = (
            target,
            str(row.get("design_natural_seq", "")).upper(),
        )
        grouped[key].append(row)
        if target and design_sequence:
            exact_keys.add((target, design_sequence))
    return dict(grouped), exact_keys, {
        "available": True,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "unique_exact_target_sequence_keys": len(exact_keys),
        "unique_target_natural_sequence_keys": len(grouped),
    }


def attach_reuse_evidence(
    row: Mapping[str, Any],
    prior_index: Mapping[Tuple[str, str], Sequence[Mapping[str, str]]],
) -> Dict[str, Any]:
    output = dict(row)
    key = (
        str(output["target_name"]).upper(),
        str(output["design_natural_seq"]).upper(),
    )
    matches = list(prior_index.get(key, []))
    output["prior_handoff_natural_sequence_match"] = int(bool(matches))
    output["prior_handoff_candidate_ids"] = ";".join(
        sorted({str(match.get("candidate_id", "")) for match in matches if match.get("candidate_id")})
    )
    output["structure_reuse_status"] = (
        "VERIFY_PDB_EXISTS_AND_PASSED_GATE_THEN_REUSE"
        if matches
        else "NEW_STRUCTURE_REQUIRED"
    )
    return output


def make_task(row: Mapping[str, Any], metadata: Mapping[str, str]) -> Dict[str, Any]:
    target = str(row["target_name"]).upper()
    rank = int(row["structure_selection_rank"])
    return {
        "suggested_job_name": f"{target}_T0_5_SERQC_{rank:02d}",
        "target_name": target,
        "temperature": 0.5,
        "selected_chain": metadata["selected_chain"],
        "receptor_chains": metadata["structure_receptor_chains"],
        "receptor_sequences_json": metadata["receptor_sequences_json"],
        "native_peptide_seq": metadata["native_peptide_seq"],
        "design_peptide_seq": row["design_seq"],
        "design_peptide_natural_seq": row["design_natural_seq"],
        "methyl_positions_1based": row["methyl_positions_1based"],
        "design_methyl_count": row["design_methyl_count"],
        "methyl_site_probability_min": row["methyl_site_probability_min"],
        "methyl_site_probability_mean": row["methyl_site_probability_mean"],
        "base_log_probability_mean": row["base_log_probability_mean"],
        "occurrence_count": row["occurrence_count"],
        "seeds_observed": row["seeds_observed"],
        "natural_aa_recovery": row["natural_aa_recovery"],
        "naturalized_variant_count": row["naturalized_variant_count"],
        "alternate_methylated_sequences": row["alternate_methylated_sequences"],
        "diversity_gate": row["diversity_gate"],
        "max_identity_to_earlier_selected": row["max_identity_to_earlier_selected"],
        "structure_selection_rank": rank,
        "source_candidate_id": row["candidate_id"],
        "source_seed": row["seed"],
        "prior_handoff_natural_sequence_match": row[
            "prior_handoff_natural_sequence_match"
        ],
        "prior_handoff_candidate_ids": row["prior_handoff_candidate_ids"],
        "structure_reuse_status": row["structure_reuse_status"],
        "permeability_status": "PENDING_UNTIL_STRUCTURE_RETURNS",
        "final_required_gate": (
            "global complex CA RMSD <3 A AND complete final-chain cyclic-peptide "
            "CA RMSD <3 A after one global alignment and best forward cyclic shift; "
            "no peptide-only refit"
        ),
    }


def write_structure_inputs(out_dir: Path, tasks: Sequence[Mapping[str, Any]]) -> None:
    input_dir = out_dir / "structure_inputs_for_shangge"
    input_dir.mkdir(parents=True, exist_ok=True)
    combined: List[str] = []
    jsonl: List[str] = []
    for task in tasks:
        receptors = json.loads(str(task["receptor_sequences_json"]))
        lines: List[str] = []
        for chain_id, sequence in receptors.items():
            lines.extend(
                [
                    f">{task['suggested_job_name']}|chain={chain_id}|role=receptor",
                    str(sequence),
                ]
            )
        lines.extend(
            [
                f">{task['suggested_job_name']}|chain={task['selected_chain']}|role=design_peptide"
                f"|methyl_positions_1based={task['methyl_positions_1based']}",
                str(task["design_peptide_natural_seq"]),
            ]
        )
        content = "\n".join(lines) + "\n"
        path = input_dir / f"{task['suggested_job_name']}.fasta"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
        combined.extend(lines)
        jsonl.append(
            json.dumps(
                {
                    "job_name": task["suggested_job_name"],
                    "target_name": task["target_name"],
                    "temperature": 0.5,
                    "receptor_sequences": receptors,
                    "design_peptide_chain": task["selected_chain"],
                    "design_peptide_natural_sequence": task[
                        "design_peptide_natural_seq"
                    ],
                    "design_peptide_methylated_sequence": task[
                        "design_peptide_seq"
                    ],
                    "methyl_positions_1based": json.loads(
                        str(task["methyl_positions_1based"])
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    combined_path = out_dir / "structure_tasks_for_shangge.fasta"
    combined_temp = combined_path.with_suffix(".fasta.tmp")
    combined_temp.write_text(
        "\n".join(combined) + ("\n" if combined else ""), encoding="utf-8"
    )
    os.replace(combined_temp, combined_path)
    jsonl_path = out_dir / "structure_tasks_for_shangge.jsonl"
    jsonl_temp = jsonl_path.with_suffix(".jsonl.tmp")
    jsonl_temp.write_text(
        "\n".join(jsonl) + ("\n" if jsonl else ""), encoding="utf-8"
    )
    os.replace(jsonl_temp, jsonl_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--candidates-csv")
    parser.add_argument("--target-manifest-csv")
    parser.add_argument("--prior-handoff-csv", default=str(DEFAULT_PRIOR_HANDOFF))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--allow-shortfall", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    plan_path = Path(args.plan).resolve()
    candidate_path = Path(args.candidates_csv or run_dir / "methylated_new_candidates.csv").resolve()
    target_manifest_path = Path(
        args.target_manifest_csv or run_dir / "target_manifest.csv"
    ).resolve()
    prior_path = Path(args.prior_handoff_csv).resolve() if args.prior_handoff_csv else None
    out_dir = Path(args.out_dir).resolve()
    for required in (plan_path, candidate_path, target_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    plan = read_json(plan_path)
    target_plan = {
        str(item["target_name"]).upper(): dict(item) for item in plan["targets"]
    }
    expected_count = int(plan.get("expected_target_count", len(target_plan)))
    if len(target_plan) != expected_count:
        raise RuntimeError(
            f"Plan target count changed: expected {expected_count}, observed {len(target_plan)}"
        )
    identity_ceiling = float(plan["sequence_identity_ceiling"])
    candidates = [add_methyl_site_statistics(row) for row in read_csv(candidate_path)]
    unexpected_targets = sorted(
        {str(row["target_name"]).upper() for row in candidates} - set(target_plan)
    )
    if unexpected_targets:
        raise RuntimeError("Candidates contain non-plan targets: " + ", ".join(unexpected_targets))
    metadata = {
        str(row["target_name"]).upper(): row for row in read_csv(target_manifest_path)
    }
    missing_metadata = sorted(set(target_plan) - set(metadata))
    if missing_metadata:
        raise RuntimeError("Missing target metadata: " + ", ".join(missing_metadata))
    prior_index, prior_exact_keys, prior_manifest = prior_handoff_index(prior_path)

    ranked_pool: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for target, target_config in target_plan.items():
        all_target_rows = [
            row for row in candidates if str(row["target_name"]).upper() == target
        ]
        target_rows = [
            row
            for row in all_target_rows
            if (target, str(row["design_seq"])) not in prior_exact_keys
        ]
        prior_exact_excluded = len(all_target_rows) - len(target_rows)
        collapsed = collapse_naturalized_variants(target_rows)
        collapsed = [attach_reuse_evidence(row, prior_index) for row in collapsed]
        for rank, row in enumerate(sorted(collapsed, key=candidate_sort_key), start=1):
            row["pre_diversity_rank"] = rank
            ranked_pool.append(row)
        quota = int(target_config["structure_quota"])
        selected = select_diverse(collapsed, quota, identity_ceiling)
        selected_rows.extend(selected)
        summary_rows.append(
            {
                "target_name": target,
                "raw_methylated_unique_candidates": len(all_target_rows),
                "prior_handoff_exact_sequences_excluded": prior_exact_excluded,
                "eligible_after_prior_exact_exclusion": len(target_rows),
                "unique_naturalized_candidates": len(collapsed),
                "planned_structure_quota": quota,
                "selected_structure_tasks": len(selected),
                "strict_diversity_tasks": sum(
                    row["diversity_gate"] == "STRICT_PASS" for row in selected
                ),
                "relaxed_diversity_tasks": sum(
                    row["diversity_gate"] == "RELAXED_FILL" for row in selected
                ),
                "potential_structure_reuse_matches": sum(
                    int(row["prior_handoff_natural_sequence_match"]) for row in selected
                ),
                "target_status": "READY" if len(selected) == quota else "SHORTFALL",
            }
        )

    tasks = [
        make_task(row, metadata[str(row["target_name"]).upper()])
        for row in selected_rows
    ]
    tasks.sort(key=lambda row: (str(row["target_name"]), int(row["structure_selection_rank"])))
    shortfalls = [row["target_name"] for row in summary_rows if row["target_status"] != "READY"]
    quality_gate = "PASS" if not shortfalls else "SHORTFALL"

    if ranked_pool:
        atomic_write_csv(
            out_dir / "ranked_candidate_pool.csv", ranked_pool, list(ranked_pool[0])
        )
    atomic_write_csv(out_dir / "target_summary.csv", summary_rows, list(summary_rows[0]))
    if tasks:
        atomic_write_csv(
            out_dir / "structure_tasks_for_shangge.csv", tasks, list(tasks[0])
        )
    write_structure_inputs(out_dir, tasks)
    manifest = {
        "quality_gate": quality_gate,
        "protocol": "serine_qc_structure_first_handoff_v1",
        "plan": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "candidate_file": str(candidate_path),
        "candidate_file_sha256": file_sha256(candidate_path),
        "candidate_rows": len(candidates),
        "prior_handoff_exact_sequences_excluded": sum(
            int(row["prior_handoff_exact_sequences_excluded"]) for row in summary_rows
        ),
        "targets": len(target_plan),
        "frozen_targets_not_regenerated": plan["frozen_targets"],
        "planned_structure_tasks": sum(
            int(item["structure_quota"]) for item in target_plan.values()
        ),
        "selected_structure_tasks": len(tasks),
        "shortfall_targets": shortfalls,
        "potential_structure_reuse_matches": sum(
            int(task["prior_handoff_natural_sequence_match"]) for task in tasks
        ),
        "prior_handoff": prior_manifest,
        "prior_handoff_policy": (
            "exact target+design_seq repeats are excluded; naturalized-sequence "
            "matches are never auto-reused and require an existing PDB that has "
            "already passed the same frozen structure gate"
        ),
        "ranking_rule": (
            "descending clean-V28 base_log_probability_mean; descending corrected "
            "methyl-site minimum probability; descending seeds observed and occurrence "
            "count; descending native recovery; deterministic sequence tie-break"
        ),
        "diversity_rule": (
            f"greedy naturalized-sequence identity < {identity_ceiling}; explicit "
            "RELAXED_FILL only when required to meet the frozen target quota"
        ),
        "permeability_status": "NOT_RUN_BEFORE_STRUCTURE_BY_EXECUTED_WORKFLOW",
        "rmsd_claim": "NONE_BEFORE_RETURNED_STRUCTURE_EVALUATION",
    }
    atomic_write_json(out_dir / "selection_manifest.json", manifest)

    print("===== STRUCTURE-FIRST HANDOFF COMPLETE =====")
    print(f"Quality gate: {quality_gate}")
    print(f"Selected tasks: {len(tasks)}")
    print(
        "Potential structure reuses: "
        f"{manifest['potential_structure_reuse_matches']} "
        "(verify both PDB existence and the frozen structure gate first)"
    )
    print(f"Handoff CSV: {out_dir / 'structure_tasks_for_shangge.csv'}")
    if shortfalls and not args.allow_shortfall:
        raise RuntimeError("Structure quota shortfall: " + ", ".join(shortfalls))


if __name__ == "__main__":
    main()
