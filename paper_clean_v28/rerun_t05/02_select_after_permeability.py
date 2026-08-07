#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select the small T=0.5 structure-prediction handoff after permeability.

Hard gates, in order:

1. candidate was newly generated and contains at least one lowercase
   N-methylation token;
2. the same permeability model produced both candidate and native-peptide
   predictions;
3. candidate permeability is strictly greater than its target's native
   peptide permeability;
4. candidates sharing the same naturalized sequence are collapsed because a
   structure predictor that receives naturalized sequences would otherwise do
   duplicate work;
5. within each target, backbone-conditioned V28 mean log probability is the
   structure-compatibility proxy, followed by permeability gain and native
   sequence recovery; a greedy natural-sequence diversity gate is applied.

No RMSD is claimed here. RMSD is measured only after the selected structures
return from HighFold/another structure predictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "rerun_temperature_0.5_multiseed"
)
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan.json")
LOG10_FLOOR = 1e-300


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def normalize_prediction_id(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/").split("/")[-1]
    for suffix in (".pdb", ".csv", ".txt"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    while text.lower().endswith("_model"):
        text = text[:-6]
    return text.casefold()


def float_value(value: object, field: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return result


def optional_float(value: object, default: float = -math.inf) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def find_column(fieldnames: Sequence[str], candidates: Sequence[str], label: str) -> str:
    by_casefold = {str(field).casefold(): str(field) for field in fieldnames if field is not None}
    for candidate in candidates:
        if candidate.casefold() in by_casefold:
            return by_casefold[candidate.casefold()]
    raise ValueError(f"Cannot find {label} column; available columns: {list(fieldnames)}")


def read_predictions(path: Path) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, Any]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Permeability prediction file is empty: {path}")
    fieldnames = list(rows[0])
    id_column = find_column(fieldnames, ["id", "permeability_id", "name"], "ID")
    prediction_column = find_column(
        fieldnames,
        ["permeability_pred", "prediction", "pred", "score"],
        "permeability prediction",
    )

    grouped: MutableMapping[str, List[float]] = defaultdict(list)
    invalid: List[Dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        normalized = normalize_prediction_id(row.get(id_column, ""))
        if not normalized:
            invalid.append({"row_number": row_number, "reason": "empty_id"})
            continue
        try:
            prediction = float_value(row.get(prediction_column, ""), prediction_column)
        except ValueError as exc:
            invalid.append(
                {"row_number": row_number, "id": row.get(id_column, ""), "reason": str(exc)}
            )
            continue
        if prediction < 0:
            invalid.append(
                {
                    "row_number": row_number,
                    "id": row.get(id_column, ""),
                    "reason": "negative_prediction_not_supported_by_raw_permeability_protocol",
                }
            )
            continue
        grouped[normalized].append(prediction)

    if invalid:
        examples = json.dumps(invalid[:10], ensure_ascii=False, indent=2)
        raise ValueError(f"Invalid permeability rows ({len(invalid)}):\n{examples}")

    predictions = {
        identifier: float(statistics.median(values)) for identifier, values in grouped.items()
    }
    replicate_counts = {identifier: len(values) for identifier, values in grouped.items()}
    metadata = {
        "rows": len(rows),
        "unique_normalized_ids": len(predictions),
        "id_column": id_column,
        "prediction_column": prediction_column,
        "duplicate_id_groups": sum(count > 1 for count in replicate_counts.values()),
        "aggregation_for_duplicate_ids": "median",
    }
    return predictions, replicate_counts, metadata


def sequence_identity(left: str, right: str) -> float:
    left = str(left).upper()
    right = str(right).upper()
    if not left or len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def candidate_sort_key(row: Mapping[str, Any]) -> Tuple[float, float, float, str]:
    """Structure proxy first, then permeability gain, then native recovery."""
    return (
        -optional_float(row.get("base_log_probability_mean")),
        -optional_float(row.get("permeability_delta_log10_vs_native")),
        -optional_float(row.get("natural_aa_recovery")),
        str(row.get("design_seq", "")),
    )


def collapse_naturalized_variants(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["design_natural_seq"]).upper()].append(row)

    collapsed: List[Dict[str, Any]] = []
    for natural_sequence, variants in sorted(grouped.items()):
        representative = max(
            variants,
            key=lambda row: (
                optional_float(row.get("permeability_delta_log10_vs_native")),
                optional_float(row.get("base_log_probability_mean")),
                optional_float(row.get("natural_aa_recovery")),
                str(row.get("design_seq", "")),
            ),
        )
        output = dict(representative)
        output["naturalized_variant_count"] = len(variants)
        output["alternate_methylated_sequences"] = ";".join(
            sorted({str(row["design_seq"]) for row in variants if row is not representative})
        )
        collapsed.append(output)
    return collapsed


def select_diverse(
    rows: Sequence[Mapping[str, Any]], quota: int, identity_ceiling: float
) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=candidate_sort_key)
    selected: List[Dict[str, Any]] = []
    skipped_for_identity: List[Tuple[Mapping[str, Any], float]] = []

    for source in ranked:
        row = dict(source)
        identities = [
            sequence_identity(
                str(row["design_natural_seq"]), str(previous["design_natural_seq"])
            )
            for previous in selected
        ]
        max_identity = max(identities) if identities else 0.0
        if not selected or max_identity < identity_ceiling:
            row["diversity_gate"] = "STRICT_PASS"
            row["max_identity_to_earlier_selected"] = max_identity
            selected.append(row)
            if len(selected) == quota:
                break
        else:
            skipped_for_identity.append((source, max_identity))

    if len(selected) < quota:
        selected_sequences = {str(row["design_seq"]) for row in selected}
        for source, _ in skipped_for_identity:
            if str(source["design_seq"]) in selected_sequences:
                continue
            row = dict(source)
            identities = [
                sequence_identity(
                    str(row["design_natural_seq"]), str(previous["design_natural_seq"])
                )
                for previous in selected
            ]
            row["diversity_gate"] = "RELAXED_FILL"
            row["max_identity_to_earlier_selected"] = max(identities) if identities else 0.0
            selected.append(row)
            selected_sequences.add(str(row["design_seq"]))
            if len(selected) == quota:
                break

    for rank, row in enumerate(selected, start=1):
        row["structure_selection_rank"] = rank
    return selected


def parse_plan(path: Path) -> Tuple[Dict[str, Dict[str, Any]], float, List[str]]:
    plan = read_json(path)
    if float(plan.get("temperature", -1)) != 0.5:
        raise ValueError("Selection plan is not the frozen T=0.5 protocol")
    targets = {
        str(item["target_name"]).upper(): dict(item) for item in plan.get("targets", [])
    }
    if len(targets) != 13:
        raise ValueError(f"Expected 13 rerun targets, found {len(targets)}")
    return targets, float(plan["sequence_identity_ceiling"]), [
        str(value).upper() for value in plan.get("frozen_targets", [])
    ]


def attach_predictions(
    candidates: Sequence[Mapping[str, str]],
    input_manifest: Sequence[Mapping[str, str]],
    predictions: Mapping[str, float],
    replicate_counts: Mapping[str, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[str], List[str]]:
    native_ids: Dict[str, str] = {}
    expected_ids: List[str] = []
    for row in input_manifest:
        normalized = normalize_prediction_id(row.get("id", ""))
        expected_ids.append(normalized)
        if str(row.get("record_type", "")).strip().lower() == "native":
            native_ids[str(row.get("target_name", "")).upper()] = normalized

    missing_expected = sorted(identifier for identifier in expected_ids if identifier not in predictions)
    unexpected = sorted(identifier for identifier in predictions if identifier not in set(expected_ids))

    native_predictions: Dict[str, float] = {}
    missing_native: List[str] = []
    for target, identifier in native_ids.items():
        if identifier not in predictions:
            missing_native.append(target)
        else:
            native_predictions[target] = predictions[identifier]

    output: List[Dict[str, Any]] = []
    for source in candidates:
        row: Dict[str, Any] = dict(source)
        target = str(row.get("target_name", "")).upper()
        identifier = normalize_prediction_id(row.get("permeability_id", ""))
        candidate_prediction = predictions.get(identifier)
        native_prediction = native_predictions.get(target)
        row["permeability_prediction_available"] = int(candidate_prediction is not None)
        row["permeability_prediction_replicates"] = int(replicate_counts.get(identifier, 0))
        row["permeability_pred"] = "" if candidate_prediction is None else candidate_prediction
        row["native_permeability_pred"] = "" if native_prediction is None else native_prediction

        if candidate_prediction is None or native_prediction is None:
            row["permeability_delta_vs_native"] = ""
            row["permeability_delta_log10_vs_native"] = ""
            row["permeability_fold_change_vs_native"] = ""
            row["permeability_improved_vs_native"] = "MISSING"
        else:
            delta = candidate_prediction - native_prediction
            candidate_log = math.log10(max(candidate_prediction, LOG10_FLOOR))
            native_log = math.log10(max(native_prediction, LOG10_FLOOR))
            row["permeability_delta_vs_native"] = delta
            row["permeability_delta_log10_vs_native"] = candidate_log - native_log
            row["permeability_fold_change_vs_native"] = 10 ** min(
                300.0, candidate_log - native_log
            )
            row["permeability_improved_vs_native"] = (
                "YES" if candidate_prediction > native_prediction else "NO"
            )
        output.append(row)

    return output, native_predictions, missing_expected, unexpected + [
        f"missing_native_target:{target}" for target in sorted(missing_native)
    ]


def make_structure_task(
    selected: Mapping[str, Any], target_meta: Mapping[str, str]
) -> Dict[str, Any]:
    target = str(selected["target_name"]).upper()
    rank = int(selected["structure_selection_rank"])
    return {
        "suggested_job_name": f"{target}_T0_5_rerun_{rank:02d}",
        "target_name": target,
        "temperature": 0.5,
        "selected_chain": target_meta["selected_chain"],
        "receptor_chains": target_meta["structure_receptor_chains"],
        "receptor_sequences_json": target_meta["receptor_sequences_json"],
        "native_peptide_seq": target_meta["native_peptide_seq"],
        "design_peptide_seq": selected["design_seq"],
        "design_peptide_natural_seq": selected["design_natural_seq"],
        "methyl_positions_1based": selected["methyl_positions_1based"],
        "design_methyl_count": selected["design_methyl_count"],
        "permeability_pred": selected["permeability_pred"],
        "native_permeability_pred": selected["native_permeability_pred"],
        "permeability_delta_vs_native": selected["permeability_delta_vs_native"],
        "permeability_delta_log10_vs_native": selected[
            "permeability_delta_log10_vs_native"
        ],
        "permeability_fold_change_vs_native": selected[
            "permeability_fold_change_vs_native"
        ],
        "permeability_improved_vs_native": selected[
            "permeability_improved_vs_native"
        ],
        "base_log_probability_mean": selected["base_log_probability_mean"],
        "natural_aa_recovery": selected["natural_aa_recovery"],
        "naturalized_variant_count": selected["naturalized_variant_count"],
        "alternate_methylated_sequences": selected["alternate_methylated_sequences"],
        "diversity_gate": selected["diversity_gate"],
        "max_identity_to_earlier_selected": selected[
            "max_identity_to_earlier_selected"
        ],
        "structure_selection_rank": rank,
        "source_candidate_id": selected["candidate_id"],
        "source_permeability_id": selected["permeability_id"],
        "source_seed": selected["seed"],
        "current_problem": selected["current_problem"],
        "final_required_gate": (
            "after structure prediction: global complex CA RMSD <3 A AND "
            "complete cyclic-peptide CA RMSD after the one global alignment and "
            "best forward cyclic shift <3 A; no peptide-only refit"
        ),
    }


def write_structure_inputs(out_dir: Path, tasks: Sequence[Mapping[str, Any]]) -> None:
    input_dir = out_dir / "structure_inputs_for_shangge"
    input_dir.mkdir(parents=True, exist_ok=True)
    combined_lines: List[str] = []
    jsonl_lines: List[str] = []

    for task in tasks:
        job_name = str(task["suggested_job_name"])
        receptors = json.loads(str(task["receptor_sequences_json"]))
        lines: List[str] = []
        for chain_id, sequence in receptors.items():
            header = f">{job_name}|chain={chain_id}|role=receptor"
            lines.extend([header, str(sequence)])
        peptide_header = (
            f">{job_name}|chain={task['selected_chain']}|role=design_peptide"
            f"|methyl_positions_1based={task['methyl_positions_1based']}"
        )
        lines.extend([peptide_header, str(task["design_peptide_natural_seq"])])
        content = "\n".join(lines) + "\n"
        temp = input_dir / f"{job_name}.fasta.tmp"
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, input_dir / f"{job_name}.fasta")
        combined_lines.extend(lines)

        jsonl_lines.append(
            json.dumps(
                {
                    "job_name": job_name,
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

    combined_temp = out_dir / "structure_tasks_for_shangge.fasta.tmp"
    combined_temp.write_text(
        "\n".join(combined_lines) + ("\n" if combined_lines else ""),
        encoding="utf-8",
    )
    os.replace(combined_temp, out_dir / "structure_tasks_for_shangge.fasta")
    jsonl_temp = out_dir / "structure_tasks_for_shangge.jsonl.tmp"
    jsonl_temp.write_text(
        "\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""),
        encoding="utf-8",
    )
    os.replace(jsonl_temp, out_dir / "structure_tasks_for_shangge.jsonl")


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    plan_path = Path(args.plan).resolve()
    candidate_path = Path(args.candidates_csv or run_dir / "methylated_new_candidates.csv")
    input_manifest_path = Path(
        args.input_manifest_csv or run_dir / "permeability_input_manifest.csv"
    )
    target_manifest_path = Path(args.target_manifest_csv or run_dir / "target_manifest.csv")
    prediction_path = Path(args.permeability_csv).resolve()
    out_dir = Path(args.out_dir or run_dir / "selected_for_structure").resolve()

    for required in (
        plan_path,
        candidate_path,
        input_manifest_path,
        target_manifest_path,
        prediction_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    target_plan, identity_ceiling, frozen_targets = parse_plan(plan_path)
    candidates = read_csv(candidate_path)
    input_manifest = read_csv(input_manifest_path)
    target_manifest_rows = read_csv(target_manifest_path)
    target_manifest = {
        str(row["target_name"]).upper(): row for row in target_manifest_rows
    }
    predictions, replicate_counts, prediction_metadata = read_predictions(prediction_path)
    attached, native_predictions, missing_expected, unexpected = attach_predictions(
        candidates, input_manifest, predictions, replicate_counts
    )

    missing_native = sorted(set(target_plan) - set(native_predictions))
    if missing_native:
        raise RuntimeError(
            "Native permeability baselines are missing for rerun targets: "
            + ", ".join(missing_native)
        )
    if missing_expected and not args.allow_partial_predictions:
        preview = "\n".join(f"  - {value}" for value in missing_expected[:20])
        raise RuntimeError(
            f"Permeability coverage gate failed: {len(missing_expected)} expected IDs are missing. "
            "Rerun the permeability model, or use --allow-partial-predictions only for an "
            f"explicit exploratory partial handoff.\n{preview}"
        )

    improved = [
        row for row in attached if row["permeability_improved_vs_native"] == "YES"
    ]
    selected_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for target in target_plan:
        target_attached = [row for row in attached if str(row["target_name"]).upper() == target]
        target_improved = [
            row for row in target_attached if row["permeability_improved_vs_native"] == "YES"
        ]
        collapsed = collapse_naturalized_variants(target_improved)
        quota = int(target_plan[target]["structure_quota"])
        selected = select_diverse(collapsed, quota, identity_ceiling)
        selected_rows.extend(selected)
        summary_rows.append(
            {
                "target_name": target,
                "current_problem": target_plan[target]["current_problem"],
                "permeability_candidates_expected": len(target_attached),
                "permeability_predictions_available": sum(
                    int(row["permeability_prediction_available"]) for row in target_attached
                ),
                "permeability_improved_candidates": len(target_improved),
                "unique_natural_sequences_improved": len(collapsed),
                "planned_structure_quota": quota,
                "selected_structure_tasks": len(selected),
                "strict_diversity_tasks": sum(
                    row["diversity_gate"] == "STRICT_PASS" for row in selected
                ),
                "relaxed_diversity_tasks": sum(
                    row["diversity_gate"] == "RELAXED_FILL" for row in selected
                ),
                "target_status": (
                    "READY"
                    if len(selected) == quota
                    else "NO_PERMEABILITY_IMPROVEMENT"
                    if not selected
                    else "SHORTFALL"
                ),
            }
        )

    structure_tasks: List[Dict[str, Any]] = []
    for selected in selected_rows:
        target = str(selected["target_name"]).upper()
        if target not in target_manifest:
            raise RuntimeError(f"Target metadata is missing for {target}")
        structure_tasks.append(make_structure_task(selected, target_manifest[target]))
    structure_tasks.sort(key=lambda row: (str(row["target_name"]), int(row["structure_selection_rank"])))

    attached_fields = list(attached[0].keys()) if attached else []
    attached_extra = [
        "permeability_prediction_available",
        "permeability_prediction_replicates",
        "permeability_pred",
        "native_permeability_pred",
        "permeability_delta_vs_native",
        "permeability_delta_log10_vs_native",
        "permeability_fold_change_vs_native",
        "permeability_improved_vs_native",
    ]
    attached_fields += [field for field in attached_extra if field not in attached_fields]
    improved_fields = attached_fields + [
        "naturalized_variant_count",
        "alternate_methylated_sequences",
    ]
    atomic_write_csv(out_dir / "candidate_pool_with_permeability.csv", attached, attached_fields)
    atomic_write_csv(out_dir / "permeability_improved_candidates.csv", improved, attached_fields)
    atomic_write_csv(
        out_dir / "target_summary.csv", summary_rows, list(summary_rows[0].keys())
    )
    task_fields = list(structure_tasks[0].keys()) if structure_tasks else list(
        make_structure_task(
            {
                "target_name": next(iter(target_plan)),
                "structure_selection_rank": 1,
                "design_seq": "",
                "design_natural_seq": "",
                "methyl_positions_1based": "[]",
                "design_methyl_count": 0,
                "permeability_pred": "",
                "native_permeability_pred": "",
                "permeability_delta_vs_native": "",
                "permeability_delta_log10_vs_native": "",
                "permeability_fold_change_vs_native": "",
                "permeability_improved_vs_native": "",
                "base_log_probability_mean": "",
                "natural_aa_recovery": "",
                "naturalized_variant_count": 0,
                "alternate_methylated_sequences": "",
                "diversity_gate": "",
                "max_identity_to_earlier_selected": "",
                "candidate_id": "",
                "permeability_id": "",
                "seed": "",
                "current_problem": "",
            },
            target_manifest[next(iter(target_plan))],
        ).keys()
    )
    atomic_write_csv(out_dir / "structure_tasks_for_shangge.csv", structure_tasks, task_fields)
    write_structure_inputs(out_dir, structure_tasks)

    ready_targets = [row["target_name"] for row in summary_rows if row["target_status"] == "READY"]
    shortfall_targets = [
        row["target_name"] for row in summary_rows if row["target_status"] != "READY"
    ]
    expected_prediction_ids = {
        normalize_prediction_id(row.get("id", "")) for row in input_manifest
    }
    quality_gate = (
        "PASS"
        if not missing_expected and not shortfall_targets
        else "EXPLORATORY_PARTIAL"
        if args.allow_partial_predictions and missing_expected
        else "NEEDS_MORE_CANDIDATES"
    )
    report = {
        "quality_gate": quality_gate,
        "protocol": "T=0.5 methylated redesign vs native permeability prescreen",
        "permeability_comparison": "candidate_raw_prediction > same-model native-peptide_raw_prediction",
        "permeability_source": str(prediction_path),
        "prediction_metadata": prediction_metadata,
        "expected_prediction_ids": len(expected_prediction_ids),
        "missing_expected_prediction_ids": missing_expected,
        "unexpected_prediction_ids": unexpected,
        "native_baselines_available": len(native_predictions),
        "candidate_rows": len(candidates),
        "candidate_predictions_available": sum(
            int(row["permeability_prediction_available"]) for row in attached
        ),
        "permeability_improved_candidates": len(improved),
        "structure_tasks": len(structure_tasks),
        "planned_structure_tasks": sum(
            int(item["structure_quota"]) for item in target_plan.values()
        ),
        "ready_targets": ready_targets,
        "shortfall_targets": shortfall_targets,
        "frozen_targets_not_regenerated": frozen_targets,
        "sequence_identity_rule": (
            f"greedy selected natural sequences require pairwise identity < {identity_ceiling}; "
            "remaining quota is explicitly marked RELAXED_FILL"
        ),
        "ranking_rule": (
            "after strict permeability improvement: descending clean-V28 backbone-conditioned "
            "mean base log probability, then descending log10 permeability gain, then native recovery"
        ),
        "rmsd_claim": "NONE_BEFORE_STRUCTURE_PREDICTION",
        "final_rmsd_gate": (
            "global complex CA RMSD <3 A and complete cyclic-peptide CA RMSD <3 A after exactly "
            "one global alignment and best forward cyclic shift; no peptide-only refit"
        ),
    }
    atomic_write_json(out_dir / "selection_manifest.json", report)

    print("\n===== PERMEABILITY PRESCREEN COMPLETE =====", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    print(f"Improved candidates: {len(improved)}", flush=True)
    print(f"Structure tasks: {len(structure_tasks)}", flush=True)
    print(f"Ready targets: {len(ready_targets)}/13", flush=True)
    if shortfall_targets:
        print("Targets needing additional local sampling: " + ", ".join(shortfall_targets), flush=True)
    print(f"Handoff CSV: {out_dir / 'structure_tasks_for_shangge.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permeability_csv", required=True)
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--candidates_csv")
    parser.add_argument("--input_manifest_csv")
    parser.add_argument("--target_manifest_csv")
    parser.add_argument("--out_dir")
    parser.add_argument("--allow-partial-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
