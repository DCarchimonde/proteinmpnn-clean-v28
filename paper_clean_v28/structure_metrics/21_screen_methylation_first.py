#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Apply the methylation gate before structural/downstream selection.

Complex order:
1. temperature == 0.5;
2. the designed sequence contains at least one lowercase methylation mark;
3. one row per target with the minimum complete best-forward-cyclic peptide
   CA RMSD (global complex RMSD and filename are deterministic tie breakers);
4. both global complex and cyclic peptide CA RMSD must be <3 A before a row is
   considered eligible for downstream interpretation.

Monomer order:
1. the E2E design contains at least one lowercase methylation mark;
2. retain all methylated rows and annotate the RMSD, permeability-direction
   and fixed-pose Rosetta-direction gates;
3. the priority subset passes all three gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


EXPECTED_TARGETS = 17
SELECTED_TEMPERATURE = 0.5
RMSD_THRESHOLD_ANGSTROM = 3.0


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_complex_path(root: Path) -> Path:
    return (
        root
        / "paper_clean_v28_outputs"
        / "structure_metrics"
        / "best_forward_cyclic_shift_ca_rmsd"
        / "best_forward_cyclic_shift_ca_rmsd_all_4108_pdbs.csv"
    )


def default_monomer_path(root: Path) -> Path:
    return (
        root
        / "paper_clean_v28_outputs"
        / "temperature_0.5_best17"
        / "monomer_all_metrics.csv"
    )


def default_out_dir(root: Path) -> Path:
    return root / "paper_clean_v28_outputs" / "methylation_first_screen"


def methyl_count(sequence: object) -> int:
    return sum(character.islower() for character in str(sequence or "").strip())


def require_columns(frame: pd.DataFrame, columns, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric_values = pd.to_numeric(series, errors="coerce")
    text_values = series.astype(str).str.strip().str.lower()
    return numeric_values.eq(1) | text_values.isin({"true", "yes", "pass", "ok"})


def screen_complex(
    source: pd.DataFrame,
    *,
    temperature: float = SELECTED_TEMPERATURE,
    rmsd_threshold: float = RMSD_THRESHOLD_ANGSTROM,
    expected_targets: int = EXPECTED_TARGETS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cyclic_column = (
        "cyclic_peptide_ca_rmsd_after_global_complex_alignment_"
        "best_forward_cyclic_shift"
    )
    require_columns(
        source,
        [
            "target_name",
            "temperature",
            "design_seq",
            "pdb_file",
            "global_complex_ca_rmsd",
            cyclic_column,
        ],
        "complex RMSD table",
    )

    work = source.copy()
    work["target_name"] = work["target_name"].astype(str).str.upper().str.strip()
    work["temperature"] = numeric(work, "temperature")
    work["global_complex_ca_rmsd"] = numeric(work, "global_complex_ca_rmsd")
    work[cyclic_column] = numeric(work, cyclic_column)
    work["design_methyl_count"] = work["design_seq"].map(methyl_count)

    at_temperature = work[
        work["temperature"].sub(float(temperature)).abs().lt(1e-9)
    ].copy()
    if at_temperature.empty:
        raise ValueError(f"No complex rows found at temperature {temperature}")

    for column in [
        "global_complex_ca_rmsd_status",
        "cyclic_peptide_ca_rmsd_status",
    ]:
        if column in at_temperature.columns:
            at_temperature = at_temperature[
                at_temperature[column].astype(str).str.strip().str.lower().eq("ok")
            ].copy()
    for column in [
        "complete_final_chain_ca_pairing_gate",
        "decoded_design_seq_matches_design_naturalized",
    ]:
        if column in at_temperature.columns:
            at_temperature = at_temperature[truthy(at_temperature[column])].copy()

    methylated = at_temperature[
        at_temperature["design_methyl_count"].gt(0)
    ].copy()
    methylated = methylated.dropna(
        subset=["global_complex_ca_rmsd", cyclic_column]
    )
    if methylated.empty:
        raise ValueError("No methylated complex candidates remain")

    methylated = methylated.sort_values(
        [
            "target_name",
            cyclic_column,
            "global_complex_ca_rmsd",
            "pdb_file",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    methylated["methylated_cyclic_rank_within_target"] = (
        methylated.groupby("target_name", sort=False).cumcount() + 1
    )
    methylated["selected_methylated_best_for_target"] = (
        methylated["methylated_cyclic_rank_within_target"].eq(1).astype(int)
    )
    methylated["passes_global_rmsd_lt3"] = (
        methylated["global_complex_ca_rmsd"].lt(rmsd_threshold).astype(int)
    )
    methylated["passes_cyclic_peptide_rmsd_lt3"] = (
        methylated[cyclic_column].lt(rmsd_threshold).astype(int)
    )
    methylated["passes_joint_rmsd_lt3"] = (
        methylated["passes_global_rmsd_lt3"].eq(1)
        & methylated["passes_cyclic_peptide_rmsd_lt3"].eq(1)
    ).astype(int)

    best = methylated[
        methylated["selected_methylated_best_for_target"].eq(1)
    ].copy()
    if len(best) != expected_targets or best["target_name"].nunique() != expected_targets:
        counts = methylated["target_name"].value_counts().sort_index().to_dict()
        raise ValueError(
            f"Expected {expected_targets} target-wise methylated rows; "
            f"observed rows={len(best)}, targets={best['target_name'].nunique()}, "
            f"candidate counts={counts}"
        )
    strict = best[best["passes_joint_rmsd_lt3"].eq(1)].copy()
    best["downstream_eligibility"] = best["passes_joint_rmsd_lt3"].map(
        {1: "eligible", 0: "generate_more_methylated_designs"}
    )
    return methylated, best, strict


def screen_monomer(
    source: pd.DataFrame,
    *,
    rmsd_threshold: float = RMSD_THRESHOLD_ANGSTROM,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rmsd_column = "naturalized_ca_rmsd_best_forward_cyclic_shift"
    permeability_delta_column = "permeability_delta_e2e_minus_reference"
    energy_delta_column = "rosetta_score_per_residue_delta_e2e_minus_reference"
    require_columns(
        source,
        [
            "sample_name",
            "e2e_design_sequence",
            "e2e_methyl_count",
            rmsd_column,
            permeability_delta_column,
            energy_delta_column,
        ],
        "monomer all-metrics table",
    )

    work = source.copy()
    work["_lowercase_methyl_count"] = work["e2e_design_sequence"].map(methyl_count)
    reported = numeric(work, "e2e_methyl_count")
    mismatch = reported.ne(work["_lowercase_methyl_count"])
    if mismatch.any():
        examples = work.loc[
            mismatch,
            ["sample_name", "e2e_design_sequence", "e2e_methyl_count"],
        ].head(10)
        raise ValueError(
            "e2e_methyl_count disagrees with lowercase sequence marks:\n"
            + examples.to_string(index=False)
        )

    selected = work[reported.gt(0)].copy()
    excluded = work[~reported.gt(0)].copy()
    selected["selection_methyl_gate"] = "PASS"
    selected["selection_rmsd_lt3"] = (
        numeric(selected, rmsd_column).lt(rmsd_threshold)
    )
    selected["selection_permeability_improved"] = numeric(
        selected, permeability_delta_column
    ).gt(0)
    selected["selection_energy_lower"] = numeric(
        selected, energy_delta_column
    ).lt(0)
    selected["selection_core_triple_pass"] = (
        selected["selection_rmsd_lt3"]
        & selected["selection_permeability_improved"]
        & selected["selection_energy_lower"]
    )
    priority = selected[selected["selection_core_triple_pass"]].copy()
    return selected, priority, excluded


def write_outputs(
    out_dir: Path,
    complex_candidates: pd.DataFrame,
    complex_best: pd.DataFrame,
    complex_strict: pd.DataFrame,
    monomer_selected: pd.DataFrame,
    monomer_priority: pd.DataFrame,
    monomer_excluded: pd.DataFrame,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "complex_methylated_candidates": out_dir
        / "complex_t05_methylated_candidates.csv",
        "complex_target_best": out_dir
        / "complex_t05_methylated_best_by_target.csv",
        "complex_strict": out_dir
        / "complex_t05_methylated_joint_rmsd_lt3.csv",
        "monomer_selected": out_dir / "monomer_methylated_106.csv",
        "monomer_priority": out_dir / "monomer_methylated_core_priority.csv",
        "monomer_excluded": out_dir / "monomer_excluded_no_methylation.csv",
    }
    for key, frame in [
        ("complex_methylated_candidates", complex_candidates),
        ("complex_target_best", complex_best),
        ("complex_strict", complex_strict),
        ("monomer_selected", monomer_selected),
        ("monomer_priority", monomer_priority),
        ("monomer_excluded", monomer_excluded),
    ]:
        frame.to_csv(outputs[key], index=False, encoding="utf-8-sig")

    report = {
        "quality_gate": "PASS",
        "rules": {
            "complex_methylation": "lowercase-count(design_seq) > 0",
            "complex_ranking": (
                "minimum complete best-forward-cyclic peptide CA RMSD per target"
            ),
            "complex_strict": "global RMSD <3 A and cyclic peptide RMSD <3 A",
            "monomer_methylation": "e2e_methyl_count > 0",
            "monomer_priority": (
                "RMSD <3 A and permeability delta >0 and "
                "Rosetta score/residue delta <0"
            ),
        },
        "counts": {
            "complex_methylated_candidates": len(complex_candidates),
            "complex_target_best": len(complex_best),
            "complex_strict": len(complex_strict),
            "complex_targets_needing_generation": int(
                complex_best["passes_joint_rmsd_lt3"].eq(0).sum()
            ),
            "monomer_selected": len(monomer_selected),
            "monomer_priority": len(monomer_priority),
            "monomer_excluded": len(monomer_excluded),
        },
        "complex_targets_needing_generation": complex_best.loc[
            complex_best["passes_joint_rmsd_lt3"].eq(0), "target_name"
        ].tolist(),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (out_dir / "methylation_first_screen_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--complex_all_csv", default=str(default_complex_path(root)))
    parser.add_argument("--monomer_all_csv", default=str(default_monomer_path(root)))
    parser.add_argument("--out_dir", default=str(default_out_dir(root)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    complex_path = Path(args.complex_all_csv).resolve()
    monomer_path = Path(args.monomer_all_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    for input_path in [complex_path, monomer_path]:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

    complex_candidates, complex_best, complex_strict = screen_complex(
        pd.read_csv(complex_path)
    )
    monomer_selected, monomer_priority, monomer_excluded = screen_monomer(
        pd.read_csv(monomer_path)
    )
    report = write_outputs(
        out_dir,
        complex_candidates,
        complex_best,
        complex_strict,
        monomer_selected,
        monomer_priority,
        monomer_excluded,
    )
    print("===== METHYLATION-FIRST SCREEN COMPLETE =====")
    for key, value in report["counts"].items():
        print(f"{key}: {value}")
    print(
        "complex targets needing generation: "
        + ", ".join(report["complex_targets_needing_generation"])
    )
    print("QUALITY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
