#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge all per-structure temperature-0.5 best17 results into one table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_ROWS = 17
SELECTED_TEMPERATURE = 0.5


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_run_dir(root: Path) -> Path:
    return root / "paper_clean_v28_outputs" / "temperature_0.5_best17"


def norm_temp(value) -> str:
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value).strip()


def naturalize(value) -> str:
    return str(value or "").strip().upper()


def add_key(df: pd.DataFrame, seq_candidates: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    seq_col = next((c for c in seq_candidates if c in out.columns), None)
    if seq_col is None:
        raise ValueError(f"No sequence key column found among: {list(seq_candidates)}")
    out["_target_key"] = out["target_name"].astype(str).str.upper().str.strip()
    out["_temperature_key"] = out["temperature"].map(norm_temp)
    out["_design_key"] = out[seq_col].astype(str).str.strip()
    out["_merge_key"] = (
        out["_target_key"] + "|" + out["_temperature_key"] + "|" + out["_design_key"]
    )
    return out


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def merge_unique(
    base: pd.DataFrame,
    other: pd.DataFrame,
    label: str,
    seq_candidates: Iterable[str],
    columns: Iterable[str] | None = None,
    allow_consistent_duplicates: bool = False,
) -> pd.DataFrame:
    keyed = add_key(other, seq_candidates)
    if keyed["_merge_key"].duplicated().any():
        if not allow_consistent_duplicates:
            duplicate = keyed.loc[
                keyed["_merge_key"].duplicated(False), "_merge_key"
            ].tolist()
            raise ValueError(f"{label} contains duplicate merge keys: {duplicate}")
        check_columns = (
            [c for c in columns if c in keyed.columns]
            if columns is not None
            else [c for c in keyed.columns if not c.startswith("_")]
        )
        inconsistent = []
        for key, group in keyed.groupby("_merge_key", dropna=False):
            for column in check_columns:
                values = group[column].dropna().astype(str).unique()
                if len(values) > 1:
                    inconsistent.append(f"{key}:{column}")
        if inconsistent:
            raise ValueError(
                f"{label} duplicate keys have inconsistent values: {inconsistent}"
            )
        keyed = keyed.drop_duplicates("_merge_key", keep="first")
    keep = ["_merge_key"]
    if columns is None:
        keep.extend([c for c in keyed.columns if not c.startswith("_")])
    else:
        keep.extend([c for c in columns if c in keyed.columns])
    keep = list(dict.fromkeys(keep))
    rename = {c: f"{label}__{c}" for c in keep if c != "_merge_key"}
    return base.merge(
        keyed[keep].rename(columns=rename),
        on="_merge_key",
        how="left",
        validate="1:1",
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def get_column(df: pd.DataFrame, names: Iterable[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def first_existing_path(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected files exist:\n" + "\n".join(map(str, paths)))


def make_final_table(merged: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=merged.index)
    direct = [
        "target_name",
        "temperature",
        "pdb_file",
        "pdb_path",
        "design_seq",
        "design_natural_seq",
        "design_length",
        "native_seq",
        "global_complex_ca_rmsd",
        "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift",
        "cyclic_peptide_ca_rmsd_after_global_complex_alignment_fixed_order",
        "cyclic_peptide_best_forward_cyclic_shift",
        "cyclic_peptide_ca_rmsd_improvement_from_forward_shift",
        "passes_global_complex_ca_rmsd_lt_threshold",
        "passes_cyclic_peptide_ca_rmsd_lt_threshold",
        "passes_joint_global_and_cyclic_peptide_lt_threshold",
    ]
    for column in direct:
        out[column] = get_column(merged, [column])

    mappings = {
        "natural_aa_recovery": ["natural_aa_recovery", "sequence__natural_aa_recovery"],
        "native_natural_seq": ["sequence__native_natural_seq"],
        "length_match": ["sequence__length_match"],
        "design_methyl_count": ["sequence__design_methyl_count"],
        "design_methyl_rate": ["sequence__design_methyl_rate"],
        "n_methylation_marks": [
            "energy__n_methylation_marks",
            "n_methylation_marks",
        ],
        "highfold_plddt_comment": [
            "highfold_plddt_comment",
            "rmsd__highfold_plddt_comment",
            "rmsd__highfold_plddt",
            "confidence__highfold_plddt",
        ],
        "highfold_ptm_comment": ["confidence__highfold_ptm"],
        "highfold_ca_bfactor_mean": [
            "highfold_ca_bfactor_mean",
            "rmsd__highfold_ca_bfactor_mean",
            "confidence__highfold_pdb_ca_bfactor_mean",
        ],
        "peptide_chain_plddt": ["rmsd__peptide_chain_plddt"],
        "peptide_chain_ptm": ["rmsd__peptide_chain_ptm"],
        "peptide_chain_iptm": ["rmsd__peptide_chain_iptm"],
        "peptide_chain_pae": ["rmsd__peptide_chain_pae"],
        "receptor_chain_plddt_mean": ["rmsd__receptor_chain_plddt_mean"],
        "peptide_receptor_iptm_mean": [
            "rmsd__peptide_receptor_iptm_mean",
            "peptide_receptor_iptm_mean",
        ],
        "peptide_receptor_inter_pae_mean": [
            "rmsd__peptide_receptor_inter_pae_mean",
            "peptide_receptor_inter_pae_mean",
        ],
        "peptide_receptor_ipae_mean": ["rmsd__peptide_receptor_ipae_mean"],
        "legacy_receptor_ca_fit_rmsd": [
            "rmsd__receptor_ca_fit_rmsd",
            "receptor_ca_fit_rmsd",
        ],
        "legacy_peptide_ca_rmsd_after_receptor_fit": [
            "rmsd__peptide_ca_rmsd_after_receptor_fit",
            "peptide_ca_rmsd_after_receptor_fit",
        ],
        "legacy_peptide_backbone_rmsd_after_receptor_fit": [
            "rmsd__peptide_backbone_rmsd_after_receptor_fit",
            "peptide_backbone_rmsd_after_receptor_fit",
        ],
        "legacy_peptide_ca_rmsd_self_superposed": [
            "rmsd__peptide_ca_rmsd_self_superposed",
            "peptide_ca_rmsd_self_superposed",
        ],
        "methyl_ca_rmsd_after_receptor_fit": [
            "methyl__methyl_ca_rmsd_after_receptor_fit"
        ],
        "methyl_backbone_rmsd_after_receptor_fit": [
            "methyl__methyl_backbone_rmsd_after_receptor_fit"
        ],
        "nonmethyl_ca_rmsd_after_receptor_fit": [
            "methyl__nonmethyl_ca_rmsd_after_receptor_fit"
        ],
        "nonmethyl_backbone_rmsd_after_receptor_fit": [
            "methyl__nonmethyl_backbone_rmsd_after_receptor_fit"
        ],
        "methyl_ca_rmsd_after_peptide_self_fit": [
            "methyl__methyl_ca_rmsd_after_peptide_self_fit"
        ],
        "nonmethyl_ca_rmsd_after_peptide_self_fit": [
            "methyl__nonmethyl_ca_rmsd_after_peptide_self_fit"
        ],
        "methyl_backbone_rmsd_after_peptide_self_fit": [
            "methyl__methyl_backbone_rmsd_after_peptide_self_fit"
        ],
        "nonmethyl_backbone_rmsd_after_peptide_self_fit": [
            "methyl__nonmethyl_backbone_rmsd_after_peptide_self_fit"
        ],
        "permeability_pred": ["permeability__permeability_pred"],
        "permeability_pred_mean": ["permeability__permeability_pred_mean"],
        "permeability_pred_median": ["permeability__permeability_pred_median"],
        "permeability_log10_positive_only": [
            "permeability__permeability_log10_positive_only"
        ],
        "permeability_log10_floor300": [
            "permeability__permeability_log10_floor300"
        ],
        "permeability_match_count": ["permeability__permeability_match_count"],
        "permeability_match_mode": ["permeability__permeability_match_mode"],
        "rosetta_complex_total_score": ["energy__rosetta_complex_total_score"],
        "rosetta_complex_score_per_residue": [
            "energy__rosetta_complex_score_per_residue"
        ],
        "rosetta_receptor_subset_score": [
            "energy__rosetta_receptor_subset_score"
        ],
        "rosetta_receptor_subset_score_per_residue": [
            "energy__rosetta_receptor_subset_score_per_residue"
        ],
        "rosetta_peptide_subset_score": ["energy__rosetta_peptide_subset_score"],
        "rosetta_peptide_subset_score_per_residue": [
            "energy__rosetta_peptide_subset_score_per_residue"
        ],
        "rosetta_cross_interface_energy_fixed_pose": [
            "energy__rosetta_cross_interface_energy_fixed_pose"
        ],
        "rosetta_cross_interface_energy_per_peptide_residue": [
            "energy__rosetta_cross_interface_energy_per_peptide_residue"
        ],
        "cross_interface_energy_lt_0": ["energy__cross_interface_energy_lt_0"],
        "rmsd_status": ["rmsd__rmsd_status"],
        "methylation_site_rmsd_status": ["methyl__site_rmsd_status"],
        "energy_status": ["energy__energy_status"],
    }
    for output_name, candidates in mappings.items():
        out[output_name] = get_column(merged, candidates)

    out["within_target_tm_diversity_1_minus_tm"] = np.nan
    out["within_target_tm_diversity_status"] = (
        "not_estimable_one_temperature_one_structure_per_target"
    )
    out["energy_success_vs_native"] = np.nan
    out["energy_stability_vs_native"] = np.nan
    out["energy_delta_status"] = (
        "not_estimable_native_reference_energy_not_computed_by_fixed_pose_workflow"
    )
    out["binding_site_recovery_ratio"] = np.nan
    out["binding_site_recovery_status"] = (
        "not_computed_no_validated_binding_site_definition_in_previous_workflow"
    )
    out["primary_rmsd_definition"] = (
        "one global PyMOL complex alignment then complete final-chain CA RMSD "
        "with best forward cyclic shift and no peptide-only second fit"
    )
    return out.sort_values("target_name").reset_index(drop=True)


def quality_report(final: pd.DataFrame) -> tuple[bool, list[str]]:
    problems = []
    if len(final) != EXPECTED_ROWS:
        problems.append(f"row_count_expected_{EXPECTED_ROWS}_observed_{len(final)}")
    if final["target_name"].astype(str).str.upper().nunique() != EXPECTED_ROWS:
        problems.append("target_count_or_uniqueness_failed")
    temp = numeric(final["temperature"])
    if not temp.sub(SELECTED_TEMPERATURE).abs().lt(1e-9).all():
        problems.append("temperature_not_uniformly_0.5")
    if final["pdb_file"].astype(str).duplicated().any():
        problems.append("duplicate_selected_pdb_file")

    required_numeric = [
        "global_complex_ca_rmsd",
        "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift",
        "highfold_ca_bfactor_mean",
        "permeability_pred",
        "rosetta_complex_score_per_residue",
        "rosetta_cross_interface_energy_fixed_pose",
    ]
    for column in required_numeric:
        n = int(numeric(final[column]).notna().sum())
        if n != EXPECTED_ROWS:
            problems.append(f"{column}_available_{n}_of_{EXPECTED_ROWS}")

    status_requirements = {
        "rmsd_status": "ok",
        "methylation_site_rmsd_status": "ok",
        "energy_status": "ok",
    }
    for column, expected in status_requirements.items():
        bad = final[column].astype(str).str.lower().ne(expected)
        if bad.any():
            problems.append(f"{column}_not_{expected}_{int(bad.sum())}_rows")
    return len(problems) == 0, problems


def main() -> int:
    root = repository_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(default_run_dir(root)))
    parser.add_argument(
        "--all_designs_csv",
        default=str(
            root
            / "paper_clean_v28_outputs"
            / "generated_fasta_clean_auto_single"
            / "all_designs.csv"
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    workspace = run_dir / "workspace"
    metrics_dir = workspace / "paper_clean_v28_outputs" / "structure_metrics"
    permeability_dir = workspace / "paper_clean_v28_outputs" / "permeability"

    selection_path = require_file(run_dir / "temperature05_best17_selection.csv")
    rmsd_path = require_file(metrics_dir / "complex_rmsd_metrics.csv")
    confidence_path = require_file(
        metrics_dir / "complex_best85_highfold_representative.csv"
    )
    methyl_path = require_file(metrics_dir / "complex_methylation_site_rmsd_by_design.csv")
    energy_path = first_existing_path(
        [
            metrics_dir / "complex_pyrosetta_energy_naturalized_best85.csv",
            metrics_dir / "complex_pyrosetta_energy_naturalized_best17.csv",
        ]
    )
    permeability_path = require_file(
        permeability_dir / "complex_permeability_best85.csv"
    )
    all_designs_path = require_file(Path(args.all_designs_csv).resolve())

    selection = add_key(pd.read_csv(selection_path), ["design_seq"])
    if len(selection) != EXPECTED_ROWS or selection["_merge_key"].duplicated().any():
        raise ValueError("Selection table must contain 17 unique target-temperature-design rows")

    merged = selection.copy()
    merged = merge_unique(
        merged,
        pd.read_csv(all_designs_path),
        "sequence",
        ["design_seq", "design_peptide_seq"],
        columns=[
            "natural_aa_recovery",
            "native_seq",
            "native_natural_seq",
            "design_natural_seq",
            "length_match",
            "design_methyl_count",
            "design_methyl_rate",
        ],
        allow_consistent_duplicates=True,
    )
    merged = merge_unique(
        merged,
        pd.read_csv(rmsd_path),
        "rmsd",
        ["design_seq"],
    )
    merged = merge_unique(
        merged,
        pd.read_csv(confidence_path),
        "confidence",
        ["design_peptide_seq", "design_seq"],
    )
    merged = merge_unique(
        merged,
        pd.read_csv(methyl_path),
        "methyl",
        ["design_seq"],
    )
    merged = merge_unique(
        merged,
        pd.read_csv(energy_path),
        "energy",
        ["design_seq"],
    )
    merged = merge_unique(
        merged,
        pd.read_csv(permeability_path),
        "permeability",
        ["design_seq", "design_peptide_seq"],
    )

    final = make_final_table(merged)
    quality_pass, problems = quality_report(final)

    final_csv = run_dir / "temperature05_best17_all_metrics.csv"
    merged_csv = run_dir / "temperature05_best17_all_metrics_audit_wide.csv"
    report_path = run_dir / "temperature05_best17_all_metrics_report.txt"
    problem_path = run_dir / "temperature05_best17_all_metrics_problem_rows.csv"

    final.to_csv(final_csv, index=False, encoding="utf-8-sig")
    merged.drop(columns=[c for c in merged.columns if c.startswith("_")]).to_csv(
        merged_csv,
        index=False,
        encoding="utf-8-sig",
    )

    problem_rows = []
    if problems:
        problem_rows = [{"problem": item} for item in problems]
    pd.DataFrame(problem_rows, columns=["problem"]).to_csv(
        problem_path,
        index=False,
        encoding="utf-8-sig",
    )

    global_rmsd = numeric(final["global_complex_ca_rmsd"])
    cyclic_rmsd = numeric(
        final[
            "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift"
        ]
    )
    interface = numeric(final["rosetta_cross_interface_energy_fixed_pose"])
    permeability = numeric(final["permeability_pred"])

    lines = [
        "===== TEMPERATURE 0.5 BEST17 ALL-METRICS REPORT =====",
        f"Expected rows: {EXPECTED_ROWS}",
        f"Observed rows: {len(final)}",
        f"Unique targets: {final['target_name'].nunique()}",
        f"Temperature 0.5 rows: {int(numeric(final['temperature']).eq(0.5).sum())}",
        "",
        "===== PRIMARY RMSD =====",
        f"Median global complex CA RMSD: {global_rmsd.median():.6f}",
        f"Global complex CA RMSD < 3 A: {int((global_rmsd < 3).sum())}/{len(final)}",
        f"Median best-forward-cyclic peptide CA RMSD: {cyclic_rmsd.median():.6f}",
        f"Cyclic peptide CA RMSD < 3 A: {int((cyclic_rmsd < 3).sum())}/{len(final)}",
        "",
        "===== ENERGY AND PERMEABILITY =====",
        f"Median cross-interface energy (REU): {interface.median():.6f}",
        f"Cross-interface energy < 0: {int((interface < 0).sum())}/{len(final)}",
        f"Permeability available: {int(permeability.notna().sum())}/{len(final)}",
        "",
        "===== NON-ESTIMABLE UNDER THE 17-STRUCTURE DESIGN =====",
        "- Within-target TM diversity (1-TM): one structure per target leaves zero within-target pairs.",
        "- Energy Success/Stability versus native: native reference energies are not produced by the fixed-pose workflow.",
        "- Binding-site recovery ratio: the previous workflow did not establish a validated binding-site definition.",
        "- These fields remain NA; they are never replaced with zero.",
        "",
        "===== METRIC SCOPE NOTES =====",
        "- Primary peptide RMSD uses one global PyMOL complex alignment, all final-chain CAs, best forward cyclic shift, and no peptide-only second fit.",
        "- Columns prefixed legacy_ reproduce older receptor-CA-fit/self-fit metrics for audit only; they are not the primary RMSD claim.",
        "- Rosetta scores are fixed-pose naturalized ref2015 scores in REU, not kcal/mol and not explicit N-methyl chemistry.",
        "- COMMENT pLDDT and mean CA B-factor are separate fields; missing ipTM/inter-PAE values remain missing.",
        "",
        f"QUALITY GATE: {'PASS' if quality_pass else 'FAIL'}",
        f"PROBLEMS: {len(problems)}",
    ]
    if problems:
        lines.extend(["", "Problem details:"] + [f"- {x}" for x in problems])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "quality_gate": "PASS" if quality_pass else "FAIL",
        "rows": len(final),
        "targets": int(final["target_name"].nunique()),
        "temperature": SELECTED_TEMPERATURE,
        "final_csv": str(final_csv),
        "audit_wide_csv": str(merged_csv),
        "report": str(report_path),
        "problems": problems,
    }
    (run_dir / "temperature05_best17_output_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("===== TEMPERATURE 0.5 BEST17 FINAL MERGE COMPLETE =====")
    print(f"rows: {len(final)}")
    print(f"unique targets: {final['target_name'].nunique()}")
    print(f"quality gate: {'PASS' if quality_pass else 'FAIL'}")
    print(f"final table: {final_csv}")
    print(f"audit-wide table: {merged_csv}")
    print(f"report: {report_path}")
    if problems:
        print("problems:")
        for item in problems:
            print(f"  - {item}")
    return 0 if quality_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
