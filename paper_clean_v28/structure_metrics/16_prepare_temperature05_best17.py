#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare an isolated temperature-0.5 best17 downstream-metric run.

The input is the corrected RMSD-best85 table produced after evaluating all
forward cyclic shifts.  Exactly one row per target at temperature 0.5 is kept.
The selected PDB is preserved exactly; it is not re-selected by pLDDT.

The script creates an isolated workspace beneath
``paper_clean_v28_outputs/temperature_0.5_best17`` so the historical best85
outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_SOURCE_ROWS = 85
EXPECTED_TARGETS = 17
SELECTED_TEMPERATURE = 0.5


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_selection_path(root: Path) -> Path:
    return (
        root
        / "paper_clean_v28_outputs"
        / "structure_metrics"
        / "best_forward_cyclic_shift_ca_rmsd"
        / "best_forward_cyclic_shift_new_rmsd_best85_all_valid.csv"
    )


def default_run_dir(root: Path) -> Path:
    return root / "paper_clean_v28_outputs" / "temperature_0.5_best17"


def norm_temp(value) -> str:
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value).strip()


def truthy_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).eq(1)


def resolve_repo_path(root: Path, value: object) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty path")
    normalized = text.replace("\\", os.sep).replace("/", os.sep)
    path = Path(normalized)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_and_select(source: pd.DataFrame, root: Path) -> pd.DataFrame:
    required = {
        "target_name",
        "temperature",
        "design_seq",
        "pdb_file",
        "pdb_path",
        "global_complex_ca_rmsd",
        "cyclic_peptide_ca_rmsd_after_global_complex_alignment_best_forward_cyclic_shift",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Selection CSV is missing required columns: {missing}")
    if len(source) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ROWS} RMSD-best85 rows, observed {len(source)}"
        )

    work = source.copy()
    work["target_name"] = work["target_name"].astype(str).str.upper().str.strip()
    work["_temperature_numeric"] = pd.to_numeric(work["temperature"], errors="coerce")
    selected = work[
        work["_temperature_numeric"].sub(SELECTED_TEMPERATURE).abs().lt(1e-9)
    ].copy()

    if len(selected) != EXPECTED_TARGETS:
        raise ValueError(
            f"Expected {EXPECTED_TARGETS} rows at temperature 0.5, observed {len(selected)}"
        )
    if selected["target_name"].nunique() != EXPECTED_TARGETS:
        counts = selected["target_name"].value_counts().to_dict()
        raise ValueError(f"Expected 17 unique targets at temperature 0.5: {counts}")

    duplicate_keys = selected.duplicated(["target_name", "_temperature_numeric"], keep=False)
    if duplicate_keys.any():
        rows = selected.loc[duplicate_keys, ["target_name", "temperature", "pdb_file"]]
        raise ValueError(f"Duplicate target-temperature rows:\n{rows.to_string(index=False)}")

    gates = {
        "global_complex_ca_rmsd_status": "ok",
        "cyclic_peptide_ca_rmsd_status": "ok",
    }
    for column, expected in gates.items():
        if column in selected.columns:
            bad = selected[column].astype(str).str.lower().ne(expected)
            if bad.any():
                raise ValueError(
                    f"{column} quality gate failed for: "
                    f"{selected.loc[bad, 'target_name'].tolist()}"
                )

    for column in [
        "complete_final_chain_ca_pairing_gate",
        "decoded_design_seq_matches_design_naturalized",
    ]:
        if column in selected.columns:
            bad = ~truthy_numeric(selected[column])
            if bad.any():
                raise ValueError(
                    f"{column} quality gate failed for: "
                    f"{selected.loc[bad, 'target_name'].tolist()}"
                )

    if "rmsd_rank_within_group" in selected.columns:
        rank = pd.to_numeric(selected["rmsd_rank_within_group"], errors="coerce")
        if not rank.eq(1).all():
            raise ValueError("Every selected row must have rmsd_rank_within_group == 1")

    resolved_paths = []
    for _, row in selected.iterrows():
        path = resolve_repo_path(root, row["pdb_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Selected PDB does not exist for {row['target_name']}: {path}"
            )
        if path.name.lower() != str(row["pdb_file"]).lower():
            raise ValueError(
                f"PDB filename/path mismatch for {row['target_name']}: "
                f"{row['pdb_file']} vs {path.name}"
            )
        resolved_paths.append(str(path))

    selected["temperature"] = SELECTED_TEMPERATURE
    selected["pdb_path_original"] = selected["pdb_path"]
    selected["pdb_path"] = resolved_paths
    selected["selection_temperature_rule"] = "temperature_equals_0.5"
    selected["selection_rmsd_rule"] = (
        "minimum complete best-forward-cyclic peptide CA RMSD within target x temperature"
    )
    selected = selected.drop(columns=["_temperature_numeric"])
    return selected.sort_values("target_name").reset_index(drop=True)


def archive_existing_workspace(workspace: Path) -> Path | None:
    if not workspace.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = workspace.with_name(f"{workspace.name}_previous_{stamp}")
    suffix = 1
    while archived.exists():
        archived = workspace.with_name(f"{workspace.name}_previous_{stamp}_{suffix}")
        suffix += 1
    workspace.rename(archived)
    return archived


def patch_expected_rows(path: Path, replacements: Iterable[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"Expected staging patch text not found in {path}: {old}")
        text = text.replace(old, new, 1)
    if text == original:
        raise RuntimeError(f"No staging patch was applied to {path}")
    path.write_text(text, encoding="utf-8")


def build_manifest(selected: pd.DataFrame) -> pd.DataFrame:
    manifest = selected.copy()
    manifest["design_peptide_seq"] = manifest["design_seq"].astype(str)
    if "design_natural_seq" not in manifest.columns:
        manifest["design_natural_seq"] = manifest["design_seq"].astype(str).str.upper()
    if "design_length" not in manifest.columns:
        manifest["design_length"] = manifest["design_seq"].astype(str).str.len()
    manifest.insert(0, "temperature05_best17_row_index", range(len(manifest)))
    return manifest


def build_selected_highfold_scores(
    selected: pd.DataFrame,
    parser_module,
    staged_pdb_dir: Path,
) -> pd.DataFrame:
    staged_pdb_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, source in selected.iterrows():
        pdb_path = Path(str(source["pdb_path"]))
        staged_pdb_path = staged_pdb_dir / pdb_path.name
        shutil.copy2(pdb_path, staged_pdb_path)
        parsed_temp, temp_folder = parser_module.parse_temp_from_path(pdb_path)
        meta = parser_module.parse_filename(pdb_path)
        scores, pdb_stats = parser_module.parse_pdb(pdb_path)
        design_seq = str(source["design_seq"])

        if meta["filename_parse_ok"] != 1:
            raise ValueError(f"Unable to parse selected PDB filename: {pdb_path.name}")
        if str(meta["target_name"]).upper() != str(source["target_name"]).upper():
            raise ValueError(
                f"Target mismatch for {pdb_path.name}: "
                f"{meta['target_name']} vs {source['target_name']}"
            )
        if meta["design_seq_from_filename"] != design_seq:
            raise ValueError(
                f"Design sequence mismatch for {pdb_path.name}: "
                f"{meta['design_seq_from_filename']} vs {design_seq}"
            )
        if parsed_temp and not math.isclose(float(parsed_temp), SELECTED_TEMPERATURE):
            raise ValueError(
                f"Selected PDB is not from temperature 0.5: {pdb_path}"
            )

        row = {
            # A workspace-relative path works unchanged in both Windows and WSL.
            "pdb_path": (Path("selected_pdbs") / pdb_path.name).as_posix(),
            "pdb_file": pdb_path.name,
            "temperature_folder": temp_folder or str(source.get("temperature_folder", "")),
            "temperature": norm_temp(source["temperature"]),
            "target_name": str(source["target_name"]).upper(),
            "file_index": meta["file_index"],
            "design_seq_from_filename": design_seq,
            "design_natural_seq_from_filename": design_seq.upper(),
            "filename_parse_ok": 1,
            "matched_all_designs_count": 1,
            "matched_all_designs": 1,
            "matched_all_designs_row_indices": "",
            "is_best_design_in_af3_manifest": 1,
        }
        row.update(pdb_stats)
        row.update(scores)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    root = repository_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_csv", default=str(default_selection_path(root)))
    parser.add_argument("--run_dir", default=str(default_run_dir(root)))
    args = parser.parse_args()

    selection_csv = resolve_repo_path(root, args.selection_csv)
    run_dir = resolve_repo_path(root, args.run_dir)
    if not selection_csv.exists():
        raise FileNotFoundError(selection_csv)

    selected = validate_and_select(pd.read_csv(selection_csv), root)
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_out = run_dir / "temperature05_best17_selection.csv"
    selected.to_csv(selection_out, index=False, encoding="utf-8")

    workspace = run_dir / "workspace"
    archived = archive_existing_workspace(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    source_paper_dir = root / "paper_clean_v28"
    stage_paper_dir = workspace / "paper_clean_v28"
    shutil.copytree(source_paper_dir, stage_paper_dir, dirs_exist_ok=True)
    shutil.copy2(root / "17_complexes_native.jsonl", workspace / "17_complexes_native.jsonl")

    stage_outputs = workspace / "paper_clean_v28_outputs"
    stage_metrics = stage_outputs / "structure_metrics"
    stage_metrics.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(selected)
    manifest.to_csv(stage_outputs / "af3_manifest.csv", index=False, encoding="utf-8")
    manifest.to_csv(
        stage_outputs / "temperature05_best17_manifest.csv",
        index=False,
        encoding="utf-8",
    )

    highfold_parser = load_module(
        root / "paper_clean_v28" / "structure_metrics" / "01_extract_highfold_scores.py",
        "temperature05_highfold_parser",
    )
    highfold = build_selected_highfold_scores(
        selected,
        highfold_parser,
        workspace / "selected_pdbs",
    )
    highfold.to_csv(
        stage_metrics / "complex_highfold_scores.csv",
        index=False,
        encoding="utf-8",
    )

    patch_expected_rows(
        stage_paper_dir / "structure_metrics" / "07_compute_methylation_site_rmsd.py",
        [("EXPECTED_ROWS = 85", "EXPECTED_ROWS = 17")],
    )
    patch_expected_rows(
        stage_paper_dir / "structure_metrics" / "10_compute_pyrosetta_energy_naturalized.py",
        [("EXPECTED_ROWS = 85", "EXPECTED_ROWS = 17")],
    )

    config = {
        "repository_root": str(root),
        "source_selection_csv": str(selection_csv),
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "selected_temperature": SELECTED_TEMPERATURE,
        "expected_rows": EXPECTED_TARGETS,
        "all_designs_csv": str(
            root
            / "paper_clean_v28_outputs"
            / "generated_fasta_clean_auto_single"
            / "all_designs.csv"
        ),
        "permeability_dir": str(
            root
            / "raw_external"
            / "pdb_permeability_v20260624"
            / "permeability_complex"
        ),
        "archived_previous_workspace": str(archived) if archived else "",
    }
    (run_dir / "temperature05_best17_run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("===== TEMPERATURE 0.5 BEST17 PREPARATION COMPLETE =====")
    print(f"source best85 rows: {EXPECTED_SOURCE_ROWS}")
    print(f"selected temperature: {SELECTED_TEMPERATURE}")
    print(f"selected rows: {len(selected)}")
    print(f"unique targets: {selected['target_name'].nunique()}")
    print(f"selected PDBs verified: {selected['pdb_path'].nunique()}/{len(selected)}")
    print(f"selection table: {selection_out}")
    print(f"isolated workspace: {workspace}")
    if archived:
        print(f"previous workspace archived: {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
