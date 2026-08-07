#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed-pose naturalized PyRosetta scoring for 151 monomer pairs.

Each sample is scored twice:

- reference_naturalized (PDB variant 2)
- e2e_naturalized (PDB variant 4)

The score function exactly follows the corrected complex workflow: ref2015 with
rama_prepro, omega, and p_aa_pp disabled.  Scores are descriptive Rosetta Energy
Units (REU) for fixed naturalized conformations; no relaxation, minimization, or
explicit N-methyl chemistry is included.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory, ScoreType


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_MONOMER_DIR = (
    ROOT
    / "paper_clean_v28_outputs"
    / "temperature_0.5_best17"
    / "monomer"
)
DEFAULT_STRUCTURE_CSV = DEFAULT_MONOMER_DIR / "monomer_structure_metrics_by_sample.csv"
DEFAULT_PDB_DIR = (
    ROOT
    / "raw_external"
    / "pdb_permeability_v20260624"
    / "pdb_monomer"
    / "pdb_monomer_hf4"
)
COMPLEX_ENERGY_HELPER = HERE / "10_compute_pyrosetta_energy_naturalized.py"
EXPECTED_SAMPLES = 151
EXPECTED_STRUCTURES = EXPECTED_SAMPLES * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score reference/e2e naturalized monomer PDBs with PyRosetta."
    )
    parser.add_argument("--structure_csv", default=str(DEFAULT_STRUCTURE_CSV))
    parser.add_argument("--pdb_dir", default=str(DEFAULT_PDB_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_MONOMER_DIR))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_energy_helper():
    spec = importlib.util.spec_from_file_location(
        "complex_naturalized_energy_helper", COMPLEX_ENERGY_HELPER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load energy helper: {COMPLEX_ENERGY_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def score_structure(
    *,
    sample_name: str,
    role: str,
    sequence: str,
    pdb_file: str,
    chain_id: str,
    pdb_dir: Path,
    scorefxn,
    helper,
) -> Dict[str, object]:
    started = time.perf_counter()
    source = (pdb_dir / pdb_file).resolve()
    result: Dict[str, object] = {
        "sample_name": sample_name,
        "structure_role": role,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "pdb_file": pdb_file,
        "pdb_path": str(source),
        "chain_id": chain_id,
        "energy_status": "failed",
        "error_stage": "precheck",
        "error_message": "",
    }
    try:
        if not source.is_file():
            raise FileNotFoundError(source)
        if not chain_id:
            raise ValueError("chain_id is empty")
        if not sequence or sequence != sequence.upper():
            raise ValueError(f"expected an uppercase naturalized sequence: {sequence!r}")

        with tempfile.TemporaryDirectory(prefix="monomer_pyrosetta_nat_") as tmp:
            canonical = Path(tmp) / source.name
            result["error_stage"] = "canonicalize_pdb"
            naturalization = helper.naturalize_peptide_pdb(
                source_path=source,
                output_path=canonical,
                peptide_chain=chain_id,
                design_seq=sequence,
            )
            for key, value in naturalization.items():
                result[f"naturalization_{key}"] = value

            result["error_stage"] = "pose_from_pdb"
            pose = pose_from_pdb(str(canonical))
            n_residues = int(pose.total_residue())
            if n_residues != len(sequence):
                raise ValueError(
                    f"loaded residue count mismatch: loaded={n_residues}, "
                    f"expected={len(sequence)}"
                )
            loaded_sequence = str(pose.sequence())
            if loaded_sequence != sequence:
                raise ValueError(
                    f"loaded sequence mismatch: loaded={loaded_sequence}, "
                    f"expected={sequence}"
                )

            result["error_stage"] = "score"
            total_score = float(scorefxn(pose))
            if not finite(total_score):
                raise ValueError(f"non-finite score: {total_score}")
            result.update(
                {
                    "n_residues_loaded": n_residues,
                    "sequence_loaded": loaded_sequence,
                    "sequence_matches_expected": 1,
                    "rosetta_total_score": total_score,
                    "rosetta_score_per_residue": total_score / n_residues,
                    "energy_status": "ok",
                    "error_stage": "",
                    "error_message": "",
                }
            )
    except Exception as exc:
        result["error_message"] = f"{type(exc).__name__}: {exc}"
        result["error_trace_tail"] = " | ".join(
            traceback.format_exc(limit=3).strip().splitlines()[-4:]
        )
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def pair_energy_rows(long_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for sample, group in long_frame.groupby("sample_name", sort=True):
        by_role = {
            str(row["structure_role"]): row
            for _, row in group.iterrows()
        }
        output: Dict[str, object] = {"sample_name": sample}
        for role in ("reference_naturalized", "e2e_naturalized"):
            item = by_role.get(role)
            prefix = "reference" if role.startswith("reference") else "e2e"
            if item is None:
                output[f"{prefix}_energy_status"] = "missing"
                continue
            for column in [
                "pdb_file",
                "sequence",
                "sequence_length",
                "n_residues_loaded",
                "sequence_matches_expected",
                "rosetta_total_score",
                "rosetta_score_per_residue",
                "energy_status",
                "error_stage",
                "error_message",
                "elapsed_seconds",
            ]:
                output[f"{prefix}_{column}"] = item.get(column)

        reference_total = output.get("reference_rosetta_total_score")
        e2e_total = output.get("e2e_rosetta_total_score")
        reference_per_residue = output.get("reference_rosetta_score_per_residue")
        e2e_per_residue = output.get("e2e_rosetta_score_per_residue")
        if all(
            finite(value)
            for value in [
                reference_total,
                e2e_total,
                reference_per_residue,
                e2e_per_residue,
            ]
        ):
            output["rosetta_total_score_delta_e2e_minus_reference"] = (
                float(e2e_total) - float(reference_total)
            )
            output["rosetta_score_per_residue_delta_e2e_minus_reference"] = (
                float(e2e_per_residue) - float(reference_per_residue)
            )
            output["e2e_lower_rosetta_total_score_than_reference"] = int(
                float(e2e_total) < float(reference_total)
            )
            output["e2e_lower_rosetta_score_per_residue_than_reference"] = int(
                float(e2e_per_residue) < float(reference_per_residue)
            )
            output["paired_energy_status"] = "ok"
        else:
            output["paired_energy_status"] = "failed"
        rows.append(output)
    return pd.DataFrame(rows)


def version_text() -> str:
    try:
        value = str(pyrosetta.version())
        return value.splitlines()[-1] if value else "not_reported"
    except Exception:
        return "not_reported"


def main() -> int:
    args = parse_args()
    structure_csv = Path(args.structure_csv).resolve()
    pdb_dir = Path(args.pdb_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not structure_csv.is_file():
        raise FileNotFoundError(structure_csv)
    if not pdb_dir.is_dir():
        raise FileNotFoundError(pdb_dir)

    source = pd.read_csv(structure_csv)
    required = {
        "sample_name",
        "reference_natural_sequence",
        "e2e_natural_sequence",
        "reference_naturalized_pdb_file",
        "e2e_naturalized_pdb_file",
        "reference_naturalized_chain",
        "e2e_naturalized_chain",
        "naturalized_structure_status",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"structure CSV missing columns: {missing}")
    if len(source) != EXPECTED_SAMPLES:
        raise ValueError(
            f"expected {EXPECTED_SAMPLES} monomer rows, observed {len(source)}"
        )
    if not source["naturalized_structure_status"].astype(str).eq("ok").all():
        failed = source.loc[
            ~source["naturalized_structure_status"].astype(str).eq("ok"),
            "sample_name",
        ].tolist()
        raise ValueError(f"input monomer structure gate is not complete: {failed}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        source = source.head(args.limit).copy()

    helper = load_energy_helper()
    init_options = (
        "-mute all -ignore_unrecognized_res -ignore_waters "
        "-load_PDB_components false -ex1 -ex2"
    )
    pyrosetta.init(init_options)
    scorefxn = ScoreFunctionFactory.create_score_function("ref2015")
    disabled_weights: Dict[str, float] = {}
    for name, score_type in [
        ("rama_prepro", ScoreType.rama_prepro),
        ("omega", ScoreType.omega),
        ("p_aa_pp", ScoreType.p_aa_pp),
    ]:
        disabled_weights[name] = float(scorefxn.get_weight(score_type))
        scorefxn.set_weight(score_type, 0.0)

    long_rows: List[Dict[str, object]] = []
    total = len(source) * 2
    counter = 0
    for _, row in source.sort_values("sample_name").iterrows():
        sample = str(row["sample_name"])
        jobs = [
            {
                "role": "reference_naturalized",
                "sequence": str(row["reference_natural_sequence"]),
                "pdb_file": str(row["reference_naturalized_pdb_file"]),
                "chain_id": str(row["reference_naturalized_chain"]),
            },
            {
                "role": "e2e_naturalized",
                "sequence": str(row["e2e_natural_sequence"]),
                "pdb_file": str(row["e2e_naturalized_pdb_file"]),
                "chain_id": str(row["e2e_naturalized_chain"]),
            },
        ]
        for job in jobs:
            counter += 1
            result = score_structure(
                sample_name=sample,
                role=job["role"],
                sequence=job["sequence"],
                pdb_file=job["pdb_file"],
                chain_id=job["chain_id"],
                pdb_dir=pdb_dir,
                scorefxn=scorefxn,
                helper=helper,
            )
            long_rows.append(result)
            print(
                f"[{counter:03d}/{total:03d}] {sample} {job['role']} -> "
                f"{result['energy_status']} "
                f"({result['elapsed_seconds']:.2f}s)",
                flush=True,
            )

    long_frame = pd.DataFrame(long_rows)
    paired = pair_energy_rows(long_frame)
    expected_samples = len(source) if args.limit is not None else EXPECTED_SAMPLES
    expected_structures = expected_samples * 2
    ok_long = long_frame["energy_status"].astype(str).eq("ok")
    ok_pairs = paired["paired_energy_status"].astype(str).eq("ok")
    quality_pass = (
        len(long_frame) == expected_structures
        and int(ok_long.sum()) == expected_structures
        and len(paired) == expected_samples
        and int(ok_pairs.sum()) == expected_samples
        and int(
            pd.to_numeric(
                long_frame["sequence_matches_expected"], errors="coerce"
            ).sum()
        )
        == expected_structures
    )

    suffix = "_smoke" if args.limit is not None else ""
    long_path = out_dir / f"monomer_pyrosetta_energy_by_structure{suffix}.csv"
    paired_path = out_dir / f"monomer_pyrosetta_energy_by_sample{suffix}.csv"
    problem_path = out_dir / f"monomer_pyrosetta_energy_problem_rows{suffix}.csv"
    report_path = out_dir / f"monomer_pyrosetta_energy_report{suffix}.txt"
    long_frame.to_csv(long_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    long_frame.loc[~ok_long].to_csv(
        problem_path, index=False, encoding="utf-8-sig"
    )

    delta = pd.to_numeric(
        paired["rosetta_score_per_residue_delta_e2e_minus_reference"],
        errors="coerce",
    )
    lower = pd.to_numeric(
        paired["e2e_lower_rosetta_score_per_residue_than_reference"],
        errors="coerce",
    )
    lines = [
        "===== MONOMER NATURALIZED FIXED-POSE PYROSETTA ENERGY REPORT =====",
        f"Run mode: {'SMOKE' if args.limit is not None else 'FULL 151-SAMPLE PANEL'}",
        f"Platform: {platform.platform()}",
        f"PyRosetta version: {version_text()}",
        f"Initialization options: {init_options}",
        "Score function: ref2015 with rama_prepro=0, omega=0, p_aa_pp=0",
        f"Original disabled weights: {disabled_weights}",
        f"Expected structures: {expected_structures}",
        f"Observed structures: {len(long_frame)}",
        f"Energy OK: {int(ok_long.sum())}/{len(long_frame)}",
        f"Expected sample pairs: {expected_samples}",
        f"Paired energy OK: {int(ok_pairs.sum())}/{len(paired)}",
        "",
        "===== PAIRED REFERENCE VS E2E RESULTS =====",
        (
            "Mean per-residue delta (e2e-reference, REU/residue): "
            f"{delta.mean():.6f}"
        ),
        (
            "Median per-residue delta (e2e-reference, REU/residue): "
            f"{delta.median():.6f}"
        ),
        (
            "E2E lower per-residue score than reference: "
            f"{int(lower.sum())}/{int(lower.notna().sum())}"
        ),
        "",
        "===== INTERPRETATION NOTES =====",
        "- Both structures are naturalized and scored in their fixed input conformations.",
        "- No FastRelax, minimization, repacking, or explicit N-methyl parameters are used.",
        "- Energies are REU, not kcal/mol.",
        "- The e2e-reference delta is descriptive and must not be presented as experimental folding free energy.",
        "- A monomer has no receptor/peptide interface, so cross-interface energy is not applicable.",
        "",
        f"QUALITY GATE: {'PASS' if quality_pass else 'FAIL'}",
        f"PROBLEMS: {int((~ok_long).sum())}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"monomer_pyrosetta_energy_manifest{suffix}.json").write_text(
        json.dumps(
            {
                "quality_gate": "PASS" if quality_pass else "FAIL",
                "samples": len(paired),
                "structures": len(long_frame),
                "energy_ok": int(ok_long.sum()),
                "paired_ok": int(ok_pairs.sum()),
                "long_csv": str(long_path),
                "paired_csv": str(paired_path),
                "report": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== MONOMER PYROSETTA ENERGY COMPLETE =====")
    print(f"structures OK: {int(ok_long.sum())}/{len(long_frame)}")
    print(f"paired samples OK: {int(ok_pairs.sum())}/{len(paired)}")
    print(f"quality gate: {'PASS' if quality_pass else 'FAIL'}")
    print(f"paired output: {paired_path}")
    if not quality_pass:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
