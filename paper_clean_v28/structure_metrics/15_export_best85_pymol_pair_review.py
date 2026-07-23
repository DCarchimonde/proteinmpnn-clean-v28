#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export all 85 single-global-align predicted/native PyMOL review pairs.

For every row this exporter independently reproduces script 13:

1. load the complete predicted and native complexes;
2. call PyMOL ``align`` exactly once on all complex C-alpha atoms;
3. do not align or fit the cyclic peptide again;
4. take the final chain from each already aligned complex;
5. calculate complete final-chain C-alpha RMSD by residue order; and
6. verify both global and peptide RMSDs against the saved best85 CSV.

The saved session therefore opens directly in the scientifically relevant
whole-complex alignment frame.  The user should rotate/zoom the view, not click
another ``align`` command.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

try:
    from pymol import cmd
except ImportError as exc:  # pragma: no cover - run in the user's PyMOL
    raise SystemExit(
        "PyMOL's Python module is required. Run with, for example:\n"
        "  pymol -cq -r paper_clean_v28/structure_metrics/"
        "15_export_best85_pymol_pair_review.py"
    ) from exc


OBJECT_PREFIX = "best85_"
PREDICTED_OBJECT = "best85_predicted_complex"
NATIVE_OBJECT = "best85_native_complex"
PREDICTED_PEPTIDE = "best85_predicted_cyclic_peptide"
NATIVE_PEPTIDE = "best85_native_cyclic_peptide"
PEPTIDE_OVERLAY = "best85_cyclic_peptide_overlay"
ALIGNMENT_OBJECT = "best85_global_complex_ca_alignment"

ALIGN_KWARGS = {
    "cutoff": 2.0,
    "cycles": 0,
    "gap": -10.0,
    "extend": -0.5,
    "max_gap": 50,
    "matrix": "BLOSUM62",
    "mobile_state": 0,
    "target_state": 0,
    "quiet": 1,
    "max_skip": 0,
    "transform": 1,
    "reset": 0,
}


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
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def resolve_script_path() -> Path:
    value = globals().get("__script__") or globals().get("__file__")
    return Path(str(value)).resolve() if value else Path.cwd().resolve()


def load_support(path: Path):
    spec = importlib.util.spec_from_file_location("best85_review_support", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import support script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sanitize_component(value: object, limit: int = 40) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-") or "unknown"
    return text[:limit]


def normalized_temperature(value: object) -> str:
    number = safe_float(value)
    if number is None:
        return sanitize_component(value, 12)
    return f"{number:g}".replace("-", "m").replace(".", "p")


def pml_quote(path: Path) -> str:
    return '"' + path.resolve().as_posix().replace('"', '\\"') + '"'


def relative_text(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path.resolve())


def remove_review_objects() -> None:
    names = set(cmd.get_names("all"))
    try:
        names.update(cmd.get_names("selections"))
    except Exception:
        pass
    for name in sorted(names, reverse=True):
        if name.startswith(OBJECT_PREFIX):
            try:
                cmd.delete(name)
            except Exception:
                pass


def validate_chain(chain: object, label: str) -> str:
    value = str(chain or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9]", value):
        raise ValueError(f"{label} must be one alphanumeric PDB chain ID, got {value!r}")
    return value


def write_pair_pml(
    path: Path,
    predicted_raw: Path,
    native_raw: Path,
    predicted_chain: str,
    native_chain: str,
    group_name: str,
) -> None:
    lines = [
        f"delete {OBJECT_PREFIX}*",
        f"load {pml_quote(predicted_raw)}, {PREDICTED_OBJECT}",
        f"load {pml_quote(native_raw)}, {NATIVE_OBJECT}",
        f"sort {PREDICTED_OBJECT}",
        f"sort {NATIVE_OBJECT}",
        (
            f"align {PREDICTED_OBJECT} and name CA, "
            f"{NATIVE_OBJECT} and name CA, "
            "cutoff=2.0, cycles=0, gap=-10.0, extend=-0.5, max_gap=50, "
            f"object={ALIGNMENT_OBJECT}, matrix=BLOSUM62, mobile_state=0, "
            "target_state=0, quiet=0, max_skip=0, transform=1, reset=0"
        ),
        (
            f"select {PREDICTED_PEPTIDE}, "
            f"{PREDICTED_OBJECT} and chain {predicted_chain}"
        ),
        (
            f"select {NATIVE_PEPTIDE}, "
            f"{NATIVE_OBJECT} and chain {native_chain}"
        ),
        (
            f"select {PEPTIDE_OVERLAY}, "
            f"{PREDICTED_PEPTIDE} or {NATIVE_PEPTIDE}"
        ),
        f"hide everything, {PREDICTED_OBJECT} or {NATIVE_OBJECT}",
        (
            f"show cartoon, ({PREDICTED_OBJECT} and not {PREDICTED_PEPTIDE}) "
            f"or ({NATIVE_OBJECT} and not {NATIVE_PEPTIDE})"
        ),
        f"show sticks, {PEPTIDE_OVERLAY}",
        f"show spheres, {PEPTIDE_OVERLAY} and name CA",
        f"color cyan, {PREDICTED_OBJECT}",
        f"color gray70, {NATIVE_OBJECT}",
        f"color orange, {PREDICTED_PEPTIDE}",
        f"color magenta, {NATIVE_PEPTIDE}",
        (
            f"set cartoon_transparency, 0.65, "
            f"{PREDICTED_OBJECT} and not {PREDICTED_PEPTIDE}"
        ),
        (
            f"set cartoon_transparency, 0.78, "
            f"{NATIVE_OBJECT} and not {NATIVE_PEPTIDE}"
        ),
        f"set stick_radius, 0.18, {PEPTIDE_OVERLAY}",
        f"set sphere_scale, 0.25, {PEPTIDE_OVERLAY} and name CA",
        "set orthoscopic, on",
        "bg_color white",
        f"group {group_name}, {PREDICTED_OBJECT} {NATIVE_OBJECT} {ALIGNMENT_OBJECT}",
        f"orient {PEPTIDE_OVERLAY}",
        f"zoom {PEPTIDE_OVERLAY}, 5",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def style_pair(
    predicted_chain: str,
    native_chain: str,
    group_name: str,
) -> None:
    predicted_chain_selection = (
        f"{PREDICTED_OBJECT} and chain {predicted_chain}"
    )
    native_chain_selection = f"{NATIVE_OBJECT} and chain {native_chain}"
    cmd.select(PREDICTED_PEPTIDE, predicted_chain_selection)
    cmd.select(NATIVE_PEPTIDE, native_chain_selection)
    cmd.select(PEPTIDE_OVERLAY, f"{PREDICTED_PEPTIDE} or {NATIVE_PEPTIDE}")

    cmd.hide("everything", f"{PREDICTED_OBJECT} or {NATIVE_OBJECT}")
    cmd.show(
        "cartoon",
        (
            f"({PREDICTED_OBJECT} and not {PREDICTED_PEPTIDE}) or "
            f"({NATIVE_OBJECT} and not {NATIVE_PEPTIDE})"
        ),
    )
    cmd.show("sticks", PEPTIDE_OVERLAY)
    cmd.show("spheres", f"{PEPTIDE_OVERLAY} and name CA")
    cmd.color("cyan", PREDICTED_OBJECT)
    cmd.color("gray70", NATIVE_OBJECT)
    cmd.color("orange", PREDICTED_PEPTIDE)
    cmd.color("magenta", NATIVE_PEPTIDE)
    cmd.set(
        "cartoon_transparency",
        0.65,
        f"{PREDICTED_OBJECT} and not {PREDICTED_PEPTIDE}",
    )
    cmd.set(
        "cartoon_transparency",
        0.78,
        f"{NATIVE_OBJECT} and not {NATIVE_PEPTIDE}",
    )
    cmd.set("stick_radius", 0.18, PEPTIDE_OVERLAY)
    cmd.set("sphere_scale", 0.25, f"{PEPTIDE_OVERLAY} and name CA")
    cmd.set("orthoscopic", 1)
    cmd.bg_color("white")
    cmd.group(
        group_name,
        f"{PREDICTED_OBJECT} {NATIVE_OBJECT} {ALIGNMENT_OBJECT}",
    )
    cmd.orient(PEPTIDE_OVERLAY)
    cmd.zoom(PEPTIDE_OVERLAY, buffer=5)


def pair_info_lines(
    index: int,
    total: int,
    row: Mapping[str, object],
    predicted_chain: str,
    native_chain: str,
    actual_global_rmsd: float,
    global_aligned_pairs: int,
    actual_peptide_rmsd: float,
    complete_peptide_pairs: int,
    predicted_count: int,
    native_count: int,
) -> List[str]:
    return [
        f"BEST85 PAIR {index:03d}/{total:03d}",
        "",
        f"target: {row.get('target_name', '')}",
        f"temperature: {row.get('temperature', '')}",
        f"design sequence: {row.get('design_seq', '')}",
        f"predicted PDB: {row.get('pdb_file', '')}",
        f"predicted cyclic-peptide chain: {predicted_chain}",
        f"native cyclic-peptide chain: {native_chain}",
        "",
        f"reproduced global-complex PyMOL CA RMSD: {actual_global_rmsd:.6f}",
        f"global PyMOL-aligned CA pairs: {global_aligned_pairs}",
        (
            "complete cyclic-peptide CA RMSD after global-complex alignment: "
            f"{actual_peptide_rmsd:.6f}"
        ),
        f"complete cyclic-peptide CA pairs: {complete_peptide_pairs}",
        f"predicted/native cyclic-peptide CA counts: {predicted_count}/{native_count}",
        (
            "naturalized design-sequence match: "
            f"{row.get('decoded_design_seq_matches_design_naturalized', '')}"
        ),
        "",
        "Colour legend:",
        "  predicted receptor: cyan",
        "  native receptor: grey",
        "  predicted cyclic peptide: orange",
        "  native cyclic peptide: magenta",
        "",
        "The visible pair uses exactly one whole-complex PyMOL CA align.",
        "The orange peptide was moved only with its complete predicted complex.",
        "The peptide RMSD uses every final-chain CA by residue order.",
        "No peptide-only align, fit, pair_fit or superposition was performed.",
        "",
        "IMPORTANT: the session is already in the required aligned frame.",
        "Rotate and zoom it; do not click Align again.",
    ]


def export_pair(
    index: int,
    total: int,
    row: Mapping[str, object],
    repo_root: Path,
    review_root: Path,
    native_records: Mapping[str, Mapping[str, object]],
    support,
    tolerance: float,
) -> dict:
    global_status = str(row.get("global_complex_ca_rmsd_status", ""))
    peptide_status = str(row.get("cyclic_peptide_ca_rmsd_status", ""))
    if global_status != "ok" or peptide_status not in ("", "ok"):
        raise ValueError(
            "best85 row status is not OK: "
            f"global={global_status!r}, peptide={peptide_status!r}"
        )

    predicted_path = support.resolve_repo_path(repo_root, row.get("pdb_path", ""))
    if not predicted_path.is_file():
        raise FileNotFoundError(f"Predicted PDB not found: {predicted_path}")

    target = str(row.get("target_name", "")).upper()
    native_record = native_records.get(target)
    if native_record is None:
        raise KeyError(f"Native record not found for target: {target}")

    predicted_meta = support.parse_predicted_ca_metadata(predicted_path)
    predicted_sequences = predicted_meta.get("chain_sequences", {})
    if not predicted_sequences:
        raise ValueError(f"No CA-containing chains in predicted PDB: {predicted_path}")
    predicted_chain = validate_chain(
        list(predicted_sequences)[-1],
        "predicted final chain",
    )
    saved_predicted_chain = str(
        row.get("predicted_peptide_chain", "")
    ).strip()
    if saved_predicted_chain and predicted_chain != saved_predicted_chain:
        raise ValueError(
            "Final-chain quality gate failed: "
            f"PDB={predicted_chain}, CSV={saved_predicted_chain}"
        )

    native_sequences = support.native_sequences(native_record)
    if not native_sequences:
        raise ValueError(f"No native chains found for {target}")
    native_chain = validate_chain(
        list(native_sequences)[-1],
        "native final chain",
    )
    saved_native_chain = str(
        row.get("native_peptide_chain")
        or row.get("native_peptide_chain_used_by_complete_positional_rmsd")
        or ""
    ).strip()
    if saved_native_chain and native_chain != saved_native_chain:
        raise ValueError(
            "Native final-chain quality gate failed: "
            f"JSONL={native_chain}, CSV={saved_native_chain}"
        )

    folder_name = (
        f"{index:03d}_{sanitize_component(target, 12)}_"
        f"t{normalized_temperature(row.get('temperature'))}_"
        f"{sanitize_component(row.get('design_seq', ''), 28)}"
    )
    pair_dir = review_root / folder_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    predicted_raw = pair_dir / "01_predicted_complex_raw.pdb"
    native_raw = pair_dir / "02_native_complex_raw.pdb"
    predicted_aligned_complex = (
        pair_dir / "03_predicted_complex_after_global_align.pdb"
    )
    native_reference_complex = (
        pair_dir / "04_native_complex_reference.pdb"
    )
    predicted_aligned_peptide = (
        pair_dir / "05_predicted_cyclic_peptide_after_global_align.pdb"
    )
    native_reference_peptide = (
        pair_dir / "06_native_cyclic_peptide_reference.pdb"
    )
    pair_pml = pair_dir / "OPEN_PAIR_IN_PYMOL.pml"
    pair_session = pair_dir / "OPEN_PAIR_IN_PYMOL.pse"
    pair_info = pair_dir / "PAIR_INFO.txt"

    shutil.copy2(predicted_path, predicted_raw)
    native_pdb = support.native_record_to_pdbstr(native_record)
    native_raw.write_text(native_pdb, encoding="utf-8")

    remove_review_objects()
    cmd.load(str(predicted_raw), PREDICTED_OBJECT)
    cmd.load(str(native_raw), NATIVE_OBJECT)
    cmd.sort(PREDICTED_OBJECT)
    cmd.sort(NATIVE_OBJECT)
    predicted_selection = (
        f"{PREDICTED_OBJECT} and chain {predicted_chain} and name CA"
    )
    native_selection = f"{NATIVE_OBJECT} and chain {native_chain} and name CA"
    predicted_count = int(cmd.count_atoms(predicted_selection))
    native_count = int(cmd.count_atoms(native_selection))
    if predicted_count <= 0 or native_count <= 0:
        raise ValueError(
            f"Empty peptide CA selection: predicted={predicted_count}, "
            f"native={native_count}"
        )

    result = cmd.align(
        f"{PREDICTED_OBJECT} and name CA",
        f"{NATIVE_OBJECT} and name CA",
        object=ALIGNMENT_OBJECT,
        **ALIGN_KWARGS,
    )
    if len(result) != 7:
        raise RuntimeError(f"Unexpected PyMOL align result: {result!r}")
    actual_global_rmsd = float(result[0])
    global_aligned_pairs = int(result[1])
    actual_peptide_rmsd, complete_peptide_pairs = (
        support.complete_positional_ca_rmsd(
            predicted_selection,
            native_selection,
        )
    )
    if (
        complete_peptide_pairs != predicted_count
        or complete_peptide_pairs != native_count
    ):
        raise ValueError(
            "Complete final-chain CA pairing gate failed: "
            f"pairs={complete_peptide_pairs}, predicted={predicted_count}, "
            f"native={native_count}"
        )

    recorded_global_rmsd = safe_float(row.get("global_complex_ca_rmsd"))
    recorded_peptide_rmsd = safe_float(
        row.get("cyclic_peptide_ca_rmsd_after_global_complex_alignment")
    )
    recorded_global_pairs_value = row.get("n_global_aligned_ca_pairs")
    recorded_complete_pairs_value = row.get(
        "n_complete_positional_peptide_ca_pairs"
    )
    try:
        recorded_global_pairs = int(float(str(recorded_global_pairs_value)))
    except (TypeError, ValueError):
        recorded_global_pairs = None
    try:
        recorded_complete_pairs = int(
            float(str(recorded_complete_pairs_value))
        )
    except (TypeError, ValueError):
        recorded_complete_pairs = None
    if recorded_global_rmsd is None or recorded_peptide_rmsd is None:
        raise ValueError("Saved global or cyclic-peptide RMSD is missing")
    if abs(actual_global_rmsd - recorded_global_rmsd) > tolerance:
        raise ValueError(
            "Global RMSD reproduction gate failed: "
            f"observed={actual_global_rmsd:.6f}, "
            f"saved={recorded_global_rmsd:.6f}, "
            f"tolerance={tolerance:.6f}"
        )
    if abs(actual_peptide_rmsd - recorded_peptide_rmsd) > tolerance:
        raise ValueError(
            "Cyclic-peptide RMSD reproduction gate failed: "
            f"observed={actual_peptide_rmsd:.6f}, "
            f"saved={recorded_peptide_rmsd:.6f}, "
            f"tolerance={tolerance:.6f}"
        )
    if (
        recorded_global_pairs is None
        or global_aligned_pairs != recorded_global_pairs
    ):
        raise ValueError(
            "Global aligned-pair reproduction gate failed: "
            f"observed={global_aligned_pairs}, "
            f"saved={recorded_global_pairs_value!r}"
        )
    if (
        recorded_complete_pairs is None
        or complete_peptide_pairs != recorded_complete_pairs
    ):
        raise ValueError(
            "Complete peptide-pair reproduction gate failed: "
            f"observed={complete_peptide_pairs}, "
            f"saved={recorded_complete_pairs_value!r}"
        )
    if str(row.get("whole_complex_align_call_count", "")) != "1":
        raise ValueError(
            "Saved one-align quality gate failed: "
            f"{row.get('whole_complex_align_call_count')!r}"
        )
    if str(row.get("cyclic_peptide_second_fit_performed", "")) != "0":
        raise ValueError(
            "Saved no-second-fit quality gate failed: "
            f"{row.get('cyclic_peptide_second_fit_performed')!r}"
        )

    group_name = (
        f"best85_pair_{index:03d}_{sanitize_component(target, 12)}_"
        f"peptide_rmsd_{actual_peptide_rmsd:.3f}".replace(".", "p")
    )
    style_pair(predicted_chain, native_chain, group_name)
    cmd.save(str(predicted_aligned_complex), PREDICTED_OBJECT)
    cmd.save(str(native_reference_complex), NATIVE_OBJECT)
    cmd.save(
        str(predicted_aligned_peptide),
        f"{PREDICTED_OBJECT} and chain {predicted_chain}",
    )
    cmd.save(
        str(native_reference_peptide),
        f"{NATIVE_OBJECT} and chain {native_chain}",
    )
    cmd.save(str(pair_session))
    write_pair_pml(
        pair_pml,
        predicted_raw,
        native_raw,
        predicted_chain,
        native_chain,
        group_name,
    )
    pair_info.write_text(
        "\n".join(
            pair_info_lines(
                index,
                total,
                row,
                predicted_chain,
                native_chain,
                actual_global_rmsd,
                global_aligned_pairs,
                actual_peptide_rmsd,
                complete_peptide_pairs,
                predicted_count,
                native_count,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "review_index": index,
        "target_name": target,
        "temperature": row.get("temperature", ""),
        "design_seq": row.get("design_seq", ""),
        "pdb_file": row.get("pdb_file", ""),
        "predicted_cyclic_peptide_chain": predicted_chain,
        "native_cyclic_peptide_chain": native_chain,
        "global_complex_ca_rmsd_saved": fmt(recorded_global_rmsd),
        "global_complex_ca_rmsd_reproduced": fmt(actual_global_rmsd),
        "n_global_pymol_aligned_ca_pairs": global_aligned_pairs,
        "cyclic_peptide_ca_rmsd_after_global_align_saved": fmt(
            recorded_peptide_rmsd
        ),
        "cyclic_peptide_ca_rmsd_after_global_align_reproduced": fmt(
            actual_peptide_rmsd
        ),
        "n_complete_positional_peptide_ca_pairs": complete_peptide_pairs,
        "n_predicted_peptide_ca": predicted_count,
        "n_native_peptide_ca": native_count,
        "naturalized_design_sequence_matches": row.get(
            "decoded_design_seq_matches_design_naturalized",
            "",
        ),
        "whole_complex_align_calls": 1,
        "cyclic_peptide_second_fit_performed": 0,
        "global_pass_lt3": row.get(
            "passes_global_complex_ca_rmsd_lt_threshold",
            "",
        ),
        "peptide_pass_lt3": row.get(
            "passes_cyclic_peptide_ca_rmsd_lt_threshold",
            "",
        ),
        "joint_pass_lt3": row.get(
            "passes_joint_global_and_cyclic_peptide_lt_threshold",
            "",
        ),
        "pair_folder": relative_text(pair_dir, repo_root),
        "open_pymol_session": relative_text(pair_session, repo_root),
        "rerun_alignment_pml": relative_text(pair_pml, repo_root),
        "manual_visual_result": "",
        "manual_notes": "",
    }


def write_navigator(
    review_root: Path,
    manifest_rows: Sequence[Mapping[str, object]],
) -> None:
    pml_paths = [
        str((review_root / Path(str(row["pair_folder"])).name
             / "OPEN_PAIR_IN_PYMOL.pml").resolve())
        for row in manifest_rows
    ]
    labels = [
        (
            f"{int(row['review_index']):03d}/"
            f"{len(manifest_rows):03d} "
            f"{row['target_name']} t={row['temperature']} "
            f"global_RMSD={row['global_complex_ca_rmsd_reproduced']} "
            f"peptide_RMSD={row['cyclic_peptide_ca_rmsd_after_global_align_reproduced']} "
            f"peptide_pairs={row['n_complete_positional_peptide_ca_pairs']}/"
            f"{row['n_native_peptide_ca']}"
        )
        for row in manifest_rows
    ]
    navigator_path = review_root / "best85_pair_navigator.py"
    source = f'''# Auto-generated by 15_export_best85_pymol_pair_review.py
from pymol import cmd

PAIR_PMLS = {pml_paths!r}
PAIR_LABELS = {labels!r}
STATE = {{"index": 0}}


def best_load(index=1):
    value = int(index)
    if value < 1 or value > len(PAIR_PMLS):
        raise ValueError(f"best_load index must be 1..{{len(PAIR_PMLS)}}")
    STATE["index"] = value - 1
    cmd.do("@" + PAIR_PMLS[STATE["index"]])
    print("\\n[BEST85 REVIEW] " + PAIR_LABELS[STATE["index"]])
    print("Commands: best_next | best_prev | best_load, N | best_info")


def best_next():
    best_load((STATE["index"] + 1) % len(PAIR_PMLS) + 1)


def best_prev():
    best_load((STATE["index"] - 1) % len(PAIR_PMLS) + 1)


def best_info():
    print("[BEST85 REVIEW] " + PAIR_LABELS[STATE["index"]])
    print("Pair PML: " + PAIR_PMLS[STATE["index"]])
    print("One whole-complex align is already applied. Do NOT align the peptide.")


def best_show_complex():
    cmd.orient("best85_predicted_complex or best85_native_complex")
    cmd.zoom("best85_predicted_complex or best85_native_complex", buffer=4)


def best_show_peptide():
    cmd.orient("best85_cyclic_peptide_overlay")
    cmd.zoom("best85_cyclic_peptide_overlay", buffer=5)


cmd.extend("best_load", best_load)
cmd.extend("best_next", best_next)
cmd.extend("best_prev", best_prev)
cmd.extend("best_info", best_info)
cmd.extend("best_show_complex", best_show_complex)
cmd.extend("best_show_peptide", best_show_peptide)
best_load(1)
'''
    navigator_path.write_text(source, encoding="utf-8")
    master_pml = review_root / "OPEN_BEST85_REVIEW.pml"
    master_pml.write_text(
        f"run {navigator_path.resolve().as_posix()}\n",
        encoding="utf-8",
    )


def write_start_here(review_root: Path) -> None:
    lines = [
        "BEST85 PYMOL PAIR REVIEW",
        "",
        "Fastest option:",
        "  Open OPEN_BEST85_REVIEW.pml in the PyMOL GUI.",
        "",
        "Then use these commands in PyMOL:",
        "  best_next          -> next predicted/native pair",
        "  best_prev          -> previous pair",
        "  best_load, 37      -> jump to pair 37",
        "  best_info          -> print current target, RMSD and pair count",
        "  best_show_complex  -> zoom to both complete complexes",
        "  best_show_peptide  -> zoom to both final peptide chains",
        "",
        "You can also open any pair folder and double-click:",
        "  OPEN_PAIR_IN_PYMOL.pse",
        "",
        "Each review shows two molecular structures:",
        "  predicted receptor = cyan",
        "  native receptor = grey",
        "  predicted cyclic peptide = orange",
        "  native cyclic peptide = magenta",
        "",
        "Every view has already received one whole-complex CA align.",
        "The peptide chains have NOT been aligned a second time.",
        "Do not click Align. Rotate/zoom the already aligned structures instead.",
        "The manifest CSV includes blank manual_visual_result and manual_notes columns.",
    ]
    (review_root / "00_START_HERE.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    metric_dir = (
        repo_root
        / "paper_clean_v28_outputs/structure_metrics/"
        "global_and_cyclic_peptide_ca_rmsd"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Export best85 after one whole-complex align; never refit the peptide."
        )
    )
    parser.add_argument(
        "--best85_csv",
        default=str(metric_dir / "global_complex_ca_rmsd_best85.csv"),
    )
    parser.add_argument(
        "--native_jsonl",
        default=str(repo_root / "17_complexes_native.jsonl"),
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
        "--review_dir",
        default=str(metric_dir / "best85_pair_review"),
    )
    parser.add_argument("--expected_rows", type=int, default=85)
    parser.add_argument("--rmsd_tolerance", type=float, default=0.005)
    return parser


def main() -> None:
    script_path = resolve_script_path()
    repo_root = script_path.parents[2]
    args, _unknown = build_parser(repo_root).parse_known_args(sys.argv[1:])
    best85_path = Path(args.best85_csv).resolve()
    native_path = Path(args.native_jsonl).resolve()
    support_path = Path(args.support_script).resolve()
    review_root = Path(args.review_dir).resolve()

    for required in (best85_path, native_path, support_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")
    if args.expected_rows <= 0:
        raise ValueError("--expected_rows must be positive")
    if args.rmsd_tolerance < 0:
        raise ValueError("--rmsd_tolerance cannot be negative")

    support = load_support(support_path)
    rows = read_csv(best85_path)
    if len(rows) != args.expected_rows:
        raise RuntimeError(
            f"best85 count gate failed: {len(rows)} != {args.expected_rows}"
        )
    native_records: Dict[str, dict] = support.load_native_records(native_path)
    review_root.mkdir(parents=True, exist_ok=True)

    print("===== EXPORT BEST85 PYMOL PAIR REVIEW =====", flush=True)
    print("repository root:", repo_root, flush=True)
    print("best85 rows:", len(rows), flush=True)
    print("review directory:", review_root, flush=True)
    print(
        "metric: one whole-complex PyMOL align, then complete final-chain CA RMSD",
        flush=True,
    )

    manifest_rows = []
    failures = []
    for index, row in enumerate(rows, start=1):
        try:
            manifest_rows.append(
                export_pair(
                    index,
                    len(rows),
                    row,
                    repo_root,
                    review_root,
                    native_records,
                    support,
                    args.rmsd_tolerance,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "review_index": index,
                    "target_name": row.get("target_name", ""),
                    "pdb_file": row.get("pdb_file", ""),
                    "error": repr(exc),
                }
            )
        if index % 10 == 0 or index == len(rows):
            print(f"processed: {index}/{len(rows)}", flush=True)

    write_csv(review_root / "best85_manual_review_manifest.csv", manifest_rows)
    write_csv(review_root / "best85_pair_export_failures.csv", failures)
    if failures or len(manifest_rows) != args.expected_rows:
        raise RuntimeError(
            "Pair-review export quality gate failed: "
            f"exported={len(manifest_rows)}, failures={len(failures)}. "
            "See best85_pair_export_failures.csv"
        )

    write_navigator(review_root, manifest_rows)
    write_start_here(review_root)
    report = [
        "===== BEST85 PYMOL PAIR REVIEW EXPORT =====",
        "",
        f"expected rows: {args.expected_rows}",
        f"exported pair folders: {len(manifest_rows)}",
        f"export failures: {len(failures)}",
        "count gate: PASS",
        (
            "saved global + peptide RMSD and pair-count reproduction gate: "
            f"PASS ({len(manifest_rows)}/{len(manifest_rows)})"
        ),
        "one whole-complex align per pair: PASS",
        "peptide-only second fit: 0/85",
        "",
        "Each pair contains:",
        "  raw predicted complex PDB",
        "  reconstructed native complex PDB",
        "  complete predicted complex after the one global align",
        "  native complex reference",
        "  predicted/native final-chain peptide PDBs in that same frame",
        "  directly openable PyMOL PSE session",
        "  reproducible one-global-align PyMOL PML script",
        "  pair metric information text",
        "",
        "Master viewer: OPEN_BEST85_REVIEW.pml",
        "Manifest/checklist: best85_manual_review_manifest.csv",
    ]
    (review_root / "best85_pair_review_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    remove_review_objects()

    print("\n===== COMPLETE =====", flush=True)
    print(f"exported pairs: {len(manifest_rows)}/{args.expected_rows}", flush=True)
    print("global + peptide RMSD reproduction: PASS", flush=True)
    print("one whole-complex align and no peptide refit: PASS", flush=True)
    print("open in PyMOL:", review_root / "OPEN_BEST85_REVIEW.pml", flush=True)
    print(
        "manual checklist:",
        review_root / "best85_manual_review_manifest.csv",
        flush=True,
    )


if __name__ in {"__main__", "pymol"}:
    main()
