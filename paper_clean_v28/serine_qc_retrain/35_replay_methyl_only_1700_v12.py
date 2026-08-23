#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent batch-one V11 methylation replay for the exact 17 x 100 set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
SEARCH_PATH = SCRIPT_PATH.with_name("33_recover_3zgc_methyl_only_v12.py")
V8_SEARCH_PATH = SCRIPT_PATH.with_name("14_directed_recovery_search_v8.py")
REANNOTATOR_PATH = SCRIPT_PATH.with_name("10_reannotate_v6_pool_serine_only_v7.py")
GENERATOR_PATH = (
    REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
)
V11_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "cyclic_native_v11_1700_monomer"
DEFAULT_SELECTION = V11_ROOT / "v12_methyl_only" / "selection_17x100"
DEFAULT_MODEL = V11_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
DEFAULT_AUDIT = V11_ROOT / "representation_audit" / "cyclic_representation_audit.json"
DEFAULT_V11_PLAN = SCRIPT_PATH.with_name(
    "target_plan_v11_cyclic_native_rmsd_priority_1700.json"
)
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_BEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "best_designs.csv"
)
DEFAULT_OUT = V11_ROOT / "v12_methyl_only" / "final_independent_replay_handoff"

TARGETS = (
    "1SFI", "3AV9", "3AVA", "3AVB", "3AVF", "3AVG", "3AVH",
    "3AVI", "3AVJ", "3AVK", "3AVM", "3AVN", "3P8F", "3WNE",
    "3ZGC", "4K1E", "4KEL",
)
QUOTA = 100
EXPECTED_ROWS = len(TARGETS) * QUOTA
THRESHOLD = 0.6
TEMPERATURE = 0.5
ATOL = 2e-6
SELECTOR_PROTOCOL = "v12_methylation_only_exact_17_x_100_selector_v1"
REPLAY_PROTOCOL = "v12_methylation_only_exact_1700_batch_one_replay_v1"

DETAIL = "1700_详细审计.csv"
CONCISE = "1700_给尚哥_极简.csv"
FASTA = "1700_给尚哥_结构输入.fasta"
SELECTOR_MANIFEST = "v12_1700_methyl_only_release_manifest.json"
REPLAY_CSV = "1700_独立逐条甲基化复算.csv"
REPLAY_MANIFEST = "v12_1700_methyl_only_independent_replay_manifest.json"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = union_fields(rows) or ["status"]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def parse_vector(value: Any, field: str, length: int) -> List[float]:
    try:
        parsed = value if isinstance(value, list) else json.loads(str(value))
        vector = [float(item) for item in parsed]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field}_malformed") from exc
    if len(vector) != length:
        raise ValueError(f"{field}_length_mismatch")
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{field}_nonfinite")
    return vector


def compare_vectors(
    field: str,
    persisted: Any,
    replayed: Any,
    length: int,
) -> tuple[List[str], float]:
    try:
        left = parse_vector(persisted, field, length)
        right = parse_vector(replayed, f"replay_{field}", length)
    except ValueError as exc:
        return [str(exc)], float("inf")
    delta = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
    return ([f"{field}_mismatch"] if delta > ATOL else []), delta


def exact_quota(rows: Sequence[Mapping[str, Any]]) -> bool:
    counts = Counter(str(row.get("target_name", "")).upper() for row in rows)
    return (
        len(rows) == EXPECTED_ROWS
        and set(counts) == set(TARGETS)
        and all(counts[target] == QUOTA for target in TARGETS)
    )


def validate_selector(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    detail_path: Path,
    concise_path: Path,
    fasta_path: Path,
    model_path: Path,
    audit_path: Path,
) -> Dict[str, bool]:
    checks = dict(manifest.get("quality_checks") or {})
    artifacts = dict(manifest.get("artifacts") or {})
    inputs = dict(manifest.get("inputs") or {})
    return {
        "selector_is_authorized_exact_1700": (
            manifest.get("quality_gate") == "PASS"
            and manifest.get("release_status")
            == "AUTHORIZED_EXACT_17_X_100_METHYLATION_ONLY_PRESTRUCTURE_HANDOFF"
            and manifest.get("protocol") == SELECTOR_PROTOCOL
            and int(manifest.get("selected_rows", -1)) == EXPECTED_ROWS
            and int(manifest.get("quota_per_target", -1)) == QUOTA
            and checks
            and all(value is True for value in checks.values())
        ),
        "selector_detail_hash_matches": (
            artifacts.get("detailed", {}).get("sha256") == sha256_file(detail_path)
        ),
        "selector_concise_hash_matches": (
            artifacts.get("shangge_concise", {}).get("sha256")
            == sha256_file(concise_path)
        ),
        "selector_fasta_hash_matches": (
            artifacts.get("shangge_fasta", {}).get("sha256") == sha256_file(fasta_path)
        ),
        "selector_model_hash_matches": (
            inputs.get("model", {}).get("sha256") == sha256_file(model_path)
        ),
        "selector_audit_hash_matches": (
            inputs.get("representation_audit", {}).get("sha256")
            == sha256_file(audit_path)
        ),
        "selector_manifest_is_distinct_input": manifest_path.parent != detail_path,
    }


def run(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("V12 independent replay requires PyTorch") from exc

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    selection_dir = Path(args.selection_dir).resolve()
    detail_path = selection_dir / DETAIL
    concise_path = selection_dir / CONCISE
    fasta_path = selection_dir / FASTA
    selector_manifest_path = selection_dir / SELECTOR_MANIFEST
    model_path = Path(args.model).resolve()
    audit_path = Path(args.representation_audit).resolve()
    v11_plan_path = Path(args.v11_plan).resolve()
    native_path = Path(args.native_jsonl).resolve()
    best_path = Path(args.best_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    required = (
        detail_path, concise_path, fasta_path, selector_manifest_path, model_path,
        audit_path, v11_plan_path, native_path, best_path, SEARCH_PATH,
        V8_SEARCH_PATH, REANNOTATOR_PATH, GENERATOR_PATH,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if out_dir == selection_dir or out_dir in {path.parent for path in required}:
        raise ValueError("Replay output must be separate from source artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    selector = read_json(selector_manifest_path)
    detailed = read_csv(detail_path)
    upstream_checks = validate_selector(
        selector,
        selector_manifest_path,
        detail_path,
        concise_path,
        fasta_path,
        model_path,
        audit_path,
    )
    upstream_checks["selector_rows_have_exact_target_quotas"] = exact_quota(detailed)
    upstream_checks["selected_ids_are_unique"] = (
        len({row.get("final_release_id", "") for row in detailed}) == EXPECTED_ROWS
        and len({row.get("candidate_id", "") for row in detailed}) == EXPECTED_ROWS
    )
    if not all(upstream_checks.values()):
        failed = [name for name, passed in upstream_checks.items() if not passed]
        raise RuntimeError("V12 replay upstream contract failed: " + ", ".join(failed))

    search = load_module("v12_replay_search", SEARCH_PATH)
    v8 = load_module("v12_replay_v8", V8_SEARCH_PATH)
    reannotator = load_module("v12_replay_reannotator", REANNOTATOR_PATH)
    generator = load_module("v12_replay_generator", GENERATOR_PATH)
    search.validate_v11_contract(torch, model_path, audit_path, v11_plan_path)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU replay requires --allow-cpu")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise RuntimeError("No CUDA device; pass --allow-cpu only for a deliberate slow run")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from paper_clean_v28.clean_v28_common import (  # pylint: disable=import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        NAT_TO_METHYL_ABS,
        NATURAL_AA_ALPHABET,
        cyclic_representation_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
    )

    best_rows = generator.read_csv(best_path)
    selected_chains = generator.selected_chain_index(best_rows)
    native_rows = generator.read_jsonl(native_path)
    native_index = {
        generator.record_name(row, index): row for index, row in enumerate(native_rows)
    }
    if set(TARGETS) - set(selected_chains) or set(TARGETS) - set(native_index):
        raise RuntimeError("V12 replay lacks one or more frozen target geometries")
    model = load_v28_model(str(model_path), device)
    model.eval()
    common = {
        "EXTENDED_AA_ALPHABET": EXTENDED_AA_ALPHABET,
        "NAT_TO_METHYL_ABS": NAT_TO_METHYL_ABS,
        "NATURAL_AA_ALPHABET": NATURAL_AA_ALPHABET,
        "cyclic_representation_known_sequence_methyl_probabilities": (
            cyclic_representation_known_sequence_methyl_probabilities
        ),
        "featurize_records": featurize_records,
    }
    replay_scorer = search.V11ReleaseFloorScorer(
        v8,
        model,
        device,
        native_index,
        selected_chains,
        1,
        torch,
        common,
        reannotator,
    )

    vector_fields = (
        "methyl_probabilities",
        "methyl_probability_representation_min",
        "methyl_probability_representation_max",
        "methyl_probability_representation_span",
        "methyl_probability_representation_std",
    )
    replay_rows: List[Dict[str, Any]] = []
    print("===== V12 INDEPENDENT BATCH-ONE METHYL REPLAY START =====", flush=True)
    for index, source in enumerate(detailed, start=1):
        target = str(source["target_name"]).upper()
        sequence = str(source["design_natural_seq"]).upper()
        payload = replay_scorer.score_full(
            target,
            [sequence],
            stage="independent final replay",
            show_progress=False,
        )[sequence]
        errors: List[str] = []
        maxima: Dict[str, float] = {}
        for field in vector_fields:
            field_errors, delta = compare_vectors(
                field,
                source.get(field, ""),
                payload.get(field, ""),
                len(sequence),
            )
            errors.extend(field_errors)
            maxima[field] = delta
        if str(source.get("design_seq", "")) != str(payload.get("design_seq", "")):
            errors.append("design_seq_mismatch")
        try:
            if int(source.get("representation_threshold_disagreement_count", -1)) != int(
                payload.get("representation_threshold_disagreement_count", -2)
            ):
                errors.append("threshold_disagreement_count_mismatch")
        except (TypeError, ValueError):
            errors.append("threshold_disagreement_count_malformed")
        if not v8.stable_cyclic_methyl_release_gate(payload, sequence):
            errors.append("replayed_strict_methylation_gate_failed")
        replay_rows.append(
            {
                "final_release_id": source["final_release_id"],
                "candidate_id": source["candidate_id"],
                "target_name": target,
                "design_natural_seq": sequence,
                "persisted_design_seq": source["design_seq"],
                "replayed_design_seq": payload["design_seq"],
                "replayed_methyl_positions_1based": payload["methyl_positions_1based"],
                "replayed_release_floor_maximum_probability": search.actionable_max(
                    sequence,
                    json.loads(str(payload["methyl_probability_representation_min"])),
                )[0],
                "maximum_probability_absolute_difference": max(maxima.values(), default=0.0),
                "vector_maximum_absolute_differences": json.dumps(maxima, sort_keys=True),
                "row_replay_status": "PASS" if not errors else "FAIL",
                "row_replay_errors": json.dumps(errors, ensure_ascii=False),
                "prestructure_base_score_replayed": 0,
                "prestructure_rmsd_replayed": 0,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(detailed):
            failures = sum(row["row_replay_status"] != "PASS" for row in replay_rows)
            print(f"Replayed {index}/{len(detailed)}; failures={failures}", flush=True)

    failures = [row for row in replay_rows if row["row_replay_status"] != "PASS"]
    replay_checks = {
        **upstream_checks,
        "every_selected_row_was_replayed_at_batch_size_one": len(replay_rows) == EXPECTED_ROWS,
        "every_batch_one_methylation_replay_passed": not failures,
        "replayed_target_quota_remains_exact_17_x_100": exact_quota(replay_rows),
        "no_base_score_was_computed_or_used": all(
            int(row["prestructure_base_score_replayed"]) == 0 for row in replay_rows
        ),
        "no_rmsd_was_computed_or_used_before_structures": all(
            int(row["prestructure_rmsd_replayed"]) == 0 for row in replay_rows
        ),
    }
    quality_gate = "PASS" if all(replay_checks.values()) else "FAIL"
    atomic_write_csv(out_dir / REPLAY_CSV, replay_rows)
    if quality_gate == "PASS":
        atomic_copy(detail_path, out_dir / DETAIL)
        atomic_copy(concise_path, out_dir / CONCISE)
        atomic_copy(fasta_path, out_dir / FASTA)

    artifacts: Dict[str, Any] = {
        "replay_csv": {
            "path": str(out_dir / REPLAY_CSV),
            "sha256": sha256_file(out_dir / REPLAY_CSV),
        }
    }
    if quality_gate == "PASS":
        for key, name in (("detailed", DETAIL), ("shangge_concise", CONCISE), ("shangge_fasta", FASTA)):
            artifacts[key] = {
                "path": str(out_dir / name),
                "sha256": sha256_file(out_dir / name),
            }
    manifest = {
        "quality_gate": quality_gate,
        "release_status": (
            "AUTHORIZED_AFTER_EXACT_1700_BATCH_ONE_METHYLATION_REPLAY"
            if quality_gate == "PASS"
            else "BLOCKED_DO_NOT_SEND_TO_SHANGGE"
        ),
        "protocol": REPLAY_PROTOCOL,
        "batch_size": 1,
        "input_rows": len(detailed),
        "replayed_rows": len(replay_rows),
        "failed_rows": len(failures),
        "temperature": TEMPERATURE,
        "threshold": THRESHOLD,
        "probability_absolute_tolerance": ATOL,
        "prestructure_base_score_policy": "NOT_COMPUTED",
        "prestructure_rmsd_policy": "WAIT_FOR_SHANGGE_STRUCTURES",
        "quality_checks": replay_checks,
        "failed_examples_first_100": failures[:100],
        "inputs": {
            "selector_manifest": {"path": str(selector_manifest_path), "sha256": sha256_file(selector_manifest_path)},
            "selector_detail": {"path": str(detail_path), "sha256": sha256_file(detail_path)},
            "selector_concise": {"path": str(concise_path), "sha256": sha256_file(concise_path)},
            "selector_fasta": {"path": str(fasta_path), "sha256": sha256_file(fasta_path)},
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "representation_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
            "v11_plan": {"path": str(v11_plan_path), "sha256": sha256_file(v11_plan_path)},
            "native_jsonl": {"path": str(native_path), "sha256": sha256_file(native_path)},
            "best_csv": {"path": str(best_path), "sha256": sha256_file(best_path)},
        },
        "artifacts": artifacts,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        },
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
    }
    atomic_write_json(out_dir / REPLAY_MANIFEST, manifest)
    print("===== V12 INDEPENDENT METHYL-ONLY REPLAY COMPLETE =====", flush=True)
    print(f"Rows: {len(replay_rows)} (17 x 100)", flush=True)
    print(f"Failures: {len(failures)}", flush=True)
    print("Base score: NOT COMPUTED", flush=True)
    print("RMSD: WAITING FOR RETURNED STRUCTURES", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        raise RuntimeError(
            f"V12 independent replay blocked the handoff; failed rows={len(failures)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", default=str(DEFAULT_SELECTION))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--representation-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--v11-plan", default=str(DEFAULT_V11_PLAN))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--best-csv", default=str(DEFAULT_BEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
