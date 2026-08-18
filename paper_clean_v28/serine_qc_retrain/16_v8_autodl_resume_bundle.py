#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export/import a hash-pinned V8 round-6 resume bundle for AutoDL.

The source Windows run must have completed the frozen six-round search.  This
tool proves that its checkpoint configuration digest belongs to the recognized
53ce92e source program, replays the exact deterministic search provenance
without another model call, and archives only the artifacts required to finish
on a destination GPU.  Import relocates path metadata while recording every
changed manifest hash.  The destination finalizer still re-scores every search
ledger row and all released candidates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
SEARCH_PATH = SCRIPT_PATH.with_name("14_directed_recovery_search_v8.py")
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"
MODEL_DIR = V8_ROOT / "model"
REPRESENTATION_DIR = V8_ROOT / "representation_audit"
BASELINE_DIR = V8_ROOT / "generation_baseline"
SEARCH_DIR = V8_ROOT / "directed_search"
PORTABLE_DIR = V8_ROOT / "portable_provenance"
PORTABLE_MANIFEST_REL = (
    "paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/"
    "portable_provenance/autodl_portable_resume_manifest.json"
)
IMPORT_MANIFEST = V8_ROOT / "autodl_import_manifest.json"
DEFAULT_BUNDLE = REPO_ROOT / "v8_autodl_resume_bundle.zip"
DEFAULT_REVIEW_BUNDLE = V8_ROOT / "serine_qc_source_scoped_hybrid_v8_review_bundle.zip"

SOURCE_COMMIT = "53ce92e5238d717fc982357b4c58f65538a8f710"
SOURCE_SEARCH_SHA256_LF = (
    "d0d3536a51ac92caabc1523e8b7418811ac71b4abf3588485055223408ea7097"
)
SOURCE_SEARCH_SHA256_CRLF = (
    "2bce6d3cb017cdacf62c130810616d08b80d73d0fc9f2dc4122c5be2aeb60a96"
)
SOURCE_SEARCH_SHA256_ALLOWED = {
    SOURCE_SEARCH_SHA256_LF,
    SOURCE_SEARCH_SHA256_CRLF,
}
SOURCE_SEARCH_REL = "paper_clean_v28/serine_qc_retrain/14_directed_recovery_search_v8.py"
EXPORT_PROTOCOL = "v8_autodl_portable_resume_export_v1"
IMPORT_PROTOCOL = "v8_autodl_portable_resume_import_v1"
RESCORE_TOLERANCE = 2e-6

V6_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_cyclic_representation_v6"
V7_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_serine_only_cyclic_v7"
V3_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_order_balanced_v3"
V6_CHECKPOINT = V6_ROOT / "model" / "frankenstein_v28_expert_heads_qc.pt"
V6_MANIFEST = V6_ROOT / "model" / "expert_heads_retrain_manifest.json"
V7_CHECKPOINT = V7_ROOT / "model" / "frankenstein_v28_serine_only_qc.pt"
V7_MANIFEST = V7_ROOT / "model" / "expert_heads_retrain_manifest.json"
TEST_JSONL = V3_ROOT / "data" / "test_serine_provenance_corrected.jsonl"

MODEL_MANIFEST = MODEL_DIR / "expert_source_composition_manifest.json"
MODEL_CHECKPOINT = MODEL_DIR / "frankenstein_v28_source_scoped_hybrid_v8.pt"
REPRESENTATION_MANIFEST = REPRESENTATION_DIR / "cyclic_representation_audit.json"
BASELINE_MANIFEST = BASELINE_DIR / "generation_manifest.json"
MANIFEST_REBASE_ORDER = (
    MODEL_MANIFEST,
    REPRESENTATION_MANIFEST,
    BASELINE_MANIFEST,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_search_module() -> Any:
    spec = importlib.util.spec_from_file_location("v8_autodl_bundle_search", SEARCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SEARCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_source_blob() -> Tuple[bytes, str]:
    completed = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{SOURCE_SEARCH_REL}"],
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Cannot read the recognized legacy search source from git: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    if sha256_bytes(completed.stdout) != SOURCE_SEARCH_SHA256_LF:
        raise RuntimeError("Recognized legacy source commit has an unexpected file hash")
    current_checkout = SEARCH_PATH.read_bytes()
    legacy_blob = (
        completed.stdout.replace(b"\n", b"\r\n")
        if b"\r\n" in current_checkout
        else completed.stdout
    )
    source_sha256 = sha256_bytes(legacy_blob)
    if source_sha256 not in SOURCE_SEARCH_SHA256_ALLOWED:
        raise RuntimeError("Legacy source line-ending normalization is unrecognized")
    return legacy_blob, source_sha256


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required resume files: " + ", ".join(missing))


def legacy_search_config(
    search: Any, source_search_sha256: str
) -> Tuple[Dict[str, Any], str]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Export requires NumPy and PyTorch from the source run") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Export must run in the same CUDA environment as the source search")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = MODEL_CHECKPOINT
    model_manifest_path = MODEL_MANIFEST
    representation_path = REPRESENTATION_MANIFEST
    plan_path = search.DEFAULT_PLAN
    native_path = search.DEFAULT_NATIVE
    historical_path = search.DEFAULT_HISTORICAL
    prior_path = search.DEFAULT_PRIOR
    require_files(
        (
            model_path,
            model_manifest_path,
            representation_path,
            BASELINE_MANIFEST,
            BASELINE_DIR / "all_candidates.csv",
            BASELINE_DIR / "unique_candidates.csv",
            BASELINE_DIR / "methylated_new_candidates.csv",
            plan_path,
            native_path,
            historical_path,
            prior_path,
            search.REANNOTATOR_PATH,
            search.GENERATOR_PATH,
            search.COMMON_PATH,
            search.MODEL_UTILS_PATH,
            search.NMETHYL_CONFIG_PATH,
        )
    )
    missing_targets = sorted(
        str(value).upper()
        for value in read_json(BASELINE_MANIFEST).get(
            "targets_without_signature_candidate", []
        )
    )
    input_hashes = {
        "model": sha256_file(model_path),
        "model_manifest": sha256_file(model_manifest_path),
        "representation_audit": sha256_file(representation_path),
        "baseline_manifest": sha256_file(BASELINE_MANIFEST),
        "baseline_all": sha256_file(BASELINE_DIR / "all_candidates.csv"),
        "baseline_unique": sha256_file(BASELINE_DIR / "unique_candidates.csv"),
        "baseline_eligible": sha256_file(
            BASELINE_DIR / "methylated_new_candidates.csv"
        ),
        "plan": sha256_file(plan_path),
        "native": sha256_file(native_path),
        "historical": sha256_file(historical_path),
        "prior": sha256_file(prior_path),
        "search_program": source_search_sha256,
        "reannotator_program": sha256_file(search.REANNOTATOR_PATH),
        "generator_program": sha256_file(search.GENERATOR_PATH),
        "common_program": sha256_file(search.COMMON_PATH),
        "model_utils_program": sha256_file(search.MODEL_UTILS_PATH),
        "nmethyl_config_program": sha256_file(search.NMETHYL_CONFIG_PATH),
    }
    config = {
        "protocol": search.V8_SEARCH_PROTOCOL,
        "input_hashes": input_hashes,
        "missing_targets": missing_targets,
        "temperature": search.TEMPERATURE,
        "threshold": search.THRESHOLD,
        "strict_operator": ">",
        "alphabet": search.NATURAL_AA,
        "3wne_radius": 2,
        "3zgc_rounds": 6,
        "3zgc_beam_width": 512,
        "3zgc_offspring_per_round": 4096,
        "methyl_batch_size": 64,
        "base_plausibility_batch_size": 32,
        "maximum_released_candidates_per_target": 200,
        "requested_device": "cuda",
        "allow_cpu": False,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "numpy_version": str(np.__version__),
        "rng": "numpy.PCG64 SeedSequence([20260817, round_index])",
        "base_plausibility_percentile": search.BASE_PERCENTILE,
        "batch_one_rescore_tolerance": search.RESCORE_TOLERANCE,
        "probability_persistence_decimal_places": 8,
        "probability_rounding_implementation": "Python round(value, 8)",
        "full_budget_no_early_stop": True,
    }
    return config, stable_json_sha256(config)


def validate_static_search_evidence(
    search: Any, source_config_sha256: str
) -> Dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Static evidence validation requires NumPy") from exc
    baseline_unique = search.read_csv(BASELINE_DIR / "unique_candidates.csv")
    missing_targets = {
        str(value).upper()
        for value in read_json(BASELINE_MANIFEST).get(
            "targets_without_signature_candidate", []
        )
    }
    search.portable_resume_expected_evidence_names(missing_targets)

    wne_rows: List[Dict[str, Any]] = []
    wne_by_sequence: Dict[str, Dict[str, Any]] = {}
    if "3WNE" in missing_targets:
        wne_path = SEARCH_DIR / "3wne_exact_search_all.csv.gz"
        wne_rows = [
            search.normalize_search_ledger_row(row)
            for row in search.read_gzip_csv(wne_path)
        ]
        ranked_wne = search.top_ranked_sequences(baseline_unique, "3WNE")
        wne_anchors = [
            (ranked_wne[0], "current_v8_baseline_top"),
            (
                search.HISTORICAL_CONTROLS["3WNE"]["sequence"],
                "withdrawn_historical_control",
            ),
            (search.NATIVE_CONTROLS["3WNE"], "native_control"),
        ]
        expected_wne = search.wne_search_provenance(wne_anchors, 2)
        wne_by_sequence = {str(row["sequence"]): row for row in wne_rows}
        if len(wne_rows) != len(wne_by_sequence) or set(wne_by_sequence) != set(
            expected_wne
        ):
            raise RuntimeError(
                "3WNE portable ledger is not the exact radius-2 budget"
            )
        for sequence, row in wne_by_sequence.items():
            search.validate_search_ledger_row(
                row, "3WNE", sequence, "exact_radius_2", expected_wne[sequence]
            )

    ranked_zgc = search.top_ranked_sequences(baseline_unique, "3ZGC")
    initial = [
        search.HISTORICAL_CONTROLS["3ZGC"]["sequence"],
        search.NATIVE_CONTROLS["3ZGC"],
        ranked_zgc[0],
    ]
    anchors = search.select_diverse_sequences(ranked_zgc[:128], 34, initial=initial)
    provenance = search.zgc_initial_anchor_provenance(anchors, ranked_zgc[0])
    checkpoints = [
        SEARCH_DIR / "checkpoints" / f"3zgc_round_{index:02d}.json.gz"
        for index in range(1, 7)
    ]
    require_files(checkpoints)
    (
        completed_round,
        seen,
        _beam,
        qualified,
        trace,
    ) = search.reconstruct_and_validate_zgc_resume(
        SEARCH_DIR,
        checkpoints,
        source_config_sha256,
        512,
        provenance,
        4096,
        np,
        None,
        validate_model_scores=False,
    )
    if completed_round != 6:
        raise RuntimeError("Portable evidence has not completed round 6")
    return {
        "missing_targets": sorted(missing_targets),
        "wne_evaluated_sequences": len(wne_by_sequence),
        "wne_strict_hits": sum(
            int(row["passes_strict_probability"]) for row in wne_rows
        ),
        "zgc_evaluated_sequences": len(seen),
        "zgc_strict_hits": len(qualified),
        "zgc_completed_round": completed_round,
        "trace_stages": len(trace),
    }


def iter_json_paths(value: Any, path: Tuple[Any, ...] = ()) -> Iterable[Tuple[Tuple[Any, ...], str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_json_paths(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_json_paths(child, (*path, index))
    elif isinstance(value, str) and value:
        yield path, value


def manifest_relocations(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    rows: List[Dict[str, Any]] = []
    for pointer, value in iter_json_paths(payload):
        candidate = Path(value)
        try:
            resolved = candidate.resolve()
            target_relative = resolved.relative_to(REPO_ROOT.resolve())
        except (OSError, ValueError):
            continue
        if not resolved.exists():
            continue
        rows.append(
            {
                "json_pointer": list(pointer),
                "source_value": value,
                "repo_relative_path": target_relative.as_posix(),
                "target_kind": "file" if resolved.is_file() else "directory",
            }
        )
    return rows


def source_inventory(search: Any, missing_targets: Iterable[str]) -> List[Path]:
    missing = {str(value).upper() for value in missing_targets}
    required = [
        V6_CHECKPOINT,
        V6_MANIFEST,
        V7_CHECKPOINT,
        V7_MANIFEST,
        TEST_JSONL,
        # This prior-handoff CSV is a generated, gitignored novelty input.  It
        # must travel with the portable bundle; the other runtime data inputs
        # are part of the pinned repository checkout on the destination.
        search.DEFAULT_PRIOR,
        SEARCH_DIR / "mandatory_length_6_7_controls.csv",
        SEARCH_DIR / "search_trace_by_round.csv",
        SEARCH_DIR / "3zgc_round_00_initial.csv.gz",
    ]
    if "3WNE" in missing:
        required.append(SEARCH_DIR / "3wne_exact_search_all.csv.gz")
    required.extend(
        SEARCH_DIR / f"3zgc_round_{index:02d}.csv.gz" for index in range(1, 7)
    )
    required.extend(
        SEARCH_DIR / "checkpoints" / f"3zgc_round_{index:02d}.json.gz"
        for index in range(1, 7)
    )
    for directory in (MODEL_DIR, REPRESENTATION_DIR, BASELINE_DIR):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        required.extend(path for path in directory.rglob("*") if path.is_file())
    unique = sorted({path.resolve() for path in required}, key=lambda path: relative(path))
    require_files(unique)
    temporary = [str(path) for path in unique if path.name.endswith(".tmp")]
    if temporary:
        raise RuntimeError("Temporary partial files must not be exported: " + ", ".join(temporary))
    return unique


def write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(2026, 8, 18, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def export_bundle(output_path: Path) -> None:
    search = load_search_module()
    legacy_blob, source_search_sha256 = git_source_blob()
    config, source_config_sha256 = legacy_search_config(
        search, source_search_sha256
    )
    checkpoint_paths = [
        SEARCH_DIR / "checkpoints" / f"3zgc_round_{index:02d}.json.gz"
        for index in range(1, 7)
    ]
    checkpoint_digests = {
        str(search.read_gzip_json(path).get("config_sha256", ""))
        for path in checkpoint_paths
    }
    if checkpoint_digests != {source_config_sha256}:
        raise RuntimeError(
            "Round checkpoints do not match the reconstructed source environment. "
            f"checkpoint={sorted(checkpoint_digests)} reconstructed={source_config_sha256}"
        )
    static_audit = validate_static_search_evidence(search, source_config_sha256)
    files = source_inventory(search, static_audit["missing_targets"])
    file_inventory = {
        relative(path): {"sha256": sha256_file(path), "size": path.stat().st_size}
        for path in files
    }
    evidence_files = {
        name: item["sha256"]
        for name, item in file_inventory.items()
        if name.startswith(relative(SEARCH_DIR) + "/")
        and (name.endswith(".csv.gz") or name.endswith(".json.gz"))
    }
    relocation = {
        relative(path): manifest_relocations(path) for path in MANIFEST_REBASE_ORDER
    }
    manifest = {
        "quality_gate": "PASS",
        "protocol": EXPORT_PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "source_search_program_sha256": source_search_sha256,
        "source_config": config,
        "source_config_sha256": source_config_sha256,
        "destination_rescore_tolerance": RESCORE_TOLERANCE,
        "destination_full_ledger_reaudit_required": True,
        "static_search_evidence_audit": static_audit,
        "files": file_inventory,
        "evidence_files": evidence_files,
        "manifest_relocations": relocation,
        "legacy_source_archive_path": (
            relative(PORTABLE_DIR / "legacy_14_directed_recovery_search_v8.py")
        ),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    progress = search.ProgressBar("export resume bundle", len(files) + 2, unit="file")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in files:
            archive.write(path, relative(path))
            progress.update(1)
        write_member(
            archive,
            manifest["legacy_source_archive_path"],
            legacy_blob,
        )
        progress.update(1)
        write_member(archive, PORTABLE_MANIFEST_REL, manifest_bytes)
        progress.update(1)
    progress.close()
    os.replace(temporary, output_path)
    print("===== V8 AUTODL RESUME BUNDLE EXPORTED =====", flush=True)
    print(f"Bundle: {output_path}", flush=True)
    print(f"SHA256: {sha256_file(output_path)}", flush=True)
    print(f"Round-6 seen: {static_audit['zgc_evaluated_sequences']:,}", flush=True)
    print(f"Round-6 strict hits: {static_audit['zgc_strict_hits']:,}", flush=True)


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Unsafe archive path: {name}")
    return path


def set_pointer(payload: Any, pointer: Sequence[Any], value: Any) -> None:
    current = payload
    for key in pointer[:-1]:
        current = current[key]
    current[pointer[-1]] = value


def import_is_reusable() -> bool:
    if not IMPORT_MANIFEST.is_file():
        return False
    payload = read_json(IMPORT_MANIFEST)
    if not (
        payload.get("quality_gate") == "PASS"
        and payload.get("protocol") == IMPORT_PROTOCOL
    ):
        return False
    current = {
        "model": MODEL_CHECKPOINT,
        "model_manifest": MODEL_MANIFEST,
        "representation_audit": REPRESENTATION_MANIFEST,
        "baseline_manifest": BASELINE_MANIFEST,
    }
    expected = dict(payload.get("current_input_hashes") or {})
    imported_files = dict(payload.get("current_imported_file_hashes") or {})
    return bool(imported_files) and set(expected) == set(current) and all(
        path.is_file() and sha256_file(path) == str(expected[name])
        for name, path in current.items()
    ) and all(
        (REPO_ROOT / name).is_file()
        and sha256_file(REPO_ROOT / name) == str(digest)
        for name, digest in dict(payload.get("evidence_files") or {}).items()
    ) and all(
        (REPO_ROOT / name).is_file()
        and sha256_file(REPO_ROOT / name) == str(digest)
        for name, digest in imported_files.items()
    )


def import_bundle(bundle_path: Path) -> None:
    if import_is_reusable():
        print(f"Portable import: reused hash-valid {IMPORT_MANIFEST}", flush=True)
        return
    bundle_path = bundle_path.resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    with zipfile.ZipFile(bundle_path, "r") as archive:
        names = set(archive.namelist())
        if PORTABLE_MANIFEST_REL not in names:
            raise RuntimeError("Archive lacks the portable resume manifest")
        manifest_bytes = archive.read(PORTABLE_MANIFEST_REL)
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if not (
            manifest.get("quality_gate") == "PASS"
            and manifest.get("protocol") == EXPORT_PROTOCOL
            and manifest.get("source_commit") == SOURCE_COMMIT
            and manifest.get("source_search_program_sha256")
            in SOURCE_SEARCH_SHA256_ALLOWED
            and float(manifest.get("destination_rescore_tolerance", -1.0))
            == RESCORE_TOLERANCE
        ):
            raise RuntimeError("Portable export manifest is failed or unrecognized")
        inventory = dict(manifest.get("files") or {})
        required_names = set(inventory) | {
            PORTABLE_MANIFEST_REL,
            str(manifest["legacy_source_archive_path"]),
        }
        if not required_names <= names:
            raise RuntimeError("Portable archive is missing one or more declared files")
        progress = load_search_module().ProgressBar(
            "import resume bundle", len(required_names), unit="file"
        )
        for name in sorted(required_names):
            safe = safe_member(name)
            target = REPO_ROOT.joinpath(*safe.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            if name in inventory and sha256_bytes(data) != str(
                inventory[name]["sha256"]
            ):
                raise RuntimeError(f"Portable archive member hash mismatch: {name}")
            if target.is_file() and sha256_file(target) != sha256_bytes(data):
                raise RuntimeError(f"Import target already exists with different bytes: {target}")
            if not target.is_file():
                target.write_bytes(data)
            progress.update(1)
        progress.close()

    relocations = dict(manifest.get("manifest_relocations") or {})
    relocated_hashes: Dict[str, Dict[str, str]] = {}
    current_model_manifest_sha = ""
    current_representation_sha = ""
    for manifest_path in MANIFEST_REBASE_ORDER:
        name = relative(manifest_path)
        source_sha = str(manifest["files"][name]["sha256"])
        if sha256_file(manifest_path) != source_sha:
            raise RuntimeError(f"Manifest changed before relocation: {manifest_path}")
        payload = read_json(manifest_path)
        for row in relocations.get(name, []):
            set_pointer(
                payload,
                row["json_pointer"],
                str(REPO_ROOT / str(row["repo_relative_path"])),
            )
        if manifest_path == REPRESENTATION_MANIFEST:
            payload["model_manifest_sha256"] = current_model_manifest_sha
        elif manifest_path == BASELINE_MANIFEST:
            payload["expert_manifest_sha256"] = current_model_manifest_sha
            heldout = dict(payload.get("cyclic_representation_heldout_audit") or {})
            heldout["sha256"] = current_representation_sha
            payload["cyclic_representation_heldout_audit"] = heldout
        atomic_write_json(manifest_path, payload)
        destination_sha = sha256_file(manifest_path)
        relocated_hashes[name] = {
            "source_sha256": source_sha,
            "destination_sha256": destination_sha,
        }
        if manifest_path == MODEL_MANIFEST:
            current_model_manifest_sha = destination_sha
        elif manifest_path == REPRESENTATION_MANIFEST:
            current_representation_sha = destination_sha

    evidence_files = dict(manifest.get("evidence_files") or {})
    for name, expected_hash in evidence_files.items():
        path = REPO_ROOT / name
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise RuntimeError(f"Imported evidence hash mismatch: {name}")
    current_input_hashes = {
        "model": sha256_file(MODEL_CHECKPOINT),
        "model_manifest": sha256_file(MODEL_MANIFEST),
        "representation_audit": sha256_file(REPRESENTATION_MANIFEST),
        "baseline_manifest": sha256_file(BASELINE_MANIFEST),
    }
    current_imported_file_hashes = {
        name: sha256_file(REPO_ROOT / name) for name in sorted(inventory)
    }
    imported = {
        "quality_gate": "PASS",
        "protocol": IMPORT_PROTOCOL,
        "source_commit": manifest["source_commit"],
        "source_search_program_sha256": manifest[
            "source_search_program_sha256"
        ],
        "source_config_sha256": manifest["source_config_sha256"],
        "portable_export_manifest": str(REPO_ROOT / PORTABLE_MANIFEST_REL),
        "portable_export_manifest_sha256": sha256_bytes(manifest_bytes),
        "bundle_sha256": sha256_file(bundle_path),
        "destination_rescore_tolerance": RESCORE_TOLERANCE,
        "destination_full_ledger_reaudit_required": True,
        "current_input_hashes": current_input_hashes,
        "current_imported_file_hashes": current_imported_file_hashes,
        "relocated_manifests": relocated_hashes,
        "evidence_files": evidence_files,
        "static_search_evidence_audit": manifest[
            "static_search_evidence_audit"
        ],
    }
    atomic_write_json(IMPORT_MANIFEST, imported)
    print("===== V8 AUTODL RESUME BUNDLE IMPORTED =====", flush=True)
    print(f"Import manifest: {IMPORT_MANIFEST}", flush=True)
    print("Path metadata rebased; scientific artifact bytes remain hash-pinned.", flush=True)


def review_inventory() -> List[Path]:
    files = [
        path
        for path in V8_ROOT.rglob("*")
        if path.is_file() and path.resolve() != DEFAULT_REVIEW_BUNDLE.resolve()
    ]
    files.extend((V6_MANIFEST, V7_MANIFEST, V6_CHECKPOINT, V7_CHECKPOINT, TEST_JSONL))
    return sorted({path.resolve() for path in files}, key=lambda path: relative(path))


def package_review(output_path: Path) -> None:
    search_manifest = read_json(SEARCH_DIR / "directed_search_manifest.json")
    final_manifest = read_json(V8_ROOT / "generation_recovered" / "generation_manifest.json")
    audit_manifest = read_json(
        V8_ROOT / "triple_audit_recovered" / "three_pass_generation_audit.json"
    )
    if not all(
        payload.get("quality_gate") == "PASS"
        for payload in (search_manifest, final_manifest, audit_manifest)
    ):
        raise RuntimeError("Review bundle cannot be created before all V8 gates pass")
    files = review_inventory()
    search = load_search_module()
    progress = search.ProgressBar("package review bundle", len(files), unit="file")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in files:
            archive.write(path, relative(path))
            progress.update(1)
    progress.close()
    os.replace(temporary, output_path)
    print("===== V8 ALL AUTOMATED GATES PASSED; MANUAL SCIENTIFIC REVIEW IS NEXT =====")
    print("Final coverage: 17/17; no formal abstention")
    print("Shang-ge handoff: NOT CREATED")
    print(f"Review bundle: {output_path}")
    print(f"Review bundle SHA256: {sha256_file(output_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output", default=str(DEFAULT_BUNDLE))
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--bundle", required=True)
    review_parser = subparsers.add_parser("package-review")
    review_parser.add_argument("--output", default=str(DEFAULT_REVIEW_BUNDLE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export_bundle(Path(args.output))
    elif args.command == "import":
        import_bundle(Path(args.bundle))
    elif args.command == "package-review":
        package_review(Path(args.output))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
