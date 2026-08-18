#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a compact, hash-indexed scientific review bundle for V8 V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"
DEFAULT_SEARCH = V8_ROOT / "directed_search_cyclic_base_v2"
DEFAULT_GENERATION = V8_ROOT / "generation_recovered_cyclic_base_v2"
DEFAULT_AUDIT = V8_ROOT / "triple_audit_recovered_cyclic_base_v2"
DEFAULT_OUTPUT = V8_ROOT / "v8_cyclic_base_v2_review_bundle.zip"
SEARCH_PROTOCOL = "cyclic_start_base_pareto_recovery_v8_v2"
FINAL_PROTOCOL = "immutable_baseline_plus_cyclic_base_recovery_overlay_v8_v2"
AUDIT_PROTOCOL = "independent_three_pass_cyclic_base_recovery_v8_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def artifact_leaves(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if set(value) >= {"path", "sha256"}:
            yield value
        else:
            for child in value.values():
                yield from artifact_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from artifact_leaves(child)


def validate_manifest_artifacts(manifest: Mapping[str, Any], root: Path) -> None:
    leaves = list(artifact_leaves(manifest.get("artifacts")))
    if not leaves:
        raise RuntimeError(f"Manifest has no artifact leaves under {root}")
    for leaf in leaves:
        path = Path(str(leaf["path"])).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Artifact escapes declared root: {path}") from exc
        if not path.is_file() or sha256_file(path) != str(leaf["sha256"]):
            raise RuntimeError(f"Artifact is absent or stale: {path}")


def write_deterministic_zip(
    output: Path,
    files: Sequence[tuple[Path, str]],
    bundle_manifest: Mapping[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    entries = sorted(files, key=lambda item: item[1])
    hashes = [f"{sha256_file(path)}  {arcname}" for path, arcname in entries]
    manifest_bytes = (
        json.dumps(bundle_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    sums_bytes = ("\n".join(hashes) + "\n").encode("ascii")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path, arcname in entries:
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        for arcname, data in (
            ("BUNDLE_MANIFEST.json", manifest_bytes),
            ("SHA256SUMS.txt", sums_bytes),
        ):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    os.replace(temporary, output)


def run(args: argparse.Namespace) -> None:
    search = Path(args.search_dir).resolve()
    generation = Path(args.generation_dir).resolve()
    audit = Path(args.audit_dir).resolve()
    output = Path(args.output).resolve()
    search_manifest_path = search / "cyclic_base_recovery_manifest.json"
    final_manifest_path = generation / "generation_manifest.json"
    audit_report_path = audit / "three_pass_generation_audit.json"
    for required in (search_manifest_path, final_manifest_path, audit_report_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    search_manifest = read_json(search_manifest_path)
    final_manifest = read_json(final_manifest_path)
    audit_report = read_json(audit_report_path)
    if not (
        search_manifest.get("quality_gate") == "PASS"
        and search_manifest.get("protocol") == SEARCH_PROTOCOL
        and final_manifest.get("quality_gate") == "PASS"
        and final_manifest.get("protocol") == FINAL_PROTOCOL
        and audit_report.get("quality_gate") == "PASS"
        and audit_report.get("protocol") == AUDIT_PROTOCOL
        and final_manifest.get("search_manifest_sha256")
        == sha256_file(search_manifest_path)
        and dict(audit_report.get("artifacts") or {})
        .get("final_manifest", {})
        .get("sha256")
        == sha256_file(final_manifest_path)
    ):
        raise RuntimeError("V8 V2 search/final/audit manifests are not a linked PASS chain")
    validate_manifest_artifacts(search_manifest, search)
    validate_manifest_artifacts(final_manifest, generation)
    validate_manifest_artifacts(audit_report, REPO_ROOT)

    requested = [
        (search_manifest_path, "search/cyclic_base_recovery_manifest.json"),
        (search / "directed_candidates.csv", "search/directed_candidates.csv"),
        (
            search / "cyclic_base_plausibility_and_position_evidence.csv",
            "search/cyclic_base_plausibility_and_position_evidence.csv",
        ),
        (
            search / "legacy_strict_hit_cyclic_reaudit.csv",
            "search/legacy_strict_hit_cyclic_reaudit.csv",
        ),
        (
            search / "baseline_cyclic_start_plausibility.csv",
            "search/baseline_cyclic_start_plausibility.csv",
        ),
        (
            search / "mandatory_length_6_7_controls.csv",
            "search/mandatory_length_6_7_controls.csv",
        ),
        (search / "search_trace_by_round.csv", "search/search_trace_by_round.csv"),
        (final_manifest_path, "generation/generation_manifest.json"),
        (
            generation / "methylated_new_candidates.csv",
            "generation/methylated_new_candidates.csv",
        ),
        (
            generation / "target_manifest.csv",
            "generation/target_manifest.csv",
        ),
        (
            generation / "generation_summary_by_target.csv",
            "generation/generation_summary_by_target.csv",
        ),
        (audit_report_path, "audit/three_pass_generation_audit.json"),
        (
            audit / "three_pass_concentration_by_target.csv",
            "audit/three_pass_concentration_by_target.csv",
        ),
        (
            audit / "av_family_physical_position_support.json",
            "audit/av_family_physical_position_support.json",
        ),
        (
            SCRIPT_PATH.with_name("17_cyclic_base_recovery_v2.py"),
            "programs/17_cyclic_base_recovery_v2.py",
        ),
        (
            SCRIPT_PATH.with_name("18_finalize_and_audit_recovery_v2.py"),
            "programs/18_finalize_and_audit_recovery_v2.py",
        ),
        (SCRIPT_PATH, "programs/19_package_v8_recovery_v2.py"),
        (
            SCRIPT_PATH.with_name("V8_CYCLIC_BASE_RECOVERY_V2.md"),
            "programs/V8_CYCLIC_BASE_RECOVERY_V2.md",
        ),
        (REPO_ROOT / "run_v8_autodl_recovery_v2.sh", "programs/run_v8_autodl_recovery_v2.sh"),
    ]
    missing = [str(path) for path, _arcname in requested if not path.is_file()]
    if missing:
        raise RuntimeError("Review bundle input is absent: " + ", ".join(missing))
    bundle_manifest = {
        "quality_gate": "PASS",
        "purpose": "SCIENTIFIC_REVIEW_ONLY_NO_STRUCTURE_HANDOFF_NO_PERMEABILITY_INPUT",
        "search_manifest_sha256": sha256_file(search_manifest_path),
        "final_manifest_sha256": sha256_file(final_manifest_path),
        "audit_report_sha256": sha256_file(audit_report_path),
        "files": {
            arcname: sha256_file(path) for path, arcname in sorted(requested, key=lambda item: item[1])
        },
    }
    write_deterministic_zip(output, requested, bundle_manifest)
    print("===== V8 V2 REVIEW BUNDLE COMPLETE =====", flush=True)
    print(f"Bundle: {output}", flush=True)
    print(f"SHA256: {sha256_file(output)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", default=str(DEFAULT_SEARCH))
    parser.add_argument("--generation-dir", default=str(DEFAULT_GENERATION))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
