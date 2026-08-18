#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic directed recovery for missing V8 target signatures.

The model is frozen before this program starts.  Search cannot alter or
"repair" model metrics; it only explores receptor/backbone-compatible natural
sequence space for 3WNE and 3ZGC under the same all-cyclic-start, all-decoder-
order, physical-position-mapped expert score used by deployment.

Historical controls are always scored but are never release eligible.  Every
released sequence must pass strict rounded probability ``>0.6``, independent
batch-one re-scoring, receptor-conditioned ProteinMPNN plausibility, historical
and prior naturalized novelty, native/current-pool exclusion, and forward
cyclic-identity exclusion.  Failure after the fixed budget remains a failure.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
# This must be set before PyTorch initializes CUDA.  The frozen search protocol
# fails rather than silently falling back to a nondeterministic CUDA kernel.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
REANNOTATOR_PATH = SCRIPT_PATH.with_name("10_reannotate_v6_pool_serine_only_v7.py")
GENERATOR_PATH = (
    REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
)
COMMON_PATH = REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"
MODEL_UTILS_PATH = REPO_ROOT / "model_utils.py"
NMETHYL_CONFIG_PATH = REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"
V8_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_source_scoped_hybrid_v8"
DEFAULT_MODEL = V8_ROOT / "model" / "frankenstein_v28_source_scoped_hybrid_v8.pt"
DEFAULT_MODEL_MANIFEST = V8_ROOT / "model" / "expert_source_composition_manifest.json"
DEFAULT_REPRESENTATION = V8_ROOT / "representation_audit" / "cyclic_representation_audit.json"
DEFAULT_BASELINE = V8_ROOT / "generation_baseline"
DEFAULT_OUT = V8_ROOT / "directed_search"
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_representation_v6.json")
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_HISTORICAL = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "all_designs.csv"
)
DEFAULT_PRIOR = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "rerun_temperature_0.5_multiseed"
    / "methylated_new_candidates.csv"
)

V8_EXPERT_PROTOCOL = (
    "canonical_shared_v6_non_ser_v7_ser_cyclic_representation_hybrid_v8"
)
V8_REPRESENTATION_PROTOCOL = (
    "cyclic_representation_frozen_audit_source_scoped_hybrid_v8"
)
V8_REPRESENTATION_AUTHORIZATION = (
    "SOURCE_SCOPED_HYBRID_V8_AUTHORIZED_FOR_DIRECTED_RECOVERY"
)
V8_BASELINE_PROTOCOL = (
    "temperature_0.5_source_scoped_hybrid_v8_reannotation_of_preserved_v6_pool"
)
V8_SEARCH_PROTOCOL = "deterministic_missing_target_directed_recovery_v8"
V8_MODEL_ARTIFACT_FILENAMES = {
    "metric_comparison": "v6_v7_v8_metric_comparison.csv",
    "serine_auc_tradeoff_audit": "serine_auc_tradeoff_audit.csv",
    "metrics_by_residue": "test_metrics_by_residue.csv",
    "position_probabilities": "test_position_probabilities.csv",
}
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
METHYLATABLE_AA = set(NATURAL_AA) - {"P"}
ALLOWED_RECOVERY_TARGETS = {"3WNE", "3ZGC"}
HISTORICAL_CONTROLS = {
    "3WNE": {"sequence": "GRKWNC", "old_design": "GrKWNC", "old_site": 2},
    "3ZGC": {"sequence": "REGGQNR", "old_design": "rEGGQNR", "old_site": 1},
}
NATIVE_CONTROLS = {"3WNE": "PKIDNG", "3ZGC": "GDEETGE"}
THRESHOLD = 0.6
TEMPERATURE = 0.5
BASE_PERCENTILE = 0.01
RESCORE_TOLERANCE = 2e-6
SEARCH_LEDGER_FIELDS = [
    "target_name",
    "sequence",
    "search_stage",
    "maximum_probability",
    "argmax_position_1based",
    "argmax_residue",
    "passes_strict_probability",
    "generation_kind",
    "parent_sequence",
    "edit_distance",
    "mutation_positions_1based",
    "rng_seed",
    "rng_draw_index",
]

PORTABLE_RESUME_IMPORT_PROTOCOL = "v8_autodl_portable_resume_import_v1"
PORTABLE_RESCORE_TOLERANCE = 2e-6
PORTABLE_SOURCE_COMMIT = "53ce92e5238d717fc982357b4c58f65538a8f710"
PORTABLE_SOURCE_SEARCH_SHA256 = {
    # Git blob / LF checkout.
    "d0d3536a51ac92caabc1523e8b7418811ac71b4abf3588485055223408ea7097",
    # Windows core.autocrlf checkout of the same Git blob.
    "2bce6d3cb017cdacf62c130810616d08b80d73d0fc9f2dc4122c5be2aeb60a96",
}


def format_duration(seconds: float) -> str:
    if not math.isfinite(float(seconds)) or float(seconds) < 0:
        return "--:--"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    """Dependency-free progress bar with rate and ETA for long GPU stages."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        unit: str = "seq",
        min_interval: float = 5.0,
    ) -> None:
        self.label = str(label)
        self.total = max(0, int(total))
        self.unit = str(unit)
        self.min_interval = max(0.2, float(min_interval))
        self.completed = 0
        self.started = time.monotonic()
        self.last_print = 0.0
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if self.total:
            self._render(force=True)

    def update(self, amount: int) -> None:
        self.completed = min(self.total, self.completed + max(0, int(amount)))
        self._render(force=self.completed >= self.total)

    def close(self) -> None:
        if self.total and self.completed < self.total:
            self.completed = self.total
            self._render(force=True)

    def _render(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self.last_print < self.min_interval:
            return
        elapsed = max(now - self.started, 1e-9)
        fraction = self.completed / self.total if self.total else 1.0
        width = 24
        filled = min(width, int(fraction * width))
        bar = "#" * filled + "-" * (width - filled)
        rate = self.completed / elapsed
        remaining = (self.total - self.completed) / rate if rate > 0 else float("inf")
        line = (
            f"[{self.label}] [{bar}] {fraction * 100:6.2f}% "
            f"{self.completed:,}/{self.total:,} {self.unit} | "
            f"{rate:,.2f} {self.unit}/s | elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(remaining)}"
        )
        if self.tty:
            print("\r" + line, end="\n" if self.completed >= self.total else "", flush=True)
        else:
            print(line, flush=True)
        self.last_print = now


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path is equal to or contains the other."""

    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def portable_resume_expected_evidence_names(
    missing_targets: Iterable[str],
) -> set[str]:
    """Return the exact round-six evidence contract for the frozen target state."""

    missing = {str(value).upper() for value in missing_targets}
    if "3ZGC" not in missing or not missing <= ALLOWED_RECOVERY_TARGETS:
        raise RuntimeError(
            "Portable round-six resume requires 3ZGC and permits only the "
            "frozen 3WNE/3ZGC recovery targets"
        )
    expected = {
        "3zgc_round_00_initial.csv.gz",
        *(f"3zgc_round_{index:02d}.csv.gz" for index in range(1, 7)),
        *(f"3zgc_round_{index:02d}.json.gz" for index in range(1, 7)),
    }
    if "3WNE" in missing:
        expected.add("3wne_exact_search_all.csv.gz")
    return expected


def validate_portable_resume_import(
    import_manifest_path: Path,
    out_dir: Path,
    model_path: Path,
    model_manifest_path: Path,
    representation_path: Path,
    baseline: Path,
) -> Dict[str, Any]:
    """Validate a hash-pinned Windows-to-AutoDL evidence import.

    Only round ledgers/checkpoints are trusted from the source runtime.  Final
    candidate annotation is recomputed on the destination GPU and the final
    three-pass audit independently re-scores every ledger row.
    """

    path = import_manifest_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if not (
        payload.get("quality_gate") == "PASS"
        and payload.get("protocol") == PORTABLE_RESUME_IMPORT_PROTOCOL
        and payload.get("source_commit") == PORTABLE_SOURCE_COMMIT
        and payload.get("source_search_program_sha256")
        in PORTABLE_SOURCE_SEARCH_SHA256
        and str(payload.get("source_config_sha256", ""))
        and float(payload.get("destination_rescore_tolerance", -1.0))
        == PORTABLE_RESCORE_TOLERANCE
    ):
        raise RuntimeError("Portable resume import manifest is failed or unrecognized")
    expected_current = {
        "model": model_path,
        "model_manifest": model_manifest_path,
        "representation_audit": representation_path,
        "baseline_manifest": baseline / "generation_manifest.json",
    }
    current_hashes = dict(payload.get("current_input_hashes") or {})
    if set(current_hashes) != set(expected_current) or any(
        not target.is_file() or sha256_file(target) != str(current_hashes[name])
        for name, target in expected_current.items()
    ):
        raise RuntimeError("Portable resume current input hash map is stale")
    evidence = dict(payload.get("evidence_files") or {})
    if not evidence:
        raise RuntimeError("Portable resume import has no evidence inventory")
    resolved_evidence: List[Path] = []
    for relative_name, expected_hash in evidence.items():
        candidate = (REPO_ROOT / str(relative_name)).resolve()
        try:
            candidate.relative_to(out_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Portable resume evidence escapes directed-search output: {relative_name}"
            ) from exc
        if not candidate.is_file() or sha256_file(candidate) != str(expected_hash):
            raise RuntimeError(f"Portable resume evidence hash mismatch: {relative_name}")
        resolved_evidence.append(candidate)
    static_audit = dict(payload.get("static_search_evidence_audit") or {})
    expected_names = portable_resume_expected_evidence_names(
        static_audit.get("missing_targets", [])
    )
    observed_names = {candidate.name for candidate in resolved_evidence}
    if not expected_names <= observed_names:
        raise RuntimeError("Portable resume evidence inventory is incomplete")
    checkpoint_digests = {
        str(read_gzip_json(candidate).get("config_sha256", ""))
        for candidate in resolved_evidence
        if candidate.name.endswith(".json.gz")
    }
    if checkpoint_digests != {str(payload["source_config_sha256"])}:
        raise RuntimeError("Portable resume checkpoint configuration digest mismatch")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def atomic_write_gzip_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as compressed:
            text_handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            writer = csv.DictWriter(
                text_handle, fieldnames=list(fieldnames), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
            text_handle.flush()
            text_handle.detach()
    os.replace(temporary, path)


def write_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as compressed:
            compressed.write(encoded)
    os.replace(temporary, path)


def read_gzip_json(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def artifact_leaves(value: Any) -> List[Mapping[str, Any]]:
    leaves: List[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            leaves.append(value)
        else:
            for child in value.values():
                leaves.extend(artifact_leaves(child))
    elif isinstance(value, list):
        for child in value:
            leaves.extend(artifact_leaves(child))
    return leaves


def artifact_hashes_match(value: Any) -> bool:
    """Recursively verify every ``{path, sha256}`` leaf in a manifest."""

    leaves = artifact_leaves(value)
    return bool(leaves) and all(
        Path(str(leaf["path"])).is_file()
        and sha256_file(Path(str(leaf["path"]))) == str(leaf["sha256"])
        for leaf in leaves
    )


def artifact_matches_exact_path(value: Any, expected_path: Path) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        return False
    declared = Path(str(value["path"])).resolve()
    expected = expected_path.resolve()
    return (
        declared == expected
        and expected.is_file()
        and sha256_file(expected) == str(value["sha256"])
    )


def artifact_map_matches_exact_paths(
    artifacts: Any, expected: Mapping[str, Path]
) -> bool:
    return (
        isinstance(artifacts, Mapping)
        and set(artifacts) == set(expected)
        and all(
            artifact_matches_exact_path(artifacts[name], path)
            for name, path in expected.items()
        )
    )


def declared_path_hash_is_current(
    payload: Mapping[str, Any],
    path_field: str,
    hash_field: str,
    expected_path: Optional[Path] = None,
) -> bool:
    declared = Path(str(payload.get(path_field, ""))).resolve()
    try:
        declared.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False
    return (
        declared.is_file()
        and (expected_path is None or declared == expected_path.resolve())
        and sha256_file(declared) == str(payload.get(hash_field, ""))
    )


def search_artifacts_match_exact(
    artifacts: Any,
    out_dir: Path,
    missing_targets: Sequence[str],
    zgc_rounds: int,
) -> bool:
    missing = set(missing_targets)
    expected_artifact_keys = {
        "controls",
        "plausibility",
        "directed_candidates",
        "trace",
    }
    if missing:
        expected_artifact_keys.add("search_ledgers")
    if "3ZGC" in missing:
        expected_artifact_keys.add("checkpoints")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifact_keys:
        return False
    top_level_ok = all(
        artifact_matches_exact_path(artifacts[name], path)
        for name, path in {
            "controls": out_dir / "mandatory_length_6_7_controls.csv",
            "plausibility": out_dir
            / "qualified_candidate_plausibility_and_novelty.csv",
            "directed_candidates": out_dir / "directed_candidates.csv",
            "trace": out_dir / "search_trace_by_round.csv",
        }.items()
    )
    expected_ledgers: Dict[str, Path] = {}
    expected_checkpoints: Dict[str, Path] = {}
    if "3WNE" in missing:
        expected_ledgers["3wne_exact_search_all.csv.gz"] = (
            out_dir / "3wne_exact_search_all.csv.gz"
        )
    if "3ZGC" in missing:
        expected_ledgers["3zgc_round_00_initial.csv.gz"] = (
            out_dir / "3zgc_round_00_initial.csv.gz"
        )
        for round_index in range(1, int(zgc_rounds) + 1):
            ledger_name = f"3zgc_round_{round_index:02d}.csv.gz"
            checkpoint_name = f"3zgc_round_{round_index:02d}.json.gz"
            expected_ledgers[ledger_name] = out_dir / ledger_name
            expected_checkpoints[checkpoint_name] = (
                out_dir / "checkpoints" / checkpoint_name
            )
    return (
        top_level_ok
        and (
            not expected_ledgers
            or artifact_map_matches_exact_paths(
                artifacts.get("search_ledgers"), expected_ledgers
            )
        )
        and (
            not expected_checkpoints
            or artifact_map_matches_exact_paths(
                artifacts.get("checkpoints"), expected_checkpoints
            )
        )
    )


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal lengths")
    return sum(a != b for a, b in zip(left, right))


def forward_cyclic_identity(sequence: str) -> str:
    value = str(sequence).upper()
    if not value:
        raise ValueError("Cyclic identity requires a non-empty sequence")
    return min(value[index:] + value[:index] for index in range(len(value)))


def hamming_neighborhood(
    anchor: str, radius: int, alphabet: str = NATURAL_AA
) -> List[str]:
    """Enumerate the exact union of Hamming shells 0..radius deterministically."""

    anchor = str(anchor).upper()
    if radius < 0 or radius > len(anchor):
        raise ValueError("Invalid Hamming radius")
    if not anchor or not set(anchor) <= set(alphabet):
        raise ValueError("Anchor contains a non-natural token")
    result = {anchor}
    positions = range(len(anchor))
    for distance in range(1, radius + 1):
        for chosen in combinations(positions, distance):
            replacements = [
                [token for token in alphabet if token != anchor[position]]
                for position in chosen
            ]
            for tokens in product(*replacements):
                sequence = list(anchor)
                for position, token in zip(chosen, tokens):
                    sequence[position] = token
                result.add("".join(sequence))
    return sorted(result)


def wne_search_provenance(
    anchors: Sequence[Tuple[str, str]], radius: int
) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Tuple[int, int, str, str]] = {}
    for priority, (anchor, source) in enumerate(anchors):
        for sequence in hamming_neighborhood(anchor, radius):
            candidate = (hamming(sequence, anchor), priority, anchor, source)
            if sequence not in selected or candidate < selected[sequence]:
                selected[sequence] = candidate
    return {
        sequence: {
            "generation_kind": "exact_hamming_neighborhood",
            "parent_sequence": anchor,
            "edit_distance": distance,
            "mutation_positions_1based": json.dumps(
                [
                    index
                    for index, (left, right) in enumerate(
                        zip(anchor, sequence), start=1
                    )
                    if left != right
                ]
            ),
            "rng_seed": "",
            "rng_draw_index": "",
            "anchor_source": source,
        }
        for sequence, (distance, _priority, anchor, source) in selected.items()
    }


def single_mutants(sequence: str, alphabet: str = NATURAL_AA) -> Iterable[str]:
    for position, original in enumerate(sequence):
        for token in alphabet:
            if token != original:
                yield sequence[:position] + token + sequence[position + 1 :]


def zgc_initial_anchor_provenance(
    anchors: Sequence[str], ranked_top: str
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for sequence in anchors:
        if sequence == HISTORICAL_CONTROLS["3ZGC"]["sequence"]:
            source = "withdrawn_historical_control"
        elif sequence == NATIVE_CONTROLS["3ZGC"]:
            source = "native_control"
        elif sequence == ranked_top:
            source = "current_v8_baseline_top"
        else:
            source = "diverse_current_v8_baseline_seed"
        result[sequence] = {
            "generation_kind": "frozen_initial_anchor",
            "parent_sequence": sequence,
            "edit_distance": 0,
            "mutation_positions_1based": "[]",
            "rng_seed": "",
            "rng_draw_index": "",
            "anchor_source": source,
        }
    return result


def zgc_round_provenance(
    beam: Sequence[Mapping[str, Any]],
    round_index: int,
    offspring_per_round: int,
    numpy_module: Any,
) -> Dict[str, Dict[str, Any]]:
    """Reproduce the complete frozen candidate/provenance set for one round."""

    generated: Dict[str, Dict[str, Any]] = {}
    for row in sorted(beam, key=lambda value: str(value["sequence"])):
        parent = str(row["sequence"])
        for mutation in single_mutants(parent):
            position = next(
                index
                for index, (left, right) in enumerate(
                    zip(parent, mutation), start=1
                )
                if left != right
            )
            generated.setdefault(
                mutation,
                {
                    "generation_kind": "complete_single_mutant",
                    "parent_sequence": parent,
                    "edit_distance": 1,
                    "mutation_positions_1based": json.dumps([position]),
                    "rng_seed": "",
                    "rng_draw_index": "",
                },
            )
    rng = numpy_module.random.Generator(
        numpy_module.random.PCG64(
            numpy_module.random.SeedSequence([20260817, int(round_index)])
        )
    )
    beam_sequences = [str(row["sequence"]) for row in beam]
    if not beam_sequences:
        raise RuntimeError("3ZGC beam is empty before a frozen search round")
    for draw_index in range(int(offspring_per_round)):
        parent = beam_sequences[int(rng.integers(0, len(beam_sequences)))]
        mutation_count = int(rng.integers(2, 5))
        positions = sorted(
            int(value)
            for value in rng.choice(7, size=mutation_count, replace=False)
        )
        child = list(parent)
        for position in positions:
            alternatives = [token for token in NATURAL_AA if token != child[position]]
            child[position] = alternatives[int(rng.integers(0, len(alternatives)))]
        child_sequence = "".join(child)
        generated.setdefault(
            child_sequence,
            {
                "generation_kind": "fixed_seed_multi_mutant",
                "parent_sequence": parent,
                "edit_distance": mutation_count,
                "mutation_positions_1based": json.dumps(
                    [position + 1 for position in positions]
                ),
                "rng_seed": f"20260817:{round_index}",
                "rng_draw_index": draw_index,
            },
        )
    return generated


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("Nearest-rank percentile requires values and 0<fraction<=1")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Nearest-rank percentile requires finite values")
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def strict_rounded_pass(value: float, threshold: float = THRESHOLD) -> bool:
    numeric = float(value)
    return (
        math.isfinite(numeric)
        and 0.0 <= numeric <= 1.0
        and round(numeric, 8) > float(threshold)
    )


def actionable_probability_max(
    sequence: str, probabilities: Sequence[float]
) -> float:
    natural = str(sequence).upper()
    values = [float(value) for value in probabilities]
    if (
        not natural
        or not set(natural) <= set(NATURAL_AA)
        or len(values) != len(natural)
        or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
    ):
        raise ValueError("Actionable methyl probability vector is invalid")
    return max(
        (
            probability
            for token, probability in zip(natural, values)
            if token in METHYLATABLE_AA
        ),
        default=0.0,
    )


def select_diverse_sequences(
    ranked_sequences: Sequence[str], count: int, initial: Sequence[str] = ()
) -> List[str]:
    """Greedy max-min Hamming selection with deterministic score-order ties."""

    unique = list(dict.fromkeys(str(value).upper() for value in ranked_sequences))
    selected = list(dict.fromkeys(str(value).upper() for value in initial))
    candidates = [value for value in unique if value not in set(selected)]
    while candidates and len(selected) < count:
        if not selected:
            chosen = candidates[0]
        else:
            indexed = {value: index for index, value in enumerate(unique)}
            chosen = max(
                candidates,
                key=lambda value: (
                    min(hamming(value, prior) for prior in selected),
                    -indexed[value],
                    value,
                ),
            )
        selected.append(chosen)
        candidates.remove(chosen)
    return selected[:count]


def select_beam(
    scored: Mapping[str, Mapping[str, Any]], width: int, length: int
) -> List[Dict[str, Any]]:
    ordered = sorted(
        (dict(value) for value in scored.values()),
        key=lambda row: (-float(row["maximum_probability"]), str(row["sequence"])),
    )
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    per_position = max(1, min(32, width // max(1, 2 * length)))
    for position in range(1, length + 1):
        candidates = [row for row in ordered if int(row["argmax_position_1based"]) == position]
        for row in candidates[:per_position]:
            sequence = str(row["sequence"])
            if sequence not in seen:
                selected.append(row)
                seen.add(sequence)
    score_fill = min(width, max(len(selected), int(width * 0.75)))
    for row in ordered:
        if len(selected) >= score_fill:
            break
        sequence = str(row["sequence"])
        if sequence not in seen:
            selected.append(row)
            seen.add(sequence)
    diversity_pool = [row for row in ordered[: max(width * 8, width)] if str(row["sequence"]) not in seen]
    minimum_distances = {
        str(row["sequence"]): min(
            hamming(str(row["sequence"]), str(prior["sequence"]))
            for prior in selected
        )
        for row in diversity_pool
    }
    while diversity_pool and len(selected) < width:
        chosen = max(
            diversity_pool,
            key=lambda row: (
                minimum_distances[str(row["sequence"])],
                float(row["maximum_probability"]),
                str(row["sequence"]),
            ),
        )
        selected.append(chosen)
        chosen_sequence = str(chosen["sequence"])
        seen.add(chosen_sequence)
        diversity_pool.remove(chosen)
        minimum_distances.pop(chosen_sequence)
        for row in diversity_pool:
            sequence = str(row["sequence"])
            minimum_distances[sequence] = min(
                minimum_distances[sequence],
                hamming(sequence, chosen_sequence),
            )
    return selected[:width]


def normalize_search_ledger_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Restore CSV evidence to the exact scalar types stored in checkpoints."""

    result = {str(key): ("" if value is None else value) for key, value in row.items()}
    for field in (
        "argmax_position_1based",
        "passes_strict_probability",
        "edit_distance",
        "rng_draw_index",
    ):
        if str(result.get(field, "")):
            result[field] = int(result[field])
    if str(result.get("maximum_probability", "")):
        result["maximum_probability"] = float(result["maximum_probability"])
    return result


def validate_search_ledger_row(
    row: Mapping[str, Any],
    target: str,
    sequence: str,
    stage: str,
    expected_provenance: Mapping[str, Any],
) -> None:
    try:
        maximum = float(row["maximum_probability"])
        argmax_position = int(row["argmax_position_1based"])
        argmax_residue = str(row["argmax_residue"])
        passes = int(row["passes_strict_probability"])
        observed_positions = [
            int(value)
            for value in json.loads(str(row["mutation_positions_1based"]))
        ]
        expected_positions = [
            int(value)
            for value in json.loads(
                str(expected_provenance["mutation_positions_1based"])
            )
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Malformed {target} provenance row: {sequence}") from exc
    expected_pass = int(
        argmax_residue in METHYLATABLE_AA and strict_rounded_pass(maximum)
    )
    comparable_fields = (
        "generation_kind",
        "parent_sequence",
        "rng_seed",
    )
    if not (
        str(row.get("target_name", "")).upper() == target
        and str(row.get("sequence", "")).upper() == sequence
        and bool(sequence)
        and set(sequence) <= set(NATURAL_AA)
        and str(row.get("search_stage", "")) == stage
        and math.isfinite(maximum)
        and 0.0 <= maximum <= 1.0
        and 1 <= argmax_position <= len(sequence)
        and argmax_residue == sequence[argmax_position - 1]
        and passes == expected_pass
        and all(
            str(row.get(field, "")) == str(expected_provenance.get(field, ""))
            for field in comparable_fields
        )
        and int(row.get("edit_distance", -1))
        == int(expected_provenance["edit_distance"])
        and observed_positions == expected_positions
        and str(row.get("rng_draw_index", ""))
        == str(expected_provenance.get("rng_draw_index", ""))
        and (
            "anchor_source" not in expected_provenance
            or str(row.get("anchor_source", ""))
            == str(expected_provenance["anchor_source"])
        )
    ):
        raise RuntimeError(
            f"{target} ledger provenance mismatch: {stage}:{sequence}"
        )


def validate_ledger_scores_against_model(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    stage: str,
    score_minimal: Any,
    score_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Recompute every persisted ledger score before it can steer a beam.

    ``MethylScorer.score_minimal`` persists eight-decimal probabilities, so a
    deterministic re-score under the frozen backend must reproduce the score,
    argmax, residue, and strict-pass bit exactly.  Hash/self-consistency alone
    is not evidence that a partial ledger came from the frozen model.
    """

    sequences = [str(row.get("sequence", "")).upper() for row in rows]
    if not sequences or any(not sequence for sequence in sequences):
        raise RuntimeError(f"{target} {stage} ledger has no model-score rows")
    if len(sequences) != len(set(sequences)):
        raise RuntimeError(f"{target} {stage} ledger has duplicate score rows")
    tolerance = float(score_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Ledger score tolerance must be finite and non-negative")
    observed = score_minimal(target, sequences, stage)
    if set(observed) != set(sequences):
        raise RuntimeError(f"{target} {stage} model re-score key mismatch")
    maximum_absolute_difference = 0.0
    for persisted in rows:
        sequence = str(persisted["sequence"]).upper()
        recomputed = observed[sequence]
        try:
            absolute_difference = abs(
                float(recomputed["maximum_probability"])
                - float(persisted["maximum_probability"])
            )
            maximum_absolute_difference = max(
                maximum_absolute_difference, absolute_difference
            )
            matches = (
                str(recomputed.get("target_name", "")).upper() == target
                and str(recomputed.get("sequence", "")).upper() == sequence
                and str(recomputed.get("search_stage", "")) == stage
                and absolute_difference <= tolerance
                and int(recomputed["argmax_position_1based"])
                == int(persisted["argmax_position_1based"])
                and str(recomputed["argmax_residue"])
                == str(persisted["argmax_residue"])
                and int(recomputed["passes_strict_probability"])
                == int(persisted["passes_strict_probability"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Malformed {target} model-score evidence: {stage}:{sequence}"
            ) from exc
        if not matches:
            raise RuntimeError(
                f"{target} ledger score is not reproduced by the frozen model: "
                f"{stage}:{sequence}"
            )
    return {
        "target_name": target,
        "stage": stage,
        "rows": len(rows),
        "score_tolerance": tolerance,
        "maximum_absolute_probability_difference": maximum_absolute_difference,
        "strict_pass_bits_match": True,
    }


def reconstruct_and_validate_zgc_resume(
    out_dir: Path,
    checkpoints: Sequence[Path],
    config_digest: str,
    beam_width: int,
    expected_initial_provenance: Mapping[str, Mapping[str, Any]],
    offspring_per_round: int,
    numpy_module: Any,
    score_minimal: Any,
    *,
    validate_model_scores: bool = True,
    score_tolerance: float = 0.0,
) -> Tuple[int, set[str], List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Rebuild resume state from complete ledgers; never trust checkpoint state alone."""

    round_numbers = [int(path.name.split("_")[-1].split(".")[0]) for path in checkpoints]
    if not round_numbers or round_numbers != list(range(1, round_numbers[-1] + 1)):
        raise RuntimeError("3ZGC checkpoint rounds are not contiguous from round 1")
    initial_path = out_dir / "3zgc_round_00_initial.csv.gz"
    if not initial_path.is_file():
        raise RuntimeError("3ZGC resume lacks the initial-anchor ledger")
    initial_rows = [normalize_search_ledger_row(row) for row in read_gzip_csv(initial_path)]
    initial_scored = {str(row["sequence"]): row for row in initial_rows}
    if (
        not initial_scored
        or len(initial_scored) != len(initial_rows)
        or set(initial_scored) != set(expected_initial_provenance)
    ):
        raise RuntimeError(
            "3ZGC initial ledger is empty, duplicated, or not the frozen anchor set"
        )
    for sequence, row in initial_scored.items():
        validate_search_ledger_row(
            row,
            "3ZGC",
            sequence,
            "beam_initial_anchors",
            expected_initial_provenance[sequence],
        )
    if validate_model_scores:
        if score_minimal is None:
            raise ValueError("Model-score validation requires a scorer")
        validate_ledger_scores_against_model(
            initial_rows,
            "3ZGC",
            "beam_initial_anchors",
            score_minimal,
            score_tolerance,
        )
    seen = set(initial_scored)
    qualified = {
        sequence: row
        for sequence, row in initial_scored.items()
        if int(row["passes_strict_probability"])
    }
    beam = select_beam(initial_scored, beam_width, 7)
    latest_trace: List[Dict[str, Any]] = []
    cumulative_expected = {
        "beam_initial_anchors": {
            "generated_unique": len(expected_initial_provenance),
            "newly_scored": len(initial_scored),
            "strict_probability_hits": len(qualified),
            "maximum_probability": max(
                float(row["maximum_probability"]) for row in initial_scored.values()
            ),
        }
    }
    replay_progress = ProgressBar(
        "3ZGC checkpoint provenance replay",
        len(checkpoints),
        unit="round",
    )

    for round_number, checkpoint_path in zip(round_numbers, checkpoints):
        generated_provenance = zgc_round_provenance(
            beam, round_number, offspring_per_round, numpy_module
        )
        expected_new_sequences = set(generated_provenance) - seen
        ledger_path = out_dir / f"3zgc_round_{round_number:02d}.csv.gz"
        if not ledger_path.is_file():
            raise RuntimeError(f"3ZGC resume lacks round-{round_number} ledger")
        round_rows = [
            normalize_search_ledger_row(row) for row in read_gzip_csv(ledger_path)
        ]
        scored = {str(row["sequence"]): row for row in round_rows}
        if (
            len(scored) != len(round_rows)
            or set(scored) & seen
            or set(scored) != expected_new_sequences
        ):
            raise RuntimeError(
                f"3ZGC round-{round_number} ledger is not the exact frozen generated budget"
            )
        for sequence, row in scored.items():
            validate_search_ledger_row(
                row,
                "3ZGC",
                sequence,
                f"beam_round_{round_number:02d}",
                generated_provenance[sequence],
            )
        if validate_model_scores:
            validate_ledger_scores_against_model(
                round_rows,
                "3ZGC",
                f"beam_round_{round_number:02d}",
                score_minimal,
                score_tolerance,
            )
        seen.update(scored)
        for sequence, row in scored.items():
            if int(row["passes_strict_probability"]):
                qualified[sequence] = row
        combined = {str(row["sequence"]): row for row in beam}
        combined.update(scored)
        beam = select_beam(combined, beam_width, 7)
        cumulative_expected[f"beam_round_{round_number:02d}"] = {
            "generated_unique": len(generated_provenance),
            "newly_scored": len(scored),
            "strict_probability_hits": len(qualified),
            "maximum_probability": max(
                (float(row["maximum_probability"]) for row in beam), default=0.0
            ),
        }

        checkpoint = read_gzip_json(checkpoint_path)
        if not (
            checkpoint.get("config_sha256") == config_digest
            and int(checkpoint.get("completed_round", -1)) == round_number
            and set(str(value) for value in checkpoint.get("seen_sequences", [])) == seen
        ):
            raise RuntimeError(f"3ZGC round-{round_number} checkpoint state/hash mismatch")
        checkpoint_qualified = {
            str(row["sequence"]): normalize_search_ledger_row(row)
            for row in checkpoint.get("qualified", [])
        }
        checkpoint_beam = [
            normalize_search_ledger_row(row) for row in checkpoint.get("beam", [])
        ]
        if checkpoint_qualified != qualified or checkpoint_beam != beam:
            raise RuntimeError(
                f"3ZGC round-{round_number} checkpoint is not reconstructible from ledgers"
            )
        trace = checkpoint.get("trace_rows")
        if not isinstance(trace, list):
            raise RuntimeError(f"3ZGC round-{round_number} checkpoint lacks trace evidence")
        zgc_trace = {
            str(row.get("stage", "")): row
            for row in trace
            if str(row.get("target_name", "")).upper() == "3ZGC"
        }
        expected_stages = {
            "beam_initial_anchors",
            *(f"beam_round_{index:02d}" for index in range(1, round_number + 1)),
        }
        if set(zgc_trace) != expected_stages:
            raise RuntimeError(f"3ZGC round-{round_number} trace stage mismatch")
        for stage, expected in cumulative_expected.items():
            observed = zgc_trace[stage]
            if not (
                int(observed.get("generated_unique", -1))
                == expected["generated_unique"]
                and int(observed.get("newly_scored", -1))
                == expected["newly_scored"]
                and int(observed.get("strict_probability_hits", -1))
                == expected["strict_probability_hits"]
                and abs(
                    float(observed.get("maximum_probability", float("nan")))
                    - float(expected["maximum_probability"])
                )
                <= 1e-12
            ):
                raise RuntimeError(f"3ZGC round-{round_number} trace values mismatch")
        latest_trace = [dict(row) for row in trace]
        replay_progress.update(1)

    replay_progress.close()
    return round_numbers[-1], seen, beam, qualified, latest_trace


class MethylScorer:
    def __init__(
        self,
        model: Any,
        device: Any,
        native_index: Mapping[str, Mapping[str, Any]],
        selected_chains: Mapping[str, str],
        batch_size: int,
        torch_module: Any,
        common: Mapping[str, Any],
        reannotator: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.native_index = native_index
        self.selected_chains = selected_chains
        self.batch_size = batch_size
        self.torch = torch_module
        self.common = common
        self.reannotator = reannotator
        self.geometry: Dict[str, Tuple[Any, ...]] = {}

    def _geometry(self, target: str, length: int) -> Tuple[Any, ...]:
        key = f"{target}:{length}"
        if key not in self.geometry:
            chain = self.selected_chains[target]
            record = self.reannotator.peptide_only_record(
                self.native_index[target], chain, "A" * length
            )
            packed = self.common["featurize_records"](
                [record], device=self.device, eval_chains="masked"
            )
            if packed is None:
                raise RuntimeError(f"Peptide-only feature construction failed for {target}")
            tensors, metas = packed
            if int(metas[0]["selected_length"]) != length:
                raise RuntimeError(f"Peptide-only length mismatch for {target}")
            self.geometry[key] = tuple(tensors[:6])
        return self.geometry[key]

    def _representations(self, target: str, sequence_batch: Sequence[str]) -> Mapping[str, Any]:
        length = len(sequence_batch[0])
        if any(len(sequence) != length for sequence in sequence_batch):
            raise ValueError("Methyl scoring batch mixes peptide lengths")
        X, _S, mask, chain_M, residue_idx, chain_encoding = self._geometry(target, length)
        alphabet = self.common["NATURAL_AA_ALPHABET"]
        S_natural = self.torch.tensor(
            [[alphabet.index(token) for token in sequence] for sequence in sequence_batch],
            device=self.device,
            dtype=self.torch.long,
        )
        current = len(sequence_batch)
        with self.torch.no_grad():
            return self.common[
                "cyclic_representation_known_sequence_methyl_probabilities"
            ](
                model=self.model,
                X=X.repeat(current, 1, 1, 1),
                S_natural=S_natural,
                mask=mask.repeat(current, 1),
                chain_M=chain_M.repeat(current, 1),
                residue_idx=residue_idx.repeat(current, 1),
                chain_encoding_all=chain_encoding.repeat(current, 1),
                temperature=TEMPERATURE,
            )

    def score_minimal(
        self, target: str, sequences: Sequence[str], stage: str
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        unique_sequences = sorted(set(sequences))
        progress = ProgressBar(
            f"{target} {stage}", len(unique_sequences), unit="seq"
        )
        for sequence_batch in chunks(unique_sequences, self.batch_size):
            representation = self._representations(target, sequence_batch)
            for row_index, sequence in enumerate(sequence_batch):
                probability = [
                    round(float(value), 8)
                    for value in representation["mean"][row_index].detach().cpu().tolist()
                ]
                if len(probability) != len(sequence) or not all(
                    math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in probability
                ):
                    raise RuntimeError(
                        f"Non-finite/out-of-range methyl probability for {target}:{sequence}"
                    )
                eligible_positions = [
                    index
                    for index, token in enumerate(sequence)
                    if token in METHYLATABLE_AA
                ]
                # The all-Pro sequence is legal in the natural alphabet but has
                # no methylatable site.  Keep it in the complete search ledger
                # without ever allowing it to qualify or crashing max([]).
                if eligible_positions:
                    best_index = max(
                        eligible_positions, key=lambda index: probability[index]
                    )
                    maximum_probability = probability[best_index]
                else:
                    best_index = max(
                        range(len(sequence)), key=lambda index: probability[index]
                    )
                    maximum_probability = 0.0
                result[sequence] = {
                    "target_name": target,
                    "sequence": sequence,
                    "search_stage": stage,
                    "maximum_probability": maximum_probability,
                    "argmax_position_1based": best_index + 1,
                    "argmax_residue": sequence[best_index],
                    "passes_strict_probability": int(
                        bool(eligible_positions)
                        and strict_rounded_pass(maximum_probability)
                    ),
                }
            progress.update(len(sequence_batch))
        progress.close()
        return result

    def score_full(
        self,
        target: str,
        sequences: Sequence[str],
        stage: str = "full annotation",
        *,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        alphabet = self.common["NATURAL_AA_ALPHABET"]
        unique_sequences = sorted(set(sequences))
        progress = (
            ProgressBar(f"{target} {stage}", len(unique_sequences), unit="seq")
            if show_progress
            else None
        )
        for sequence_batch in chunks(unique_sequences, self.batch_size):
            representation = self._representations(target, sequence_batch)
            for row_index, sequence in enumerate(sequence_batch):
                result[sequence] = self.reannotator.annotation_payload(
                    sequence,
                    representation,
                    row_index,
                    THRESHOLD,
                    alphabet,
                    self.common["EXTENDED_AA_ALPHABET"],
                    self.common["NAT_TO_METHYL_ABS"],
                )
            if progress is not None:
                progress.update(len(sequence_batch))
        if progress is not None:
            progress.close()
        return result


class BasePlausibilityScorer:
    """Receptor-conditioned base-head log probability averaged over L orders."""

    def __init__(
        self,
        model: Any,
        device: Any,
        target_records: Mapping[str, Mapping[str, Any]],
        batch_size: int,
        torch_module: Any,
        functional: Any,
        common: Mapping[str, Any],
    ) -> None:
        self.model = model
        self.device = device
        self.target_records = target_records
        self.batch_size = batch_size
        self.torch = torch_module
        self.functional = functional
        self.common = common
        self.features: Dict[str, Tuple[Any, ...]] = {}

    def _features(self, target: str) -> Tuple[Any, ...]:
        if target not in self.features:
            packed = self.common["featurize_records"](
                [self.target_records[target]],
                device=self.device,
                eval_chains="masked",
                max_peptide_len=30,
            )
            if packed is None:
                raise RuntimeError(f"Complex feature construction failed for {target}")
            tensors, _metas = packed
            self.features[target] = tuple(tensors[:6])
        return self.features[target]

    def score(
        self,
        target: str,
        sequences: Sequence[str],
        stage: str = "base plausibility",
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        alphabet = self.common["NATURAL_AA_ALPHABET"]
        unique_sequences = sorted(set(sequences))
        progress = ProgressBar(
            f"{target} {stage}", len(unique_sequences), unit="seq"
        )
        for sequence_batch in chunks(unique_sequences, self.batch_size):
            X, S_true, mask, chain_M, residue_idx, chain_encoding = self._features(target)
            selected = self.torch.nonzero(
                (chain_M[0] * mask[0]) > 0.0, as_tuple=False
            ).squeeze(-1)
            length = int(selected.numel())
            if any(len(sequence) != length for sequence in sequence_batch):
                raise RuntimeError(f"Base plausibility length mismatch for {target}")
            current = len(sequence_batch)
            Xb = X.repeat(current, 1, 1, 1)
            Sb = S_true.repeat(current, 1).clone()
            maskb = mask.repeat(current, 1)
            chainb = chain_M.repeat(current, 1)
            residueb = residue_idx.repeat(current, 1)
            encodingb = chain_encoding.repeat(current, 1)
            natural = self.torch.tensor(
                [[alphabet.index(token) for token in sequence] for sequence in sequence_batch],
                device=self.device,
                dtype=self.torch.long,
            )
            Sb[:, selected] = natural
            total = self.torch.zeros(current, device=self.device)
            with self.torch.no_grad():
                for shift in range(length):
                    requested = selected.roll(shifts=-shift).unsqueeze(0).repeat(current, 1)
                    order = self.common["complete_decoding_order"](
                        chainb, maskb, requested
                    )
                    base_logits, _experts = self.model(
                        Xb,
                        Sb,
                        maskb,
                        chainb,
                        residueb,
                        encodingb,
                        decoding_order=order,
                    )
                    log_probability = self.functional.log_softmax(base_logits, dim=-1)
                    selected_log_probability = log_probability[:, selected].gather(
                        -1, natural.unsqueeze(-1)
                    ).squeeze(-1)
                    total += selected_log_probability.mean(dim=1)
            values = (total / length).detach().cpu().tolist()
            for sequence, value in zip(sequence_batch, values):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise RuntimeError(
                        f"Non-finite base plausibility for {target}:{sequence}"
                    )
                result[sequence] = numeric
            progress.update(len(sequence_batch))
        progress.close()
        return result


def validate_baseline(
    baseline: Path,
    model_path: Path,
    model_manifest_path: Path,
    representation_path: Path,
    plan_path: Path,
    native_path: Path,
    historical_path: Path,
    prior_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, str]], List[Dict[str, str]]]:
    manifest_path = baseline / "generation_manifest.json"
    unique_path = baseline / "unique_candidates.csv"
    target_manifest_path = baseline / "target_manifest.csv"
    summary_path = baseline / "generation_summary_by_target.csv"
    for required in (manifest_path, unique_path, target_manifest_path, summary_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = read_json(manifest_path)
    model_manifest = read_json(model_manifest_path)
    representation = read_json(representation_path)
    model_artifacts = dict(model_manifest.get("artifacts") or {})
    if not (
        model_manifest.get("quality_gate") == "PASS"
        and model_manifest.get("protocol") == V8_EXPERT_PROTOCOL
        and int(model_manifest.get("audit_batch_size", -1)) == 8
        and model_manifest.get("checkpoint_artifact_sha256") == sha256_file(model_path)
        and model_manifest.get("composer_program_sha256") == sha256_file(
            SCRIPT_PATH.with_name("12_compose_source_scoped_hybrid_v8.py")
        )
        and model_manifest.get("trainer_program_sha256") == sha256_file(
            SCRIPT_PATH.with_name("02_retrain_canonical_expert_heads.py")
        )
        and model_manifest.get("common_program_sha256") == sha256_file(COMMON_PATH)
        and model_manifest.get("model_utils_program_sha256")
        == sha256_file(MODEL_UTILS_PATH)
        and model_manifest.get("nmethyl_config_program_sha256")
        == sha256_file(NMETHYL_CONFIG_PATH)
        and artifact_map_matches_exact_paths(
            model_artifacts,
            {
                name: model_manifest_path.parent / filename
                for name, filename in V8_MODEL_ARTIFACT_FILENAMES.items()
            },
        )
    ):
        raise RuntimeError("V8 model is absent, failed, or stale")
    representation_artifacts = dict(representation.get("artifacts") or {})
    representation_best_path = Path(str(representation.get("best_csv", ""))).resolve()
    try:
        representation_best_path.relative_to(REPO_ROOT.resolve())
        representation_best_is_current = (
            representation_best_path.is_file()
            and sha256_file(representation_best_path)
            == str(representation.get("best_csv_sha256", ""))
        )
    except ValueError:
        representation_best_is_current = False
    if not (
        representation.get("quality_gate") == "PASS"
        and representation.get("protocol") == V8_REPRESENTATION_PROTOCOL
        and representation.get("release_authorization")
        == V8_REPRESENTATION_AUTHORIZATION
        and int(representation.get("audit_batch_size", -1)) == 8
        and representation.get("model_sha256") == sha256_file(model_path)
        and representation.get("model_manifest_sha256")
        == sha256_file(model_manifest_path)
        and representation.get("representation_auditor_program_sha256")
        == sha256_file(SCRIPT_PATH.with_name("13_audit_source_scoped_hybrid_v8.py"))
        and representation.get("equivariance_auditor_program_sha256")
        == sha256_file(
            SCRIPT_PATH.with_name("07_audit_cyclic_representation_equivariance.py")
        )
        and representation.get("common_program_sha256") == sha256_file(COMMON_PATH)
        and representation.get("model_utils_program_sha256")
        == sha256_file(MODEL_UTILS_PATH)
        and representation.get("nmethyl_config_program_sha256")
        == sha256_file(NMETHYL_CONFIG_PATH)
        and declared_path_hash_is_current(
            representation, "test_jsonl", "test_jsonl_sha256"
        )
        and declared_path_hash_is_current(
            representation,
            "native_jsonl",
            "native_jsonl_sha256",
            native_path,
        )
        and declared_path_hash_is_current(
            representation, "best_csv", "best_csv_sha256"
        )
        and declared_path_hash_is_current(
            representation, "plan", "plan_sha256", plan_path
        )
        and artifact_map_matches_exact_paths(
            representation_artifacts,
            {
                "frozen_test_positions": representation_path.parent
                / "frozen_test_position_probabilities.csv",
                "length_metrics": representation_path.parent
                / "frozen_test_metrics_by_length.csv",
                "native_probabilities": representation_path.parent
                / "native_target_representation_probabilities.csv",
                "native_summary": representation_path.parent
                / "native_target_representation_summary.csv",
            },
        )
        and representation_best_is_current
    ):
        raise RuntimeError("V8 representation audit is absent, failed, or stale")
    missing = {str(value).upper() for value in manifest.get("targets_without_signature_candidate", [])}
    coverage_check = "every_target_has_at_least_one_novel_methylated_signature_candidate"
    false_checks = {
        name for name, passed in dict(manifest.get("quality_checks") or {}).items() if not passed
    }
    baseline_artifacts = dict(manifest.get("candidate_artifacts") or {})
    if manifest.get("protocol") != V8_BASELINE_PROTOCOL:
        raise RuntimeError("V8 baseline uses the wrong protocol")
    if not (
        manifest.get("model_sha256") == sha256_file(model_path)
        and float(manifest.get("temperature", -1.0)) == TEMPERATURE
        and float(manifest.get("methyl_threshold", -1.0)) == THRESHOLD
        and manifest.get("strict_threshold_operator") == ">"
        and int(manifest.get("scoring_batch_size", -1)) == 8
        and manifest.get("summary_score_label") == "v8"
        and manifest.get("expert_scope") == "residue-source-scoped-hybrid"
        and manifest.get("model_expert_qc_protocol") == V8_EXPERT_PROTOCOL
        and dict(manifest.get("cyclic_representation_heldout_audit") or {}).get(
            "sha256"
        )
        == sha256_file(representation_path)
        and manifest.get("reannotator_program_sha256") == sha256_file(REANNOTATOR_PATH)
        and manifest.get("generator_program_sha256") == sha256_file(GENERATOR_PATH)
        and manifest.get("common_program_sha256") == sha256_file(COMMON_PATH)
        and manifest.get("model_utils_program_sha256") == sha256_file(MODEL_UTILS_PATH)
        and manifest.get("nmethyl_config_program_sha256")
        == sha256_file(NMETHYL_CONFIG_PATH)
        and manifest.get("plan_sha256") == sha256_file(plan_path)
        and manifest.get("native_jsonl_sha256") == sha256_file(native_path)
        and manifest.get("historical_design_csv_sha256")
        == sha256_file(historical_path)
        and manifest.get("prior_handoff_csv_sha256") == sha256_file(prior_path)
        and artifact_map_matches_exact_paths(
            baseline_artifacts,
            {
                "all": baseline / "all_candidates.csv",
                "unique": baseline / "unique_candidates.csv",
                "eligible": baseline / "methylated_new_candidates.csv",
                "target_manifest": baseline / "target_manifest.csv",
                "target_summary": baseline / "generation_summary_by_target.csv",
            },
        )
    ):
        raise RuntimeError("V8 baseline belongs to a different model")
    if missing:
        if not (
            manifest.get("quality_gate") == "FAIL"
            and
            bool(manifest.get("directed_recovery_eligible"))
            and false_checks == {coverage_check}
            and missing <= ALLOWED_RECOVERY_TARGETS
        ):
            raise RuntimeError(
                "Baseline failure is not an isolated recoverable 3WNE/3ZGC coverage state"
            )
    elif manifest.get("quality_gate") != "PASS" or false_checks:
        raise RuntimeError("Baseline has no missing target but is not PASS")
    unique_rows = read_csv(unique_path)
    target_rows = read_csv(target_manifest_path)
    return manifest, unique_rows, target_rows


def top_ranked_sequences(rows: Sequence[Mapping[str, str]], target: str) -> List[str]:
    target_rows = [row for row in rows if str(row["target_name"]).upper() == target]
    if not target_rows:
        raise RuntimeError(f"No V8 baseline sequences are available for {target}")

    def actionable_maximum(row: Mapping[str, str]) -> float:
        sequence = str(row.get("design_natural_seq", "")).upper()
        try:
            raw = json.loads(str(row.get("methyl_probabilities", "")))
            probabilities = [float(value) for value in raw]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Malformed baseline probability vector for {target}:{sequence}"
            ) from exc
        try:
            return actionable_probability_max(sequence, probabilities)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid baseline probability vector for {target}:{sequence}"
            ) from exc

    return [
        str(row["design_natural_seq"]).upper()
        for row in sorted(
            target_rows,
            key=lambda row: (
                -actionable_maximum(row),
                str(row.get("design_natural_seq", "")),
            ),
        )
    ]


def exclusion_keys(
    rows: Sequence[Mapping[str, Any]], target: str
) -> Tuple[set[str], set[str]]:
    natural: set[str] = set()
    cyclic: set[str] = set()
    for row in rows:
        if str(row.get("target_name", "")).upper() != target:
            continue
        sequence = str(
            row.get("design_natural_seq") or row.get("design_seq") or ""
        ).upper()
        if sequence:
            natural.add(sequence)
            cyclic.add(forward_cyclic_identity(sequence))
    return natural, cyclic


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("V8 directed recovery requires NumPy and PyTorch") from exc
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = Path(args.model_path).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    representation_path = Path(args.representation_audit).resolve()
    baseline = Path(args.baseline_run_dir).resolve()
    plan_path = Path(args.plan).resolve()
    native_path = Path(args.native_jsonl).resolve()
    historical_path = Path(args.historical_designs_csv).resolve()
    prior_path = Path(args.prior_handoff_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    immutable_inputs = (
        model_path,
        model_manifest_path,
        representation_path,
        baseline,
        plan_path,
        native_path,
        historical_path,
        prior_path,
        SCRIPT_PATH,
        REANNOTATOR_PATH,
        GENERATOR_PATH,
        COMMON_PATH,
        MODEL_UTILS_PATH,
        NMETHYL_CONFIG_PATH,
    )
    overlapping = [path for path in immutable_inputs if paths_overlap(out_dir, path)]
    if overlapping:
        raise ValueError(
            "Directed-search output overlaps an immutable input: "
            + ", ".join(str(path) for path in overlapping)
        )
    for required in (
        model_path,
        model_manifest_path,
        representation_path,
        plan_path,
        native_path,
        historical_path,
        prior_path,
        REANNOTATOR_PATH,
        GENERATOR_PATH,
        COMMON_PATH,
        MODEL_UTILS_PATH,
        NMETHYL_CONFIG_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    baseline_manifest, baseline_unique, baseline_target_rows = validate_baseline(
        baseline,
        model_path,
        model_manifest_path,
        representation_path,
        plan_path,
        native_path,
        historical_path,
        prior_path,
    )
    missing_targets = [
        str(value).upper()
        for value in baseline_manifest.get("targets_without_signature_candidate", [])
    ]
    selected_chains = {
        str(row["target_name"]).upper(): str(row["selected_chain"])
        for row in baseline_target_rows
    }
    if not ALLOWED_RECOVERY_TARGETS <= set(selected_chains):
        raise RuntimeError("Baseline selected-chain map lacks 3WNE/3ZGC")

    input_hashes = {
        "model": sha256_file(model_path),
        "model_manifest": sha256_file(model_manifest_path),
        "representation_audit": sha256_file(representation_path),
        "baseline_manifest": sha256_file(baseline / "generation_manifest.json"),
        "baseline_all": sha256_file(baseline / "all_candidates.csv"),
        "baseline_unique": sha256_file(baseline / "unique_candidates.csv"),
        "baseline_eligible": sha256_file(baseline / "methylated_new_candidates.csv"),
        "plan": sha256_file(plan_path),
        "native": sha256_file(native_path),
        "historical": sha256_file(historical_path),
        "prior": sha256_file(prior_path),
        "search_program": sha256_file(SCRIPT_PATH),
        "reannotator_program": sha256_file(REANNOTATOR_PATH),
        "generator_program": sha256_file(GENERATOR_PATH),
        "common_program": sha256_file(COMMON_PATH),
        "model_utils_program": sha256_file(MODEL_UTILS_PATH),
        "nmethyl_config_program": sha256_file(NMETHYL_CONFIG_PATH),
    }
    portable_import: Optional[Dict[str, Any]] = None
    portable_import_path: Optional[Path] = None
    if str(args.portable_resume_manifest).strip():
        portable_import_path = Path(args.portable_resume_manifest).resolve()
        portable_import = validate_portable_resume_import(
            portable_import_path,
            out_dir,
            model_path,
            model_manifest_path,
            representation_path,
            baseline,
        )
    config = {
        "protocol": V8_SEARCH_PROTOCOL,
        "input_hashes": input_hashes,
        "missing_targets": sorted(missing_targets),
        "temperature": TEMPERATURE,
        "threshold": THRESHOLD,
        "strict_operator": ">",
        "alphabet": NATURAL_AA,
        "3wne_radius": int(args.wne_radius),
        "3zgc_rounds": int(args.zgc_rounds),
        "3zgc_beam_width": int(args.zgc_beam_width),
        "3zgc_offspring_per_round": int(args.zgc_offspring_per_round),
        "methyl_batch_size": int(args.batch_size),
        "base_plausibility_batch_size": int(args.base_batch_size),
        "maximum_released_candidates_per_target": int(args.max_release_per_target),
        "requested_device": str(args.device),
        "allow_cpu": bool(args.allow_cpu),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
        ),
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
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
        "base_plausibility_percentile": BASE_PERCENTILE,
        "batch_one_rescore_tolerance": RESCORE_TOLERANCE,
        "probability_persistence_decimal_places": 8,
        "probability_rounding_implementation": "Python round(value, 8)",
        "full_budget_no_early_stop": True,
        "resume_mode": (
            "HASH_PINNED_CROSS_RUNTIME_EVIDENCE_IMPORT_WITH_DESTINATION_REAUDIT"
            if portable_import is not None
            else "SAME_RUNTIME_LEDGER_MODEL_REPLAY"
        ),
        "portable_resume_import_sha256": (
            sha256_file(portable_import_path)
            if portable_import_path is not None
            else None
        ),
    }
    config_digest = stable_json_sha256(config)
    existing_manifest = out_dir / "directed_search_manifest.json"
    if existing_manifest.is_file():
        existing = read_json(existing_manifest)
        if existing.get("config_sha256") != config_digest:
            raise RuntimeError("Existing directed search belongs to a different configuration")
        if existing.get("quality_gate") == "PASS":
            existing_artifacts = dict(existing.get("artifacts") or {})
            if not search_artifacts_match_exact(
                existing_artifacts,
                out_dir,
                missing_targets,
                int(args.zgc_rounds),
            ):
                raise RuntimeError("Passed directed-search artifacts are absent or stale")
            print("Directed search: reused passed V8 result", flush=True)
            return
        if not args.resume:
            raise FileExistsError("Failed/partial directed search exists; pass --resume")
    elif out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError("Directed search output exists; pass --resume after inspection")
    out_dir.mkdir(parents=True, exist_ok=True)

    reannotator = load_module("source_scoped_v8_reannotator", REANNOTATOR_PATH)
    generator = load_module("source_scoped_v8_generator", GENERATOR_PATH)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from paper_clean_v28.clean_v28_common import (  # pylint: disable=import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        NAT_TO_METHYL_ABS,
        NATURAL_AA_ALPHABET,
        complete_decoding_order,
        cyclic_representation_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
    )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU search requires --allow-cpu")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise RuntimeError("No CUDA device is available; pass --allow-cpu knowingly")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    native_rows = generator.read_jsonl(native_path)
    native_index = {
        generator.record_name(row, index): row for index, row in enumerate(native_rows)
    }
    model = load_v28_model(str(model_path), device)
    model.eval()
    common = {
        "EXTENDED_AA_ALPHABET": EXTENDED_AA_ALPHABET,
        "NAT_TO_METHYL_ABS": NAT_TO_METHYL_ABS,
        "NATURAL_AA_ALPHABET": NATURAL_AA_ALPHABET,
        "complete_decoding_order": complete_decoding_order,
        "cyclic_representation_known_sequence_methyl_probabilities": (
            cyclic_representation_known_sequence_methyl_probabilities
        ),
        "featurize_records": featurize_records,
    }
    methyl_scorer = MethylScorer(
        model,
        device,
        native_index,
        selected_chains,
        int(args.batch_size),
        torch,
        common,
        reannotator,
    )
    target_records, _target_manifest = generator.prepare_target_records(
        native_rows, selected_chains, sorted(ALLOWED_RECOVERY_TARGETS)
    )
    base_scorer = BasePlausibilityScorer(
        model,
        device,
        target_records,
        int(args.base_batch_size),
        torch,
        functional,
        common,
    )

    control_rows: List[Dict[str, Any]] = []
    control_sequences_by_target: Dict[str, List[str]] = {}
    for target in sorted(ALLOWED_RECOVERY_TARGETS):
        selected_chain = selected_chains[target]
        source_record = native_index[target]
        actual_native = str(source_record.get(f"seq_chain_{selected_chain}", "")).upper()
        coordinate_lengths = {
            atom: len(source_record.get(f"{atom}_chain_{selected_chain}", []))
            for atom in ("N", "CA", "C", "O")
        }
        expected_length = len(NATIVE_CONTROLS[target])
        if actual_native != NATIVE_CONTROLS[target] or any(
            length != expected_length for length in coordinate_lengths.values()
        ):
            raise RuntimeError(
                f"{target} selected-chain native sequence/geometry changed: "
                f"{actual_native}, {coordinate_lengths}"
            )
        control_sequences = [
            HISTORICAL_CONTROLS[target]["sequence"],
            NATIVE_CONTROLS[target],
        ]
        control_sequences_by_target[target] = control_sequences
        full = methyl_scorer.score_full(target, control_sequences)
        for control_type, sequence in zip(("withdrawn_historical", "native"), control_sequences):
            payload = full[sequence]
            probabilities = [
                float(value)
                for value in json.loads(str(payload["methyl_probabilities"]))
            ]
            actionable_maximum = actionable_probability_max(sequence, probabilities)
            old_site = (
                int(HISTORICAL_CONTROLS[target]["old_site"])
                if control_type == "withdrawn_historical"
                else None
            )
            control_rows.append(
                {
                    "target_name": target,
                    "selected_chain": selected_chain,
                    "control_type": control_type,
                    "natural_sequence": sequence,
                    "withdrawn_old_design": (
                        HISTORICAL_CONTROLS[target]["old_design"]
                        if control_type == "withdrawn_historical"
                        else ""
                    ),
                    "length": len(sequence),
                    "native_selected_chain_sequence": actual_native,
                    "native_atom_coordinate_lengths": json.dumps(
                        coordinate_lengths, sort_keys=True
                    ),
                    "predicted_design_seq": payload["design_seq"],
                    "predicted_methyl_positions_1based": payload[
                        "methyl_positions_1based"
                    ],
                    "maximum_probability": actionable_maximum,
                    "old_methyl_position_1based": old_site if old_site is not None else "",
                    "old_site_probability": (
                        probabilities[old_site - 1] if old_site is not None else ""
                    ),
                    "methyl_probabilities": payload["methyl_probabilities"],
                    "representation_min": payload[
                        "methyl_probability_representation_min"
                    ],
                    "representation_max": payload[
                        "methyl_probability_representation_max"
                    ],
                    "representation_span": payload[
                        "methyl_probability_representation_span"
                    ],
                    "decoder_order_ensemble_size": payload[
                        "annotation_decoder_order_ensemble_size"
                    ],
                    "representation_ensemble_size": payload[
                        "annotation_representation_ensemble_size"
                    ],
                    "release_eligibility": "CONTROL_ONLY_NEVER_RELEASE",
                }
            )
    atomic_write_csv(
        out_dir / "mandatory_length_6_7_controls.csv",
        control_rows,
        list(control_rows[0]),
    )

    trace_rows: List[Dict[str, Any]] = []
    trace_path = out_dir / "search_trace_by_round.csv"
    if args.resume and trace_path.is_file():
        trace_rows = [dict(row) for row in read_csv(trace_path)]
    all_qualified: Dict[str, Dict[str, Dict[str, Any]]] = {
        target: {} for target in ALLOWED_RECOVERY_TARGETS
    }
    evaluated_counts: Counter[str] = Counter()
    evaluated_sequences: Dict[str, set[str]] = {
        target: set() for target in ALLOWED_RECOVERY_TARGETS
    }

    if "3WNE" in missing_targets:
        ranked = top_ranked_sequences(baseline_unique, "3WNE")
        anchors = [
            (ranked[0], "current_v8_baseline_top"),
            (HISTORICAL_CONTROLS["3WNE"]["sequence"], "withdrawn_historical_control"),
            (NATIVE_CONTROLS["3WNE"], "native_control"),
        ]
        search_provenance = wne_search_provenance(
            anchors, int(args.wne_radius)
        )
        search_sequences = sorted(search_provenance)
        wne_ledger_path = out_dir / "3wne_exact_search_all.csv.gz"
        if portable_import is not None:
            if not wne_ledger_path.is_file():
                raise RuntimeError("Portable resume lacks the 3WNE source ledger")
            ledger = [
                normalize_search_ledger_row(row)
                for row in read_gzip_csv(wne_ledger_path)
            ]
            scored = {str(row["sequence"]): row for row in ledger}
            if len(scored) != len(ledger) or set(scored) != set(search_sequences):
                raise RuntimeError(
                    "Portable 3WNE ledger is not the exact frozen radius-2 budget"
                )
        else:
            scored = methyl_scorer.score_minimal(
                "3WNE", search_sequences, "exact_radius_2"
            )
        for sequence, row in scored.items():
            if portable_import is None:
                row.update(search_provenance[sequence])
            validate_search_ledger_row(
                row,
                "3WNE",
                sequence,
                "exact_radius_2",
                search_provenance[sequence],
            )
        ledger = list(scored.values())
        if portable_import is None:
            atomic_write_gzip_csv(
                wne_ledger_path,
                ledger,
                list(ledger[0]) if ledger else SEARCH_LEDGER_FIELDS,
            )
        evaluated_counts["3WNE"] = len(scored)
        evaluated_sequences["3WNE"] = set(scored)
        for sequence, row in scored.items():
            if int(row["passes_strict_probability"]):
                all_qualified["3WNE"][sequence] = row
        trace_rows.append(
            {
                "target_name": "3WNE",
                "stage": "exact_radius_2",
                "generated_unique": len(search_sequences),
                "newly_scored": len(scored),
                "strict_probability_hits": len(all_qualified["3WNE"]),
                "maximum_probability": max(
                    float(row["maximum_probability"]) for row in scored.values()
                ),
            }
        )
        atomic_write_csv(trace_path, trace_rows, list(trace_rows[0]))

    if "3ZGC" in missing_targets:
        ranked = top_ranked_sequences(baseline_unique, "3ZGC")
        initial = [
            HISTORICAL_CONTROLS["3ZGC"]["sequence"],
            NATIVE_CONTROLS["3ZGC"],
            ranked[0],
        ]
        anchors = select_diverse_sequences(ranked[:128], 34, initial=initial)
        initial_provenance = zgc_initial_anchor_provenance(anchors, ranked[0])
        checkpoint_dir = out_dir / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("3zgc_round_*.json.gz"))
        start_round = 1
        seen: set[str]
        beam: List[Dict[str, Any]]
        if checkpoints and args.resume:
            checkpoint_config_digest = (
                str(portable_import["source_config_sha256"])
                if portable_import is not None
                else config_digest
            )
            (
                completed_round,
                seen,
                beam,
                reconstructed_qualified,
                reconstructed_trace,
            ) = reconstruct_and_validate_zgc_resume(
                out_dir,
                checkpoints,
                checkpoint_config_digest,
                int(args.zgc_beam_width),
                initial_provenance,
                int(args.zgc_offspring_per_round),
                np,
                methyl_scorer.score_minimal,
                validate_model_scores=portable_import is None,
            )
            if completed_round > int(args.zgc_rounds):
                raise RuntimeError("3ZGC checkpoint exceeds the frozen round budget")
            start_round = completed_round + 1
            evaluated_sequences["3ZGC"] = set(seen)
            all_qualified["3ZGC"] = reconstructed_qualified
            trace_rows = reconstructed_trace
            evaluated_counts["3ZGC"] = len(seen)
        else:
            initial_scored = methyl_scorer.score_minimal(
                "3ZGC", anchors, "beam_initial_anchors"
            )
            for sequence, row in initial_scored.items():
                row.update(initial_provenance[sequence])
                validate_search_ledger_row(
                    row,
                    "3ZGC",
                    sequence,
                    "beam_initial_anchors",
                    initial_provenance[sequence],
                )
            seen = set(initial_scored)
            evaluated_sequences["3ZGC"] = set(seen)
            beam = select_beam(initial_scored, int(args.zgc_beam_width), 7)
            for sequence, row in initial_scored.items():
                if int(row["passes_strict_probability"]):
                    all_qualified["3ZGC"][sequence] = row
            evaluated_counts["3ZGC"] = len(seen)
            initial_ledger = list(initial_scored.values())
            atomic_write_gzip_csv(
                out_dir / "3zgc_round_00_initial.csv.gz",
                initial_ledger,
                list(initial_ledger[0]),
            )
            trace_rows.append(
                {
                    "target_name": "3ZGC",
                    "stage": "beam_initial_anchors",
                    "generated_unique": len(anchors),
                    "newly_scored": len(initial_scored),
                    "strict_probability_hits": len(all_qualified["3ZGC"]),
                    "maximum_probability": max(
                        float(row["maximum_probability"])
                        for row in initial_scored.values()
                    ),
                }
            )
            atomic_write_csv(trace_path, trace_rows, list(trace_rows[0]))

        for round_index in range(start_round, int(args.zgc_rounds) + 1):
            generated_provenance = zgc_round_provenance(
                beam,
                round_index,
                int(args.zgc_offspring_per_round),
                np,
            )
            generated = set(generated_provenance)
            to_score = sorted(generated - seen)
            scored = methyl_scorer.score_minimal(
                "3ZGC", to_score, f"beam_round_{round_index:02d}"
            )
            for sequence, row in scored.items():
                row.update(generated_provenance[sequence])
                validate_search_ledger_row(
                    row,
                    "3ZGC",
                    sequence,
                    f"beam_round_{round_index:02d}",
                    generated_provenance[sequence],
                )
            seen.update(scored)
            evaluated_sequences["3ZGC"] = set(seen)
            evaluated_counts["3ZGC"] = len(seen)
            for sequence, row in scored.items():
                if int(row["passes_strict_probability"]):
                    all_qualified["3ZGC"][sequence] = row
            combined = {str(row["sequence"]): row for row in beam}
            combined.update(scored)
            beam = select_beam(combined, int(args.zgc_beam_width), 7)
            ledger = list(scored.values())
            atomic_write_gzip_csv(
                out_dir / f"3zgc_round_{round_index:02d}.csv.gz",
                ledger,
                list(ledger[0]) if ledger else SEARCH_LEDGER_FIELDS,
            )
            maximum = max(
                [float(row["maximum_probability"]) for row in beam], default=0.0
            )
            trace_rows.append(
                {
                    "target_name": "3ZGC",
                    "stage": f"beam_round_{round_index:02d}",
                    "generated_unique": len(generated),
                    "newly_scored": len(scored),
                    "strict_probability_hits": len(all_qualified["3ZGC"]),
                    "maximum_probability": maximum,
                }
            )
            atomic_write_csv(trace_path, trace_rows, list(trace_rows[0]))
            write_gzip_json(
                checkpoint_dir / f"3zgc_round_{round_index:02d}.json.gz",
                {
                    "config_sha256": config_digest,
                    "completed_round": round_index,
                    "seen_sequences": sorted(seen),
                    "beam": beam,
                    "qualified": list(all_qualified["3ZGC"].values()),
                    "trace_rows": trace_rows,
                },
            )

    historical_rows = read_csv(historical_path)
    prior_rows = read_csv(prior_path)
    release_rows: List[Dict[str, Any]] = []
    plausibility_rows: List[Dict[str, Any]] = []
    for target in missing_targets:
        target_pool_sequences = sorted(
            {
                str(row["design_natural_seq"]).upper()
                for row in baseline_unique
                if str(row["target_name"]).upper() == target
            }
        )
        pool_base = base_scorer.score(
            target,
            target_pool_sequences,
            stage="baseline plausibility floor",
        )
        floor = nearest_rank_percentile(list(pool_base.values()), BASE_PERCENTILE)
        qualified_sequences = sorted(all_qualified[target])
        candidate_base = (
            base_scorer.score(
                target,
                qualified_sequences,
                stage="strict-hit plausibility",
            )
            if qualified_sequences
            else {}
        )
        historical_natural, historical_cyclic = exclusion_keys(historical_rows, target)
        prior_natural, prior_cyclic = exclusion_keys(prior_rows, target)
        pool_natural, pool_cyclic = exclusion_keys(baseline_unique, target)
        native_natural = {NATIVE_CONTROLS[target]}
        native_cyclic = {forward_cyclic_identity(value) for value in native_natural}
        full_payload = (
            methyl_scorer.score_full(
                target,
                qualified_sequences,
                stage="strict-hit full annotation",
            )
            if qualified_sequences
            else {}
        )
        accepted_cyclic: set[str] = set()
        batch_one_scorer = MethylScorer(
            model,
            device,
            native_index,
            selected_chains,
            1,
            torch,
            common,
            reannotator,
        )
        ordered = sorted(
            qualified_sequences,
            key=lambda sequence: (
                -float(all_qualified[target][sequence]["maximum_probability"]),
                -float(candidate_base[sequence]),
                sequence,
            ),
        )
        batch_one_progress = ProgressBar(
            f"{target} eligibility + batch-one audit",
            len(ordered),
            unit="candidate",
        )
        for sequence in ordered:
            batch_one_progress.update(1)
            search_row = all_qualified[target][sequence]
            search_maximum = float(search_row["maximum_probability"])
            if not strict_rounded_pass(search_maximum):
                raise RuntimeError(
                    f"Qualified search row is not a finite strict hit: {target}:{sequence}"
                )
            cyclic_key = forward_cyclic_identity(sequence)
            duplicate_reason = ""
            if sequence in historical_natural or cyclic_key in historical_cyclic:
                duplicate_reason = "historical_4115"
            elif sequence in prior_natural or cyclic_key in prior_cyclic:
                duplicate_reason = "prior_1333"
            elif sequence in pool_natural or cyclic_key in pool_cyclic:
                duplicate_reason = "current_31500_pool"
            elif sequence in native_natural or cyclic_key in native_cyclic:
                duplicate_reason = "native"
            elif cyclic_key in accepted_cyclic:
                duplicate_reason = "accepted_forward_cyclic_equivalent"
            base_pass = float(candidate_base[sequence]) >= floor
            payload = full_payload[sequence]
            full_probabilities = [
                float(value)
                for value in json.loads(str(payload["methyl_probabilities"]))
            ]
            full_max = actionable_probability_max(sequence, full_probabilities)
            qualified_full_difference = abs(full_max - search_maximum)
            if not all(
                math.isfinite(value)
                for value in (
                    float(candidate_base[sequence]),
                    float(floor),
                    full_max,
                    qualified_full_difference,
                )
            ):
                raise RuntimeError(
                    f"Non-finite candidate evidence for {target}:{sequence}"
                )
            persisted_pass = (
                int(payload["design_methyl_count"]) > 0
                and strict_rounded_pass(full_max)
            )
            eligibility = (
                not duplicate_reason
                and base_pass
                and persisted_pass
                and qualified_full_difference <= RESCORE_TOLERANCE
            )
            plausibility_rows.append(
                {
                    "target_name": target,
                    "sequence": sequence,
                    "search_stage": search_row["search_stage"],
                    "search_maximum_probability": search_maximum,
                    "qualified_full_maximum_probability": full_max,
                    "qualified_full_rescore_absolute_difference": (
                        qualified_full_difference
                    ),
                    "base_log_probability_mean_all_orders": candidate_base[sequence],
                    "base_plausibility_floor_1pct": floor,
                    "passes_base_plausibility": int(base_pass),
                    "duplicate_reason": duplicate_reason,
                    "pre_rescore_release_eligible": int(eligibility),
                }
            )
            if not eligibility or sum(row["target_name"] == target for row in release_rows) >= int(args.max_release_per_target):
                continue
            # Independent batch-one scoring is deliberately separate from the
            # search batches and is the final probability gate.
            independent = batch_one_scorer.score_full(
                target,
                [sequence],
                stage="batch-one release audit",
                show_progress=False,
            )[sequence]
            independent_probabilities = [
                float(value)
                for value in json.loads(str(independent["methyl_probabilities"]))
            ]
            independent_max = actionable_probability_max(
                sequence, independent_probabilities
            )
            difference = abs(independent_max - search_maximum)
            if not (
                math.isfinite(independent_max)
                and math.isfinite(difference)
                and int(independent["design_methyl_count"]) > 0
                and strict_rounded_pass(independent_max)
                and difference <= RESCORE_TOLERANCE
            ):
                continue
            accepted_cyclic.add(cyclic_key)
            release_rows.append(
                {
                    "candidate_id": f"v8dir_{target.lower()}_{len(release_rows)+1:04d}",
                    "target_name": target,
                    "selected_chain": selected_chains[target],
                    "design_seq": independent["design_seq"],
                    "design_natural_seq": sequence,
                    "native_seq": NATIVE_CONTROLS[target],
                    "native_length": len(sequence),
                    "design_length": len(sequence),
                    "length_match": 1,
                    "valid_token_gate": 1,
                    "temperature": TEMPERATURE,
                    "methyl_threshold": THRESHOLD,
                    "strict_threshold_operator": ">",
                    "candidate_origin": "DETERMINISTIC_DIRECTED_SEARCH",
                    "search_stage": search_row["search_stage"],
                    "search_maximum_probability": search_maximum,
                    "qualified_full_maximum_probability": full_max,
                    "qualified_full_rescore_absolute_difference": (
                        qualified_full_difference
                    ),
                    "batch_one_maximum_probability": independent_max,
                    "batch_rescore_absolute_difference": difference,
                    "base_log_probability_mean": "",
                    "base_log_probability_mean_all_orders": candidate_base[sequence],
                    "base_plausibility_context_policy": (
                        "native_complex_longest_receptor_visible_all_peptide_decoder_orders_mean"
                    ),
                    "base_plausibility_floor_1pct": floor,
                    "forward_cyclic_identity": cyclic_key,
                    "control_or_candidate": "NOVEL_RECOVERY_CANDIDATE",
                    **independent,
                    "sampling_context_policy": (
                        "DETERMINISTIC_DIRECTED_SEARCH_NO_AUTOREGRESSIVE_SAMPLING"
                    ),
                    "sampling_path_annotation_status": "NOT_APPLICABLE_DIRECTED_SEARCH",
                    "sampling_path_methyl_probabilities": "",
                    "decoding_order_absolute": "",
                    "seed": "",
                    "draw_index_within_seed": "",
                    "occurrence_count": 1,
                    "seeds_observed": "DIRECTED_SEARCH_NOT_SAMPLED",
                    "seen_in_historical_4115_exact": 0,
                    "seen_in_historical_4115_naturalized": 0,
                    "seen_in_historical_4115": 0,
                    "seen_in_prior_1333_exact": 0,
                    "seen_in_prior_1333_naturalized": 0,
                    "seen_in_prior_1333": 0,
                    "passes_methylation_hard_gate": 1,
                    "eligible_for_new_permeability_screen": 0,
                    "permeability_screen_authorized_in_this_release": 0,
                    "permeability_eligibility_status": (
                        "DEFERRED_PENDING_GLOBAL_AND_CYCLIC_RMSD_LT_3A"
                    ),
                    "eligible_for_manual_structure_review": 1,
                    "permeability_id": "",
                }
            )
        batch_one_progress.close()

    atomic_write_csv(
        out_dir / "qualified_candidate_plausibility_and_novelty.csv",
        plausibility_rows,
        list(plausibility_rows[0]) if plausibility_rows else [
            "target_name",
            "sequence",
            "search_stage",
            "search_maximum_probability",
            "qualified_full_maximum_probability",
            "qualified_full_rescore_absolute_difference",
            "base_log_probability_mean_all_orders",
            "base_plausibility_floor_1pct",
            "passes_base_plausibility",
            "duplicate_reason",
            "pre_rescore_release_eligible",
        ],
    )
    atomic_write_csv(
        out_dir / "directed_candidates.csv",
        release_rows,
        list(release_rows[0]) if release_rows else [
            "candidate_id",
            "target_name",
            "design_seq",
            "design_natural_seq",
        ],
    )
    trace_rows = list(
        {
            (str(row["target_name"]), str(row["stage"])): row
            for row in trace_rows
        }.values()
    )
    trace_rows.sort(key=lambda row: (str(row["target_name"]), str(row["stage"])))
    atomic_write_csv(
        trace_path,
        trace_rows,
        list(trace_rows[0]) if trace_rows else [
            "target_name",
            "stage",
            "generated_unique",
            "newly_scored",
            "strict_probability_hits",
            "maximum_probability",
        ],
    )

    expected_ledger_paths: List[Path] = []
    expected_checkpoint_paths: List[Path] = []
    if "3WNE" in missing_targets:
        expected_ledger_paths.append(out_dir / "3wne_exact_search_all.csv.gz")
    if "3ZGC" in missing_targets:
        expected_ledger_paths.extend(
            [out_dir / "3zgc_round_00_initial.csv.gz"]
            + [
                out_dir / f"3zgc_round_{round_index:02d}.csv.gz"
                for round_index in range(1, int(args.zgc_rounds) + 1)
            ]
        )
        expected_checkpoint_paths.extend(
            out_dir / "checkpoints" / f"3zgc_round_{round_index:02d}.json.gz"
            for round_index in range(1, int(args.zgc_rounds) + 1)
        )
    missing_evidence = [
        str(path)
        for path in [*expected_ledger_paths, *expected_checkpoint_paths]
        if not path.is_file()
    ]
    unexpected_evidence = [
        str(path)
        for path in [
            *sorted(out_dir.glob("*.csv.gz")),
            *sorted((out_dir / "checkpoints").glob("*.json.gz")),
        ]
        if path not in {*expected_ledger_paths, *expected_checkpoint_paths}
    ]
    if missing_evidence or unexpected_evidence:
        raise RuntimeError(
            "Fixed-budget search evidence inventory mismatch; missing="
            + ", ".join(missing_evidence)
            + "; unexpected="
            + ", ".join(unexpected_evidence)
        )
    ledger_sequences: Dict[str, set[str]] = {
        target: set() for target in missing_targets
    }
    ledger_strict_hits: Dict[str, set[str]] = {
        target: set() for target in missing_targets
    }
    duplicate_ledger_sequences: List[str] = []
    for ledger_path in expected_ledger_paths:
        for row in read_gzip_csv(ledger_path):
            target = str(row.get("target_name", "")).upper()
            sequence = str(row.get("sequence", "")).upper()
            if target not in ledger_sequences or not sequence:
                raise RuntimeError(f"Malformed search ledger row in {ledger_path}")
            if sequence in ledger_sequences[target]:
                duplicate_ledger_sequences.append(f"{target}:{sequence}")
            ledger_sequences[target].add(sequence)
            if int(row.get("passes_strict_probability", 0)) == 1:
                ledger_strict_hits[target].add(sequence)
    ledger_union_matches_evaluated = all(
        ledger_sequences[target] == evaluated_sequences[target]
        for target in missing_targets
    )
    ledger_strict_hits_match_qualified = all(
        ledger_strict_hits[target] == set(all_qualified[target])
        for target in missing_targets
    )

    release_counts = Counter(str(row["target_name"]) for row in release_rows)
    missing_after_search = [target for target in missing_targets if release_counts[target] < 1]
    control_checks = {
        "both_historical_and_native_length_controls_scored": len(control_rows) == 4,
        "control_identities_selected_chains_and_native_geometry_are_exact": (
            {
                (
                    str(row["target_name"]),
                    str(row["control_type"]),
                    str(row["natural_sequence"]),
                    str(row["selected_chain"]),
                )
                for row in control_rows
            }
            == {
                ("3WNE", "withdrawn_historical", "GRKWNC", "C"),
                ("3WNE", "native", "PKIDNG", "C"),
                ("3ZGC", "withdrawn_historical", "REGGQNR", "C"),
                ("3ZGC", "native", "GDEETGE", "C"),
            }
            and all(
                set(json.loads(str(row["native_atom_coordinate_lengths"])).values())
                == {int(row["length"])}
                for row in control_rows
            )
        ),
        "control_lengths_are_exactly_6_and_7": all(
            int(row["length"]) == (6 if row["target_name"] == "3WNE" else 7)
            for row in control_rows
        ),
        "control_vectors_and_ensembles_match_length": all(
            len(json.loads(str(row["methyl_probabilities"]))) == int(row["length"])
            and len(json.loads(str(row["representation_min"]))) == int(row["length"])
            and len(json.loads(str(row["representation_max"]))) == int(row["length"])
            and len(json.loads(str(row["representation_span"]))) == int(row["length"])
            and int(row["decoder_order_ensemble_size"]) == int(row["length"])
            and int(row["representation_ensemble_size"]) == int(row["length"])
            for row in control_rows
        ),
        "controls_are_never_release_eligible": all(
            row["release_eligibility"] == "CONTROL_ONLY_NEVER_RELEASE"
            for row in control_rows
        ),
    }
    quality_checks = {
        "model_representation_and_baseline_hashes_are_pinned": True,
        "baseline_failure_is_only_allowed_target_coverage": True,
        **control_checks,
        "search_is_restricted_to_missing_3wne_or_3zgc": (
            set(missing_targets) <= ALLOWED_RECOVERY_TARGETS
        ),
        "complete_search_ledgers_and_checkpoints_are_persisted": (
            not missing_evidence
            and not unexpected_evidence
            and not duplicate_ledger_sequences
            and ledger_union_matches_evaluated
            and ledger_strict_hits_match_qualified
        ),
        "full_frozen_budget_ran_without_early_stop": (
            "3ZGC" not in missing_targets
            or {
                str(row["stage"])
                for row in trace_rows
                if str(row["target_name"]) == "3ZGC"
            }
            >= {
                "beam_initial_anchors",
                *(
                f"beam_round_{round_index:02d}"
                for round_index in range(1, int(args.zgc_rounds) + 1)
                ),
            }
        ),
        "all_missing_targets_have_a_novel_plausible_strict_candidate": (
            not missing_after_search
        ),
        "released_candidates_pass_batch_one_rescore": all(
            strict_rounded_pass(float(row["batch_one_maximum_probability"]))
            and float(row["batch_rescore_absolute_difference"]) <= RESCORE_TOLERANCE
            for row in release_rows
        ),
        "no_formal_abstention_or_threshold_change": True,
        "portable_resume_source_is_hash_pinned_and_destination_reaudit_required": (
            portable_import is None
            or portable_import.get("quality_gate") == "PASS"
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    artifacts: Dict[str, Any] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in {
            "controls": out_dir / "mandatory_length_6_7_controls.csv",
            "plausibility": out_dir / "qualified_candidate_plausibility_and_novelty.csv",
            "directed_candidates": out_dir / "directed_candidates.csv",
            "trace": out_dir / "search_trace_by_round.csv",
        }.items()
    }
    if expected_ledger_paths:
        artifacts["search_ledgers"] = {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in expected_ledger_paths
        }
    if expected_checkpoint_paths:
        artifacts["checkpoints"] = {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in expected_checkpoint_paths
        }
    manifest = {
        "quality_gate": quality_gate,
        "release_status": (
            "READY_FOR_RECOVERY_OVERLAY_AUDIT_NO_STRUCTURE_HANDOFF"
            if quality_gate == "PASS"
            else "BLOCKED_FIXED_SEARCH_BUDGET_DID_NOT_RECOVER_ALL_TARGETS"
        ),
        "protocol": V8_SEARCH_PROTOCOL,
        "config": config,
        "config_sha256": config_digest,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "model_sha256": sha256_file(model_path),
        "baseline_manifest_sha256": input_hashes["baseline_manifest"],
        "temperature": TEMPERATURE,
        "methyl_threshold": THRESHOLD,
        "strict_threshold_operator": ">",
        "missing_targets_before_search": missing_targets,
        "missing_targets_after_search": missing_after_search,
        "evaluated_sequence_counts": dict(sorted(evaluated_counts.items())),
        "evaluated_sequence_sha256_by_target": {
            target: hashlib.sha256(
                ("\n".join(sorted(evaluated_sequences[target])) + "\n").encode("ascii")
            ).hexdigest()
            for target in sorted(missing_targets)
        },
        "strict_probability_hit_counts": {
            target: len(all_qualified[target]) for target in sorted(missing_targets)
        },
        "released_candidate_counts": dict(sorted(release_counts.items())),
        "released_candidates": len(release_rows),
        "structure_status": "NOT_PREDICTED_REQUIRES_GLOBAL_AND_CYCLIC_RMSD_LT_3A",
        "structure_handoff_status": "NOT_CREATED_PENDING_MANUAL_REVIEW",
        "permeability_status": "DEFERRED_UNTIL_STRUCTURE_GATES_PASS",
        "quality_checks": quality_checks,
        "artifacts": artifacts,
        "checkpoint_config_sha256": (
            str(portable_import["source_config_sha256"])
            if portable_import is not None
            else config_digest
        ),
        "resume_provenance": (
            {
                "mode": config["resume_mode"],
                "source_commit": portable_import["source_commit"],
                "source_search_program_sha256": portable_import[
                    "source_search_program_sha256"
                ],
                "source_config_sha256": portable_import["source_config_sha256"],
                "portable_import_manifest": str(portable_import_path),
                "portable_import_manifest_sha256": sha256_file(portable_import_path),
                "destination_full_ledger_reaudit_required": True,
                "destination_rescore_tolerance": PORTABLE_RESCORE_TOLERANCE,
            }
            if portable_import is not None
            else None
        ),
    }
    atomic_write_json(existing_manifest, manifest)

    # Prove that the immutable baseline remained byte-identical during search.
    after_hashes = {
        "baseline_manifest": sha256_file(baseline / "generation_manifest.json"),
        "baseline_all": sha256_file(baseline / "all_candidates.csv"),
        "baseline_unique": sha256_file(baseline / "unique_candidates.csv"),
        "baseline_eligible": sha256_file(baseline / "methylated_new_candidates.csv"),
    }
    for name, observed in after_hashes.items():
        if observed != input_hashes[name]:
            raise RuntimeError(f"Immutable baseline changed during search: {name}")

    print("===== V8 DETERMINISTIC DIRECTED RECOVERY SEARCH COMPLETE =====", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    print(f"Missing before: {', '.join(missing_targets) if missing_targets else 'none'}", flush=True)
    print(
        f"Missing after: {', '.join(missing_after_search) if missing_after_search else 'none'}",
        flush=True,
    )
    print(f"Released recovery candidates: {len(release_rows)}", flush=True)
    if quality_gate != "PASS":
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError("V8 directed recovery failed: " + ", ".join(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-manifest", default=str(DEFAULT_MODEL_MANIFEST))
    parser.add_argument("--representation-audit", default=str(DEFAULT_REPRESENTATION))
    parser.add_argument("--baseline-run-dir", default=str(DEFAULT_BASELINE))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--historical-designs-csv", default=str(DEFAULT_HISTORICAL))
    parser.add_argument("--prior-handoff-csv", default=str(DEFAULT_PRIOR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-batch-size", type=int, default=32)
    parser.add_argument("--wne-radius", type=int, default=2)
    parser.add_argument("--zgc-rounds", type=int, default=6)
    parser.add_argument("--zgc-beam-width", type=int, default=512)
    parser.add_argument("--zgc-offspring-per-round", type=int, default=4096)
    parser.add_argument("--max-release-per-target", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--portable-resume-manifest",
        default="",
        help=(
            "Hash-pinned AutoDL import manifest for a completed Windows round-6 "
            "ledger/checkpoint set. Final candidates and the independent final "
            "ledger audit are still recomputed on the destination GPU."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = (
        args.batch_size,
        args.base_batch_size,
        args.zgc_rounds,
        args.zgc_beam_width,
        args.zgc_offspring_per_round,
        args.max_release_per_target,
    )
    if any(int(value) <= 0 for value in positive):
        raise ValueError("All search sizes and budgets must be positive")
    if int(args.wne_radius) != 2:
        raise ValueError("3WNE protocol is frozen to exact Hamming radius 2")
    frozen = {
        "--batch-size": (int(args.batch_size), 64),
        "--base-batch-size": (int(args.base_batch_size), 32),
        "--zgc-rounds": (int(args.zgc_rounds), 6),
        "--zgc-beam-width": (int(args.zgc_beam_width), 512),
        "--zgc-offspring-per-round": (int(args.zgc_offspring_per_round), 4096),
        "--max-release-per-target": (int(args.max_release_per_target), 200),
    }
    changed = [name for name, (observed, expected) in frozen.items() if observed != expected]
    if changed:
        raise ValueError(
            "V8 directed-search protocol has frozen numerical budgets: "
            + ", ".join(changed)
        )
    run(args)


if __name__ == "__main__":
    main()
