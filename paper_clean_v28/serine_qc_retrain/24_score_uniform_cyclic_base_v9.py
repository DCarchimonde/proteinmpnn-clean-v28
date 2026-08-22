#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exact receptor-visible cyclic-base scoring for the V9 release pool.

For every target/sequence this scorer jointly rotates peptide coordinates and
sequence through all L physical starts, resets the peptide residue index, and
evaluates all L causal decoder orders at each start.  The scalar for one start
is the mean natural-base log probability across peptide sites and decoder
orders.  A per-target nearest-rank 1% floor is frozen from the complete unique
generation pool; only candidate means at or above that target-specific floor
may proceed to the independent 17 x 100 selector.
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
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
GENERATOR_PATH = REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_BEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "best_designs.csv"
)
DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_cyclic_stability_v9_1700.json")
EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_"
    "cyclic_stability_worst_start_v9"
)
V11_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_cyclic_native_relative_positions_v11"
)
AUDIT_PROTOCOL = "cyclic_stability_worst_start_heldout_gate_v9"
AUDIT_AUTHORIZATION = "CYCLIC_STABILITY_V9_VALIDATED_FOR_UNIFORM_REGENERATION"
V11_AUDIT_PROTOCOL = "cyclic_native_relative_positions_heldout_gate_v11"
V11_AUDIT_AUTHORIZATION = (
    "CYCLIC_NATIVE_V11_VALIDATED_FOR_RMSD_PRIORITY_REGENERATION"
)
SCORE_PROTOCOL = "receptor_visible_all_physical_starts_all_decoder_orders_exact_v9"
FLOOR_POLICY = (
    "per_target_bottom_1pct_current_pool_outlier_filter_"
    "not_independent_calibration_v9"
)
FLOOR_FRACTION = 0.01
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
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


def atomic_write_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_gzip_json(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Malformed target scoring checkpoint: {path}")
    return payload


def text_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def torch_runtime_contract(torch_module: Any, device: Any) -> Dict[str, Any]:
    """Capture every runtime switch that may change persisted numeric scores."""
    device_type = str(getattr(device, "type", str(device).split(":", 1)[0]))
    cuda_active = device_type == "cuda"
    cuda = getattr(torch_module, "cuda", None)
    backend = getattr(torch_module, "backends", None)
    cudnn = getattr(backend, "cudnn", None)
    cuda_backend = getattr(backend, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    gpu: Dict[str, Any] | None = None
    if cuda_active:
        device_index = getattr(device, "index", None)
        if device_index is None:
            device_index = int(cuda.current_device())
        properties = cuda.get_device_properties(device_index)
        gpu = {
            "index": int(device_index),
            "name": str(cuda.get_device_name(device_index)),
            "capability": [
                int(value) for value in cuda.get_device_capability(device_index)
            ],
            "total_memory": int(getattr(properties, "total_memory", -1)),
            "multi_processor_count": int(
                getattr(properties, "multi_processor_count", -1)
            ),
            "uuid": str(getattr(properties, "uuid", "")),
        }
    deterministic_warn_only = None
    warn_only_reader = getattr(
        torch_module, "is_deterministic_algorithms_warn_only_enabled", None
    )
    if callable(warn_only_reader):
        deterministic_warn_only = bool(warn_only_reader())
    float32_precision_reader = getattr(torch_module, "get_float32_matmul_precision", None)
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(getattr(torch_module.version, "cuda", None)),
        "cudnn_version": (
            int(cudnn.version())
            if cudnn is not None and cudnn.version() is not None
            else None
        ),
        "device": str(device),
        "device_type": device_type,
        "gpu": gpu,
        "deterministic": {
            "algorithms_enabled": bool(
                torch_module.are_deterministic_algorithms_enabled()
            ),
            "algorithms_warn_only": deterministic_warn_only,
            "cudnn_deterministic": bool(getattr(cudnn, "deterministic", False)),
            "cudnn_benchmark": bool(getattr(cudnn, "benchmark", False)),
            "cudnn_allow_tf32": bool(getattr(cudnn, "allow_tf32", False)),
            "cuda_matmul_allow_tf32": bool(
                getattr(matmul_backend, "allow_tf32", False)
            ),
            "float32_matmul_precision": (
                str(float32_precision_reader())
                if callable(float32_precision_reader)
                else None
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }


def scoring_input_records(
    *,
    candidate_path: Path,
    baseline_path: Path,
    model_path: Path,
    generation_manifest_path: Path,
    audit_path: Path,
    native_path: Path,
    best_path: Path,
    plan_path: Path,
) -> Dict[str, Dict[str, str]]:
    """Build path-bound records used by the runner's PASS-cache validator."""
    return {
        "candidate_csv": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "baseline_csv": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "generation_manifest": {
            "path": str(generation_manifest_path),
            "sha256": sha256_file(generation_manifest_path),
        },
        "representation_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
        },
        "native_jsonl": {
            "path": str(native_path),
            "sha256": sha256_file(native_path),
        },
        "best_csv": {"path": str(best_path), "sha256": sha256_file(best_path)},
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
    }


def target_checkpoint_is_reusable(
    checkpoint: Mapping[str, Any],
    *,
    target: str,
    config_sha256: str,
    program_hashes: Mapping[str, str],
    runtime_contract: Mapping[str, Any],
    scoring_parameters: Mapping[str, Any],
    sequence_set_sha256: str,
    sequences: Sequence[str],
) -> bool:
    scores = checkpoint.get("scores")
    return bool(
        checkpoint.get("protocol") == SCORE_PROTOCOL
        and checkpoint.get("target_name") == target
        and checkpoint.get("config_sha256") == config_sha256
        and all(
            checkpoint.get(name) == value for name, value in program_hashes.items()
        )
        and checkpoint.get("runtime_contract") == dict(runtime_contract)
        and checkpoint.get("scoring_parameters") == dict(scoring_parameters)
        and checkpoint.get("sequence_set_sha256") == sequence_set_sha256
        and int(checkpoint.get("sequence_count", -1)) == len(sequences)
        and isinstance(scores, dict)
        and set(scores) == set(sequences)
        and all(
            score_payload_is_valid(sequence, scores[sequence])
            for sequence in sequences
        )
    )


def score_payload_is_valid(sequence: str, payload: object) -> bool:
    """Reject truncated, non-finite, or internally inconsistent score caches."""
    if not isinstance(payload, Mapping):
        return False
    length = len(sequence)
    try:
        matrix = json.loads(
            str(payload["cyclic_base_log_probability_start_by_decoder_order"])
        )
        by_start = json.loads(str(payload["cyclic_base_log_probability_by_start"]))
        if not (
            payload.get("cyclic_base_score_protocol") == SCORE_PROTOCOL
            and int(payload.get("cyclic_base_physical_start_count", -1)) == length
            and int(payload.get("cyclic_base_decoder_order_count_per_start", -1))
            == length
            and int(payload.get("cyclic_base_total_ensemble_size", -1))
            == length * length
            and isinstance(matrix, list)
            and len(matrix) == length
            and all(isinstance(row, list) and len(row) == length for row in matrix)
            and isinstance(by_start, list)
            and len(by_start) == length
        ):
            return False
        matrix_values = [[float(value) for value in row] for row in matrix]
        start_values = [float(value) for value in by_start]
        scalars = {
            "mean": float(payload["cyclic_base_log_probability_mean"]),
            "minimum": float(payload["cyclic_base_log_probability_min"]),
            "maximum": float(payload["cyclic_base_log_probability_max"]),
            "span": float(payload["cyclic_base_log_probability_span"]),
            "std": float(payload["cyclic_base_log_probability_std"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not all(
        math.isfinite(value)
        for value in [
            *[item for row in matrix_values for item in row],
            *start_values,
            *scalars.values(),
        ]
    ):
        return False
    recomputed_by_start = [sum(row) / length for row in matrix_values]
    mean = sum(recomputed_by_start) / length
    minimum = min(recomputed_by_start)
    maximum = max(recomputed_by_start)
    std = math.sqrt(
        sum((value - mean) ** 2 for value in recomputed_by_start) / length
    )
    expected_scalars = {
        "mean": mean,
        "minimum": minimum,
        "maximum": maximum,
        "span": maximum - minimum,
        "std": std,
    }
    return all(
        math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
        for observed, expected in zip(start_values, recomputed_by_start)
    ) and all(
        math.isclose(scalars[name], expected, rel_tol=1e-12, abs_tol=1e-12)
        for name, expected in expected_scalars.items()
    )


def batches(values: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 < fraction <= 1.0:
        raise ValueError("Nearest-rank percentile requires values and 0<fraction<=1")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Nearest-rank percentile requires finite values")
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("v9_t05_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generator: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def natural_sequence(row: Mapping[str, Any]) -> str:
    value = str(row.get("design_natural_seq") or row.get("design_seq") or "").upper()
    if not value or not set(value) <= set(NATURAL_AA):
        raise ValueError(f"Invalid natural candidate sequence: {value!r}")
    return value


class ExactCyclicBaseScorer:
    """Differentiation-free exact L physical starts x L decoder orders scorer."""

    def __init__(
        self,
        model: Any,
        device: Any,
        target_records: Mapping[str, Mapping[str, Any]],
        torch_module: Any,
        functional: Any,
        featurize_records: Any,
        complete_decoding_order: Any,
        batch_size: int,
    ) -> None:
        self.model = model
        self.device = device
        self.target_records = target_records
        self.torch = torch_module
        self.functional = functional
        self.featurize_records = featurize_records
        self.complete_decoding_order = complete_decoding_order
        self.batch_size = batch_size
        self.features: Dict[str, Tuple[Any, ...]] = {}

    def _features(self, target: str) -> Tuple[Any, ...]:
        if target not in self.features:
            packed = self.featurize_records(
                [self.target_records[target]],
                device=self.device,
                eval_chains="masked",
                max_peptide_len=30,
            )
            if packed is None:
                raise RuntimeError(f"Feature construction failed for {target}")
            tensors, _metas = packed
            self.features[target] = tuple(tensors[:6])
        return self.features[target]

    def score(self, target: str, sequences: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        unique = sorted(set(str(value).upper() for value in sequences))
        alphabet_index = {token: index for index, token in enumerate(NATURAL_AA)}
        X, S_true, mask, chain_M, residue_idx, chain_encoding = self._features(target)
        selected = self.torch.nonzero(
            (chain_M[0] * mask[0]) > 0.0, as_tuple=False
        ).squeeze(-1)
        length = int(selected.numel())
        if length <= 0:
            raise RuntimeError(f"No designed peptide positions for {target}")
        canonical_residue_idx = self.torch.arange(
            length, device=self.device, dtype=residue_idx.dtype
        )
        chain_values = self.torch.unique(chain_encoding[0, selected])
        if int(chain_values.numel()) != 1:
            raise RuntimeError(f"Selected peptide is not one chain for {target}")

        for sequence_batch in batches(unique, self.batch_size):
            if any(len(sequence) != length for sequence in sequence_batch):
                raise RuntimeError(f"Candidate length mismatch for {target}")
            current = len(sequence_batch)
            natural = self.torch.tensor(
                [[alphabet_index[token] for token in sequence] for sequence in sequence_batch],
                device=self.device,
                dtype=self.torch.long,
            )
            decoder_score_matrices: List[Any] = []
            with self.torch.no_grad():
                for physical_shift in range(length):
                    Xb = X.repeat(current, 1, 1, 1)
                    Sb = S_true.repeat(current, 1).clone()
                    maskb = mask.repeat(current, 1)
                    chainb = chain_M.repeat(current, 1)
                    residueb = residue_idx.repeat(current, 1).clone()
                    encodingb = chain_encoding.repeat(current, 1).clone()
                    rolled_coordinates = self.torch.roll(
                        X[0, selected], shifts=-physical_shift, dims=0
                    )
                    Xb[:, selected] = rolled_coordinates.unsqueeze(0).repeat(
                        current, 1, 1, 1
                    )
                    rolled_natural = self.torch.roll(
                        natural, shifts=-physical_shift, dims=1
                    )
                    Sb[:, selected] = rolled_natural
                    residueb[:, selected] = canonical_residue_idx.unsqueeze(0).repeat(
                        current, 1
                    )
                    encodingb[:, selected] = chain_values[0]
                    decoder_scores: List[Any] = []
                    for decoder_shift in range(length):
                        requested = self.torch.roll(
                            selected, shifts=-decoder_shift
                        ).unsqueeze(0).repeat(current, 1)
                        order = self.complete_decoding_order(chainb, maskb, requested)
                        base_logits, _expert_logits = self.model(
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
                            -1, rolled_natural.unsqueeze(-1)
                        ).squeeze(-1)
                        decoder_scores.append(selected_log_probability.mean(dim=1))
                    decoder_score_matrices.append(
                        self.torch.stack(decoder_scores, dim=1)
                    )
            score_cube = (
                self.torch.stack(decoder_score_matrices, dim=1)
                .detach()
                .cpu()
                .tolist()
            )
            for sequence, raw_matrix in zip(sequence_batch, score_cube):
                start_by_decoder = [
                    [round(float(value), 8) for value in decoder_values]
                    for decoder_values in raw_matrix
                ]
                by_start = [
                    sum(decoder_values) / len(decoder_values)
                    for decoder_values in start_by_decoder
                ]
                mean = sum(by_start) / len(by_start)
                minimum = min(by_start)
                maximum = max(by_start)
                std = math.sqrt(
                    sum((value - mean) ** 2 for value in by_start) / len(by_start)
                )
                result[sequence] = {
                    "cyclic_base_score_protocol": SCORE_PROTOCOL,
                    "cyclic_base_log_probability_start_by_decoder_order": json.dumps(
                        start_by_decoder
                    ),
                    "cyclic_base_log_probability_by_start": json.dumps(by_start),
                    "cyclic_base_log_probability_mean": mean,
                    "cyclic_base_log_probability_min": minimum,
                    "cyclic_base_log_probability_max": maximum,
                    "cyclic_base_log_probability_span": maximum - minimum,
                    "cyclic_base_log_probability_std": std,
                    "cyclic_base_physical_start_count": length,
                    "cyclic_base_decoder_order_count_per_start": length,
                    "cyclic_base_total_ensemble_size": length * length,
                }
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--representation-audit", required=True)
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--best-csv", default=str(DEFAULT_BEST))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Exact cyclic-base scoring requires PyTorch") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CPU scoring requires explicit --allow-cpu")
    device = torch.device(args.device)
    runtime_contract = torch_runtime_contract(torch, device)
    scoring_parameters = {
        "batch_size": int(args.batch_size),
        "device_argument": str(args.device),
        "allow_cpu": bool(args.allow_cpu),
        "floor_policy": FLOOR_POLICY,
        "floor_fraction": FLOOR_FRACTION,
    }

    generator = load_generator()
    clean_dir = REPO_ROOT / "paper_clean_v28"
    if str(clean_dir) not in sys.path:
        sys.path.insert(0, str(clean_dir))
    from clean_v28_common import (  # pylint: disable=import-error,import-outside-toplevel
        complete_decoding_order,
        featurize_records,
        load_v28_model,
    )

    candidate_path = Path(args.candidate_csv).resolve()
    baseline_path = Path(args.baseline_csv).resolve()
    model_path = Path(args.model).resolve()
    generation_manifest_path = Path(args.generation_manifest).resolve()
    audit_path = Path(args.representation_audit).resolve()
    native_path = Path(args.native_jsonl).resolve()
    best_path = Path(args.best_csv).resolve()
    plan_path = Path(args.plan).resolve()
    out_dir = Path(args.out_dir).resolve()
    for required in (
        candidate_path,
        baseline_path,
        model_path,
        generation_manifest_path,
        audit_path,
        native_path,
        best_path,
        plan_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    output_csv = out_dir / "candidates_exact_cyclic_base_scored.csv"
    output_pass = out_dir / "candidates_exact_cyclic_base_pass.csv"
    output_summary = out_dir / "cyclic_base_floor_by_target.csv"
    output_manifest = out_dir / "exact_cyclic_base_scoring_manifest.json"
    checkpoint_dir = out_dir / "target_checkpoints"
    existing = [
        path
        for path in (output_csv, output_pass, output_summary, output_manifest)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Exact cyclic-base output exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()
        for target in (
            "1SFI", "3AV9", "3AVA", "3AVB", "3AVF", "3AVG", "3AVH",
            "3AVI", "3AVJ", "3AVK", "3AVM", "3AVN", "3P8F", "3WNE",
            "3ZGC", "4K1E", "4KEL",
        ):
            checkpoint = checkpoint_dir / f"{target.lower()}_exact_base.json.gz"
            if checkpoint.is_file():
                checkpoint.unlink()

    plan = read_json(plan_path)
    generation_manifest = read_json(generation_manifest_path)
    audit = read_json(audit_path)
    model_sha256 = sha256_file(model_path)
    plan_sha256 = sha256_file(plan_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    metadata = (
        dict(checkpoint.get("expert_head_qc_metadata", {}))
        if isinstance(checkpoint, Mapping)
        else {}
    )
    del checkpoint
    model_expert_protocol = str(metadata.get("protocol", ""))
    model_is_v11 = model_expert_protocol == V11_EXPERT_PROTOCOL
    upstream_checks = {
        "checkpoint_is_promoted_v9": (
            model_expert_protocol in {EXPERT_PROTOCOL, V11_EXPERT_PROTOCOL}
            and float(metadata.get("worst_start_bce_weight", 0.0)) > 0.0
            and float(metadata.get("representation_consistency_weight", 0.0)) > 0.0
            and bool(metadata.get("full_physical_start_by_full_decoder_order_grid"))
            and float(metadata.get("training_ensemble_temperature", -1.0)) == 0.5
            and "full_physical_start_x_full_decoder_order_grid"
            in str(metadata.get("training_objective", ""))
            and (
                not model_is_v11
                or (
                    bool(metadata.get("cyclic_relative_positions"))
                    and float(metadata.get("base_sequence_loss_weight", 0.0))
                    > 0.0
                )
            )
        ),
        "generation_manifest_pass": generation_manifest.get("quality_gate") == "PASS",
        "generation_model_hash_matches": generation_manifest.get("model_sha256") == model_sha256,
        "generation_protocol_matches_plan": generation_manifest.get("protocol") == plan.get("protocol"),
        "generation_expert_protocol_matches_checkpoint": (
            generation_manifest.get("model_expert_qc_protocol")
            == model_expert_protocol
        ),
        "candidate_pool_matches_generation_methylated_bytes": (
            generation_manifest.get("methylated_new_candidates_csv_sha256")
            == sha256_file(candidate_path)
        ),
        "baseline_pool_matches_generation_unique_bytes": (
            generation_manifest.get("unique_candidates_csv_sha256")
            == sha256_file(baseline_path)
        ),
        "heldout_audit_is_authorized": (
            audit.get("quality_gate") == "PASS"
            and audit.get("protocol")
            == (V11_AUDIT_PROTOCOL if model_is_v11 else AUDIT_PROTOCOL)
            and audit.get("release_authorization")
            == (
                V11_AUDIT_AUTHORIZATION
                if model_is_v11
                else AUDIT_AUTHORIZATION
            )
            and audit.get("model_expert_qc_protocol") == model_expert_protocol
            and audit.get("model_sha256") == model_sha256
            and audit.get("plan_sha256") == plan_sha256
        ),
    }
    if not all(upstream_checks.values()):
        failed = [name for name, passed in upstream_checks.items() if not passed]
        raise RuntimeError("Exact cyclic-base upstream gate failed: " + ", ".join(failed))

    target_names = [str(item["target_name"]).upper() for item in plan["targets"]]
    candidate_rows = read_csv(candidate_path)
    baseline_rows = read_csv(baseline_path)
    candidate_by_target: MutableMapping[str, List[Dict[str, str]]] = defaultdict(list)
    baseline_by_target: MutableMapping[str, List[Dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        candidate_by_target[str(row.get("target_name", "")).upper()].append(row)
    for row in baseline_rows:
        baseline_by_target[str(row.get("target_name", "")).upper()].append(row)
    if set(candidate_by_target) != set(target_names) or set(baseline_by_target) != set(target_names):
        raise RuntimeError("Candidate/baseline target sets must exactly match the frozen 17")

    best_rows = read_csv(best_path)
    selected_chains = generator.selected_chain_index(best_rows)
    native_rows = generator.read_jsonl(native_path)
    target_records, _target_manifest = generator.prepare_target_records(
        native_rows, selected_chains, target_names
    )
    model = load_v28_model(str(model_path), device)
    model.eval()
    scorer = ExactCyclicBaseScorer(
        model,
        device,
        target_records,
        torch,
        functional,
        featurize_records,
        complete_decoding_order,
        args.batch_size,
    )

    enriched: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    checkpoint_artifacts: List[Dict[str, Any]] = []
    checkpoint_program_hashes = {
        "scorer_program_sha256": sha256_file(SCRIPT_PATH),
        "generator_dependency_sha256": sha256_file(GENERATOR_PATH),
        "clean_v28_common_dependency_sha256": sha256_file(
            REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"
        ),
        "model_utils_dependency_sha256": sha256_file(
            REPO_ROOT / "model_utils.py"
        ),
        "nmethyl_config_dependency_sha256": sha256_file(
            REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"
        ),
    }
    checkpoint_config_sha256 = text_sha256(
        [
            SCORE_PROTOCOL,
            FLOOR_POLICY,
            model_sha256,
            plan_sha256,
            sha256_file(candidate_path),
            sha256_file(baseline_path),
            sha256_file(native_path),
            sha256_file(best_path),
            json.dumps(scoring_parameters, sort_keys=True, separators=(",", ":")),
            json.dumps(runtime_contract, sort_keys=True, separators=(",", ":")),
            *[
                f"{name}={value}"
                for name, value in sorted(checkpoint_program_hashes.items())
            ],
        ]
    )
    for target in target_names:
        candidate_sequences = [natural_sequence(row) for row in candidate_by_target[target]]
        baseline_sequences = [natural_sequence(row) for row in baseline_by_target[target]]
        union = sorted(set(candidate_sequences) | set(baseline_sequences))
        union_sha256 = text_sha256(union)
        checkpoint_path = checkpoint_dir / f"{target.lower()}_exact_base.json.gz"
        if checkpoint_path.is_file():
            checkpoint = read_gzip_json(checkpoint_path)
            if not target_checkpoint_is_reusable(
                checkpoint,
                target=target,
                config_sha256=checkpoint_config_sha256,
                program_hashes=checkpoint_program_hashes,
                runtime_contract=runtime_contract,
                scoring_parameters=scoring_parameters,
                sequence_set_sha256=union_sha256,
                sequences=union,
            ):
                raise RuntimeError(
                    f"Target checkpoint is stale or malformed; use --overwrite: {checkpoint_path}"
                )
            score = dict(checkpoint["scores"])
            checkpoint_reused = True
        else:
            score = scorer.score(target, union)
            atomic_write_gzip_json(
                checkpoint_path,
                {
                    "protocol": SCORE_PROTOCOL,
                    "target_name": target,
                    "config_sha256": checkpoint_config_sha256,
                    **checkpoint_program_hashes,
                    "runtime_contract": runtime_contract,
                    "scoring_parameters": scoring_parameters,
                    "sequence_set_sha256": union_sha256,
                    "sequence_count": len(union),
                    "scores": score,
                },
            )
            checkpoint_reused = False
        checkpoint_artifacts.append(
            {
                "target_name": target,
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "reused": checkpoint_reused,
                "sequence_count": len(union),
            }
        )
        baseline_means = [
            round(float(score[sequence]["cyclic_base_log_probability_mean"]), 8)
            for sequence in sorted(set(baseline_sequences))
        ]
        floor = nearest_rank_percentile(baseline_means, FLOOR_FRACTION)
        passed = 0
        for source in candidate_by_target[target]:
            sequence = natural_sequence(source)
            row = dict(source)
            row.update(score[sequence])
            row["cyclic_base_floor_policy"] = FLOOR_POLICY
            row["cyclic_base_floor"] = floor
            row["cyclic_base_gate_pass"] = int(
                round(float(row["cyclic_base_log_probability_mean"]), 8) >= floor
            )
            passed += int(row["cyclic_base_gate_pass"])
            enriched.append(row)
        summary_rows.append(
            {
                "target_name": target,
                "baseline_rows": len(baseline_by_target[target]),
                "baseline_unique_natural_sequences": len(set(baseline_sequences)),
                "candidate_rows": len(candidate_by_target[target]),
                "candidate_rows_passing_exact_cyclic_base_floor": passed,
                "floor_fraction": FLOOR_FRACTION,
                "nearest_rank": max(1, math.ceil(FLOOR_FRACTION * len(baseline_means))),
                "cyclic_base_floor": floor,
                "floor_policy": FLOOR_POLICY,
            }
        )

    atomic_write_csv(output_csv, enriched, union_fields(enriched))
    passing_rows = [row for row in enriched if int(row["cyclic_base_gate_pass"]) == 1]
    atomic_write_csv(output_pass, passing_rows, union_fields(enriched))
    atomic_write_csv(output_summary, summary_rows, list(summary_rows[0]))
    manifest = {
        "quality_gate": "PASS",
        "protocol": SCORE_PROTOCOL,
        "floor_policy": FLOOR_POLICY,
        "floor_fraction": FLOOR_FRACTION,
        "floor_scope_limitation": (
            "The floor is estimated from the same target's current unique "
            "generation pool. It is a weak bottom-1% outlier filter, not an "
            "independent base-plausibility calibration or publication benchmark."
        ),
        "upstream_checks": upstream_checks,
        "target_count": len(target_names),
        "candidate_rows": len(candidate_rows),
        "scored_candidate_rows": len(enriched),
        "candidate_rows_passing_exact_floor": len(passing_rows),
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "runtime_contract": runtime_contract,
        "scoring_parameters": scoring_parameters,
        "checkpoint_config_sha256": checkpoint_config_sha256,
        "model_sha256": model_sha256,
        "plan_sha256": plan_sha256,
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "dependencies": {
            "generator_module": {
                "path": str(GENERATOR_PATH),
                "sha256": sha256_file(GENERATOR_PATH),
            },
            "clean_v28_common": {
                "path": str(REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"),
                "sha256": sha256_file(REPO_ROOT / "paper_clean_v28" / "clean_v28_common.py"),
            },
            "model_utils": {
                "path": str(REPO_ROOT / "model_utils.py"),
                "sha256": sha256_file(REPO_ROOT / "model_utils.py"),
            },
            "nmethyl_config": {
                "path": str(REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"),
                "sha256": sha256_file(REPO_ROOT / "nmethyl" / "utils" / "nmethyl_config.py"),
            },
        },
        "inputs": scoring_input_records(
            candidate_path=candidate_path,
            baseline_path=baseline_path,
            model_path=model_path,
            generation_manifest_path=generation_manifest_path,
            audit_path=audit_path,
            native_path=native_path,
            best_path=best_path,
            plan_path=plan_path,
        ),
        "target_summary": summary_rows,
        "target_checkpoints": checkpoint_artifacts,
        "artifacts": {
            "scored_candidates": {"path": str(output_csv), "sha256": sha256_file(output_csv)},
            "passing_candidates": {"path": str(output_pass), "sha256": sha256_file(output_pass)},
            "floor_by_target": {"path": str(output_summary), "sha256": sha256_file(output_summary)},
        },
    }
    atomic_write_json(output_manifest, manifest)
    print("===== V9 EXACT CYCLIC-BASE SCORING COMPLETE =====", flush=True)
    print(f"Targets: {len(target_names)}", flush=True)
    print(f"Candidates: {len(enriched)}", flush=True)
    print(f"Scored CSV: {output_csv}", flush=True)


if __name__ == "__main__":
    main()
