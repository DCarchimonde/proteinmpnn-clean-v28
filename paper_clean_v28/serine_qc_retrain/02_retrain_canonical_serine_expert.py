#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrain only the Ser expert inside the canonical clean-V28 network.

No simplified surrogate network and no weight splicing are used.  The full
clean-V28 forward pass produces the hidden representation used at inference;
only ``experts[Ser]`` has gradients.  Every other tensor must remain bitwise
identical to ``frankenstein_v28.pt``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nmethyl.utils.nmethyl_config import (  # noqa: E402
    EXTENDED_AA_ALPHABET,
    EXTENDED_AA_TO_INDEX,
    NATURAL_AA_ALPHABET,
)
from paper_clean_v28.clean_v28_common import (  # noqa: E402
    N_NATURAL,
    X_INDEX,
    binary_metrics,
    featurize_records,
    load_v28_model,
    naturalize_tensor_for_input,
    read_jsonl,
    roc_auc_score_simple,
)


DEFAULT_DATA_DIR = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_retrain" / "data"
DEFAULT_OUT = REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_retrain" / "model"
EXPECTED_TRAIN_COUNTS = {"S": 242, "s": 50, "P": 307, "p": 0}
EXPECTED_TEST_COUNTS = {"S": 62, "s": 12, "P": 83, "p": 0}
SER_INDEX = NATURAL_AA_ALPHABET.index("S")
SER_METHYL_INDEX = EXTENDED_AA_TO_INDEX["s"]
ALLOWED_CHANGED_STATE_KEYS = {
    f"experts.{SER_INDEX}.weight",
    f"experts.{SER_INDEX}.bias",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hashes(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, str]:
    return {key: tensor_sha256(value) for key, value in sorted(state_dict.items())}


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


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sequence_counts(records: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for key, value in record.items():
            if key.startswith("seq_chain_"):
                counts.update(str(value))
    return counts


def require_corrected_counts(
    records: Sequence[Mapping[str, Any]], expected: Mapping[str, int], label: str
) -> Counter[str]:
    counts = sequence_counts(records)
    observed = {token: counts[token] for token in expected}
    if observed != dict(expected):
        raise RuntimeError(
            f"{label} is not the pinned provenance-corrected dataset: "
            f"expected {dict(expected)}, observed {observed}"
        )
    return counts


def batches(records: Sequence[Mapping[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield [dict(value) for value in records[start : start + batch_size]]


def train_serine_expert(
    model: torch.nn.Module,
    train_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> List[Dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    serine_expert = model.experts[SER_INDEX]
    for parameter in serine_expert.parameters():
        parameter.requires_grad_(True)

    train_counts = sequence_counts(train_records)
    positive_count = int(train_counts["s"])
    negative_count = int(train_counts["S"])
    if positive_count <= 0 or negative_count <= 0:
        raise RuntimeError("Ser expert requires both natural S and methyl s in training")
    positive_weight = negative_count / positive_count

    optimizer = torch.optim.AdamW(
        serine_expert.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history: List[Dict[str, Any]] = []

    # Inference-mode hidden representations are the training domain.  Dropout
    # stays disabled even while the linear Ser expert is optimized.
    model.eval()
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_records)))
        random.Random(seed + epoch).shuffle(order)
        shuffled = [train_records[index] for index in order]
        total_loss = 0.0
        total_positions = 0
        positive_seen = 0
        negative_seen = 0

        for batch in batches(shuffled, batch_size):
            packed = featurize_records(batch, device=device, eval_chains="masked")
            if packed is None:
                continue
            tensors, _metas = packed
            X, S_label, mask, chain_M, residue_idx, chain_encoding_all, real_pos = tensors
            valid = (
                (mask > 0)
                & (chain_M > 0)
                & (real_pos > 0)
                & (S_label != X_INDEX)
            )
            true_base = naturalize_tensor_for_input(S_label)
            selected = valid & (true_base == SER_INDEX)
            if not bool(selected.any()):
                continue

            optimizer.zero_grad(set_to_none=True)
            # The methyl label must never enter the sequence encoder.  This is
            # the same strict known-sequence inference domain used by the clean
            # evaluator: both S and s are presented to the trunk as natural S.
            S_forward = naturalize_tensor_for_input(S_label)
            _base_logits, expert_logits = model(
                X,
                S_forward,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
            )
            logits = expert_logits[..., SER_INDEX][selected]
            labels = (S_label[selected] == SER_METHYL_INDEX).to(dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(serine_expert.parameters(), 5.0)
            optimizer.step()

            count = int(labels.numel())
            total_positions += count
            positive_seen += int(labels.sum().item())
            negative_seen += count - int(labels.sum().item())
            total_loss += float(loss.item()) * count

        if total_positions != positive_count + negative_count:
            raise RuntimeError(
                f"Epoch {epoch} Ser coverage changed: expected "
                f"{positive_count + negative_count}, observed {total_positions}"
            )
        row = {
            "epoch": epoch,
            "mean_weighted_bce": total_loss / total_positions,
            "ser_positions": total_positions,
            "methyl_s_positions": positive_seen,
            "natural_S_positions": negative_seen,
            "positive_weight": positive_weight,
            "learning_rate": learning_rate,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs}: Ser BCE={row['mean_weighted_bce']:.6f}",
            flush=True,
        )
    return history


def evaluate(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    threshold: float,
    checkpoint_label: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    position_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(batches(records, batch_size)):
            packed = featurize_records(batch, device=device, eval_chains="masked")
            if packed is None:
                continue
            tensors, metas = packed
            X, S_label, mask, chain_M, residue_idx, chain_encoding_all, real_pos = tensors
            valid = (
                (mask > 0)
                & (chain_M > 0)
                & (real_pos > 0)
                & (S_label != X_INDEX)
            )
            # Prevent target leakage: lowercase methyl labels are evaluation
            # targets only, never inputs to the canonical model trunk.
            S_forward = naturalize_tensor_for_input(S_label)
            _base_logits, expert_logits = model(
                X,
                S_forward,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
            )
            true_base = naturalize_tensor_for_input(S_label)
            known_logits = torch.gather(
                expert_logits, -1, true_base.unsqueeze(-1)
            ).squeeze(-1)
            probability = torch.sigmoid(known_logits)

            for row_index, meta in enumerate(metas):
                for position in torch.where(valid[row_index])[0].cpu().tolist():
                    base_index = int(true_base[row_index, position].item())
                    target_index = int(S_label[row_index, position].item())
                    position_rows.append(
                        {
                            "checkpoint": checkpoint_label,
                            "sample_name": meta["name"],
                            "batch_index": batch_index,
                            "position_in_model_0based": position,
                            "target_token": EXTENDED_AA_ALPHABET[target_index],
                            "base_token": NATURAL_AA_ALPHABET[base_index],
                            "is_methyl_true": int(target_index >= N_NATURAL),
                            "probability_methyl": float(
                                probability[row_index, position].item()
                            ),
                        }
                    )

    y_all = np.asarray([row["is_methyl_true"] for row in position_rows], dtype=np.int64)
    p_all = np.asarray([row["probability_methyl"] for row in position_rows], dtype=np.float64)
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(position_rows):
        grouped_indices[str(row["base_token"])].append(index)

    per_residue: List[Dict[str, Any]] = []
    for base_token in NATURAL_AA_ALPHABET:
        idx = np.asarray(grouped_indices.get(base_token, []), dtype=np.int64)
        y = y_all[idx] if len(idx) else np.asarray([], dtype=np.int64)
        p = p_all[idx] if len(idx) else np.asarray([], dtype=np.float64)
        threshold_metrics = binary_metrics(y, p, [threshold])[0] if len(idx) else {}
        per_residue.append(
            {
                "checkpoint": checkpoint_label,
                "base_token": base_token,
                "positions": int(len(idx)),
                "natural_negatives": int(np.sum(y == 0)),
                "methyl_positives": int(np.sum(y == 1)),
                "auc": roc_auc_score_simple(y, p),
                **threshold_metrics,
            }
        )

    serine = next(row for row in per_residue if row["base_token"] == "S")
    non_ser_idx = np.asarray(
        [index for index, row in enumerate(position_rows) if row["base_token"] != "S"],
        dtype=np.int64,
    )
    overall_threshold = binary_metrics(y_all, p_all, [threshold])[0]
    non_ser_threshold = binary_metrics(
        y_all[non_ser_idx], p_all[non_ser_idx], [threshold]
    )[0]
    summary = {
        "checkpoint": checkpoint_label,
        "positions": len(position_rows),
        "threshold": threshold,
        "overall_auc": roc_auc_score_simple(y_all, p_all),
        "overall_at_threshold": overall_threshold,
        "non_ser_auc": roc_auc_score_simple(y_all[non_ser_idx], p_all[non_ser_idx]),
        "non_ser_at_threshold": non_ser_threshold,
        "serine": serine,
    }
    return summary, per_residue, position_rows


def compare_non_ser_predictions(
    baseline_rows: Sequence[Mapping[str, Any]],
    corrected_rows: Sequence[Mapping[str, Any]],
) -> float:
    def key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
        return (
            str(row["sample_name"]),
            int(row["position_in_model_0based"]),
            str(row["base_token"]),
        )

    baseline = {
        key(row): float(row["probability_methyl"])
        for row in baseline_rows
        if str(row["base_token"]) != "S"
    }
    corrected = {
        key(row): float(row["probability_methyl"])
        for row in corrected_rows
        if str(row["base_token"]) != "S"
    }
    if baseline.keys() != corrected.keys():
        raise RuntimeError("Non-Ser evaluation rows changed after Ser-only retraining")
    return max(
        (abs(baseline[key_value] - corrected[key_value]) for key_value in baseline),
        default=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(REPO_ROOT / "frankenstein_v28.pt"))
    parser.add_argument(
        "--train-jsonl",
        default=str(DEFAULT_DATA_DIR / "train_serine_provenance_corrected.jsonl"),
    )
    parser.add_argument(
        "--test-jsonl",
        default=str(DEFAULT_DATA_DIR / "test_serine_provenance_corrected.jsonl"),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-fail-on-quality-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between zero and one")

    model_path = Path(args.model_path).resolve()
    train_path = Path(args.train_jsonl).resolve()
    test_path = Path(args.test_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    for required in (model_path, train_path, test_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_deterministic_seed(args.seed)

    train_records = read_jsonl(str(train_path))
    test_records = read_jsonl(str(test_path))
    require_corrected_counts(train_records, EXPECTED_TRAIN_COUNTS, "train")
    require_corrected_counts(test_records, EXPECTED_TEST_COUNTS, "test")

    print(f"Loading canonical clean-V28 checkpoint: {model_path}", flush=True)
    model = load_v28_model(str(model_path), device)
    before_hashes = state_hashes(model.state_dict())
    baseline_summary, baseline_per_residue, baseline_positions = evaluate(
        model,
        test_records,
        device,
        args.batch_size,
        args.threshold,
        "baseline_frankenstein_v28",
    )

    history = train_serine_expert(
        model,
        train_records,
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
    )
    after_hashes = state_hashes(model.state_dict())
    changed_keys = sorted(
        key for key in before_hashes if before_hashes[key] != after_hashes[key]
    )
    if set(changed_keys) != ALLOWED_CHANGED_STATE_KEYS:
        raise RuntimeError(
            "Ser-only state isolation failed: expected changes "
            f"{sorted(ALLOWED_CHANGED_STATE_KEYS)}, observed {changed_keys}"
        )

    corrected_summary, corrected_per_residue, corrected_positions = evaluate(
        model,
        test_records,
        device,
        args.batch_size,
        args.threshold,
        "serine_qc_retrained",
    )
    non_ser_max_delta = compare_non_ser_predictions(
        baseline_positions, corrected_positions
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "frankenstein_v28_serine_qc.pt"
    candidate_checkpoint_path = out_dir / "frankenstein_v28_serine_qc.candidate.pt"
    checkpoint_payload = {
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "serine_qc_metadata": {
            "protocol": "canonical_clean_v28_serine_expert_only_v1",
            "parent_checkpoint_sha256": file_sha256(model_path),
            "train_jsonl_sha256": file_sha256(train_path),
            "test_jsonl_sha256": file_sha256(test_path),
            "changed_state_keys": changed_keys,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "threshold": args.threshold,
            "seed": args.seed,
        },
    }
    temporary_checkpoint = candidate_checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint_payload, temporary_checkpoint)
    os.replace(temporary_checkpoint, candidate_checkpoint_path)

    # Strictly reload the candidate artifact through the production loader.
    # It is promoted to the production filename only after every quality gate
    # below passes.
    reloaded = load_v28_model(str(candidate_checkpoint_path), device)
    reload_hashes = state_hashes(reloaded.state_dict())
    if reload_hashes != after_hashes:
        raise RuntimeError("Saved checkpoint failed strict state round-trip")

    serine = corrected_summary["serine"]
    quality_checks = {
        "only_ser_expert_changed": set(changed_keys) == ALLOWED_CHANGED_STATE_KEYS,
        "non_ser_probabilities_bitwise_stable": non_ser_max_delta == 0.0,
        "serine_test_has_both_classes": (
            int(serine["natural_negatives"]) == EXPECTED_TEST_COUNTS["S"]
            and int(serine["methyl_positives"]) == EXPECTED_TEST_COUNTS["s"]
        ),
        "serine_auc_ge_0_75": serine["auc"] is not None and float(serine["auc"]) >= 0.75,
        "serine_recall_at_0_6_ge_0_50": float(serine["recall"]) >= 0.50,
        "serine_fpr_at_0_6_le_0_20": float(serine["false_positive_rate"]) <= 0.20,
        "overall_auc_ge_0_85": (
            corrected_summary["overall_auc"] is not None
            and float(corrected_summary["overall_auc"]) >= 0.85
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    if quality_gate == "PASS":
        os.replace(candidate_checkpoint_path, checkpoint_path)
        checkpoint_artifact_path = checkpoint_path
    else:
        checkpoint_artifact_path = candidate_checkpoint_path
    manifest = {
        "quality_gate": quality_gate,
        "protocol": "canonical_clean_v28_serine_expert_only_v1",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parent_checkpoint": str(model_path),
        "parent_checkpoint_sha256": file_sha256(model_path),
        "checkpoint_ready_for_generation": quality_gate == "PASS",
        "output_checkpoint": str(checkpoint_path) if quality_gate == "PASS" else None,
        "candidate_checkpoint": str(checkpoint_artifact_path),
        "checkpoint_artifact_sha256": file_sha256(checkpoint_artifact_path),
        "changed_state_keys": changed_keys,
        "unchanged_state_key_count": len(before_hashes) - len(changed_keys),
        "non_ser_probability_max_abs_delta": non_ser_max_delta,
        "alphabet": EXTENDED_AA_ALPHABET,
        "alphabet_size": len(EXTENDED_AA_ALPHABET),
        "proline_policy": "no p output token; P expert and alphabet remain unchanged",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "train_jsonl": str(train_path),
            "train_jsonl_sha256": file_sha256(train_path),
            "test_jsonl": str(test_path),
            "test_jsonl_sha256": file_sha256(test_path),
        },
        "baseline_test": baseline_summary,
        "corrected_test": corrected_summary,
        "quality_checks": quality_checks,
    }
    atomic_write_json(out_dir / "serine_retrain_manifest.json", manifest)
    atomic_write_csv(
        out_dir / "training_history.csv", history, list(history[0])
    )
    metric_rows = baseline_per_residue + corrected_per_residue
    atomic_write_csv(
        out_dir / "test_metrics_by_residue.csv",
        metric_rows,
        list(metric_rows[0]),
    )
    position_rows = baseline_positions + corrected_positions
    atomic_write_csv(
        out_dir / "test_position_probabilities.csv",
        position_rows,
        list(position_rows[0]),
    )

    print("===== CANONICAL SER EXPERT RETRAIN COMPLETE =====", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    print(f"Changed tensors: {', '.join(changed_keys)}", flush=True)
    print(f"Non-Ser max probability delta: {non_ser_max_delta:.12g}", flush=True)
    print(
        "Ser test: AUC={auc:.4f}, recall={recall:.4f}, FPR={false_positive_rate:.4f}".format(
            **serine
        ),
        flush=True,
    )
    if quality_gate == "PASS":
        print(f"Production checkpoint: {checkpoint_path}", flush=True)
    else:
        print(
            f"Diagnostic candidate only (not production): {candidate_checkpoint_path}",
            flush=True,
        )

    if quality_gate != "PASS" and not args.no_fail_on_quality_gate:
        failed = [name for name, passed in quality_checks.items() if not passed]
        raise RuntimeError("Quality gate failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
