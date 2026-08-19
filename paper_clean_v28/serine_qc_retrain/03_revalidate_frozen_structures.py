#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Revalidate the seven frozen T=0.5 structures with the final expert checkpoint.

This step does not regenerate sequences or structures.  It re-scores each
already-passed naturalized peptide on its native peptide backbone with every
receptor chain removed, matching the expert-head train/test domain.  It never
overwrites the accepted compound's sequence or methyl sites: disagreement is
reported as provenance, not silently converted into a different compound.  The
existing HighFold structure remains reusable because HighFold received the
identical naturalized sequence.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nmethyl.utils.nmethyl_config import (  # noqa: E402
    EXTENDED_AA_ALPHABET,
    NATURAL_AA_ALPHABET,
)
from paper_clean_v28.clean_v28_common import (  # noqa: E402
    NAT_TO_METHYL_ABS,
    cyclic_known_sequence_methyl_probabilities,
    featurize_records,
    load_v28_model,
    naturalize_tensor_for_input,
    read_jsonl,
)


DEFAULT_PLAN = SCRIPT_PATH.with_name("target_plan_structure_failures.json")
DEFAULT_MODEL = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "serine_qc_order_balanced_v3"
    / "model"
    / "frankenstein_v28_expert_heads_qc.pt"
)
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_OUT = (
    REPO_ROOT / "paper_clean_v28_outputs" / "serine_qc_peptide_only_v4" / "bridge"
)
REQUIRED_ORDER_BALANCED_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_corrected_labels_order_balanced_v3"
)
PEPTIDE_ONLY_ANNOTATION_MODE = (
    "peptide_only_cyclic_order_ensemble_known_natural_sequence"
)
PEPTIDE_ONLY_ANNOTATION_CONTEXT = (
    "peptide_chain_only_no_visible_receptor_chains"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def record_name(record: Mapping[str, Any], fallback: int) -> str:
    return str(
        record.get("name")
        or record.get("pdb")
        or record.get("pdb_id")
        or record.get("id")
        or f"record_{fallback}"
    ).upper()


def chain_ids(record: Mapping[str, Any]) -> List[str]:
    ordered: List[str] = []
    for chain in list(record.get("masked_list", [])) + list(record.get("visible_list", [])):
        chain = str(chain)
        if f"seq_chain_{chain}" in record and chain not in ordered:
            ordered.append(chain)
    for key in record:
        if key.startswith("seq_chain_"):
            chain = key[len("seq_chain_") :]
            if chain not in ordered:
                ordered.append(chain)
    return ordered


def prepare_candidate_record(
    source: Mapping[str, Any], selected_chain: str, design_natural: str
) -> Dict[str, Any]:
    available = chain_ids(source)
    if selected_chain not in available:
        raise RuntimeError(f"Selected chain {selected_chain} is absent from {record_name(source, 0)}")
    native_length = len(str(source.get(f"seq_chain_{selected_chain}", "")))
    if native_length != len(design_natural):
        raise RuntimeError(
            f"Candidate/native length mismatch for {record_name(source, 0)}: "
            f"{len(design_natural)} != {native_length}"
        )
    record: Dict[str, Any] = {
        "name": record_name(source, 0),
        "seq": design_natural,
        f"seq_chain_{selected_chain}": design_natural,
    }
    for atom_name in ("N", "CA", "C", "O"):
        key = f"{atom_name}_chain_{selected_chain}"
        if key not in source:
            raise RuntimeError(f"Missing {key} for {record_name(source, 0)}")
        record[key] = copy.deepcopy(source[key])
    record["masked_list"] = [selected_chain]
    record["visible_list"] = []
    return record


def score_one(
    model: torch.nn.Module,
    device: torch.device,
    source: Mapping[str, Any],
    target: str,
    evidence: Mapping[str, Any],
    temperature: float,
    threshold: float,
) -> Dict[str, Any]:
    old_design = str(evidence["design_seq"])
    retained_methyl_positions = [
        index for index, token in enumerate(old_design, start=1) if token.islower()
    ]
    design_natural = old_design.upper()
    selected_chain = str(evidence["selected_chain"])
    record = prepare_candidate_record(source, selected_chain, design_natural)
    packed = featurize_records([record], device=device, eval_chains="masked")
    if packed is None:
        raise RuntimeError(f"Feature construction failed for frozen target {target}")
    tensors, metas = packed
    X, S_label, mask, chain_M, residue_idx, chain_encoding_all, real_pos = tensors
    valid = (
        (mask > 0)
        & (chain_M > 0)
        & (real_pos > 0)
    )
    S_forward = naturalize_tensor_for_input(S_label)
    with torch.no_grad():
        base_logits, _expert_logits = model(
            X, S_forward, mask, chain_M, residue_idx, chain_encoding_all
        )
        raw_probability_full, raw_order_std_full = (
            cyclic_known_sequence_methyl_probabilities(
                model,
                X,
                S_forward,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
                temperature=1.0,
            )
        )
        scaled_probability_full, scaled_order_std_full = (
            cyclic_known_sequence_methyl_probabilities(
                model,
                X,
                S_forward,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
                temperature=temperature,
            )
        )
    positions = torch.where(valid[0])[0]
    if int(positions.numel()) != len(design_natural):
        raise RuntimeError(
            f"Frozen target {target} selected positions changed: "
            f"{int(positions.numel())} != {len(design_natural)}"
        )
    base_indices = S_forward[0, positions]
    raw_probabilities = raw_probability_full[0, positions]
    scaled_probabilities = scaled_probability_full[0, positions]
    raw_order_std = raw_order_std_full[0, positions]
    scaled_order_std = scaled_order_std_full[0, positions]
    base_log_probabilities = F.log_softmax(base_logits[0, positions], dim=-1).gather(
        1, base_indices.unsqueeze(-1)
    ).squeeze(-1)

    final_tokens: List[str] = []
    for base_index, probability in zip(
        base_indices.detach().cpu().tolist(),
        scaled_probabilities.detach().cpu().tolist(),
    ):
        base_token = NATURAL_AA_ALPHABET[int(base_index)]
        methyl_index = NAT_TO_METHYL_ABS.get(int(base_index))
        if methyl_index is not None and float(probability) > threshold:
            final_tokens.append(EXTENDED_AA_ALPHABET[int(methyl_index)])
        else:
            final_tokens.append(base_token)
    final_design = "".join(final_tokens)
    final_positions = [
        index for index, token in enumerate(final_design, start=1) if token.islower()
    ]
    natural_invariant = final_design.upper() == design_natural
    if not natural_invariant:
        raise RuntimeError(f"Final-model annotation changed the natural sequence for {target}")

    exact_annotation_match = final_design == old_design
    return {
        "target_name": target,
        "selected_chain": selected_chain,
        "candidate_origin": "PRE_QC_GENERATION_AUDITED_BY_FINAL_CHECKPOINT",
        "old_design_seq": old_design,
        "retained_result_design_seq": old_design,
        "design_natural_seq": design_natural,
        "final_model_suggested_design_seq": final_design,
        "old_methyl_positions_1based": json.dumps(retained_methyl_positions),
        "retained_design_has_methylation": int(bool(retained_methyl_positions)),
        "final_methyl_positions_1based": json.dumps(final_positions),
        "final_design_methyl_count": len(final_positions),
        "exact_methyl_annotation_match": int(exact_annotation_match),
        "model_bridge_status": (
            "FINAL_MODEL_EXACT_ANNOTATION_MATCH"
            if exact_annotation_match
            else "FINAL_MODEL_DISAGREES_RETAIN_PRE_QC_RESULT"
        ),
        "result_sequence_changed": 0,
        "strict_naturalized_input": 1,
        "temperature": temperature,
        "methyl_threshold": threshold,
        "expert_probabilities_raw": json.dumps(
            [round(float(value), 8) for value in raw_probabilities.detach().cpu().tolist()]
        ),
        "expert_probabilities_temperature_scaled": json.dumps(
            [round(float(value), 8) for value in scaled_probabilities.detach().cpu().tolist()]
        ),
        "expert_probability_order_std_raw": json.dumps(
            [round(float(value), 8) for value in raw_order_std.detach().cpu().tolist()]
        ),
        "expert_probability_order_std_temperature_scaled": json.dumps(
            [round(float(value), 8) for value in scaled_order_std.detach().cpu().tolist()]
        ),
        "annotation_mode": PEPTIDE_ONLY_ANNOTATION_MODE,
        "annotation_context_policy": PEPTIDE_ONLY_ANNOTATION_CONTEXT,
        "annotation_visible_receptor_chains": 0,
        "annotation_order_ensemble_size": int(positions.numel()),
        "base_log_probability_mean": float(base_log_probabilities.mean().item()),
        "global_complex_ca_rmsd": float(evidence["global_rmsd"]),
        "cyclic_peptide_ca_rmsd": float(evidence["cyclic_rmsd"]),
        "frozen_joint_structure_gate_pass": int(
            float(evidence["global_rmsd"]) < 3.0
            and float(evidence["cyclic_rmsd"]) < 3.0
        ),
        "natural_sequence_invariant": int(natural_invariant),
        "structure_reuse_allowed": int(bool(retained_methyl_positions)),
        "structure_action": (
            "KEEP_EXISTING_PDB_NO_HIGHFOLD_RERUN"
            if retained_methyl_positions
            else "WITHDRAW_NON_METHYLATED_ROW_NOT_A_VALID_COMPOUND"
        ),
        "paper_annotation_action": (
            "KEEP_PRE_QC_RESULT_AND_REPORT_FINAL_MODEL_AGREEMENT"
            if exact_annotation_match
            else "KEEP_PRE_QC_RESULT_AND_REPORT_FINAL_MODEL_DISAGREEMENT"
        ),
        "downstream_compound_identity_action": (
            "USE_RETAINED_PRE_QC_DESIGN_SEQ; DO_NOT_SUBSTITUTE_MODEL_SUGGESTION"
        ),
        "model_context_selected_length": int(metas[0]["selected_length"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    plan_path = Path(args.plan).resolve()
    native_path = Path(args.native_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    for required in (model_path, plan_path, native_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    checkpoint = torch.load(model_path, map_location="cpu")
    metadata = (
        dict(checkpoint.get("expert_head_qc_metadata", {}))
        if isinstance(checkpoint, Mapping)
        else {}
    )
    if not (
        str(metadata.get("protocol", "")) == REQUIRED_ORDER_BALANCED_EXPERT_PROTOCOL
        and int(metadata.get("minimum_order_coverage_epochs", 0)) >= 30
        and str(metadata.get("training_decoding_order_policy", ""))
        == "epoch_indexed_cyclic_designed_position_rotation"
        and str(metadata.get("deployment_annotation_policy", ""))
        == "complete_natural_sequence_all_cyclic_rotations_probability_mean"
    ):
        raise RuntimeError(
            "Frozen-target bridge requires the order-balanced v3 expert checkpoint"
        )
    del checkpoint
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frozen_targets = [str(value).upper() for value in plan["frozen_targets"]]
    evidence = plan["frozen_target_evidence"]
    if set(frozen_targets) != set(evidence):
        raise RuntimeError("Frozen target list and evidence keys differ")
    native_rows = read_jsonl(str(native_path))
    native_index = {
        record_name(row, index): row for index, row in enumerate(native_rows)
    }
    missing = sorted(set(frozen_targets) - set(native_index))
    if missing:
        raise RuntimeError("Frozen targets missing from native JSONL: " + ", ".join(missing))

    model = load_v28_model(str(model_path), device)
    model.eval()
    rows = [
        score_one(
            model,
            device,
            native_index[target],
            target,
            evidence[target],
            float(plan["temperature"]),
            float(plan["methyl_threshold"]),
        )
        for target in frozen_targets
    ]
    quality_checks = {
        "frozen_target_count_matches_plan": len(rows) == len(frozen_targets),
        "all_previously_passed_same_joint_structure_gate": all(
            int(row["frozen_joint_structure_gate_pass"]) == 1 for row in rows
        ),
        "all_natural_sequences_invariant": all(
            int(row["natural_sequence_invariant"]) == 1 for row in rows
        ),
        "all_accepted_result_sequences_unchanged": all(
            int(row["result_sequence_changed"]) == 0 for row in rows
        ),
        "all_frozen_compounds_contain_at_least_one_methyl_token": all(
            int(row["retained_design_has_methylation"]) == 1 for row in rows
        ),
        "all_existing_structures_reusable": all(
            int(row["structure_reuse_allowed"]) == 1 for row in rows
        ),
        "all_expert_annotations_use_training_matched_peptide_only_context": all(
            row["annotation_mode"] == PEPTIDE_ONLY_ANNOTATION_MODE
            and row["annotation_context_policy"] == PEPTIDE_ONLY_ANNOTATION_CONTEXT
            and int(row["annotation_visible_receptor_chains"]) == 0
            for row in rows
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    atomic_write_csv(out_dir / "frozen_target_final_model_bridge.csv", rows, list(rows[0]))
    manifest = {
        "quality_gate": quality_gate,
        "protocol": "final_order_balanced_expert_checkpoint_frozen_structure_bridge_v3_peptide_only",
        "model_expert_qc_protocol": metadata.get("protocol"),
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "plan_path": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "native_jsonl": str(native_path),
        "native_jsonl_sha256": file_sha256(native_path),
        "frozen_targets": frozen_targets,
        "frozen_target_rows": len(rows),
        "exact_annotation_matches": sum(
            int(row["exact_methyl_annotation_match"]) for row in rows
        ),
        "final_model_annotation_disagreements": sum(
            row["model_bridge_status"]
            == "FINAL_MODEL_DISAGREES_RETAIN_PRE_QC_RESULT"
            for row in rows
        ),
        "annotation_mode": PEPTIDE_ONLY_ANNOTATION_MODE,
        "annotation_context_policy": PEPTIDE_ONLY_ANNOTATION_CONTEXT,
        "annotation_visible_receptor_chains": 0,
        "train_deployment_context_match": True,
        "quality_checks": quality_checks,
        "scientific_scope": (
            "No sequence or structure was regenerated. Existing PDBs are reused only "
            "when the retained result is actually methylated and the final checkpoint "
            "preserves the identical naturalized sequence. A structure-only row with "
            "zero methyl tokens is withdrawn rather than mislabeled as an accepted "
            "compound. Final-model annotation disagreement is reported and never "
            "substituted silently."
        ),
    }
    atomic_write_json(out_dir / "frozen_target_bridge_manifest.json", manifest)
    print("===== FROZEN TARGET FINAL-MODEL BRIDGE COMPLETE =====", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    print(
        "Frozen structures eligible after methyl gate: "
        f"{sum(int(row['structure_reuse_allowed']) for row in rows)} / {len(rows)}",
        flush=True,
    )
    print(
        f"Exact annotations: {manifest['exact_annotation_matches']} / {len(rows)}; "
        "final-model disagreements: "
        f"{manifest['final_model_annotation_disagreements']} / {len(rows)}",
        flush=True,
    )
    print(f"Bridge CSV: {out_dir / 'frozen_target_final_model_bridge.csv'}", flush=True)
    if quality_gate != "PASS":
        raise RuntimeError("Frozen target final-model bridge failed")


if __name__ == "__main__":
    main()
