#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V13 fixed-budget methyl-conditioned generation and 17 x 100 handoff.

This is an honest yield experiment, not a quota top-up engine.  A small pilot
chooses one global product-of-experts strength across all 17 targets; pilot rows
are never eligible for handoff.  Each target then receives exactly 250 fresh
draws.  At least half of all raw draws must pass the strict all-cyclic-start
``representation_min > 0.6`` gate and at least 100 forward-cyclic-unique hits
must remain.  A failed target is reported and blocks the entire handoff; no
additional draws, replayed historical sequences, local search, base filter, or
pre-structure RMSD score can fill the deficit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
GENERATOR_PATH = REPO_ROOT / "paper_clean_v28" / "rerun_t05" / "01_generate_t05_multiseed.py"
REANNOTATOR_PATH = SCRIPT_PATH.with_name("10_reannotate_v6_pool_serine_only_v7.py")
PLAN_PATH = SCRIPT_PATH.with_name("target_plan_v13_fixed_budget_methyl_yield_1700.json")
V13_ROOT = REPO_ROOT / "paper_clean_v28_outputs" / "methyl_yield_v13_1700"
DEFAULT_MODEL = V13_ROOT / "model" / "frankenstein_v28_short_length_balanced_v13.pt"
DEFAULT_MODEL_MANIFEST = V13_ROOT / "model" / "v13_short_length_retrain_manifest.json"
DEFAULT_AUDIT = V13_ROOT / "representation_audit" / "cyclic_representation_audit.json"
DEFAULT_NATIVE = REPO_ROOT / "17_complexes_native.jsonl"
DEFAULT_BEST = (
    REPO_ROOT
    / "paper_clean_v28_outputs"
    / "generated_fasta_clean_auto_single"
    / "best_designs.csv"
)
DEFAULT_OUT = V13_ROOT / "fixed_budget_generation"
V13_EXPERT_PROTOCOL = (
    "canonical_clean_v28_all_expert_heads_cyclic_native_"
    "short_length_balanced_v13"
)
V13_AUDIT_PROTOCOL = "cyclic_native_short_length_balanced_heldout_gate_v13"
V13_AUTHORIZATION = (
    "CYCLIC_NATIVE_V13_VALIDATED_FOR_FIXED_BUDGET_METHYL_YIELD_EVALUATION"
)
VECTOR_FIELDS = (
    "methyl_probabilities",
    "methyl_probability_representation_min",
    "methyl_probability_representation_max",
    "methyl_probability_representation_span",
    "methyl_probability_representation_std",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = union_fields(rows)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def forward_cyclic_identity(sequence: str) -> str:
    if not sequence:
        raise ValueError("Cannot canonicalize an empty cyclic sequence")
    return min(sequence[index:] + sequence[:index] for index in range(len(sequence)))


def target_names(plan: Mapping[str, Any]) -> List[str]:
    names = [str(row["target_name"]).upper() for row in plan["targets"]]
    if (
        len(names) != int(plan["expected_target_count"])
        or len(names) != 17
        or len(set(names)) != 17
    ):
        raise RuntimeError("V13 plan must contain exactly 17 unique targets")
    return names


def validate_contract(
    plan_path: Path,
    model_path: Path,
    model_manifest_path: Path,
    audit_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    plan = read_json(plan_path)
    manifest = read_json(model_manifest_path)
    audit = read_json(audit_path)
    model_sha = sha256_file(model_path)
    if not (
        plan.get("protocol")
        == "v13_fixed_budget_native_methyl_conditioned_17_target_1700"
        and float(plan.get("temperature", -1.0)) == 0.5
        and float(plan.get("methyl_threshold", -1.0)) == 0.6
        and int(plan.get("batch_size", -1)) == 8
        and int(plan.get("final_independent_draws_per_target", -1)) == 250
        and float(plan.get("minimum_raw_strict_methyl_hit_rate_per_target", -1.0))
        == 0.5
        and int(plan.get("minimum_forward_cyclic_unique_strict_hits_per_target", -1))
        == 100
        and int(plan.get("final_release_quota_per_target", -1)) == 100
        and not bool(plan.get("calibration_rows_eligible_for_handoff", True))
        and plan.get("topup_policy")
        == "FORBIDDEN_FIXED_BUDGET_FAILURE_IS_REPORTED_NOT_FILLED"
    ):
        raise RuntimeError("V13 fixed-budget plan contract is incomplete or changed")
    target_names(plan)
    if not (
        manifest.get("quality_gate") == "PASS"
        and manifest.get("protocol") == V13_EXPERT_PROTOCOL
        and manifest.get("checkpoint_ready_for_generation") is True
        and manifest.get("checkpoint_artifact_sha256") == model_sha
        and Path(str(manifest.get("output_checkpoint", ""))).resolve()
        == model_path.resolve()
        and all(bool(value) for value in manifest.get("quality_checks", {}).values())
    ):
        raise RuntimeError("V13 model manifest is absent, failed, or stale")
    if not (
        audit.get("quality_gate") == "PASS"
        and audit.get("protocol") == V13_AUDIT_PROTOCOL
        and audit.get("release_authorization") == V13_AUTHORIZATION
        and audit.get("model_sha256") == model_sha
        and audit.get("plan_sha256") == sha256_file(plan_path)
        and all(bool(value) for value in audit.get("quality_checks", {}).values())
    ):
        raise RuntimeError("V13 representation audit is absent, failed, or stale")
    return plan, manifest, audit


def choose_global_strength(
    summary_rows: Sequence[Mapping[str, Any]], strengths: Sequence[float]
) -> Tuple[float, List[Dict[str, Any]]]:
    ranking: List[Dict[str, Any]] = []
    for strength in strengths:
        rows = [
            row
            for row in summary_rows
            if math.isclose(float(row["guidance_strength"]), float(strength))
        ]
        if len(rows) != 17:
            raise RuntimeError(f"Calibration summary is incomplete for strength {strength}")
        rates = [float(row["raw_strict_methyl_hit_rate"]) for row in rows]
        unique = [int(row["forward_cyclic_unique_strict_hits"]) for row in rows]
        ranking.append(
            {
                "guidance_strength": float(strength),
                "minimum_target_hit_rate": min(rates),
                "mean_target_hit_rate": sum(rates) / len(rates),
                "minimum_target_unique_hits": min(unique),
                "mean_target_unique_hits": sum(unique) / len(unique),
            }
        )
    ranking.sort(
        key=lambda row: (
            -float(row["minimum_target_hit_rate"]),
            -float(row["mean_target_hit_rate"]),
            -int(row["minimum_target_unique_hits"]),
            -float(row["mean_target_unique_hits"]),
            float(row["guidance_strength"]),
        )
    )
    return float(ranking[0]["guidance_strength"]), ranking


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], targets: Sequence[str], strength: float
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for target in targets:
        selected = [row for row in rows if str(row["target_name"]) == target]
        hits = [row for row in selected if int(row["raw_strict_methyl_hit"]) == 1]
        identities = {
            forward_cyclic_identity(str(row["design_seq"])) for row in hits
        }
        result.append(
            {
                "target_name": target,
                "guidance_strength": float(strength),
                "raw_draws": len(selected),
                "raw_strict_methyl_hits": len(hits),
                "raw_strict_methyl_hit_rate": len(hits) / len(selected),
                "forward_cyclic_unique_strict_hits": len(identities),
            }
        )
    return result


def sample_target(
    *,
    target: str,
    features: Sequence[Any],
    metadata: Mapping[str, Any],
    count: int,
    stage: str,
    base_seed: int,
    strength: float,
    batch_size: int,
    model: Any,
    generator: Any,
    torch: Any,
    functional: Any,
    common: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    effective_seed = int(base_seed) * 100_000 + int(generator.stable_target_offset(target))
    generator.torch_seed_all(torch, effective_seed)
    rows: List[Dict[str, Any]] = []
    while len(rows) < count:
        current = min(batch_size, count - len(rows))
        generated = generator.generate_batch(
            model=model,
            features=features,
            batch_size=current,
            temperature=0.5,
            methyl_threshold=0.6,
            torch_module=torch,
            functional=functional,
            extended_alphabet=common["EXTENDED_AA_ALPHABET"],
            x_index=int(common["EXTENDED_AA_TO_INDEX"]["X"]),
            natural_to_methyl=common["NAT_TO_METHYL_ABS"],
            complete_order_fn=common["complete_decoding_order"],
            ensemble_probability_fn=common[
                "cyclic_representation_known_sequence_methyl_probabilities"
            ],
            peptide_only_tensors_fn=common["peptide_only_annotation_tensors"],
            methyl_guidance_positions_1based=None,
            methyl_guidance_strength=float(strength),
            methyl_guidance_forbidden_parent_tokens=None,
            methyl_guidance_context_policy="release_peptide_only",
            methyl_guidance_mode="until_provisional_hit",
        )
        for payload in generated:
            draw = len(rows) + 1
            sequence = str(payload["design_seq"])
            rows.append(
                {
                    "candidate_id": (
                        f"v13_{stage}_{target.lower()}_s{base_seed}_{draw:04d}"
                    ),
                    "target_name": target,
                    "stage": stage,
                    "base_seed": int(base_seed),
                    "effective_seed": effective_seed,
                    "draw_index": draw,
                    "guidance_strength": float(strength),
                    "selected_chain": metadata["selected_chain"],
                    "generation_receptor_chain": metadata[
                        "generation_receptor_chain"
                    ],
                    "structure_receptor_chains": metadata[
                        "structure_receptor_chains"
                    ],
                    "native_seq": metadata["native_peptide_seq"],
                    "native_length": int(metadata["native_peptide_length"]),
                    "raw_strict_methyl_hit": int(
                        generator.stable_cyclic_release_gate(payload)
                    ),
                    "forward_cyclic_identity": forward_cyclic_identity(sequence),
                    "prestructure_base_score_computed": 0,
                    "prestructure_rmsd_computed": 0,
                    **payload,
                }
            )
    return rows


class BatchOneMethylScorer:
    def __init__(
        self,
        model: Any,
        device: Any,
        native_index: Mapping[str, Mapping[str, Any]],
        selected_chains: Mapping[str, str],
        torch: Any,
        common: Mapping[str, Any],
        reannotator: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.native_index = native_index
        self.selected_chains = selected_chains
        self.torch = torch
        self.common = common
        self.reannotator = reannotator

    def score(self, target: str, natural_sequence: str) -> Dict[str, Any]:
        record = self.reannotator.peptide_only_record(
            self.native_index[target],
            self.selected_chains[target],
            natural_sequence,
        )
        packed = self.common["featurize_records"](
            [record], device=self.device, eval_chains="masked"
        )
        if packed is None:
            raise RuntimeError(f"Batch-one feature construction failed for {target}")
        tensors, metas = packed
        if int(metas[0]["selected_length"]) != len(natural_sequence):
            raise RuntimeError(f"Batch-one length mismatch for {target}")
        X, _S, mask, chain_M, residue_idx, chain_encoding = tensors[:6]
        alphabet = self.common["NATURAL_AA_ALPHABET"]
        S_natural = self.torch.tensor(
            [[alphabet.index(token) for token in natural_sequence]],
            device=self.device,
            dtype=self.torch.long,
        )
        with self.torch.no_grad():
            representation = self.common[
                "cyclic_representation_known_sequence_methyl_probabilities"
            ](
                model=self.model,
                X=X,
                S_natural=S_natural,
                mask=mask,
                chain_M=chain_M,
                residue_idx=residue_idx,
                chain_encoding_all=chain_encoding,
                temperature=0.5,
            )
        return self.reannotator.annotation_payload(
            natural_sequence,
            representation,
            0,
            0.6,
            self.common["NATURAL_AA_ALPHABET"],
            self.common["EXTENDED_AA_ALPHABET"],
            self.common["NAT_TO_METHYL_ABS"],
        )


def vector_difference(left: Any, right: Any) -> float:
    try:
        a = [float(value) for value in json.loads(str(left))]
        b = [float(value) for value in json.loads(str(right))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return math.inf
    if len(a) != len(b):
        return math.inf
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def exact_quota(rows: Sequence[Mapping[str, Any]]) -> bool:
    counts = Counter(str(row["target_name"]).upper() for row in rows)
    return len(rows) == 1700 and len(counts) == 17 and set(counts.values()) == {100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(PLAN_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-manifest", default=str(DEFAULT_MODEL_MANIFEST))
    parser.add_argument("--representation-audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--native-jsonl", default=str(DEFAULT_NATIVE))
    parser.add_argument("--best-csv", default=str(DEFAULT_BEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-seed", type=int, default=73001)
    parser.add_argument("--final-seed", type=int, default=91001)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("V13 generation requires NumPy and PyTorch") from exc
    if args.batch_size <= 0 or args.calibration_seed <= 0 or args.final_seed <= 0:
        raise ValueError("Batch size and seeds must be positive")
    if int(args.batch_size) != 8:
        raise ValueError("V13 fixed numerical contract requires --batch-size 8")
    if args.calibration_seed == args.final_seed:
        raise ValueError("Calibration and final seeds must be disjoint")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    plan_path = Path(args.plan).resolve()
    model_path = Path(args.model).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    audit_path = Path(args.representation_audit).resolve()
    native_path = Path(args.native_jsonl).resolve()
    best_path = Path(args.best_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    for required in (
        plan_path,
        model_path,
        model_manifest_path,
        audit_path,
        native_path,
        best_path,
        GENERATOR_PATH,
        REANNOTATOR_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"V13 output is non-empty; inspect it or pass --overwrite: {out_dir}"
            )
        backup = out_dir.with_name(
            out_dir.name + f".superseded.{time.time_ns()}"
        )
        os.replace(out_dir, backup)
        print(f"Preserved prior V13 evidence at: {backup}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan, model_manifest, audit = validate_contract(
        plan_path, model_path, model_manifest_path, audit_path
    )
    targets = target_names(plan)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        if not args.allow_cpu:
            raise RuntimeError("CPU generation requires --allow-cpu")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise RuntimeError("No CUDA device; use --allow-cpu only deliberately")

    generator = load_module("v13_generator", GENERATOR_PATH)
    reannotator = load_module("v13_reannotator", REANNOTATOR_PATH)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from paper_clean_v28.clean_v28_common import (  # pylint: disable=import-outside-toplevel
        EXTENDED_AA_ALPHABET,
        EXTENDED_AA_TO_INDEX,
        NAT_TO_METHYL_ABS,
        NATURAL_AA_ALPHABET,
        complete_decoding_order,
        cyclic_representation_known_sequence_methyl_probabilities,
        featurize_records,
        load_v28_model,
        peptide_only_annotation_tensors,
    )

    common = {
        "EXTENDED_AA_ALPHABET": EXTENDED_AA_ALPHABET,
        "EXTENDED_AA_TO_INDEX": EXTENDED_AA_TO_INDEX,
        "NAT_TO_METHYL_ABS": NAT_TO_METHYL_ABS,
        "NATURAL_AA_ALPHABET": NATURAL_AA_ALPHABET,
        "complete_decoding_order": complete_decoding_order,
        "cyclic_representation_known_sequence_methyl_probabilities": (
            cyclic_representation_known_sequence_methyl_probabilities
        ),
        "featurize_records": featurize_records,
        "peptide_only_annotation_tensors": peptide_only_annotation_tensors,
    }
    best_rows = generator.read_csv(best_path)
    selected_chains = generator.selected_chain_index(best_rows)
    native_rows = generator.read_jsonl(native_path)
    target_records, target_manifest = generator.prepare_target_records(
        native_rows, selected_chains, targets
    )
    metadata_by_target = {
        str(row["target_name"]).upper(): row for row in target_manifest
    }
    native_index = {
        generator.record_name(row, index): row for index, row in enumerate(native_rows)
    }
    model = load_v28_model(str(model_path), device)
    model.eval()

    features_by_target: Dict[str, Sequence[Any]] = {}
    for target in targets:
        packed = featurize_records(
            [target_records[target]], device=device, eval_chains="masked", max_peptide_len=30
        )
        if packed is None:
            raise RuntimeError(f"Feature construction failed for {target}")
        features, metas = packed
        if int(metas[0]["selected_length"]) != int(
            metadata_by_target[target]["native_peptide_length"]
        ):
            raise RuntimeError(f"Peptide geometry/sequence mismatch for {target}")
        features_by_target[target] = features

    strengths = [float(value) for value in plan["guidance_strength_candidates"]]
    pilot_draws = int(plan["calibration_draws_per_target_per_strength"])
    calibration_rows: List[Dict[str, Any]] = []
    calibration_summary: List[Dict[str, Any]] = []
    print("===== V13 GLOBAL GUIDANCE CALIBRATION START =====", flush=True)
    for strength_index, strength in enumerate(strengths):
        strength_rows: List[Dict[str, Any]] = []
        for target in targets:
            rows = sample_target(
                target=target,
                features=features_by_target[target],
                metadata=metadata_by_target[target],
                count=pilot_draws,
                stage=f"calibration_g{strength:g}",
                base_seed=int(args.calibration_seed) + strength_index * 10_000,
                strength=strength,
                batch_size=args.batch_size,
                model=model,
                generator=generator,
                torch=torch,
                functional=functional,
                common=common,
            )
            strength_rows.extend(rows)
            calibration_rows.extend(rows)
            hits = sum(int(row["raw_strict_methyl_hit"]) for row in rows)
            print(
                f"[pilot g={strength:g}] {target}: {hits}/{pilot_draws} "
                f"({hits / pilot_draws:.1%})",
                flush=True,
            )
        calibration_summary.extend(summarize_rows(strength_rows, targets, strength))
        atomic_write_csv(out_dir / "calibration_all_rows.csv", calibration_rows)
        atomic_write_csv(out_dir / "calibration_summary_by_target.csv", calibration_summary)
    selected_strength, strength_ranking = choose_global_strength(
        calibration_summary, strengths
    )
    atomic_write_csv(out_dir / "calibration_strength_ranking.csv", strength_ranking)
    print(f"Selected one global guidance strength: {selected_strength:g}", flush=True)

    final_draws = int(plan["final_independent_draws_per_target"])
    final_rows: List[Dict[str, Any]] = []
    print("===== V13 FIXED-BUDGET FINAL SAMPLING START =====", flush=True)
    for target in targets:
        rows = sample_target(
            target=target,
            features=features_by_target[target],
            metadata=metadata_by_target[target],
            count=final_draws,
            stage="final_fixed_budget",
            base_seed=int(args.final_seed),
            strength=selected_strength,
            batch_size=args.batch_size,
            model=model,
            generator=generator,
            torch=torch,
            functional=functional,
            common=common,
        )
        final_rows.extend(rows)
        atomic_write_csv(out_dir / "final_fixed_budget_all_rows.csv", final_rows)
        hits = sum(int(row["raw_strict_methyl_hit"]) for row in rows)
        unique = len(
            {
                str(row["forward_cyclic_identity"])
                for row in rows
                if int(row["raw_strict_methyl_hit"]) == 1
            }
        )
        print(
            f"[final] {target}: strict={hits}/{final_draws} ({hits/final_draws:.1%}), "
            f"cyclic-unique={unique}",
            flush=True,
        )

    final_summary = summarize_rows(final_rows, targets, selected_strength)
    minimum_rate = float(plan["minimum_raw_strict_methyl_hit_rate_per_target"])
    minimum_unique = int(plan["minimum_forward_cyclic_unique_strict_hits_per_target"])
    for row in final_summary:
        row["minimum_required_raw_hit_rate"] = minimum_rate
        row["minimum_required_cyclic_unique_hits"] = minimum_unique
        row["raw_hit_rate_gate"] = int(
            float(row["raw_strict_methyl_hit_rate"]) >= minimum_rate
        )
        row["unique_hit_gate"] = int(
            int(row["forward_cyclic_unique_strict_hits"]) >= minimum_unique
        )
        row["target_yield_gate"] = int(
            int(row["raw_hit_rate_gate"]) == 1 and int(row["unique_hit_gate"]) == 1
        )
    atomic_write_csv(out_dir / "final_yield_summary_by_target.csv", final_summary)
    failed_targets = [
        str(row["target_name"]) for row in final_summary
        if int(row["target_yield_gate"]) != 1
    ]

    base_manifest: Dict[str, Any] = {
        "protocol": "v13_fixed_budget_methyl_yield_and_1700_handoff",
        "quality_gate": "PENDING",
        "release_status": "BLOCKED_PENDING_BATCH_ONE_REPLAY",
        "temperature": 0.5,
        "methyl_threshold": 0.6,
        "strict_operator": ">",
        "global_guidance_strength": selected_strength,
        "calibration_rows_are_ineligible_for_handoff": True,
        "calibration_draws": len(calibration_rows),
        "final_draws": len(final_rows),
        "final_draws_per_target": final_draws,
        "minimum_raw_strict_methyl_hit_rate_per_target": minimum_rate,
        "minimum_forward_cyclic_unique_strict_hits_per_target": minimum_unique,
        "failed_yield_targets": failed_targets,
        "topup_draws": 0,
        "historical_rows_replayed_into_handoff": 0,
        "directed_or_local_search_rows": 0,
        "prestructure_base_score_policy": "NOT_COMPUTED_OR_USED",
        "prestructure_rmsd_policy": "WAIT_FOR_SHANGGE_STRUCTURES",
        "inputs": {
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "model_manifest": {
                "path": str(model_manifest_path),
                "sha256": sha256_file(model_manifest_path),
            },
            "representation_audit": {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
            },
            "native_jsonl": {"path": str(native_path), "sha256": sha256_file(native_path)},
            "best_csv": {"path": str(best_path), "sha256": sha256_file(best_path)},
        },
        "program": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "dependencies": {
            "generator": {"path": str(GENERATOR_PATH), "sha256": sha256_file(GENERATOR_PATH)},
            "reannotator": {
                "path": str(REANNOTATOR_PATH),
                "sha256": sha256_file(REANNOTATOR_PATH),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "batch_size": int(args.batch_size),
        },
    }
    if failed_targets:
        base_manifest.update(
            {
                "quality_gate": "FAIL",
                "release_status": "BLOCKED_FIXED_BUDGET_YIELD_GATE_FAILED",
                "quality_checks": {
                    "every_target_raw_hit_rate_ge_0_5": all(
                        int(row["raw_hit_rate_gate"]) == 1 for row in final_summary
                    ),
                    "every_target_has_100_cyclic_unique_strict_hits": all(
                        int(row["unique_hit_gate"]) == 1 for row in final_summary
                    ),
                    "no_topup_or_search_was_used": True,
                },
            }
        )
        atomic_write_json(out_dir / "v13_fixed_budget_manifest.json", base_manifest)
        raise RuntimeError(
            "V13 fixed-budget yield gate failed; no handoff was created: "
            + ", ".join(failed_targets)
        )

    selected_rows: List[Dict[str, Any]] = []
    for target in targets:
        seen: set[str] = set()
        target_hits = sorted(
            (
                row for row in final_rows
                if str(row["target_name"]) == target
                and int(row["raw_strict_methyl_hit"]) == 1
            ),
            key=lambda row: int(row["draw_index"]),
        )
        for row in target_hits:
            identity = str(row["forward_cyclic_identity"])
            if identity in seen:
                continue
            seen.add(identity)
            selected = dict(row)
            selected["target_release_index"] = len(seen)
            selected["final_release_id"] = f"V13_{target}_{len(seen):03d}"
            selected_rows.append(selected)
            if len(seen) == 100:
                break
    if not exact_quota(selected_rows):
        raise RuntimeError("V13 internal selection failed exact 17 x 100 quota")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    replay_model = load_v28_model(str(model_path), device)
    replay_model.eval()
    scorer = BatchOneMethylScorer(
        replay_model,
        device,
        native_index,
        selected_chains,
        torch,
        common,
        reannotator,
    )
    replay_rows: List[Dict[str, Any]] = []
    print("===== V13 INDEPENDENT BATCH-ONE REPLAY START =====", flush=True)
    for index, source in enumerate(selected_rows, start=1):
        target = str(source["target_name"])
        natural = str(source["design_natural_seq"]).upper()
        replayed = scorer.score(target, natural)
        differences = {
            field: vector_difference(source.get(field, ""), replayed.get(field, ""))
            for field in VECTOR_FIELDS
        }
        errors: List[str] = []
        if str(source["design_seq"]) != str(replayed["design_seq"]):
            errors.append("design_seq_mismatch")
        if int(replayed["stable_cyclic_release_gate"]) != 1:
            errors.append("strict_methyl_gate_failed")
        if int(replayed["representation_threshold_disagreement_count"]) != 0:
            errors.append("cyclic_threshold_disagreement")
        if any(value > 1e-7 for value in differences.values()):
            errors.append("probability_vector_mismatch")
        replay_rows.append(
            {
                "final_release_id": source["final_release_id"],
                "candidate_id": source["candidate_id"],
                "target_name": target,
                "batch_size": 1,
                "persisted_design_seq": source["design_seq"],
                "replayed_design_seq": replayed["design_seq"],
                "maximum_probability_absolute_difference": max(
                    differences.values(), default=0.0
                ),
                "vector_maximum_absolute_differences": json.dumps(
                    differences, sort_keys=True
                ),
                "row_replay_status": "PASS" if not errors else "FAIL",
                "row_replay_errors": json.dumps(errors, ensure_ascii=False),
            }
        )
        if index == 1 or index % 100 == 0 or index == len(selected_rows):
            failures = sum(row["row_replay_status"] != "PASS" for row in replay_rows)
            print(f"Replayed {index}/1700; failures={failures}", flush=True)
    atomic_write_csv(out_dir / "v13_1700_batch_one_replay.csv", replay_rows)
    replay_failures = [row for row in replay_rows if row["row_replay_status"] != "PASS"]

    concise_rows: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []
    fasta_lines: List[str] = []
    replay_by_id = {str(row["final_release_id"]): row for row in replay_rows}
    for row in selected_rows:
        release_id = str(row["final_release_id"])
        replay = replay_by_id[release_id]
        detailed_rows.append({**row, **replay})
        concise_rows.append(
            {
                "sequence_id": release_id,
                "target_name": row["target_name"],
                "peptide_chain": row["selected_chain"],
                "design_seq": row["design_seq"],
                "design_natural_seq": row["design_natural_seq"],
            }
        )
        fasta_lines.extend(
            [
                f">{release_id}|target={row['target_name']}|chain={row['selected_chain']}",
                str(row["design_seq"]),
            ]
        )

    quality_checks = {
        "calibration_rows_are_excluded": all(
            str(row["stage"]) == "final_fixed_budget" for row in selected_rows
        ),
        "every_target_used_exactly_250_final_draws": (
            len(final_rows) == 17 * 250
            and set(Counter(str(row["target_name"]) for row in final_rows).values())
            == {250}
        ),
        "every_target_raw_hit_rate_ge_0_5": all(
            int(row["raw_hit_rate_gate"]) == 1 for row in final_summary
        ),
        "every_target_has_100_cyclic_unique_strict_hits": all(
            int(row["unique_hit_gate"]) == 1 for row in final_summary
        ),
        "selected_rows_are_exactly_17_x_100": exact_quota(selected_rows),
        "every_selected_row_passed_independent_batch_one_replay": not replay_failures,
        "no_topup_historical_replay_or_directed_search": True,
        "no_prestructure_base_or_rmsd_score": all(
            int(row["prestructure_base_score_computed"]) == 0
            and int(row["prestructure_rmsd_computed"]) == 0
            for row in selected_rows
        ),
    }
    quality_gate = "PASS" if all(quality_checks.values()) else "FAIL"
    if quality_gate == "PASS":
        atomic_write_csv(out_dir / "1700_详细审计.csv", detailed_rows)
        atomic_write_csv(out_dir / "1700_给尚哥_极简.csv", concise_rows)
        atomic_write_text(out_dir / "1700_给尚哥_结构输入.fasta", "\n".join(fasta_lines) + "\n")
    base_manifest.update(
        {
            "quality_gate": quality_gate,
            "release_status": (
                "AUTHORIZED_17_X_100_FOR_SHANGGE_STRUCTURE_GENERATION"
                if quality_gate == "PASS"
                else "BLOCKED_BATCH_ONE_REPLAY_FAILED"
            ),
            "selected_rows": len(selected_rows),
            "batch_one_replay_failures": len(replay_failures),
            "quality_checks": quality_checks,
        }
    )
    artifacts = {
        "calibration_all_rows": out_dir / "calibration_all_rows.csv",
        "calibration_summary": out_dir / "calibration_summary_by_target.csv",
        "calibration_strength_ranking": out_dir / "calibration_strength_ranking.csv",
        "final_all_rows": out_dir / "final_fixed_budget_all_rows.csv",
        "final_yield_summary": out_dir / "final_yield_summary_by_target.csv",
        "batch_one_replay": out_dir / "v13_1700_batch_one_replay.csv",
    }
    if quality_gate == "PASS":
        artifacts.update(
            {
                "detailed_handoff": out_dir / "1700_详细审计.csv",
                "shangge_concise": out_dir / "1700_给尚哥_极简.csv",
                "shangge_fasta": out_dir / "1700_给尚哥_结构输入.fasta",
            }
        )
    base_manifest["artifacts"] = {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in artifacts.items()
    }
    atomic_write_json(out_dir / "v13_fixed_budget_manifest.json", base_manifest)
    print("===== V13 FIXED-BUDGET PIPELINE COMPLETE =====", flush=True)
    print(f"Global guidance strength: {selected_strength:g}", flush=True)
    print(f"Rows for Shangge: {len(selected_rows)} (17 x 100)", flush=True)
    print("RMSD: NOT COMPUTED; waiting for Shangge structures", flush=True)
    print(f"Quality gate: {quality_gate}", flush=True)
    if quality_gate != "PASS":
        raise RuntimeError(
            f"V13 batch-one replay blocked handoff; failures={len(replay_failures)}"
        )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
