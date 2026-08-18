from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RETRAIN_DIR = ROOT / "paper_clean_v28" / "serine_qc_retrain"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v8_composer = load_module(
    "source_scoped_hybrid_v8_composer",
    RETRAIN_DIR / "12_compose_source_scoped_hybrid_v8.py",
)
v8_auditor = load_module(
    "source_scoped_hybrid_v8_auditor",
    RETRAIN_DIR / "13_audit_source_scoped_hybrid_v8.py",
)
v8_search = load_module(
    "source_scoped_hybrid_v8_search",
    RETRAIN_DIR / "14_directed_recovery_search_v8.py",
)
v8_finalizer = load_module(
    "source_scoped_hybrid_v8_finalizer",
    RETRAIN_DIR / "15_finalize_and_audit_recovery_v8.py",
)
v8_bundle = load_module(
    "source_scoped_hybrid_v8_autodl_bundle",
    RETRAIN_DIR / "16_v8_autodl_resume_bundle.py",
)


class FakeTensor:
    """Small tensor-shaped sentinel that keeps these tests Torch-independent."""

    def __init__(self, label: str, cloned: bool = False):
        self.label = label
        self.cloned = cloned

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeTensor(self.label, cloned=True)


class SourceScopedHybridV8Tests(unittest.TestCase):
    def test_v8_programs_parse_and_import_without_torch(self):
        for name in (
            "12_compose_source_scoped_hybrid_v8.py",
            "13_audit_source_scoped_hybrid_v8.py",
            "14_directed_recovery_search_v8.py",
            "15_finalize_and_audit_recovery_v8.py",
        ):
            source = (RETRAIN_DIR / name).read_text(encoding="utf-8")
            ast.parse(source, filename=name)

    def test_source_scoped_tensor_route_is_exact_and_cloned(self):
        serine_index = v8_composer.NATURAL_AA_ALPHABET.index("S")
        for expert_index in range(len(v8_composer.NATURAL_AA_ALPHABET)):
            expected = "v7_serine" if expert_index == serine_index else "v6_non_ser"
            for suffix in ("weight", "bias"):
                self.assertEqual(
                    v8_composer.source_for_state_key(
                        f"experts.{expert_index}.{suffix}"
                    ),
                    expected,
                )
        self.assertEqual(
            v8_composer.source_for_state_key("encoder.layers.0.weight"),
            "canonical_shared",
        )

        keys = (
            "encoder.layers.0.weight",
            "experts.0.weight",
            f"experts.{serine_index}.weight",
            f"experts.{serine_index}.bias",
            "experts.19.bias",
        )
        canonical = {key: FakeTensor(f"canonical:{key}") for key in keys}
        v6 = {key: FakeTensor(f"v6:{key}") for key in keys}
        v7 = {key: FakeTensor(f"v7:{key}") for key in keys}
        result = v8_composer.compose_state_dict(canonical, v6, v7)

        self.assertEqual(
            result["encoder.layers.0.weight"].label,
            "canonical:encoder.layers.0.weight",
        )
        self.assertEqual(result["experts.0.weight"].label, "v6:experts.0.weight")
        self.assertEqual(result["experts.19.bias"].label, "v6:experts.19.bias")
        self.assertEqual(
            result[f"experts.{serine_index}.weight"].label,
            f"v7:experts.{serine_index}.weight",
        )
        self.assertEqual(
            result[f"experts.{serine_index}.bias"].label,
            f"v7:experts.{serine_index}.bias",
        )
        self.assertTrue(all(value.cloned for value in result.values()))
        self.assertTrue(all(result[key] is not canonical[key] for key in result))

    def test_composition_position_audit_rejects_nonfinite_values(self):
        row = {
            "probability_methyl_deployment_scaled": 0.7,
            "probability_order_std": 0.1,
            "probability_representation_std": 0.2,
            "probability_representation_span": 0.3,
        }
        v8_composer.validate_finite_position_rows([row], "finite")
        bad = dict(row)
        bad["probability_representation_span"] = float("nan")
        with self.assertRaises(RuntimeError):
            v8_composer.validate_finite_position_rows([bad], "bad")

    def test_serine_auc_tradeoff_is_reported_without_faking_noninferiority(self):
        v6_counts = {"tp": 10, "tn": 56, "fp": 6, "fn": 2}
        v7_counts = {"tp": 10, "tn": 57, "fp": 5, "fn": 2}

        def summary(auc, observed_counts):
            return {
                "serine": {
                    **observed_counts,
                    "auc": auc,
                }
            }

        v6_auc = 714.0 / 744.0
        v7_auc = 712.0 / 744.0
        audit = v8_composer.serine_auc_tradeoff_audit(
            summary(v6_auc, v6_counts),
            summary(v7_auc, v7_counts),
            summary(v7_auc, v7_counts),
            0.6,
        )
        self.assertEqual(audit["positive_negative_pair_count"], 744)
        self.assertAlmostEqual(audit["v8_minus_v6_auc"], -2.0 / 744.0)
        self.assertAlmostEqual(
            audit["v8_minus_v6_auc_positive_negative_pair_equivalent"], -2.0
        )
        self.assertEqual(audit["v8_auc_direction_vs_v6"], "lower")
        self.assertFalse(audit["v8_threshold_confusion_matches_v6"])
        self.assertTrue(audit["v8_threshold_confusion_is_non_degrading_vs_v6"])
        self.assertTrue(audit["v8_threshold_confusion_matches_v7"])
        self.assertTrue(audit["v8_auc_matches_v7_within_tolerance"])
        self.assertIn("do not assert zero-margin", audit["auc_gate_policy"])

        rows = v8_composer.serine_auc_tradeoff_rows(audit)
        self.assertEqual(
            {row["metric"] for row in rows},
            {
                "serine_auc",
                "serine_auc_positive_negative_pair_equivalent",
                "serine_tp_at_0_6",
                "serine_tn_at_0_6",
                "serine_fp_at_0_6",
                "serine_fn_at_0_6",
            },
        )
        auc_row = next(row for row in rows if row["metric"] == "serine_auc")
        self.assertEqual(
            auc_row["promotion_role"], "report_only_with_absolute_safety_floor"
        )
        fp_row = next(row for row in rows if row["metric"] == "serine_fp_at_0_6")
        self.assertEqual(
            fp_row["promotion_role"],
            "v8_must_be_less_than_or_equal_to_v6",
        )

        changed_counts = {"tp": 9, "tn": 57, "fp": 5, "fn": 3}
        changed = v8_composer.serine_auc_tradeoff_audit(
            summary(v6_auc, v6_counts),
            summary(v7_auc, changed_counts),
            summary(v7_auc, changed_counts),
            0.6,
        )
        self.assertFalse(
            changed["v8_threshold_confusion_is_non_degrading_vs_v6"]
        )
        with self.assertRaises(RuntimeError):
            v8_composer.serine_auc_tradeoff_audit(
                summary(v6_auc, v6_counts),
                summary(float("nan"), v7_counts),
                summary(v7_auc, v7_counts),
                0.6,
            )

    def test_six_residue_radius_two_neighborhood_has_5530_sorted_members(self):
        anchor = "GRKWNC"
        first = v8_search.hamming_neighborhood(anchor, radius=2)
        second = v8_search.hamming_neighborhood(anchor, radius=2)
        expected_count = 1 + 6 * 19 + 15 * 19 * 19
        self.assertEqual(expected_count, 5530)
        self.assertEqual(len(first), expected_count)
        self.assertEqual(first, sorted(first))
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertIn(anchor, first)
        self.assertTrue(all(v8_search.hamming(anchor, row) <= 2 for row in first))

    def test_strict_probability_gate_rounds_to_eight_decimal_places(self):
        self.assertFalse(v8_search.strict_rounded_pass(0.6))
        self.assertFalse(v8_search.strict_rounded_pass(0.600000004))
        self.assertTrue(v8_search.strict_rounded_pass(0.600000006))
        self.assertTrue(v8_search.strict_rounded_pass(0.60000001))
        self.assertFalse(v8_search.strict_rounded_pass(float("nan")))
        self.assertFalse(v8_search.strict_rounded_pass(float("inf")))
        self.assertFalse(v8_search.strict_rounded_pass(1.01))

    def test_baseline_seed_ranking_excludes_non_methylatable_proline_signal(self):
        rows = [
            {
                "target_name": "3WNE",
                "design_natural_seq": "PAAAAA",
                "methyl_probability_max": "0.99",
                "methyl_probabilities": "[0.99, 0.1, 0.1, 0.1, 0.1, 0.1]",
            },
            {
                "target_name": "3WNE",
                "design_natural_seq": "AAAAAA",
                "methyl_probability_max": "0.70",
                "methyl_probabilities": "[0.70, 0.1, 0.1, 0.1, 0.1, 0.1]",
            },
        ]
        self.assertEqual(
            v8_search.top_ranked_sequences(rows, "3WNE"),
            ["AAAAAA", "PAAAAA"],
        )
        self.assertEqual(
            v8_search.actionable_probability_max("PPPPPP", [0.99] * 6), 0.0
        )
        rows[0]["methyl_probabilities"] = "[NaN, 0.1]"
        with self.assertRaises(RuntimeError):
            v8_search.top_ranked_sequences(rows, "3WNE")

    def test_final_annotation_contract_fails_closed_on_nonfinite_vectors(self):
        row = {
            "candidate_id": "finite",
            "design_seq": "aC",
            "design_natural_seq": "AC",
            "design_methyl_count": 1,
            "design_methyl_rate": 0.5,
            "methyl_positions_1based": "[1]",
            "methyl_probabilities": "[0.7, 0.2]",
            "methyl_probability_order_std": "[0.01, 0.02]",
            "methyl_probability_order_std_max": 0.02,
            "methyl_probability_representation_std": "[0.03, 0.04]",
            "methyl_probability_representation_std_max": 0.04,
            "methyl_probability_representation_min": "[0.69, 0.19]",
            "methyl_probability_representation_max": "[0.71, 0.21]",
            "methyl_probability_representation_span": "[0.02, 0.02]",
            "methyl_probability_representation_span_max": 0.02,
            "methyl_probability_min": 0.2,
            "methyl_probability_mean": 0.45,
            "methyl_probability_max": 0.7,
            "methyl_site_probability_min": 0.7,
            "methyl_site_probability_mean": 0.7,
            "methyl_site_probability_max": 0.7,
            "annotation_decoder_order_ensemble_size": 2,
            "annotation_representation_ensemble_size": 2,
        }
        self.assertEqual(v8_finalizer.validate_annotation_row(row), [])
        row["methyl_probabilities"] = "[NaN, 0.2]"
        self.assertTrue(
            any(
                "non-finite" in error
                for error in v8_finalizer.validate_annotation_row(row)
            )
        )
        self.assertFalse(math.isfinite(float("nan")))
        self.assertFalse(v8_finalizer.strict_rounded_pass(0.6000000001))
        self.assertTrue(v8_finalizer.strict_rounded_pass(0.60000001))
        borderline = dict(row)
        borderline.update(
            {
                "candidate_id": "borderline",
                "methyl_probabilities": "[0.6000000001, 0.2]",
                "methyl_probability_min": 0.2,
                "methyl_probability_mean": 0.40000000005,
                "methyl_probability_max": 0.6000000001,
                "methyl_site_probability_min": 0.6000000001,
                "methyl_site_probability_mean": 0.6000000001,
                "methyl_site_probability_max": 0.6000000001,
            }
        )
        borderline_errors = v8_finalizer.validate_annotation_row(borderline)
        self.assertTrue(
            any(
                "strict threshold" in error or "8 decimals" in error
                for error in borderline_errors
            )
        )

    def test_output_paths_cannot_overlap_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            nested = root / "nested" / "output"
            sibling = root.parent / f"{root.name}_sibling"
            self.assertTrue(v8_search.paths_overlap(root, nested))
            self.assertTrue(v8_finalizer.paths_overlap(nested, root))
            self.assertFalse(v8_search.paths_overlap(root, sibling))

    def test_forward_cyclic_identity_collapses_rotations_but_not_reversal(self):
        sequence = "ACDEFG"
        rotations = [
            sequence[index:] + sequence[:index] for index in range(len(sequence))
        ]
        identities = {v8_search.forward_cyclic_identity(row) for row in rotations}
        self.assertEqual(identities, {"ACDEFG"})
        self.assertEqual(v8_search.forward_cyclic_identity("acdefg"), "ACDEFG")
        self.assertNotEqual(
            v8_search.forward_cyclic_identity(sequence),
            v8_search.forward_cyclic_identity(sequence[::-1]),
        )

    def test_nearest_rank_percentile_uses_ceil_rank(self):
        values = list(reversed(range(1, 101)))
        self.assertEqual(v8_search.nearest_rank_percentile(values, 0.01), 1.0)
        self.assertEqual(v8_search.nearest_rank_percentile(values, 0.10), 10.0)
        self.assertEqual(v8_search.nearest_rank_percentile([4, 1, 9], 0.5), 4.0)
        with self.assertRaises(ValueError):
            v8_search.nearest_rank_percentile([], 0.01)

    def test_beam_and_diversity_helpers_are_deterministic(self):
        rows = [
            ("AAA", 0.91, 1),
            ("AAC", 0.91, 2),
            ("ACA", 0.89, 3),
            ("CAA", 0.87, 1),
            ("CCC", 0.86, 2),
            ("CCD", 0.85, 3),
            ("CDC", 0.84, 1),
            ("DCC", 0.83, 2),
        ]
        scored = {
            sequence: {
                "sequence": sequence,
                "maximum_probability": probability,
                "argmax_position_1based": position,
            }
            for sequence, probability, position in rows
        }
        reversed_scored = dict(reversed(list(scored.items())))
        first = v8_search.select_beam(scored, width=5, length=3)
        second = v8_search.select_beam(reversed_scored, width=5, length=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len({row["sequence"] for row in first}), 5)

        ranking = [row[0] for row in rows]
        diverse_first = v8_search.select_diverse_sequences(ranking, count=5)
        diverse_second = v8_search.select_diverse_sequences(ranking, count=5)
        self.assertEqual(diverse_first, diverse_second)
        self.assertEqual(len(diverse_first), len(set(diverse_first)))

    def test_incremental_beam_selection_is_exactly_equivalent_to_frozen_reference(self):
        def frozen_reference(scored, width, length):
            ordered = sorted(
                (dict(value) for value in scored.values()),
                key=lambda row: (
                    -float(row["maximum_probability"]),
                    str(row["sequence"]),
                ),
            )
            selected = []
            seen = set()
            per_position = max(1, min(32, width // max(1, 2 * length)))
            for position in range(1, length + 1):
                candidates = [
                    row
                    for row in ordered
                    if int(row["argmax_position_1based"]) == position
                ]
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
            diversity_pool = [
                row
                for row in ordered[: max(width * 8, width)]
                if str(row["sequence"]) not in seen
            ]
            while diversity_pool and len(selected) < width:
                chosen = max(
                    diversity_pool,
                    key=lambda row: (
                        min(
                            v8_search.hamming(
                                str(row["sequence"]), str(prior["sequence"])
                            )
                            for prior in selected
                        ),
                        float(row["maximum_probability"]),
                        str(row["sequence"]),
                    ),
                )
                selected.append(chosen)
                seen.add(str(chosen["sequence"]))
                diversity_pool.remove(chosen)
            return selected[:width]

        alphabet = v8_search.NATURAL_AA
        for seed in range(5):
            rng = random.Random(seed)
            scored = {}
            for index in range(240):
                value = index
                tokens = []
                for _ in range(7):
                    tokens.append(alphabet[value % len(alphabet)])
                    value //= len(alphabet)
                sequence = "".join(tokens)
                scored[sequence] = {
                    "sequence": sequence,
                    "maximum_probability": round(rng.random(), 8),
                    "argmax_position_1based": rng.randrange(1, 8),
                }
            expected = frozen_reference(scored, width=64, length=7)
            observed = v8_search.select_beam(scored, width=64, length=7)
            self.assertEqual(observed, expected)

    def test_search_evidence_gzip_is_byte_deterministic_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv.gz"
            second = root / "second.csv.gz"
            rows = [{"target_name": "3ZGC", "sequence": "GDEETGE"}]
            fields = ["target_name", "sequence"]
            v8_search.atomic_write_gzip_csv(first, rows, fields)
            v8_search.atomic_write_gzip_csv(second, rows, fields)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            first_json = root / "first.json.gz"
            second_json = root / "second.json.gz"
            payload = {"completed_round": 1, "seen_sequences": ["GDEETGE"]}
            v8_search.write_gzip_json(first_json, payload)
            v8_search.write_gzip_json(second_json, payload)
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())

            artifact = {
                "ledger": {
                    "path": str(first),
                    "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                }
            }
            self.assertTrue(v8_search.artifact_hashes_match(artifact))
            self.assertFalse(v8_search.artifact_hashes_match({}))
            first.write_bytes(first.read_bytes() + b"tamper")
            self.assertFalse(v8_search.artifact_hashes_match(artifact))

    def test_search_and_finalizer_accept_the_complete_v8_model_artifact_contract(self):
        expected = {
            "metric_comparison": "v6_v7_v8_metric_comparison.csv",
            "serine_auc_tradeoff_audit": "serine_auc_tradeoff_audit.csv",
            "metrics_by_residue": "test_metrics_by_residue.csv",
            "position_probabilities": "test_position_probabilities.csv",
        }
        self.assertEqual(v8_search.V8_MODEL_ARTIFACT_FILENAMES, expected)
        self.assertEqual(v8_finalizer.V8_MODEL_ARTIFACT_FILENAMES, expected)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            expected_paths = {}
            for logical_name, filename in expected.items():
                path = root / filename
                path.write_text(logical_name, encoding="utf-8")
                artifacts[logical_name] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                expected_paths[logical_name] = path
            for module in (v8_search, v8_finalizer):
                self.assertTrue(
                    module.artifact_map_matches_exact_paths(
                        artifacts, expected_paths
                    )
                )
                legacy = dict(artifacts)
                legacy.pop("serine_auc_tradeoff_audit")
                self.assertFalse(
                    module.artifact_map_matches_exact_paths(legacy, expected_paths)
                )

    def test_zgc_resume_state_is_reconstructed_from_ledgers(self):
        def provenance(sequence, source="test_anchor"):
            return {
                "generation_kind": "frozen_initial_anchor",
                "parent_sequence": sequence,
                "edit_distance": 0,
                "mutation_positions_1based": "[]",
                "rng_seed": "",
                "rng_draw_index": "",
                "anchor_source": source,
            }

        def scored(sequence, probability, passes, stage, source):
            argmax = next(
                (index for index, token in enumerate(sequence) if token != "P"), 0
            )
            row = {
                "target_name": "3ZGC",
                "sequence": sequence,
                "search_stage": stage,
                "maximum_probability": probability,
                "argmax_position_1based": argmax + 1,
                "argmax_residue": sequence[argmax],
                "passes_strict_probability": passes,
            }
            row.update(source)
            return row

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial_provenance = {
                "AAAAAAA": provenance("AAAAAAA", "current_v8_baseline_top"),
                "CCCCCCC": provenance("CCCCCCC", "diverse_current_v8_baseline_seed"),
            }
            initial = [
                scored(
                    sequence,
                    0.7 if sequence == "AAAAAAA" else 0.5,
                    int(sequence == "AAAAAAA"),
                    "beam_initial_anchors",
                    initial_provenance[sequence],
                )
                for sequence in sorted(initial_provenance)
            ]
            v8_search.atomic_write_gzip_csv(
                root / "3zgc_round_00_initial.csv.gz", initial, list(initial[0])
            )
            initial_map = {row["sequence"]: row for row in initial}
            initial_beam = v8_search.select_beam(initial_map, width=2, length=7)
            generated = v8_search.zgc_round_provenance(
                initial_beam, 1, 0, np
            )
            expected_new = sorted(set(generated) - set(initial_map))
            strict_hit = next(
                sequence
                for sequence in expected_new
                if sequence[0] != "P"
            )
            round_one = [
                scored(
                    sequence,
                    0.8 if sequence == strict_hit else 0.5,
                    int(sequence == strict_hit),
                    "beam_round_01",
                    generated[sequence],
                )
                for sequence in expected_new
            ]
            ground_truth_by_stage = {
                "beam_initial_anchors": {
                    row["sequence"]: dict(row) for row in initial
                },
                "beam_round_01": {
                    row["sequence"]: dict(row) for row in round_one
                },
            }

            def fake_score_minimal(target, sequences, stage):
                self.assertEqual(target, "3ZGC")
                source = ground_truth_by_stage[stage]
                return {
                    sequence: {
                        field: source[sequence][field]
                        for field in (
                            "target_name",
                            "sequence",
                            "search_stage",
                            "maximum_probability",
                            "argmax_position_1based",
                            "argmax_residue",
                            "passes_strict_probability",
                        )
                    }
                    for sequence in sorted(set(sequences))
                }

            v8_search.atomic_write_gzip_csv(
                root / "3zgc_round_01.csv.gz", round_one, list(round_one[0])
            )
            combined = {row["sequence"]: row for row in initial_beam}
            combined.update({row["sequence"]: row for row in round_one})
            beam = v8_search.select_beam(combined, width=2, length=7)
            qualified = [
                row for row in [*initial, *round_one]
                if int(row["passes_strict_probability"])
            ]
            trace = [
                {
                    "target_name": "3ZGC",
                    "stage": "beam_initial_anchors",
                    "generated_unique": 2,
                    "newly_scored": 2,
                    "strict_probability_hits": 1,
                    "maximum_probability": 0.7,
                },
                {
                    "target_name": "3ZGC",
                    "stage": "beam_round_01",
                    "generated_unique": len(generated),
                    "newly_scored": len(round_one),
                    "strict_probability_hits": 2,
                    "maximum_probability": 0.8,
                },
            ]
            checkpoint_dir = root / "checkpoints"
            checkpoint = checkpoint_dir / "3zgc_round_01.json.gz"
            payload = {
                "config_sha256": "cfg",
                "completed_round": 1,
                "seen_sequences": sorted(set(initial_map) | set(expected_new)),
                "beam": beam,
                "qualified": qualified,
                "trace_rows": trace,
            }
            v8_search.write_gzip_json(checkpoint, payload)
            completed, seen, rebuilt_beam, rebuilt_qualified, rebuilt_trace = (
                v8_search.reconstruct_and_validate_zgc_resume(
                    root,
                    [checkpoint],
                    "cfg",
                    2,
                    initial_provenance,
                    0,
                    np,
                    fake_score_minimal,
                )
            )
            self.assertEqual(completed, 1)
            self.assertEqual(seen, set(initial_map) | set(expected_new))
            self.assertEqual(rebuilt_beam, beam)
            self.assertEqual(set(rebuilt_qualified), {"AAAAAAA", strict_hit})
            self.assertEqual(rebuilt_trace, trace)

            tampered_round = [dict(row) for row in round_one]
            tampered_round[0]["parent_sequence"] = "GGGGGGG"
            v8_search.atomic_write_gzip_csv(
                root / "3zgc_round_01.csv.gz",
                tampered_round,
                list(tampered_round[0]),
            )
            with self.assertRaises(RuntimeError):
                v8_search.reconstruct_and_validate_zgc_resume(
                    root,
                    [checkpoint],
                    "cfg",
                    2,
                    initial_provenance,
                    0,
                    np,
                    fake_score_minimal,
                )
            v8_search.atomic_write_gzip_csv(
                root / "3zgc_round_01.csv.gz", round_one, list(round_one[0])
            )
            tampered_score = [dict(row) for row in round_one]
            tampered_score[0]["maximum_probability"] = 0.55
            v8_search.atomic_write_gzip_csv(
                root / "3zgc_round_01.csv.gz",
                tampered_score,
                list(tampered_score[0]),
            )
            with self.assertRaises(RuntimeError):
                v8_search.reconstruct_and_validate_zgc_resume(
                    root,
                    [checkpoint],
                    "cfg",
                    2,
                    initial_provenance,
                    0,
                    np,
                    fake_score_minimal,
                )
            v8_search.atomic_write_gzip_csv(
                root / "3zgc_round_01.csv.gz", round_one, list(round_one[0])
            )
            payload["qualified"].append(
                scored(
                    "EEEEEEE",
                    0.9,
                    1,
                    "injected",
                    provenance("EEEEEEE"),
                )
            )
            v8_search.write_gzip_json(checkpoint, payload)
            with self.assertRaises(RuntimeError):
                v8_search.reconstruct_and_validate_zgc_resume(
                    root,
                    [checkpoint],
                    "cfg",
                    2,
                    initial_provenance,
                    0,
                    np,
                    fake_score_minimal,
                )

    def test_finalizer_artifact_root_and_record_name_guards(self):
        self.assertEqual(v8_finalizer.record_name({"pdb": "3zgc"}, 0), "3ZGC")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "artifact.txt"
            inside.write_text("evidence", encoding="utf-8")
            leaf = {
                "path": str(inside),
                "sha256": hashlib.sha256(inside.read_bytes()).hexdigest(),
            }
            self.assertTrue(
                v8_finalizer.artifacts_are_hash_pinned_under({"x": leaf}, root)
            )
            self.assertFalse(v8_finalizer.artifacts_are_hash_pinned_under({}, root))
            temporary = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=root.parent, delete=False
            )
            outside = Path(temporary.name)
            try:
                temporary.write("outside")
                temporary.close()
                outside_leaf = {
                    "path": str(outside),
                    "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                }
                self.assertFalse(
                    v8_finalizer.artifacts_are_hash_pinned_under(
                        {"x": outside_leaf}, root
                    )
                )
            finally:
                if not temporary.closed:
                    temporary.close()
                outside.unlink(missing_ok=True)

    def test_runner_reuses_sources_without_training_v6_or_force_switch(self):
        runner = (
            ROOT
            / "paper_clean_v28"
            / "run_serine_qc_source_scoped_hybrid_v8.ps1"
        )
        if not runner.is_file():
            self.skipTest("V8 launcher is being assembled in parallel")
        source = runner.read_text(encoding="utf-8")
        self.assertNotIn("[switch]$Force", source)
        self.assertNotIn("if ($Force", source)
        self.assertIn("02_retrain_canonical_expert_heads.py", source)
        self.assertNotIn(
            '$Arguments = @(\n            $TrainerProgram,', source
        )
        self.assertNotIn('"V8 expert retrain"', source)
        self.assertNotIn("08_resume_cyclic_representation_v6_quota.py", source)
        self.assertNotIn("09_finalize_cyclic_representation_v6_exhaustion.py", source)
        self.assertNotIn("run_serine_qc_cyclic_representation_v6.ps1", source)
        self.assertIn("12_compose_source_scoped_hybrid_v8.py", source)
        self.assertIn("14_directed_recovery_search_v8.py", source)
        self.assertIn("15_finalize_and_audit_recovery_v8.py", source)
        self.assertIn('"--recovery-mode"', source)
        self.assertIn('"--summary-score-label", "v8"', source)
        self.assertIn("SearchBatchSize=64 and BaseBatchSize=32", source)
        self.assertIn("position_auditor_program_sha256", source)
        self.assertIn('Filter "3zgc_round_*.json.gz"', source)

    def test_runner_accepts_the_pinned_legacy_v6_manifest_without_v7_scope_fields(self):
        runner = (
            ROOT
            / "paper_clean_v28"
            / "run_serine_qc_source_scoped_hybrid_v8.ps1"
        )
        source = runner.read_text(encoding="utf-8")
        self.assertIn("function Assert-LegacyV6SourceModel", source)
        self.assertIn(
            "$V6Source = Assert-LegacyV6SourceModel "
            "$V6Checkpoint $V6ExpertManifest $V6ExpertProtocol",
            source,
        )
        self.assertNotIn("$V6Source = Assert-SourceModel", source)
        self.assertIn("$ExpertIndex -lt 20", source)
        self.assertIn(
            'Assert-SameStringSet -Observed @($Manifest.changed_non_expert_keys) '
            '-Expected @()',
            source,
        )
        self.assertIn(
            "all 20 expert linear heads are retrained",
            source,
        )
        self.assertIn(
            "-not [bool]$Manifest.training.cyclic_representation_augmentation",
            source,
        )
        self.assertIn(
            '$V7Source = Assert-SourceModel $V7Checkpoint',
            source,
        )

    def test_runner_executes_embedded_python_without_powershell_c_quoting(self):
        runner = (
            ROOT
            / "paper_clean_v28"
            / "run_serine_qc_source_scoped_hybrid_v8.ps1"
        )
        source = runner.read_text(encoding="utf-8")
        self.assertIn("function Invoke-PythonProgram", source)
        self.assertIn(
            'Invoke-PythonProgram $ResolvedPython $Probe "Python/PyTorch preflight"',
            source,
        )
        self.assertIn(
            'Invoke-PythonProgram $ResolvedPython $CudaProbe "CUDA preflight"',
            source,
        )
        self.assertNotIn('@("-c", $Probe)', source)
        self.assertNotIn('"CUDA preflight" @("-c"', source)
        self.assertNotIn("NaturalExpertTokensJson", source)
        self.assertIn(
            '"--expected-active-expert-tokens-csv", $NaturalExpertTokensCsv',
            source,
        )
        programs = re.findall(
            r"\$(?:Probe|CudaProbe) = '([^'\r\n]+)'",
            source,
        )
        self.assertEqual(len(programs), 2)
        for program in programs:
            compile(program, "<V8 PowerShell embedded Python>", "exec")

    def test_runner_safely_retries_only_the_exact_pre_fix_auc_failure(self):
        runner = (
            ROOT
            / "paper_clean_v28"
            / "run_serine_qc_source_scoped_hybrid_v8.ps1"
        )
        source = runner.read_text(encoding="utf-8")
        self.assertIn("function Assert-RetryablePreFixV8ModelFailure", source)
        self.assertIn(
            'Assert-SameStringSet -Observed $FalseChecks -Expected '
            '@("serine_auc_is_non_inferior_to_v6")',
            source,
        )
        self.assertIn('$Arguments += "--overwrite"', source)
        self.assertIn(
            '"serine_auc_tradeoff_audit"',
            source,
        )
        self.assertIn(
            "serine_threshold_operating_point_is_non_degrading_vs_v6",
            source,
        )
        self.assertNotIn("frozen_source_route_comparison", source)

    def test_search_budget_trace_and_candidate_provenance_are_hard_gated(self):
        source = (RETRAIN_DIR / "14_directed_recovery_search_v8.py").read_text(
            encoding="utf-8"
        )
        for token in (
            '"--zgc-rounds": (int(args.zgc_rounds), 6)',
            '"--zgc-beam-width": (int(args.zgc_beam_width), 512)',
            '"--zgc-offspring-per-round": (int(args.zgc_offspring_per_round), 4096)',
            '"trace_rows": trace_rows',
            '"complete_search_ledgers_and_checkpoints_are_persisted"',
            '"DETERMINISTIC_DIRECTED_SEARCH_NO_AUTOREGRESSIVE_SAMPLING"',
            '"base_log_probability_mean_all_orders"',
            '"qualified_full_rescore_absolute_difference"',
            '"deterministic_algorithms_enabled"',
        ):
            self.assertIn(token, source)
        self.assertNotIn("hybrid_metadata = dict(v7_metadata)", (
            RETRAIN_DIR / "12_compose_source_scoped_hybrid_v8.py"
        ).read_text(encoding="utf-8"))

    def test_cross_runtime_ledger_tolerance_never_changes_strict_pass_bits(self):
        persisted = [{
            "target_name": "3ZGC",
            "sequence": "AAAAAAA",
            "search_stage": "round",
            "maximum_probability": 0.61000000,
            "argmax_position_1based": 1,
            "argmax_residue": "A",
            "passes_strict_probability": 1,
        }]

        def close_score(_target, sequences, _stage):
            return {
                sequence: {
                    **persisted[0],
                    "sequence": sequence,
                    "maximum_probability": 0.61000150,
                }
                for sequence in sequences
            }

        with self.assertRaises(RuntimeError):
            v8_search.validate_ledger_scores_against_model(
                persisted, "3ZGC", "round", close_score
            )
        audit = v8_search.validate_ledger_scores_against_model(
            persisted, "3ZGC", "round", close_score, 2e-6
        )
        self.assertEqual(audit["rows"], 1)
        self.assertLessEqual(audit["maximum_absolute_probability_difference"], 2e-6)

        def changed_gate(_target, sequences, _stage):
            return {
                sequence: {
                    **persisted[0],
                    "sequence": sequence,
                    "maximum_probability": 0.59999999,
                    "passes_strict_probability": 0,
                }
                for sequence in sequences
            }

        with self.assertRaises(RuntimeError):
            v8_search.validate_ledger_scores_against_model(
                persisted, "3ZGC", "round", changed_gate, 0.02
            )

    def test_autodl_resume_bundle_and_runner_preserve_full_destination_reaudit(self):
        self.assertEqual(
            v8_bundle.SOURCE_COMMIT,
            "53ce92e5238d717fc982357b4c58f65538a8f710",
        )
        self.assertEqual(v8_bundle.RESCORE_TOLERANCE, 2e-6)
        payload = {"a": {"b": ["old"]}}
        v8_bundle.set_pointer(payload, ["a", "b", 0], "new")
        self.assertEqual(payload, {"a": {"b": ["new"]}})
        with self.assertRaises(RuntimeError):
            v8_bundle.safe_member("../escape")

        runner = (ROOT / "run_v8_autodl_resume.sh").read_text(encoding="utf-8")
        self.assertIn("--portable-resume-manifest", runner)
        self.assertIn("15_finalize_and_audit_recovery_v8.py", runner)
        self.assertIn("package-review", runner)
        self.assertIn("--batch-size 64", runner)
        self.assertIn("--base-batch-size 32", runner)
        self.assertIn("V8 AUTODL RUNTIME PREFLIGHT PASSED", runner)
        self.assertIn("current_imported_file_hashes", runner)
        self.assertIn("OMP_NUM_THREADS=16", runner)

        bundle_source = (
            RETRAIN_DIR / "16_v8_autodl_resume_bundle.py"
        ).read_text(encoding="utf-8")
        self.assertIn("search.DEFAULT_PRIOR", bundle_source)
        self.assertIn(
            '"current_imported_file_hashes": current_imported_file_hashes',
            bundle_source,
        )

        recovery = (
            ROOT / "recover_v8_autodl_legacy_bundle.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "0e916c311956108269b57960fc2ca18d388a3713",
            recovery,
        )
        self.assertIn("COMPLETE PRE-RUN AUDIT PASSED", recovery)
        self.assertIn("source_config", recovery)
        self.assertIn("input_hashes", recovery)
        self.assertIn("OMP_NUM_THREADS=16", recovery)

        search_source = (
            RETRAIN_DIR / "14_directed_recovery_search_v8.py"
        ).read_text(encoding="utf-8")
        finalizer_source = (
            RETRAIN_DIR / "15_finalize_and_audit_recovery_v8.py"
        ).read_text(encoding="utf-8")
        self.assertIn("class ProgressBar", search_source)
        self.assertIn("destination_full_ledger_reaudit_required", search_source)
        self.assertIn(
            "portable_cross_runtime_full_ledger_reaudit_passes_within_tolerance",
            finalizer_source,
        )

    def test_round_six_portable_evidence_matches_the_frozen_missing_target_subset(self):
        zgc_only = v8_search.portable_resume_expected_evidence_names(["3ZGC"])
        self.assertNotIn("3wne_exact_search_all.csv.gz", zgc_only)
        self.assertIn("3zgc_round_06.csv.gz", zgc_only)
        self.assertIn("3zgc_round_06.json.gz", zgc_only)
        self.assertEqual(len(zgc_only), 13)

        both = v8_search.portable_resume_expected_evidence_names(
            ["3WNE", "3ZGC"]
        )
        self.assertEqual(both - zgc_only, {"3wne_exact_search_all.csv.gz"})
        with self.assertRaises(RuntimeError):
            v8_search.portable_resume_expected_evidence_names(["3WNE"])
        with self.assertRaises(RuntimeError):
            v8_search.portable_resume_expected_evidence_names(
                ["3ZGC", "UNSUPPORTED"]
            )

    def test_finalizer_cannot_emit_handoff_or_permeability_inputs(self):
        source = (
            RETRAIN_DIR / "15_finalize_and_audit_recovery_v8.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("03_select_structure_first_handoff.py", source)
        self.assertNotIn("structure_tasks_for_shangge.csv", source)
        self.assertNotIn("permeability_input_manifest.csv\" =", source)
        self.assertIn('"structure_handoff_is_not_created"', source)
        self.assertIn('"permeability_remains_deferred"', source)
        self.assertIn('"NOT_CREATED_PENDING_MANUAL_REVIEW"', source)
        self.assertIn(
            '"DEFERRED_UNTIL_RETURNED_STRUCTURES_PASS_BOTH_RMSD_GATES"', source
        )


if __name__ == "__main__":
    unittest.main()
