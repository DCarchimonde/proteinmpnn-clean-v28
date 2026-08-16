from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
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

    def test_frozen_source_route_uses_pre_v8_manifests_not_runtime_replay(self):
        def fixed(counts):
            return v8_composer.threshold_metrics_from_counts(counts, 0.6)

        serine_counts = dict(v8_composer.EXPECTED_FROZEN_SERINE_CONFUSION)
        v6_counts = dict(v8_composer.EXPECTED_FROZEN_V6_CONFUSION)
        v7_counts = dict(v8_composer.EXPECTED_FROZEN_V7_CONFUSION)
        v6_non_ser = {
            name: v6_counts[name] - serine_counts[name]
            for name in ("tp", "tn", "fp", "fn")
        }
        v6 = {
            "positions": 1505,
            "threshold": 0.6,
            "deployment_temperature": 0.5,
            "overall_at_threshold": fixed(v6_counts),
            "non_ser_at_threshold": fixed(v6_non_ser),
            "serine": {**fixed(serine_counts), "auc": 0.9475806451612904},
        }
        v7 = {
            "positions": 1505,
            "threshold": 0.6,
            "deployment_temperature": 0.5,
            "overall_at_threshold": fixed(v7_counts),
            "non_ser_at_threshold": fixed(
                {
                    name: v7_counts[name] - serine_counts[name]
                    for name in ("tp", "tn", "fp", "fn")
                }
            ),
            "serine": {**fixed(serine_counts), "auc": 0.956989247311828},
        }
        route = v8_composer.frozen_source_route(
            {"corrected_test": v6},
            {"corrected_test": v7},
            0.6,
            0.5,
        )
        self.assertEqual(
            route["v8_routed"]["overall_at_threshold"]["tp"], 210
        )
        self.assertEqual(
            route["v8_routed"]["overall_at_threshold"]["fp"], 35
        )
        self.assertAlmostEqual(
            route["v8_routed"]["overall_at_threshold"]["recall"],
            210 / 261,
        )
        self.assertAlmostEqual(
            route["v8_routed"]["overall_at_threshold"]["f1"],
            420 / 506,
        )
        self.assertGreaterEqual(
            route["v8_routed"]["serine_auc"], route["v6"]["serine_auc"]
        )
        rows = v8_composer.frozen_source_comparison_rows(route)
        self.assertEqual({row["metric"] for row in rows}, {
            "recall_at_0_6",
            "f1_at_0_6",
            "true_positives_at_0_6",
            "false_negatives_at_0_6",
            "false_positives_at_0_6",
            "true_negatives_at_0_6",
            "serine_auc",
        })

        tampered = dict(v6)
        tampered["overall_at_threshold"] = fixed(
            {"tp": 211, "tn": 1209, "fp": 35, "fn": 50}
        )
        with self.assertRaises(RuntimeError):
            v8_composer.frozen_source_route(
                {"corrected_test": tampered},
                {"corrected_test": v7},
                0.6,
                0.5,
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
            '"frozen_source_route_comparison"',
            source,
        )
        self.assertIn(
            "frozen_source_routed_recall_is_non_inferior_to_v6",
            source,
        )

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
