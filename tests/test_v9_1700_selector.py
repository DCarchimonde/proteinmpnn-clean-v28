from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


selector = load_module(
    "v9_1700_selector",
    ROOT
    / "paper_clean_v28"
    / "serine_qc_retrain"
    / "23_select_and_audit_v9_1700.py",
)


def valid_candidate(
    *,
    target: str = "1SFI",
    candidate_id: str = "candidate-1",
    design_seq: str = "AcDE",
):
    length = len(design_seq)
    natural = design_seq.upper()
    methylated = [index for index, token in enumerate(design_seq) if token.islower()]
    assert len(methylated) == 1
    methyl_index = methylated[0]

    assert length == 4
    background_by_start = [0.20, 0.25, 0.25, 0.30]
    methyl_by_start = [0.72, 0.74, 0.74, 0.80]
    representation_by_start = [
        [
            methyl_by_start[start] if position == methyl_index else background
            for position in range(length)
        ]
        for start, background in enumerate(background_by_start)
    ]
    columns = list(zip(*representation_by_start))
    minima = [min(values) for values in columns]
    means = [sum(values) / length for values in columns]
    maxima = [max(values) for values in columns]
    spans = [maximum - minimum for minimum, maximum in zip(minima, maxima)]
    representation_std = [
        math.sqrt(sum((value - means[index]) ** 2 for value in values) / length)
        for index, values in enumerate(columns)
    ]

    by_start = [-1.0 - 0.1 * index for index in range(length)]
    start_by_decoder_order = [
        [value - 0.03, value - 0.01, value + 0.01, value + 0.03]
        for value in by_start
    ]
    base_mean = sum(by_start) / length
    base_min = min(by_start)
    base_max = max(by_start)
    base_std = math.sqrt(
        sum((value - base_mean) ** 2 for value in by_start) / length
    )
    positions = [methyl_index + 1]
    return {
        "target_name": target,
        "candidate_id": candidate_id,
        "native_seq": "PPPP"[:length],
        "native_length": length,
        "temperature": 0.5,
        "methyl_threshold": 0.6,
        "design_seq": design_seq,
        "design_natural_seq": natural,
        "design_length": length,
        "length_match": 1,
        "valid_token_gate": 1,
        "design_methyl_count": 1,
        "design_methyl_rate": 1 / length,
        "natural_aa_recovery": 0.0,
        "methyl_positions_1based": json.dumps(positions),
        "methyl_probabilities": json.dumps(means),
        "methyl_probability_representation_min": json.dumps(minima),
        "methyl_probability_representation_max": json.dumps(maxima),
        "methyl_probability_representation_span": json.dumps(spans),
        "methyl_probability_representation_std": json.dumps(representation_std),
        "methyl_probability_representation_std_max": max(representation_std),
        "methyl_probability_representation_by_start": json.dumps(
            representation_by_start
        ),
        "methyl_probability_order_std": json.dumps([0.03] * length),
        "methyl_probability_order_std_max": 0.03,
        "methyl_site_representation_floor_min": minima[methyl_index],
        "methyl_probability_representation_span_max": max(spans),
        "representation_threshold_disagreement_positions_1based": "[]",
        "representation_threshold_disagreement_count": 0,
        "stable_cyclic_release_gate": 1,
        "passes_methylation_hard_gate": 1,
        "eligible_for_new_permeability_screen": 1,
        "seen_in_historical_4115": 0,
        "seen_in_historical_4115_exact": 0,
        "seen_in_historical_4115_naturalized": 0,
        "seen_in_prior_1333": 0,
        "seen_in_prior_1333_exact": 0,
        "seen_in_prior_1333_naturalized": 0,
        "annotation_mode": selector.ANNOTATION_MODE,
        "annotation_context_policy": "peptide_chain_only_no_visible_receptor_chains",
        "annotation_visible_receptor_chains": 0,
        "sampling_context_policy": "native_complex_longest_receptor_visible",
        "annotation_ranking_probability_policy": selector.RANKING_POLICY,
        "annotation_release_probability_policy": selector.RELEASE_POLICY,
        "annotation_representation_ensemble_size": length,
        "annotation_decoder_order_ensemble_size": length,
        "annotation_order_ensemble_size": length,
        "annotation_total_probability_ensemble_size": length**2,
        "cyclic_base_score_protocol": selector.CYCLIC_BASE_PROTOCOL,
        "cyclic_base_floor_policy": selector.CYCLIC_BASE_FLOOR_POLICY,
        "cyclic_base_log_probability_by_start": json.dumps(by_start),
        "cyclic_base_log_probability_start_by_decoder_order": json.dumps(
            start_by_decoder_order
        ),
        "cyclic_base_log_probability_mean": base_mean,
        "cyclic_base_log_probability_min": base_min,
        "cyclic_base_log_probability_max": base_max,
        "cyclic_base_log_probability_span": base_max - base_min,
        "cyclic_base_log_probability_std": base_std,
        "cyclic_base_floor": base_mean - 0.01,
        "cyclic_base_physical_start_count": length,
        "cyclic_base_decoder_order_count_per_start": length,
        "cyclic_base_total_ensemble_size": length**2,
        "cyclic_base_gate_pass": 1,
    }


class CandidateValidationTests(unittest.TestCase):
    def test_complete_stable_representation_min_candidate_passes(self):
        candidate, errors = selector.validate_candidate(
            valid_candidate(), {"1SFI"}
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["_methyl_positions"], [2])
        self.assertEqual(candidate["release_min_argmax_position_1based"], 2)
        self.assertEqual(candidate["ranking_mean_argmax_position_1based"], 2)

    def test_representation_mean_cannot_rescue_a_straddling_minimum(self):
        row = valid_candidate()
        matrix = json.loads(row["methyl_probability_representation_by_start"])
        matrix[0][1] = 0.59
        columns = list(zip(*matrix))
        minima = [min(values) for values in columns]
        means = [sum(values) / len(values) for values in columns]
        maxima = [max(values) for values in columns]
        standard_deviations = [
            math.sqrt(
                sum((value - means[index]) ** 2 for value in values) / len(values)
            )
            for index, values in enumerate(columns)
        ]
        row["methyl_probability_representation_by_start"] = json.dumps(matrix)
        row["methyl_probability_representation_min"] = json.dumps(minima)
        row["methyl_probabilities"] = json.dumps(means)
        row["methyl_probability_representation_max"] = json.dumps(maxima)
        row["methyl_probability_representation_std"] = json.dumps(
            standard_deviations
        )
        spans = [
            maximum - minimum
            for minimum, maximum in zip(minima, maxima)
        ]
        row["methyl_probability_representation_span"] = json.dumps(spans)
        row["methyl_probability_representation_span_max"] = max(spans)
        candidate, errors = selector.validate_candidate(row, {"1SFI"})
        self.assertIsNone(candidate)
        self.assertIn("cyclic_start_threshold_disagreement", errors)
        self.assertIn(
            "lowercase_pattern_not_recomputed_from_representation_min", errors
        )

    def test_threshold_boundary_cannot_hide_behind_loose_summary_tolerance(self):
        row = valid_candidate()
        matrix = json.loads(row["methyl_probability_representation_by_start"])
        matrix[0][1] = 0.59999951
        columns = list(zip(*matrix))
        means = [sum(values) / len(values) for values in columns]
        maxima = [max(values) for values in columns]
        standard_deviations = [
            math.sqrt(
                sum((value - means[index]) ** 2 for value in values) / len(values)
            )
            for index, values in enumerate(columns)
        ]
        stored_minima = [min(values) for values in columns]
        stored_minima[1] = 0.60000001
        spans = [
            maximum - minimum
            for minimum, maximum in zip(stored_minima, maxima)
        ]
        row["methyl_probability_representation_by_start"] = json.dumps(matrix)
        row["methyl_probabilities"] = json.dumps(means)
        row["methyl_probability_representation_min"] = json.dumps(stored_minima)
        row["methyl_probability_representation_max"] = json.dumps(maxima)
        row["methyl_probability_representation_span"] = json.dumps(spans)
        row["methyl_probability_representation_span_max"] = max(spans)
        row["methyl_probability_representation_std"] = json.dumps(
            standard_deviations
        )
        row["methyl_probability_representation_std_max"] = max(
            standard_deviations
        )
        row["methyl_site_representation_floor_min"] = stored_minima[1]
        candidate, errors = selector.validate_candidate(row, {"1SFI"})
        self.assertIsNone(candidate)
        self.assertIn("representation_by_start_summary_recompute_mismatch", errors)

    def test_missing_duplicate_exclusion_evidence_fails_closed(self):
        row = valid_candidate()
        del row["seen_in_prior_1333_naturalized"]
        candidate, errors = selector.validate_candidate(row, {"1SFI"})
        self.assertIsNone(candidate)
        self.assertIn("seen_in_prior_1333_naturalized_missing", errors)

    def test_annotation_context_and_l_squared_counts_are_hard_checks(self):
        row = valid_candidate()
        row["annotation_visible_receptor_chains"] = 1
        row["annotation_total_probability_ensemble_size"] = 15
        candidate, errors = selector.validate_candidate(row, {"1SFI"})
        self.assertIsNone(candidate)
        self.assertIn("annotation_visible_receptor_chains_not_zero", errors)
        self.assertIn("total_probability_ensemble_size_not_L_squared", errors)

    def test_cyclic_base_vector_is_recomputed_and_must_have_l_starts(self):
        row = valid_candidate()
        row["cyclic_base_log_probability_mean"] = -99
        row["cyclic_base_physical_start_count"] = 3
        candidate, errors = selector.validate_candidate(row, {"1SFI"})
        self.assertIsNone(candidate)
        self.assertIn("cyclic_base_summary_recompute_mismatch", errors)
        self.assertIn("cyclic_base_physical_start_count_not_L", errors)

    def test_cyclic_base_floor_comparison_uses_persisted_eight_decimals(self):
        row = valid_candidate()
        mean = float(row["cyclic_base_log_probability_mean"])
        row["cyclic_base_floor"] = round(mean, 8) + 0.00000001
        candidate, errors = selector.validate_candidate(row, {"1SFI"})
        self.assertIsNone(candidate)
        self.assertIn("cyclic_base_mean_below_frozen_target_floor", errors)

    def test_ranking_scalar_tampering_is_rejected(self):
        row = valid_candidate()
        row["methyl_site_representation_floor_min"] = 0.99
        row["methyl_probability_representation_span_max"] = 0.99
        row["methyl_probability_representation_std_max"] = 0.99
        row["methyl_probability_order_std_max"] = 0.99
        candidate, errors = selector.validate_candidate(row, {"1SFI"})
        self.assertIsNone(candidate)
        self.assertIn("methyl_site_representation_floor_scalar_mismatch", errors)
        self.assertIn("representation_span_max_scalar_mismatch", errors)
        self.assertIn("representation_std_max_scalar_mismatch", errors)
        self.assertIn("decoder_order_std_max_scalar_mismatch", errors)

    def test_basic_sequence_diagnostic_tampering_is_rejected(self):
        row = valid_candidate()
        row["design_length"] = 5
        row["native_length"] = 5
        row["length_match"] = 0
        row["valid_token_gate"] = 0
        row["design_methyl_count"] = 2
        row["design_methyl_rate"] = 0.5
        row["natural_aa_recovery"] = 1.0
        candidate, errors = selector.validate_candidate(row, {"1SFI"})
        self.assertIsNone(candidate)
        self.assertIn("design_length_mismatch", errors)
        self.assertIn("native_length_mismatch", errors)
        self.assertIn("length_match_gate_mismatch", errors)
        self.assertIn("valid_token_gate_not_one", errors)
        self.assertIn("design_methyl_count_mismatch", errors)
        self.assertIn("design_methyl_rate_mismatch", errors)
        self.assertIn("natural_aa_recovery_mismatch", errors)

    def test_wrong_length_candidate_is_hard_blocked_even_when_flag_is_honest(self):
        row = valid_candidate()
        row["native_seq"] = "PPPPP"
        row["native_length"] = 5
        row["length_match"] = 0
        candidate, errors = selector.validate_candidate(row, {"1SFI"})
        self.assertIsNone(candidate)
        self.assertIn("length_match_gate_not_one", errors)


class DeduplicationAndSelectionTests(unittest.TestCase):
    @staticmethod
    def internal_row(candidate_id: str, natural: str, score: float):
        return {
            "candidate_id": candidate_id,
            "design_seq": natural,
            "design_natural_seq": natural,
            "_natural_cyclic_key": selector.canonical_rotation(natural),
            "_methyl_positions": [1],
            "_primary_methyl_position": 1,
            "ranking_mean_argmax_position_1based": 1,
            "release_min_argmax_position_1based": 1,
            "_base_score": score,
            "_release_floor": 0.7,
            "_representation_span": 0.1,
        }

    def test_forward_rotations_collapse_to_best_ranked_candidate(self):
        rows = [
            self.internal_row("weaker", "CDEA", -2.0),
            self.internal_row("best", "ACDE", -1.0),
            self.internal_row("other", "ACDF", -1.5),
        ]
        deduplicated = selector.deduplicate_cyclic(rows)
        self.assertEqual([row["candidate_id"] for row in deduplicated], ["best", "other"])

    def test_selection_is_deterministic_under_input_permutation(self):
        rows = [
            self.internal_row("c", "ACDF", -1.2),
            self.internal_row("a", "ACDE", -1.0),
            self.internal_row("b", "ACDG", -1.1),
        ]
        forward = selector.select_diverse(rows, quota=3, frontier_multiplier=1)
        reverse = selector.select_diverse(list(reversed(rows)), quota=3, frontier_multiplier=1)
        self.assertEqual(
            [row["candidate_id"] for row in forward],
            [row["candidate_id"] for row in reverse],
        )

    def test_selection_searches_the_complete_scored_pool(self):
        source = Path(selector.__file__).read_text(encoding="utf-8")
        self.assertIn("frontier = ranked", source)
        self.assertNotIn("ranked[: max(quota, quota * frontier_multiplier)]", source)

    def test_external_exclusion_canonicalizes_forward_rotation(self):
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "old.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["target_name", "design_seq"])
                writer.writeheader()
                writer.writerow({"target_name": "1SFI", "design_seq": "CDEA"})
            natural, cyclic = selector.exclusion_keys([path])
        self.assertIn(("1SFI", "CDEA"), natural)
        self.assertIn(("1SFI", selector.canonical_rotation("ACDE")), cyclic)


class MethylResidueConcentrationTests(unittest.TestCase):
    @staticmethod
    def selected_rows(residues):
        rows = []
        for index, residue in enumerate(residues):
            position = index % 4 + 1
            natural = list("ACDE")
            natural[position - 1] = residue
            marked = list(natural)
            marked[position - 1] = marked[position - 1].lower()
            rows.append(
                {
                    "candidate_id": f"candidate-{index:03d}",
                    "design_seq": "".join(marked),
                    "design_natural_seq": "".join(natural),
                    "_natural_cyclic_key": f"cyclic-key-{index:03d}",
                    "_methyl_positions": [position],
                    "_primary_methyl_position": position,
                    "ranking_mean_argmax_position_1based": position,
                    "release_min_argmax_position_1based": position,
                }
            )
        return rows

    def test_projected_site_share_accepts_string_residue_keys(self):
        self.assertEqual(selector.projected_site_share({}, 0, ["S"]), 1.0)
        self.assertEqual(
            selector.projected_site_share({"S": 1}, 1, ["A"]), 0.5
        )

    def test_all_serine_target_fails_even_with_balanced_positions(self):
        rows = self.selected_rows(["S"] * 100)
        summary = selector.target_summary("3ZGC", rows, rows, quota=100)

        self.assertEqual(summary["maximum_single_position_share"], 0.25)
        self.assertTrue(summary["position_concentration_pass"])
        self.assertEqual(summary["maximum_single_methyl_residue_share"], 1.0)
        self.assertFalse(summary["methyl_residue_concentration_pass"])

    def test_balanced_mixed_methyl_residues_pass_both_concentration_gates(self):
        residues = ["ACDE"[index % 4] for index in range(100)]
        rows = self.selected_rows(residues)
        summary = selector.target_summary("1SFI", rows, rows, quota=100)

        self.assertEqual(summary["maximum_single_position_share"], 0.25)
        self.assertEqual(summary["maximum_single_methyl_residue_share"], 0.25)
        self.assertTrue(summary["position_concentration_pass"])
        self.assertTrue(summary["methyl_residue_concentration_pass"])


class UpstreamPlanTests(unittest.TestCase):
    def test_repository_plan_has_exact_frozen_unique_17_targets(self):
        plan = selector.read_json(selector.DEFAULT_PLAN)
        names = [str(item["target_name"]).upper() for item in plan["targets"]]
        self.assertEqual(len(names), 17)
        self.assertEqual(len(set(names)), 17)
        self.assertEqual(set(names), set(selector.FROZEN_TARGETS))
        self.assertEqual(plan["final_release_quota_per_target"], 100)
        self.assertEqual(plan["initial_stable_pool_quota_per_target"], 120)
        self.assertEqual(
            plan["annotation_context_policy"],
            "peptide_chain_only_no_visible_receptor_chains",
        )
        self.assertEqual(
            plan["sampling_context_policy"],
            "native_complex_longest_receptor_visible",
        )
        self.assertEqual(
            plan["annotation_ranking_probability_policy"],
            selector.RANKING_POLICY,
        )
        self.assertEqual(
            plan["annotation_release_probability_policy"],
            selector.RELEASE_POLICY,
        )
        self.assertTrue(
            all(item["structure_quota"] == 120 for item in plan["targets"])
        )

    @staticmethod
    def write_json(path: Path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def upstream_fixture(self, temp: Path, plan):
        plan_path = temp / "plan.json"
        manifest_path = temp / "manifest.json"
        audit_path = temp / "audit.json"
        model_path = temp / "model.pt"
        self.write_json(plan_path, plan)
        model_path.write_bytes(b"frozen-v9-checkpoint")
        model_hash = selector.sha256_file(model_path)
        manifest = {
            "quality_gate": "PASS",
            "quality_checks": {"generation_check": True},
            "protocol": plan["protocol"],
            "model_expert_qc_protocol": selector.EXPERT_PROTOCOL,
            "model_sha256": model_hash,
            "annotation_stability_audit": {"quality_gate": "PASS"},
        }
        audit = {
            "quality_gate": "PASS",
            "quality_checks": {"heldout_check": True},
            "protocol": selector.AUDIT_PROTOCOL,
            "release_authorization": selector.AUDIT_AUTHORIZATION,
            "model_expert_qc_protocol": selector.EXPERT_PROTOCOL,
            "model_sha256": model_hash,
            "plan_sha256": selector.sha256_file(plan_path),
            "cyclic_representation_ensemble_heldout": {
                "representation_threshold_disagreement_positions": 0
            },
        }
        self.write_json(manifest_path, manifest)
        self.write_json(audit_path, audit)
        return plan_path, manifest_path, manifest, audit_path, audit, model_path

    def test_upstream_gate_rejects_duplicate_or_substituted_target_plan(self):
        plan = selector.read_json(selector.DEFAULT_PLAN)
        plan["targets"][-1]["target_name"] = plan["targets"][0]["target_name"]
        with tempfile.TemporaryDirectory() as temp_name:
            paths = self.upstream_fixture(Path(temp_name), plan)
            checks = selector.validate_upstream(paths[0], plan, *paths[1:])
        self.assertFalse(checks["plan_is_v9_17_target_t05_threshold_06"])

    def test_upstream_gate_binds_checkpoint_bytes(self):
        plan = selector.read_json(selector.DEFAULT_PLAN)
        with tempfile.TemporaryDirectory() as temp_name:
            paths = self.upstream_fixture(Path(temp_name), plan)
            checks = selector.validate_upstream(paths[0], plan, *paths[1:])
        self.assertTrue(all(checks.values()))


class ReleaseViewConsistencyTests(unittest.TestCase):
    FIELDS = [
        "final_release_id",
        "candidate_id",
        "target_name",
        "design_seq",
        "design_natural_seq",
        "methyl_positions_1based",
    ]

    @classmethod
    def rows(cls):
        return [
            {
                "final_release_id": f"v9_1sfi_{index + 1:04d}",
                "candidate_id": f"candidate-{index + 1}",
                "target_name": "1SFI",
                "design_seq": "AcDE",
                "design_natural_seq": "ACDE",
                "methyl_positions_1based": "[2]",
            }
            for index in range(1700)
        ]

    @classmethod
    def write_views(cls, temp: Path, rows):
        detail = temp / "detail.csv"
        concise = temp / "concise.csv"
        fasta = temp / "sequences.fasta"
        selector.atomic_write_csv(detail, rows, cls.FIELDS)
        selector.atomic_write_csv(concise, rows, cls.FIELDS)
        fasta_lines = []
        for row in rows:
            fasta_lines.extend(
                [
                    f">{row['final_release_id']}|{row['target_name']}|"
                    f"candidate={row['candidate_id']}|marked={row['design_seq']}|"
                    f"methyl_positions={row['methyl_positions_1based']}",
                    row["design_natural_seq"],
                ]
            )
        selector.atomic_write_text(fasta, "\n".join(fasta_lines) + "\n")
        return detail, concise, fasta

    def test_reopened_detailed_concise_and_fasta_views_match(self):
        with tempfile.TemporaryDirectory() as temp_name:
            paths = self.write_views(Path(temp_name), self.rows())
            checks = selector.verify_release_views(*paths)
        self.assertTrue(all(checks.values()))

    def test_reopened_view_mismatch_blocks_consistency_gate(self):
        rows = self.rows()
        concise_rows = [dict(row) for row in rows]
        concise_rows[19]["design_seq"] = "ACdE"
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            detail, _unused, fasta = self.write_views(temp, rows)
            concise = temp / "concise.csv"
            selector.atomic_write_csv(concise, concise_rows, self.FIELDS)
            checks = selector.verify_release_views(detail, concise, fasta)
        self.assertFalse(
            checks["reopened_detailed_and_concise_views_match_exactly"]
        )
        self.assertTrue(
            checks["reopened_fasta_ids_headers_and_sequences_match_detailed_csv"]
        )


class SelectorEndToEndTests(unittest.TestCase):
    @staticmethod
    def generated_rows():
        alphabet = "ACDEFGHIKLMNQRSTVWY"
        methyl_residues = "ACDE"
        observed_cyclic = set()
        rows = []
        global_index = 0
        for target in selector.FROZEN_TARGETS:
            for local_index in range(100):
                methyl_index = local_index % 4
                nonce = 0
                while True:
                    digest = hashlib.sha256(
                        f"{target}:{local_index}:{nonce}".encode("ascii")
                    ).digest()
                    natural = [alphabet[value % len(alphabet)] for value in digest[:4]]
                    natural[methyl_index] = methyl_residues[methyl_index]
                    natural_sequence = "".join(natural)
                    cyclic_key = selector.canonical_rotation(natural_sequence)
                    if cyclic_key not in observed_cyclic:
                        observed_cyclic.add(cyclic_key)
                        break
                    nonce += 1
                marked = list(natural_sequence)
                marked[methyl_index] = marked[methyl_index].lower()
                rows.append(
                    valid_candidate(
                        target=target,
                        candidate_id=f"{target.lower()}-{local_index:03d}",
                        design_seq="".join(marked),
                    )
                )
                global_index += 1
        assert global_index == 1700
        return rows

    @staticmethod
    def write_csv(path: Path, rows):
        selector.atomic_write_csv(path, rows, selector.union_fields(rows))

    def test_exact_17_by_100_release_writes_consistent_handoff(self):
        rows = self.generated_rows()
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            candidates_path = temp / "passing_candidates.csv"
            baseline_path = temp / "unique_candidates.csv"
            historical_path = temp / "historical.csv"
            prior_path = temp / "prior.csv"
            manifest_path = temp / "generation_manifest.json"
            audit_path = temp / "heldout_audit.json"
            cyclic_manifest_path = temp / "cyclic_base_manifest.json"
            model_path = temp / "model.pt"
            out_dir = temp / "handoff"
            plan_path = selector.DEFAULT_PLAN.resolve()

            self.write_csv(candidates_path, rows)
            self.write_csv(
                baseline_path,
                [
                    {
                        "target_name": target,
                        "design_seq": "ACDE",
                        "design_natural_seq": "ACDE",
                    }
                    for target in selector.FROZEN_TARGETS
                ],
            )
            self.write_csv(
                historical_path,
                [{"target_name": "1SFI", "design_seq": "PPPA"}],
            )
            self.write_csv(
                prior_path,
                [{"target_name": "1SFI", "design_seq": "PPPC"}],
            )
            model_path.write_bytes(b"v9-promoted-checkpoint")
            model_hash = selector.sha256_file(model_path)
            plan = selector.read_json(plan_path)
            generation_manifest = {
                "quality_gate": "PASS",
                "quality_checks": {"all_generation_checks": True},
                "protocol": plan["protocol"],
                "model_expert_qc_protocol": selector.EXPERT_PROTOCOL,
                "model_sha256": model_hash,
                "annotation_stability_audit": {"quality_gate": "PASS"},
                "historical_design_csv_sha256": selector.sha256_file(
                    historical_path
                ),
                "prior_handoff_csv_sha256": selector.sha256_file(prior_path),
                "methylated_new_candidates_csv_sha256": selector.sha256_file(
                    candidates_path
                ),
                "unique_candidates_csv_sha256": selector.sha256_file(
                    baseline_path
                ),
            }
            selector.atomic_write_json(manifest_path, generation_manifest)
            heldout_audit = {
                "quality_gate": "PASS",
                "quality_checks": {"all_heldout_checks": True},
                "protocol": selector.AUDIT_PROTOCOL,
                "release_authorization": selector.AUDIT_AUTHORIZATION,
                "model_expert_qc_protocol": selector.EXPERT_PROTOCOL,
                "model_sha256": model_hash,
                "plan_sha256": selector.sha256_file(plan_path),
                "cyclic_representation_ensemble_heldout": {
                    "representation_threshold_disagreement_positions": 0
                },
            }
            selector.atomic_write_json(audit_path, heldout_audit)
            floor = float(rows[0]["cyclic_base_floor"])
            cyclic_manifest = {
                "quality_gate": "PASS",
                "protocol": selector.CYCLIC_BASE_PROTOCOL,
                "floor_policy": selector.CYCLIC_BASE_FLOOR_POLICY,
                "model_sha256": model_hash,
                "plan_sha256": selector.sha256_file(plan_path),
                "inputs": {
                    "candidate_csv": {
                        "sha256": selector.sha256_file(candidates_path)
                    },
                    "baseline_csv": {
                        "sha256": selector.sha256_file(baseline_path)
                    },
                    "generation_manifest": {
                        "sha256": selector.sha256_file(manifest_path)
                    },
                },
                "target_summary": [
                    {
                        "target_name": target,
                        "baseline_unique_natural_sequences": 1,
                        "nearest_rank": 1,
                        "floor_fraction": 0.01,
                        "cyclic_base_floor": floor,
                    }
                    for target in selector.FROZEN_TARGETS
                ],
                "artifacts": {
                    "passing_candidates": {
                        "sha256": selector.sha256_file(candidates_path)
                    }
                },
            }
            selector.atomic_write_json(cyclic_manifest_path, cyclic_manifest)

            argv = [
                str(selector.SCRIPT_PATH),
                "--candidates",
                str(candidates_path),
                "--generation-manifest",
                str(manifest_path),
                "--heldout-audit",
                str(audit_path),
                "--cyclic-base-manifest",
                str(cyclic_manifest_path),
                "--plan",
                str(plan_path),
                "--model",
                str(model_path),
                "--exclusion-csv",
                str(historical_path),
                "--exclusion-csv",
                str(prior_path),
                "--out-dir",
                str(out_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                selector.main()

            report = selector.read_json(out_dir / "v9_1700_release_audit.json")
            concise = selector.read_csv(out_dir / "1700_给尚哥_极简.csv")
            detailed = selector.read_csv(out_dir / "1700_详细审计.csv")
            self.assertEqual(report["quality_gate"], "PASS")
            self.assertEqual(report["selected_rows"], 1700)
            self.assertEqual(len(concise), 1700)
            self.assertEqual(len(detailed), 1700)
            self.assertTrue(all(report["quality_checks"].values()))
            self.assertEqual(
                len({row["design_natural_seq"] for row in detailed}), 1700
            )
            self.assertEqual(
                len(
                    {
                        selector.canonical_rotation(row["design_natural_seq"])
                        for row in detailed
                    }
                ),
                1700,
            )


if __name__ == "__main__":
    unittest.main()
