from __future__ import annotations

import ast
import csv
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETRAIN_DIR = ROOT / "paper_clean_v28" / "serine_qc_retrain"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v7_reannotator = load_module(
    "serine_only_v7_reannotator",
    RETRAIN_DIR / "10_reannotate_v6_pool_serine_only_v7.py",
)
v7_auditor = load_module(
    "serine_only_v7_auditor",
    RETRAIN_DIR / "11_triple_audit_serine_only_v7.py",
)

from nmethyl.utils.nmethyl_config import (  # noqa: E402
    EXTENDED_AA_ALPHABET,
    NATURAL_AA_ALPHABET,
    NMETHYL_TO_NATURAL_MAPPING,
)


NAT_TO_METHYL_ABS = {}
for methyl_relative_index, natural_index in NMETHYL_TO_NATURAL_MAPPING.items():
    NAT_TO_METHYL_ABS.setdefault(
        int(natural_index), len(NATURAL_AA_ALPHABET) + int(methyl_relative_index)
    )


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class SerineOnlyV7Tests(unittest.TestCase):
    def test_all_v7_python_programs_parse_without_importing_torch(self):
        for name in (
            "02_retrain_canonical_expert_heads.py",
            "07_audit_cyclic_representation_equivariance.py",
            "10_reannotate_v6_pool_serine_only_v7.py",
            "11_triple_audit_serine_only_v7.py",
        ):
            ast.parse((RETRAIN_DIR / name).read_text(encoding="utf-8"), filename=name)

    def test_serine_only_protocol_changes_exactly_ser_weight_and_bias(self):
        trainer = (RETRAIN_DIR / "02_retrain_canonical_expert_heads.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('SERINE_EXPERT_INDEX = NATURAL_AA_ALPHABET.index("S")', trainer)
        self.assertIn('choices=("all", "serine-only")', trainer)
        self.assertIn("expected_changed_keys", trainer)
        self.assertIn("maximum_non_ser_probability_difference", trainer)
        self.assertIn("all_non_target_tensors_are_bitwise_parent_identical", trainer)

    def test_strict_threshold_and_no_proline_methyl_token(self):
        representation = {
            "mean": [FakeTensor([0.6, 0.99, 0.60000001, 0.1])],
            "decoder_order_std_mean": [FakeTensor([0.0, 0.0, 0.0, 0.0])],
            "representation_std": [FakeTensor([0.0, 0.0, 0.0, 0.0])],
            "representation_min": [FakeTensor([0.6, 0.99, 0.60000001, 0.1])],
            "representation_max": [FakeTensor([0.6, 0.99, 0.60000001, 0.1])],
            "representation_span": [FakeTensor([0.0, 0.0, 0.0, 0.0])],
            "representation_count": [FakeTensor([4.0, 4.0, 4.0, 4.0])],
        }
        payload = v7_reannotator.annotation_payload(
            "SPRA",
            representation,
            0,
            0.6,
            NATURAL_AA_ALPHABET,
            EXTENDED_AA_ALPHABET,
            NAT_TO_METHYL_ABS,
        )
        self.assertEqual(payload["design_seq"], "SPrA")
        self.assertEqual(json.loads(payload["methyl_positions_1based"]), [3])
        self.assertEqual(payload["design_methyl_count"], 1)
        self.assertEqual(payload["sampling_path_methyl_probabilities"], "")

    def test_preserved_source_validator_requires_unique_ids_and_natural_rows(self):
        rows = [
            {
                "candidate_id": "a",
                "target_name": "T1",
                "selected_chain": "C",
                "design_natural_seq": "SRA",
                "design_seq": "SrA",
                "design_length": "3",
                "native_length": "3",
            },
            {
                "candidate_id": "b",
                "target_name": "T2",
                "selected_chain": "D",
                "design_natural_seq": "GDE",
                "design_seq": "GdE",
                "design_length": "3",
                "native_length": "3",
            },
        ]
        result = v7_reannotator.validate_source_rows(
            rows,
            ["T1", "T2"],
            expected_raw_rows=2,
            expected_target_count=2,
        )
        self.assertEqual(result["unique_target_natural_sequence_groups"], 2)
        rows[1]["candidate_id"] = "a"
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            v7_reannotator.validate_source_rows(
                rows,
                ["T1", "T2"],
                expected_raw_rows=2,
                expected_target_count=2,
            )

    def test_cyclic_distance_matrix_detects_physical_shift(self):
        coordinates = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 1.0],
        ]
        reference = v7_auditor.distance_matrix(coordinates)
        shifted_coordinates = coordinates[1:] + coordinates[:1]
        shifted = v7_auditor.distance_matrix(shifted_coordinates)
        scores = [
            v7_auditor.cyclic_matrix_rmse(reference, shifted, shift)
            for shift in range(4)
        ]
        self.assertEqual(scores.index(min(scores)), 3)
        self.assertAlmostEqual(min(scores), 0.0, places=12)

    def test_real_3av_backbones_keep_the_same_physical_cyclic_index(self):
        native_rows = {
            row["name"].upper(): row
            for row in (
                json.loads(line)
                for line in (ROOT / "17_complexes_native.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        }
        selected_chains = {}
        with (
            ROOT
            / "paper_clean_v28_outputs"
            / "generated_fasta_clean_auto_single"
            / "best_designs.csv"
        ).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                target = str(row["target_name"]).upper()
                if target in v7_auditor.AV_FAMILY:
                    selected_chains.setdefault(target, str(row["selected_chains"]))

        reference_chain = selected_chains["3AV9"]
        reference = v7_auditor.distance_matrix(
            native_rows["3AV9"][f"CA_chain_{reference_chain}"]
        )
        observed = {}
        for target in sorted(v7_auditor.AV_FAMILY):
            chain = selected_chains[target]
            query = v7_auditor.distance_matrix(
                native_rows[target][f"CA_chain_{chain}"]
            )
            scores = [
                v7_auditor.cyclic_matrix_rmse(reference, query, shift)
                for shift in range(len(query))
            ]
            observed[target] = scores.index(min(scores))
        self.assertEqual(observed, {target: 0 for target in v7_auditor.AV_FAMILY})

    def test_launcher_is_non_destructive_and_disallows_abstention_success(self):
        launcher = (ROOT / "run_serine_qc_serine_only_cyclic_v7.ps1").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("[switch]$Force", launcher)
        self.assertNotIn("Remove-Item -LiteralPath $V6Root", launcher)
        self.assertIn("Target coverage:       17/17; no formal abstention", launcher)
        self.assertIn("Canonical frankenstein_v28.pt changed", launcher)
        self.assertIn('$ExpectedChanged = @("experts.15.bias", "experts.15.weight")', launcher)

    def test_launcher_embedded_python_programs_compile(self):
        launcher = (ROOT / "run_serine_qc_serine_only_cyclic_v7.ps1").read_text(
            encoding="utf-8"
        )
        here_strings = re.findall(r"\$Program = @'\n(.*?)\n'@", launcher, flags=re.DOTALL)
        self.assertEqual(len(here_strings), 1)
        for program in here_strings:
            compile(program, "<V7 PowerShell embedded Python>", "exec")
        probes = re.findall(r"\$Probe = '([^'\r\n]+)'", launcher)
        self.assertEqual(len(probes), 1)
        for program in probes:
            compile(program, "<V7 PowerShell probe>", "exec")

    def test_v7_audit_has_no_formal_abstention_escape_hatch(self):
        auditor = (RETRAIN_DIR / "11_triple_audit_serine_only_v7.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"all_17_targets_have_at_least_one_signature_candidate"', auditor)
        self.assertIn('"formal_target_abstention_is_absent"', auditor)
        self.assertNotIn("FORMAL_ABSTENTION_MINIMUM_TOPUP_DRAWS", auditor)


if __name__ == "__main__":
    unittest.main()
