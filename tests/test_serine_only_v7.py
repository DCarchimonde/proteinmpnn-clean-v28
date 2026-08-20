from __future__ import annotations

import ast
import csv
import importlib.util
import json
import re
import subprocess
import sys
import textwrap
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


TORCH_ACTIVE_BATCH_PROGRAM = textwrap.dedent(
    r"""
    import importlib.util
    import sys
    from pathlib import Path

    import torch

    root = Path(sys.argv[1]).resolve()
    trainer_path = (
        root
        / "paper_clean_v28"
        / "serine_qc_retrain"
        / "02_retrain_canonical_expert_heads.py"
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("serine_only_v7_trainer", trainer_path)
    assert spec and spec.loader
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)

    ser = trainer.SERINE_EXPERT_INDEX
    labels = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    assert not trainer.batch_has_active_expert_positions(labels, valid, (ser,))

    labels[1, 1] = ser
    assert trainer.batch_has_active_expert_positions(labels, valid, (ser,))

    labels[1, 1] = trainer.EXTENDED_AA_ALPHABET.index("s")
    assert trainer.batch_has_active_expert_positions(labels, valid, (ser,))

    valid[1, 1] = False
    assert not trainer.batch_has_active_expert_positions(labels, valid, (ser,))
    assert trainer.batch_has_active_expert_positions(labels, valid, tuple(range(20)))
    """
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

    def test_scope_limited_empty_batches_are_guarded_before_forward_and_optimizer(self):
        trainer = (RETRAIN_DIR / "02_retrain_canonical_expert_heads.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(trainer)
        functions = {
            node.name: ast.get_source_segment(trainer, node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        training = functions["train_all_expert_heads"]
        validation = functions["validation_balanced_bce"]
        helper_call = "if not batch_has_active_expert_positions("
        self.assertLess(training.index(helper_call), training.index("optimizer.zero_grad"))
        self.assertLess(training.index(helper_call), training.index("model("))
        self.assertIn("skipped_no_active_position_batches += 1", training)
        self.assertIn("active_position_coverage_verified", training)
        self.assertLess(
            validation.index(helper_call),
            validation.index(
                "cyclic_representation_known_sequence_methyl_probabilities("
            ),
        )

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

    def test_mean_above_threshold_cannot_override_worst_start_failure(self):
        representation = {
            "mean": [FakeTensor([0.9, 0.1])],
            "decoder_order_std_mean": [FakeTensor([0.0, 0.0])],
            "representation_std": [FakeTensor([0.15, 0.0])],
            "representation_min": [FakeTensor([0.59, 0.1])],
            "representation_max": [FakeTensor([0.95, 0.1])],
            "representation_span": [FakeTensor([0.36, 0.0])],
            "representation_count": [FakeTensor([2.0, 2.0])],
        }
        payload = v7_reannotator.annotation_payload(
            "SA",
            representation,
            0,
            0.6,
            NATURAL_AA_ALPHABET,
            EXTENDED_AA_ALPHABET,
            NAT_TO_METHYL_ABS,
        )
        self.assertEqual(payload["design_seq"], "SA")
        self.assertEqual(payload["design_methyl_count"], 0)
        self.assertEqual(payload["representation_threshold_disagreement_count"], 1)
        self.assertEqual(payload["stable_cyclic_release_gate"], 0)

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


class SerineOnlyActiveBatchTorchTests(unittest.TestCase):
    """Exercise the V7 active-position guard in a clean Torch process."""

    @classmethod
    def setUpClass(cls):
        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if probe.returncode != 0 and "No module named 'torch'" in probe.stderr:
            raise unittest.SkipTest("torch is not installed in this test environment")
        if probe.returncode != 0:
            raise RuntimeError(
                "PyTorch failed in a clean test process:\n"
                + probe.stdout
                + probe.stderr
            )

    def test_serine_only_guard_distinguishes_empty_active_and_masked_batches(self):
        completed = subprocess.run(
            [sys.executable, "-c", TORCH_ACTIVE_BATCH_PROGRAM, str(ROOT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                "isolated V7 active-position guard failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
