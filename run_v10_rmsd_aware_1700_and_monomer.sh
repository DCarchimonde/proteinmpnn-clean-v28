#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${V10_PYTHON:-python}"
OUTPUT_ROOT="${V10_OUTPUT_ROOT:-$REPO_ROOT/paper_clean_v28_outputs/rmsd_aware_v10_1700_monomer}"
PLAN="${V10_PLAN:-$REPO_ROOT/paper_clean_v28/serine_qc_retrain/target_plan_v10_rmsd_priority_1700.json}"
PARENT_MODEL="${V10_PARENT_MODEL:-$REPO_ROOT/frankenstein_v28.pt}"
TRAIN_JSONL="${V10_TRAIN_JSONL:-$REPO_ROOT/v9_inputs/train_serine_provenance_corrected.jsonl}"
TEST_JSONL="${V10_TEST_JSONL:-$REPO_ROOT/v9_inputs/test_serine_provenance_corrected.jsonl}"
NATIVE_JSONL="${V10_NATIVE_JSONL:-$REPO_ROOT/17_complexes_native.jsonl}"
BEST_CSV="${V10_BEST_CSV:-$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv}"
HISTORICAL_CSV="${V10_HISTORICAL_CSV:-$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv}"
PRIOR_CSV="${V10_PRIOR_CSV:-$REPO_ROOT/v9_inputs/methylated_new_candidates.csv}"
RMSD_DEVELOPMENT_CSV="${V10_RMSD_DEVELOPMENT_CSV:-$REPO_ROOT/v10_inputs/six_non3av_t05_joint_rmsd_476.csv}"
ORIGINAL_MONOMER_CORRECTED_CSV="${V10_ORIGINAL_MONOMER_CSV:-$REPO_ROOT/v10_inputs/monomer_corrected_1505_original_v28.csv}"
POSITION_CONCENTRATION_POLICY="${V10_POSITION_POLICY:-$REPO_ROOT/v10_inputs/evidence_aware_position_concentration_policy.json}"
PANDAS_SPEC="${V10_PANDAS_SPEC:-pandas==2.2.3}"

TRAINER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/02_retrain_canonical_expert_heads.py"
AUDITOR="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/07_audit_cyclic_representation_equivariance.py"
GENERATOR="$REPO_ROOT/paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py"
TOPUP="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/25_resume_cyclic_stability_v9_quota.py"
BASE_SCORER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/24_score_uniform_cyclic_base_v9.py"
SELECTOR="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/23_select_and_audit_v9_1700.py"
REPLAYER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/26_independent_replay_and_package_v9_1700.py"
RMSD_RANKER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/27_calibrate_and_apply_rmsd_ranker_v10.py"
MONOMER_FINALIZER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/28_finalize_monomer_v10.py"
RMSD_REPLAYER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/29_independent_rmsd_priority_replay_v10.py"
REPORT_BUILDER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/30_build_v10_prestructure_report.py"
MONOMER_EVALUATOR="$REPO_ROOT/paper_clean_v28/01_eval_clean_model.py"

MODEL_DIR="$OUTPUT_ROOT/model"
MODEL="$MODEL_DIR/frankenstein_v28_expert_heads_qc.pt"
MODEL_MANIFEST="$MODEL_DIR/expert_heads_retrain_manifest.json"
AUDIT_DIR="$OUTPUT_ROOT/representation_audit"
AUDIT_JSON="$AUDIT_DIR/cyclic_representation_audit.json"
GENERATION_DIR="$OUTPUT_ROOT/generation"
GENERATION_MANIFEST="$GENERATION_DIR/generation_manifest.json"
BASE_DIR="$OUTPUT_ROOT/exact_cyclic_base"
BASE_MANIFEST="$BASE_DIR/exact_cyclic_base_scoring_manifest.json"
BASE_PASS="$BASE_DIR/candidates_exact_cyclic_base_pass.csv"
RMSD_DIR="$OUTPUT_ROOT/rmsd_priority_ranker"
RMSD_MANIFEST="$RMSD_DIR/rmsd_ranker_v10_manifest.json"
RMSD_MODELS="$RMSD_DIR/rmsd_ranker_models_v10.json"
RMSD_SCORED="$RMSD_DIR/candidates_rmsd_priority_scored.csv"
SELECTION_DIR="$OUTPUT_ROOT/selection_17x100"
SELECTION_MANIFEST="$SELECTION_DIR/v9_1700_release_audit.json"
FINAL_DIR="$OUTPUT_ROOT/final_independent_replay_handoff"
FINAL_MANIFEST="$FINAL_DIR/v9_1700_independent_replay_manifest.json"
V10_FINAL_DIR="$OUTPUT_ROOT/final_v10_handoff"
V10_FINAL_MANIFEST="$V10_FINAL_DIR/v10_1700_final_manifest.json"
MONOMER_EVAL_DIR="$OUTPUT_ROOT/monomer_sequence_eval"
PARENT_MONOMER_EVAL_DIR="$OUTPUT_ROOT/monomer_parent_sequence_eval"
MONOMER_DIR="$OUTPUT_ROOT/monomer_final"
MONOMER_MANIFEST="$MONOMER_DIR/monomer_v10_manifest.json"
REPORT_DIR="$OUTPUT_ROOT/prestructure_report"
REPORT_MANIFEST="$REPORT_DIR/v10_prestructure_report_manifest.json"
TRANSFER_ARCHIVE="$OUTPUT_ROOT/v10_autodl_to_windows_handoff.tar.gz"
LOG="$OUTPUT_ROOT/v10_1700_monomer_pipeline.log"

export OMP_NUM_THREADS="${V10_OMP_THREADS:-16}"
export MKL_NUM_THREADS="${V10_MKL_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${V10_OPENBLAS_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${V10_NUMEXPR_NUM_THREADS:-16}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

manifest_passes() {
  "$PYTHON_BIN" - "$@" <<'PY'
import hashlib
import json
import pathlib
import re
import sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8-sig"))
checks = payload.get("quality_checks", payload.get("upstream_checks", {}))
passed = payload.get("quality_gate") == "PASS"
if isinstance(checks, dict) and checks:
    passed = passed and all(bool(value) for value in checks.values())

def sha256_file(candidate):
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def values(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from values(value)
    elif isinstance(node, list):
        for value in node:
            yield from values(value)

# Every artifact is bound to its own recorded path and digest.  Merely finding
# a digest elsewhere in the JSON is intentionally insufficient: two 1700-row
# views cannot be swapped and still pass this cache check.
records = {}
def collect(node):
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            records[str(pathlib.Path(node["path"]).resolve())] = node["sha256"].lower()
        for key, value in node.items():
            if key.endswith("_path") and isinstance(value, str):
                hash_value = node.get(key[:-5] + "_sha256")
                if isinstance(hash_value, str):
                    records[str(pathlib.Path(value).resolve())] = hash_value.lower()
            collect(value)
    elif isinstance(node, list):
        for value in node:
            collect(value)
collect(payload)
# A PASS cache is valid only while every locally named program, dependency,
# input, checkpoint and artifact still has its recorded bytes.  This prevents
# an older PASS manifest from silently surviving a git pull that changed code.
for resolved, expected in records.items():
    artifact = pathlib.Path(resolved)
    if not artifact.is_file() or sha256_file(artifact) != expected:
        passed = False
for raw in sys.argv[2:]:
    artifact = pathlib.Path(raw)
    resolved = str(artifact.resolve())
    if (
        not artifact.is_file()
        or resolved not in records
        or sha256_file(artifact) != records[resolved]
    ):
        passed = False
raise SystemExit(0 if passed else 1)
PY
}

verify_frozen_inputs() {
  "$PYTHON_BIN" - \
    "$PARENT_MODEL" "bab7b8a010114fc52c749fab1914d9d8ae561ddca45d6d7a0fbec3f9f5ac5b2e" \
    "$TRAIN_JSONL" "98c73a832e3e46820018354ca50a378739a0871c68dc983b9cb0868d4834b2c1" \
    "$TEST_JSONL" "56f877bb998701149954b8c01e86b59ecb8503b01742bfd8200e985b564d236b" \
    "$PRIOR_CSV" "6c7b20e96d8b75fa8c09e5d773326b1c38be7bea84e1bf87f86c27d1894d06f3" \
    "$NATIVE_JSONL" "853d6c2e7075e016989ccd22cf888da20b2b69fb2e1a99036edba8231b3f8816" \
    "$BEST_CSV" "86373590f317a6746826f6caa6de8921acb0937c1e606f2421b224fd1f70ff72" \
    "$HISTORICAL_CSV" "4cf203f8951f7090bd82cba1eb5455682f4049d7a8b8b4bd901ecaf242b39580" \
    "$PLAN" "65a6b1def84f9271a82da740995f4fc67911a607abb31e2b5ff14d19b85118ac" \
    "$RMSD_DEVELOPMENT_CSV" "d754c905e00d03c18ce0610b740c9bd6da09ee0a9e9d5d7ce953dc73d86aad05" \
    "$ORIGINAL_MONOMER_CORRECTED_CSV" "c9c709521b83523c82dd83eb376da0d7f88be3147521d5c80526b6306f92fc62" \
    "$POSITION_CONCENTRATION_POLICY" "28b41461138cd719dc0f8e0210e35071fbbb3c3ec7ad13a0f03bd12baff1744b" <<'PY'
import hashlib
import pathlib
import sys

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

failures = []
for index in range(1, len(sys.argv), 2):
    path = pathlib.Path(sys.argv[index])
    expected = sys.argv[index + 1]
    observed = sha256_file(path) if path.is_file() else "MISSING"
    if observed != expected:
        failures.append(f"{path}: expected {expected}, observed {observed}")
if failures:
    raise SystemExit("ERROR: frozen V10 input SHA-256 mismatch:\n" + "\n".join(failures))
print("Frozen V10 input SHA-256 contract: PASS")
PY
}

ensure_v10_regression_dependencies() {
  if "$PYTHON_BIN" -c 'import pandas' >/dev/null 2>&1; then
    return
  fi

  echo "V10 dependency bootstrap: pandas is missing; installing $PANDAS_SPEC"
  "$PYTHON_BIN" -m pip install \
    --disable-pip-version-check \
    --no-input \
    "$PANDAS_SPEC"

  "$PYTHON_BIN" -c 'import pandas' >/dev/null
}

final_handoff_passes() {
  manifest_passes "$FINAL_MANIFEST" \
    "$REPLAYER" "$SELECTION_MANIFEST" "$MODEL" "$AUDIT_JSON" "$BASE_MANIFEST" \
    "$PLAN" "$NATIVE_JSONL" "$BEST_CSV" \
    "$FINAL_DIR/1700_详细审计.csv" \
    "$FINAL_DIR/1700_给尚哥_极简.csv" \
    "$FINAL_DIR/1700_给尚哥_结构输入.fasta" \
    "$FINAL_DIR/v9_1700_independent_replay.csv" || return 1
  "$PYTHON_BIN" - \
    "$FINAL_DIR/1700_详细审计.csv" \
    "$FINAL_DIR/1700_给尚哥_极简.csv" \
    "$FINAL_DIR/1700_给尚哥_结构输入.fasta" \
    "$FINAL_DIR/v9_1700_independent_replay.csv" <<'PY'
import collections
import csv
import pathlib
import sys

detail, concise, fasta, replay = map(pathlib.Path, sys.argv[1:])
def csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

detail_rows = csv_rows(detail)
concise_rows = csv_rows(concise)
replay_rows = csv_rows(replay)
fasta_records = sum(
    line.startswith(">")
    for line in fasta.read_text(encoding="utf-8-sig").splitlines()
)
target_counts = collections.Counter(
    str(row.get("target_name", "")).upper() for row in detail_rows
)
passed = (
    len(detail_rows) == len(concise_rows) == len(replay_rows) == fasta_records == 1700
    and len(target_counts) == 17
    and set(target_counts.values()) == {100}
    and all(row.get("row_replay_status") == "PASS" for row in replay_rows)
)
raise SystemExit(0 if passed else 1)
PY
}

v10_final_handoff_passes() {
  manifest_passes "$V10_FINAL_MANIFEST" \
    "$RMSD_REPLAYER" "$FINAL_MANIFEST" "$SELECTION_MANIFEST" "$RMSD_MANIFEST" "$RMSD_MODELS" \
    "$V10_FINAL_DIR/1700_详细审计.csv" \
    "$V10_FINAL_DIR/1700_给尚哥_极简.csv" \
    "$V10_FINAL_DIR/1700_给尚哥_结构输入.fasta" \
    "$V10_FINAL_DIR/v10_rmsd_priority_replay.csv" || return 1
  "$PYTHON_BIN" - \
    "$V10_FINAL_DIR/1700_详细审计.csv" \
    "$V10_FINAL_DIR/v10_rmsd_priority_replay.csv" <<'PY'
import collections
import csv
import pathlib
import sys
detail_path, replay_path = map(pathlib.Path, sys.argv[1:])
def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
detail = rows(detail_path)
replay = rows(replay_path)
counts = collections.Counter(str(row.get("target_name", "")).upper() for row in detail)
passed = (
    len(detail) == len(replay) == 1700
    and len(counts) == 17
    and set(counts.values()) == {100}
    and all(any(token.islower() for token in row.get("design_seq", "")) for row in detail)
    and all(row.get("row_rmsd_priority_replay_status") == "PASS" for row in replay)
)
raise SystemExit(0 if passed else 1)
PY
}

monomer_handoff_passes() {
  manifest_passes "$MONOMER_MANIFEST" \
    "$MONOMER_FINALIZER" "$MONOMER_EVALUATOR" "$MODEL" "$PARENT_MODEL" "$AUDIT_JSON" \
    "$MONOMER_EVAL_DIR/eval_manifest.json" \
    "$MONOMER_EVAL_DIR/position_predictions.csv" \
    "$PARENT_MONOMER_EVAL_DIR/eval_manifest.json" \
    "$PARENT_MONOMER_EVAL_DIR/position_predictions.csv" \
    "$AUDIT_DIR/heldout_position_probabilities.csv" \
    "$AUDIT_DIR/native_target_representation_probabilities.csv" \
    "$MONOMER_DIR/monomer_v10_position_comparison_1505.csv" \
    "$MONOMER_DIR/monomer_v10_metrics.csv" \
    "$MONOMER_DIR/monomer_v10_threshold_curves.csv" \
    "$MONOMER_DIR/monomer_v10_by_residue.csv" \
    "$MONOMER_DIR/monomer_v10_by_company_rosetta_panel.csv" \
    "$MONOMER_DIR/monomer_v10_per_sample.csv" \
    "$MONOMER_DIR/monomer_v10_paired_original_v28_comparison.csv" \
    "$MONOMER_DIR/native17_v10_all_negative_control.csv" \
    "$MONOMER_DIR/monomer_v10_design_manifest_151.csv" \
    "$MONOMER_DIR/monomer_v10_structure_input_if_reprediction_needed.fasta" || return 1
  "$PYTHON_BIN" - \
    "$MONOMER_DIR/monomer_v10_position_comparison_1505.csv" \
    "$MONOMER_DIR/monomer_v10_design_manifest_151.csv" <<'PY'
import csv
import pathlib
import sys
def count(path):
    with pathlib.Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))
raise SystemExit(0 if count(sys.argv[1]) == 1505 and count(sys.argv[2]) == 151 else 1)
PY
}

require_empty_stage() {
  local stage_dir="$1"
  local stage_name="$2"
  if [[ -d "$stage_dir" ]] && [[ -n "$(find "$stage_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: $stage_name has partial/unverified files: $stage_dir" >&2
    echo "Use a new V10_OUTPUT_ROOT; this launcher never deletes scientific evidence." >&2
    exit 1
  fi
}

echo "===== V10 CYCLIC-STABLE + RMSD-PRIORITY 17 x 100 + MONOMER PIPELINE ====="
echo "Output root: $OUTPUT_ROOT"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
}
for required in \
  "$PLAN" "$PARENT_MODEL" "$TRAIN_JSONL" "$TEST_JSONL" "$NATIVE_JSONL" \
  "$BEST_CSV" "$HISTORICAL_CSV" "$PRIOR_CSV" "$RMSD_DEVELOPMENT_CSV" \
  "$ORIGINAL_MONOMER_CORRECTED_CSV" "$POSITION_CONCENTRATION_POLICY" "$TRAINER" "$AUDITOR" \
  "$GENERATOR" "$TOPUP" "$BASE_SCORER" "$SELECTOR" "$REPLAYER" \
  "$RMSD_RANKER" "$MONOMER_FINALIZER" "$RMSD_REPLAYER" \
  "$REPORT_BUILDER" "$MONOMER_EVALUATOR"; do
  [[ -s "$required" ]] || {
    echo "ERROR: required input/program is missing or empty: $required" >&2
    exit 1
  }
done
verify_frozen_inputs
ensure_v10_regression_dependencies

"$PYTHON_BIN" - <<'PY'
import json
import sys
import pandas
import torch
if sys.version_info < (3, 10):
    raise SystemExit("ERROR: Python >=3.10 is required")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA PyTorch is required for V10 retraining/replay")
print(json.dumps({
    "python": sys.version.split()[0],
    "pandas": pandas.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
}, ensure_ascii=False))
PY

if [[ "${V10_SKIP_TESTS:-0}" != "1" ]]; then
  echo "[1/10] Full code regression"
  (cd "$REPO_ROOT" && "$PYTHON_BIN" -m unittest discover -s tests)
else
  echo "[1/10] Full code regression skipped only because V10_SKIP_TESTS=1"
fi

if [[ -f "$MODEL" ]] && manifest_passes "$MODEL_MANIFEST" \
  "$TRAINER" "$PARENT_MODEL" "$TRAIN_JSONL" "$TEST_JSONL" "$MODEL"; then
  echo "[2/10] Reusing promoted, manifest-PASS cyclic-stable checkpoint"
else
  require_empty_stage "$MODEL_DIR" "V9 model stage"
  echo "[2/10] Retraining all expert heads from canonical frankenstein_v28.pt"
  "$PYTHON_BIN" "$TRAINER" \
    --model-path "$PARENT_MODEL" \
    --train-jsonl "$TRAIN_JSONL" \
    --test-jsonl "$TEST_JSONL" \
    --out-dir "$MODEL_DIR" \
    --epochs "${V10_EPOCHS:-80}" \
    --batch-size "${V10_TRAIN_BATCH_SIZE:-16}" \
    --deployment-temperature 0.5 \
    --threshold 0.6 \
    --worst-start-bce-weight 1.0 \
    --representation-consistency-weight 0.25 \
    --expert-scope all \
    --expected-parent-sha256 "bab7b8a010114fc52c749fab1914d9d8ae561ddca45d6d7a0fbec3f9f5ac5b2e" \
    --expected-train-sha256 "98c73a832e3e46820018354ca50a378739a0871c68dc983b9cb0868d4834b2c1" \
    --expected-test-sha256 "56f877bb998701149954b8c01e86b59ecb8503b01742bfd8200e985b564d236b" \
    --require-frozen-input-sha256 \
    --cyclic-representation-augmentation
fi

if [[ -f "$AUDIT_JSON" ]] && manifest_passes "$AUDIT_JSON" \
  "$AUDITOR" "$MODEL" "$TEST_JSONL" "$NATIVE_JSONL" "$BEST_CSV" "$PLAN" \
  "$AUDIT_DIR/heldout_position_probabilities.csv" \
  "$AUDIT_DIR/native_target_representation_probabilities.csv" \
  "$AUDIT_DIR/native_target_representation_summary.csv"; then
  echo "[3/10] Reusing hash-pinned PASS representation audit"
else
  require_empty_stage "$AUDIT_DIR" "V9 held-out audit stage"
  echo "[3/10] Auditing 151 monomers and all 17 native targets over the full cyclic grid"
  "$PYTHON_BIN" "$AUDITOR" \
    --model-path "$MODEL" \
    --test-jsonl "$TEST_JSONL" \
    --native-jsonl "$NATIVE_JSONL" \
    --best-csv "$BEST_CSV" \
    --plan "$PLAN" \
    --out-dir "$AUDIT_DIR" \
    --batch-size "${V10_AUDIT_BATCH_SIZE:-8}" \
    --temperature 0.5 \
    --threshold 0.6 \
    --device cuda
fi

if [[ -f "$MONOMER_MANIFEST" ]] && monomer_handoff_passes; then
  echo "[4/10] Reusing hash-pinned PASS V10 monomer sequence audit"
else
  require_empty_stage "$MONOMER_EVAL_DIR" "V10 monomer evaluation stage"
  require_empty_stage "$PARENT_MONOMER_EVAL_DIR" "parent monomer evaluation stage"
  require_empty_stage "$MONOMER_DIR" "V10 monomer finalization stage"
  echo "[4/10] Recomputing deterministic parent and V10 151-monomer metrics"
  "$PYTHON_BIN" "$MONOMER_EVALUATOR" \
    --model_path "$PARENT_MODEL" \
    --data_jsonl "$TEST_JSONL" \
    --mode monomer \
    --eval_chains masked \
    --batch_size "${V10_MONOMER_BATCH_SIZE:-16}" \
    --seed 0 \
    --thresholds 0.6 \
    --out_dir "$PARENT_MONOMER_EVAL_DIR"
  "$PYTHON_BIN" "$MONOMER_EVALUATOR" \
    --model_path "$MODEL" \
    --data_jsonl "$TEST_JSONL" \
    --mode monomer \
    --eval_chains masked \
    --batch_size "${V10_MONOMER_BATCH_SIZE:-16}" \
    --seed 0 \
    --thresholds 0.6 \
    --out_dir "$MONOMER_EVAL_DIR"
  "$PYTHON_BIN" "$MONOMER_FINALIZER" \
    --v10-position-csv "$MONOMER_EVAL_DIR/position_predictions.csv" \
    --v10-eval-manifest "$MONOMER_EVAL_DIR/eval_manifest.json" \
    --parent-position-csv "$PARENT_MONOMER_EVAL_DIR/position_predictions.csv" \
    --parent-eval-manifest "$PARENT_MONOMER_EVAL_DIR/eval_manifest.json" \
    --cyclic-position-csv "$AUDIT_DIR/heldout_position_probabilities.csv" \
    --native-position-csv "$AUDIT_DIR/native_target_representation_probabilities.csv" \
    --cyclic-audit-manifest "$AUDIT_JSON" \
    --original-v28-corrected-csv "$ORIGINAL_MONOMER_CORRECTED_CSV" \
    --v10-model "$MODEL" \
    --parent-model "$PARENT_MODEL" \
    --out-dir "$MONOMER_DIR"
fi
monomer_handoff_passes || {
  echo "ERROR: V10 monomer audit did not pass" >&2
  exit 1
}

if [[ ! -f "$GENERATION_MANIFEST" ]]; then
  require_empty_stage "$GENERATION_DIR" "V9 generation stage"
  echo "[5/10] Generating the initial 42,500 T=0.5 draws"
  set +e
  "$PYTHON_BIN" "$GENERATOR" \
    --plan "$PLAN" \
    --model_path "$MODEL" \
    --native_jsonl "$NATIVE_JSONL" \
    --best_csv "$BEST_CSV" \
    --old_designs_csv "$HISTORICAL_CSV" \
    --prior_designs_csv "$PRIOR_CSV" \
    --position-concentration-policy "$POSITION_CONCENTRATION_POLICY" \
    --out_dir "$GENERATION_DIR" \
    --batch_size "${V10_GENERATION_BATCH_SIZE:-16}" \
    --device cuda \
    --cyclic-representation-ensemble \
    --representation-audit-json "$AUDIT_JSON" \
    --defer-permeability-until-structure
  generation_exit=$?
  set -e
  echo "Initial generation exit code: $generation_exit (quota-only failure may be resumed)"
fi

set +e
"$PYTHON_BIN" - "$GENERATION_MANIFEST" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    print("generation manifest missing", file=sys.stderr)
    raise SystemExit(20)
payload = json.loads(path.read_text(encoding="utf-8-sig"))
checks = payload.get("quality_checks", {})
failed = sorted(name for name, passed in checks.items() if not bool(passed))
if payload.get("quality_gate") == "PASS" and not failed:
    raise SystemExit(0)
if failed == ["every_target_meets_pre_structure_candidate_quota"]:
    print("quota-only generation shortfall: adaptive top-up authorized")
    raise SystemExit(10)
print("non-quota generation failures: " + ", ".join(failed), file=sys.stderr)
raise SystemExit(20)
PY
generation_status=$?
set -e
if [[ "$generation_status" == "10" ]]; then
  echo "[5/10] Stable-candidate quota shortfall: resuming only deficient targets"
  "$PYTHON_BIN" "$TOPUP" \
    --plan "$PLAN" \
    --model-path "$MODEL" \
    --source-run-dir "$GENERATION_DIR" \
    --out-dir "$GENERATION_DIR" \
    --representation-audit-json "$AUDIT_JSON" \
    --native-jsonl "$NATIVE_JSONL" \
    --old-designs-csv "$HISTORICAL_CSV" \
    --prior-designs-csv "$PRIOR_CSV" \
    --position-concentration-policy "$POSITION_CONCENTRATION_POLICY" \
    --batch-size "${V10_GENERATION_BATCH_SIZE:-16}" \
    --draws-per-reserve-seed "${V10_TOPUP_DRAWS_PER_SEED:-1000}" \
    --max-topup-draws-per-target "${V10_MAX_TOPUP_DRAWS_PER_TARGET:-12000}" \
    --quota-margin "${V10_TOPUP_MARGIN:-10}" \
    --device cuda
elif [[ "$generation_status" != "0" ]]; then
  echo "ERROR: generation failed a scientific gate; top-up is forbidden" >&2
  exit "$generation_status"
fi
manifest_passes "$GENERATION_MANIFEST" \
  "$GENERATOR" "$MODEL" "$AUDIT_JSON" "$POSITION_CONCENTRATION_POLICY" \
  "$GENERATION_DIR/all_candidates.csv" "$GENERATION_DIR/target_manifest.csv" \
  "$GENERATION_DIR/methylated_new_candidates.csv" \
  "$GENERATION_DIR/unique_candidates.csv" \
  "$GENERATION_DIR/generation_summary_by_target.csv" || {
  echo "ERROR: V9 generation/top-up manifest is not PASS" >&2
  exit 1
}

if [[ -f "$BASE_MANIFEST" ]] && manifest_passes "$BASE_MANIFEST" \
  "$BASE_SCORER" "$MODEL" "$GENERATION_MANIFEST" "$AUDIT_JSON" "$PLAN" \
  "$BASE_DIR/candidates_exact_cyclic_base_scored.csv" \
  "$BASE_PASS" \
  "$BASE_DIR/cyclic_base_floor_by_target.csv"; then
  echo "[6/10] Reusing exact cyclic-base scores"
else
  echo "[6/10] Exact receptor-visible L x L cyclic-base scoring (per-target resumable)"
  "$PYTHON_BIN" "$BASE_SCORER" \
    --candidate-csv "$GENERATION_DIR/methylated_new_candidates.csv" \
    --baseline-csv "$GENERATION_DIR/unique_candidates.csv" \
    --model "$MODEL" \
    --generation-manifest "$GENERATION_MANIFEST" \
    --representation-audit "$AUDIT_JSON" \
    --native-jsonl "$NATIVE_JSONL" \
    --best-csv "$BEST_CSV" \
    --plan "$PLAN" \
    --out-dir "$BASE_DIR" \
    --batch-size "${V10_BASE_BATCH_SIZE:-32}" \
    --device cuda
fi

if [[ -f "$RMSD_MANIFEST" ]] && manifest_passes "$RMSD_MANIFEST" \
  "$RMSD_RANKER" "$REPO_ROOT/paper_clean_v28/serine_qc_retrain/rmsd_ranker_v10.py" \
  "$RMSD_DEVELOPMENT_CSV" "$BASE_PASS" "$RMSD_MODELS" "$RMSD_SCORED" \
  "$RMSD_DIR/rmsd_ranker_oof_predictions_476.csv"; then
  echo "[7/10] Reusing frozen PASS target-held-out RMSD-priority ranker"
else
  require_empty_stage "$RMSD_DIR" "V10 RMSD-priority ranker stage"
  echo "[7/10] Calibrating six-target leave-one-target-out RMSD priority and scoring pool"
  "$PYTHON_BIN" "$RMSD_RANKER" \
    --development-csv "$RMSD_DEVELOPMENT_CSV" \
    --candidate-csv "$BASE_PASS" \
    --out-dir "$RMSD_DIR"
fi

if [[ -f "$SELECTION_MANIFEST" ]] && manifest_passes "$SELECTION_MANIFEST" \
  "$SELECTOR" "$MODEL" "$GENERATION_MANIFEST" "$AUDIT_JSON" "$BASE_MANIFEST" "$PLAN" \
  "$RMSD_MANIFEST" "$RMSD_SCORED" \
  "$SELECTION_DIR/1700_详细审计.csv" \
  "$SELECTION_DIR/1700_给尚哥_极简.csv" \
  "$SELECTION_DIR/1700_给尚哥_结构输入.fasta" \
  "$SELECTION_DIR/selection_summary_by_target.csv" \
  "$SELECTION_DIR/candidate_validation_problems.csv"; then
  echo "[8/10] Reusing exact RMSD-priority 17 x 100 PASS selection"
else
  require_empty_stage "$SELECTION_DIR" "V10 17 x 100 selection stage"
  echo "[8/10] Independent RMSD-priority exact-1700 selection and concentration audit"
  "$PYTHON_BIN" "$SELECTOR" \
    --candidates "$BASE_PASS" \
    --generation-manifest "$GENERATION_MANIFEST" \
    --heldout-audit "$AUDIT_JSON" \
    --cyclic-base-manifest "$BASE_MANIFEST" \
    --plan "$PLAN" \
    --model "$MODEL" \
    --exclusion-csv "$HISTORICAL_CSV" \
    --exclusion-csv "$PRIOR_CSV" \
    --rmsd-priority-csv "$RMSD_SCORED" \
    --rmsd-priority-manifest "$RMSD_MANIFEST" \
    --out-dir "$SELECTION_DIR"
fi

if [[ -f "$FINAL_MANIFEST" ]] && final_handoff_passes; then
  echo "[9/10] Reusing final batch-size-1 independent methyl/base replay"
else
  require_empty_stage "$FINAL_DIR" "V10 lower-layer independent replay stage"
  echo "[9/10] Batch-size-1 methyl/base replay of all 1,700"
  "$PYTHON_BIN" "$REPLAYER" \
    --selector-manifest "$SELECTION_MANIFEST" \
    --detailed-csv "$SELECTION_DIR/1700_详细审计.csv" \
    --selector-concise "$SELECTION_DIR/1700_给尚哥_极简.csv" \
    --selector-fasta "$SELECTION_DIR/1700_给尚哥_结构输入.fasta" \
    --model "$MODEL" \
    --heldout-audit "$AUDIT_JSON" \
    --scorer-manifest "$BASE_MANIFEST" \
    --native-jsonl "$NATIVE_JSONL" \
    --best-csv "$BEST_CSV" \
    --plan "$PLAN" \
    --out-dir "$FINAL_DIR" \
    --device cuda
fi

final_handoff_passes || {
  echo "ERROR: final independent replay did not pass" >&2
  exit 1
}

if [[ -f "$V10_FINAL_MANIFEST" ]] && v10_final_handoff_passes; then
  echo "[10/10] Reusing independent RMSD-priority replay and V10 final package"
else
  require_empty_stage "$V10_FINAL_DIR" "V10 final RMSD-priority replay stage"
  echo "[10/10] Independently replaying RMSD priorities and packaging final 17 x 100"
  "$PYTHON_BIN" "$RMSD_REPLAYER" \
    --v9-final-dir "$FINAL_DIR" \
    --selector-manifest "$SELECTION_MANIFEST" \
    --ranker-manifest "$RMSD_MANIFEST" \
    --ranker-models "$RMSD_MODELS" \
    --out-dir "$V10_FINAL_DIR"
fi
v10_final_handoff_passes || {
  echo "ERROR: V10 final handoff did not pass" >&2
  exit 1
}
if [[ -f "$REPORT_MANIFEST" ]] && manifest_passes "$REPORT_MANIFEST" \
  "$REPORT_BUILDER" "$GENERATION_MANIFEST" "$GENERATION_DIR/generation_summary_by_target.csv" \
  "$SELECTION_MANIFEST" "$SELECTION_DIR/selection_summary_by_target.csv" \
  "$RMSD_MANIFEST" "$V10_FINAL_MANIFEST" "$MONOMER_MANIFEST" \
  "$REPORT_DIR/v10_prestructure_funnel_by_target.csv" \
  "$REPORT_DIR/v10_prestructure_audit_cn.md"; then
  echo "[10/10] Reusing hash-pinned Chinese pre-structure audit report"
else
  require_empty_stage "$REPORT_DIR" "V10 pre-structure report stage"
  "$PYTHON_BIN" "$REPORT_BUILDER" \
    --generation-manifest "$GENERATION_MANIFEST" \
    --generation-summary "$GENERATION_DIR/generation_summary_by_target.csv" \
    --selector-manifest "$SELECTION_MANIFEST" \
    --selector-summary "$SELECTION_DIR/selection_summary_by_target.csv" \
    --ranker-manifest "$RMSD_MANIFEST" \
    --v10-final-manifest "$V10_FINAL_MANIFEST" \
    --monomer-manifest "$MONOMER_MANIFEST" \
    --out-dir "$REPORT_DIR"
fi

archive_tmp="$TRANSFER_ARCHIVE.tmp"
tar -czf "$archive_tmp" -C "$OUTPUT_ROOT" \
  model \
  representation_audit \
  generation/generation_manifest.json \
  generation/generation_summary_by_target.csv \
  exact_cyclic_base/exact_cyclic_base_scoring_manifest.json \
  rmsd_priority_ranker \
  selection_17x100/selection_summary_by_target.csv \
  selection_17x100/v9_1700_release_audit.json \
  final_independent_replay_handoff/v9_1700_independent_replay_manifest.json \
  final_v10_handoff \
  monomer_sequence_eval \
  monomer_parent_sequence_eval \
  monomer_final \
  prestructure_report
mv -f "$archive_tmp" "$TRANSFER_ARCHIVE"
sha256sum "$TRANSFER_ARCHIVE" > "$TRANSFER_ARCHIVE.sha256"
echo "===== V10 ALL PRE-STRUCTURE GATES PASSED ====="
echo "IMPORTANT: RMSD improvement is predicted, not proven until returned structures are audited."
echo "Detailed: $V10_FINAL_DIR/1700_详细审计.csv"
echo "Shang-ge concise: $V10_FINAL_DIR/1700_给尚哥_极简.csv"
echo "Shang-ge FASTA: $V10_FINAL_DIR/1700_给尚哥_结构输入.fasta"
echo "V10 final manifest: $V10_FINAL_MANIFEST"
echo "Monomer manifest: $MONOMER_MANIFEST"
echo "Chinese audit report: $REPORT_DIR/v10_prestructure_audit_cn.md"
echo "AutoDL -> Windows archive: $TRANSFER_ARCHIVE"
echo "Archive SHA-256: $TRANSFER_ARCHIVE.sha256"
