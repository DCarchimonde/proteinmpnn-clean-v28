#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${V9_PYTHON:-python}"
OUTPUT_ROOT="${V9_OUTPUT_ROOT:-$REPO_ROOT/paper_clean_v28_outputs/cyclic_stability_v9_1700}"
PLAN="${V9_PLAN:-$REPO_ROOT/paper_clean_v28/serine_qc_retrain/target_plan_cyclic_stability_v9_1700.json}"
PARENT_MODEL="${V9_PARENT_MODEL:-$REPO_ROOT/frankenstein_v28.pt}"
TRAIN_JSONL="${V9_TRAIN_JSONL:-$REPO_ROOT/v9_inputs/train_serine_provenance_corrected.jsonl}"
TEST_JSONL="${V9_TEST_JSONL:-$REPO_ROOT/v9_inputs/test_serine_provenance_corrected.jsonl}"
NATIVE_JSONL="${V9_NATIVE_JSONL:-$REPO_ROOT/17_complexes_native.jsonl}"
BEST_CSV="${V9_BEST_CSV:-$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv}"
HISTORICAL_CSV="${V9_HISTORICAL_CSV:-$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv}"
PRIOR_CSV="${V9_PRIOR_CSV:-$REPO_ROOT/v9_inputs/methylated_new_candidates.csv}"

TRAINER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/02_retrain_canonical_expert_heads.py"
AUDITOR="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/07_audit_cyclic_representation_equivariance.py"
GENERATOR="$REPO_ROOT/paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py"
TOPUP="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/25_resume_cyclic_stability_v9_quota.py"
BASE_SCORER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/24_score_uniform_cyclic_base_v9.py"
SELECTOR="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/23_select_and_audit_v9_1700.py"
REPLAYER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/26_independent_replay_and_package_v9_1700.py"

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
SELECTION_DIR="$OUTPUT_ROOT/selection_17x100"
SELECTION_MANIFEST="$SELECTION_DIR/v9_1700_release_audit.json"
FINAL_DIR="$OUTPUT_ROOT/final_independent_replay_handoff"
FINAL_MANIFEST="$FINAL_DIR/v9_1700_independent_replay_manifest.json"
LOG="$OUTPUT_ROOT/v9_1700_pipeline.log"

export OMP_NUM_THREADS="${V9_OMP_THREADS:-16}"
export MKL_NUM_THREADS="${V9_MKL_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${V9_OPENBLAS_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${V9_NUMEXPR_THREADS:-16}"
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

# Every explicitly required artifact must still exist and its current digest
# must be recorded in the PASS manifest.  A stale JSON boolean is never enough.
recorded_hashes = {
    str(value).lower()
    for key, value in values(payload)
    if "sha256" in str(key).lower()
    and isinstance(value, str)
    and re.fullmatch(r"[0-9a-fA-F]{64}", value)
}
for raw in sys.argv[2:]:
    artifact = pathlib.Path(raw)
    if not artifact.is_file() or sha256_file(artifact) not in recorded_hashes:
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
    "$PLAN" "41f5d7ead7c922016c73042bc39cac47569ab4264b0b7b8c1416da7af8a68dff" <<'PY'
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
    raise SystemExit("ERROR: frozen V9 input SHA-256 mismatch:\n" + "\n".join(failures))
print("Frozen V9 input SHA-256 contract: PASS")
PY
}

final_handoff_passes() {
  manifest_passes "$FINAL_MANIFEST" \
    "$SELECTION_MANIFEST" "$MODEL" "$AUDIT_JSON" "$BASE_MANIFEST" \
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

require_empty_stage() {
  local stage_dir="$1"
  local stage_name="$2"
  if [[ -d "$stage_dir" ]] && [[ -n "$(find "$stage_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: $stage_name has partial/unverified files: $stage_dir" >&2
    echo "Use a new V9_OUTPUT_ROOT; this launcher never deletes scientific evidence." >&2
    exit 1
  fi
}

echo "===== V9 CYCLIC STABILITY 17 x 100 PIPELINE ====="
echo "Output root: $OUTPUT_ROOT"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
}
for required in \
  "$PLAN" "$PARENT_MODEL" "$TRAIN_JSONL" "$TEST_JSONL" "$NATIVE_JSONL" \
  "$BEST_CSV" "$HISTORICAL_CSV" "$PRIOR_CSV" "$TRAINER" "$AUDITOR" \
  "$GENERATOR" "$TOPUP" "$BASE_SCORER" "$SELECTOR" "$REPLAYER"; do
  [[ -s "$required" ]] || {
    echo "ERROR: required input/program is missing or empty: $required" >&2
    exit 1
  }
done
verify_frozen_inputs

"$PYTHON_BIN" - <<'PY'
import json
import sys
import torch
if sys.version_info < (3, 10):
    raise SystemExit("ERROR: Python >=3.10 is required")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA PyTorch is required for V9 retraining/replay")
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
}, ensure_ascii=False))
PY

if [[ "${V9_SKIP_TESTS:-0}" != "1" ]]; then
  echo "[1/7] Full code regression"
  (cd "$REPO_ROOT" && "$PYTHON_BIN" -m unittest discover -s tests)
else
  echo "[1/7] Full code regression skipped only because V9_SKIP_TESTS=1"
fi

if [[ -f "$MODEL" ]] && manifest_passes "$MODEL_MANIFEST" "$MODEL"; then
  echo "[2/7] Reusing promoted, manifest-PASS V9 checkpoint"
else
  require_empty_stage "$MODEL_DIR" "V9 model stage"
  echo "[2/7] Retraining all expert heads from canonical frankenstein_v28.pt"
  "$PYTHON_BIN" "$TRAINER" \
    --model-path "$PARENT_MODEL" \
    --train-jsonl "$TRAIN_JSONL" \
    --test-jsonl "$TEST_JSONL" \
    --out-dir "$MODEL_DIR" \
    --epochs "${V9_EPOCHS:-80}" \
    --batch-size "${V9_TRAIN_BATCH_SIZE:-16}" \
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
  "$MODEL" "$TEST_JSONL" "$NATIVE_JSONL" "$PLAN" \
  "$AUDIT_DIR/heldout_position_probabilities.csv" \
  "$AUDIT_DIR/native_target_representation_probabilities.csv" \
  "$AUDIT_DIR/native_target_representation_summary.csv"; then
  echo "[3/7] Reusing hash-pinned PASS internal-development representation audit"
else
  require_empty_stage "$AUDIT_DIR" "V9 held-out audit stage"
  echo "[3/7] Auditing 151 held-out records and all 17 native targets"
  "$PYTHON_BIN" "$AUDITOR" \
    --model-path "$MODEL" \
    --test-jsonl "$TEST_JSONL" \
    --native-jsonl "$NATIVE_JSONL" \
    --best-csv "$BEST_CSV" \
    --plan "$PLAN" \
    --out-dir "$AUDIT_DIR" \
    --batch-size "${V9_AUDIT_BATCH_SIZE:-8}" \
    --temperature 0.5 \
    --threshold 0.6 \
    --device cuda
fi

if [[ ! -f "$GENERATION_MANIFEST" ]]; then
  require_empty_stage "$GENERATION_DIR" "V9 generation stage"
  echo "[4/7] Generating the initial 42,500 T=0.5 draws"
  set +e
  "$PYTHON_BIN" "$GENERATOR" \
    --plan "$PLAN" \
    --model_path "$MODEL" \
    --native_jsonl "$NATIVE_JSONL" \
    --best_csv "$BEST_CSV" \
    --old_designs_csv "$HISTORICAL_CSV" \
    --prior_designs_csv "$PRIOR_CSV" \
    --out_dir "$GENERATION_DIR" \
    --batch_size "${V9_GENERATION_BATCH_SIZE:-16}" \
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
  echo "[4/7] Stable-candidate quota shortfall: resuming only deficient targets"
  "$PYTHON_BIN" "$TOPUP" \
    --plan "$PLAN" \
    --model-path "$MODEL" \
    --source-run-dir "$GENERATION_DIR" \
    --out-dir "$GENERATION_DIR" \
    --representation-audit-json "$AUDIT_JSON" \
    --native-jsonl "$NATIVE_JSONL" \
    --old-designs-csv "$HISTORICAL_CSV" \
    --prior-designs-csv "$PRIOR_CSV" \
    --batch-size "${V9_GENERATION_BATCH_SIZE:-16}" \
    --draws-per-reserve-seed "${V9_TOPUP_DRAWS_PER_SEED:-1000}" \
    --max-topup-draws-per-target "${V9_MAX_TOPUP_DRAWS_PER_TARGET:-12000}" \
    --quota-margin "${V9_TOPUP_MARGIN:-10}" \
    --device cuda
elif [[ "$generation_status" != "0" ]]; then
  echo "ERROR: generation failed a scientific gate; top-up is forbidden" >&2
  exit "$generation_status"
fi
manifest_passes "$GENERATION_MANIFEST" \
  "$MODEL" "$AUDIT_JSON" \
  "$GENERATION_DIR/methylated_new_candidates.csv" \
  "$GENERATION_DIR/unique_candidates.csv" || {
  echo "ERROR: V9 generation/top-up manifest is not PASS" >&2
  exit 1
}

if [[ -f "$BASE_MANIFEST" ]] && manifest_passes "$BASE_MANIFEST" \
  "$MODEL" "$GENERATION_MANIFEST" "$AUDIT_JSON" "$PLAN" \
  "$BASE_DIR/candidates_exact_cyclic_base_scored.csv" \
  "$BASE_PASS" \
  "$BASE_DIR/cyclic_base_floor_by_target.csv"; then
  echo "[5/7] Reusing exact cyclic-base scores"
else
  echo "[5/7] Exact receptor-visible L x L cyclic-base scoring (per-target resumable)"
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
    --batch-size "${V9_BASE_BATCH_SIZE:-32}" \
    --device cuda
fi

if [[ -f "$SELECTION_MANIFEST" ]] && manifest_passes "$SELECTION_MANIFEST" \
  "$MODEL" "$GENERATION_MANIFEST" "$AUDIT_JSON" "$BASE_MANIFEST" "$PLAN" \
  "$SELECTION_DIR/1700_详细审计.csv" \
  "$SELECTION_DIR/1700_给尚哥_极简.csv" \
  "$SELECTION_DIR/1700_给尚哥_结构输入.fasta"; then
  echo "[6/7] Reusing exact 17 x 100 PASS selection"
else
  require_empty_stage "$SELECTION_DIR" "V9 17 x 100 selection stage"
  echo "[6/7] Independent exact-1700 selection and concentration audit"
  "$PYTHON_BIN" "$SELECTOR" \
    --candidates "$BASE_PASS" \
    --generation-manifest "$GENERATION_MANIFEST" \
    --heldout-audit "$AUDIT_JSON" \
    --cyclic-base-manifest "$BASE_MANIFEST" \
    --plan "$PLAN" \
    --model "$MODEL" \
    --exclusion-csv "$HISTORICAL_CSV" \
    --exclusion-csv "$PRIOR_CSV" \
    --out-dir "$SELECTION_DIR"
fi

if [[ -f "$FINAL_MANIFEST" ]] && final_handoff_passes; then
  echo "[7/7] Reusing final batch-size-1 independent replay handoff"
else
  require_empty_stage "$FINAL_DIR" "V9 final independent replay stage"
  echo "[7/7] Batch-size-1 methyl/base replay of all 1,700 and final packaging"
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
echo "===== V9 ALL GATES PASSED: FINAL 17 x 100 HANDOFF READY ====="
echo "Detailed: $FINAL_DIR/1700_详细审计.csv"
echo "Shang-ge concise: $FINAL_DIR/1700_给尚哥_极简.csv"
echo "Shang-ge FASTA: $FINAL_DIR/1700_给尚哥_结构输入.fasta"
echo "Replay manifest: $FINAL_MANIFEST"
