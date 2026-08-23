#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${V13_PYTHON:-python}"
V11_ROOT="${V11_OUTPUT_ROOT:-$REPO_ROOT/paper_clean_v28_outputs/cyclic_native_v11_1700_monomer}"
OUTPUT_ROOT="${V13_OUTPUT_ROOT:-$REPO_ROOT/paper_clean_v28_outputs/methyl_yield_v13_1700}"
TRAINER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/36_retrain_short_length_balanced_v13.py"
AUDITOR="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/07_audit_cyclic_representation_equivariance.py"
GENERATOR="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/37_generate_fixed_budget_methyl_yield_v13.py"
PLAN="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/target_plan_v13_fixed_budget_methyl_yield_1700.json"
V11_MODEL="$V11_ROOT/model/frankenstein_v28_expert_heads_qc.pt"
V11_MANIFEST="$V11_ROOT/model/expert_heads_retrain_manifest.json"
TRAIN_JSONL="${V13_TRAIN_JSONL:-$REPO_ROOT/v9_inputs/train_serine_provenance_corrected.jsonl}"
TEST_JSONL="${V13_TEST_JSONL:-$REPO_ROOT/v9_inputs/test_serine_provenance_corrected.jsonl}"
NATIVE_JSONL="${V13_NATIVE_JSONL:-$REPO_ROOT/17_complexes_native.jsonl}"
BEST_CSV="${V13_BEST_CSV:-$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv}"
MODEL_DIR="$OUTPUT_ROOT/model"
MODEL="$MODEL_DIR/frankenstein_v28_short_length_balanced_v13.pt"
MODEL_MANIFEST="$MODEL_DIR/v13_short_length_retrain_manifest.json"
AUDIT_DIR="$OUTPUT_ROOT/representation_audit"
AUDIT_JSON="$AUDIT_DIR/cyclic_representation_audit.json"
GENERATION_DIR="$OUTPUT_ROOT/fixed_budget_generation"
GENERATION_MANIFEST="$GENERATION_DIR/v13_fixed_budget_manifest.json"
LOG="$OUTPUT_ROOT/v13_fixed_budget_pipeline.log"

export OMP_NUM_THREADS="${V13_OMP_THREADS:-16}"
export MKL_NUM_THREADS="${V13_MKL_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${V13_OPENBLAS_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${V13_NUMEXPR_NUM_THREADS:-16}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

manifest_passes() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8-sig"))
checks = payload.get("quality_checks", {})
passed = payload.get("quality_gate") == "PASS" and bool(checks) and all(bool(v) for v in checks.values())
def sha256_file(candidate):
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def visit(node):
    nonlocal_passed = True
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            artifact = pathlib.Path(node["path"])
            nonlocal_passed = artifact.is_file() and sha256_file(artifact) == node["sha256"]
        for value in node.values():
            nonlocal_passed = visit(value) and nonlocal_passed
    elif isinstance(node, list):
        for value in node:
            nonlocal_passed = visit(value) and nonlocal_passed
    return nonlocal_passed
passed = passed and visit(payload)
raise SystemExit(0 if passed else 1)
PY
}

echo "===== V13 FIXED-BUDGET METHYL-YIELD PIPELINE ====="
echo "Repository: $REPO_ROOT"
echo "Commit: $(git rev-parse HEAD)"
echo "Scientific contract: one global policy, 250 final draws/target, >=50% strict hits, >=100 cyclic-unique hits"
echo "Forbidden: historical replay, local search, deficit top-up, pre-structure base/RMSD ranking"

echo "[1/5] Code regression"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py'

echo "[2/5] Verify promoted V11 parent"
"$PYTHON_BIN" - "$V11_MODEL" "$V11_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import sys
model, manifest = map(pathlib.Path, sys.argv[1:])
if not model.is_file() or not manifest.is_file():
    raise SystemExit("V11 promoted checkpoint/manifest is missing; do not retrain from an unknown parent")
payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
digest = hashlib.sha256(model.read_bytes()).hexdigest()
if not (
    payload.get("quality_gate") == "PASS"
    and payload.get("protocol") == "canonical_clean_v28_all_expert_heads_cyclic_native_relative_positions_v11"
    and payload.get("checkpoint_artifact_sha256") == digest
):
    raise SystemExit("V11 parent checkpoint is not the promoted hash-bound model")
print("Promoted V11 parent: PASS", digest)
PY

echo "[3/5] Train short-length-balanced V13 expert heads"
if manifest_passes "$MODEL_MANIFEST"; then
  echo "V13 model cache: PASS"
else
  if [[ -d "$MODEL_DIR" ]]; then
    mv "$MODEL_DIR" "${MODEL_DIR}.superseded.$(date -u '+%Y%m%dT%H%M%SZ')"
  fi
  "$PYTHON_BIN" "$TRAINER" \
    --model-path "$V11_MODEL" \
    --train-jsonl "$TRAIN_JSONL" \
    --test-jsonl "$TEST_JSONL" \
    --out-dir "$MODEL_DIR"
fi

echo "[4/5] Independent cyclic representation audit"
if manifest_passes "$AUDIT_JSON"; then
  echo "V13 representation audit cache: PASS"
else
  if [[ -d "$AUDIT_DIR" ]]; then
    mv "$AUDIT_DIR" "${AUDIT_DIR}.superseded.$(date -u '+%Y%m%dT%H%M%SZ')"
  fi
  "$PYTHON_BIN" "$AUDITOR" \
    --model-path "$MODEL" \
    --required-expert-protocol \
      canonical_clean_v28_all_expert_heads_cyclic_native_short_length_balanced_v13 \
    --test-jsonl "$TEST_JSONL" \
    --native-jsonl "$NATIVE_JSONL" \
    --best-csv "$BEST_CSV" \
    --plan "$PLAN" \
    --out-dir "$AUDIT_DIR"
fi

echo "[5/5] Fixed-budget yield experiment and 17 x 100 batch-one replay"
if manifest_passes "$GENERATION_MANIFEST"; then
  echo "V13 generation/handoff cache: PASS"
else
  overwrite_args=()
  if [[ -d "$GENERATION_DIR" ]] && [[ -n "$(find "$GENERATION_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    if [[ "${V13_OVERWRITE_FAILED_GENERATION:-0}" != "1" ]]; then
      echo "Existing V13 generation evidence was preserved."
      echo "Inspect: $GENERATION_MANIFEST"
      echo "A deliberate fresh rerun requires V13_OVERWRITE_FAILED_GENERATION=1."
      exit 1
    fi
    overwrite_args+=(--overwrite)
  fi
  "$PYTHON_BIN" "$GENERATOR" \
    --plan "$PLAN" \
    --model "$MODEL" \
    --model-manifest "$MODEL_MANIFEST" \
    --representation-audit "$AUDIT_JSON" \
    --native-jsonl "$NATIVE_JSONL" \
    --best-csv "$BEST_CSV" \
    --out-dir "$GENERATION_DIR" \
    "${overwrite_args[@]}"
fi

echo "===== V13 ALL GATES PASSED ====="
echo "Yield table: $GENERATION_DIR/final_yield_summary_by_target.csv"
echo "Shangge CSV: $GENERATION_DIR/1700_给尚哥_极简.csv"
echo "Shangge FASTA: $GENERATION_DIR/1700_给尚哥_结构输入.fasta"
echo "RMSD remains intentionally uncomputed until structures return."
