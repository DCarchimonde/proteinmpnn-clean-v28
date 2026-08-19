#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON="python"
TASK_SOURCE_REPO=""
TASK_EPOCHS="80"
TASK_BATCH_SIZE="32"
TASK_ALLOW_CPU="0"
TASK_FORCE="0"
TASK_PRIOR_HANDOFF=""
TASK_RELEASE_HANDOFF="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) TASK_PYTHON="$2"; shift 2 ;;
    --source-repo) TASK_SOURCE_REPO="$2"; shift 2 ;;
    --epochs) TASK_EPOCHS="$2"; shift 2 ;;
    --batch-size) TASK_BATCH_SIZE="$2"; shift 2 ;;
    --prior-handoff-csv) TASK_PRIOR_HANDOFF="$2"; shift 2 ;;
    --allow-cpu) TASK_ALLOW_CPU="1"; shift ;;
    --force) TASK_FORCE="1"; shift ;;
    --release-handoff) TASK_RELEASE_HANDOFF="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

SOURCE_COMMIT="28dff152d83623dfb322480413b7dc889f8537a4"
OUTPUT_ROOT="$SCRIPT_DIR/paper_clean_v28_outputs/serine_qc_peptide_only_v4"
DATA_OUT="$OUTPUT_ROOT/data"
MODEL_OUT="$OUTPUT_ROOT/model"
BRIDGE_OUT="$OUTPUT_ROOT/bridge"
GENERATION_OUT="$OUTPUT_ROOT/generation"
AUDIT_OUT="$OUTPUT_ROOT/triple_audit"
HANDOFF_OUT="$OUTPUT_ROOT/handoff"
AUDIT_BUNDLE="$OUTPUT_ROOT/serine_qc_peptide_only_v4_review_bundle.zip"
PLAN="$SCRIPT_DIR/paper_clean_v28/serine_qc_retrain/target_plan_structure_failures.json"
PARENT_CHECKPOINT="$SCRIPT_DIR/frankenstein_v28.pt"
CORRECTED_CHECKPOINT="$MODEL_OUT/frankenstein_v28_expert_heads_qc.pt"
if [[ -z "$TASK_PRIOR_HANDOFF" ]]; then
  PRIOR_HANDOFF="$SCRIPT_DIR/paper_clean_v28_outputs/rerun_temperature_0.5_multiseed/methylated_new_candidates.csv"
else
  PRIOR_HANDOFF="$TASK_PRIOR_HANDOFF"
fi

"$TASK_PYTHON" -c 'import json, numpy, torch; print(json.dumps({"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'
if [[ "$TASK_ALLOW_CPU" != "1" ]]; then
  "$TASK_PYTHON" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)'
fi

"$TASK_PYTHON" "$SCRIPT_DIR/paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py" \
  --plan "$PLAN" \
  --prior_designs_csv "$PRIOR_HANDOFF" \
  --validate-prior-designs-only

if [[ -z "$TASK_SOURCE_REPO" ]]; then
  TASK_SOURCE_REPO="$SCRIPT_DIR/.serine_qc_source/ProteinMPNN"
  if [[ ! -d "$TASK_SOURCE_REPO/.git" ]]; then
    if [[ -e "$TASK_SOURCE_REPO" ]]; then
      echo "Managed source path exists but is not a Git checkout: $TASK_SOURCE_REPO" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$TASK_SOURCE_REPO")"
    git clone --filter=blob:none --sparse https://github.com/DCarchimonde/ProteinMPNN.git "$TASK_SOURCE_REPO"
  fi
  git -C "$TASK_SOURCE_REPO" fetch origin "$SOURCE_COMMIT" --depth 1
  git -C "$TASK_SOURCE_REPO" sparse-checkout set nmethyl_data/raw_pdb nmethyl_data/training_set nmethyl_data/test_set
  git -C "$TASK_SOURCE_REPO" checkout --detach "$SOURCE_COMMIT"
else
  OBSERVED_SOURCE_COMMIT="$(git -C "$TASK_SOURCE_REPO" rev-parse HEAD)"
  if [[ "$OBSERVED_SOURCE_COMMIT" != "$SOURCE_COMMIT" ]]; then
    echo "--source-repo must be pinned to $SOURCE_COMMIT; observed $OBSERVED_SOURCE_COMMIT" >&2
    exit 2
  fi
fi

TRAIN_JSONL="$TASK_SOURCE_REPO/nmethyl_data/training_set/train.jsonl"
TEST_JSONL="$TASK_SOURCE_REPO/nmethyl_data/test_set/test.jsonl"
RAW_PDB_DIR="$TASK_SOURCE_REPO/nmethyl_data/raw_pdb"
if [[ ! -f "$PRIOR_HANDOFF" ]]; then
  echo "Prior 1,333-row handoff is required: $PRIOR_HANDOFF" >&2
  echo "Pass --prior-handoff-csv /path/to/methylated_new_candidates.csv if it moved." >&2
  exit 2
fi

cd "$SCRIPT_DIR"
"$TASK_PYTHON" paper_clean_v28/serine_qc_retrain/01_rebuild_provenance_labels.py \
  --train-jsonl "$TRAIN_JSONL" \
  --test-jsonl "$TEST_JSONL" \
  --raw-pdb-dir "$RAW_PDB_DIR" \
  --out-dir "$DATA_OUT" \
  --source-commit "$SOURCE_COMMIT"

TRAIN_ARGS=(
  paper_clean_v28/serine_qc_retrain/02_retrain_canonical_expert_heads.py
  --model-path "$PARENT_CHECKPOINT"
  --train-jsonl "$DATA_OUT/train_serine_provenance_corrected.jsonl"
  --test-jsonl "$DATA_OUT/test_serine_provenance_corrected.jsonl"
  --out-dir "$MODEL_OUT"
  --epochs "$TASK_EPOCHS"
  --batch-size "$TASK_BATCH_SIZE"
)
if [[ "$TASK_ALLOW_CPU" == "1" ]]; then TRAIN_ARGS+=(--allow-cpu); fi
"$TASK_PYTHON" "${TRAIN_ARGS[@]}"

BRIDGE_ARGS=(
  paper_clean_v28/serine_qc_retrain/03_revalidate_frozen_structures.py
  --model-path "$CORRECTED_CHECKPOINT"
  --plan "$PLAN"
  --native-jsonl "$SCRIPT_DIR/17_complexes_native.jsonl"
  --out-dir "$BRIDGE_OUT"
)
if [[ "$TASK_ALLOW_CPU" == "1" ]]; then BRIDGE_ARGS+=(--allow-cpu); fi
"$TASK_PYTHON" "${BRIDGE_ARGS[@]}"

GENERATION_ARGS=(
  paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py
  --plan "$PLAN"
  --model_path "$CORRECTED_CHECKPOINT"
  --out_dir "$GENERATION_OUT"
  --batch_size "$TASK_BATCH_SIZE"
  --prior_designs_csv "$PRIOR_HANDOFF"
  --defer-permeability-until-structure
)
if [[ "$TASK_ALLOW_CPU" == "1" ]]; then GENERATION_ARGS+=(--device auto --allow-cpu); else GENERATION_ARGS+=(--device cuda); fi
if [[ "$TASK_FORCE" == "1" ]]; then GENERATION_ARGS+=(--overwrite); fi
set +e
"$TASK_PYTHON" "${GENERATION_ARGS[@]}"
GENERATION_EXIT_CODE=$?
set -e

AUDIT_EXIT_CODE=1
if [[ -f "$GENERATION_OUT/generation_manifest.json" ]]; then
  set +e
  "$TASK_PYTHON" paper_clean_v28/serine_qc_retrain/04_triple_audit_generation.py \
    --run-dir "$GENERATION_OUT" \
    --plan "$PLAN" \
    --prior-handoff-csv "$PRIOR_HANDOFF" \
    --out-dir "$AUDIT_OUT"
  AUDIT_EXIT_CODE=$?
  set -e
fi

BUNDLE_SOURCES=()
for FILE in \
  "$MODEL_OUT/expert_heads_retrain_manifest.json" \
  "$MODEL_OUT/training_history.csv" \
  "$MODEL_OUT/test_metrics_by_residue.csv" \
  "$MODEL_OUT/test_position_probabilities.csv" \
  "$BRIDGE_OUT/frozen_target_bridge_manifest.json" \
  "$BRIDGE_OUT/frozen_target_final_model_bridge.csv" \
  "$GENERATION_OUT/generation_manifest.json" \
  "$GENERATION_OUT/generation_summary_by_target.csv" \
  "$GENERATION_OUT/all_candidates.csv" \
  "$GENERATION_OUT/unique_candidates.csv" \
  "$GENERATION_OUT/methylated_new_candidates.csv" \
  "$AUDIT_OUT/three_pass_generation_audit.json" \
  "$AUDIT_OUT/three_pass_concentration_by_target.csv"
do
  if [[ -f "$FILE" ]]; then BUNDLE_SOURCES+=("$FILE"); fi
done
if [[ ${#BUNDLE_SOURCES[@]} -gt 0 ]]; then
  "$TASK_PYTHON" - "$AUDIT_BUNDLE" "${BUNDLE_SOURCES[@]}" <<'PY'
import sys
import zipfile
from pathlib import Path

destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for value in sys.argv[2:]:
        path = Path(value)
        archive.write(path, arcname=path.name)
PY
fi

if [[ "$GENERATION_EXIT_CODE" -ne 0 ]]; then
  echo "Generation was blocked; upload $AUDIT_BUNDLE and do not release a handoff." >&2
  exit "$GENERATION_EXIT_CODE"
fi
if [[ "$AUDIT_EXIT_CODE" -ne 0 ]]; then
  echo "Independent three-pass audit was blocked; upload $AUDIT_BUNDLE and do not release a handoff." >&2
  exit "$AUDIT_EXIT_CODE"
fi

if [[ "$TASK_RELEASE_HANDOFF" == "1" ]]; then
  SELECT_ARGS=(
    paper_clean_v28/serine_qc_retrain/03_select_structure_first_handoff.py
    --run-dir "$GENERATION_OUT"
    --plan "$PLAN"
    --out-dir "$HANDOFF_OUT"
    --prior-handoff-csv "$PRIOR_HANDOFF"
  )
  "$TASK_PYTHON" "${SELECT_ARGS[@]}"
fi

echo "AUTOMATED V4 QUALITY GATES PASSED"
echo "Corrected checkpoint: $CORRECTED_CHECKPOINT"
echo "Frozen-target bridge: $BRIDGE_OUT/frozen_target_final_model_bridge.csv"
echo "Manual-review bundle: $AUDIT_BUNDLE"
if [[ "$TASK_RELEASE_HANDOFF" == "1" ]]; then
  echo "Shang-ge handoff: $HANDOFF_OUT/structure_tasks_for_shangge.csv"
else
  echo "Release status: HOLD FOR MANUAL REVIEW; no Shang-ge handoff was created"
fi
echo "Permeability: DEFERRED until returned structures pass the structure gate"
