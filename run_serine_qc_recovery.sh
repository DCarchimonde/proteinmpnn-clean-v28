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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) TASK_PYTHON="$2"; shift 2 ;;
    --source-repo) TASK_SOURCE_REPO="$2"; shift 2 ;;
    --epochs) TASK_EPOCHS="$2"; shift 2 ;;
    --batch-size) TASK_BATCH_SIZE="$2"; shift 2 ;;
    --prior-handoff-csv) TASK_PRIOR_HANDOFF="$2"; shift 2 ;;
    --allow-cpu) TASK_ALLOW_CPU="1"; shift ;;
    --force) TASK_FORCE="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

SOURCE_COMMIT="28dff152d83623dfb322480413b7dc889f8537a4"
OUTPUT_ROOT="$SCRIPT_DIR/paper_clean_v28_outputs/serine_qc_retrain"
DATA_OUT="$OUTPUT_ROOT/data"
MODEL_OUT="$OUTPUT_ROOT/model"
TEST_EVAL_OUT="$MODEL_OUT/full_corrected_test_eval"
BRIDGE_OUT="$OUTPUT_ROOT/bridge"
GENERATION_OUT="$OUTPUT_ROOT/generation"
HANDOFF_OUT="$OUTPUT_ROOT/handoff"
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

"$TASK_PYTHON" paper_clean_v28/01_eval_clean_model.py \
  --model_path "$CORRECTED_CHECKPOINT" \
  --data_jsonl "$DATA_OUT/test_serine_provenance_corrected.jsonl" \
  --mode monomer \
  --eval_chains masked \
  --batch_size "$TASK_BATCH_SIZE" \
  --thresholds "0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99" \
  --out_dir "$TEST_EVAL_OUT"

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
"$TASK_PYTHON" "${GENERATION_ARGS[@]}"

SELECT_ARGS=(
  paper_clean_v28/serine_qc_retrain/03_select_structure_first_handoff.py
  --run-dir "$GENERATION_OUT"
  --plan "$PLAN"
  --out-dir "$HANDOFF_OUT"
)
SELECT_ARGS+=(--prior-handoff-csv "$PRIOR_HANDOFF")
"$TASK_PYTHON" "${SELECT_ARGS[@]}"

echo "ALL QUALITY GATES PASSED"
echo "Corrected checkpoint: $CORRECTED_CHECKPOINT"
echo "Frozen-target bridge: $BRIDGE_OUT/frozen_target_final_model_bridge.csv"
echo "Shang-ge handoff: $HANDOFF_OUT/structure_tasks_for_shangge.csv"
echo "Permeability: DEFERRED until returned structures pass the structure gate"
