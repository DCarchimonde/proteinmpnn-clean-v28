#!/usr/bin/env bash
set -Eeuo pipefail

TASK_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_BUNDLE="${1:-/root/autodl-tmp/v8_autodl_resume_bundle.zip}"
TASK_PYTHON="${PYTHON:-python}"
TASK_V8_ROOT="$TASK_REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8"
TASK_IMPORT_MANIFEST="$TASK_V8_ROOT/autodl_import_manifest.json"
TASK_LOG="$TASK_V8_ROOT/v8_autodl_resume.log"

mkdir -p "$TASK_V8_ROOT"
exec > >(tee -a "$TASK_LOG") 2>&1

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "V8 AUTODL PORTABLE RESUME + FULL DESTINATION REAUDIT"
echo "Repository: $TASK_REPO"
echo "Bundle:     $TASK_BUNDLE"
echo "Python:     $TASK_PYTHON"
echo "============================================================"

if [[ ! -f "$TASK_BUNDLE" ]]; then
  echo "ERROR: resume bundle not found: $TASK_BUNDLE" >&2
  exit 2
fi

"$TASK_PYTHON" - <<'PY'
import json
import sys

try:
    import numpy
    import torch
except ImportError as exc:
    raise SystemExit(f"Missing Python dependency: {exc}") from exc

if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 is required; observed {sys.version.split()[0]}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this AutoDL image")

capability = torch.cuda.get_device_capability(0)
arches = set(torch.cuda.get_arch_list())
required_arch = f"sm_{capability[0]}{capability[1]}"
if arches and required_arch not in arches:
    raise SystemExit(
        f"Installed PyTorch lacks native {required_arch} support; available={sorted(arches)}"
    )

print(json.dumps({
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": capability,
    "native_arch": required_arch,
    "device_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
}, ensure_ascii=False))
PY

cd "$TASK_REPO"

"$TASK_PYTHON" \
  paper_clean_v28/serine_qc_retrain/16_v8_autodl_resume_bundle.py \
  import --bundle "$TASK_BUNDLE"

"$TASK_PYTHON" \
  paper_clean_v28/serine_qc_retrain/14_directed_recovery_search_v8.py \
  --model-path "$TASK_V8_ROOT/model/frankenstein_v28_source_scoped_hybrid_v8.pt" \
  --model-manifest "$TASK_V8_ROOT/model/expert_source_composition_manifest.json" \
  --representation-audit "$TASK_V8_ROOT/representation_audit/cyclic_representation_audit.json" \
  --baseline-run-dir "$TASK_V8_ROOT/generation_baseline" \
  --plan "$TASK_REPO/paper_clean_v28/serine_qc_retrain/target_plan_cyclic_representation_v6.json" \
  --native-jsonl "$TASK_REPO/17_complexes_native.jsonl" \
  --historical-designs-csv "$TASK_REPO/paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv" \
  --prior-handoff-csv "$TASK_REPO/paper_clean_v28_outputs/rerun_temperature_0.5_multiseed/methylated_new_candidates.csv" \
  --out-dir "$TASK_V8_ROOT/directed_search" \
  --batch-size 64 \
  --base-batch-size 32 \
  --wne-radius 2 \
  --zgc-rounds 6 \
  --zgc-beam-width 512 \
  --zgc-offspring-per-round 4096 \
  --max-release-per-target 200 \
  --device cuda \
  --resume \
  --portable-resume-manifest "$TASK_IMPORT_MANIFEST"

"$TASK_PYTHON" \
  paper_clean_v28/serine_qc_retrain/15_finalize_and_audit_recovery_v8.py \
  --model-path "$TASK_V8_ROOT/model/frankenstein_v28_source_scoped_hybrid_v8.pt" \
  --model-manifest "$TASK_V8_ROOT/model/expert_source_composition_manifest.json" \
  --representation-audit "$TASK_V8_ROOT/representation_audit/cyclic_representation_audit.json" \
  --baseline-run-dir "$TASK_V8_ROOT/generation_baseline" \
  --search-dir "$TASK_V8_ROOT/directed_search" \
  --plan "$TASK_REPO/paper_clean_v28/serine_qc_retrain/target_plan_cyclic_representation_v6.json" \
  --native-jsonl "$TASK_REPO/17_complexes_native.jsonl" \
  --historical-designs-csv "$TASK_REPO/paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv" \
  --prior-handoff-csv "$TASK_REPO/paper_clean_v28_outputs/rerun_temperature_0.5_multiseed/methylated_new_candidates.csv" \
  --out-dir "$TASK_V8_ROOT/generation_recovered" \
  --audit-out-dir "$TASK_V8_ROOT/triple_audit_recovered" \
  --device cuda \
  --overwrite

"$TASK_PYTHON" \
  paper_clean_v28/serine_qc_retrain/16_v8_autodl_resume_bundle.py \
  package-review

echo "Log: $TASK_LOG"
