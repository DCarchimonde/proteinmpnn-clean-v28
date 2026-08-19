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
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

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

"$TASK_PYTHON" - "$TASK_REPO" "$TASK_V8_ROOT" "$TASK_IMPORT_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


repo = Path(sys.argv[1]).resolve()
v8_root = Path(sys.argv[2]).resolve()
import_manifest_path = Path(sys.argv[3]).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if not import_manifest_path.is_file():
    raise SystemExit(f"Runtime preflight missing import manifest: {import_manifest_path}")
import_manifest = json.loads(import_manifest_path.read_text(encoding="utf-8"))
imported_files = dict(import_manifest.get("current_imported_file_hashes") or {})
if not imported_files:
    raise SystemExit(
        "Runtime preflight requires a bundle exported with the complete portable "
        "input contract"
    )
for name, expected in imported_files.items():
    path = repo / name
    if not path.is_file():
        raise SystemExit(f"Runtime preflight missing imported file: {path}")
    if sha256_file(path) != str(expected):
        raise SystemExit(f"Runtime preflight imported-file hash mismatch: {path}")

required = [
    repo / "17_complexes_native.jsonl",
    repo / "model_utils.py",
    repo / "nmethyl/utils/nmethyl_config.py",
    repo / "paper_clean_v28/clean_v28_common.py",
    repo / "paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py",
    repo / "paper_clean_v28/serine_qc_retrain/10_reannotate_v6_pool_serine_only_v7.py",
    repo / "paper_clean_v28/serine_qc_retrain/11_triple_audit_serine_only_v7.py",
    repo / "paper_clean_v28/serine_qc_retrain/14_directed_recovery_search_v8.py",
    repo / "paper_clean_v28/serine_qc_retrain/15_finalize_and_audit_recovery_v8.py",
    repo / "paper_clean_v28/serine_qc_retrain/target_plan_cyclic_representation_v6.json",
    repo / "paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv",
    repo / "paper_clean_v28_outputs/rerun_temperature_0.5_multiseed/methylated_new_candidates.csv",
    v8_root / "model/frankenstein_v28_source_scoped_hybrid_v8.pt",
    v8_root / "model/expert_source_composition_manifest.json",
    v8_root / "representation_audit/cyclic_representation_audit.json",
    v8_root / "generation_baseline/all_candidates.csv",
    v8_root / "generation_baseline/unique_candidates.csv",
    v8_root / "generation_baseline/methylated_new_candidates.csv",
    v8_root / "generation_baseline/target_manifest.csv",
    v8_root / "generation_baseline/generation_summary_by_target.csv",
    v8_root / "generation_baseline/generation_manifest.json",
    v8_root / "directed_search/mandatory_length_6_7_controls.csv",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(
        "Runtime preflight found all missing inputs before GPU work:\n- "
        + "\n- ".join(missing)
    )
print(
    f"===== V8 AUTODL RUNTIME PREFLIGHT PASSED: "
    f"{len(imported_files)} imported + {len(required)} runtime files =====",
    flush=True,
)
PY

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
