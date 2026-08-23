#!/usr/bin/env bash
# Resume from the completed V11 model/audit/pool and produce the real requested
# pre-structure deliverable: 17 targets x exactly 100 strict methylated sequences.
# No base-floor release gate and no pre-structure RMSD ranking are run here.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${V12_PYTHON:-python}"
V11_ROOT="${V11_OUTPUT_ROOT:-$REPO_ROOT/paper_clean_v28_outputs/cyclic_native_v11_1700_monomer}"
V12_ROOT="${V12_OUTPUT_ROOT:-$V11_ROOT/v12_methyl_only}"
MODEL="$V11_ROOT/model/frankenstein_v28_expert_heads_qc.pt"
AUDIT="$V11_ROOT/representation_audit/cyclic_representation_audit.json"
GENERATION="$V11_ROOT/generation"
MONOMER="$V11_ROOT/monomer_final_v11/monomer_v10_final_manifest.json"
V11_PLAN="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/target_plan_v11_cyclic_native_rmsd_priority_1700.json"
V12_PLAN="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/target_plan_v12_methyl_only_1700.json"
SEARCH="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/33_recover_3zgc_methyl_only_v12.py"
SELECTOR="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/34_select_methyl_only_1700_v12.py"
REPLAYER="$REPO_ROOT/paper_clean_v28/serine_qc_retrain/35_replay_methyl_only_1700_v12.py"
SEARCH_DIR="$V12_ROOT/3zgc_directed_search"
SELECTION_DIR="$V12_ROOT/selection_17x100"
FINAL_DIR="$V12_ROOT/final_independent_replay_handoff"
ARCHIVE="$V12_ROOT/v12_1700_给尚哥_甲基化序列_待结构.tar.gz"

echo "===== V12 REAL PRE-STRUCTURE TASK: METHYLATION-ONLY 17 x 100 ====="
echo "Reusing V11 model/audit/pool: $V11_ROOT"
echo "Output root: $V12_ROOT"
echo "Base score gate: NOT USED"
echo "RMSD ranking: NOT AVAILABLE BEFORE SHANG-GE RETURNS STRUCTURES"

for required in \
  "$MODEL" "$AUDIT" "$GENERATION/generation_manifest.json" \
  "$GENERATION/all_candidates.csv" "$GENERATION/methylated_new_candidates.csv" \
  "$V11_PLAN" "$V12_PLAN" "$SEARCH" "$SELECTOR" "$REPLAYER" \
  "$REPO_ROOT/17_complexes_native.jsonl" \
  "$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv" \
  "$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv" \
  "$REPO_ROOT/v9_inputs/methylated_new_candidates.csv" \
  "$REPO_ROOT/v10_inputs/six_non3av_t05_joint_rmsd_476.csv"; do
  [[ -s "$required" ]] || {
    echo "ERROR: required V11/V12 input is missing or empty: $required" >&2
    exit 20
  }
done

"$PYTHON_BIN" - <<'PY'
import json
import sys
import torch
if sys.version_info < (3, 10):
    raise SystemExit("ERROR: Python >=3.10 is required")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA PyTorch is required for V12 model replay")
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
}, ensure_ascii=False))
PY

echo "[1/5] V12 code tests"
(cd "$REPO_ROOT" && "$PYTHON_BIN" -m unittest \
  tests.test_v12_methyl_only_prestructure \
  tests.test_v11_generation_recovery \
  tests.test_v11_cyclic_native_model)

echo "[2/5] Same-checkpoint 3ZGC replay and bounded complete-sequence recovery"
search_args=(
  --model "$MODEL"
  --representation-audit "$AUDIT"
  --generation-dir "$GENERATION"
  --v11-plan "$V11_PLAN"
  --native-jsonl "$REPO_ROOT/17_complexes_native.jsonl"
  --best-csv "$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv"
  --historical-csv "$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv"
  --prior-csv "$REPO_ROOT/v9_inputs/methylated_new_candidates.csv"
  --rmsd-history-csv "$REPO_ROOT/v10_inputs/six_non3av_t05_joint_rmsd_476.csv"
  --out-dir "$SEARCH_DIR"
  --quota 100
  --rounds "${V12_3ZGC_ROUNDS:-6}"
  --beam-width "${V12_3ZGC_BEAM_WIDTH:-512}"
  --random-offspring-per-round "${V12_3ZGC_RANDOM_OFFSPRING:-4096}"
  --batch-size "${V12_SEARCH_BATCH_SIZE:-64}"
  --device cuda
)
if [[ -z "${V12_3ZGC_EXTRA_SEED_CSV:-}" ]]; then
  # If the earlier V8 scientific ledger is still present on this AutoDL
  # instance, replay its 3ZGC strict-hit sequences under V11 before searching.
  # It is seed evidence only: no V8 probability/base decision is trusted.
  for candidate in \
    /root/autodl-tmp/proteinmpnn-clean-v28*/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/direct*search/qualified_candidate_plausibility_and_novelty.csv; do
    if [[ -s "$candidate" ]]; then
      V12_3ZGC_EXTRA_SEED_CSV="$candidate"
      echo "Found prior V8 3ZGC sequence ledger for same-V11 replay: $candidate"
      break
    fi
  done
fi
if [[ -n "${V12_3ZGC_EXTRA_SEED_CSV:-}" ]]; then
  [[ -s "$V12_3ZGC_EXTRA_SEED_CSV" ]] || {
    echo "ERROR: V12_3ZGC_EXTRA_SEED_CSV does not exist: $V12_3ZGC_EXTRA_SEED_CSV" >&2
    exit 21
  }
  search_args+=(--seed-csv "$V12_3ZGC_EXTRA_SEED_CSV")
fi
if "$PYTHON_BIN" - "$SEARCH_DIR" "$SEARCH" "$MODEL" "$AUDIT" <<'PY'
import hashlib, json, pathlib, sys
root, program, model, audit = map(pathlib.Path, sys.argv[1:])
manifest_path = root / "3zgc_methyl_only_search_manifest.json"
release_path = root / "3zgc_exact_100_methylated.csv"
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
if not manifest_path.is_file() or not release_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
ok = (
    manifest.get("quality_gate") == "PASS"
    and manifest.get("release_status")
    == "AUTHORIZED_3ZGC_EXACT_100_METHYLATION_ONLY_PRESTRUCTURE_ROWS"
    and manifest.get("program", {}).get("sha256") == digest(program)
    and manifest.get("inputs", {}).get("model", {}).get("sha256") == digest(model)
    and manifest.get("inputs", {}).get("representation_audit", {}).get("sha256")
    == digest(audit)
    and manifest.get("artifacts", {}).get("exact_100_release", {}).get("sha256")
    == digest(release_path)
)
raise SystemExit(0 if ok else 1)
PY
then
  echo "[2/5] Reusing hash-pinned PASS exact-100 3ZGC recovery"
else
  "$PYTHON_BIN" "$SEARCH" "${search_args[@]}"
fi

echo "[3/5] Exact 17 x 100 methylation-only selection"
if "$PYTHON_BIN" - "$SELECTION_DIR" "$SELECTOR" "$MODEL" "$AUDIT" \
  "$GENERATION/generation_manifest.json" "$SEARCH_DIR/3zgc_methyl_only_search_manifest.json" \
  "$V12_PLAN" <<'PY'
import hashlib, json, pathlib, sys
root, program, model, audit, generation, zgc, plan = map(pathlib.Path, sys.argv[1:])
manifest_path = root / "v12_1700_methyl_only_release_manifest.json"
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
if not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
artifacts = manifest.get("artifacts", {})
required = ["detailed", "shangge_concise", "shangge_fasta"]
ok = (
    manifest.get("quality_gate") == "PASS"
    and manifest.get("selected_rows") == 1700
    and manifest.get("program", {}).get("sha256") == digest(program)
    and manifest.get("inputs", {}).get("model", {}).get("sha256") == digest(model)
    and manifest.get("inputs", {}).get("representation_audit", {}).get("sha256")
    == digest(audit)
    and manifest.get("inputs", {}).get("generation_manifest", {}).get("sha256")
    == digest(generation)
    and manifest.get("inputs", {}).get("zgc_search_manifest", {}).get("sha256")
    == digest(zgc)
    and manifest.get("inputs", {}).get("plan", {}).get("sha256") == digest(plan)
    and all(
        pathlib.Path(artifacts[name]["path"]).is_file()
        and artifacts[name]["sha256"] == digest(pathlib.Path(artifacts[name]["path"]))
        for name in required
    )
)
raise SystemExit(0 if ok else 1)
PY
then
  echo "[3/5] Reusing hash-pinned PASS exact 17 x 100 selection"
else
  "$PYTHON_BIN" "$SELECTOR" \
    --generation-dir "$GENERATION" \
    --zgc-dir "$SEARCH_DIR" \
    --plan "$V12_PLAN" \
    --model "$MODEL" \
    --representation-audit "$AUDIT" \
    --historical-csv "$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv" \
    --prior-csv "$REPO_ROOT/v9_inputs/methylated_new_candidates.csv" \
    --out-dir "$SELECTION_DIR"
fi

echo "[4/5] Independent batch-one replay of every one of the 1,700 rows"
if "$PYTHON_BIN" - "$FINAL_DIR" "$REPLAYER" "$MODEL" "$AUDIT" \
  "$SELECTION_DIR/v12_1700_methyl_only_release_manifest.json" "$V11_PLAN" \
  "$REPO_ROOT/17_complexes_native.jsonl" \
  "$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv" <<'PY'
import hashlib, json, pathlib, sys
root, program, model, audit, selector, plan, native, best = map(pathlib.Path, sys.argv[1:])
manifest_path = root / "v12_1700_methyl_only_independent_replay_manifest.json"
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
if not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
artifacts = manifest.get("artifacts", {})
required = ["detailed", "shangge_concise", "shangge_fasta", "replay_csv"]
ok = (
    manifest.get("quality_gate") == "PASS"
    and manifest.get("replayed_rows") == 1700
    and manifest.get("program", {}).get("sha256") == digest(program)
    and manifest.get("inputs", {}).get("model", {}).get("sha256") == digest(model)
    and manifest.get("inputs", {}).get("representation_audit", {}).get("sha256")
    == digest(audit)
    and manifest.get("inputs", {}).get("selector_manifest", {}).get("sha256")
    == digest(selector)
    and manifest.get("inputs", {}).get("v11_plan", {}).get("sha256") == digest(plan)
    and manifest.get("inputs", {}).get("native_jsonl", {}).get("sha256")
    == digest(native)
    and manifest.get("inputs", {}).get("best_csv", {}).get("sha256") == digest(best)
    and all(
        pathlib.Path(artifacts[name]["path"]).is_file()
        and artifacts[name]["sha256"] == digest(pathlib.Path(artifacts[name]["path"]))
        for name in required
    )
)
raise SystemExit(0 if ok else 1)
PY
then
  echo "[4/5] Reusing hash-pinned PASS 1,700-row batch-one replay"
else
  "$PYTHON_BIN" "$REPLAYER" \
    --selection-dir "$SELECTION_DIR" \
    --model "$MODEL" \
    --representation-audit "$AUDIT" \
    --v11-plan "$V11_PLAN" \
    --native-jsonl "$REPO_ROOT/17_complexes_native.jsonl" \
    --best-csv "$REPO_ROOT/paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv" \
    --out-dir "$FINAL_DIR" \
    --device cuda
fi

echo "[5/5] Packaging only the verified sequence handoff and audit evidence"
[[ -s "$MONOMER" ]] || echo "NOTE: V11 monomer manifest is absent; complex 1700 handoff is unaffected."
archive_tmp="${ARCHIVE}.tmp.$$"
tar -czf "$archive_tmp" -C "$V12_ROOT" \
  3zgc_directed_search/3zgc_same_v11_seed_replay.csv \
  3zgc_directed_search/3zgc_search_trace.csv \
  3zgc_directed_search/3zgc_methyl_only_search_manifest.json \
  selection_17x100/selection_summary_by_target.csv \
  selection_17x100/candidate_validation_problems.csv \
  selection_17x100/v12_1700_methyl_only_release_manifest.json \
  final_independent_replay_handoff/1700_详细审计.csv \
  final_independent_replay_handoff/1700_给尚哥_极简.csv \
  final_independent_replay_handoff/1700_给尚哥_结构输入.fasta \
  final_independent_replay_handoff/1700_独立逐条甲基化复算.csv \
  final_independent_replay_handoff/v12_1700_methyl_only_independent_replay_manifest.json
mv -f "$archive_tmp" "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

echo "===== V12 ALL PRE-STRUCTURE METHYLATION GATES PASSED ====="
echo "Rows: 1,700 = 17 targets x exactly 100"
echo "All rows: strict V11 representation-min methylation PASS"
echo "Observed RMSD: intentionally pending Shang-ge structures"
echo "Shang-ge CSV: $FINAL_DIR/1700_给尚哥_极简.csv"
echo "Shang-ge FASTA: $FINAL_DIR/1700_给尚哥_结构输入.fasta"
echo "Audit archive: $ARCHIVE"
echo "Archive SHA-256: $ARCHIVE.sha256"
