#!/usr/bin/env bash
set -Eeuo pipefail

export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16

WORK="/root/autodl-tmp"
UPLOAD="$WORK/methylated_new_candidates.csv"
REPO="$WORK/proteinmpnn-clean-v28-v8-autodl"
BUNDLE="$WORK/v8_autodl_resume_bundle.zip"
V8_ROOT="$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8"
IMPORT_MANIFEST="$V8_ROOT/autodl_import_manifest.json"
RECOVERY_LOG="$WORK/v8_recovery_after_prior.log"
PIDFILE="$WORK/v8_autodl_resume.pid"
EXPECTED_SOURCE="0e916c311956108269b57960fc2ca18d388a3713"

echo "[1/3] 验证补传文件及全部运行输入"
[[ -s "$UPLOAD" ]] || {
    echo "ERROR: 请先上传到 $UPLOAD"
    exit 1
}
[[ -s "$BUNDLE" && -s "$IMPORT_MANIFEST" ]] || {
    echo "ERROR: 旧迁移包或已导入 manifest 不存在"
    exit 1
}
if [[ -f "$REPO/.source_commit" ]]; then
    grep -qx "$EXPECTED_SOURCE" "$REPO/.source_commit" || {
        echo "ERROR: 当前 AutoDL 源码版本不是旧包对应的 $EXPECTED_SOURCE"
        exit 1
    }
fi

python - "$UPLOAD" "$REPO" "$V8_ROOT" "$IMPORT_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys


upload = pathlib.Path(sys.argv[1]).resolve()
repo = pathlib.Path(sys.argv[2]).resolve()
v8_root = pathlib.Path(sys.argv[3]).resolve()
import_manifest_path = pathlib.Path(sys.argv[4]).resolve()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


imported = json.loads(import_manifest_path.read_text(encoding="utf-8"))
portable_path = pathlib.Path(imported["portable_export_manifest"]).resolve()
portable = json.loads(portable_path.read_text(encoding="utf-8"))
expected_prior = str(portable["source_config"]["input_hashes"]["prior"])
observed_prior = sha256_file(upload)
if observed_prior != expected_prior:
    raise SystemExit(
        "ERROR: 补传文件不是本次搜索使用的 prior CSV；"
        f" expected={expected_prior}; observed={observed_prior}"
    )

destination = (
    repo
    / "paper_clean_v28_outputs"
    / "rerun_temperature_0.5_multiseed"
    / "methylated_new_candidates.csv"
)
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.is_file():
    if sha256_file(destination) != expected_prior:
        raise SystemExit(f"ERROR: 目标位置已有不同文件：{destination}")
else:
    shutil.copy2(upload, destination)
if sha256_file(destination) != expected_prior:
    raise SystemExit("ERROR: prior CSV 复制后哈希不一致")

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
    destination,
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
        "ERROR: 一次性输入审计发现以下缺项：\n- " + "\n- ".join(missing)
    )

current_paths = {
    "model": v8_root / "model/frankenstein_v28_source_scoped_hybrid_v8.pt",
    "model_manifest": v8_root / "model/expert_source_composition_manifest.json",
    "representation_audit": v8_root / "representation_audit/cyclic_representation_audit.json",
    "baseline_manifest": v8_root / "generation_baseline/generation_manifest.json",
}
for name, expected in dict(imported.get("current_input_hashes") or {}).items():
    path = current_paths[name]
    if sha256_file(path) != str(expected):
        raise SystemExit(f"ERROR: 已导入核心输入哈希不一致：{path}")

evidence = dict(imported.get("evidence_files") or {})
for relative, expected in evidence.items():
    path = repo / relative
    if not path.is_file() or sha256_file(path) != str(expected):
        raise SystemExit(f"ERROR: 搜索 ledger/checkpoint 哈希不一致：{path}")

print(f"Prior SHA256: {observed_prior}")
print(
    f"===== COMPLETE PRE-RUN AUDIT PASSED: "
    f"{len(required)} runtime files + {len(evidence)} evidence files ====="
)
PY

echo "[2/3] 原地续跑，不重跑六轮搜索"
if [[ -e "$RECOVERY_LOG" ]]; then
    mv "$RECOVERY_LOG" "${RECOVERY_LOG}.previous.$(date +%s)"
fi

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
    bash "$REPO/run_v8_autodl_resume.sh" "$BUNDLE" \
    >> "$RECOVERY_LOG" 2>&1 < /dev/null &
RUN_PID=$!
printf '%s\n' "$RUN_PID" > "$PIDFILE"
sleep 15
if ! kill -0 "$RUN_PID" 2>/dev/null; then
    echo "ERROR: 续跑任务启动后退出，完整末尾日志如下："
    tail -n 220 "$RECOVERY_LOG"
    exit 1
fi

echo "[3/3] 续跑已启动，PID=$RUN_PID；现在显示实时进度。"
echo "关闭网页不会停止；Ctrl+C 只停止查看日志。"
tail -n 120 -F "$RECOVERY_LOG"

