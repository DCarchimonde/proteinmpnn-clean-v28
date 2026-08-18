#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${V8_AUTODL_WORK:-/root/autodl-tmp}"
LOG="${V8_V2_LOG:-$WORK/v8_cyclic_base_recovery_v2.log}"
PIDFILE="${V8_V2_PIDFILE:-$WORK/v8_cyclic_base_recovery_v2.pid}"
PYTHON_BIN="${V8_PYTHON:-python}"
SEARCH="$REPO/paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py"
FINAL="$REPO/paper_clean_v28/serine_qc_retrain/18_finalize_and_audit_recovery_v2.py"
PACKAGE="$REPO/paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py"
LEGACY="$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/directed_search"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

mkdir -p "$WORK"

echo "[1/4] V2 完整预检：GPU、源码、旧六轮证据和不可变输入"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "ERROR: Python 不存在：$PYTHON_BIN" >&2
  exit 1
}
command -v nohup >/dev/null || {
  echo "ERROR: nohup 不存在" >&2
  exit 1
}
for required in "$SEARCH" "$FINAL" "$PACKAGE"; do
  [[ -s "$required" ]] || {
    echo "ERROR: 缺少 V2 程序：$required" >&2
    exit 1
  }
done

"$PYTHON_BIN" - "$REPO" "$LEGACY" <<'PY'
import json
import pathlib
import py_compile
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
legacy = pathlib.Path(sys.argv[2]).resolve()
programs = [
    repo / "paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py",
    repo / "paper_clean_v28/serine_qc_retrain/18_finalize_and_audit_recovery_v2.py",
    repo / "paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py",
]
for program in programs:
    py_compile.compile(str(program), doraise=True)

manifest_path = legacy / "directed_search_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"ERROR: 缺少旧六轮搜索清单：{manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
false_checks = {
    name
    for name, passed in dict(manifest.get("quality_checks") or {}).items()
    if not passed
}
expected_false = {"all_missing_targets_have_a_novel_plausible_strict_candidate"}
if not (
    manifest.get("protocol") == "deterministic_missing_target_directed_recovery_v8"
    and manifest.get("quality_gate") == "FAIL"
    and false_checks == expected_false
    and manifest.get("missing_targets_before_search") == ["3ZGC"]
    and manifest.get("missing_targets_after_search") == ["3ZGC"]
    and int(dict(manifest.get("strict_probability_hit_counts") or {}).get("3ZGC", -1)) > 0
):
    raise SystemExit("ERROR: 旧结果不是可恢复的精确 3ZGC 单门失败状态")

required = [
    repo / "paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/model/frankenstein_v28_source_scoped_hybrid_v8.pt",
    repo / "paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/model/expert_source_composition_manifest.json",
    repo / "paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/representation_audit/cyclic_representation_audit.json",
    repo / "paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/generation_baseline/generation_manifest.json",
    repo / "paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/generation_baseline/unique_candidates.csv",
    repo / "17_complexes_native.jsonl",
    repo / "paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv",
    repo / "paper_clean_v28_outputs/rerun_temperature_0.5_multiseed/methylated_new_candidates.csv",
]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit("ERROR: 运行输入缺失：" + ", ".join(missing))
print(
    "Legacy evidence: "
    f"evaluated={manifest['evaluated_sequence_counts']['3ZGC']:,}; "
    f"strict_hits={manifest['strict_probability_hit_counts']['3ZGC']:,}"
)
PY

"$PYTHON_BIN" - <<'PY'
import json
import sys
import numpy
import torch

if sys.version_info < (3, 10):
    raise SystemExit(f"ERROR: Python 必须 >=3.10，当前 {sys.version.split()[0]}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch 检测不到 CUDA")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
required = f"sm_{capability[0]}{capability[1]}"
arches = set(torch.cuda.get_arch_list())
if required not in arches:
    raise SystemExit(f"ERROR: PyTorch 不原生支持 {required}; available={sorted(arches)}")
if "5090" not in name:
    raise SystemExit(f"ERROR: 本协议要求 RTX 5090，当前 {name}")
print(json.dumps({
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": name,
    "capability": capability,
    "native_arch": required,
}, ensure_ascii=False))
PY

echo "[2/4] 检查是否已有同一 V2 任务"
if [[ -f "$PIDFILE" ]]; then
  existing_pid="$(tr -cd '0-9' < "$PIDFILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "V2 任务已经在后台运行：PID=$existing_pid"
    echo "查看进度：tail -n 120 -F $LOG"
    exit 0
  fi
fi

echo "[3/4] 后台启动：先纠正 2881 个候选的环起点基线审计，必要时自动跑六轮双目标搜索"
nohup bash -lc '
  set -Eeuo pipefail
  export OMP_NUM_THREADS=16
  export MKL_NUM_THREADS=16
  export OPENBLAS_NUM_THREADS=16
  export NUMEXPR_NUM_THREADS=16
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONUNBUFFERED=1
  cd "$1"
  printf "\n===== V8 V2 launch %s =====\n" "$(date --iso-8601=seconds)"
  "$2" "$3" --device cuda --resume
  "$2" "$4" --device cuda --overwrite
  "$2" "$5"
  echo "===== ALL V8 V2 AUTOMATED GATES PASSED ====="
' v8-v2 "$REPO" "$PYTHON_BIN" "$SEARCH" "$FINAL" "$PACKAGE" >>"$LOG" 2>&1 &
job_pid=$!
printf '%s\n' "$job_pid" > "$PIDFILE"

echo "[4/4] 等待 20 秒确认没有启动即退"
if ! "$PYTHON_BIN" - "$job_pid" <<'PY'
import os
import sys
import time

pid = int(sys.argv[1])
for _ in range(20):
    time.sleep(1)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit(1)
PY
then
  if grep -q 'ALL V8 V2 AUTOMATED GATES PASSED' "$LOG"; then
    echo "===== V8 V2 已在 20 秒内全部完成 ====="
    tail -n 80 "$LOG" || true
    exit 0
  else
    echo "ERROR: V2 任务启动后退出，末尾日志如下：" >&2
    tail -n 160 "$LOG" >&2 || true
    exit 1
  fi
fi

echo "===== V8 V2 BACKGROUND JOB IS HEALTHY: PID=$job_pid ====="
echo "这里表示准备和启动完成，不代表科学计算已经结束。"
echo "网页/SSH 断开不会停止任务。查看进度："
echo "tail -n 120 -F $LOG"
echo "确认最终完成："
echo "grep -E 'ALL V8 V2 AUTOMATED GATES PASSED|Quality gate:|ERROR|Traceback' $LOG | tail -n 40"
echo "最终审计包："
echo "$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/v8_cyclic_base_v2_review_bundle.zip"
echo "当前日志末尾："
tail -n 40 "$LOG" || true
