#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${V8_AUTODL_WORK:-/root/autodl-tmp}"
LOG="${V8_V4_LOG:-$WORK/v8_methyl_first_recovery_v4.log}"
PIDFILE="${V8_V4_PIDFILE:-$WORK/v8_methyl_first_recovery_v4.pid}"
PYTHON_BIN="${V8_PYTHON:-python}"
V8_ROOT="$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8"
SEARCH_PROGRAM="$REPO/paper_clean_v28/serine_qc_retrain/21_methyl_first_joint_recovery_v4.py"
AUDIT_PROGRAM="$REPO/paper_clean_v28/serine_qc_retrain/22_audit_and_package_methyl_first_v4.py"
V3_PROGRAM="$REPO/paper_clean_v28/serine_qc_retrain/20_full_frontier_recovery_v3.py"
V2_PROGRAM="$REPO/paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py"
PRIOR_V2="$V8_ROOT/directed_search_cyclic_base_v2"
PRIOR_V3="$V8_ROOT/directed_search_cyclic_base_v3_full_frontier"
V4_SEARCH="$V8_ROOT/directed_search_methyl_first_v4"
V4_AUDIT="$V8_ROOT/independent_audit_methyl_first_v4"
V4_BUNDLE="$V8_ROOT/v8_methyl_first_v4_review_bundle.zip"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

mkdir -p "$WORK"

echo "[1/4] V4 离线预检：固定源码、V3 零释放证据、甲基优先规则和 GPU"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "ERROR: Python 不存在：$PYTHON_BIN" >&2
  exit 1
}
command -v nohup >/dev/null || {
  echo "ERROR: nohup 不存在" >&2
  exit 1
}
for required in "$SEARCH_PROGRAM" "$AUDIT_PROGRAM" "$V2_PROGRAM" "$V3_PROGRAM"; do
  [[ -s "$required" ]] || {
    echo "ERROR: 缺少 V4 程序：$required" >&2
    exit 1
  }
done

"$PYTHON_BIN" - "$REPO" "$PRIOR_V2" "$PRIOR_V3" <<'PY'
import hashlib
import json
import pathlib
import py_compile
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
prior_v2 = pathlib.Path(sys.argv[2]).resolve()
prior_v3 = pathlib.Path(sys.argv[3]).resolve()
programs = [
    repo / "paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py",
    repo / "paper_clean_v28/serine_qc_retrain/20_full_frontier_recovery_v3.py",
    repo / "paper_clean_v28/serine_qc_retrain/21_methyl_first_joint_recovery_v4.py",
    repo / "paper_clean_v28/serine_qc_retrain/22_audit_and_package_methyl_first_v4.py",
]
for program in programs:
    py_compile.compile(str(program), doraise=True)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def leaves(value):
    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value):
            yield value
        else:
            for child in value.values():
                yield from leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from leaves(child)

v2_path = prior_v2 / "cyclic_base_recovery_manifest.json"
v3_path = prior_v3 / "cyclic_base_recovery_manifest.json"
for required in (v2_path, v3_path):
    if not required.is_file():
        raise SystemExit(f"ERROR: 缺少不可变失败清单：{required}")
v2 = json.loads(v2_path.read_text(encoding="utf-8-sig"))
v3 = json.loads(v3_path.read_text(encoding="utf-8-sig"))
v2_false = {name for name, passed in v2["quality_checks"].items() if not passed}
v3_false = {name for name, passed in v3["quality_checks"].items() if not passed}
only_false = {"at_least_one_real_3zgc_candidate_is_released"}
if not (
    v2.get("protocol") == "cyclic_start_base_pareto_recovery_v8_v2"
    and v2.get("quality_gate") == "FAIL"
    and v2_false == only_false
    and int(v2.get("released_candidates", -1)) == 0
    and int(v2.get("conditional_rounds_completed", -1)) == 6
    and v3.get("protocol") == "full_legacy_frontier_cyclic_base_recovery_v8_v3"
    and v3.get("quality_gate") == "FAIL"
    and v3_false == only_false
    and v3.get("release_status")
    == "BLOCKED_FIXED_V3_FULL_FRONTIER_BUDGET_DID_NOT_RECOVER_3ZGC"
    and int(v3.get("released_candidates", -1)) == 0
    and int(v3.get("conditional_rounds_completed", -1)) == 6
    and int(v3.get("legacy_full_frontier_rows", -1)) == 268365
    and int(v3.get("prior_v2_methyl_screen_rows_reused", -1)) == 159329
    and int(v3.get("legacy_non_strict_bridge_rows_exactly_scored", -1)) == 16384
    and v3.get("missing_targets_after_search") == ["3ZGC"]
    and v3["config"].get("threshold") == 0.6
    and v3["config"].get("base_percentile") == 0.01
    and abs(float(v3.get("cyclic_base_floor_1pct")) + 2.094945192337036) <= 2e-6
    and v3["config"].get("prior_v2_manifest_sha256") == sha256_file(v2_path)
):
    raise SystemExit("ERROR: 当前目录不是完整、唯一门失败的 V2/V3 固定状态")
for label, manifest, root in (("V2", v2, prior_v2), ("V3", v3, prior_v3)):
    artifacts = list(leaves(manifest.get("artifacts")))
    if not artifacts:
        raise SystemExit(f"ERROR: {label} 清单没有文件证据")
    for item in artifacts:
        path = pathlib.Path(str(item["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(f"ERROR: {label} 文件越界：{path}") from exc
        if not path.is_file() or sha256_file(path) != str(item["sha256"]):
            raise SystemExit(f"ERROR: {label} 文件缺失或哈希失配：{path}")
print(
    "V4 inputs: prior_seen=501,537; prior_exact=69,413; "
    "prior_joint_hits=0; only_missing=3ZGC; methyl_gate=>0.6; "
    "base_floor=-2.09494519; non_methyl_advisor_rows=FORBIDDEN"
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
if required not in set(torch.cuda.get_arch_list()):
    raise SystemExit(f"ERROR: PyTorch 不原生支持 {required}")
if "5090" not in name:
    raise SystemExit(f"ERROR: 本协议按 RTX 5090 冻结，当前 {name}")
print(json.dumps({
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": name,
    "native_arch": required,
}, ensure_ascii=False))
PY

echo "[2/4] 检查是否已有同一 V4 任务"
if [[ -f "$PIDFILE" ]]; then
  existing_pid="$(tr -cd '0-9' < "$PIDFILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    if "$PYTHON_BIN" - "$existing_pid" "$REPO" "$SEARCH_PROGRAM" <<'PY'
import pathlib
import sys
pid = int(sys.argv[1])
repo = str(pathlib.Path(sys.argv[2]).resolve())
program = str(pathlib.Path(sys.argv[3]).resolve())
try:
    command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
except (FileNotFoundError, PermissionError, ProcessLookupError):
    raise SystemExit(1)
raise SystemExit(0 if repo in command and program in command else 1)
PY
    then
      echo "V4 已在后台运行：PID=$existing_pid"
      echo "查看进度：tail -n 120 -F $LOG"
      exit 0
    fi
    rm -f -- "$PIDFILE"
  fi
fi

echo "[3/4] 后台启动最终固定预算：只接受甲基硬门命中，再算 exact base"
nohup bash -lc '
  set -Eeuo pipefail
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
  export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
  cd "$1"
  printf "\n===== V8 V4 methyl-first launch %s =====\n" "$(date --iso-8601=seconds)"
  "$2" "$3" --device cuda --resume --prior-v2-dir "$5" --prior-v3-dir "$6" --out-dir "$7"
  "$2" "$4" --device cuda --overwrite --search-dir "$7" --audit-dir "$8" --bundle "$9"
  "$2" - "$7/methyl_first_v4_manifest.json" <<'PY'
import json
import pathlib
import sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
if manifest.get("execution_audit_gate") != "PASS":
    raise SystemExit("V4 execution audit did not pass")
if manifest.get("scientific_joint_gate") == "PASS":
    print("===== V8 V4 SCIENTIFIC JOINT GATE PASSED =====")
else:
    print("===== V8 V4 ZERO JOINT HITS; METHYLATED BASE-NEAR-MISS REVIEW ONLY =====")
PY
  echo "===== ALL V8 V4 METHYL-FIRST AUTOMATED AUDITS PASSED ====="
' v8-v4 "$REPO" "$PYTHON_BIN" "$SEARCH_PROGRAM" "$AUDIT_PROGRAM" \
  "$PRIOR_V2" "$PRIOR_V3" "$V4_SEARCH" "$V4_AUDIT" "$V4_BUNDLE" \
  >>"$LOG" 2>&1 &
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
  if grep -q 'ALL V8 V4 METHYL-FIRST AUTOMATED AUDITS PASSED' "$LOG"; then
    echo "===== V8 V4 已在 20 秒内完成 ====="
    tail -n 120 "$LOG" || true
    exit 0
  fi
  echo "ERROR: V4 启动后退出，末尾日志如下：" >&2
  tail -n 240 "$LOG" >&2 || true
  exit 1
fi

echo "===== V8 V4 BACKGROUND JOB IS HEALTHY: PID=$job_pid ====="
echo "预计 RTX 5090 约 25–35 分钟；SSH 断开不会停止。"
echo "查看进度：tail -n 120 -F $LOG"
echo "确认最终状态："
echo "grep -E 'ALL V8 V4|SCIENTIFIC JOINT|ZERO JOINT|Execution audit|Independent audit|ERROR|Traceback' $LOG | tail -n 80"
echo "最终审计包：$V4_BUNDLE"
echo "若双门仍为 0，尚哥审阅表：$V4_SEARCH/methylated_base_near_miss_for_shangge_review.csv"
echo "当前日志末尾："
tail -n 50 "$LOG" || true
