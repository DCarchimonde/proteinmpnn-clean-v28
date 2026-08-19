#!/usr/bin/env bash
set -Eeuo pipefail

WORK="${V8_AUTODL_WORK:-/root/autodl-tmp}"
REPO="${V8_AUTODL_REPO:-$WORK/proteinmpnn-clean-v28-v8-autodl}"
PATCH_COMMIT="b86aee7562822637c49799121e2ad04ca24d8061"
SEARCH_REL="paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py"
DOC_REL="paper_clean_v28/serine_qc_retrain/V8_CYCLIC_BASE_RECOVERY_V2.md"
SEARCH="$REPO/$SEARCH_REL"
DOC="$REPO/$DOC_REL"
RUNNER="$REPO/run_v8_autodl_recovery_v2.sh"
LOG="$WORK/v8_cyclic_base_recovery_v2.log"
PIDFILE="$WORK/v8_cyclic_base_recovery_v2.pid"
V2_OUT="$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/directed_search_cyclic_base_v2"
NEW_SEARCH_SHA="8c3321710c5a0dd8fd961aaa4d797932f9e85e6bf6e3f754127b9070920dde6e"
OLD_SEARCH_SHA="05ac59b96f3a9b6fdce87ecabdde614d2ed2d0f03ef37469ee2a569a41ceda62"
NEW_DOC_SHA="f5c99ac2f95aacf962637fb6b72eda70d3d2ed206f938ceb247d089283f96c8d"
RAW_ROOT="https://raw.githubusercontent.com/DCarchimonde/proteinmpnn-clean-v28/$PATCH_COMMIT"
SEARCH_NEW="$WORK/17_cyclic_base_recovery_v2.py.$PATCH_COMMIT.new"
DOC_NEW="$WORK/V8_CYCLIC_BASE_RECOVERY_V2.md.$PATCH_COMMIT.new"

[[ -d "$REPO" ]] || {
  echo "ERROR: 旧 V8 仓库不存在：$REPO" >&2
  exit 1
}
[[ -d "$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/directed_search" ]] || {
  echo "ERROR: 旧六轮 V8 证据不存在；不能跳过已经完成的 268,365 条搜索" >&2
  exit 1
}
for command_name in curl sha256sum python install; do
  command -v "$command_name" >/dev/null || {
    echo "ERROR: 缺少命令：$command_name" >&2
    exit 1
  }
done

download_verified() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  if [[ -s "$destination" ]] \
    && printf '%s  %s\n' "$expected_sha" "$destination" | sha256sum -c - >/dev/null 2>&1; then
    echo "复用已核验的小补丁：$destination"
    return
  fi
  local partial="${destination}.download"
  curl --fail --location --show-error \
    --connect-timeout 30 --max-time 600 \
    --retry 30 --retry-delay 3 --retry-max-time 3600 --retry-all-errors \
    --output "$partial" "$url"
  printf '%s  %s\n' "$expected_sha" "$partial" | sha256sum -c -
  mv -f -- "$partial" "$destination"
}

echo "[1/6] 下载并核验精确小补丁 $PATCH_COMMIT（约 0.1 MB）"
download_verified "$RAW_ROOT/$SEARCH_REL" "$SEARCH_NEW" "$NEW_SEARCH_SHA"
download_verified "$RAW_ROOT/$DOC_REL" "$DOC_NEW" "$NEW_DOC_SHA"

echo "[2/6] 核验现有 V2 支撑程序；不改模型、基线或旧六轮证据"
printf '%s  %s\n' \
  "9b43c4832483e11a65b1e2a8cf49941d88162c506805143b71e5652ea5147228" \
  "$REPO/paper_clean_v28/serine_qc_retrain/18_finalize_and_audit_recovery_v2.py" \
  | sha256sum -c -
printf '%s  %s\n' \
  "8f04000377236bb76208f3237613968e380ad6c7fe5b48bfc9287223a319de04" \
  "$REPO/paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py" \
  | sha256sum -c -
printf '%s  %s\n' \
  "12f9d318f1c99d3d8151bed5b9e655d8bf0592951eab98fe4c0c5f73445d2f36" \
  "$RUNNER" \
  | sha256sum -c -

[[ -s "$SEARCH" ]] || {
  echo "ERROR: 当前 V2 搜索程序不存在：$SEARCH" >&2
  exit 1
}
CURRENT_SEARCH_SHA="$(sha256sum "$SEARCH" | awk '{print $1}')"
case "$CURRENT_SEARCH_SHA" in
  "$OLD_SEARCH_SHA"|"$NEW_SEARCH_SHA") ;;
  *)
    echo "ERROR: 当前 V2 搜索程序既不是已核验旧版，也不是修复版：$CURRENT_SEARCH_SHA" >&2
    exit 1
    ;;
esac

echo "[3/6] 精确识别并停止旧版卡住的 V2 进程"
EXISTING_PID=""
if [[ -f "$PIDFILE" ]]; then
  EXISTING_PID="$(tr -cd '0-9' < "$PIDFILE")"
fi
if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
  if [[ "$CURRENT_SEARCH_SHA" == "$NEW_SEARCH_SHA" ]]; then
    echo "修复版 V2 已经在运行：PID=$EXISTING_PID"
    echo "查看进度：tail -n 120 -F $LOG"
    exit 0
  fi
  python - "$EXISTING_PID" "$REPO" "$SEARCH" <<'PY'
import os
import pathlib
import signal
import sys
import time

root_pid = int(sys.argv[1])
repo = str(pathlib.Path(sys.argv[2]).resolve())
search = str(pathlib.Path(sys.argv[3]).resolve())

def command_line(pid):
    try:
        return pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""

def parent_pid(pid):
    try:
        suffix = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1]
        return int(suffix.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return -1

def process_ids():
    return sorted(
        int(path.name)
        for path in pathlib.Path("/proc").iterdir()
        if path.name.isdigit()
    )

children = {}
for pid in process_ids():
    children.setdefault(parent_pid(pid), []).append(pid)
tree = []
pending = [root_pid]
while pending:
    pid = pending.pop()
    if pid in tree:
        continue
    tree.append(pid)
    pending.extend(children.get(pid, []))
combined = "\n".join(command_line(pid) for pid in tree)
if search not in combined or repo not in combined:
    raise SystemExit(
        f"ERROR: PID {root_pid} 不是预期 V2 任务；为避免误杀，已停止修复"
    )
print(f"Stopping verified old V2 process tree: {tree}", flush=True)
for pid in reversed(tree):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
deadline = time.monotonic() + 20.0
while time.monotonic() < deadline:
    alive = [pid for pid in tree if pathlib.Path(f"/proc/{pid}").exists()]
    if not alive:
        break
    time.sleep(0.25)
else:
    for pid in reversed(alive):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1.0)
    alive = [pid for pid in tree if pathlib.Path(f"/proc/{pid}").exists()]
    if alive:
        raise SystemExit(f"ERROR: 旧 V2 进程仍未退出：{alive}")
print("Verified old V2 process stopped", flush=True)
PY
  rm -f -- "$PIDFILE"
else
  echo "没有存活的旧 PID；继续安装修复版。"
  rm -f -- "$PIDFILE"
fi

echo "[4/6] 处理旧实现的轮内状态并安装修复版"
if [[ "$CURRENT_SEARCH_SHA" == "$OLD_SEARCH_SHA" ]] \
  && { [[ -f "$V2_OUT/v2_resume_state.json" ]] \
    || [[ -f "$V2_OUT/cyclic_base_recovery_manifest.json" ]]; }; then
  BACKUP="${V2_OUT}.pre_stall_hotfix_$(date +%Y%m%dT%H%M%S)"
  mv -- "$V2_OUT" "$BACKUP"
  echo "旧实现的配置绑定状态已完整保留到：$BACKUP"
fi
install -m 0644 "$SEARCH_NEW" "$SEARCH"
install -m 0644 "$DOC_NEW" "$DOC"
printf '%s  %s\n' "$NEW_SEARCH_SHA" "$SEARCH" | sha256sum -c -
printf '%s  %s\n' "$NEW_DOC_SHA" "$DOC" | sha256sum -c -

echo "[5/6] 编译检查并验证卡点修复契约"
python -m py_compile "$SEARCH"
python - "$SEARCH" <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("v8_v2_stall_hotfix_check", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = [{"sequence": value} for value in (
    "AAAAAAA", "CCCCCCC", "DDDDDDD", "EEEEEEE", "FFFFFFF", "GGGGGGG",
)]
observed = module.deterministic_diversity_fill(rows[1:], [rows[0]], 4)
if len(observed) != 4 or len({row["sequence"] for row in observed}) != 4:
    raise SystemExit("ERROR: V2 增量多样性选择自检失败")
if module.V2_INFLIGHT_PROTOCOL != "hash_pinned_v2_round_inflight_resume_v1":
    raise SystemExit("ERROR: V2 轮内哈希续跑契约缺失")
print("V2 stall hotfix self-check: PASS")
PY

echo "[6/6] 重新启动完整 V2 后台流程"
exec "$RUNNER"
