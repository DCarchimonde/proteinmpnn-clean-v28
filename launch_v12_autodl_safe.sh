#!/usr/bin/env bash
# Detached supervisor for V12.  A scientific/technical stop is recorded in
# files and never propagates errexit into the user's interactive terminal.

set +e
set +u
set +o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$REPO_ROOT/run_v12_methyl_only_1700.sh"
LOG="${V12_LAUNCHER_LOG:-/root/autodl-tmp/v12_launcher.log}"
PID_FILE="${V12_LAUNCHER_PID:-/root/autodl-tmp/v12_launcher.pid}"
EXIT_FILE="${V12_LAUNCHER_EXIT:-/root/autodl-tmp/v12_launcher.exitcode}"
STATUS_FILE="${V12_LAUNCHER_STATUS:-/root/autodl-tmp/v12_launcher.status}"
LOCK_FILE="${V12_LAUNCHER_LOCK:-/root/autodl-tmp/v12_launcher.lock}"

write_status() {
  local state="$1"
  local detail="$2"
  local temporary="${STATUS_FILE}.tmp.$$"
  {
    echo "state=$state"
    echo "detail=$detail"
    date -u '+updated_utc=%Y-%m-%dT%H:%M:%SZ'
    echo "repo=$REPO_ROOT"
    echo "log=$LOG"
  } > "$temporary"
  mv -f "$temporary" "$STATUS_FILE"
}

worker() {
  cd "$REPO_ROOT" || {
    echo 90 > "$EXIT_FILE"
    write_status "TECHNICAL_STOP" "repository_unavailable"
    return 0
  }
  exec 9>"$LOCK_FILE"
  if command -v flock >/dev/null 2>&1; then
    flock -n 9 || {
      echo "V12 supervisor: another worker owns the lock; no duplicate started."
      write_status "ALREADY_RUNNING" "lock_is_held"
      return 0
    }
  fi
  write_status "RUNNING" "v12_pipeline_started"
  echo "===== V12 SUPERVISOR START $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  echo "Repository: $REPO_ROOT"
  echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  bash "$RUNNER"
  local pipeline_code=$?
  echo "$pipeline_code" > "$EXIT_FILE"
  if [[ "$pipeline_code" -eq 0 ]]; then
    write_status "COMPLETED" "exact_17x100_methyl_only_handoff_passed"
    echo "===== V12 SUPERVISOR COMPLETE: ALL GATES PASSED ====="
  else
    write_status "STOPPED_WITH_PRESERVED_EVIDENCE" "pipeline_exit_${pipeline_code}"
    echo "===== V12 SUPERVISOR STOPPED SAFELY (pipeline exit $pipeline_code) ====="
    echo "No completed evidence was deleted; the interactive terminal remains usable."
  fi
  echo "Exit-code record: $EXIT_FILE"
  echo "Status record: $STATUS_FILE"
  return 0
}

if [[ "${1:-}" == "--worker" ]]; then
  worker
  exit 0
fi

if [[ ! -s "$RUNNER" ]]; then
  echo "无法启动：缺少 $RUNNER"
  write_status "TECHNICAL_STOP" "runner_missing"
  exit 0
fi

if [[ -s "$PID_FILE" ]]; then
  prior_pid="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$prior_pid" ]] && kill -0 "$prior_pid" 2>/dev/null; then
    echo "V12 已在后台运行，PID=$prior_pid；没有重复启动。"
    echo "状态：cat $STATUS_FILE"
    echo "日志：tail -n 160 -F $LOG"
    exit 0
  fi
fi

if command -v pgrep >/dev/null 2>&1; then
  conflicting="$(pgrep -af '[r]un_v1[12]_.*1700.*[.]sh' | head -n 5)"
  if [[ -n "$conflicting" ]]; then
    echo "检测到仍在运行的 V11/V12 主流程；为避免 GPU 冲突，没有重复启动："
    echo "$conflicting"
    exit 0
  fi
fi

available_kb="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -z "$available_kb" || ! "$available_kb" =~ ^[0-9]+$ ]]; then
  echo "无法可靠读取磁盘余量，未启动。"
  write_status "TECHNICAL_STOP" "disk_space_unknown"
  exit 0
fi
if (( available_kb < 10 * 1024 * 1024 )); then
  echo "剩余磁盘不足 10 GiB，未启动，防止搜索证据写满磁盘。"
  df -h "$REPO_ROOT"
  write_status "TECHNICAL_STOP" "less_than_10GiB_free"
  exit 0
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$PID_FILE")"
printf 'PENDING\n' > "$EXIT_FILE"
nohup bash "$0" --worker >> "$LOG" 2>&1 < /dev/null &
worker_pid=$!
echo "$worker_pid" > "$PID_FILE"
sleep 1
if kill -0 "$worker_pid" 2>/dev/null; then
  echo "V12 已安全转入后台，PID=$worker_pid"
  echo "断开网页或 SSH 不会停止任务，脚本失败也不会让当前终端闪退。"
  echo "状态：cat $STATUS_FILE"
  echo "日志：tail -n 160 -F $LOG"
  echo "退出码：cat $EXIT_FILE（结束后生成；0 才是完整 PASS）"
else
  echo "后台监督进程未保持运行；请查看日志，当前终端仍会保留。"
  tail -n 80 "$LOG" 2>/dev/null
fi
exit 0
