#!/usr/bin/env bash
# Start or supervise V11 without inheriting the interactive terminal's
# errexit/nounset/pipefail options.  The worker always records its real pipeline
# exit code and then exits cleanly, so an SSH/network interruption cannot make
# the terminal appear to "flash-exit" or kill the GPU job.

set +e
set +u
set +o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$REPO_ROOT/run_v11_cyclic_native_1700_and_monomer.sh"
LOG="${V11_LAUNCHER_LOG:-/root/autodl-tmp/v11_launcher.log}"
PID_FILE="${V11_LAUNCHER_PID:-/root/autodl-tmp/v11_launcher.pid}"
EXIT_FILE="${V11_LAUNCHER_EXIT:-/root/autodl-tmp/v11_launcher.exitcode}"
STATUS_FILE="${V11_LAUNCHER_STATUS:-/root/autodl-tmp/v11_launcher.status}"
LOCK_FILE="${V11_LAUNCHER_LOCK:-/root/autodl-tmp/v11_launcher.lock}"

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
    echo "V11 supervisor: repository is unavailable: $REPO_ROOT"
    echo 90 > "$EXIT_FILE"
    write_status "TECHNICAL_STOP" "repository_unavailable"
    return 0
  }
  exec 9>"$LOCK_FILE"
  if command -v flock >/dev/null 2>&1; then
    flock -n 9 || {
      echo "V11 supervisor: another worker owns $LOCK_FILE; no duplicate was started"
      write_status "ALREADY_RUNNING" "lock_is_held"
      return 0
    }
  fi
  write_status "RUNNING" "pipeline_started"
  echo "===== V11 SUPERVISOR START $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  echo "Repository: $REPO_ROOT"
  echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  export V11_INSTALL_OPTIONAL_PANDAS="${V11_INSTALL_OPTIONAL_PANDAS:-0}"
  bash "$RUNNER"
  local pipeline_code=$?
  echo "$pipeline_code" > "$EXIT_FILE"
  if [[ "$pipeline_code" -eq 0 ]]; then
    write_status "COMPLETED" "all_prestructure_gates_passed"
    echo "===== V11 SUPERVISOR COMPLETE: ALL GATES PASSED ====="
  else
    write_status "STOPPED_WITH_PRESERVED_EVIDENCE" "pipeline_exit_${pipeline_code}"
    echo "===== V11 SUPERVISOR STOPPED SAFELY (pipeline exit $pipeline_code) ====="
    echo "No completed evidence was deleted. Review the final 160 log lines before resuming."
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
    echo "V11 已在后台运行，PID=$prior_pid；没有重复启动。"
    echo "状态：cat $STATUS_FILE"
    echo "日志：tail -n 160 -F $LOG"
    exit 0
  fi
fi
if command -v pgrep >/dev/null 2>&1; then
  existing_runner="$(pgrep -f '[r]un_v11_cyclic_native_1700_and_monomer.sh' | head -n 1)"
  if [[ -n "$existing_runner" ]]; then
    echo "检测到已有 V11 主流程 PID=$existing_runner；没有重复启动。"
    echo "日志：tail -n 160 -F $LOG"
    exit 0
  fi
fi

available_kb="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -z "$available_kb" || ! "$available_kb" =~ ^[0-9]+$ ]]; then
  echo "无法可靠读取剩余磁盘空间，未启动；请先运行：df -h $REPO_ROOT"
  write_status "TECHNICAL_STOP" "disk_space_unknown"
  exit 0
fi
if (( available_kb < 20 * 1024 * 1024 )); then
  echo "剩余磁盘不足 20 GiB，未启动，防止中途写满磁盘。"
  df -h "$REPO_ROOT"
  write_status "TECHNICAL_STOP" "less_than_20GiB_free"
  exit 0
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$PID_FILE")"
rm -f "$EXIT_FILE"
nohup bash "$0" --worker >> "$LOG" 2>&1 < /dev/null &
worker_pid=$!
echo "$worker_pid" > "$PID_FILE"
sleep 1
if kill -0 "$worker_pid" 2>/dev/null; then
  echo "V11 已安全转入后台，PID=$worker_pid"
  echo "不会跟随 SSH/网页终端断开而停止，也不会让当前终端闪退。"
  echo "状态：cat $STATUS_FILE"
  echo "日志：tail -n 160 -F $LOG"
  echo "退出码：cat $EXIT_FILE（任务结束后生成；0 表示全部门禁通过）"
else
  echo "后台监督进程未保持运行；请查看日志，但当前终端仍会正常保留。"
  tail -n 80 "$LOG" 2>/dev/null
fi
exit 0
