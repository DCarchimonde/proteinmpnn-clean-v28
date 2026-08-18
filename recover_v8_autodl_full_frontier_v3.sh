#!/usr/bin/env bash
set -Eeuo pipefail

WORK="${V8_AUTODL_WORK:-/root/autodl-tmp}"
REPO="${V8_AUTODL_REPO:-$WORK/proteinmpnn-clean-v28-v8-autodl}"
REMOTE="https://github.com/DCarchimonde/proteinmpnn-clean-v28.git"
BRANCH="fix/serine-provenance-retrain-2026"
BASE_COMMIT="0e916c311956108269b57960fc2ca18d388a3713"
PAYLOAD_COMMIT="1f085e00f71049683eb535201ea576b5bbe6b2cc"

# This is the complete tracked-file delta from the AutoDL source bundle's
# pinned BASE_COMMIT to the reviewed V3 PAYLOAD_COMMIT.  Keeping the complete
# delta here prevents a new test file from being paired with stale support
# modules or launchers in a destination that intentionally has no .git tree.
FILES=(
  paper_clean_v28/serine_qc_retrain/16_v8_autodl_resume_bundle.py
  paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py
  paper_clean_v28/serine_qc_retrain/18_finalize_and_audit_recovery_v2.py
  paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py
  paper_clean_v28/serine_qc_retrain/20_full_frontier_recovery_v3.py
  paper_clean_v28/serine_qc_retrain/README.md
  paper_clean_v28/serine_qc_retrain/V8_CYCLIC_BASE_RECOVERY_V2.md
  paper_clean_v28/serine_qc_retrain/V8_FULL_FRONTIER_RECOVERY_V3.md
  recover_v8_autodl_cyclic_v2.sh
  recover_v8_autodl_legacy_bundle.sh
  run_v8_autodl_recovery_v2.sh
  run_v8_autodl_recovery_v3.sh
  run_v8_autodl_resume.sh
  tests/test_source_scoped_hybrid_v8.py
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "[1/6] 核验现有 AutoDL 目录和不可变搜索证据"
[[ -d "$REPO" ]] || die "找不到现有 AutoDL 目录：$REPO"
for command_name in git tar sha256sum python install mktemp; do
  command -v "$command_name" >/dev/null || die "缺少命令：$command_name"
done
if [[ -s "$REPO/.source_commit" ]]; then
  grep -qx "$BASE_COMMIT" "$REPO/.source_commit" || die \
    "现有目录的基础源码不是已核验版本 $BASE_COMMIT"
fi
for required in \
  "$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/directed_search/directed_search_manifest.json" \
  "$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/directed_search_cyclic_base_v2/cyclic_base_recovery_manifest.json"; do
  [[ -s "$required" ]] || die "旧六轮或 V2 证据缺失；未修改源码：$required"
done

echo "[2/6] 在临时 Git 目录拉取固定提交；不要求目标目录含 .git"
STAGE="$(mktemp -d /tmp/v8-v3-bootstrap.XXXXXX)"
git -C "$STAGE" init -q
git -C "$STAGE" remote add origin "$REMOTE"
for attempt in $(seq 1 30); do
  if git -C "$STAGE" \
      -c http.lowSpeedLimit=1 \
      -c http.lowSpeedTime=30 \
      fetch --no-tags origin "$BRANCH"; then
    break
  fi
  if [[ "$attempt" -eq 30 ]]; then
    die "GitHub 连续 30 次拉取失败；目标目录未被修改"
  fi
  sleep 3
done
git -C "$STAGE" merge-base --is-ancestor "$PAYLOAD_COMMIT" FETCH_HEAD || die \
  "固定提交 $PAYLOAD_COMMIT 不在远端分支中；目标目录未被修改"

echo "[3/6] 提取并逐文件核验完整 14 文件差集"
PAYLOAD="$STAGE/payload"
mkdir -p "$PAYLOAD"
git -C "$STAGE" archive "$PAYLOAD_COMMIT" -- "${FILES[@]}" |
  tar -x -C "$PAYLOAD"
for relative in "${FILES[@]}"; do
  [[ -s "$PAYLOAD/$relative" ]] || die "固定提交缺少文件：$relative"
  expected="$(git -C "$STAGE" rev-parse "$PAYLOAD_COMMIT:$relative")"
  actual="$(git -C "$STAGE" hash-object -- "$PAYLOAD/$relative")"
  [[ "$actual" == "$expected" ]] || die "下载对象哈希不符：$relative"
done

echo "[4/6] 备份现有源码并安装已核验完整差集"
BACKUP="$WORK/v8_v3_source_backup_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$BACKUP"
existing=()
for relative in "${FILES[@]}"; do
  if [[ -e "$REPO/$relative" ]]; then
    existing+=("$relative")
  fi
done
if [[ "${#existing[@]}" -gt 0 ]]; then
  tar -C "$REPO" -cf - "${existing[@]}" | tar -x -C "$BACKUP"
fi
for relative in "${FILES[@]}"; do
  target="$REPO/$relative"
  mkdir -p "$(dirname "$target")"
  mode=0644
  if [[ "$relative" == *.sh ]]; then
    mode=0755
  fi
  install -m "$mode" "$PAYLOAD/$relative" "$target"
done

echo "[5/6] 写入后复核、40 项回归测试和 Shell 语法检查"
for relative in "${FILES[@]}"; do
  expected="$(git -C "$STAGE" rev-parse "$PAYLOAD_COMMIT:$relative")"
  actual="$(git -C "$STAGE" hash-object -- "$REPO/$relative")"
  [[ "$actual" == "$expected" ]] || die "写入后对象哈希不符：$relative"
done
cd "$REPO"
python tests/test_source_scoped_hybrid_v8.py
for shell_program in \
  recover_v8_autodl_cyclic_v2.sh \
  recover_v8_autodl_legacy_bundle.sh \
  run_v8_autodl_recovery_v2.sh \
  run_v8_autodl_recovery_v3.sh \
  run_v8_autodl_resume.sh; do
  bash -n "$shell_program"
done
echo "完整源码差集、对象哈希和 40 项回归测试：PASS"
echo "可恢复源码备份：$BACKUP"

if [[ "${V8_V3_BOOTSTRAP_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "V3 bootstrap validate-only mode: PASS（未启动 GPU 任务）"
  exit 0
fi

echo "[6/6] 执行 V3 全量预检并自动启动完整后台流水线"
bash ./run_v8_autodl_recovery_v3.sh
