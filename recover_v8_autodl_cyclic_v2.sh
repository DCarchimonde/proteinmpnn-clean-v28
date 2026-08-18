#!/usr/bin/env bash
set -Eeuo pipefail

WORK="/root/autodl-tmp"
REPO="$WORK/proteinmpnn-clean-v28-v8-autodl"
COMMIT="cda853e790f1cf37c0e5cdb2c01bf4dcd11f014b"
ARCHIVE="$WORK/v8_cyclic_base_v2_${COMMIT}.tar.gz"
SOURCE_URL="https://codeload.github.com/DCarchimonde/proteinmpnn-clean-v28/tar.gz/$COMMIT"

[[ -d "$REPO" ]] || {
  echo "ERROR: 旧 V8 仓库不存在：$REPO" >&2
  exit 1
}
[[ -d "$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/directed_search" ]] || {
  echo "ERROR: 旧六轮 V8 证据不存在；不能跳过已经完成的 268,365 条搜索" >&2
  exit 1
}
command -v curl >/dev/null || {
  echo "ERROR: curl 不存在" >&2
  exit 1
}
command -v tar >/dev/null || {
  echo "ERROR: tar 不存在" >&2
  exit 1
}
command -v sha256sum >/dev/null || {
  echo "ERROR: sha256sum 不存在" >&2
  exit 1
}

echo "[1/4] 下载精确 V2 提交 $COMMIT（只下载源码，不下载模型或旧结果）"
if [[ -s "$ARCHIVE" ]] && tar -tzf "$ARCHIVE" >/dev/null 2>&1; then
  echo "复用已下载且 tar 结构有效的源码包：$ARCHIVE"
else
  DOWNLOAD="${ARCHIVE}.download"
  curl --fail --location --show-error \
    --connect-timeout 30 --max-time 600 \
    --retry 30 --retry-delay 3 --retry-max-time 3600 --retry-all-errors \
    --output "$DOWNLOAD" "$SOURCE_URL"
  tar -tzf "$DOWNLOAD" >/dev/null
  mv -f "$DOWNLOAD" "$ARCHIVE"
fi

TEMP_DIR="$(mktemp -d "$WORK/v8-cyclic-v2-source.XXXXXX")"
cleanup() {
  if [[ -n "${TEMP_DIR:-}" && "$TEMP_DIR" == "$WORK"/v8-cyclic-v2-source.* ]]; then
    rm -rf -- "$TEMP_DIR"
  fi
}
trap cleanup EXIT
tar -xzf "$ARCHIVE" -C "$TEMP_DIR"
SOURCE_ROOT="$TEMP_DIR/proteinmpnn-clean-v28-$COMMIT"
[[ -d "$SOURCE_ROOT" ]] || {
  echo "ERROR: GitHub 源码包目录结构不符" >&2
  exit 1
}

echo "[2/4] 核验五个 V2 文件的精确 SHA256"
(
  cd "$SOURCE_ROOT"
  sha256sum -c <<'SHA256S'
05ac59b96f3a9b6fdce87ecabdde614d2ed2d0f03ef37469ee2a569a41ceda62  paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py
9b43c4832483e11a65b1e2a8cf49941d88162c506805143b71e5652ea5147228  paper_clean_v28/serine_qc_retrain/18_finalize_and_audit_recovery_v2.py
8f04000377236bb76208f3237613968e380ad6c7fe5b48bfc9287223a319de04  paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py
cbd6aa6d99cccc0e97f2d8592a80fe3c315807c150326a4c64c5c98412f49536  paper_clean_v28/serine_qc_retrain/V8_CYCLIC_BASE_RECOVERY_V2.md
12f9d318f1c99d3d8151bed5b9e655d8bf0592951eab98fe4c0c5f73445d2f36  run_v8_autodl_recovery_v2.sh
SHA256S
)

echo "[3/4] 只安装已核验的 V2 程序；不改模型、基线、六轮 ledger/checkpoint"
install -m 0644 \
  "$SOURCE_ROOT/paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py" \
  "$REPO/paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py"
install -m 0644 \
  "$SOURCE_ROOT/paper_clean_v28/serine_qc_retrain/18_finalize_and_audit_recovery_v2.py" \
  "$REPO/paper_clean_v28/serine_qc_retrain/18_finalize_and_audit_recovery_v2.py"
install -m 0644 \
  "$SOURCE_ROOT/paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py" \
  "$REPO/paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py"
install -m 0644 \
  "$SOURCE_ROOT/paper_clean_v28/serine_qc_retrain/V8_CYCLIC_BASE_RECOVERY_V2.md" \
  "$REPO/paper_clean_v28/serine_qc_retrain/V8_CYCLIC_BASE_RECOVERY_V2.md"
install -m 0755 \
  "$SOURCE_ROOT/run_v8_autodl_recovery_v2.sh" \
  "$REPO/run_v8_autodl_recovery_v2.sh"

echo "[4/4] 启动完整 V2 后台流程"
exec "$REPO/run_v8_autodl_recovery_v2.sh"
