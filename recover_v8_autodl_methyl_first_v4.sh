#!/usr/bin/env bash
set -Eeuo pipefail

WORK="${V8_AUTODL_WORK:-/root/autodl-tmp}"
REPO="${V8_AUTODL_REPO:-$WORK/proteinmpnn-clean-v28-v8-autodl}"
REMOTE="${V8_V4_REMOTE:-https://github.com/DCarchimonde/proteinmpnn-clean-v28.git}"
BRANCH="${V8_V4_BRANCH:-agent/v8-final-methyl-first-v4}"
BASE_PAYLOAD_COMMIT="1f085e00f71049683eb535201ea576b5bbe6b2cc"
PAYLOAD_COMMIT="31c624a4c3503d23ae08f762aafde6559be8d21e"

FILES=(
  paper_clean_v28/serine_qc_retrain/21_methyl_first_joint_recovery_v4.py
  paper_clean_v28/serine_qc_retrain/22_audit_and_package_methyl_first_v4.py
  paper_clean_v28/serine_qc_retrain/README.md
  paper_clean_v28/serine_qc_retrain/V8_METHYL_FIRST_RECOVERY_V4.md
  run_v8_autodl_recovery_v4.sh
  tests/test_source_scoped_hybrid_v8.py
)

UNCHANGED_RUNTIME=(
  paper_clean_v28/serine_qc_retrain/14_directed_recovery_search_v8.py
  paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py
  paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py
  paper_clean_v28/serine_qc_retrain/20_full_frontier_recovery_v3.py
  paper_clean_v28/clean_v28_common.py
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "[1/7] 核验现有 AutoDL V3 失败证据；不重跑、不删除任何结果"
[[ -d "$REPO" ]] || die "找不到现有 AutoDL 目录：$REPO"
for command_name in git tar sha256sum python install mktemp; do
  command -v "$command_name" >/dev/null || die "缺少命令：$command_name"
done
for required in \
  "$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/directed_search_cyclic_base_v2/cyclic_base_recovery_manifest.json" \
  "$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/directed_search_cyclic_base_v3_full_frontier/cyclic_base_recovery_manifest.json"; do
  [[ -s "$required" ]] || die "V2/V3 固定证据缺失；目标源码未修改：$required"
done

python - "$REPO" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
v8 = repo / "paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8"
v2_path = v8 / "directed_search_cyclic_base_v2/cyclic_base_recovery_manifest.json"
v3_path = v8 / "directed_search_cyclic_base_v3_full_frontier/cyclic_base_recovery_manifest.json"

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

v2 = json.loads(v2_path.read_text(encoding="utf-8-sig"))
v3 = json.loads(v3_path.read_text(encoding="utf-8-sig"))
only_false = {"at_least_one_real_3zgc_candidate_is_released"}
if not (
    v2.get("quality_gate") == "FAIL"
    and {name for name, passed in v2["quality_checks"].items() if not passed}
    == only_false
    and int(v2.get("released_candidates", -1)) == 0
    and v3.get("quality_gate") == "FAIL"
    and {name for name, passed in v3["quality_checks"].items() if not passed}
    == only_false
    and v3.get("release_status")
    == "BLOCKED_FIXED_V3_FULL_FRONTIER_BUDGET_DID_NOT_RECOVER_3ZGC"
    and int(v3.get("released_candidates", -1)) == 0
    and int(v3.get("conditional_rounds_completed", -1)) == 6
    and v3.get("missing_targets_after_search") == ["3ZGC"]
    and abs(float(v3.get("cyclic_base_floor_1pct")) + 2.094945192337036) <= 2e-6
    and v3["config"].get("threshold") == 0.6
    and v3["config"].get("prior_v2_manifest_sha256") == sha256_file(v2_path)
):
    raise SystemExit("ERROR: 不是已核验的 V2/V3 唯一双门零命中状态")
for manifest, root in ((v2, v2_path.parent), (v3, v3_path.parent)):
    for item in leaves(manifest.get("artifacts")):
        path = pathlib.Path(str(item["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(f"ERROR: 证据文件越界：{path}") from exc
        if not path.is_file() or sha256_file(path) != str(item["sha256"]):
            raise SystemExit(f"ERROR: 证据文件缺失或哈希失配：{path}")
print("V2/V3 zero-release evidence: HASH-PINNED PASS")
PY

echo "[2/7] 在临时 Git 目录拉取固定 V4 提交；目标目录无需 .git"
STAGE="$(mktemp -d /tmp/v8-v4-bootstrap.XXXXXX)"
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
    die "GitHub 连续 30 次拉取失败；目标目录未修改"
  fi
  sleep 3
done
git -C "$STAGE" merge-base --is-ancestor "$PAYLOAD_COMMIT" FETCH_HEAD || die \
  "固定 V4 提交 $PAYLOAD_COMMIT 不在远端分支；目标目录未修改"

echo "[3/7] 核验 V3 已安装运行时仍与固定基础提交一致"
for relative in "${UNCHANGED_RUNTIME[@]}"; do
  [[ -s "$REPO/$relative" ]] || die "现有 V3 运行时缺失：$relative"
  expected="$(git -C "$STAGE" rev-parse "$BASE_PAYLOAD_COMMIT:$relative")"
  actual="$(git -C "$STAGE" hash-object -- "$REPO/$relative")"
  [[ "$actual" == "$expected" ]] || die \
    "现有 V3 运行时被修改，拒绝覆盖：$relative"
done

echo "[4/7] 提取并逐文件核验完整 6 文件 V4 载荷"
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

echo "[5/7] 备份现有同名源码并安装 V4；旧 V2/V3 输出保持只读"
BACKUP="$WORK/v8_v4_source_backup_$(date +%Y%m%d_%H%M%S)_$$"
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

echo "[6/7] 写入后复核、46 项专项测试、130 项全仓库回归和 Shell 语法"
for relative in "${FILES[@]}"; do
  expected="$(git -C "$STAGE" rev-parse "$PAYLOAD_COMMIT:$relative")"
  actual="$(git -C "$STAGE" hash-object -- "$REPO/$relative")"
  [[ "$actual" == "$expected" ]] || die "写入后对象哈希不符：$relative"
done
cd "$REPO"
python tests/test_source_scoped_hybrid_v8.py
python -m unittest discover -s tests -p 'test_*.py'
bash -n run_v8_autodl_recovery_v4.sh
echo "完整 V4 载荷、对象哈希、46 项专项与全仓库回归：PASS"
echo "可恢复源码备份：$BACKUP"

if [[ "${V8_V4_BOOTSTRAP_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "V4 bootstrap validate-only mode: PASS（未启动 GPU 任务）"
  exit 0
fi

echo "[7/7] 执行 V4 全量预检并自动启动后台固定预算"
bash ./run_v8_autodl_recovery_v4.sh
