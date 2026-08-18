#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${V8_AUTODL_WORK:-/root/autodl-tmp}"
LOG="${V8_V3_LOG:-$WORK/v8_full_frontier_recovery_v3.log}"
PIDFILE="${V8_V3_PIDFILE:-$WORK/v8_full_frontier_recovery_v3.pid}"
PYTHON_BIN="${V8_PYTHON:-python}"
V8_ROOT="$REPO/paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8"
SEARCH="$REPO/paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py"
FINAL="$REPO/paper_clean_v28/serine_qc_retrain/18_finalize_and_audit_recovery_v2.py"
PACKAGE="$REPO/paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py"
FRONTIER="$REPO/paper_clean_v28/serine_qc_retrain/20_full_frontier_recovery_v3.py"
LEGACY="$V8_ROOT/directed_search"
PRIOR_V2="$V8_ROOT/directed_search_cyclic_base_v2"
V3_SEARCH="$V8_ROOT/directed_search_cyclic_base_v3_full_frontier"
V3_GENERATION="$V8_ROOT/generation_recovered_cyclic_base_v3_full_frontier"
V3_AUDIT="$V8_ROOT/triple_audit_recovered_cyclic_base_v3_full_frontier"
V3_BUNDLE="$V8_ROOT/v8_cyclic_base_v3_full_frontier_review_bundle.zip"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

mkdir -p "$WORK"

echo "[1/4] V3 完整离线预检：源码、GPU、旧六轮证据和已完成 V2 失败"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "ERROR: Python 不存在：$PYTHON_BIN" >&2
  exit 1
}
command -v nohup >/dev/null || {
  echo "ERROR: nohup 不存在" >&2
  exit 1
}
for required in "$SEARCH" "$FINAL" "$PACKAGE" "$FRONTIER"; do
  [[ -s "$required" ]] || {
    echo "ERROR: 缺少 V3 程序：$required" >&2
    exit 1
  }
done

"$PYTHON_BIN" - "$REPO" "$LEGACY" "$PRIOR_V2" <<'PY'
import csv
import gzip
import hashlib
import json
import math
import pathlib
import py_compile
import shutil
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
legacy = pathlib.Path(sys.argv[2]).resolve()
prior = pathlib.Path(sys.argv[3]).resolve()
programs = [
    repo / "paper_clean_v28/serine_qc_retrain/17_cyclic_base_recovery_v2.py",
    repo / "paper_clean_v28/serine_qc_retrain/18_finalize_and_audit_recovery_v2.py",
    repo / "paper_clean_v28/serine_qc_retrain/19_package_v8_recovery_v2.py",
    repo / "paper_clean_v28/serine_qc_retrain/20_full_frontier_recovery_v3.py",
]
for program in programs:
    py_compile.compile(str(program), doraise=True)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def artifact_leaves(value):
    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value):
            yield value
        else:
            for child in value.values():
                yield from artifact_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from artifact_leaves(child)

legacy_manifest_path = legacy / "directed_search_manifest.json"
prior_manifest_path = prior / "cyclic_base_recovery_manifest.json"
for required in (legacy_manifest_path, prior_manifest_path):
    if not required.is_file():
        raise SystemExit(f"ERROR: 缺少不可变清单：{required}")
legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8-sig"))
prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8-sig"))
legacy_false = {
    name
    for name, passed in dict(legacy_manifest.get("quality_checks") or {}).items()
    if not passed
}
prior_false = {
    name
    for name, passed in dict(prior_manifest.get("quality_checks") or {}).items()
    if not passed
}
if not (
    legacy_manifest.get("protocol")
    == "deterministic_missing_target_directed_recovery_v8"
    and legacy_manifest.get("quality_gate") == "FAIL"
    and legacy_false == {"all_missing_targets_have_a_novel_plausible_strict_candidate"}
    and legacy_manifest.get("missing_targets_before_search") == ["3ZGC"]
    and legacy_manifest.get("missing_targets_after_search") == ["3ZGC"]
    and int(dict(legacy_manifest.get("evaluated_sequence_counts") or {}).get("3ZGC", -1))
    == 268365
    and int(dict(legacy_manifest.get("strict_probability_hit_counts") or {}).get("3ZGC", -1))
    == 2881
):
    raise SystemExit("ERROR: 旧六轮证据不是精确的 268365/2881 单门失败状态")
if not (
    prior_manifest.get("protocol") == "cyclic_start_base_pareto_recovery_v8_v2"
    and prior_manifest.get("quality_gate") == "FAIL"
    and prior_false == {"at_least_one_real_3zgc_candidate_is_released"}
    and prior_manifest.get("release_status")
    == "BLOCKED_FIXED_V2_BUDGET_DID_NOT_RECOVER_3ZGC"
    and int(prior_manifest.get("conditional_rounds_completed", -1)) == 6
    and int(prior_manifest.get("legacy_strict_hits_reaudited", -1)) == 2881
    and int(prior_manifest.get("released_candidates", -1)) == 0
    and prior_manifest.get("missing_targets_before_search") == ["3ZGC"]
    and prior_manifest.get("missing_targets_after_search") == ["3ZGC"]
):
    raise SystemExit("ERROR: V2 目录不是刚完成的精确六轮单门失败状态")
for label, manifest, root in (
    ("旧六轮", legacy_manifest, legacy),
    ("V2", prior_manifest, prior),
):
    leaves = list(artifact_leaves(manifest.get("artifacts")))
    if not leaves:
        raise SystemExit(f"ERROR: {label}清单没有可核验文件")
    for leaf in leaves:
        path = pathlib.Path(str(leaf.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(f"ERROR: {label}清单中的文件越界：{path}") from exc
        if not path.is_file() or sha256_file(path) != str(leaf.get("sha256", "")):
            raise SystemExit(f"ERROR: {label}文件缺失或哈希失配：{path}")

if not (
    prior_manifest.get("model_sha256") == legacy_manifest.get("model_sha256")
    and prior_manifest.get("legacy_manifest_sha256")
    == sha256_file(legacy_manifest_path)
    and prior_manifest.get("baseline_manifest_sha256")
    == legacy_manifest.get("baseline_manifest_sha256")
):
    raise SystemExit("ERROR: V2、旧六轮、模型或基线的哈希链不连续")

serine = dict(prior_manifest.get("serine_provenance_gate") or {})
source_by_residue = dict(serine.get("expert_source_by_residue") or {})
bitwise = dict(serine.get("bitwise_quality_checks") or {})
if not (
    serine.get("quality_gate") == "PASS"
    and serine.get("literal_normalization")
    == "lowercase_design_s_naturalizes_to_uppercase_S"
    and serine.get("serine_expert_source") == "v7_serine"
    and serine.get("non_ser_expert_source") == "v6_non_ser"
    and source_by_residue.get("S") == "v7_serine"
    and len(source_by_residue) == 20
    and all(
        source == ("v7_serine" if residue == "S" else "v6_non_ser")
        for residue, source in source_by_residue.items()
    )
    and bitwise
    and all(bitwise.values())
    and dict(prior_manifest.get("quality_checks") or {}).get(
        "physical_position_vectors_and_argmax_are_persisted"
    )
    is True
    and dict(prior_manifest.get("quality_checks") or {}).get(
        "cyclic_base_uses_joint_coordinate_sequence_roll_and_residue_index_reset"
    )
    is True
):
    raise SystemExit("ERROR: s→S 来源或物理点位修复证据不完整")

screen_sequences = set()
strict_screen_sequences = set()
argmax_positions = set()
screen_artifacts = dict(
    dict(prior_manifest.get("artifacts") or {}).get("conditional_methyl_screens")
    or {}
)
exact_artifacts = dict(
    dict(prior_manifest.get("artifacts") or {}).get(
        "conditional_cyclic_base_shortlists"
    )
    or {}
)
for leaf in screen_artifacts.values():
    with gzip.open(
        pathlib.Path(str(leaf["path"])), "rt", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            sequence = str(row.get("sequence", "")).upper()
            maximum = float(row.get("maximum_probability", "nan"))
            argmax = int(row.get("argmax_position_1based", -1))
            strict = int(row.get("passes_strict_probability", -1))
            if not (
                len(sequence) == 7
                and sequence not in screen_sequences
                and math.isfinite(maximum)
                and 0.0 <= maximum <= 1.0
                and 1 <= argmax <= 7
                and str(row.get("argmax_residue", "")) == sequence[argmax - 1]
                and strict == int(round(maximum, 8) > 0.6)
            ):
                raise SystemExit("ERROR: V2 methyl-screen 行重复或字段损坏")
            screen_sequences.add(sequence)
            argmax_positions.add(argmax)
            if strict:
                strict_screen_sequences.add(sequence)

exact_sequences = set()
for leaf in exact_artifacts.values():
    with gzip.open(
        pathlib.Path(str(leaf["path"])), "rt", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            sequence = str(row.get("sequence", "")).upper()
            if len(sequence) != 7 or sequence in exact_sequences:
                raise SystemExit("ERROR: V2 exact cyclic-base 行重复或字段损坏")
            exact_sequences.add(sequence)
if not (
    len(screen_sequences) == 159329
    and len(exact_sequences) == 6 * 4096
    and strict_screen_sequences <= exact_sequences
    and argmax_positions == set(range(1, 8))
):
    raise SystemExit(
        "ERROR: V2 完整前沿数量、strict 保留或 1–7 物理位置覆盖不符"
    )

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
if shutil.disk_usage(repo).free < 2 * 1024**3:
    raise SystemExit("ERROR: 仓库所在磁盘可用空间不足 2 GiB")
print(
    "V3 inputs: legacy=268,365; legacy_strict=2,881; "
    "V2_screen=159,329; V2_exact=24,576; V2_rounds=6; "
    "prior_released=0; only_missing=3ZGC; s_to_S=PASS; positions_1_to_7=PASS"
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

echo "[2/4] 检查是否已有同一 V3 任务"
if [[ -f "$PIDFILE" ]]; then
  existing_pid="$(tr -cd '0-9' < "$PIDFILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    if "$PYTHON_BIN" - "$existing_pid" "$REPO" "$SEARCH" <<'PY'
import pathlib
import sys

pid = int(sys.argv[1])
repo = str(pathlib.Path(sys.argv[2]).resolve())
search = str(pathlib.Path(sys.argv[3]).resolve())
try:
    command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(
        b"\0", b" "
    ).decode("utf-8", errors="replace")
except (FileNotFoundError, PermissionError, ProcessLookupError):
    raise SystemExit(1)
raise SystemExit(0 if repo in command and search in command else 1)
PY
    then
      echo "V3 任务已经在后台运行：PID=$existing_pid"
      echo "查看进度：tail -n 120 -F $LOG"
      exit 0
    fi
    echo "忽略已被其他进程复用的旧 PID：$existing_pid"
    rm -f -- "$PIDFILE"
  fi
fi

echo "[3/4] 后台启动：只搜索缺失的 3ZGC；其余 16 靶点复用不可变基线并仅做终审"
nohup bash -lc '
  set -Eeuo pipefail
  export OMP_NUM_THREADS=16
  export MKL_NUM_THREADS=16
  export OPENBLAS_NUM_THREADS=16
  export NUMEXPR_NUM_THREADS=16
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONUNBUFFERED=1
  cd "$1"
  printf "\n===== V8 V3 full-frontier launch %s =====\n" "$(date --iso-8601=seconds)"
  "$2" "$3" \
    --device cuda --resume --frontier-v3 \
    --prior-v2-dir "$6" --out-dir "$7"
  "$2" "$4" \
    --device cuda --overwrite \
    --search-dir "$7" --out-dir "$8" --audit-out-dir "$9"
  "$2" "$5" \
    --search-dir "$7" --generation-dir "$8" --audit-dir "$9" \
    --output "${10}"
  echo "===== ALL V8 V3 FULL-FRONTIER AUTOMATED GATES PASSED ====="
' v8-v3 "$REPO" "$PYTHON_BIN" "$SEARCH" "$FINAL" "$PACKAGE" \
  "$PRIOR_V2" "$V3_SEARCH" "$V3_GENERATION" "$V3_AUDIT" "$V3_BUNDLE" \
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
  if grep -q 'ALL V8 V3 FULL-FRONTIER AUTOMATED GATES PASSED' "$LOG"; then
    echo "===== V8 V3 已在 20 秒内全部完成 ====="
    tail -n 100 "$LOG" || true
    exit 0
  fi
  echo "ERROR: V3 任务启动后退出，末尾日志如下：" >&2
  tail -n 200 "$LOG" >&2 || true
  exit 1
fi

echo "===== V8 V3 BACKGROUND JOB IS HEALTHY: PID=$job_pid ====="
echo "这里仅表示预检和后台启动完成，不代表科学计算结束。"
echo "SSH/网页断开不会停止任务。查看进度："
echo "tail -n 120 -F $LOG"
echo "确认最终结果："
echo "grep -E 'ALL V8 V3 FULL-FRONTIER AUTOMATED GATES PASSED|Quality gate:|Released 3ZGC|ERROR|Traceback' $LOG | tail -n 60"
echo "最终审计包："
echo "$V3_BUNDLE"
echo "当前日志末尾："
tail -n 50 "$LOG" || true
