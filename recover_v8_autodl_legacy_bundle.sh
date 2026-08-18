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

echo "[1/4] 验证补传文件、旧迁移包、源码和六轮证据"
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
if [[ -s "$PIDFILE" ]]; then
    OLD_PID="$(tr -cd '0-9' < "$PIDFILE")"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: PID=$OLD_PID 的 V8 任务仍在运行；拒绝重复启动"
        exit 1
    fi
fi

python - "$UPLOAD" "$REPO" "$V8_ROOT" "$IMPORT_MANIFEST" <<'PY'
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
from typing import Any, Mapping


upload = pathlib.Path(sys.argv[1]).resolve()
repo = pathlib.Path(sys.argv[2]).resolve()
v8_root = pathlib.Path(sys.argv[3]).resolve()
import_manifest_path = pathlib.Path(sys.argv[4]).resolve()

model_path = v8_root / "model/frankenstein_v28_source_scoped_hybrid_v8.pt"
model_manifest_path = v8_root / "model/expert_source_composition_manifest.json"
representation_path = v8_root / "representation_audit/cyclic_representation_audit.json"
baseline_dir = v8_root / "generation_baseline"
baseline_manifest_path = baseline_dir / "generation_manifest.json"
plan_path = repo / "paper_clean_v28/serine_qc_retrain/target_plan_cyclic_representation_v6.json"
native_path = repo / "17_complexes_native.jsonl"
historical_path = repo / "paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv"
prior_path = repo / "paper_clean_v28_outputs/rerun_temperature_0.5_multiseed/methylated_new_candidates.csv"
retrain_dir = repo / "paper_clean_v28/serine_qc_retrain"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: pathlib.Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_under_repo(path: pathlib.Path, label: str) -> pathlib.Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {label} 路径逃出仓库：{resolved}") from exc
    if not resolved.is_file():
        raise SystemExit(f"ERROR: {label} 文件不存在：{resolved}")
    return resolved


def verify_text_newline_equivalence(
    expected: str,
    target: pathlib.Path,
    label: str,
) -> tuple[str, str]:
    data = target.read_bytes()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(
            f"ERROR: {label} 不是 UTF-8 文本，不能进行跨平台换行核验：{target}"
        ) from exc
    lf = data.replace(b"\r\n", b"\n")
    if b"\r" in lf:
        raise SystemExit(
            f"ERROR: {label} 含独立 CR 字节，拒绝当作普通 CRLF/LF 差异：{target}"
        )
    crlf = lf.replace(b"\n", b"\r\n")
    if data not in {lf, crlf}:
        raise SystemExit(f"ERROR: {label} 换行格式混杂：{target}")
    variants = {
        sha256_bytes(lf): "LF",
        sha256_bytes(crlf): "CRLF",
    }
    if expected not in variants:
        raise SystemExit(
            f"ERROR: {label} 不是仅 CRLF/LF 差异；"
            f" manifest={expected}; destination={sha256_bytes(data)}; "
            f"LF={sha256_bytes(lf)}; CRLF={sha256_bytes(crlf)}; file={target}"
        )
    observed = sha256_bytes(data)
    return variants[expected], variants[observed]


def rebase_hash_field(
    payload: dict[str, Any],
    manifest_name: str,
    field: str,
    target: pathlib.Path,
    changes: list[dict[str, str]],
) -> None:
    target = require_under_repo(target, f"{manifest_name}.{field}")
    expected = str(payload.get(field, ""))
    observed = sha256_file(target)
    if len(expected) != 64:
        raise SystemExit(f"ERROR: {manifest_name}.{field} 缺少合法 SHA256")
    if expected == observed:
        return
    source_style, destination_style = verify_text_newline_equivalence(
        expected, target, f"{manifest_name}.{field}"
    )
    payload[field] = observed
    changes.append(
        {
            "manifest": manifest_name,
            "field": field,
            "path": str(target),
            "source_sha256": expected,
            "destination_sha256": observed,
            "verified_relation": f"{source_style}_TO_{destination_style}_ONLY",
        }
    )


def declared_target(
    payload: Mapping[str, Any],
    manifest_name: str,
    path_field: str,
    expected_path: pathlib.Path | None = None,
) -> pathlib.Path:
    declared = require_under_repo(
        pathlib.Path(str(payload.get(path_field, ""))),
        f"{manifest_name}.{path_field}",
    )
    if expected_path is not None and declared != expected_path.resolve():
        raise SystemExit(
            f"ERROR: {manifest_name}.{path_field} 目标路径错误；"
            f" declared={declared}; expected={expected_path.resolve()}"
        )
    return declared


def set_dependency_hash(
    payload: dict[str, Any],
    manifest_name: str,
    field: str,
    allowed_old: set[str],
    destination_sha: str,
    changes: list[dict[str, str]],
) -> None:
    source_sha = str(payload.get(field, ""))
    if source_sha == destination_sha:
        return
    if source_sha not in allowed_old:
        raise SystemExit(
            f"ERROR: {manifest_name}.{field} 不属于本次已验证的清单级联；"
            f" declared={source_sha}; allowed={sorted(allowed_old)}"
        )
    payload[field] = destination_sha
    changes.append(
        {
            "manifest": manifest_name,
            "field": field,
            "source_sha256": source_sha,
            "destination_sha256": destination_sha,
            "verified_relation": "RELOCATED_MANIFEST_DEPENDENCY",
        }
    )


# Recheck the exact legacy executable blobs before touching manifest metadata.
expected_blobs = {
    repo / "run_v8_autodl_resume.sh": "1869637f81eb10f67a66f1a0dc5f125cedc1b71c",
    retrain_dir / "14_directed_recovery_search_v8.py": "e7c6fabb12e348e0c81cd9d12eed8f44bc32f758",
    retrain_dir / "15_finalize_and_audit_recovery_v8.py": "14f6e281e53e4722eba81b9c430a479770c80a75",
    retrain_dir / "16_v8_autodl_resume_bundle.py": "43bb3224cc0bdacd2ebefad69f481d2a15560dbd",
}
for path, expected_blob in expected_blobs.items():
    if not path.is_file() or git_blob_sha1(path) != expected_blob:
        raise SystemExit(f"ERROR: 精确提交源码核验失败：{path}")

imported = read_json(import_manifest_path)
portable_path = require_under_repo(
    pathlib.Path(str(imported.get("portable_export_manifest", ""))),
    "portable_export_manifest",
)
portable = read_json(portable_path)
expected_prior = str(portable["source_config"]["input_hashes"]["prior"])
observed_prior = sha256_file(upload)
if observed_prior != expected_prior:
    raise SystemExit(
        "ERROR: 补传文件不是本次搜索使用的 prior CSV；"
        f" expected={expected_prior}; observed={observed_prior}"
    )

prior_path.parent.mkdir(parents=True, exist_ok=True)
if prior_path.is_file():
    if sha256_file(prior_path) != expected_prior:
        raise SystemExit(f"ERROR: 目标位置已有不同文件：{prior_path}")
else:
    shutil.copy2(upload, prior_path)
if sha256_file(prior_path) != expected_prior:
    raise SystemExit("ERROR: prior CSV 复制后哈希不一致")

required = [
    native_path,
    repo / "model_utils.py",
    repo / "nmethyl/utils/nmethyl_config.py",
    repo / "paper_clean_v28/clean_v28_common.py",
    repo / "paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py",
    retrain_dir / "02_retrain_canonical_expert_heads.py",
    retrain_dir / "07_audit_cyclic_representation_equivariance.py",
    retrain_dir / "10_reannotate_v6_pool_serine_only_v7.py",
    retrain_dir / "11_triple_audit_serine_only_v7.py",
    retrain_dir / "12_compose_source_scoped_hybrid_v8.py",
    retrain_dir / "13_audit_source_scoped_hybrid_v8.py",
    retrain_dir / "14_directed_recovery_search_v8.py",
    retrain_dir / "15_finalize_and_audit_recovery_v8.py",
    plan_path,
    historical_path,
    prior_path,
    model_path,
    model_manifest_path,
    representation_path,
    baseline_dir / "all_candidates.csv",
    baseline_dir / "unique_candidates.csv",
    baseline_dir / "methylated_new_candidates.csv",
    baseline_dir / "target_manifest.csv",
    baseline_dir / "generation_summary_by_target.csv",
    baseline_manifest_path,
    v8_root / "directed_search/mandatory_length_6_7_controls.csv",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("ERROR: 一次性输入审计发现以下缺项：\n- " + "\n- ".join(missing))

current_paths = {
    "model": model_path,
    "model_manifest": model_manifest_path,
    "representation_audit": representation_path,
    "baseline_manifest": baseline_manifest_path,
}
before_input_hashes = {
    name: sha256_file(path) for name, path in current_paths.items()
}
expected_current = dict(imported.get("current_input_hashes") or {})
if set(expected_current) != set(current_paths):
    raise SystemExit("ERROR: 旧导入清单的核心输入集合不完整")
for name, observed in before_input_hashes.items():
    if observed != str(expected_current[name]):
        raise SystemExit(
            f"ERROR: 修复前核心输入已被改动：{current_paths[name]}; "
            f"expected={expected_current[name]}; observed={observed}"
        )

evidence = dict(imported.get("evidence_files") or {})
if not evidence:
    raise SystemExit("ERROR: 导入清单缺少六轮 evidence inventory")
for relative, expected in evidence.items():
    path = require_under_repo(repo / relative, f"evidence:{relative}")
    if sha256_file(path) != str(expected):
        raise SystemExit(f"ERROR: 搜索 ledger/checkpoint 哈希不一致：{path}")

# Compute all three rebased manifests in memory first. No scientific artifact
# is modified; only hashes proven to differ solely by CRLF/LF are rewritten.
newline_changes: list[dict[str, str]] = []
dependency_changes: list[dict[str, str]] = []
model = read_json(model_manifest_path)
model_before_sha = before_input_hashes["model_manifest"]
for field, target in {
    "composer_program_sha256": retrain_dir / "12_compose_source_scoped_hybrid_v8.py",
    "trainer_program_sha256": retrain_dir / "02_retrain_canonical_expert_heads.py",
    "common_program_sha256": repo / "paper_clean_v28/clean_v28_common.py",
    "model_utils_program_sha256": repo / "model_utils.py",
    "nmethyl_config_program_sha256": repo / "nmethyl/utils/nmethyl_config.py",
}.items():
    rebase_hash_field(model, "model", field, target, newline_changes)
model_test = declared_target(model, "model", "test_jsonl")
rebase_hash_field(model, "model", "test_jsonl_sha256", model_test, newline_changes)
model_data = json_bytes(model)
model_after_sha = sha256_bytes(model_data)

representation = read_json(representation_path)
representation_before_sha = before_input_hashes["representation_audit"]
set_dependency_hash(
    representation,
    "representation",
    "model_manifest_sha256",
    {model_before_sha, model_after_sha},
    model_after_sha,
    dependency_changes,
)
for field, target in {
    "representation_auditor_program_sha256": retrain_dir / "13_audit_source_scoped_hybrid_v8.py",
    "equivariance_auditor_program_sha256": retrain_dir / "07_audit_cyclic_representation_equivariance.py",
    "common_program_sha256": repo / "paper_clean_v28/clean_v28_common.py",
    "model_utils_program_sha256": repo / "model_utils.py",
    "nmethyl_config_program_sha256": repo / "nmethyl/utils/nmethyl_config.py",
}.items():
    rebase_hash_field(representation, "representation", field, target, newline_changes)
for path_field, hash_field, expected_path in (
    ("test_jsonl", "test_jsonl_sha256", None),
    ("native_jsonl", "native_jsonl_sha256", native_path),
    ("best_csv", "best_csv_sha256", None),
    ("plan", "plan_sha256", plan_path),
):
    target = declared_target(
        representation, "representation", path_field, expected_path
    )
    rebase_hash_field(
        representation, "representation", hash_field, target, newline_changes
    )
representation_data = json_bytes(representation)
representation_after_sha = sha256_bytes(representation_data)

baseline = read_json(baseline_manifest_path)
baseline_before_sha = before_input_hashes["baseline_manifest"]
set_dependency_hash(
    baseline,
    "baseline",
    "expert_manifest_sha256",
    {model_before_sha, model_after_sha},
    model_after_sha,
    dependency_changes,
)
heldout = dict(baseline.get("cyclic_representation_heldout_audit") or {})
set_dependency_hash(
    heldout,
    "baseline.cyclic_representation_heldout_audit",
    "sha256",
    {representation_before_sha, representation_after_sha},
    representation_after_sha,
    dependency_changes,
)
rebase_hash_field(
    heldout,
    "baseline.cyclic_representation_heldout_audit",
    "plan_sha256",
    plan_path,
    newline_changes,
)
baseline["cyclic_representation_heldout_audit"] = heldout
for field, target in {
    "reannotator_program_sha256": retrain_dir / "10_reannotate_v6_pool_serine_only_v7.py",
    "generator_program_sha256": repo / "paper_clean_v28/rerun_t05/01_generate_t05_multiseed.py",
    "common_program_sha256": repo / "paper_clean_v28/clean_v28_common.py",
    "model_utils_program_sha256": repo / "model_utils.py",
    "nmethyl_config_program_sha256": repo / "nmethyl/utils/nmethyl_config.py",
    "plan_sha256": plan_path,
    "native_jsonl_sha256": native_path,
    "historical_design_csv_sha256": historical_path,
    "prior_handoff_csv_sha256": prior_path,
}.items():
    rebase_hash_field(baseline, "baseline", field, target, newline_changes)
for path_field, expected_path in (
    ("plan", plan_path),
    ("native_jsonl", native_path),
    ("historical_design_csv", historical_path),
    ("prior_handoff_csv", prior_path),
):
    declared_target(baseline, "baseline", path_field, expected_path)
baseline_data = json_bytes(baseline)
baseline_after_sha = sha256_bytes(baseline_data)

after_input_hashes = {
    "model": before_input_hashes["model"],
    "model_manifest": model_after_sha,
    "representation_audit": representation_after_sha,
    "baseline_manifest": baseline_after_sha,
}
existing_rebase = imported.get("legacy_cross_platform_manifest_rebase")
if existing_rebase is None:
    imported["legacy_cross_platform_manifest_rebase"] = {
        "quality_gate": "PASS",
        "protocol": "v8_legacy_autodl_crlf_lf_manifest_rebase_v1",
        "scientific_artifact_bytes_changed": False,
        "verification_rule": "SHA256 matched exact UTF-8 LF or CRLF byte variant only",
        "newline_hash_rebases": newline_changes,
        "dependency_hash_rebases": dependency_changes,
        "source_current_input_hashes": before_input_hashes,
        "destination_current_input_hashes": after_input_hashes,
    }
else:
    if not (
        isinstance(existing_rebase, Mapping)
        and existing_rebase.get("quality_gate") == "PASS"
        and existing_rebase.get("protocol")
        == "v8_legacy_autodl_crlf_lf_manifest_rebase_v1"
        and existing_rebase.get("scientific_artifact_bytes_changed") is False
    ):
        raise SystemExit("ERROR: 已存在但无法识别的跨平台清单修复记录")
imported["current_input_hashes"] = after_input_hashes
imported_data = json_bytes(imported)

# Commit the fully precomputed metadata update atomically file by file.
for path, data in (
    (model_manifest_path, model_data),
    (representation_path, representation_data),
    (baseline_manifest_path, baseline_data),
    (import_manifest_path, imported_data),
):
    if path.read_bytes() != data:
        atomic_write(path, data)

# Re-read and prove the destination state and all actual phase-14 gates before
# the shell is allowed to launch any GPU process.
for name, expected in after_input_hashes.items():
    if sha256_file(current_paths[name]) != expected:
        raise SystemExit(f"ERROR: 清单修复后核心输入哈希错误：{current_paths[name]}")

search_path = retrain_dir / "14_directed_recovery_search_v8.py"
spec = importlib.util.spec_from_file_location("v8_preflight_search", search_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"ERROR: 无法导入真实 V8 校验器：{search_path}")
search = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = search
spec.loader.exec_module(search)
search.validate_baseline(
    baseline_dir,
    model_path,
    model_manifest_path,
    representation_path,
    plan_path,
    native_path,
    historical_path,
    prior_path,
)

# The finalizer adds five source-model path/hash checks beyond phase 14.
model_current = read_json(model_manifest_path)
for label, path_field, hash_field in (
    ("canonical", "canonical_checkpoint", "canonical_checkpoint_sha256"),
    ("v6_checkpoint", "v6_checkpoint", "v6_checkpoint_sha256"),
    ("v6_manifest", "v6_manifest", "v6_manifest_sha256"),
    ("v7_checkpoint", "v7_checkpoint", "v7_checkpoint_sha256"),
    ("v7_manifest", "v7_manifest", "v7_manifest_sha256"),
):
    target = declared_target(model_current, "model", path_field)
    if sha256_file(target) != str(model_current.get(hash_field, "")):
        raise SystemExit(f"ERROR: finalizer 模型来源校验失败：{label} -> {target}")

print(f"Prior SHA256: {observed_prior}")
print(
    "Cross-platform manifest audit: PASS; "
    f"newline fields rebased={len(newline_changes)}; "
    f"dependency fields rebased={len(dependency_changes)}"
)
print(
    f"===== COMPLETE PRE-RUN AUDIT PASSED: {len(required)} runtime files + "
    f"{len(evidence)} evidence files + model/representation/baseline/finalizer gates ====="
)
PY

echo "[2/4] 编译检查三个实际执行程序"
python -m py_compile \
    "$REPO/paper_clean_v28/serine_qc_retrain/14_directed_recovery_search_v8.py" \
    "$REPO/paper_clean_v28/serine_qc_retrain/15_finalize_and_audit_recovery_v8.py" \
    "$REPO/paper_clean_v28/serine_qc_retrain/16_v8_autodl_resume_bundle.py"

echo "[3/4] 原地续跑；复用六轮搜索，只在 RTX 5090 上做规定的重评分和最终审计"
if [[ -e "$RECOVERY_LOG" ]]; then
    mv "$RECOVERY_LOG" "${RECOVERY_LOG}.previous.$(date +%s)"
fi

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
    bash "$REPO/run_v8_autodl_resume.sh" "$BUNDLE" \
    >> "$RECOVERY_LOG" 2>&1 < /dev/null &
RUN_PID=$!
printf '%s\n' "$RUN_PID" > "$PIDFILE"

echo "[4/4] 等待 20 秒确认没有启动即退"
sleep 20
if ! kill -0 "$RUN_PID" 2>/dev/null; then
    echo "ERROR: 全量预检虽通过，但任务在 20 秒内退出；完整末尾日志如下："
    tail -n 260 "$RECOVERY_LOG"
    exit 1
fi

echo "===== V8 BACKGROUND JOB IS HEALTHY: PID=$RUN_PID ====="
echo "这里表示‘准备和启动全部完成’，不是科学计算已经跑完。"
echo "网页/SSH 断开不会停止任务。查看进度："
echo "tail -n 120 -F $RECOVERY_LOG"
echo "确认最终完成："
echo "grep -E 'ALL AUTOMATED GATES PASSED|Quality gate:|ERROR|Traceback' $RECOVERY_LOG | tail -n 30"
echo "当前日志末尾："
tail -n 80 "$RECOVERY_LOG"
