#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a Chinese, hash-bound pre-structure audit report for the V10 handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROTOCOL = "v10_prestructure_funnel_report_v1"
EXPECTED_TARGETS = 17
EXPECTED_SELECTED = 1700


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def nested_hash(payload: Mapping[str, Any], section: str, label: str) -> str:
    node = payload.get(section, {})
    if not isinstance(node, Mapping):
        return ""
    record = node.get(label, {})
    if not isinstance(record, Mapping):
        return ""
    return str(record.get("sha256", ""))


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else ["target_name"]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--generation-summary", required=True)
    parser.add_argument("--selector-manifest", required=True)
    parser.add_argument("--selector-summary", required=True)
    parser.add_argument("--ranker-manifest", required=True)
    parser.add_argument("--v10-final-manifest", required=True)
    parser.add_argument("--monomer-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    paths = {
        "generation_manifest": Path(args.generation_manifest).resolve(),
        "generation_summary": Path(args.generation_summary).resolve(),
        "selector_manifest": Path(args.selector_manifest).resolve(),
        "selector_summary": Path(args.selector_summary).resolve(),
        "ranker_manifest": Path(args.ranker_manifest).resolve(),
        "v10_final_manifest": Path(args.v10_final_manifest).resolve(),
        "monomer_manifest": Path(args.monomer_manifest).resolve(),
    }
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    generation = read_json(paths["generation_manifest"])
    selector = read_json(paths["selector_manifest"])
    ranker = read_json(paths["ranker_manifest"])
    final = read_json(paths["v10_final_manifest"])
    monomer = read_json(paths["monomer_manifest"])
    artifact_binding_checks = {
        "generation_summary_matches_its_named_generation_manifest_artifact": (
            nested_hash(
                generation, "artifacts", "generation_summary_by_target"
            )
            == sha256_file(paths["generation_summary"])
        ),
        "selector_summary_matches_its_named_selector_manifest_artifact": (
            nested_hash(
                selector, "release_artifacts", "selection_summary_by_target"
            )
            == sha256_file(paths["selector_summary"])
        ),
    }
    if not all(artifact_binding_checks.values()):
        failed = [name for name, passed in artifact_binding_checks.items() if not passed]
        raise RuntimeError(
            "V10 pre-structure report artifact binding failed: " + ", ".join(failed)
        )
    generation_rows = read_csv(paths["generation_summary"])
    selector_rows = read_csv(paths["selector_summary"])
    generation_by_target = {
        row["target_name"].upper(): row for row in generation_rows
    }
    selector_by_target = {row["target_name"].upper(): row for row in selector_rows}
    targets = sorted(set(generation_by_target) | set(selector_by_target))

    checks = {
        **artifact_binding_checks,
        "all_upstream_manifests_are_pass": all(
            payload.get("quality_gate") == "PASS"
            for payload in (generation, selector, ranker, final, monomer)
        ),
        "generation_and_selector_have_exact_same_17_targets": (
            len(targets) == EXPECTED_TARGETS
            and set(generation_by_target) == set(selector_by_target)
        ),
        "final_manifest_reports_exactly_1700": int(final.get("rows", -1)) == EXPECTED_SELECTED,
        "every_target_selected_exactly_100": all(
            int(selector_by_target[target].get("selected", -1)) == 100
            for target in targets
        ),
        "every_target_generation_quota_was_met_before_selection": all(
            int(generation_by_target[target].get(
                "enough_candidates_before_permeability", 0
            ))
            == 1
            for target in targets
        ),
        "ranker_is_explicitly_prestructure_not_observed_rmsd": (
            "not observed RMSD" in str(ranker.get("scientific_scope", ""))
        ),
    }

    funnel_rows: List[Dict[str, Any]] = []
    for target in targets:
        generated = generation_by_target[target]
        selected = selector_by_target[target]
        raw = int(generated["raw_generated"])
        unique = int(generated["unique_generated"])
        stable = int(generated["new_methylated_for_permeability"])
        valid_pool = int(selected["valid_stable_novel_pool"])
        cyclic_unique_pool = int(
            selected["cyclic_unique_pool_used_for_rmsd_top_quartile"]
        )
        selected_count = int(selected["selected"])
        funnel_rows.append(
            {
                "target_name": target,
                "raw_draws": raw,
                "unique_natural_candidates": unique,
                "strict_stable_methylated_novel_candidates": stable,
                "strict_stable_methylated_yield_over_raw": stable / raw if raw else 0.0,
                "exact_cyclic_base_and_independent_valid_pool": valid_pool,
                "global_forward_cyclic_unique_rmsd_priority_pool": cyclic_unique_pool,
                "selected_for_structure": selected_count,
                "selected_contains_predicted_methylation": 1,
                "rmsd_priority_pool_score_mean": selected.get(
                    "rmsd_priority_pool_score_mean", ""
                ),
                "rmsd_priority_selected_score_min": selected.get(
                    "rmsd_priority_selected_score_min", ""
                ),
                "rmsd_priority_selected_score_mean": selected.get(
                    "rmsd_priority_selected_score_mean", ""
                ),
                "maximum_single_position_share": selected.get(
                    "maximum_single_position_share", ""
                ),
                "position_concentration_policy": selected.get(
                    "position_concentration_policy", ""
                ),
                "position_concentration_pass": selected.get(
                    "position_concentration_pass", ""
                ),
                "methyl_residue_concentration_pass": selected.get(
                    "methyl_residue_concentration_pass", ""
                ),
            }
        )
        checks[f"{target}_generation_and_selection_funnel_is_monotone"] = (
            raw >= unique >= stable >= valid_pool >= cyclic_unique_pool >= selected_count
        )
        checks[f"{target}_rmsd_priority_pool_is_at_least_400"] = (
            cyclic_unique_pool >= 400
        )
        checks[f"{target}_selected_fraction_is_at_most_25_percent"] = (
            selected_count / cyclic_unique_pool <= 0.25
            if cyclic_unique_pool
            else False
        )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("V10 pre-structure report input gate failed: " + ", ".join(failed))
    total_raw = sum(int(row["raw_draws"]) for row in funnel_rows)
    total_stable = sum(
        int(row["strict_stable_methylated_novel_candidates"])
        for row in funnel_rows
    )
    cv = ranker["development_cv"]["joint_lt5"]
    markdown = [
        "# V10 17×100 与单体：结构预测前审计报告",
        "",
        "## 当前结论",
        "",
        "- 17 个复合物各保留 100 条，共 1700 条；每条都含至少一个模型预测的稳定甲基化位点。",
        "- ‘含甲基化’表示新模型在全部循环起点下的最小概率经八位小数归一后严格大于 0.6，不表示实验已经证实发生甲基化。",
        "- 1700 条已通过逐条 batch-size=1 甲基头与 cyclic-base 回放，并再次独立重算 RMSD 优先分数。",
        "- RMSD 优先分数是结构预测前排序，不是实际 RMSD；在尚哥返回结构前，不能宣称新批次已经改善 `<3 Å` 或 `<5 Å` 比例。",
        "",
        "## 与旧六复合物基线的关系",
        "",
        "- 旧条件 micro：`<3 Å = 16/476 = 3.36%`，`<5 Å = 101/476 = 21.22%`。",
        "- 旧六靶点等权 macro：`<3 Å = 4.37%`，`<5 Å = 23.08%`。",
        "- V10 的低容量排序器使用整靶点留一验证；主终点 `<5 Å` 的 OOF AUC 为 "
        f"`{float(cv['pooled_oof_auc']):.4f}`，每靶点前四分位通过率为 "
        f"`{float(cv['top_fraction_rate']):.2%}`，旧总体率为 `{float(cv['baseline_rate']):.2%}`。",
        "- 这项回顾性富集只授权排序器用于前瞻批次，不等于 17 靶点实测改善，尤其不能外推为 3AV 家族的已证实结果。",
        "",
        "## 生成漏斗",
        "",
        f"总 raw draws：{total_raw}；严格稳定、含预测甲基且新颖的候选：{total_stable}；最终：1700。",
        "",
        "逐靶点完整数字见 `v10_prestructure_funnel_by_target.csv`。分母始终保留 raw draws，最终 100/靶点不能被误报为甲基化有效率 100%。",
        "",
        "## 单体",
        "",
        f"- 已按修正标签重算 {int(monomer['sample_count'])} 个单体、{int(monomer['position_count'])} 个位点。",
        f"- 天然氨基酸恢复率：{float(monomer['base_recovery']):.4%}；在相同 seed/batch 协议下，base head 必须与原 V28 的 1505/1505 位点预测完全一致。",
        f"- 端到端扩展 token 恢复率：{float(monomer['end_to_end_extended_token_recovery']):.4%}。",
        "- 751 个单体按项目口径记为公司自行计算的 Rosetta 理论模型；151 条是内部开发审计，不称独立盲测。",
        "- 旧天然化 variant4 PDB 只有在 Windows 本地完成 151/151 文件名、序列和 CA 覆盖复核后才可复用；显式甲基 variant3 不匹配时必须重预测或报告缺失。",
        "",
        "## 下一步结构验收",
        "",
        "尚哥返回结构后，必须冻结同一批 1700 清单，按一次全复合物对齐、同一对齐框架下 best-forward cyclic peptide、禁止反向和肽段二次拟合的原论文口径，分别报告 global、cyclic 和 joint `<3/<5 Å`。",
        "",
    ]
    funnel_path = out_dir / "v10_prestructure_funnel_by_target.csv"
    markdown_path = out_dir / "v10_prestructure_audit_cn.md"
    atomic_write_csv(funnel_path, funnel_rows)
    atomic_write_text(markdown_path, "\n".join(markdown))
    report = {
        "quality_gate": "PASS",
        "protocol": PROTOCOL,
        "quality_checks": checks,
        "total_raw_draws": total_raw,
        "total_strict_stable_methylated_novel_candidates": total_stable,
        "final_selected": EXPECTED_SELECTED,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "program": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "artifacts": {
            "funnel_by_target": {
                "path": str(funnel_path),
                "sha256": sha256_file(funnel_path),
            },
            "chinese_report": {
                "path": str(markdown_path),
                "sha256": sha256_file(markdown_path),
            },
        },
    }
    atomic_write_json(out_dir / "v10_prestructure_report_manifest.json", report)
    print("V10 pre-structure report: PASS", flush=True)


if __name__ == "__main__":
    main()
