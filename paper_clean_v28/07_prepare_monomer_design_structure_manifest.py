#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import math
from collections import defaultdict


DEFAULT_THRESHOLD = 0.6
THRESHOLD_OPERATOR = ">"
ROUNDING_POLICY = "round(prob,8)"


def safe_name(x):
    x = str(x)
    out = []
    for ch in x:
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def validate_probability(prob):
    try:
        prob = float(prob)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid methylation probability: {prob!r}") from exc
    if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
        raise ValueError(
            f"methylation probability must be finite and within [0, 1]: {prob!r}"
        )
    return prob


def validate_threshold(threshold):
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"methylation threshold must be finite and within [0, 1]: {threshold!r}"
        )
    return threshold


def passes_methylation_threshold(prob, threshold):
    prob = validate_probability(prob)
    return round(prob, 8) > threshold


def lower_if_methyl(aa, prob, threshold):
    if passes_methylation_threshold(prob, threshold):
        return aa.lower()
    return aa.upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--input_mode", default="strict_naturalized_input")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    threshold = validate_threshold(args.threshold)

    rows = read_csv(args.position_csv)

    selected = [r for r in rows if r["input_mode"] == args.input_mode]

    groups = defaultdict(list)
    for r in selected:
        key = (r["sample_name"], r["selected_chains"])
        groups[key].append(r)

    out = []

    for (sample_name, selected_chains), items in groups.items():
        items = sorted(items, key=lambda r: int(r["position_in_model"]))

        reference_original = "".join(r["target_token"] for r in items)
        reference_natural = "".join(r["true_base_token"].upper() for r in items)

        # classifier-only 口径：保留原始天然氨基酸，只用模型判断哪些位置甲基化
        known_base_design = "".join(
            lower_if_methyl(
                r["true_base_token"],
                r["prob_methyl_known_sequence"],
                threshold
            )
            for r in items
        )

        # end-to-end 口径：氨基酸也用模型预测的 pred_base_token，甲基化也用 end_to_end 概率
        e2e_design = "".join(
            lower_if_methyl(
                r["pred_base_token"],
                r["prob_methyl_end_to_end"],
                threshold
            )
            for r in items
        )

        known_methyl_positions = [
            str(i + 1)
            for i, r in enumerate(items)
            if passes_methylation_threshold(
                r["prob_methyl_known_sequence"], threshold
            )
        ]

        e2e_methyl_positions = [
            str(i + 1)
            for i, r in enumerate(items)
            if passes_methylation_threshold(r["prob_methyl_end_to_end"], threshold)
        ]

        out.append({
            "suggested_job_name": safe_name(f"{sample_name}_{args.input_mode}_e2e"),
            "sample_name": sample_name,
            "selected_chains": selected_chains,
            "input_mode": args.input_mode,
            "threshold": threshold,
            "threshold_operator": THRESHOLD_OPERATOR,
            "rounding_policy": ROUNDING_POLICY,

            "reference_original_sequence": reference_original,
            "reference_natural_sequence": reference_natural,

            "known_base_design_sequence": known_base_design,
            "known_base_sequence_for_structure_prediction": known_base_design.upper(),
            "known_base_methyl_positions_1based": ",".join(known_methyl_positions),
            "known_base_methyl_count": len(known_methyl_positions),

            "e2e_design_sequence": e2e_design,
            "e2e_sequence_for_structure_prediction": e2e_design.upper(),
            "e2e_methyl_positions_1based": ",".join(e2e_methyl_positions),
            "e2e_methyl_count": len(e2e_methyl_positions),

            "sequence_length": len(items),
            "note": "真正代表模型生成后序列的是 e2e_design_sequence；如果结构预测平台不支持小写甲基化残基，就用 e2e_sequence_for_structure_prediction，同时保留 e2e_methyl_positions_1based。",
        })

    write_csv(args.out_csv, out)
    print("已生成:", args.out_csv)
    print("任务数:", len(out))
    print("input_mode:", args.input_mode)
    print("threshold:", threshold)
    print("threshold_operator:", THRESHOLD_OPERATOR)
    print("rounding_policy:", ROUNDING_POLICY)


if __name__ == "__main__":
    main()
