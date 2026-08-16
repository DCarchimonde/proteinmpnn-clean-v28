# ProteinMPNN Clean V28 Handoff Package

本仓库用于 clean V28 结果交接。

## 给师兄的文件

单体：

```text
paper_clean_v28_outputs/monomer_design_structure_manifest.csv
nmethyl_data/test_set/test.jsonl
```

复合物不用重新预测结构，复用之前结果。

复合物对齐文件：

```text
paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv
paper_clean_v28_outputs/af3_manifest.csv
```

## 当前恢复流程：Ser 来源质控、循环起点不变性、结构先行（V8）

当前只运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\paper_clean_v28\run_serine_qc_source_scoped_hybrid_v8.ps1
```

V7 的 Ser 来源修复正确，但把另外 19 个 experts 退回 canonical 后，冻结
Recall@0.6 从 0.8046 降到 0.5096：少 77 个 TP，只少 21 个 FP。V8 不再训练，按
固定来源规则组合 canonical shared tensors、V6 non-Ser experts 和 V7 Ser
expert；必须逐位证明 non-Ser 概率继承 V6、Ser 概率继承 V7，并在内部冻结审计
上对 V6 的 Recall/F1 非劣后才继续。之后只读重标注现有 31,500 条 V6 自然
序列，对实际缺失的 3WNE/3ZGC 做固定预算确定性搜索，再做独立 overlay 三审。
只有 17/17、无正式弃权才打 checksum-indexed 人工复核 ZIP；不会创建尚哥
handoff，结构返回并同时通过两项 `<3 Å` RMSD 门前也禁止透膜步骤。V5/V6/V7
launcher 仅保留用于历史复现与失败诊断。

这里使用过的 151-record / 1,505-position 集合不是新的盲测，只能作为 V6/V7/V8
成对内部审计；论文最终主张仍需要新的 outer split 或真正 blind set。
完整证据、门槛和输出说明见：

```text
paper_clean_v28/serine_qc_retrain/README.md
```

## 历史 T=0.5 预筛流程

`run_t05_rerun.ps1` 和 `paper_clean_v28/rerun_t05/` 保留用于复现此前
13-target、透膜预筛在前的历史批次，不能用于当前恢复交付。

```text
paper_clean_v28/rerun_t05/README.md
```
