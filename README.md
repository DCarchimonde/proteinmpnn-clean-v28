# ProteinMPNN Clean V28 Handoff Package

本仓库用于 clean V28 结果交接。

## 当前入口：V10 循环稳定、RMSD 优先、17×100 与单体重算

V8 的训练、部署和释放合同存在确定错误；旧 6,964 条候选中有 5,196 条跨循环
表示发生 `0.6` 阈值分歧。V10 从原始 `frankenstein_v28.pt` 重新训练甲基
expert heads，硬执行 `round(min_probability, 8) > 0.6`、零分歧、严格去重和
证据感知的塌缩门；每靶点先取得至少 500 条严格候选，最终 100 条只能来自
去重后不少于 400 条池的 RMSD-priority 前四分位。旧 V8 候选不会混入新结果。

六个非 3AV 复合物旧结果的主要瓶颈是环肽 pose，而非全复合物 global RMSD。
V10 因而增加冻结的、按 target 留一外验证的低容量 RMSD 优先排序器，并重跑
151 个单体的序列/甲基化/循环表示指标。该排序分数只用于结构预测前优先级，
真正的 RMSD 改善必须等尚哥返回新结构后按同一协议复算。

在Linux/CUDA环境运行：

```bash
bash run_v10_rmsd_aware_1700_and_monomer.sh
```

当前 GitHub 只保存代码和冻结输入，不保存被忽略的 AutoDL GPU 输出，也不包含
新 PDB。全部机器边界、放行门和 Windows 后续命令见
`V10_TASK2_TASK3_运行与验收.md`。`run_cyclic_stability_v9_1700.sh` 与
`V9_CYCLIC_STABILITY_17X100_运行与验收.md` 仅保留为历史版本。

## 给师兄的文件

单体：

```text
paper_clean_v28_outputs/monomer_design_structure_manifest.csv
nmethyl_data/test_set/test.jsonl
```

以下“复用旧复合物结构”只适用于历史 V28 结果；V10 新生成的 1,700 条必须
重新预测结构，不能复用旧 PDB。

复合物对齐文件：

```text
paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv
paper_clean_v28_outputs/af3_manifest.csv
```

## 历史恢复流程：Ser 来源质控、循环起点不变性、结构先行（V8）

以下入口只为历史复现保留，**不要用于当前 Task 2/3**：

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
