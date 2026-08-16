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

## 当前恢复流程：Ser 来源质控、循环起点不变性、结构先行（V7）

当前只运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_serine_only_cyclic_v7.ps1
```

V6 把只涉及 Ser 的来源标签修复错误地扩大到全部 20 个 experts，导致 3ZGC
等非 Ser 预测退化。V7 从 canonical V28 只重训 Ser expert，并以逐张量 hash 和
非 Ser held-out 概率零差异证明其余模型未被改动。它保留、hash 固定并直接
重标注现有 31,500 条 V6 自然序列，不重采样、不降阈值，也不接受 3ZGC 弃权
来换 PASS。只有 17/17 均有新颖候选且独立三审通过才打人工复核 ZIP；不会创建
尚哥 handoff，结构返回前也禁止透膜步骤。V5/V6 脚本只保留用于历史复现。
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
