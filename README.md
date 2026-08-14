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

## 当前恢复流程：Ser 来源质控、解码顺序平衡、结构先行

当前应运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_recovery.ps1
```

该流程从固定旧提交的原始 PDB 重建 Ser 标签，冻结共享主干和 base head、
用循环顺序平衡协议重训 canonical V28 的完整 20-expert 模块；7 个已过
结构门的 T=0.5 靶点仅用最终 checkpoint 重评分并复用结构，只为 10 个
未通过靶点重新生成。生成后执行独立三遍结果审计，并默认停在人工复核，
不会自动创建给尚哥的包；结构返回前也禁止生成透膜输入。完整证据、门槛
和输出说明见：

```text
paper_clean_v28/serine_qc_retrain/README.md
```

## 历史 T=0.5 预筛流程

`run_t05_rerun.ps1` 和 `paper_clean_v28/rerun_t05/` 保留用于复现此前
13-target、透膜预筛在前的历史批次，不能用于当前恢复交付。

```text
paper_clean_v28/rerun_t05/README.md
```
