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

## 当前恢复流程：Ser 来源质控、循环起点不变性、结构先行（V6）

若从未生成 V6，首次完整运行才使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_cyclic_representation_v6.ps1 -Force
```

若 V6 模型、19,500 条初始生成或后续补采样已经存在，**不要再用 `-Force`**。
使用下面命令保留已有结果；达到累计补采样上限仍为零产出的靶点会登记为明确的
模型弃权，而不会继续盲抽或降低阈值：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_cyclic_representation_v6.ps1 -ResumeQuota
```

V5 已撤回：3AV9 的旧结构通过行没有甲基 token，且 V5 的甲基候选全部集中
在数组第 7 位。V6 从 canonical V28 重新训练完整 20-expert 模块；训练和部署
都联合轮换环肽序列、标签、N/CA/C/O 坐标及 residue index，并把概率映射回
物理残基。旧 7 条不再冻结，全部 17 个靶点重新生成。独立 test、生成与三遍
审计通过后仍只产出人工复核包，不自动创建尚哥 handoff；结构返回前也禁止
透膜步骤。完整证据、门槛和输出说明见：

```text
paper_clean_v28/serine_qc_retrain/README.md
```

## 历史 T=0.5 预筛流程

`run_t05_rerun.ps1` 和 `paper_clean_v28/rerun_t05/` 保留用于复现此前
13-target、透膜预筛在前的历史批次，不能用于当前恢复交付。

```text
paper_clean_v28/rerun_t05/README.md
```
