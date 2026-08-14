# 温度 0.5 多种子重跑与本地预筛

> **历史流程警告**：本目录保留的是此前“透膜预筛在结构之前”的
> 13-target 批次，只用于复现，不用于当前交付。当前实际顺序必须是
> 结构先行、透膜后置；请改用 `run_serine_qc_recovery.ps1` 和
> `paper_clean_v28/serine_qc_retrain/README.md`。
>
> 旧结果还存在外层随机采样顺序与模型内部固定因果顺序不一致的问题；
> 现有 872 条第 7 位集中结果已撤回。当前生成器只能与带有
> `corrected_labels_order_balanced_v3` 元数据的新 checkpoint 配套用于
> Ser-QC 恢复计划，并在结果门通过后停在人工复核。

## 目的

只重跑温度 `0.5` 下尚未同时达到以下目标的 13 个复合物：

```text
3P8F 3ZGC 3AV9 3AVA 3AVB 3AVF 3AVJ
3AVG 3AVI 3AVK 3AVH 3AVM 3AVN
```

已经达到“环肽 RMSD 小且透膜性提高”的以下 4 个先冻结，不重复消耗机器：

```text
1SFI 3WNE 4K1E 4KEL
```

固定协议：

- 生成温度：`0.5`
- 甲基化阈值：`0.6`
- 基础随机种子：`101, 202, 303, 404, 505`
- 本地原始生成量：`13,500`
- 尚哥结构任务上限：`185`
- 透膜性提高：同一模型下 `P(甲基化重新设计肽) > P(native 原始肽)`
- 最终结构门槛：整体复合物 CA RMSD `<3 Å`，并且在同一次整体对齐后、只允许正向循环移位、不做肽单独拟合的完整环肽 CA RMSD `<3 Å`

## 第一次运行：生成并准备透膜性输入

在 Windows 的 GPU/PyTorch 环境（此前使用的 `wain`）和仓库根目录运行：

```powershell
git pull origin main
```

```powershell
powershell -ExecutionPolicy Bypass -File .\run_t05_rerun.ps1
```

默认 `BatchSize=16`。若 RTX 4060 显存不足，用：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_t05_rerun.ps1 -BatchSize 8 -Force
```

生成完成后，控制器会停在透膜性阶段，因为 clean 仓库只有既往透膜性结果和合并脚本，没有可调用的透膜性推理模型。它不会用替代模型或旧分数冒充新候选结果。

需要把以下文件交给此前完全相同的透膜性模型：

```text
paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\permeability_input.csv
```

其中包括：

- 所有“新生成、至少一个甲基化位点、且未在历史 4,115 条池中出现”的候选；
- 17 条 native 原始肽，`methy_index=[]`，用于统一的 native 基线。

透膜性模型输出至少需要保留：

```text
id, permeability_pred
```

推荐保存为：

```text
paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\permeability_predictions.csv
```

## 第二次运行：自动筛到给尚哥的少量结构任务

预测文件按上述默认名称放好后，运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_t05_rerun.ps1 -SkipGeneration
```

如果预测文件在其他位置：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_t05_rerun.ps1 `
  -SkipGeneration `
  -PermeabilityCsv "E:\path\to\permeability_predictions.csv"
```

控制器会自动执行：

1. 核验候选及 17 条 native 的预测覆盖率；
2. 只保留严格满足 `P_design > P_native` 的甲基化候选；
3. 合并天然化序列相同但甲基化位置不同的重复结构任务；
4. 先按 clean V28 在 native backbone 上的平均 log probability 选择结构兼容性更好的候选，再比较透膜性提高和 native recovery；
5. 按天然序列 `<80%` 一致性尽量保留多样性，不足配额时才标记 `RELAXED_FILL`；
6. 为尚哥输出 CSV、合并 FASTA、JSONL 和逐任务 FASTA。

主交付文件：

```text
paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\selected_for_structure\structure_tasks_for_shangge.csv
paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\selected_for_structure\structure_inputs_for_shangge\
paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\selected_for_structure\target_summary.csv
paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\selected_for_structure\selection_manifest.json
```

只有 `selection_manifest.json` 为 `quality_gate=PASS` 时，才代表13个目标都达到计划的本地预筛配额。`NEEDS_MORE_CANDIDATES` 会列出需要继续本地加采样的目标，已有合格任务仍会保留，不能用未提高透膜性的候选硬补。

## 采样量为什么不是每个目标都相同

`target_plan.json` 按旧池难度分配：

- `3P8F、3ZGC`：每个种子100条，共500条/目标；
- `3AV9、3AVA、3AVB、3AVF、3AVJ、3AVH、3AVM、3AVN`：每个种子200条，共1,000条/目标；
- `3AVG、3AVI、3AVK`：每个种子300条，共1,500条/目标。

`batch_size`只控制显存和速度，不改变计划生成总数。`seed`负责独立可复现抽样，`temperature`仍固定0.5，不通过继续升温来盲目扩大序列空间。

## 给尚哥的说明

```text
尚哥，我先固定温度0.5，不让公司机器盲跑全部新序列。本地会用5个固定随机种子生成13,500条，先硬筛甲基化、相对native透膜性严格提高，再按模型结构兼容性和序列多样性压缩到最多185个结构任务。您只需要跑最终CSV/FASTA里的任务。结构回来后我再按同一整体对齐计算整体RMSD和完整环肽RMSD，两项都<3 Å才算最终通过。
```
