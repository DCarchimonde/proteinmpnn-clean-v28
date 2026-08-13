# Ser 来源质控后的最小恢复流程

本目录修复一个有明确 PDB 来源证据的标签问题，并把恢复范围限制在当前
T=0.5 结构门未通过的复合物。它不是一次全模型重做。

## 已确认的问题

旧预处理把天然残基表和 N-甲基残基表按 `residue_name` 直接合并。两张表
都曾包含 `SER`，导致普通 `ATOM-SER` 被静默覆盖成小写 `s`。固定来源为
`DCarchimonde/ProteinMPNN` 提交：

```text
28dff152d83623dfb322480413b7dc889f8537a4
```

按 PDB 的 record type 和 `CN` 原子重建后：

| split | rows | `s -> S` | natural `S` | methyl `s` | natural `P` | methyl `p` |
|---|---:|---:|---:|---:|---:|---:|
| train | 600 | 242 | 242 | 50 | 307 | 0 |
| test | 151 | 62 | 62 | 12 | 83 | 0 |

解析规则是：

- `ATOM-SER` 始终是天然 `S`；
- 带 `CN` 的 `HETATM-5JP` 是甲基 `s`；
- 历史数据中一个带 `CN` 的 `HETATM-SER` 也确认为甲基 `s`；
- 模糊的 `HETATM-SER` 或缺少 `CN` 的甲基组分直接报错；
- `p` 没有正样本，因此不新增 token、不改变 40 字符字母表或 checkpoint 维度。

训练和测试 JSONL 的语义哈希、行数、标签变化类型及上述计数均为硬门槛。
坐标和所有非序列字段保持不变。

## 最小训练边界

`02_retrain_canonical_serine_expert.py` 使用生产推理相同的完整 clean-V28
网络，只更新 `experts[15]`（Ser）的 weight 和 bias。主干、base head、
其余 19 个专家和字母表全部冻结。训练与评估时先将标签 `s` 天然化为
输入 `S`，避免把答案 token 泄漏给主干。

保存前后的 state hash 必须证明只有以下两个 tensor 改变：

```text
experts.15.weight
experts.15.bias
```

同时要求非 Ser 测试概率逐位不变、保存后的 checkpoint 能被生产 loader
严格回读，并通过固定的 Ser/总体测试门槛。任何门槛失败都会停止后续生成。

## 冻结与重跑范围

结构门定义固定为：一次全复合物对齐后，global complex CA RMSD `< 3 Å`，
并且完整末链环肽在最佳正向循环移位下 CA RMSD `< 3 Å`；不允许肽链二次拟合。

已通过并冻结 7 个 T=0.5 靶点：

```text
1SFI  3AV9  3P8F  3WNE  3ZGC  4K1E  4KEL
```

这 7 条已接受设计均不含小写 `s`，而且 Ser-only checkpoint 修改不会改变
它们使用的任何专家。它们不会重新生成，也不会重新交结构。

仅重跑 10 个当前未通过结构门的靶点：

```text
3AVA  3AVB  3AVF  3AVG  3AVH
3AVI  3AVJ  3AVK  3AVM  3AVN
```

固定 5 个 seed，共生成 11,500 条原始候选；按未改动的 base likelihood、
纠正后的甲基位点置信度、跨 seed 稳定性和序列多样性，最多选 150 个结构
任务。若本地还保留先前的 `methylated_new_candidates.csv`，完全相同的
`target + design_seq` 会被排除，不重复交付。

旧 T=0.5 结构失败不能归因于这个 Ser 标签问题；本次修复只解释并纠正
新批次异常集中的 `s` 输出。结构结果仍须重新计算后才能判断是否通过。

## 一键运行

Windows（默认寻找当前环境或名为 `wain` 的 Conda 环境）：

```powershell
git pull
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_recovery.ps1
```

显式指定 Python：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_recovery.ps1 `
  -Python "E:\path\to\wain\python.exe"
```

若旧的 1,333 条交接表不在默认输出目录，可显式传入，完全相同的序列会被
排除：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_recovery.ps1 `
  -PriorHandoffCsv "E:\path\to\methylated_new_candidates.csv"
```

AutoDL/Linux：

```bash
git pull
bash run_serine_qc_recovery.sh --python python
```

对应的 Linux 参数是 `--prior-handoff-csv /path/to/methylated_new_candidates.csv`。

正常流程需要 CUDA。仅调试时才可显式传 `-AllowCpu` 或 `--allow-cpu`。
输出目录若已有部分生成结果，脚本会停止；确认要重跑同一隔离目录时传
`-Force` 或 `--force`。

## 输出与真实先后顺序

一键脚本依次执行：

1. 从固定旧提交和 751 个 PDB 重建 train/test 标签；
2. 只重训 canonical Ser expert 并执行隔离质量门；
3. 只为 10 个失败靶点生成新 T=0.5 候选；
4. 生成给尚哥的结构任务表。

交付文件：

```text
paper_clean_v28_outputs/serine_qc_retrain/handoff/structure_tasks_for_shangge.csv
paper_clean_v28_outputs/serine_qc_retrain/handoff/structure_tasks_for_shangge.fasta
paper_clean_v28_outputs/serine_qc_retrain/handoff/selection_manifest.json
```

本流程强制 `STRUCTURE_FIRST_THEN_PERMEABILITY`。结构返回前不会生成
`permeability_input.csv`，也不会运行透膜模型。透膜性只能在返回结构通过
同一结构门之后另行筛选。
