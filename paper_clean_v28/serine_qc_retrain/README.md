# Ser 来源质控 + 解码顺序平衡恢复流程（v3）

本目录修复一个有明确 PDB 来源证据的标签问题，重新训练生产网络中的
完整专家模块，并修复训练、生成和最终甲基位点注释之间的解码顺序不一致。
昂贵的结构补跑仍限制在当前 T=0.5 结构门未通过的复合物。最终只产出一个
新的完整 checkpoint；共享主干和天然氨基酸 base head 仍逐字节冻结。

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

## 新确认的点位问题

S 来源修复和点位异常是两个独立问题。旧 V28 训练在训练态使用随机解码顺序；
此前的 Ser-QC 重训为了关闭 dropout 把模型保持在 `eval()`，却没有显式传入
解码顺序，因此专家头只见过固定顺序。生成端又随机选择下一个肽位点，但
模型内部仍使用固定因果掩码。外层采样顺序、模型可见上下文和专家头训练
上下文三者不一致。

这个错误在去掉小写 token 回灌后不再被旧泄漏掩盖，表现为新 872 条中
873 个甲基位点有 833 个落在第 7 位（95.42%），并且 11,500 条原始生成中
有 1,052 个重复天然序列组，其中 208 组得到不同甲基标注。该 872 条结果
因此明确撤回，不能交结构；不能通过删除第 7 位或重排表格补救。

v3 的闭环是：

- 专家头训练使用 epoch-indexed 循环解码顺序；至少 30 个 epoch，覆盖冻结的
  30-residue 肽长度上限内每一个相对解码深度；
- 天然序列采样的外层随机顺序被原样传入模型因果掩码；
- 完整天然序列生成后，用所有循环移位做确定性专家概率集成，每个位点恰好
  在每个相对深度出现一次；
- 同一 `target + naturalized sequence` 必须得到同一标注和同一概率；
- 全局或单靶点的点位、残基、采样步异常集中会直接阻断交付。

## 训练边界：完整专家模块，而不是只改 Ser

`02_retrain_canonical_expert_heads.py` 使用生产推理相同的完整 clean-V28
网络，同时更新 `experts[0]` 到 `experts[19]` 的 weight 和 bias。共享
ProteinMPNN 主干、序列 embedding、decoder 和天然氨基酸 base head 全部
逐字节冻结。

600 条纠正训练记录先按记录级确定性拆分为 development-train 和 validation；
原来的 151 条 test 不参与 epoch 选择或 early stopping，只在最终 checkpoint
确定后用于质量门和固定报告，不把结果反馈给训练。每个 forward 前，所有
小写甲基标签都转换成天然母体作为模型输入，避免答案通过 `W_s` 泄漏。
训练 forward 显式使用循环解码顺序；前 30 个 epoch 完成顺序覆盖之前，不允许
选择 checkpoint 或 early stop。validation、独立 test 和生产注释均使用同一
完整天然序列循环顺序集成口径。

19 个有正负训练支持的专家使用各自的 class weight；Pro 没有 `p` 正样本，
其专家只学习天然 P 的负类 veto，且生成器仍没有 `P -> p` 映射。

保存前后的 state hash 必须证明恰好 40 个 expert tensor 改变：

```text
experts.0.weight / experts.0.bias
...
experts.19.weight / experts.19.bias
```

任何非 expert tensor 改变都直接失败。保存后的 checkpoint 必须由生产 loader
严格回读，并通过固定的 validation、Ser、macro-AUC、总体
precision/recall/FPR 和无 `p` 假阳性门槛；失败时只保留 `.candidate.pt`
诊断文件，后续生成自动停止。阈值类晋级指标使用与生产生成完全相同的口径：
对每个循环解码顺序计算 `sigmoid(logit / 0.5)`，取确定性均值后执行严格
`> 0.6`。旧的固定顺序 evaluator 不再作为 v3 checkpoint 的晋级证据。

## 冻结与重跑范围

结构门定义固定为：一次全复合物对齐后，global complex CA RMSD `< 3 Å`，
并且完整末链环肽在最佳正向循环移位下 CA RMSD `< 3 Å`；不允许肽链二次拟合。

已通过并冻结 7 个 T=0.5 靶点：

```text
1SFI  3AV9  3P8F  3WNE  3ZGC  4K1E  4KEL
```

这 7 条已接受设计均不含小写 `s`。它们不会重新生成，也不会重新交结构；
新 checkpoint 训练完成后先运行一次零成本 bridge，用最终专家模块统一
重评分，但绝不覆盖这 7 条已经接受的序列或甲基位点；若最终模型不同意旧
注释，bridge 会把不一致写入溯源，而不是静默换成另一个化合物。HighFold
使用的天然化序列没有变化，所以既有 PDB 和结构门结果原样复用。
这 7 条的候选来源必须如实记为“pre-QC generation, audited by the final
checkpoint”，不能写成由新 checkpoint 重新采样；最终报告和后续生成只保留
一个生产 checkpoint。

仅重跑 10 个当前未通过结构门的靶点：

```text
3AVA  3AVB  3AVF  3AVG  3AVH
3AVI  3AVJ  3AVK  3AVM  3AVN
```

固定 5 个 seed，共生成 11,500 条原始候选；按未改动的 base likelihood、
纠正后的甲基位点置信度、跨 seed 稳定性和序列多样性，最多选 150 个结构
任务。旧 4,115 条和先前 1,333 条 `methylated_new_candidates.csv` 都作为
强制输入执行两层硬去重：`target + design_seq` 精确重复、以及
`target + naturalized_seq` 天然化重复均不允许进入新候选池或交接表。
1,333 条文件缺失或行数不是 1,333 时，流程会在训练前停止，不能靠
“新模型应该不会重复”来猜。

生成时小写 token 只记录最终甲基化注释；后续自回归步骤只接收其天然母体，
外层随机生成顺序同时传给模型因果掩码。天然序列完成后才进行循环顺序集成
复评；生成路径上的临时专家概率只留作审计，不决定最终小写位点。

旧 T=0.5 结构失败不能归因于 Ser 标签或解码顺序问题。新序列仍须重新生成、
通过结果审计并重新计算结构，才能判断是否通过；现有 872 条不能复用或仅
重新标注，因为它们的天然序列本身也是在不一致因果掩码下采样的。

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

旧的 1,333 条交接表是强制证据。默认路径不存在时必须显式传入；精确序列
和天然化序列重复都会被排除：

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

默认运行**不会**创建给尚哥的交接包。自动门通过后只生成：

```text
paper_clean_v28_outputs/serine_qc_order_balanced_v3/
  serine_qc_order_balanced_v3_review_bundle.zip
```

必须把该 review bundle 做第三遍人工科学复核；只有明确放行后，才可再次传
`-ReleaseHandoff`（Linux 为 `--release-handoff`）。不能为了提前拿到 CSV
绕过这一停点。

## 输出与真实先后顺序

一键脚本依次执行：

1. 从固定旧提交和 751 个 PDB 重建 train/test 标签；
2. 用顺序平衡协议重训 canonical 全部 20 个 expert heads；
3. 用循环顺序集成完成 validation 和 151 条独立 test 晋级门；
4. 对已通过的 7 个靶点做同口径 bridge，复用既有结构；
5. 只为 10 个失败靶点生成 11,500 条新 T=0.5 原始候选；
6. 生成器内执行第一轮结果门；
7. `04_triple_audit_generation.py` 独立重算完整性、结果分布和去重/配额三遍审计；
8. 停在人工复核，不自动生成给尚哥的表。

人工复核文件：

```text
paper_clean_v28_outputs/serine_qc_order_balanced_v3/serine_qc_order_balanced_v3_review_bundle.zip
paper_clean_v28_outputs/serine_qc_order_balanced_v3/model/expert_heads_retrain_manifest.json
paper_clean_v28_outputs/serine_qc_order_balanced_v3/generation/generation_manifest.json
paper_clean_v28_outputs/serine_qc_order_balanced_v3/triple_audit/three_pass_generation_audit.json
```

最终条数由重跑和审计决定，不预设为 872，也不为了凑数降低阈值或删除异常
位点。只有人工复核放行后才会在 `handoff/` 下生成简化结构任务表。

本流程强制 `STRUCTURE_FIRST_THEN_PERMEABILITY`。结构返回前不会生成
`permeability_input.csv`，也不会运行透膜模型。透膜性只能在返回结构通过
同一结构门之后另行筛选。
