# Ser 来源质控与循环起点不变性恢复（V7）

## 先说结论

V7 取代 V6。**不要再运行 V6 的 `-Force` 或 `-ResumeQuota`，也不要把 3ZGC 的
`MODEL_ABSTAINS` 当最终科学结论。** V6 的 31,500 条自然序列和 base-head
采样统计保留为只读输入；V7 不再抽样，而是修正模型训练范围后统一重新标注。

根因是训练范围扩大错了：PDB 来源修复只把普通 `ATOM-SER` 的错误小写 `s`
改回天然 `S`，没有改变 R、G、L 等另外 19 个专家的标签；V3/V6 却重训了全部
20 个 expert heads。3ZGC 在 V6 中 13,000 次抽样零产出（最高概率约 0.195）
不是“种子不够”，而是非 Ser 专家也被不必要地改写了。历史上 3ZGC 的结构
通过候选 `rEGGQNR` 和 3WNE 的 `GrKWNC` 都依赖 R 专家，这与退化方向一致。

V7 的硬约束是：

- 从 canonical `frankenstein_v28.pt` 开始，只训练 Ser expert 的 weight/bias；
- 共享 trunk、base head 和其余 19 个 experts 必须逐张量 SHA-256 不变；
- 独立 test 上所有非 Ser 概率必须与 parent **精确相等**；
- 继续使用“所有循环起点 × 所有 decoder order，再映射回物理残基”的口径；
- 直接重标注已审计的 31,500 条 V6 自然序列，不重新生成、不降低 `>0.6`
  阈值、不继承 V6 的 sampling-path expert 概率；
- 不允许正式弃权来换 PASS：17 个靶点都至少要有 1 个新颖甲基候选；
- 只生成人工复核包，不生成尚哥 handoff；结构返回前不跑透膜性。

一次运行（脚本会自动复用已经 PASS 的阶段，避免重复长跑）：

```powershell
cd E:\ProteinMPNN_work\proteinmpnn-clean-v28
git fetch origin
git switch fix/serine-provenance-retrain-2026
git pull --ff-only origin fix/serine-provenance-retrain-2026
python -m unittest discover -s tests -p "test_*.py"
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_serine_only_cyclic_v7.ps1
```

成功终端必须同时显示：

```text
V7 ALL AUTOMATED GATES PASSED; MANUAL SCIENTIFIC REVIEW IS NEXT
Target coverage:       17/17; no formal abstention
Shang-ge handoff:      NOT CREATED
```

人工复核文件是：

```text
paper_clean_v28_outputs/serine_qc_serine_only_cyclic_v7/
  serine_qc_serine_only_cyclic_v7_review_bundle.zip
```

任一关失败时，脚本退出非零并保留诊断；不会生成 ZIP，也不会打印最终绿色成功。
`-ReviewOnly` 只适用于所有 V7 产物已经 PASS 后的快速重审和重新打包。

## V5/V6 历史问题与证据（保留用于复现，不再作为当前运行说明）

V5 已撤回。不要发送下面任何旧文件：

- V5 的 3,389 条 `methylated_new_candidates.csv`；
- V5 选出的 150 条结构任务；
- 两个 V5 尚哥交接 ZIP。

这不是说 3,389 条已经逐条证明“化学上都错”，而是它们共用的上游甲基位点
判定存在未排除的固定数组位置偏差，整批因而没有达到可交付证据标准。V6 会
重新训练、重新判分并重新生成全部 17 个靶点；不会直接删除第 7 位，也不会把
旧结果重新排个表冒充修复。

## 3AV9 到底错在哪里

旧冻结表把 3AV9 的 `QICGRGRG` 算进了“已通过 7 条”，因为它的两项结构指标
都小于 3 Å。但该序列：

```text
design_seq = QICGRGRG
old_methyl_positions_1based = []
```

也就是一个甲基 token 都没有。它只满足“天然序列结构相似”，不满足“甲基化
候选化合物”这个前提。旧 bridge 只检查两项 RMSD，没有检查
`design_methyl_count > 0`，所以把两个不同问题混在了一起。

正确的逐级门是：

1. 候选必须至少含一个由最终模型判定的甲基位点；
2. 必须通过历史 4,115 条、先前 1,333 条及天然化序列去重；
3. 之后才送结构；
4. 返回结构必须同时满足 global complex CA RMSD `< 3 Å` 和完整环肽 cyclic
   CA RMSD `< 3 Å`；
5. 结构通过后才进入透膜性比较。

因此 3AV9 必须重新生成。旧 7 条也不再被 grandfather：另外 6 条的 pre-QC
甲基注释与校正后 checkpoint 不一致，所以 V6 对 17 个靶点统一重训、重算，
不保留任何旧甲基位点。

## 为什么 V5 的 3,389 条全在第 7 位

S 标签错误和第 7 位问题是两层独立错误。

### 第一层：S 来源标签

旧预处理把天然残基表和 N-甲基残基表按 `residue_name` 直接合并，两张表都含
`SER`，导致普通 `ATOM-SER` 被覆盖成小写 `s`。现在按 PDB record type 和
`CN` 原子重建：

- `ATOM-SER` 一定是天然 `S`；
- 带 `CN` 的 `HETATM-5JP` 是 N-甲基 `s`；
- 历史中一个带 `CN` 的 `HETATM-SER` 也确认为 `s`；
- 模糊记录直接报错，不能猜。

固定计数为：train 600 条，其中 `S=242, s=50`；独立 test 151 条，其中
`S=62, s=12`。小写标签在模型 forward 前全部转回天然母体，避免答案从序列
embedding 泄漏。

### 第二层：环肽数组起点

V3 的确重训了全部 20 个 expert heads，也把采样顺序传进了因果 mask；但它的
“cyclic order ensemble”只改变**解码先后顺序**。它没有一起轮换：

- 环肽天然序列；
- 每个残基的 N/CA/C/O 坐标；
- 从 0 开始的 `residue_idx`。

ProteinMPNN 看到的仍是一个有首尾边界的线性数组。因此，“每个位点都经历过
各种解码深度”并不等于“每个物理残基都经历过各种数组位置”。V5 的 7,791
条唯一天然序列中，4,347 条没有甲基，3,444 条有甲基且全部落在数组第 7 位；
再排除历史/先前重复后剩 3,389 条。这种整批结果必须按上游表示偏差处理，后面
再做三遍 CSV 审计或结构来源解释都不能把它变成有效结果。

“以前通过的化合物不全在第 7 位”并不反驳这一点：那些序列来自不同的 pre-QC
模型、标签和生成口径，其中 3AV9 甚至根本没有甲基。旧位点不能拿来证明 V5
这次全第 7 位是正常的；同样，也不能仅凭旧位点不同就认定旧位点正确。

## V6 具体怎样修

V6 从仓库根目录的 canonical `frankenstein_v28.pt` 重新训练全部 20 个 expert
heads，共享 ProteinMPNN trunk、sequence embedding、decoder 和天然氨基酸
base head 保持逐字节冻结。

训练时，对每条环肽枚举所有等价循环起点；序列、甲基标签和 N/CA/C/O 坐标
一起平移，并把 `residue_idx` 重置为 `0..L-1`。训练 epoch 仍轮换因果解码
顺序，所以每个真实物理残基会覆盖所有数组位置和相对解码深度。验证、独立
test 和最终生成都使用同一口径：

```text
每个循环起点 × 每个解码顺序
        ↓
把概率逆映射回原始物理残基
        ↓
对同一物理残基求平均
        ↓
严格 probability > 0.6 才写小写甲基 token
```

生成前有独立 test 硬门：151 条 test 未参与 train/validation 拆分和 epoch
选择；AUC、precision、recall、FPR、Ser 和 Pro 门任一失败，流程立即停止，不会
开始 17 靶点生成。checkpoint、test、计划文件的 SHA-256 会绑定在审计报告中，
不能拿别的模型报告冒充。

V6 对全部 17 个靶点计划生成 19,500 条 raw draws。最终
`methylated_new_candidates.csv` 保存**所有**满足甲基硬门且去重后的新候选，
不是只保留 150 条；最终多少条由新模型真实结果决定，不能预先承诺仍为 3,389。
计划里的 245 只是将来人工放行后用于检查每个靶点是否有足够结构覆盖的最低
配额总和，本次命令不会生成尚哥交接包，也不会把候选截成 245 条。

## 3ZGC 的固定预算耗尽不是“再换种子就会好”

完整 V6 运行中，3ZGC 的 5 个初始种子共 1,000 条，加上 12 个互不重叠的
reserve seeds 共 12,000 条，合计 13,000 条，在冻结的循环表示平均和严格
`>0.6` 阈值下仍为 0 个可释放的新甲基候选。这个结果不能用下面任何办法“补齐”：

- 继续一轮又一轮盲抽新种子；
- 把阈值从 0.6 降低；
- 恢复 V6 以前依赖数组起点的旧甲基标注；
- 伪造 3ZGC 已满足 `10` 条结构配额。

这里的 `10` 是结构筛查覆盖目标，不是要求模型必须给出阳性的真实标签。固定
12,000 条补采样预算用尽且独立复算仍为 0 后，科学上正确的终态是
`MODEL_ABSTAINS`：明确记录 3ZGC 无可释放候选，不为它创建结构任务；其余 16 个
靶点按各自冻结配额继续人工复核。对当前这批结果，这等价于 17 个靶点中 16 个
有候选覆盖、有效结构复核计划由 245 降为 235；不是宣称“17/17 全部达标”。

该终态只更新 summary、target manifest、generation manifest 和独立审计文件。
`all_candidates.csv`、`unique_candidates.csv`、
`methylated_new_candidates.csv` 的 SHA-256 在前后必须完全相同，否则立即失败。

## Windows 一键运行（历史 V6，仅供复现，不要作为当前命令执行）

必须在真正的仓库目录运行，而不是上一层 `E:\ProteinMPNN_work`：

```powershell
cd E:\ProteinMPNN_work\proteinmpnn-clean-v28
git fetch origin
git switch fix/serine-provenance-retrain-2026
git pull --ff-only origin fix/serine-provenance-retrain-2026
git rev-parse --short HEAD
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_cyclic_representation_v6.ps1 -Force
```

脚本会依次执行：

1. 检查 CUDA、canonical checkpoint、校正后的 train/test 和历史去重文件；
2. 使用所有循环起点重训 20 个 expert heads；
3. 跑 151 条独立 test 与 17 个 native target 的循环表示审计；
4. 只有前述门全部 PASS 才重新生成全部 17 个靶点；
5. 对生成结果做独立三遍审计；
6. 只打人工复核包，明确不创建尚哥 handoff。

`-Force` 只清理并重建隔离目录：

```text
paper_clean_v28_outputs/serine_qc_cyclic_representation_v6/
```

它不会删除 canonical checkpoint、V3/V4/V5 原始结果或历史 CSV。若显存不足，
可降低批大小，例如：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_cyclic_representation_v6.ps1 `
  -Force -TrainingBatchSize 4 -AuditBatchSize 4 -GenerationBatchSize 4
```

## 历史 V6 正常完成后看什么

终端最后应显示：

```text
V6 AUTOMATED GATES PASSED; MANUAL REVIEW IS STILL REQUIRED
Shang-ge handoff:     NOT CREATED
```

然后上传这个文件进行人工复核：

```text
paper_clean_v28_outputs/serine_qc_cyclic_representation_v6/
  serine_qc_cyclic_representation_v6_review_bundle.zip
```

关键证据也会单独保留：

```text
model/expert_heads_retrain_manifest.json
representation_audit/cyclic_representation_audit.json
generation/generation_manifest.json
generation/methylated_new_candidates.csv
triple_audit/three_pass_generation_audit.json
```

如果流程报错，不要继续运行 V5 的 `-ReviewOnly -ReleaseHandoff`，也不要手工挑
CSV。把完整终端日志和当时已有的 V6 review ZIP（若已生成）发回来定位。

如果训练、151 条 test、循环表示审计和 19,500 条生成都已完成，唯一失败项是
`every_target_meets_pre_structure_candidate_quota`，不要再次使用 `-Force`。保留
现有 V6 模型与全部候选，只对缺额靶点使用独立 reserve seeds 补采样：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_cyclic_representation_v6.ps1 -ResumeQuota
```

该模式会先用不导入 Torch 的 finalizer 核对 checkpoint、plan、representation
audit、全部候选和固定预算。若已经保留 12,000 条补采样且目标仍是零产出，它
会直接登记正式模型弃权，跳过 GPU；若预算尚未用尽，才自动读取缺额靶点，每
200 条重新计算一次冻结配额。12,000 是**每靶点累计总上限**，不是每次命令都
能再抽 12,000。阈值仍严格为 `>0.6`，原始 19,500 行会备份并逐行保留。之后
继续三遍独立审计和打人工复核 ZIP，不会重训、重跑已完成的 17 靶点，也不会
创建尚哥 handoff。`-ResumeQuota` 与 `-Force` 禁止同时使用。

Windows 全量单测中的两个 Torch 数值测试使用干净的 Python 子进程，避免已经
导入 NumPy 的测试进程再加载 `libomp.dll`/`libiomp5md.dll`。不要设置不安全的
`KMP_DUPLICATE_LIB_OK=TRUE`。当前恢复命令为：

```powershell
python -m unittest discover -s tests -p "test_*.py"
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_cyclic_representation_v6.ps1 -ResumeQuota
```

已经完成一次 V6 后，仅重新审计和打包可运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_cyclic_representation_v6.ps1 -ReviewOnly
```

V6 人工复核通过前，结构交接和透膜性步骤都保持阻断。
