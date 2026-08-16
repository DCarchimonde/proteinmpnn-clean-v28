# Ser 来源质控与循环起点不变性恢复（V8）

## 先说结论

V8 取代 V7 作为当前恢复流程。**不要重训，也不要运行 V6 的 `-Force` 或
`-ResumeQuota`。** canonical、已通过的 V6/V7 checkpoint 与 representation
audit，以及 V6 的 31,500 条自然序列池，全部作为 hash 固定的只读输入。

V7 的 Ser 来源修复本身是对的，但“只改了 Ser 标签”不等于“部署时另外 19 个
expert 应退回 canonical”。全部 expert 都要在新的循环表示下部署；V6 对非 Ser
heads 的循环表示训练没有受到 Ser 标签梯度污染，因为每个天然母体残基选择独立
的线性 head。V7 把 19 个非 Ser heads 退回 canonical，实际形成训练/部署表示不
匹配。冻结的 1,505 位点成对审计给出了直接证据：

| 冻结阈值 `>0.6` | V6 | V7 | 变化 |
| --- | ---: | ---: | ---: |
| Recall | 0.8046 | 0.5096 | −0.2950 |
| TP / FN | 210 / 51 | 133 / 128 | 少 77 个 TP |
| FP / TN | 35 / 1209 | 14 / 1230 | 少 21 个 FP |

Ser 子集的阈值混淆未变；77 个新增 FN 全是非 Ser。因此这不是空 batch，也不是
阈值偶然波动，而是 19 个 non-Ser heads 被明确回滚的结果。此前 Recall 下限
`0.40` 过宽，不能阻止这种明显退化。该 151-record 集在 V3/V6/V7 中已被多次
查看，只能称为**冻结成对内部审计**，不能包装成新的盲测；论文最终主张仍需新
outer split 或真正 blind set。

V8 不训练、不平均权重、不调阈值；来源规则是在 V6/V7 成对审计暴露问题后提出、
在组合并评估 V8 前冻结的。因此它是 post-hoc 内部恢复候选，不是新的盲测模型：

- shared trunk、embedding、decoder 与 base head：canonical clean V28；
- 19 个 non-Ser experts：循环表示训练后的 V6；
- Ser expert：来源修复后的 V7；
- 每个非 Ser 位点概率必须与 V6 一致，每个 Ser 位点概率必须与 V7 一致；
- Recall 与 F1 在冻结 `>0.6` 口径下必须不劣于 V6，Ser AUC 必须不劣于 V6；
- 在通过模型与表示审计后，重标注只读的 31,500-row V6 pool；
- 只对实际缺失的 3WNE/3ZGC 做固定预算、可复现的定向搜索；长度 6/7 的历史与
  native controls 无论是否缺靶都必须复算，control 永远不能进入释放候选；
- 最终必须 17/17、无正式弃权；不降 `>0.6` 阈值；搜索不能修改模型指标；
- 只生成人工复核 ZIP，不生成尚哥 handoff，也不生成 permeability input。

一次运行（已 PASS 的 V8 stage 会按 manifest/hash 复用；普通 partial 目录原样
保留并停止，只有带配置 hash 的定向搜索 checkpoint 可以继续）：

```powershell
cd E:\ProteinMPNN_work\proteinmpnn-clean-v28
git fetch origin
git switch fix/serine-provenance-retrain-2026
git pull --ff-only origin fix/serine-provenance-retrain-2026
python -m unittest discover -s tests -p "test_*.py"
powershell -ExecutionPolicy Bypass -File .\paper_clean_v28\run_serine_qc_source_scoped_hybrid_v8.ps1
```

成功终端必须同时显示：

```text
V8 ALL AUTOMATED GATES PASSED; MANUAL SCIENTIFIC REVIEW IS NEXT
Final coverage:       17/17; no formal abstention
Shang-ge handoff:      NOT CREATED
```

人工复核文件是：

```text
paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/
  serine_qc_source_scoped_hybrid_v8_review_bundle.zip
```

ZIP 同时保留既有 V7 `15/17` 失败诊断、V8 模型三方比较、表示审计、V8 baseline
真实覆盖、长度 6/7 controls、搜索 trace/压缩 ledger/checkpoint/候选、最终三审
以及 before/after SHA-256。任一关失败时脚本退出非零并保留诊断；不会打印最终
绿色成功，也不会创建任何结构交接或透膜输入。

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
`KMP_DUPLICATE_LIB_OK=TRUE`。历史 V6 配额恢复命令为：

```powershell
python -m unittest discover -s tests -p "test_*.py"
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_cyclic_representation_v6.ps1 -ResumeQuota
```

已经完成一次 V6 后，仅重新审计和打包可运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_serine_qc_cyclic_representation_v6.ps1 -ReviewOnly
```

V6 人工复核通过前，结构交接和透膜性步骤都保持阻断。
