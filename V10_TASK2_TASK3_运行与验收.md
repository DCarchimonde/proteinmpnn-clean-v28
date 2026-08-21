# V10：V8 修复、17×100 生成、单体重算与机器文件台账

## 先说清楚当前边界

这次交付的是一条新的 **V10 系统流程**，不是把旧 V8 候选重新过滤后换名：

1. 从原始 `frankenstein_v28.pt` 重新训练甲基 expert heads；
2. 用完整循环表示网格修复 V8 的甲基释放错误；
3. 对新生成池做严格甲基化、去重、base plausibility 和位点塌缩审计；
4. 用旧六复合物 476 条结构标签校准一个低容量、留一靶点外验证的 RMSD 优先排序器；
5. 冻结后从 17 个复合物各选 100 条，共 1,700 条；
6. 用 batch size 1 独立重放全部 1,700 条；
7. 用同一个新 checkpoint 重跑 151 个单体的序列、甲基化和循环表示指标；
8. 把 AutoDL 结果打包给 Windows 本地，先审计旧 PDB 能否逐条复用，再重算单体结构、能量和已有通透性指标。

当前仓库只包含代码和冻结输入，**不包含已经在 GPU 上产生的新 checkpoint、1,700 条成品或新结构**。只有 AutoDL 实跑全部硬门通过后，才会出现这些结果。RMSD 优先分数是结构预测前的排序依据，不是实测 RMSD；真正的 `<3 Å` / `<5 Å` 改善必须等尚哥返回结构后按同一协议复算。

## Task 2：V8 到底错在哪里

### 已经被证据确定的问题

旧 V8 有明确的训练—部署—释放合同错误：

- 训练时每个循环起点只优化一个 decoder order，部署时却平均全部 decoder orders；
- 正式甲基化释放依据是跨循环起点的平均概率，最差起点及跨阈值分歧只作诊断；
- 位点高度集中只记录、不阻断；
- 6,964 条旧候选中仅 1,768 条在全部循环表示下保持相同的 `>0.6` 硬判，5,196 条存在阈值分歧；
- 旧 3AV 候选 4,080/4,080 的最高甲基位点都在物理第 7 位。

这些证据足以判定 V8 **不能发布**。但是，现有材料不能证明“训练/部署不一致”是 3AV 第 7 位偏好的唯一原因，因为部分 native control 也偏好该位点。因此 V10 修的是可证明有错的释放合同，并用针对 3AV 的塌缩硬门阻止异常批次发布，而不是编造唯一生物学根因。

### V10 的甲基化修复

- 训练使用完整“物理循环起点 × decoder order”的 `L×L` 网格；
- 正样本按跨起点最小概率、负样本按跨起点最大概率优化，并加入表示一致性损失；
- 所有正式甲基标记统一为 `round(probability_min, 8) > 0.6`；等于 0.6 不通过；
- 每条候选必须至少有一个小写甲基化 token，且所有循环表示的硬判完全一致；
- 17 个靶点采用同一新 checkpoint，旧 V8 的 6,964 条候选不得流入新结果；
- `v9_inputs/methylated_new_candidates.csv` 的 1,333 条仅用于“历史序列排除”，不是新候选，也不是旧 6,964 条 V8 池；
- 3AV 的第 7 位集中仍为硬失败；六个非 3AV 靶点仅对旧结构成功中已有支持的主位点采用证据豁免，但残基类型塌缩仍为硬失败。

## 六复合物 RMSD 为什么低，以及 V10 做了什么

旧 T=0.5、严格甲基化保留后的 476 条中：

| 指标 | `<3 Å` | `<5 Å` |
|---|---:|---:|
| 全复合物 global RMSD | 469/476（98.53%） | 476/476（100.00%） |
| global 与环肽 pose 联合通过 | 16/476（3.36%） | 101/476（21.22%） |
| 按全部 544 条原始生成计算 | 16/544（2.94%） | 101/544（18.57%） |
| 六靶点等权 macro 基线 | 4.37% | 23.08% |

所以主要瓶颈是“一次全复合物对齐后”的环肽结合姿态，不是受体整体结构。单纯修甲基 expert heads 不会自动提高 RMSD，因为 shared trunk、decoder 和天然氨基酸 base head 都被冻结。

V10 因而新增一个独立的结构优先层：

- 冻结开发数据：`v10_inputs/six_non3av_t05_joint_rmsd_476.csv`；
- 主目标：joint `<5 Å`，共有 101 个阳性；
- `<3 Å` 只有 16 个阳性，仅作为次要描述终点，不用于夸大模型能力；
- `<3 Å` 模型只输出描述性分数，不参与候选排序、tie-break 或放行；正式排序只使用 joint `<5 Å`；
- 排序器是确定性、低容量、带 L2 正则的序列组成逻辑回归；不使用目标 ID，避免直接记住六个靶点；
- 外层按 target 做 leave-one-target-out，避免同一靶点的行随机泄漏；
- 回顾性外层 AUC 为 0.5827；预注册 top quartile 为 40/120（33.33%）joint `<5 Å`，高于全体 101/476（21.22%）。

这说明旧数据里有弱但可复现的富集信号，所以它可以用于**送结构前优先级**。它不等于新 1,700 条已经达到 33.33%，也不证明 11 个 3AV 靶点的结构泛化。若该外层验证门不通过，V10 会硬停，不生成可交付 1,700 条。

### 尚哥返回结构后的公平验收

1,700 条名单必须在看到结构前冻结。结构返回后按旧论文完全相同的口径计算：

- 同一结构预测器版本；
- 一次全复合物 CA 对齐；
- global RMSD 与同一坐标系下的 final-chain best-forward cyclic-shift RMSD；
- 只允许正向循环移位，不允许反向，不允许对环肽二次拟合；
- 分别报告 global、cyclic 和 joint；
- 六个重叠靶点同时报告 micro、等权 macro、每靶点比例、Wilson 区间和以靶点为单位的 bootstrap 差值；
- 除“选中 100 条”的条件率外，还必须报告“成功数 ÷ 达到配额前全部 raw draws”的端到端率。

旧的公平比较基线固定为：micro `3.36% / 21.22%`，macro `4.37% / 23.08%`，raw 端到端 `2.94% / 18.57%`。在新结构回来前，任何文件都不得写“RMSD 已提高”。

## Task 3：17×100 的生成与放行合同

流程固定 `T=0.5`。每个靶点完整记录：

`raw draws → exact unique → 严格稳定且含甲基 → exact cyclic-base gate → 全局 forward-cyclic unique → RMSD-priority 前四分位 → selected 100`

初始生成硬目标为每靶点至少 500 条严格稳定的新候选；跨靶点与循环去重、base gate 后仍必须每靶点至少保留 400 条，最终 100 条只能来自该池按 joint `<5 Å` 分数排序的前 25%。任一靶点不足都硬停，不补空行、不放宽阈值。

最终放行要求：

- 恰好 17 个靶点，每个 100 条，总计 1,700；
- 每条都有至少一个由 `round(min_probability, 8) > 0.6` 得到的小写甲基 token；
- 跨循环表示硬判分歧为 0；
- marked sequence、natural sequence、forward-cyclic natural sequence 和历史池去重全部通过；
- exact receptor-visible `L×L` cyclic-base 打分通过；
- 3AV 第 7 位异常集中、无证据支持的物理位点集中、单残基类型集中均按冻结策略审计；
- RMSD 优先分数与冻结模型逐行重算一致；
- 全部 1,700 条用 batch size 1 独立重放；
- manifest 与每个成品文件执行“路径 → SHA-256”绑定复核，不能用另一个文件里碰巧相同的摘要蒙混；
- 最终 manifest 为 `PASS`，所有硬门均为真。

这里的“发生甲基化”是**模型预测的稳定甲基化标记**，不是实验化学反应已经发生。最终文件仍需人工检查后才发送给尚哥。

## 单体重跑与 751 个公司结构

751 个单体 PDB 是公司内部采用 Rosetta 生成的计算理论结构：600 个用于训练，151 个用于内部测试/开发审计。这里不再纠结“Baker 33”来源，也不把公司数据当异常；只保留两个科学边界：它们不是实验结构，而且 151 条已被多个版本用于调试，因此结果叫内部配对审计，不叫新的独立盲测。

AutoDL 用新 checkpoint 重算：

- 151 个单体、1,505 个真实位点；
- base amino-acid recovery；
- known-sequence 与 end-to-end ROC-AUC、PR-AUC；
- 严格 0.6 下 TP/TN/FP/FN、Accuracy、Precision、Recall、F1、FPR 和预测阳性率；
- 扩展 token recovery、甲基残基精确恢复；
- 20 种残基分层，尤其 Ser 与 Pro；
- 循环表示 mean/min/max/span/std、跨起点硬判分歧；
- 151 条中至少一个稳定甲基位点的样本比例和甲基数分布；
- 17 个 native complex 全阴性位点控制；
- 与原始 V28 的 1,505 位逐位配对比较。

因为 base head 和 trunk 冻结，AutoDL 会在相同 `seed=0`、相同 batch size 和同一测试文件下，分别重跑原始 `frankenstein_v28.pt` 与 V10；两者的 1,505 个 base argmax 必须 **1,505/1,505 完全一致**，不一致即硬失败。历史修正表仍用于原始 V28 指标的配对比较，但不再用不同随机 decoder 轨迹证明逐位冻结。冻结原始 V28 的历史基线为：RAA 16.08%，known AUC 0.9003，end-to-end AUC 0.9082；严格 0.6 的 end-to-end 混淆矩阵为 TP=111、TN=1230、FP=14、FN=150。

单体结构、能量和已有通透性数据在 Windows 本地重算。旧 PDB 只允许按以下规则复用：

- variant 2 参考天然化结构：151/151 文件名序列、链长和 CA 完整性都匹配才整体授权；
- variant 4 V10 end-to-end 天然化结构：同样必须 151/151 全部匹配；
- variant 3 显式甲基结构：只有大小写敏感 marked sequence 完全相同的样本才逐条授权，不能整体默认复用；
- 审计只读，不删除、不重命名、不改写任何 PDB；审计后库存哈希变化会阻止下游计算。

若 variant 2/4 审计失败，就不能拿旧 PDB 冒充 V10，需要只补跑未匹配的结构后再计算。

## GitHub / AutoDL / Windows 本地文件台账

| 位置 | 当前应有内容 | 当前不应假定存在的内容 |
|---|---|---|
| GitHub 分支 `fix/v10-rmsd-aware-1700-monomer` | V10 代码、测试、原始 `frankenstein_v28.pt`、冻结 train/test/native、476 条 RMSD 开发表、1,505 位原始 V28 修正表、位点策略 | 新 checkpoint、实际 1,700 条、任何新 PDB；输出目录被 `.gitignore` 排除 |
| AutoDL 的仓库副本 | `git pull` 后运行 GPU 训练、审计、生成、打分、筛选、单体序列重算 | Windows 旧 HighFold PDB、Windows PyMOL/PyRosetta 结果 |
| AutoDL 输出根 | `paper_clean_v28_outputs/rmsd_aware_v10_1700_monomer/` | 不能通过 GitHub `pull` 获得，必须从该 AutoDL 实例下载 |
| Windows 本地仓库 | 旧复合物 PDB 的精确目录为 `E:\ProteinMPNN_work\proteinmpnn-clean-v28\raw_external\pdb_highfold_temperature\pdb_highfold4_t05\`；旧单体 PDB 为 `E:\ProteinMPNN_work\proteinmpnn-clean-v28\raw_external\pdb_permeability_v20260624\pdb_monomer\pdb_monomer_hf4\` | 不会自动拥有 AutoDL 的 V10 manifest 和 1,700 条；V10 新 1,700 条结构也不在旧复合物目录，必须先下载交接包并另等尚哥返回新结构 |
| Windows 历史结果目录 | `E:\ProteinMPNN_work\proteinmpnn-clean-v28\paper_clean_v28_outputs\temperature_0.5_best17\` 中是此前 17 复合物/151 单体的旧 V28 结构计算结果 | 不能把该目录中的旧结果改名当作 V10 单体或 V10 新 1,700 条结果；V10 Windows 重算写入独立的 `rmsd_aware_v10_1700_monomer\windows_structure_recalculation\` |
| 尚哥/结构预测端 | 收到人工确认后的 1,700 条 FASTA/极简表并返回结构 | 不参与 V10 checkpoint 训练或 AutoDL 本地序列审计 |

历史路径 `/root/autodl-tmp/proteinmpnn-clean-v28-v8-autodl` 只是在旧记录中出现，当前会话无法直接查看该远端实例，所以不得写成“已确认仍存在”。在 AutoDL 实际仓库根目录运行命令即可。

## 发布前已做的本地验证

- 218 项自动回归通过，3 项需要当前环境不具备的 PyTorch/CUDA，因而按测试合同跳过；
- 全部变更 Python 文件通过 `py_compile`，Linux 一键入口通过 `bash -n`，Git 差异无空白或格式错误；
- 非 GPU 合成链验证了 476 条 RMSD 校准、17×400 候选到17×100筛选、1,700 条独立排序回放、1,505 位单体收口及所有篡改阻断门；
- 当前 Linux 环境没有真实 CUDA、Windows PowerShell 和公司本地 PDB，所以它不能替代 AutoDL GPU 数值实跑与 Windows PDB 实跑。

## AutoDL：唯一推荐入口

在 **AutoDL 的仓库根目录** 执行一条命令：

```bash
git fetch origin fix/v10-rmsd-aware-1700-monomer && (git switch fix/v10-rmsd-aware-1700-monomer || git switch -c fix/v10-rmsd-aware-1700-monomer --track origin/fix/v10-rmsd-aware-1700-monomer) && git pull --ff-only && bash run_v10_rmsd_aware_1700_and_monomer.sh
```

需要 Python ≥ 3.10、CUDA 版 PyTorch 和可用 GPU。默认不覆盖半成品；某阶段存在未通过或无法校验的旧文件时会要求换新的 `V10_OUTPUT_ROOT`，不会删除科学证据。

全部通过后最重要的 AutoDL 文件是：

```text
paper_clean_v28_outputs/rmsd_aware_v10_1700_monomer/
├── final_v10_handoff/
│   ├── 1700_详细审计.csv
│   ├── 1700_给尚哥_极简.csv
│   ├── 1700_给尚哥_结构输入.fasta
│   ├── v10_rmsd_priority_replay.csv
│   └── v10_1700_final_manifest.json
├── monomer_final/
│   ├── monomer_v10_position_comparison_1505.csv
│   ├── monomer_v10_metrics.csv
│   ├── monomer_v10_threshold_curves.csv
│   ├── monomer_v10_by_residue.csv
│   ├── monomer_v10_by_company_rosetta_panel.csv
│   ├── monomer_v10_per_sample.csv
│   ├── monomer_v10_paired_original_v28_comparison.csv
│   ├── native17_v10_all_negative_control.csv
│   ├── monomer_v10_design_manifest_151.csv
│   └── monomer_v10_manifest.json
├── monomer_parent_sequence_eval/    # 原始 V28、seed=0 的配对冻结重跑
├── monomer_sequence_eval/           # V10、seed=0 的同协议重跑
├── prestructure_report/v10_prestructure_audit_cn.md
├── v10_autodl_to_windows_handoff.tar.gz
└── v10_autodl_to_windows_handoff.tar.gz.sha256
```

只有 `final_v10_handoff/v10_1700_final_manifest.json` 和 `monomer_final/monomer_v10_manifest.json` 都为 `PASS`，且中文预结构报告无人工异议，才进入“发给尚哥”步骤。

## Windows：下载 AutoDL 包后再运行

先把 AutoDL 的 `v10_autodl_to_windows_handoff.tar.gz` 和 `.sha256` 下载到 Windows，校验摘要，并解压到
`E:\ProteinMPNN_work\proteinmpnn-clean-v28\paper_clean_v28_outputs\rmsd_aware_v10_1700_monomer\`。不能只单独复制一个 design CSV。解压后下面两个文件必须保持在同一个 `monomer_final` 目录：

```text
monomer_final\monomer_v10_design_manifest_151.csv
monomer_final\monomer_v10_manifest.json
```

Windows 入口会重新验证 `monomer_v10_manifest.json` 的协议、总质量门、每一项 `quality_checks`，并要求其中 `artifacts.design_manifest.sha256` 与 Windows 上 CSV 的实际 bytes 完全一致；AutoDL 记录的原绝对路径不要求在 Windows 存在。旧 CSV、被改写的 CSV，或只有手工写成 `PASS` 的复用审计都不能进入结构重算。

然后在 Windows 仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_v10_windows_structure_recalculation.ps1
```

该入口会先运行只读 PDB 复用审计；只有 151 个 variant 2/4 全部授权后，才继续单体结构、置信度、已有通透性和 PyRosetta 能量重算。AutoDL 命令不读取 Windows PDB，Windows 命令也不重新训练模型。

## 出错时不要做的事

- 不要运行旧 `run_v8_*` 或把旧 6,964 条候选混入 V10；
- 不要把 1,333 条历史排除池当新生成结果；
- 不要为凑满 100 条而放宽 `>0.6`、跨起点一致性或去重门；
- 不要在看到结构后改动已经冻结的 1,700 条名单，再把结果称为前瞻验证；
- 不要把 RMSD 优先分数写成实测 RMSD；
- 不要在 AutoDL 上假定 Windows 的旧 PDB 已存在，也不要在 Windows 上用 `git pull` 寻找被忽略的 AutoDL 输出。
