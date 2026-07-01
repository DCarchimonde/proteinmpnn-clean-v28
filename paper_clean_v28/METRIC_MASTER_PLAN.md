# ProteinMPNN Clean v28 项目总控文档

本文件是 `proteinmpnn-clean-v28` 的实验、指标、数据和后续任务总控表。之后所有新脚本、新结果和论文表述都必须对齐本文件，避免把不同口径混在一起。

## 0. 核心原则

1. 不编造缺失数据。
2. 不把 confidence score 当作 energy。
3. 不把 oracle best-of-N 写成真实无监督筛选。
4. 不把 peptide self-superposed RMSD 写成 binding pose RMSD。
5. 不把没有正样本的数据集报告为 F1 / Recall / AUC。
6. 所有结构指标必须经过质量闸门。
7. 所有训练泄露结论必须有数据清单或审计报告支持。

## 1. 模型与代码框架

### 1.1 最终模型

- 模型 checkpoint：`frankenstein_v28.pt`
- clean evaluation 代码目录：`paper_clean_v28/`
- clean output 目录：`paper_clean_v28_outputs/`

### 1.2 clean evaluation 模块

| 文件 | 作用 | 当前状态 |
|---|---|---|
| `paper_clean_v28/01_eval_clean_model.py` | 单体 / 天然复合物短肽 clean evaluation | 已用 |
| `paper_clean_v28/02_score_generated_fastas.py` | 评价不同温度生成 FASTA | 已用，主口径为 `auto_single` |
| `paper_clean_v28/03_prepare_structure_manifest.py` | 准备 best85 结构预测清单 | 已用 |
| `paper_clean_v28/structure_metrics/01_extract_highfold_scores.py` | 提取 HighFold PDB 分数并匹配 all_designs / best85 | 已用 |
| `paper_clean_v28/structure_metrics/02_audit_best85_structure_coverage.py` | 审计 best85 是否有 PDB | 已用 |
| `paper_clean_v28/structure_metrics/03_audit_complex_chain_mapping.py` | 审计 predicted/native chain mapping | 已用 |
| `paper_clean_v28/structure_metrics/04_compute_complex_rmsd.py` | 计算复合物 RMSD | 已用 |
| `paper_clean_v28/structure_metrics/05_validate_complex_structure_metrics.py` | 结构质量闸门 | 已用，PASS |
| `paper_clean_v28/06_available_data_leakage_audit.py` | 当前仓库可用数据泄露审计 | 待运行 |

## 2. 数据集状态

| 数据集 | 路径 | 用途 | 当前状态 |
|---|---|---|---|
| 单体测试集 | `nmethyl_data/test_set/test.jsonl` | methylation classifier / monomer clean evaluation | 已测 |
| 17 个复合物天然结构 | `17_complexes_native.jsonl` | 天然复合物短肽 clean evaluation / native reference structure | 已测 |
| 不同温度生成 FASTA | `all_temperature_results/` | 复合物生成序列评价 | 已测 |
| best85 结构预测清单 | `paper_clean_v28_outputs/af3_manifest.csv` | 结构预测任务清单 | 已生成 |
| HighFold PDB | `raw_external/pdb_highfold_temperature/` | 结构指标 | 本地存在，不进 GitHub |
| 训练集 | 当前仓库无 train 文件 | 完整 train/test leakage audit | 未能完成，需要补训练集清单 |
| permeability full result | 待师兄返回 | permeability 指标 | 未完成 |
| energy result | 未提供 | Success / Stability lower energy than native | 未完成 |
| TM-score / diversity result | 未提供 | structural diversity = 1 - TMScore | 未完成 |

## 3. 已完成指标

### 3.1 单体测试集 clean evaluation

输出目录：`paper_clean_v28_outputs/monomer_clean/`

主口径：`strict_naturalized_input`

| 指标 | 数值 | 可否报告 | 备注 |
|---|---:|---|---|
| clean evaluation positions | 1505 | 可以 | padding 已排除 |
| base recovery / RAA | 16.08% | 可以 | strict naturalized input |
| known-sequence methylation AUC | 0.9562 | 可以 | 已知序列条件 |
| known-sequence methylation F1 | 80.14% | 可以 | threshold = 0.3 |
| end-to-end methylation AUC | 0.8352 | 可以 | base prediction + methylation |
| end-to-end methylation F1 | 60.91% | 可以 | threshold = 0.3 |

### 3.2 复合物天然短肽 clean evaluation

输出目录：`paper_clean_v28_outputs/complex_native_clean/`

主口径：`strict_naturalized_input` + `auto_single`

| 指标 | 数值 | 可否报告 | 备注 |
|---|---:|---|---|
| clean evaluation positions | 251 | 可以 | 17 个复合物短肽位点 |
| base recovery | 21.12% | 可以 | 天然复合物短肽 |
| methyl positive count | 0 | 必须说明 | 没有甲基化正样本 |
| methylation F1 / Recall / AUC | 不可报告 | 不可 | 因为 positive = 0 |
| false positive rate / predicted methyl rate | 可以 | 可以 | 作为负样本集误报分析 |

### 3.3 生成 FASTA 评价

输出目录：`paper_clean_v28_outputs/generated_fasta_clean_auto_single/`

| 指标 | 数值 | 可否报告 | 备注 |
|---|---:|---|---|
| native targets | 17 | 可以 | 17 complexes |
| temperatures | 5 | 可以 | 0.01, 0.10, 0.20, 0.30, 0.50 |
| raw designs | 4115 | 可以 | 原始生成序列 |
| unique designs | 4015 | 可以 | 去重后 |
| best rows | 85 | 可以 | 17 × 5 oracle best-of-N |
| chain matching warnings | 0 | 可以 | 说明 auto_single 匹配干净 |

注意：`best_designs.csv` 是 oracle best-of-N subset，按 native-sequence recovery 选择。论文中必须写成上限分析或 downstream structure subset，不能写成模型真实无监督筛选。

### 3.4 Complex structure metrics

输出目录：`paper_clean_v28_outputs/structure_metrics/`

| 指标 | 当前状态 | 可否报告 | 备注 |
|---|---|---|---|
| best85 structure coverage | 81/85 OK, 4 missing | 可以阶段性报告 | 4 条 PDB 等师兄补跑 |
| chain mapping audit | 81 OK, 4 missing | 可以 | HETATM / modified residue 已处理 |
| RMSD_CA | 81/85 已测 | 可以阶段性报告 | 以 receptor-fit peptide CA RMSD 为 binding pose 主口径 |
| Backbone RMSD | 81/85 已测 | 可以阶段性报告 | N/CA/C backbone |
| peptide self-superposed CA RMSD | 81/85 已测 | 可以 | 只反映 peptide 自身构象 |
| Designability RMSD < 2 / < 5 | 可从 RMSD 派生 | 可以但要谨慎 | 当前按 receptor-fit peptide CA RMSD |
| pLDDT / ipTM / inter-PAE | 部分缺失 | 只可按 available 报告 | 47/81 OK rows 缺 HighFold score |
| quality gate | PASS | 必须引用 | 但有 warnings |

结构质量闸门当前关键信息：

- Kabsch self-test RMSD ≈ 1.0e-15。
- `complex_chain_mapping_audit.csv`：85 行。
- `complex_rmsd_metrics.csv`：85 行。
- `rmsd_status`：81 `ok`，4 `skip_not_ok_chain_mapping`。
- `QUALITY GATE: PASS`。
- Warnings：HighFold pLDDT / ipTM / inter-PAE 有 47/81 缺失。

## 4. 未完成或不能报告的指标

| 指标 | 当前状态 | 下一步 |
|---|---|---|
| Full Rosetta train/test leakage audit | 不能完成 | 当前仓库没有 train 文件，需要找训练集或训练清单 |
| 4 条 missing PDB | 未完成 | 等师兄补跑后重跑 01→02→03→04→05 |
| HighFold score completeness | 部分缺失 | 检查 PDB COMMENT 是否缺失；能做 pLDDT fallback 需明确 source，不能补编 ipTM/PAE |
| All-atom RMSD | 未做 | 需要非标准/甲基化残基原子映射 |
| Methylation-site CA/Backbone/All RMSD | 未做 | 先定义 methylation site 和邻域窗口 |
| BSR binding site ratio | 未做 | 需要定义 binding site cutoff，例如 peptide-receptor 原子距离 < 4/5/8 Å |
| Success = peptide lower energy than native | 未做 | 需要能量输出，不能用 pLDDT/ipTM/RMSD 替代 |
| Stability = complex lower energy than native | 未做 | 需要能量输出，不能用 confidence 替代 |
| Diversity = 1 - TMScore | 未做 | 需要结构 pairwise TM-score 或替代工具 |
| Membrane permeability | 未完成 | 等师兄 full permeability CSV |
| Final integrated table | 未完成 | 等 structure/permeability/energy/diversity 补齐后合并 |

## 5. 必须做的图

### 5.1 Sequence figures

1. 单体 known-sequence vs end-to-end methylation ROC。
2. 单体 known-sequence vs end-to-end PR curve。
3. 单体 threshold-F1 curve。
4. 生成 FASTA 不同温度 mean/best recovery 柱状图或箱线图。
5. 不同 target recovery 热图：target × temperature。
6. raw vs unique count by temperature。
7. best85 methylation rate / methyl count distribution。

### 5.2 Structure figures

1. receptor-fit peptide CA RMSD by temperature。
2. receptor-fit peptide backbone RMSD by temperature。
3. peptide self-superposed RMSD by temperature。
4. receptor fit RMSD sanity plot。
5. Designability success rate by temperature：RMSD < 2 Å / < 5 Å。
6. RMSD vs HighFold pLDDT/ipTM/inter-PAE scatter，仅对有分数的行作图。
7. target-level RMSD heatmap。
8. 缺失 PDB / 缺失 HighFold score audit 图或表。

### 5.3 Later figures

1. Permeability distribution by temperature / target。
2. Recovery vs permeability scatter。
3. RMSD vs permeability scatter。
4. Energy delta vs RMSD / permeability，如果能量结果可用。
5. Diversity 1 - TMScore distribution，如果 TM-score 可用。

## 6. 下一步任务顺序

### Immediate tasks

1. 运行 `paper_clean_v28/06_available_data_leakage_audit.py`。
2. 查看并提交 available-data leakage audit 输出。
3. 找训练集或训练清单，用于完整 train/test leakage audit。
4. 等师兄补回 4 条 missing PDB。
5. 等师兄返回 full permeability CSV。

### After missing PDB returns

重跑：

```bash
python paper_clean_v28/structure_metrics/01_extract_highfold_scores.py
python paper_clean_v28/structure_metrics/02_audit_best85_structure_coverage.py
python paper_clean_v28/structure_metrics/03_audit_complex_chain_mapping.py
python paper_clean_v28/structure_metrics/04_compute_complex_rmsd.py
python paper_clean_v28/structure_metrics/05_validate_complex_structure_metrics.py
```

目标：把 81/85 结构结果补成 85/85，并重新确认 `QUALITY GATE: PASS`。

### After permeability returns

1. 写 permeability merge 脚本。
2. 将 permeability 合并到 best85 / structure metrics。
3. 生成 final candidate table。
4. 画 recovery / RMSD / permeability 联合分析图。

### If energy results become available

1. 合并 peptide energy / complex energy。
2. 计算 Success：peptide lower energy than native。
3. 计算 Stability：complex lower energy than native。
4. 注意：没有 energy 文件之前，不能报告 Success/Stability energy 指标。

## 7. 论文表述红线

1. 不能写“天然复合物短肽 methylation F1 = 0”，应该写“not reportable because there are no positive methylated residues”。
2. 不能写“best85 是模型自动筛选最佳设计”，应该写“oracle best-of-N subset selected by native-sequence recovery”。
3. 不能写“pLDDT/ipTM 代表能量稳定性”，confidence 和 energy 是不同概念。
4. 不能把 `peptide_ca_rmsd_self_superposed` 当成 binding pose 成功标准。
5. 不能在 HighFold score 缺失时填 0 或随便均值补全。
6. 不能声称完整 train/test leakage audit 已完成，除非训练集文件或训练清单真的可用。

## 8. Codex / ChatGPT 分工建议

- ChatGPT 当前对话：负责科研口径、指标解释、质量审查、实验路线。
- Codex：负责具体改文件、跑测试、提交 PR/commit。

给 Codex 的固定提示：

```text
请先阅读 paper_clean_v28/METRIC_MASTER_PLAN.md。不要改变科研口径，不要编造缺失指标，不要把 confidence 当 energy，不要把 oracle best-of-N 写成无监督筛选。修改后必须说明运行了哪些检查，哪些数据仍然缺失。
```
