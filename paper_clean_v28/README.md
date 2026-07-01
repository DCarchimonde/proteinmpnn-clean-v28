# paper_clean_v28

这是给 `frankenstein_v28.pt` 单独建立的干净评价工作区。这个文件夹不改动原来的训练、生成、旧评估脚本；论文主结果优先从这里和 `paper_clean_v28_outputs/` 重新产出。

## 当前主口径

1. 最终模型：`frankenstein_v28.pt`。
2. 单体测试集：`nmethyl_data/test_set/test.jsonl`。
3. 复合物天然结构数据：`17_complexes_native.jsonl`。
4. 生成序列：`all_temperature_results/`。
5. 复合物 generated FASTA 的主口径使用 `auto_single`，输出目录为 `paper_clean_v28_outputs/generated_fasta_clean_auto_single/`。
6. 旧口径 `short` / `generated_fasta_clean` 不再作为主结果使用，因为多短链复合物中仅按短链筛选可能选错 peptide chain。

## 目标

1. 统一字母表，只使用 `nmethyl/utils/nmethyl_config.py`。
2. 统一模型结构，只针对最终模型 `frankenstein_v28.pt`。
3. 严格排除 padding 位点。
4. 单体、天然复合物短肽、生成 FASTA、结构指标分开评价。
5. 明确区分：
   - 基础氨基酸恢复率 / RAA / base recovery；
   - 已知序列条件下的甲基化位点预测；
   - 端到端设计加甲基化预测；
   - oracle best-of-N 生成序列分析；
   - 复合物结构 pose RMSD 与 peptide self-superposed RMSD。
6. 为后续结构预测、permeability、能量、多样性和论文作图输出干净表格。

## 推荐目录

在仓库根目录下运行所有脚本，所有新结果统一放在：

```text
paper_clean_v28_outputs/
```

不要再和旧的 `run_v28_robust`、`generated_peptides`、`temperature_sequence_metrics` 混在一起。

## 第一步：单体甲基化与序列评价

```bash
python paper_clean_v28/01_eval_clean_model.py \
  --model_path ./frankenstein_v28.pt \
  --data_jsonl nmethyl_data/test_set/test.jsonl \
  --mode monomer \
  --eval_chains masked \
  --batch_size 16 \
  --out_dir paper_clean_v28_outputs/monomer_clean
```

重点看输出：

```text
paper_clean_v28_outputs/monomer_clean/summary.json
paper_clean_v28_outputs/monomer_clean/position_predictions.csv
paper_clean_v28_outputs/monomer_clean/threshold_metrics.csv
```

论文主口径优先看 `strict_naturalized_input`。当前已经得到：1505 个评价位点，strict naturalized 输入下 base recovery = 16.08%，known-sequence methylation F1 = 80.14%，end-to-end methylation F1 = 60.91%。

## 第二步：复合物天然短肽评价

```bash
python paper_clean_v28/01_eval_clean_model.py \
  --model_path ./frankenstein_v28.pt \
  --data_jsonl 17_complexes_native.jsonl \
  --mode complex \
  --eval_chains auto_single \
  --max_peptide_len 30 \
  --batch_size 1 \
  --out_dir paper_clean_v28_outputs/complex_native_clean
```

注意：天然复合物短肽目前没有甲基化正样本，因此不能报告复合物天然短肽的 methylation F1、Recall 或 AUC。可以报告 base recovery、false positive rate、predicted methylation rate 等。

## 第三步：评价已经生成的 FASTA

```bash
python paper_clean_v28/02_score_generated_fastas.py \
  --native_jsonl 17_complexes_native.jsonl \
  --fasta_dir all_temperature_results \
  --out_dir paper_clean_v28_outputs/generated_fasta_clean_auto_single \
  --eval_chains auto_single \
  --max_peptide_len 30
```

这个脚本用于评价生成序列本身，不需要模型权重。当前 `generated_fasta_clean_auto_single/report.json` 显示：17 个 target，4115 条 raw designs，4015 条 unique designs，85 条 best rows，warnings = 0。

重要口径：`best_designs.csv` 是按 native-sequence recovery 选择的 oracle best-of-N 子集，用于下游结构分析。论文里不能写成“模型无监督自动筛选出了最佳序列”。

## 第四步：准备结构预测清单

```bash
python paper_clean_v28/03_prepare_structure_manifest.py \
  --best_csv paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv \
  --native_jsonl 17_complexes_native.jsonl \
  --out_csv paper_clean_v28_outputs/af3_manifest.csv
```

`af3_manifest.csv` 用于整理 best85 中哪些序列需要拿去预测结构。

## 第五步：复合物结构指标

结构预测 PDB 不进 GitHub，默认放在本地 gitignore 路径：

```text
raw_external/pdb_highfold_temperature/
```

结构指标脚本依次运行：

```bash
python paper_clean_v28/structure_metrics/01_extract_highfold_scores.py
python paper_clean_v28/structure_metrics/02_audit_best85_structure_coverage.py
python paper_clean_v28/structure_metrics/03_audit_complex_chain_mapping.py
python paper_clean_v28/structure_metrics/04_compute_complex_rmsd.py
python paper_clean_v28/structure_metrics/05_validate_complex_structure_metrics.py
```

当前结构质量闸门：85 行，81 条 RMSD OK，4 条 skipped missing PDB，`QUALITY GATE: PASS`，但有 HighFold score completeness warnings。HighFold pLDDT / ipTM / inter-PAE 对 T0.20/T0.30/T0.50 不完整，不能编造缺失值。

主结构口径：

- `peptide_ca_rmsd_after_receptor_fit`：先用 receptor 对齐，再看 peptide binding pose 偏移。
- `peptide_backbone_rmsd_after_receptor_fit`：对应 backbone pose RMSD。
- `peptide_ca_rmsd_self_superposed`：只反映 peptide 自身构象，不等价于 binding pose RMSD。
- `success_rate_ca_rmsd_lt_2` / `success_rate_ca_rmsd_lt_5`：从 receptor-fit peptide CA RMSD 派生。

## 第六步：available-data leakage audit

当前仓库只包含 `nmethyl_data/test_set/test.jsonl`，没有训练集文件，因此不能完成完整 Rosetta train/test leakage audit。可以先运行 available-data audit：

```bash
python paper_clean_v28/06_available_data_leakage_audit.py
```

该脚本会检查：

1. test.jsonl 内部重复；
2. test.jsonl 与 17 个复合物天然序列的重叠；
3. test.jsonl 与 generated FASTA / best85 / af3_manifest 设计序列的重叠；
4. 当前仓库是否存在 train/valid 文件候选。

完整 train/test leakage audit 需要补充训练集文件或训练数据清单。

## 当前结论边界

- 旧的 38% 氨基酸恢复率不能作为干净主结果。
- `frankenstein_v28.pt` 先作为当前最终模型进行完整干净评价。
- Generated FASTA 的 best85 是 oracle best-of-N subset，不是无监督筛选结果。
- 天然复合物短肽没有甲基化正样本，因此不能报告 methylation F1、Recall、AUC。
- RMSD 结果目前是 81/85 阶段性完成，4 条 missing PDB 等补跑。
- pLDDT / ipTM / PAE 不完整时必须保留 missing 或明确 fallback，不能编造。
- Success / Stability 如果定义为 lower energy than native，必须等待能量计算；不能用 RMSD、pLDDT、ipTM 或 PAE 冒充 energy。
- All-atom RMSD 和 methylation-site all-atom RMSD 需要非标准/甲基化残基原子映射，不能草率计算。
