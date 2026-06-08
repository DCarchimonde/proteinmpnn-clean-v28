# ProteinMPNN Clean V28 Evaluation Package

本仓库是 `frankenstein_v28.pt` 的干净评价和结构预测交接包。

## 目录说明

```text
paper_clean_v28/                         干净评价脚本
paper_clean_v28_outputs/monomer_clean/   单体测试集干净评价结果
paper_clean_v28_outputs/complex_native_clean/ 复合物天然短肽干净评价结果
paper_clean_v28_outputs/generated_fasta_clean_auto_single/ 生成 FASTA 干净评价结果
paper_clean_v28_outputs/af3_manifest.csv 给结构预测使用的 85 个任务清单
frankenstein_v28.pt                      最终模型
17_complexes_native.jsonl                17 个天然复合物数据
nmethyl_data/test_set/test.jsonl          单体测试集
all_temperature_results/                 已生成的 FASTA 序列
```

## 已确认结果

### 单体测试集

```text
真实评价位点：1505
基础氨基酸恢复率：16.08%
甲基化正样本数：323
已知序列甲基化 F1：80.14%
端到端甲基化 F1：60.91%
```

推荐论文主口径：`strict_naturalized_input` 下的 `known_sequence_methylation`。

### 复合物天然短肽

```text
真实短肽位点：251
基础氨基酸恢复率：21.12%
甲基化正样本数：0
```

复合物天然短肽没有甲基化正样本，因此不能报告甲基化召回率、F1 或 AUC，只能报告误报率或预测甲基化比例。

### 生成 FASTA

使用 `auto_single` 口径重新对齐：

```text
天然复合物目标数：17
原始生成序列数：4115
去重后生成序列数：4015
最佳设计条目数：85
警告数：0
```

`85 = 17 个目标 × 5 个温度`。

## 重新运行

```bash
bash run_reproduce_clean_eval.sh
```

## 结构预测

给结构预测使用的任务清单：

```text
paper_clean_v28_outputs/af3_manifest.csv
```

重要字段：

```text
design_peptide_seq           保留小写字母，表示 N-甲基化残基
design_peptide_natural_seq   将小写甲基化残基还原成普通天然氨基酸后的序列
design_methyl_count          设计序列中的甲基化位点数量
natural_aa_recovery          设计短肽相对天然短肽的天然氨基酸恢复率
```

如果结构预测平台不支持 N-甲基化残基，先用 `design_peptide_natural_seq` 预测结构，同时单独记录小写位点作为甲基化位点。
