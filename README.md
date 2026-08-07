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

## 温度 0.5 多种子重跑

只重跑尚未同时达到“透膜性相对 native 提高 + 双 RMSD <3 Å”的13个目标，并先在本地将13,500条序列压缩为最多185个结构任务：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_t05_rerun.ps1
```

完整口径和透膜性预测返回后的第二步见：

```text
paper_clean_v28/rerun_t05/README.md
```
