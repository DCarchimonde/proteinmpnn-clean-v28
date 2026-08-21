# Task 1：原始 Frankenstein-v28 中文论文与完整数据

本目录把原始 `frankenstein_v28.pt` 的单体、6 个非 3AV 复合物主分析，以及 17 靶点探索性结果整理为一篇可直接审阅的中文 LaTeX 论文。

冻结模型 SHA-256：

`bab7b8a010114fc52c749fab1914d9d8ae561ddca45d6d7a0fbec3f9f5ac5b2e`

## 直接阅读

- `main.pdf`：已经编译好的中文论文
- `main.tex` 与 `sections/`：完整 LaTeX 源文件
- `figures/`：论文图的 PDF、PNG 和 SVG 版本
- `data/`：论文主表、476 条完整审计、101 条 `<5 Å` 和 16 条 `<3 Å` 名单
- `supplementary/`：Excel 工作簿、历史阈值截图与指标提纲截图
- `SOURCE_MANIFEST.json`：关键源文件的路径、大小和 SHA-256

## 核心口径

- 单体数据：历史合并集共 751 条（训练 600、留出测试 151）。逐一复核的 751 个原始 PDB 全部带有 Rosetta 理论模型标记，其中 406 个来自 Rosetta-2023 面板、345 个来自 Rosetta-2025 面板；没有可核验的“Baker 33 真结构”子集。
- 单体测试：151 条、1505 个真实位点；Ser 来源修正后 261 阳性、1244 阴性；阈值严格 `p > 0.6`。天然化氨基酸恢复率为 16.08%，完整扩展 token 恢复率为 14.88%；端到端甲基二分类召回为 42.53%，而同时恢复正确天然氨基酸身份与甲基状态的真实甲基残基为 7.66%。
- 单体逐位复算：`data/Monomer_Corrected_1505.csv` 是正文主分析表；`data/Monomer_RawPredictions_4515.csv` 保留旧标签，仅用于追溯冻结概率，不能直接作为修正真值表。
- 单体来源审计：`data/Monomer_PDB_Provenance_751.csv` 逐文件列出 Rosetta 标记、版本、split、SHA-256 与 Git blob；`data/Monomer_Train_Test_Exact_Overlap_Audit.json` 记录精确名称/天然化序列重叠为 0（未做同源聚类）。
- 标准单表 CSV：`data/Sequence_Main_Metrics_Corrected.csv`、`data/Monomer_Threshold_Curves_Corrected.csv` 和 `data/Monomer_Historical_Labels_2.csv` 可直接用普通 `read_csv` 读取；同名工作簿原样导出表仅用于追溯。
- 六复合物：仅 1SFI、3P8F、3WNE、3ZGC、4K1E、4KEL；仅温度 0.5；544 条原始序列中 476 条包含保存的甲基标记。
- RMSD：一次全复合物 CA 对齐后，在同一坐标系计算完整末链肽的 best-forward 循环移位 CA RMSD；联合门要求全局和环肽 RMSD 同时达标。
- 六复合物结果：保留后联合 `<3 Å` 为 16/476（3.36%），联合 `<5 Å` 为 101/476（21.22%）；以 544 条原始生成作分母，端到端产率分别为 2.94% 和 18.57%。
- 17 靶点 best85：按天然序列恢复率挑选，属于 oracle/选择条件化结果，只作探索性证据。

## 重编译

仓库已包含 `main.pdf`。如需修改 LaTeX 后重编译：

```powershell
# Windows PowerShell，在本目录运行
.\build.ps1
```

```bash
# Linux/macOS，在本目录运行
bash build.sh
```

需要 XeLaTeX 和 BibTeX。素材脚本 `scripts/prepare_paper_assets.py` 只整理已有文件，不会重跑模型、结构预测、通透性或 PyRosetta；若需更新素材，先安装 `requirements-assets.txt` 中固定的 Python 依赖。fresh clone 缺少维护者工作区或原上传目录时会复用随稿的冻结数据、工作簿和截图。重新扫描751个历史 PDB 时，需按 `SOURCE_MANIFEST.json` 检出冻结的旧仓库提交，再运行 `scripts/audit_monomer_pdb_provenance.py`。

## 中文字体

`fonts/NotoSansSC-Task1-Subset.ttf` 是仅保留本文所需字符的自包含子集字体，源自 Noto Sans SC，并已改用非保留字体名。字体及其衍生子集按 `fonts/OFL.txt` 中的 SIL Open Font License 1.1 分发；原版权声明、来源版本和修改记录见 `fonts/FONT_NOTICE.txt`。这样构建不依赖系统安装 `ctex` 或中文字库。
