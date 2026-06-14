#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_extract_highfold_scores.py

从 HighFold 预测复合物 PDB 里提取置信度分数，并和 clean V28 的 all_designs/af3_manifest 做匹配审计。

特点：
- 只用 Python 标准库，不需要 torch。
- 可以在 Windows + PowerShell + conda 环境下直接运行。
- 原始大文件放在 raw_external/，不进入 GitHub。
- 输出结果统一放到 paper_clean_v28_outputs/structure_metrics/。
"""

import argparse
import csv
import re
from collections