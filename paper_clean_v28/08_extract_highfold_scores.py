#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract HighFold PDB scores and audit matching to clean V28 designs.
CPU only. Put raw PDB folders under raw_external/; outputs go to paper_clean_v28_outputs/structure_metrics/.
"""
import argparse, csv, re
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean

TEMP_MAP = {"pdb_highfold4_t001":0.01,"pdb_highfold4_t01":0.1,"pdb_highfold4_t02":0.2,"pdb_highfold4_t03":0.3,"pdb_highfold4_t05":0.5}

def ntemp(x):
    if x in (None, ""): return ""
    return