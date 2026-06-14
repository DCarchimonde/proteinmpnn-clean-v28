#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract HighFold COMMENT scores and audit exact sequence matches."""
import csv,re,argparse
from pathlib import Path
from collections import defaultdict,Counter
T={'pdb_highfold4_t001':'0.01','pdb_highfold4_t01':'0.1','pdb_highfold4_t02':'0.2','pdb_highfold4_t03':'0.3','pdb_highfold4_t05':'0.5'}
def rd(p):
    with open(p,encoding='utf-8',newline