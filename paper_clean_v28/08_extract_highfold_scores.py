#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_extract_highfold_scores.py

Purpose
-------
Extract HighFold/AlphaFold-like confidence scores from predicted complex PDB files,
then audit whether those PDB files can be matched back to clean V28 generated designs.

This script is intentionally CPU-only and does NOT require torch.
It is safe to run locally on Windows.

Default expected external data layout