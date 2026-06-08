#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1

python paper_clean_v28/01_eval_clean_model.py \
  --model_path ./frankenstein_v28.pt \
  --data_jsonl nmethyl_data/test_set/test.jsonl \
  --mode monomer \
  --eval_chains masked \
  --batch_size 16 \
  --out_dir paper_clean_v28_outputs/monomer_clean

python paper_clean_v28/01_eval_clean_model.py \
  --model_path ./frankenstein_v28.pt \
  --data_jsonl 17_complexes_native.jsonl \
  --mode complex \
  --eval_chains short \
  --max_peptide_len 30 \
  --batch_size 1 \
  --out_dir paper_clean_v28_outputs/complex_native_clean

python paper_clean_v28/02_score_generated_fastas.py \
  --native_jsonl 17_complexes_native.jsonl \
  --fasta_dir all_temperature_results \
  --out_dir paper_clean_v28_outputs/generated_fasta_clean_auto_single \
  --eval_chains auto_single \
  --max_peptide_len 30

python paper_clean_v28/03_prepare_structure_manifest.py \
  --best_csv paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv \
  --native_jsonl 17_complexes_native.jsonl \
  --out_csv paper_clean_v28_outputs/af3_manifest.csv
