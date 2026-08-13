# Clean V28 repository file guide

This repository is a clean handoff package for the V28 N-methylation ProteinMPNN evaluation and structure-prediction workflow.

## Files to send for structure prediction

### Complex designs

File:

```text
paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv
```

Use this file when all complex generated designs are required. The structure-prediction sequence column is `design_seq`, which is column K in the current CSV.

If only the selected best designs are required, use:

```text
paper_clean_v28_outputs/af3_manifest.csv
```

This contains the selected 85 complex best-design tasks.

### Monomer designs

File:

```text
paper_clean_v28_outputs/monomer_design_structure_manifest.csv
```

Important columns:

- `reference_original_sequence`: original Rosetta/test-set sequence. Lowercase letters mark original N-methylated residues.
- `reference_natural_sequence`: original sequence after converting lowercase methylation markers to normal uppercase amino acids.
- `known_base_design_sequence`: original base sequence with model-predicted methylation marks. This is classifier-only, not the full end-to-end generated sequence.
- `known_base_sequence_for_structure_prediction`: uppercase version of `known_base_design_sequence`.
- `e2e_design_sequence`: end-to-end model-designed monomer sequence. Lowercase letters mark model-predicted methylation sites.
- `e2e_sequence_for_structure_prediction`: uppercase version of `e2e_design_sequence`. Use this for structure prediction if the structure-prediction platform does not support N-methylated residue tokens.
- `e2e_methyl_positions_1based`: 1-based methylation positions predicted by the model.

For the current monomer structure-prediction handoff, use `e2e_sequence_for_structure_prediction` as the main structure input and keep `e2e_methyl_positions_1based` for methylation-site analysis.

## Core code files

### `paper_clean_v28/clean_v28_common.py`
Shared model-loading, alphabet, featurization, sequence-cleaning, and metric utilities used by the clean evaluation scripts.

### `paper_clean_v28/01_eval_clean_model.py`
Runs clean monomer and complex-native evaluation using the V28 model. It produces `summary.json`, `position_predictions.csv`, `threshold_metrics.csv`, and `sample_manifest.csv`.

### `paper_clean_v28/02_score_generated_fastas.py`
Scores generated complex FASTA files against the correct native peptide chain using the `auto_single` chain-matching logic. It produces complex design recovery summaries and best-design selections.

### `paper_clean_v28/03_prepare_structure_manifest.py`
Builds the complex structure-prediction manifest from complex best-design results. Output: `paper_clean_v28_outputs/af3_manifest.csv`.

### `paper_clean_v28/04_audit_native_chains.py`
Audits native complex chain IDs, chain lengths, and peptide-chain choices. Output: `paper_clean_v28_outputs/native_chain_audit.csv`.

### `paper_clean_v28/07_prepare_monomer_design_structure_manifest.py`
Rebuilds monomer end-to-end designed sequences from `paper_clean_v28_outputs/monomer_clean/position_predictions.csv`. Output: `paper_clean_v28_outputs/monomer_design_structure_manifest.csv`.

## Main data files

### `frankenstein_v28.pt`
Final V28 model checkpoint.

### `model_utils.py`
Model utility code required by the V28 model.

### `nmethyl/utils/nmethyl_config.py`
N-methylation alphabet and token configuration.

### `nmethyl_data/test_set/test.jsonl`
Rosetta/test-set monomer data used for clean monomer evaluation.

### `17_complexes_native.jsonl`
Native complex data for the 17 complex targets.

### `all_temperature_results/`
Original generated complex FASTA files across temperatures. These are the source FASTA files used to build complex design scoring outputs.

## Main output files

### `paper_clean_v28_outputs/monomer_clean/summary.json`
Summary metrics for monomer clean evaluation.

### `paper_clean_v28_outputs/monomer_clean/position_predictions.csv`
Per-position monomer predictions. This is the source used to rebuild `monomer_design_structure_manifest.csv`.

### `paper_clean_v28_outputs/monomer_clean/threshold_metrics.csv`
Monomer methylation metrics across thresholds.

### `paper_clean_v28_outputs/complex_native_clean/summary.json`
Summary metrics for complex-native clean evaluation.

### `paper_clean_v28_outputs/generated_fasta_clean_auto_single/all_designs.csv`
All complex generated designs with clean chain matching and recovery scores. Use column `design_seq` for complex structure prediction when all designs are required.

### `paper_clean_v28_outputs/generated_fasta_clean_auto_single/unique_designs.csv`
Deduplicated complex generated designs.

### `paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv`
Selected best complex designs by target and temperature.

### `paper_clean_v28_outputs/generated_fasta_clean_auto_single/summary_by_temperature.csv`
Complex recovery and methylation-rate summary by temperature.

### `paper_clean_v28_outputs/generated_fasta_clean_auto_single/summary_by_target.csv`
Complex recovery and methylation-rate summary by target.

### `paper_clean_v28_outputs/generated_fasta_clean_auto_single/summary_by_target_temperature.csv`
Complex recovery and methylation-rate summary by both target and temperature.

### `paper_clean_v28_outputs/generated_fasta_clean_auto_single/report.json`
Machine-readable summary of the complex generated FASTA scoring run.

### `paper_clean_v28_outputs/generated_fasta_clean_auto_single/warnings.csv`
Warnings from the complex generated FASTA scoring run. The clean run should have zero chain-matching warnings.

### `paper_clean_v28_outputs/af3_manifest.csv`
Selected 85 complex best-design structure-prediction tasks.

### `paper_clean_v28_outputs/monomer_design_structure_manifest.csv`
Monomer end-to-end model-designed sequence manifest for structure prediction.

### `paper_clean_v28_outputs/native_chain_audit.csv`
Audit file for native complex chain selection and chain lengths.

### `paper_clean_v28_outputs/structure_manifest_warnings.csv`
Warnings from complex structure-manifest generation.

## Reproduction helper files

### `run_reproduce_clean_eval.sh`
One-command script to rerun the clean evaluation workflow.

### `requirements_minimal.txt`
Minimal Python dependencies.

### `CHECK_FILE_SIZES.sh`
Checks whether any files are too large for normal GitHub upload.

### `.gitignore`
Clean-repository ignore rules.

## Ser provenance recovery

### `paper_clean_v28/serine_qc_retrain/README.md`
Evidence, frozen scope, quality gates, and exact commands for the Ser
provenance correction and structure-first recovery.

### `run_serine_qc_recovery.ps1` / `run_serine_qc_recovery.sh`
One-command Windows and AutoDL/Linux launchers. They rebuild labels from the
pinned raw PDB source, retrain only the canonical Ser expert, regenerate only
the ten failed T=0.5 targets, and create the structure handoff. Permeability is
deliberately deferred until structures return.
