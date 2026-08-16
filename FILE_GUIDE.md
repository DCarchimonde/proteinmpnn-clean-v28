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

## Ser provenance and cyclic-representation recovery

### `paper_clean_v28/serine_qc_retrain/README.md`
Evidence for the Ser correction, the withdrawn non-methylated 3AV9 row, the
fixed tensor-position-7 root cause, V6 quality gates, and exact commands.

### `paper_clean_v28/run_serine_qc_source_scoped_hybrid_v8.ps1`
Current Windows launcher. It performs no training and never reruns V6. It pins
and reuses the passed canonical, V6 and V7 artifacts; composes canonical shared
tensors + V6 non-Ser experts + V7 Ser expert; and requires exact per-position
inheritance plus V6-noninferior frozen Recall/F1 before continuing. It then
reannotates the immutable 31,500-row pool, performs deterministic fixed-budget
recovery only for actually missing 3WNE/3ZGC targets, and runs the independent
overlay audit. Existing partial stage directories are preserved. The only
packaged deliverable is a checksum-indexed manual-review ZIP; no structure
handoff or permeability input is created.
This is explicitly a post-hoc internal recovery workflow; publication metrics
still require a new outer/blind evaluation.

### `paper_clean_v28/serine_qc_retrain/12_compose_source_scoped_hybrid_v8.py`
Composes the fixed residue-source checkpoint without averaging or optimization.
The paired 1,505-position audit proves non-Ser probabilities come from V6 and
Ser probabilities come from V7. The repeatedly inspected 151-record set is
explicitly documented as an internal frozen audit rather than a new blind test.

### `paper_clean_v28/serine_qc_retrain/13_audit_source_scoped_hybrid_v8.py`
Audits cyclic-start/decoder-order remapping, V6-noninferior sensitivity, all 17
native-chain mappings, and explicit length-6/length-7 strata before authorizing
directed recovery.

### `paper_clean_v28/serine_qc_retrain/14_directed_recovery_search_v8.py`
Scores mandatory historical/native length-6/7 controls and applies the frozen
deterministic search budget only to missing 3WNE/3ZGC targets. Released rows
must pass strict rounded `>0.6`, batch-one rescore, base-head plausibility and
historical/prior/native/current-pool plus forward-cyclic novelty gates. Search
metrics are kept separate from model metrics.

### `paper_clean_v28/serine_qc_retrain/15_finalize_and_audit_recovery_v8.py`
Overlays independently rescored directed rows on the immutable V8 baseline in
a new directory, verifies 17/17 coverage, 3AV physical-position support,
novelty and workflow ordering, and blocks handoff/permeability artifacts.

### `run_serine_qc_serine_only_cyclic_v7.ps1`
Historical V7 launcher. Its Ser provenance repair is retained as the source of
the V8 Ser expert, but its rollback of all 19 non-Ser experts caused frozen
Recall@0.6 to fall from 0.8046 to 0.5096 (77 true positives lost). Do not use it
as the current recovery command. Its preserved 15/17 generation failure is
included in the V8 review ZIP as diagnostic evidence.

### `paper_clean_v28/serine_qc_retrain/10_reannotate_v6_pool_serine_only_v7.py`
Scores each unique target/natural sequence once, propagates one canonical
annotation payload to repeats, recomputes novelty, and preserves all base-model
sampling statistics. Defaults reproduce Ser-only V7; explicit protocol/scope/
authorization arguments let the V8 launcher use the same audited reannotation
logic and record an isolated missing-target coverage state for recovery.

### `paper_clean_v28/serine_qc_retrain/11_triple_audit_serine_only_v7.py`
Independently reconstructs strict-threshold annotations, aggregation, position
and decoder-step distributions, novelty, and 17/17 coverage. It treats nearest
held-out backbone similarity as a diagnostic rather than methylation truth; the
actual returned structures remain subject to both frozen RMSD gates.

### `run_serine_qc_cyclic_representation_v6.ps1`
Historical V6 launcher; do not use for the current repair. It retrains all 20 expert heads with every equivalent
cyclic sequence/coordinate start, runs the held-out test gate, regenerates all
17 targets, and performs the independent three-pass result audit. It stops at a
manual-review bundle and never creates a structure handoff. The older V5
handoff path is withdrawn and release-blocked. `-ResumeQuota` preserves a
completed V6 checkpoint and original 19,500 draws, samples only quota-shortfall
targets up to a cumulative fixed budget, and then continues the same three-pass
review packaging. If that fixed budget has already produced zero candidates,
the launcher records an explicit model abstention without loading Torch or
sampling again.

### `paper_clean_v28/serine_qc_retrain/08_resume_cyclic_representation_v6_quota.py`
In-place, hash-pinned V6 quota recovery. It retains every pre-resume candidate,
uses disjoint reserve seeds only for targets below the frozen structure quota,
and records exact initial/top-up row accounting for the independent audit.

### `paper_clean_v28/serine_qc_retrain/09_finalize_cyclic_representation_v6_exhaustion.py`
Metadata-only terminal audit for a V6 target with zero novel candidates after
the complete initial pool and 12,000 fixed-budget top-up rows. It preserves all
candidate CSV hashes, does not lower the 0.6 threshold, and records
`MODEL_ABSTAINS` instead of pretending that the frozen structure quota passed.
