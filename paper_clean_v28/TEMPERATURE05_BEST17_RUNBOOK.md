# Temperature 0.5 best17 + monomer downstream metrics

This workflow has two explicit analysis panels:

1. **Complex panel:** the corrected RMSD-best structure for each of the 17
   targets at generation temperature 0.5.
2. **Monomer panel:** all 151 reference/e2e monomer pairs represented by the
   existing 560 HighFold PDB files.

The complex selection is never re-ranked by pLDDT.  Each target keeps the exact
temperature-0.5 PDB with minimum complete final-chain peptide Cα RMSD after one
global complex alignment and all forward cyclic register shifts.

## One-command run

Run from `(wain) PowerShell` at the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\paper_clean_v28\structure_metrics\run_temperature05_best17_all.ps1
```

If complex steps 1–7 already completed and the earlier run stopped at the
WSL/PyRosetta path error, resume without rebuilding those outputs:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\paper_clean_v28\structure_metrics\run_temperature05_best17_all.ps1 `
  -StartStep 8
```

The resumed command performs complex PyRosetta scoring and finalization, then
continues automatically through the complete monomer workflow and final
workbook.

## Twelve stages

1. Select and isolate the 17 exact temperature-0.5 complex PDBs.
2. Audit complex PDB coverage.
3. Audit complex peptide/receptor chain mapping.
4. Recompute legacy receptor-fit complex metrics for audit.
5. Attach complex HighFold confidence fields.
6. Recompute complex methylation/non-methylation position metrics.
7. Merge complex permeability.
8. Recompute naturalized fixed-pose complex PyRosetta energy.
9. Merge the 17-row complex table and enforce its quality gate.
10. Recompute monomer structure, confidence, TM, and permeability metrics.
11. Recompute 302 monomer PyRosetta scores (151 reference + 151 e2e).
12. Create the joint Excel workbook and final quality gate.

## Monomer PDB mapping

The 560 PDB filenames encode four structure roles:

| Filename variant | Role | Coverage | Use |
|---:|---|---:|---|
| 1 | reference, explicit methylation | 125/151 | sensitivity subset |
| 2 | reference, naturalized | 151/151 | primary |
| 3 | e2e design, explicit methylation | 133/151 | sensitivity subset |
| 4 | e2e design, naturalized | 151/151 | primary |

Variants 2 and 4 form the complete primary panel.  Variants 1 and 3 are both
available for 110 samples and are reported as an explicit-methylation
sensitivity subset.  Missing variant-1/3 files are not converted to zeros.

The monomer workflow verifies every filename sequence against
`monomer_design_structure_manifest.csv` before calculating metrics.

## Monomer metric definitions

- **Primary CA RMSD:** e2e naturalized structure self-superposed to the
  naturalized reference-sequence structure; all forward cyclic shifts are
  tested and reverse order is disallowed.
- **Backbone RMSD:** N/CA/C RMSD using the same best-shift CA transform.
- **Methyl/non-methyl RMSD:** residue deviations grouped by lowercase positions
  in `e2e_design_sequence`.
- **TM/diversity:** best-forward-shift symmetric TM-score and `1 - TM`.
- **Confidence:** CA B-factor/pLDDT for reference and e2e structures; COMMENT
  pLDDT/pTM are retained separately when present.
- **Permeability:** exact sequence match against monomer permeability CSV files
  discovered under `raw_external/pdb_permeability_v20260624/`.
- **Energy:** fixed-pose naturalized ref2015 total score and score per residue
  for reference and e2e structures, plus `e2e - reference` deltas.

These monomer comparisons are between two HighFold predictions.  They measure
predicted conformational change/designability and must not be described as
experimental native-structure validation.

## Metrics that are not applicable or not estimable

### Complex temperature-0.5 panel

- Original within-target TM diversity is not estimable because one structure
  per target leaves zero within-target pairs.
- Native-relative energy Success/Stability is not estimable because native
  reference energies are not computed by the fixed-pose workflow.
- BSR remains unavailable because no validated binding-site definition/cutoff
  has been established.

### Single-chain monomers

- cross-interface energy;
- receptor-fit binding-pose RMSD;
- ipTM and inter-chain PAE;
- receptor-defined binding-site recovery;
- all-atom RMSD between different reference/e2e sequences.

These fields are recorded as `NA` with an explicit status, never as zero.

## Software environments

- Windows conda environment `wain`: audits, RMSD, confidence, TM-score,
  permeability, merging, and Excel output.
- WSL2 distribution `Ubuntu`, conda environment `pyrosetta_eval`: complex and
  monomer PyRosetta scoring.
- The Windows environment must retain `tmtools==0.3.0`, as used by the earlier
  complex TM-diversity workflow.
- Excel creation uses `openpyxl`.

Windows drive paths are converted directly to WSL mount paths, for example
`E:\work` → `/mnt/e/work`; raw backslash paths are not passed through
`wslpath`.  The controller also constructs each WSL Bash program as one
semicolon-delimited line.  This prevents a Windows CRLF checkout from turning
Bash's `pipefail` option into the invalid token `pipefail\\r`.

## Outputs

All outputs remain isolated under:

```text
paper_clean_v28_outputs/temperature_0.5_best17/
```

Primary deliverables:

```text
temperature05_best17_and_monomer_all_metrics.xlsx
temperature05_best17_and_monomer_metric_summary.csv
temperature05_best17_all_metrics.csv
monomer_all_metrics.csv
temperature05_best17_and_monomer_quality_gate.csv
temperature05_best17_and_monomer_report.txt
run_console.log
```

The Excel workbook contains:

- `Metric_summary`
- `Complex_best17`
- `Monomer_151`
- `Monomer_model`
- `Quality_gate`
- `Definitions`

The run is complete only when the console ends with:

```text
===== ALL DONE =====
```

and the final report contains:

```text
QUALITY GATE: PASS
PROBLEMS: 0
```
