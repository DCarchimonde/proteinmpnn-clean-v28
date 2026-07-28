# Temperature 0.5 best17 downstream metrics

This workflow uses the corrected RMSD-best85 table and retains only the 17
target-specific winners at generation temperature 0.5.  Within each target,
the selected row is the minimum complete final-chain peptide Cα RMSD after one
global PyMOL complex alignment and the best forward cyclic shift.

## One-command run

Run from `(wain) PowerShell` at the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\paper_clean_v28\structure_metrics\run_temperature05_best17_all.ps1
```

The controller uses:

- Windows conda environment `wain` for structure audits, confidence,
  methylation-position metrics, permeability merging, and the final table;
- WSL2 distribution `Ubuntu` and conda environment `pyrosetta_eval` for
  naturalized fixed-pose PyRosetta energy.

Historical best85 outputs are not overwritten.  The isolated run is written to:

```text
paper_clean_v28_outputs/temperature_0.5_best17/
```

Primary outputs:

```text
temperature05_best17_all_metrics.csv
temperature05_best17_all_metrics_audit_wide.csv
temperature05_best17_all_metrics_report.txt
temperature05_best17_all_metrics_problem_rows.csv
run_console.log
```

The run is complete only when the report ends with:

```text
QUALITY GATE: PASS
PROBLEMS: 0
```

## Metrics that remain unavailable by definition

- The earlier within-target structural diversity metric used all five
  temperatures and generated `17 × C(5,2) = 170` TM-align pairs.  With only one
  temperature-0.5 structure per target, there are zero within-target pairs.
  The final table therefore records this metric as `NA`, not zero.
- Energy-based Success/Stability relative to native requires native reference
  energies.  The fixed-pose naturalized PyRosetta workflow does not calculate
  those reference energies, so the two delta labels remain `NA`.
- A binding-site recovery ratio remains `NA` because the previous workflow did
  not establish a validated binding-site definition.

Missing or scientifically non-estimable values are never replaced with zero.
