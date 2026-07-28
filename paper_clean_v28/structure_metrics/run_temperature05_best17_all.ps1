[CmdletBinding()]
param(
    [string]$SelectionCsv = "",
    [string]$RunDir = "",
    [string]$WindowsCondaEnv = "wain",
    [string]$WslDistribution = "Ubuntu",
    [string]$WslCondaRoot = "/home/aaron/miniconda3",
    [string]$PyRosettaEnv = "pyrosetta_eval"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

if ([string]::IsNullOrWhiteSpace($SelectionCsv)) {
    $SelectionCsv = Join-Path $RepoRoot "paper_clean_v28_outputs\structure_metrics\best_forward_cyclic_shift_ca_rmsd\best_forward_cyclic_shift_new_rmsd_best85_all_valid.csv"
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunDir = Join-Path $RepoRoot "paper_clean_v28_outputs\temperature_0.5_best17"
}

$SelectionCsv = [System.IO.Path]::GetFullPath($SelectionCsv)
$RunDir = [System.IO.Path]::GetFullPath($RunDir)
$AllDesigns = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\all_designs.csv"
$PermeabilityDir = Join-Path $RepoRoot "raw_external\pdb_permeability_v20260624\permeability_complex"
$PrepareScript = Join-Path $ScriptDir "16_prepare_temperature05_best17.py"
$FinalizeScript = Join-Path $ScriptDir "17_finalize_temperature05_best17.py"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    Write-Host ""
    Write-Host "===== $Label =====" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $SelectionCsv -PathType Leaf)) {
    throw "Selection CSV not found: $SelectionCsv"
}
if (-not (Test-Path $AllDesigns -PathType Leaf)) {
    throw "all_designs.csv not found: $AllDesigns"
}
if (-not (Test-Path $PermeabilityDir -PathType Container)) {
    throw "Permeability directory not found: $PermeabilityDir"
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$LogPath = Join-Path $RunDir "run_console.log"
Start-Transcript -Path $LogPath -Force | Out-Null

try {
    Invoke-Checked "1/9 Prepare isolated temperature-0.5 best17 workspace" {
        conda run --no-capture-output -n $WindowsCondaEnv python $PrepareScript `
            --selection_csv $SelectionCsv `
            --run_dir $RunDir
    }

    $Workspace = Join-Path $RunDir "workspace"
    $StageScripts = Join-Path $Workspace "paper_clean_v28\structure_metrics"
    $StageMetrics = Join-Path $Workspace "paper_clean_v28_outputs\structure_metrics"
    $StagePermeability = Join-Path $Workspace "paper_clean_v28_outputs\permeability"
    $StageManifest = Join-Path $Workspace "paper_clean_v28_outputs\temperature05_best17_manifest.csv"
    $StageRmsd = Join-Path $StageMetrics "complex_rmsd_metrics.csv"

    Push-Location $Workspace
    try {
        Invoke-Checked "2/9 Audit the 17 exact selected PDBs" {
            conda run --no-capture-output -n $WindowsCondaEnv python `
                (Join-Path $StageScripts "02_audit_best85_structure_coverage.py")
        }
        Invoke-Checked "3/9 Audit peptide and receptor chain mapping" {
            conda run --no-capture-output -n $WindowsCondaEnv python `
                (Join-Path $StageScripts "03_audit_complex_chain_mapping.py")
        }
        Invoke-Checked "4/9 Recompute legacy receptor-fit metrics for audit" {
            conda run --no-capture-output -n $WindowsCondaEnv python `
                (Join-Path $StageScripts "04_compute_complex_rmsd.py")
        }
        Invoke-Checked "5/9 Attach HighFold confidence and CA B-factor metrics" {
            conda run --no-capture-output -n $WindowsCondaEnv python `
                (Join-Path $StageScripts "06_apply_highfold_plddt_bfactor_fallback.py")
        }
        Invoke-Checked "6/9 Recompute methylation-position structural metrics" {
            conda run --no-capture-output -n $WindowsCondaEnv python `
                (Join-Path $StageScripts "07_compute_methylation_site_rmsd.py")
        }
        Invoke-Checked "7/9 Merge permeability for the exact 17 selected sequences" {
            conda run --no-capture-output -n $WindowsCondaEnv python `
                (Join-Path $Workspace "paper_clean_v28\08_merge_complex_permeability.py") `
                --permeability_dir $PermeabilityDir `
                --all_designs_csv $AllDesigns `
                --best85_csv $StageManifest `
                --rmsd_csv $StageRmsd `
                --out_dir $StagePermeability
        }
    }
    finally {
        Pop-Location
    }

    Invoke-Checked "8/9 Recompute naturalized fixed-pose PyRosetta energy" {
        $WorkspaceWsl = (
            wsl.exe -d $WslDistribution -- wslpath -a $Workspace
        ).Trim()
        if ([string]::IsNullOrWhiteSpace($WorkspaceWsl)) {
            throw "Unable to convert the isolated workspace path for WSL."
        }
        $BashCommand = @"
set -euo pipefail
source '$WslCondaRoot/etc/profile.d/conda.sh'
cd '$WorkspaceWsl'
conda run --no-capture-output -n '$PyRosettaEnv' python 'paper_clean_v28/structure_metrics/10_compute_pyrosetta_energy_naturalized.py'
"@
        wsl.exe -d $WslDistribution -- bash -lc $BashCommand
    }

    Invoke-Checked "9/9 Merge all metrics and enforce the 17-row quality gate" {
        conda run --no-capture-output -n $WindowsCondaEnv python $FinalizeScript `
            --run_dir $RunDir `
            --all_designs_csv $AllDesigns
    }

    $FinalCsv = Join-Path $RunDir "temperature05_best17_all_metrics.csv"
    $FinalReport = Join-Path $RunDir "temperature05_best17_all_metrics_report.txt"
    Write-Host ""
    Write-Host "===== ALL DONE =====" -ForegroundColor Green
    Write-Host "Final 17-row table: $FinalCsv"
    Write-Host "Quality report: $FinalReport"
    Write-Host "Console log: $LogPath"
}
finally {
    Stop-Transcript | Out-Null
}
