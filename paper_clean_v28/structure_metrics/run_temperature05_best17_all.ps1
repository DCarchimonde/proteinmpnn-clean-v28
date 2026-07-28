[CmdletBinding()]
param(
    [string]$SelectionCsv = "",
    [string]$RunDir = "",
    [string]$MonomerPdbDir = "",
    [string]$MonomerPermeabilityRoot = "",
    [string]$WindowsCondaEnv = "wain",
    [string]$WslDistribution = "Ubuntu",
    [string]$WslCondaRoot = "/home/aaron/miniconda3",
    [string]$PyRosettaEnv = "pyrosetta_eval",
    [ValidateRange(1, 12)][int]$StartStep = 1
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
if ([string]::IsNullOrWhiteSpace($MonomerPdbDir)) {
    $MonomerPdbDir = Join-Path $RepoRoot "raw_external\pdb_permeability_v20260624\pdb_monomer\pdb_monomer_hf4"
}
if ([string]::IsNullOrWhiteSpace($MonomerPermeabilityRoot)) {
    $MonomerPermeabilityRoot = Join-Path $RepoRoot "raw_external\pdb_permeability_v20260624"
}

$SelectionCsv = [System.IO.Path]::GetFullPath($SelectionCsv)
$RunDir = [System.IO.Path]::GetFullPath($RunDir)
$MonomerPdbDir = [System.IO.Path]::GetFullPath($MonomerPdbDir)
$MonomerPermeabilityRoot = [System.IO.Path]::GetFullPath($MonomerPermeabilityRoot)

$AllDesigns = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\all_designs.csv"
$ComplexPermeabilityDir = Join-Path $RepoRoot "raw_external\pdb_permeability_v20260624\permeability_complex"
$MonomerDesignManifest = Join-Path $RepoRoot "paper_clean_v28_outputs\monomer_design_structure_manifest.csv"
$MonomerReferenceManifest = Join-Path $RepoRoot "paper_clean_v28_outputs\monomer_structure_manifest.csv"
$MonomerModelSummary = Join-Path $RepoRoot "paper_clean_v28_outputs\monomer_clean\summary.json"

$PrepareScript = Join-Path $ScriptDir "16_prepare_temperature05_best17.py"
$FinalizeComplexScript = Join-Path $ScriptDir "17_finalize_temperature05_best17.py"
$MonomerStructureScript = Join-Path $ScriptDir "18_compute_monomer_structure_metrics.py"
$MonomerEnergyScript = Join-Path $ScriptDir "19_compute_monomer_pyrosetta_energy.py"
$FinalizeAllScript = Join-Path $ScriptDir "20_finalize_temperature05_best17_and_monomer.py"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    Write-Host ""
    Write-Host "===== $Label =====" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $Command
    $ExitCode = $LASTEXITCODE
    if ($null -eq $ExitCode) {
        $ExitCode = 0
    }
    if ($ExitCode -ne 0) {
        throw "$Label failed with exit code $ExitCode"
    }
}

function Convert-WindowsPathToWslMountPath {
    param(
        [Parameter(Mandatory = $true)][string]$WindowsPath
    )
    $FullPath = [System.IO.Path]::GetFullPath($WindowsPath)
    if ($FullPath -notmatch "^([A-Za-z]):[\\/](.*)$") {
        throw (
            "Only drive-letter Windows paths can be converted safely for WSL. " +
            "Observed: $FullPath"
        )
    }
    $Drive = $Matches[1].ToLowerInvariant()
    $Rest = $Matches[2] -replace "\\", "/"
    return "/mnt/$Drive/$Rest"
}

function Assert-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Purpose
    )
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Cannot continue: missing $Purpose file: $Path"
    }
}

function Assert-RequiredDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Purpose
    )
    if (-not (Test-Path $Path -PathType Container)) {
        throw "Cannot continue: missing $Purpose directory: $Path"
    }
}

Assert-RequiredFile $SelectionCsv "corrected RMSD-best85 selection"
Assert-RequiredFile $AllDesigns "complex all_designs"
Assert-RequiredDirectory $ComplexPermeabilityDir "complex permeability"
Assert-RequiredFile $MonomerDesignManifest "monomer design manifest"
Assert-RequiredFile $MonomerReferenceManifest "monomer reference manifest"
Assert-RequiredFile $MonomerModelSummary "monomer clean-evaluation summary"
Assert-RequiredDirectory $MonomerPdbDir "560-PDB monomer input"
Assert-RequiredDirectory $MonomerPermeabilityRoot "monomer permeability search root"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$LogPath = Join-Path $RunDir "run_console.log"
if ($StartStep -eq 1) {
    Start-Transcript -Path $LogPath -Force | Out-Null
}
else {
    Start-Transcript -Path $LogPath -Append | Out-Null
}

try {
    Write-Host "Requested start step: $StartStep"
    if ($StartStep -le 1) {
        Invoke-Checked "1/12 Prepare isolated temperature-0.5 best17 workspace" {
            conda run --no-capture-output -n $WindowsCondaEnv python $PrepareScript `
                --selection_csv $SelectionCsv `
                --run_dir $RunDir
        }
    }

    $Workspace = Join-Path $RunDir "workspace"
    if (-not (Test-Path $Workspace -PathType Container)) {
        throw "Isolated complex workspace not found: $Workspace. Resume from step 1 first."
    }
    $StageScripts = Join-Path $Workspace "paper_clean_v28\structure_metrics"
    $StageMetrics = Join-Path $Workspace "paper_clean_v28_outputs\structure_metrics"
    $StagePermeability = Join-Path $Workspace "paper_clean_v28_outputs\permeability"
    $StageManifest = Join-Path $Workspace "paper_clean_v28_outputs\temperature05_best17_manifest.csv"
    $StageRmsd = Join-Path $StageMetrics "complex_rmsd_metrics.csv"

    Push-Location $Workspace
    try {
        if ($StartStep -le 2) {
            Invoke-Checked "2/12 Audit the 17 exact selected complex PDBs" {
                conda run --no-capture-output -n $WindowsCondaEnv python `
                    (Join-Path $StageScripts "02_audit_best85_structure_coverage.py")
            }
        }
        if ($StartStep -le 3) {
            Invoke-Checked "3/12 Audit complex peptide and receptor chain mapping" {
                conda run --no-capture-output -n $WindowsCondaEnv python `
                    (Join-Path $StageScripts "03_audit_complex_chain_mapping.py")
            }
        }
        if ($StartStep -le 4) {
            Invoke-Checked "4/12 Recompute legacy complex receptor-fit metrics" {
                conda run --no-capture-output -n $WindowsCondaEnv python `
                    (Join-Path $StageScripts "04_compute_complex_rmsd.py")
            }
        }
        if ($StartStep -le 5) {
            Invoke-Checked "5/12 Attach complex HighFold confidence metrics" {
                conda run --no-capture-output -n $WindowsCondaEnv python `
                    (Join-Path $StageScripts "06_apply_highfold_plddt_bfactor_fallback.py")
            }
        }
        if ($StartStep -le 6) {
            Invoke-Checked "6/12 Recompute complex methylation-position metrics" {
                conda run --no-capture-output -n $WindowsCondaEnv python `
                    (Join-Path $StageScripts "07_compute_methylation_site_rmsd.py")
            }
        }
        if ($StartStep -le 7) {
            Invoke-Checked "7/12 Merge complex permeability for the exact 17 sequences" {
                conda run --no-capture-output -n $WindowsCondaEnv python `
                    (Join-Path $Workspace "paper_clean_v28\08_merge_complex_permeability.py") `
                    --permeability_dir $ComplexPermeabilityDir `
                    --all_designs_csv $AllDesigns `
                    --best85_csv $StageManifest `
                    --rmsd_csv $StageRmsd `
                    --out_dir $StagePermeability
            }
        }
    }
    finally {
        Pop-Location
    }

    if ($StartStep -le 8) {
        Assert-RequiredFile $StageRmsd "step 4 complex RMSD"
        Assert-RequiredFile `
            (Join-Path $StageMetrics "complex_best85_highfold_representative.csv") `
            "step 5 complex confidence"
        Assert-RequiredFile `
            (Join-Path $StageMetrics "complex_methylation_site_rmsd_by_design.csv") `
            "step 6 complex methylation-position RMSD"
        Assert-RequiredFile `
            (Join-Path $StagePermeability "complex_permeability_best85.csv") `
            "step 7 complex permeability"

        Invoke-Checked "8/12 Recompute complex naturalized fixed-pose PyRosetta energy" {
            $WorkspaceWsl = Convert-WindowsPathToWslMountPath $Workspace
            Write-Host "Windows complex workspace: $Workspace"
            Write-Host "WSL complex workspace: $WorkspaceWsl"
            $BashCommand = @"
set -euo pipefail
test -d '$WorkspaceWsl'
test -f '$WslCondaRoot/etc/profile.d/conda.sh'
source '$WslCondaRoot/etc/profile.d/conda.sh'
cd '$WorkspaceWsl'
conda run --no-capture-output -n '$PyRosettaEnv' python 'paper_clean_v28/structure_metrics/10_compute_pyrosetta_energy_naturalized.py'
"@
            wsl.exe -d $WslDistribution -- bash -lc $BashCommand
        }
    }

    if ($StartStep -le 9) {
        Invoke-Checked "9/12 Merge complex metrics and enforce the 17-row gate" {
            conda run --no-capture-output -n $WindowsCondaEnv python $FinalizeComplexScript `
                --run_dir $RunDir `
                --all_designs_csv $AllDesigns
        }
    }

    $ComplexFinalCsv = Join-Path $RunDir "temperature05_best17_all_metrics.csv"
    Assert-RequiredFile $ComplexFinalCsv "step 9 complex final table"

    $MonomerOutDir = Join-Path $RunDir "monomer"
    if ($StartStep -le 10) {
        Invoke-Checked "10/12 Recompute all applicable monomer structure/confidence/TM/permeability metrics" {
            conda run --no-capture-output -n $WindowsCondaEnv python $MonomerStructureScript `
                --design_manifest $MonomerDesignManifest `
                --reference_manifest $MonomerReferenceManifest `
                --pdb_dir $MonomerPdbDir `
                --permeability_root $MonomerPermeabilityRoot `
                --out_dir $MonomerOutDir
        }
    }

    $MonomerStructureCsv = Join-Path $MonomerOutDir "monomer_structure_metrics_by_sample.csv"
    Assert-RequiredFile $MonomerStructureCsv "step 10 monomer structure table"

    if ($StartStep -le 11) {
        Invoke-Checked "11/12 Recompute 302 naturalized monomer PyRosetta scores" {
            $RepoRootWsl = Convert-WindowsPathToWslMountPath $RepoRoot
            $MonomerPdbDirWsl = Convert-WindowsPathToWslMountPath $MonomerPdbDir
            $MonomerOutDirWsl = Convert-WindowsPathToWslMountPath $MonomerOutDir
            $MonomerStructureCsvWsl = Convert-WindowsPathToWslMountPath $MonomerStructureCsv
            Write-Host "Windows monomer PDB directory: $MonomerPdbDir"
            Write-Host "WSL monomer PDB directory: $MonomerPdbDirWsl"
            $BashCommand = @"
set -euo pipefail
test -d '$RepoRootWsl'
test -d '$MonomerPdbDirWsl'
test -f '$MonomerStructureCsvWsl'
test -f '$WslCondaRoot/etc/profile.d/conda.sh'
source '$WslCondaRoot/etc/profile.d/conda.sh'
cd '$RepoRootWsl'
conda run --no-capture-output -n '$PyRosettaEnv' python \
  'paper_clean_v28/structure_metrics/19_compute_monomer_pyrosetta_energy.py' \
  --structure_csv '$MonomerStructureCsvWsl' \
  --pdb_dir '$MonomerPdbDirWsl' \
  --out_dir '$MonomerOutDirWsl'
"@
            wsl.exe -d $WslDistribution -- bash -lc $BashCommand
        }
    }

    $MonomerEnergyCsv = Join-Path $MonomerOutDir "monomer_pyrosetta_energy_by_sample.csv"
    Assert-RequiredFile $MonomerEnergyCsv "step 11 monomer paired energy table"

    if ($StartStep -le 12) {
        Invoke-Checked "12/12 Build the final complex + monomer workbook and quality gate" {
            conda run --no-capture-output -n $WindowsCondaEnv python $FinalizeAllScript `
                --run_dir $RunDir `
                --model_summary $MonomerModelSummary
        }
    }

    $FinalWorkbook = Join-Path $RunDir "temperature05_best17_and_monomer_all_metrics.xlsx"
    $FinalReport = Join-Path $RunDir "temperature05_best17_and_monomer_report.txt"
    Write-Host ""
    Write-Host "===== ALL DONE =====" -ForegroundColor Green
    Write-Host "Complex 17-row CSV: $ComplexFinalCsv"
    Write-Host "Monomer 151-row CSV: $(Join-Path $RunDir 'monomer_all_metrics.csv')"
    Write-Host "Final Excel workbook: $FinalWorkbook"
    Write-Host "Final quality report: $FinalReport"
    Write-Host "Console log: $LogPath"
}
finally {
    Stop-Transcript | Out-Null
}
