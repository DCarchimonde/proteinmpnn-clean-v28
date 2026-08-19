param(
    [string]$Python = "",
    [string]$CondaEnvironment = "wain",
    [string]$PriorHandoffCsv = "",
    [int]$BatchSize = 32,
    [switch]$AllowCpu,
    [switch]$Force,
    [switch]$ReleaseHandoff
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$V3Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_order_balanced_v3"
$V4Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_peptide_only_v4"
$ModelOut = Join-Path $V3Root "model"
$Checkpoint = Join-Path $ModelOut "frankenstein_v28_expert_heads_qc.pt"
$SourceGeneration = Join-Path $V3Root "generation"
$BridgeOut = Join-Path $V4Root "bridge"
$GenerationOut = Join-Path $V4Root "generation"
$AuditOut = Join-Path $V4Root "triple_audit"
$HandoffOut = Join-Path $V4Root "handoff"
$AuditBundle = Join-Path $V4Root "serine_qc_peptide_only_v4_review_bundle.zip"
$Plan = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\target_plan_structure_failures.json"
$Bridge = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\03_revalidate_frozen_structures.py"
$Rescorer = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\05_rescore_existing_generation_peptide_only.py"
$Auditor = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\04_triple_audit_generation.py"
$Selector = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\03_select_structure_first_handoff.py"
$NativeJsonl = Join-Path $RepoRoot "17_complexes_native.jsonl"
$HistoricalCsv = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\all_designs.csv"
if ([string]::IsNullOrWhiteSpace($PriorHandoffCsv)) {
    $PriorHandoff = Join-Path $RepoRoot "paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\methylated_new_candidates.csv"
} elseif ([System.IO.Path]::IsPathRooted($PriorHandoffCsv)) {
    $PriorHandoff = [System.IO.Path]::GetFullPath($PriorHandoffCsv)
} else {
    $PriorHandoff = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PriorHandoffCsv))
}

function Assert-LastExitCode {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Resolve-PythonExecutable {
    if (-not [string]::IsNullOrWhiteSpace($Python)) {
        return $Python
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $Candidate = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            return $Candidate
        }
    }
    $Conda = Get-Command conda.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $Conda) {
        try {
            $Info = (& $Conda.Source env list --json | Out-String) | ConvertFrom-Json
            foreach ($EnvironmentPath in @($Info.envs)) {
                if ((Split-Path -Leaf $EnvironmentPath) -ieq $CondaEnvironment) {
                    $Candidate = Join-Path $EnvironmentPath "python.exe"
                    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
                        return $Candidate
                    }
                }
            }
        } catch {
            Write-Host "Conda environment discovery warning: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    return "python"
}

function Invoke-PythonProgram {
    param(
        [string]$PythonPath,
        [string]$Program,
        [string]$Stage
    )
    $Stem = "proteinmpnn_serine_v4_$([Guid]::NewGuid().ToString('N'))"
    $ProgramPath = Join-Path ([System.IO.Path]::GetTempPath()) ($Stem + ".py")
    $StdoutPath = Join-Path ([System.IO.Path]::GetTempPath()) ($Stem + ".stdout.txt")
    $StderrPath = Join-Path ([System.IO.Path]::GetTempPath()) ($Stem + ".stderr.txt")
    try {
        $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($ProgramPath, $Program, $Utf8NoBom)
        & $PythonPath $ProgramPath 1> $StdoutPath 2> $StderrPath
        $ProgramExitCode = $LASTEXITCODE
        foreach ($OutputPath in @($StdoutPath, $StderrPath)) {
            if (Test-Path -LiteralPath $OutputPath -PathType Leaf) {
                Get-Content -LiteralPath $OutputPath | ForEach-Object { Write-Host $_ }
            }
        }
        if ($ProgramExitCode -ne 0) {
            throw "$Stage failed with exit code $ProgramExitCode"
        }
    } finally {
        foreach ($TemporaryPath in @($ProgramPath, $StdoutPath, $StderrPath)) {
            if (Test-Path -LiteralPath $TemporaryPath) {
                Remove-Item -LiteralPath $TemporaryPath -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

$RequiredInputs = @(
    $Checkpoint,
    (Join-Path $SourceGeneration "all_candidates.csv"),
    (Join-Path $SourceGeneration "generation_manifest.json"),
    (Join-Path $SourceGeneration "target_manifest.csv"),
    $Plan,
    $NativeJsonl,
    $HistoricalCsv,
    $PriorHandoff,
    $Bridge,
    $Rescorer,
    $Auditor,
    $Selector
)
foreach ($Required in $RequiredInputs) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required V3 recovery input is missing: $Required"
    }
}
if ((Test-Path -LiteralPath (Join-Path $GenerationOut "generation_manifest.json")) -and -not $Force) {
    throw "V4 output already exists. Review it first, or rerun intentionally with -Force: $GenerationOut"
}
if ($Force) {
    foreach ($StaleOutput in @(
        (Join-Path $AuditOut "three_pass_generation_audit.json"),
        (Join-Path $AuditOut "three_pass_concentration_by_target.csv"),
        $AuditBundle
    )) {
        if (Test-Path -LiteralPath $StaleOutput -PathType Leaf) {
            Remove-Item -LiteralPath $StaleOutput -Force
        }
    }
}

$ResolvedPython = Resolve-PythonExecutable
Write-Host "============================================================"
Write-Host "SERINE QC PEPTIDE-ONLY ANNOTATION RECOVERY V4"
Write-Host "Repository: $RepoRoot"
Write-Host "Python:     $ResolvedPython"
Write-Host "Action:     reuse V3 checkpoint + 11,500 natural sequences"
Write-Host "Skipped:    no retraining; no base-sequence resampling"
Write-Host "============================================================"

$ProbeCode = 'import json, numpy, torch; print(json.dumps({"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'
Invoke-PythonProgram $ResolvedPython $ProbeCode "Python/PyTorch preflight"
if (-not $AllowCpu) {
    $ProbeCode = 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)'
    Invoke-PythonProgram $ResolvedPython $ProbeCode "CUDA preflight"
}

Push-Location $RepoRoot
try {
    $BridgeArguments = @(
        $Bridge,
        "--model-path", $Checkpoint,
        "--plan", $Plan,
        "--native-jsonl", $NativeJsonl,
        "--out-dir", $BridgeOut
    )
    if ($AllowCpu) { $BridgeArguments += "--allow-cpu" }
    & $ResolvedPython @BridgeArguments
    Assert-LastExitCode "Frozen passed-target peptide-only bridge"

    $RecoveryArguments = @(
        $Rescorer,
        "--plan", $Plan,
        "--model-path", $Checkpoint,
        "--source-run-dir", $SourceGeneration,
        "--out-dir", $GenerationOut,
        "--native-jsonl", $NativeJsonl,
        "--old-designs-csv", $HistoricalCsv,
        "--prior-designs-csv", $PriorHandoff,
        "--batch-size", $BatchSize
    )
    if ($AllowCpu) { $RecoveryArguments += @("--device", "auto", "--allow-cpu") }
    else { $RecoveryArguments += @("--device", "cuda") }
    if ($Force) { $RecoveryArguments += "--overwrite" }
    & $ResolvedPython @RecoveryArguments
    $RecoveryExitCode = $LASTEXITCODE

    $AuditExitCode = 1
    if (Test-Path -LiteralPath (Join-Path $GenerationOut "generation_manifest.json") -PathType Leaf) {
        & $ResolvedPython $Auditor `
            --run-dir $GenerationOut `
            --plan $Plan `
            --prior-handoff-csv $PriorHandoff `
            --out-dir $AuditOut
        $AuditExitCode = $LASTEXITCODE
    }

    $BundleSources = @(
        (Join-Path $ModelOut "expert_heads_retrain_manifest.json"),
        (Join-Path $ModelOut "training_history.csv"),
        (Join-Path $ModelOut "test_metrics_by_residue.csv"),
        (Join-Path $ModelOut "test_position_probabilities.csv"),
        (Join-Path $BridgeOut "frozen_target_bridge_manifest.json"),
        (Join-Path $BridgeOut "frozen_target_final_model_bridge.csv"),
        (Join-Path $GenerationOut "generation_manifest.json"),
        (Join-Path $GenerationOut "generation_summary_by_target.csv"),
        (Join-Path $GenerationOut "all_candidates.csv"),
        (Join-Path $GenerationOut "unique_candidates.csv"),
        (Join-Path $GenerationOut "methylated_new_candidates.csv"),
        (Join-Path $AuditOut "three_pass_generation_audit.json"),
        (Join-Path $AuditOut "three_pass_concentration_by_target.csv")
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    if ($BundleSources.Count -gt 0) {
        Compress-Archive -LiteralPath $BundleSources -DestinationPath $AuditBundle -Force
    }

    if ($RecoveryExitCode -ne 0) {
        throw "V4 recovery was blocked with exit code $RecoveryExitCode; outputs and review bundle were preserved"
    }
    if ($AuditExitCode -ne 0) {
        throw "Independent V4 three-pass audit was blocked with exit code $AuditExitCode; review bundle was preserved"
    }

    if ($ReleaseHandoff) {
        & $ResolvedPython $Selector `
            --run-dir $GenerationOut `
            --plan $Plan `
            --out-dir $HandoffOut `
            --prior-handoff-csv $PriorHandoff
        Assert-LastExitCode "Structure-first shortlist"
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "AUTOMATED V4 QUALITY GATES PASSED" -ForegroundColor Green
Write-Host "Reused checkpoint:    $Checkpoint"
Write-Host "Reused natural rows:  $(Join-Path $SourceGeneration 'all_candidates.csv')"
Write-Host "Corrected candidates: $(Join-Path $GenerationOut 'methylated_new_candidates.csv')"
Write-Host "Manual-review bundle: $AuditBundle"
if ($ReleaseHandoff) {
    Write-Host "Shang-ge handoff:     $(Join-Path $HandoffOut 'structure_tasks_for_shangge.csv')"
} else {
    Write-Host "Release status:       HOLD FOR MANUAL REVIEW; no Shang-ge handoff was created" -ForegroundColor Yellow
}
Write-Host "Permeability:         DEFERRED until returned structures pass the structure gate"
