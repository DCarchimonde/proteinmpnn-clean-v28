param(
    [string]$Python = "",
    [string]$CondaEnvironment = "wain",
    [string]$SourceRepo = "",
    [string]$PriorHandoffCsv = "",
    [int]$Epochs = 80,
    [int]$BatchSize = 32,
    [switch]$AllowCpu,
    [switch]$Force,
    [switch]$ReleaseHandoff
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceCommit = "28dff152d83623dfb322480413b7dc889f8537a4"
$SourceUrl = "https://github.com/DCarchimonde/ProteinMPNN.git"
$OutputRoot = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_order_balanced_v3"
$DataOut = Join-Path $OutputRoot "data"
$ModelOut = Join-Path $OutputRoot "model"
$BridgeOut = Join-Path $OutputRoot "bridge"
$GenerationOut = Join-Path $OutputRoot "generation"
$AuditOut = Join-Path $OutputRoot "triple_audit"
$HandoffOut = Join-Path $OutputRoot "handoff"
$AuditBundle = Join-Path $OutputRoot "serine_qc_order_balanced_v3_review_bundle.zip"
$Plan = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\target_plan_structure_failures.json"
$LabelBuilder = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\01_rebuild_provenance_labels.py"
$Trainer = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\02_retrain_canonical_expert_heads.py"
$Bridge = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\03_revalidate_frozen_structures.py"
$Generator = Join-Path $RepoRoot "paper_clean_v28\rerun_t05\01_generate_t05_multiseed.py"
$Auditor = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\04_triple_audit_generation.py"
$Selector = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\03_select_structure_first_handoff.py"
$ParentCheckpoint = Join-Path $RepoRoot "frankenstein_v28.pt"
$CorrectedCheckpoint = Join-Path $ModelOut "frankenstein_v28_expert_heads_qc.pt"
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
    # Windows PowerShell 5.1 can corrupt nested quotes in `python -c` and can
    # hide the useful tail of a native stderr traceback.  Use a temporary .py
    # file plus captured stdout/stderr, as in the validated T=0.5 launcher.
    $Stem = "proteinmpnn_serine_qc_$([Guid]::NewGuid().ToString('N'))"
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

function Prepare-PinnedSourceRepo {
    if (-not [string]::IsNullOrWhiteSpace($SourceRepo)) {
        $Resolved = [System.IO.Path]::GetFullPath($SourceRepo)
        if (-not (Test-Path -LiteralPath (Join-Path $Resolved ".git") -PathType Container)) {
            throw "-SourceRepo is not a Git checkout: $Resolved"
        }
        $Observed = (& git -C $Resolved rev-parse HEAD).Trim()
        Assert-LastExitCode "Source repository revision check"
        if ($Observed -ne $SourceCommit) {
            throw "-SourceRepo must be pinned to $SourceCommit; observed $Observed"
        }
        return $Resolved
    }

    $Managed = Join-Path $RepoRoot ".serine_qc_source\ProteinMPNN"
    if (-not (Test-Path -LiteralPath (Join-Path $Managed ".git") -PathType Container)) {
        if (Test-Path -LiteralPath $Managed) {
            throw "Managed source path exists but is not a Git checkout: $Managed"
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Managed) | Out-Null
        & git clone --filter=blob:none --sparse $SourceUrl $Managed
        Assert-LastExitCode "Sparse source clone"
    }
    & git -C $Managed fetch origin $SourceCommit --depth 1
    Assert-LastExitCode "Pinned source fetch"
    & git -C $Managed sparse-checkout set nmethyl_data/raw_pdb nmethyl_data/training_set nmethyl_data/test_set
    Assert-LastExitCode "Pinned source sparse checkout"
    & git -C $Managed checkout --detach $SourceCommit
    Assert-LastExitCode "Pinned source checkout"
    return $Managed
}

$ResolvedPython = Resolve-PythonExecutable
Write-Host "============================================================"
Write-Host "SERINE QC + ORDER-BALANCED EXPERT/GENERATION RECOVERY V3"
Write-Host "Repository: $RepoRoot"
Write-Host "Python:     $ResolvedPython"
Write-Host "============================================================"

$ProbeCode = 'import json, numpy, torch; print(json.dumps({"python_torch": torch.__version__, "cuda": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'
Invoke-PythonProgram $ResolvedPython $ProbeCode "Python/PyTorch preflight"
if (-not $AllowCpu) {
    $ProbeCode = 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)'
    Invoke-PythonProgram $ResolvedPython $ProbeCode "CUDA preflight"
}

& $ResolvedPython $Generator `
    --plan $Plan `
    --prior_designs_csv $PriorHandoff `
    --validate-prior-designs-only
Assert-LastExitCode "Prior 1,333-row handoff preflight"

$PinnedSource = Prepare-PinnedSourceRepo
$TrainJsonl = Join-Path $PinnedSource "nmethyl_data\training_set\train.jsonl"
$TestJsonl = Join-Path $PinnedSource "nmethyl_data\test_set\test.jsonl"
$RawPdbDir = Join-Path $PinnedSource "nmethyl_data\raw_pdb"
foreach ($Required in @($TrainJsonl, $TestJsonl, $RawPdbDir, $ParentCheckpoint, $Plan, $PriorHandoff)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required input is missing: $Required"
    }
}

Push-Location $RepoRoot
try {
    & $ResolvedPython $LabelBuilder `
        --train-jsonl $TrainJsonl `
        --test-jsonl $TestJsonl `
        --raw-pdb-dir $RawPdbDir `
        --out-dir $DataOut `
        --source-commit $SourceCommit
    Assert-LastExitCode "Ser provenance rebuild"

    $TrainArguments = @(
        $Trainer,
        "--model-path", $ParentCheckpoint,
        "--train-jsonl", (Join-Path $DataOut "train_serine_provenance_corrected.jsonl"),
        "--test-jsonl", (Join-Path $DataOut "test_serine_provenance_corrected.jsonl"),
        "--out-dir", $ModelOut,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize
    )
    if ($AllowCpu) { $TrainArguments += "--allow-cpu" }
    & $ResolvedPython @TrainArguments
    Assert-LastExitCode "Canonical all-expert-head retraining"

    $BridgeArguments = @(
        $Bridge,
        "--model-path", $CorrectedCheckpoint,
        "--plan", $Plan,
        "--native-jsonl", (Join-Path $RepoRoot "17_complexes_native.jsonl"),
        "--out-dir", $BridgeOut
    )
    if ($AllowCpu) { $BridgeArguments += "--allow-cpu" }
    & $ResolvedPython @BridgeArguments
    Assert-LastExitCode "Frozen passed-target final-model bridge"

    $GenerateArguments = @(
        $Generator,
        "--plan", $Plan,
        "--model_path", $CorrectedCheckpoint,
        "--out_dir", $GenerationOut,
        "--batch_size", $BatchSize,
        "--prior_designs_csv", $PriorHandoff,
        "--defer-permeability-until-structure"
    )
    if ($AllowCpu) { $GenerateArguments += @("--device", "auto", "--allow-cpu") }
    else { $GenerateArguments += @("--device", "cuda") }
    if ($Force) { $GenerateArguments += "--overwrite" }
    & $ResolvedPython @GenerateArguments
    $GenerationExitCode = $LASTEXITCODE

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

    if ($GenerationExitCode -ne 0) {
        throw "Failed-target T=0.5 generation was blocked with exit code $GenerationExitCode; upload the review bundle, do not release a handoff"
    }
    if ($AuditExitCode -ne 0) {
        throw "Independent three-pass audit was blocked with exit code $AuditExitCode; upload the review bundle, do not release a handoff"
    }

    if ($ReleaseHandoff) {
        $SelectArguments = @(
            $Selector,
            "--run-dir", $GenerationOut,
            "--plan", $Plan,
            "--out-dir", $HandoffOut,
            "--prior-handoff-csv", $PriorHandoff
        )
        & $ResolvedPython @SelectArguments
        Assert-LastExitCode "Structure-first shortlist"
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "AUTOMATED V3 QUALITY GATES PASSED" -ForegroundColor Green
Write-Host "Corrected checkpoint: $CorrectedCheckpoint"
Write-Host "Frozen-target bridge: $(Join-Path $BridgeOut 'frozen_target_final_model_bridge.csv')"
Write-Host "Manual-review bundle: $AuditBundle"
if ($ReleaseHandoff) {
    Write-Host "Shang-ge handoff:     $(Join-Path $HandoffOut 'structure_tasks_for_shangge.csv')"
} else {
    Write-Host "Release status:       HOLD FOR MANUAL REVIEW; no Shang-ge handoff was created" -ForegroundColor Yellow
}
Write-Host "Permeability:         DEFERRED until returned structures pass the structure gate"
