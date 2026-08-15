param(
    [string]$Python = "",
    [string]$CondaEnvironment = "wain",
    [int]$BatchSize = 16,
    [switch]$AllowCpu,
    [switch]$Force,
    [switch]$ReleaseHandoff
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$V3Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_order_balanced_v3"
$V4Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_peptide_only_v4"
$V5Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_structural_support_v5"
$Checkpoint = Join-Path $V3Root "model\frankenstein_v28_expert_heads_qc.pt"
$TrainJsonl = Join-Path $V3Root "data\train_serine_provenance_corrected.jsonl"
$TestJsonl = Join-Path $V3Root "data\test_serine_provenance_corrected.jsonl"
$V3Generation = Join-Path $V3Root "generation"
$V4Generation = Join-Path $V4Root "generation"
$V5Generation = Join-Path $V5Root "generation"
$AuditOut = Join-Path $V5Root "triple_audit"
$HandoffOut = Join-Path $V5Root "handoff"
$ReviewBundle = Join-Path $V5Root "serine_qc_structural_support_v5_review_bundle.zip"
$Plan = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\target_plan_structure_failures.json"
$V4Rescorer = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\05_rescore_existing_generation_peptide_only.py"
$V5Topup = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\06_top_up_quota_and_finalize_v5.py"
$Auditor = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\04_triple_audit_generation.py"
$Selector = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\03_select_structure_first_handoff.py"
$NativeJsonl = Join-Path $RepoRoot "17_complexes_native.jsonl"
$HistoricalCsv = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\all_designs.csv"
$PriorHandoff = Join-Path $RepoRoot "paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\methylated_new_candidates.csv"

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
    $Stem = "proteinmpnn_serine_v5_$([Guid]::NewGuid().ToString('N'))"
    $ProgramPath = Join-Path ([System.IO.Path]::GetTempPath()) ($Stem + ".py")
    try {
        $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($ProgramPath, $Program, $Utf8NoBom)
        & $PythonPath $ProgramPath
        if ($LASTEXITCODE -ne 0) {
            throw "$Stage failed with exit code $LASTEXITCODE"
        }
    } finally {
        if (Test-Path -LiteralPath $ProgramPath) {
            Remove-Item -LiteralPath $ProgramPath -Force -ErrorAction SilentlyContinue
        }
    }
}

$RequiredInputs = @(
    $Checkpoint,
    $TrainJsonl,
    $TestJsonl,
    (Join-Path $V3Generation "all_candidates.csv"),
    (Join-Path $V3Generation "generation_manifest.json"),
    (Join-Path $V3Generation "target_manifest.csv"),
    $Plan,
    $NativeJsonl,
    $HistoricalCsv,
    $PriorHandoff,
    $V4Rescorer,
    $V5Topup,
    $Auditor,
    $Selector
)
foreach ($Required in $RequiredInputs) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required Ser-QC input is missing: $Required"
    }
}
if ((Test-Path -LiteralPath (Join-Path $V5Generation "generation_manifest.json")) -and -not $Force) {
    throw "V5 output already exists. Review it first, or rerun intentionally with -Force: $V5Generation"
}

$ResolvedPython = Resolve-PythonExecutable
Write-Host "============================================================"
Write-Host "SERINE QC STRUCTURAL-SUPPORT + QUOTA RECOVERY V5"
Write-Host "Repository: $RepoRoot"
Write-Host "Python:     $ResolvedPython"
Write-Host "Model:      reuse the already promoted V3 retrained checkpoint"
Write-Host "Pool:       retain all 11,500 audited V4 natural-sequence rows"
Write-Host "Top-up:     sample only a target that is still below its frozen quota"
Write-Host "Audit:      held-out backbone support + independent three-pass checks"
Write-Host "============================================================"

$ProbeCode = 'import json, numpy, torch; print(json.dumps({"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'
Invoke-PythonProgram $ResolvedPython $ProbeCode "Python/PyTorch preflight"
if (-not $AllowCpu) {
    $ProbeCode = 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)'
    Invoke-PythonProgram $ResolvedPython $ProbeCode "CUDA preflight"
}

Push-Location $RepoRoot
try {
    $V4ManifestPath = Join-Path $V4Generation "generation_manifest.json"
    if (-not (Test-Path -LiteralPath $V4ManifestPath -PathType Leaf)) {
        Write-Host "V4 annotated pool is absent; rebuilding it once from the audited V3 rows."
        $V4Arguments = @(
            $V4Rescorer,
            "--plan", $Plan,
            "--model-path", $Checkpoint,
            "--source-run-dir", $V3Generation,
            "--out-dir", $V4Generation,
            "--native-jsonl", $NativeJsonl,
            "--old-designs-csv", $HistoricalCsv,
            "--prior-designs-csv", $PriorHandoff,
            "--batch-size", 32,
            "--overwrite"
        )
        if ($AllowCpu) { $V4Arguments += @("--device", "auto", "--allow-cpu") }
        else { $V4Arguments += @("--device", "cuda") }
        & $ResolvedPython @V4Arguments
        $V4ExitCode = $LASTEXITCODE
        if (-not (Test-Path -LiteralPath $V4ManifestPath -PathType Leaf)) {
            throw "V4 source reconstruction did not produce its diagnostic manifest (exit $V4ExitCode)"
        }
        if ($V4ExitCode -ne 0) {
            Write-Host "V4 stopped at its old concentration/quota gate as expected; V5 will validate the exact failed checks before continuing." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Reusing completed V4 annotated pool: $V4Generation"
    }

    $TopupArguments = @(
        $V5Topup,
        "--plan", $Plan,
        "--model-path", $Checkpoint,
        "--source-run-dir", $V4Generation,
        "--out-dir", $V5Generation,
        "--native-jsonl", $NativeJsonl,
        "--old-designs-csv", $HistoricalCsv,
        "--prior-designs-csv", $PriorHandoff,
        "--batch-size", $BatchSize
    )
    if ($AllowCpu) { $TopupArguments += @("--device", "auto", "--allow-cpu") }
    else { $TopupArguments += @("--device", "cuda") }
    if ($Force) { $TopupArguments += "--overwrite" }
    & $ResolvedPython @TopupArguments
    Assert-LastExitCode "V5 adaptive quota top-up"

    & $ResolvedPython $Auditor `
        --run-dir $V5Generation `
        --plan $Plan `
        --prior-handoff-csv $PriorHandoff `
        --train-jsonl $TrainJsonl `
        --test-jsonl $TestJsonl `
        --native-jsonl $NativeJsonl `
        --out-dir $AuditOut
    Assert-LastExitCode "Independent V5 three-pass audit"

    $BundleSources = @(
        (Join-Path $V3Root "model\expert_heads_retrain_manifest.json"),
        (Join-Path $V3Root "model\test_metrics_by_residue.csv"),
        (Join-Path $V4Generation "generation_manifest.json"),
        (Join-Path $V5Generation "generation_manifest.json"),
        (Join-Path $V5Generation "generation_summary_by_target.csv"),
        (Join-Path $V5Generation "all_candidates.csv"),
        (Join-Path $V5Generation "unique_candidates.csv"),
        (Join-Path $V5Generation "methylated_new_candidates.csv"),
        (Join-Path $AuditOut "three_pass_generation_audit.json"),
        (Join-Path $AuditOut "three_pass_concentration_by_target.csv"),
        (Join-Path $AuditOut "structural_position_support.json")
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    Compress-Archive -LiteralPath $BundleSources -DestinationPath $ReviewBundle -Force

    if ($ReleaseHandoff) {
        & $ResolvedPython $Selector `
            --run-dir $V5Generation `
            --plan $Plan `
            --triple-audit-json (Join-Path $AuditOut "three_pass_generation_audit.json") `
            --out-dir $HandoffOut `
            --prior-handoff-csv $PriorHandoff
        Assert-LastExitCode "Structure-first shortlist"
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "AUTOMATED V5 QUALITY GATES PASSED" -ForegroundColor Green
Write-Host "Final candidates:     $(Join-Path $V5Generation 'methylated_new_candidates.csv')"
Write-Host "Structural evidence:  $(Join-Path $AuditOut 'structural_position_support.json')"
Write-Host "Manual-review bundle: $ReviewBundle"
if ($ReleaseHandoff) {
    Write-Host "Shang-ge handoff:     $(Join-Path $HandoffOut 'structure_tasks_for_shangge.csv')"
} else {
    Write-Host "Release status:       READY FOR MANUAL SCIENTIFIC REVIEW; no Shang-ge handoff was created" -ForegroundColor Yellow
}
Write-Host "Permeability:         DEFERRED until returned structures pass the structure gate"
