param(
    [string]$Python = "",
    [string]$CondaEnvironment = "wain",
    [int]$TrainingBatchSize = 8,
    [int]$AuditBatchSize = 8,
    [int]$GenerationBatchSize = 8,
    [switch]$AllowCpu,
    [switch]$Force,
    [switch]$ReviewOnly,
    [switch]$ResumeQuota
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$V3Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_order_balanced_v3"
$V6Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_cyclic_representation_v6"
$V6ModelOut = Join-Path $V6Root "model"
$RepresentationAuditOut = Join-Path $V6Root "representation_audit"
$GenerationOut = Join-Path $V6Root "generation"
$TripleAuditOut = Join-Path $V6Root "triple_audit"
$ReviewBundle = Join-Path $V6Root "serine_qc_cyclic_representation_v6_review_bundle.zip"

$ParentCheckpoint = Join-Path $RepoRoot "frankenstein_v28.pt"
$Checkpoint = Join-Path $V6ModelOut "frankenstein_v28_expert_heads_qc.pt"
$ExpertManifest = Join-Path $V6ModelOut "expert_heads_retrain_manifest.json"
$TrainJsonl = Join-Path $V3Root "data\train_serine_provenance_corrected.jsonl"
$TestJsonl = Join-Path $V3Root "data\test_serine_provenance_corrected.jsonl"
$NativeJsonl = Join-Path $RepoRoot "17_complexes_native.jsonl"
$BestCsv = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\best_designs.csv"
$HistoricalCsv = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\all_designs.csv"
$PriorHandoff = Join-Path $RepoRoot "paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\methylated_new_candidates.csv"
$Plan = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\target_plan_cyclic_representation_v6.json"
$Trainer = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\02_retrain_canonical_expert_heads.py"
$RepresentationAuditor = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\07_audit_cyclic_representation_equivariance.py"
$QuotaResumer = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\08_resume_cyclic_representation_v6_quota.py"
$Generator = Join-Path $RepoRoot "paper_clean_v28\rerun_t05\01_generate_t05_multiseed.py"
$TripleAuditor = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\04_triple_audit_generation.py"
$RepresentationAuditReport = Join-Path $RepresentationAuditOut "cyclic_representation_audit.json"
$GenerationManifest = Join-Path $GenerationOut "generation_manifest.json"

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
        [string]$Stage,
        [string[]]$Arguments = @()
    )
    $Stem = "proteinmpnn_serine_v6_$([Guid]::NewGuid().ToString('N'))"
    $ProgramPath = Join-Path ([System.IO.Path]::GetTempPath()) ($Stem + ".py")
    try {
        $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($ProgramPath, $Program, $Utf8NoBom)
        & $PythonPath $ProgramPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Stage failed with exit code $LASTEXITCODE"
        }
    } finally {
        if (Test-Path -LiteralPath $ProgramPath) {
            Remove-Item -LiteralPath $ProgramPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Compress-PortableArchive {
    param(
        [string]$PythonPath,
        [string]$SourceDirectory,
        [string]$DestinationPath,
        [string]$Stage
    )
    $PortableArchiveProgram = @'
import sys
import zipfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
if not source.is_dir():
    raise SystemExit(f"archive source is not a directory: {source}")
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    destination.unlink()
files = sorted(path for path in source.rglob("*") if path.is_file())
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in files:
        archive.write(path, arcname=path.relative_to(source).as_posix())
with zipfile.ZipFile(destination, "r") as archive:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise SystemExit("portable archive contains duplicate member names")
    if any("\\" in name for name in names):
        raise SystemExit("portable archive contains a backslash member path")
    bad_member = archive.testzip()
    if bad_member is not None:
        raise SystemExit(f"portable archive CRC failure: {bad_member}")
'@
    Invoke-PythonProgram `
        -PythonPath $PythonPath `
        -Program $PortableArchiveProgram `
        -Stage $Stage `
        -Arguments @($SourceDirectory, $DestinationPath)
}

if ($TrainingBatchSize -le 0 -or $AuditBatchSize -le 0 -or $GenerationBatchSize -le 0) {
    throw "TrainingBatchSize, AuditBatchSize, and GenerationBatchSize must all be positive"
}
if ($ReviewOnly -and $ResumeQuota) {
    throw "Choose exactly one recovery mode: -ReviewOnly or -ResumeQuota"
}
if ($ResumeQuota -and $Force) {
    throw "Do not combine -ResumeQuota with -Force; quota resume must preserve V6 outputs"
}

$RequiredInputs = @(
    $TrainJsonl,
    $TestJsonl,
    $NativeJsonl,
    $BestCsv,
    $HistoricalCsv,
    $PriorHandoff,
    $Plan,
    $Trainer,
    $RepresentationAuditor,
    $QuotaResumer,
    $Generator,
    $TripleAuditor
)
foreach ($Required in $RequiredInputs) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required V6 input is missing: $Required"
    }
}
if ((-not $ReviewOnly) -and (-not $ResumeQuota) -and -not (Test-Path -LiteralPath $ParentCheckpoint -PathType Leaf)) {
    throw "Canonical parent checkpoint is missing: $ParentCheckpoint"
}
if ($ReviewOnly -or $ResumeQuota) {
    foreach ($RequiredReviewInput in @($Checkpoint, $ExpertManifest, $RepresentationAuditReport)) {
        if (-not (Test-Path -LiteralPath $RequiredReviewInput -PathType Leaf)) {
            throw "Existing-artifact mode requires a completed V6 artifact: $RequiredReviewInput"
        }
    }
}
if (($ReviewOnly -or $ResumeQuota) -and -not (Test-Path -LiteralPath $GenerationManifest -PathType Leaf)) {
    throw "Existing-artifact mode requires an existing V6 generation: $GenerationManifest"
}
if ((-not $ReviewOnly) -and (-not $ResumeQuota) -and (Test-Path -LiteralPath $V6Root)) {
    $ExistingV6Files = @(Get-ChildItem -LiteralPath $V6Root -Force -ErrorAction SilentlyContinue)
    if ($ExistingV6Files.Count -gt 0 -and -not $Force) {
        throw "V6 output already exists. Use -ReviewOnly, or rerun intentionally with -Force: $V6Root"
    }
    if ($ExistingV6Files.Count -gt 0 -and $Force) {
        Remove-Item -LiteralPath $V6Root -Recurse -Force
    }
}

$ResolvedPython = Resolve-PythonExecutable
Write-Host "============================================================"
Write-Host "SERINE QC CYCLIC-REPRESENTATION RECOVERY V6"
Write-Host "Repository: $RepoRoot"
Write-Host "Python:     $ResolvedPython"
Write-Host "Targets:    all 17; no pre-QC methyl annotation is grandfathered"
Write-Host "3AV9:       regenerated because its old T=0.5 structure row had zero methyl tokens"
Write-Host "Training:   all 20 expert heads retrained from canonical V28 with every cyclic start"
Write-Host "Annotation: all cyclic sequence/coordinate starts + all decoder orders"
Write-Host "Mapping:    every probability is mapped back to the original physical residue"
Write-Host "Release:    review bundle only; structure handoff remains blocked"
if ($ReviewOnly) {
    Write-Host "Mode:       re-audit/package existing V6; no GPU scoring or sampling"
} elseif ($ResumeQuota) {
    Write-Host "Mode:       preserve trained V6 + 19,500 draws; top up quota-shortfall targets only"
} else {
    Write-Host "Mode:       representation-augmented retraining, held-out gate, then full regeneration"
}
Write-Host "============================================================"

Push-Location $RepoRoot
try {
    if ((-not $ReviewOnly) -and (-not $ResumeQuota)) {
        $ProbeCode = 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'
        Invoke-PythonProgram $ResolvedPython $ProbeCode "Python/PyTorch preflight"
        if (-not $AllowCpu) {
            $CudaCode = 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)'
            Invoke-PythonProgram $ResolvedPython $CudaCode "CUDA preflight"
        }

        $TrainingArguments = @(
            $Trainer,
            "--model-path", $ParentCheckpoint,
            "--train-jsonl", $TrainJsonl,
            "--test-jsonl", $TestJsonl,
            "--out-dir", $V6ModelOut,
            "--epochs", 80,
            "--batch-size", $TrainingBatchSize,
            "--learning-rate", 0.001,
            "--validation-fraction", 0.20,
            "--early-stopping-patience", 12,
            "--threshold", 0.6,
            "--deployment-temperature", 0.5,
            "--seed", 42,
            "--cyclic-representation-augmentation"
        )
        if ($AllowCpu) { $TrainingArguments += "--allow-cpu" }
        & $ResolvedPython @TrainingArguments
        Assert-LastExitCode "Cyclic-representation expert-head retraining"

        $AuditArguments = @(
            $RepresentationAuditor,
            "--model-path", $Checkpoint,
            "--test-jsonl", $TestJsonl,
            "--native-jsonl", $NativeJsonl,
            "--best-csv", $BestCsv,
            "--plan", $Plan,
            "--out-dir", $RepresentationAuditOut,
            "--batch-size", $AuditBatchSize,
            "--temperature", 0.5,
            "--threshold", 0.6,
            "--overwrite"
        )
        if ($AllowCpu) { $AuditArguments += @("--device", "auto", "--allow-cpu") }
        else { $AuditArguments += @("--device", "cuda") }
        & $ResolvedPython @AuditArguments
        Assert-LastExitCode "Held-out cyclic-representation audit"

        $GenerationArguments = @(
            $Generator,
            "--plan", $Plan,
            "--model_path", $Checkpoint,
            "--native_jsonl", $NativeJsonl,
            "--best_csv", $BestCsv,
            "--old_designs_csv", $HistoricalCsv,
            "--prior_designs_csv", $PriorHandoff,
            "--out_dir", $GenerationOut,
            "--batch_size", $GenerationBatchSize,
            "--overwrite",
            "--defer-permeability-until-structure",
            "--cyclic-representation-ensemble",
            "--representation-audit-json", $RepresentationAuditReport
        )
        if ($AllowCpu) { $GenerationArguments += @("--device", "auto", "--allow-cpu") }
        else { $GenerationArguments += @("--device", "cuda") }
        & $ResolvedPython @GenerationArguments
        Assert-LastExitCode "Full 17-target V6 regeneration"
    } elseif ($ResumeQuota) {
        $ProbeCode = 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'
        Invoke-PythonProgram $ResolvedPython $ProbeCode "Python/PyTorch quota-resume preflight"
        if (-not $AllowCpu) {
            $CudaCode = 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)'
            Invoke-PythonProgram $ResolvedPython $CudaCode "CUDA quota-resume preflight"
        }
        $ResumeArguments = @(
            $QuotaResumer,
            "--plan", $Plan,
            "--model-path", $Checkpoint,
            "--source-run-dir", $GenerationOut,
            "--out-dir", $GenerationOut,
            "--representation-audit-json", $RepresentationAuditReport,
            "--native-jsonl", $NativeJsonl,
            "--old-designs-csv", $HistoricalCsv,
            "--prior-designs-csv", $PriorHandoff,
            "--batch-size", $GenerationBatchSize
        )
        if ($AllowCpu) { $ResumeArguments += @("--device", "auto", "--allow-cpu") }
        else { $ResumeArguments += @("--device", "cuda") }
        & $ResolvedPython @ResumeArguments
        Assert-LastExitCode "V6 shortfall-target quota resume"
    } else {
        Write-Host "GPU step:   skipped; reusing existing V6 rows"
    }

    & $ResolvedPython $TripleAuditor `
        --run-dir $GenerationOut `
        --plan $Plan `
        --prior-handoff-csv $PriorHandoff `
        --train-jsonl $TrainJsonl `
        --test-jsonl $TestJsonl `
        --native-jsonl $NativeJsonl `
        --out-dir $TripleAuditOut
    Assert-LastExitCode "Independent V6 three-pass audit"

    $ReviewStaging = Join-Path $V6Root "review_bundle_staging"
    if (Test-Path -LiteralPath $ReviewStaging) {
        Remove-Item -LiteralPath $ReviewStaging -Recurse -Force
    }
    New-Item -ItemType Directory -Path $ReviewStaging -Force | Out-Null
    $BundleFileMap = [ordered]@{
        "v6_expert_heads_retrain_manifest.json" = $ExpertManifest
        "v6_target_plan.json" = $Plan
        "v6_cyclic_representation_audit.json" = $RepresentationAuditReport
        "v6_heldout_position_probabilities.csv" = (Join-Path $RepresentationAuditOut "heldout_position_probabilities.csv")
        "v6_native_target_representation_summary.csv" = (Join-Path $RepresentationAuditOut "native_target_representation_summary.csv")
        "v6_native_target_representation_probabilities.csv" = (Join-Path $RepresentationAuditOut "native_target_representation_probabilities.csv")
        "v6_generation_manifest.json" = $GenerationManifest
        "v6_target_manifest.csv" = (Join-Path $GenerationOut "target_manifest.csv")
        "v6_generation_summary_by_target.csv" = (Join-Path $GenerationOut "generation_summary_by_target.csv")
        "v6_all_candidates.csv" = (Join-Path $GenerationOut "all_candidates.csv")
        "v6_unique_candidates.csv" = (Join-Path $GenerationOut "unique_candidates.csv")
        "v6_methylated_new_candidates.csv" = (Join-Path $GenerationOut "methylated_new_candidates.csv")
        "v6_three_pass_generation_audit.json" = (Join-Path $TripleAuditOut "three_pass_generation_audit.json")
        "v6_three_pass_concentration_by_target.csv" = (Join-Path $TripleAuditOut "three_pass_concentration_by_target.csv")
        "v6_structural_position_support.json" = (Join-Path $TripleAuditOut "structural_position_support.json")
        "v3_train_serine_provenance_corrected.jsonl" = $TrainJsonl
        "v3_test_serine_provenance_corrected.jsonl" = $TestJsonl
        "native_17_complexes.jsonl" = $NativeJsonl
        "historical_4115_all_designs.csv" = $HistoricalCsv
        "prior_1333_methylated_new_candidates.csv" = $PriorHandoff
    }
    try {
        $ReviewContents = @()
        foreach ($Entry in $BundleFileMap.GetEnumerator()) {
            $ArchiveName = [string]$Entry.Key
            $SourcePath = [string]$Entry.Value
            if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
                throw "Review bundle source is missing: $SourcePath"
            }
            Copy-Item -LiteralPath $SourcePath -Destination (Join-Path $ReviewStaging $ArchiveName) -Force
            $ReviewContents += [ordered]@{
                archive_name = $ArchiveName
                source_path = (Resolve-Path -LiteralPath $SourcePath).Path
                sha256 = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
                bytes = (Get-Item -LiteralPath $SourcePath).Length
            }
        }
        $Generation = Get-Content -LiteralPath $GenerationManifest -Raw | ConvertFrom-Json
        $Representation = Get-Content -LiteralPath $RepresentationAuditReport -Raw | ConvertFrom-Json
        $Triple = Get-Content -LiteralPath (Join-Path $TripleAuditOut "three_pass_generation_audit.json") -Raw | ConvertFrom-Json
        $ReviewManifest = [ordered]@{
            protocol = "serine_qc_cyclic_representation_v6_manual_review_bundle_v1"
            representation_quality_gate = $Representation.quality_gate
            generation_quality_gate = $Generation.quality_gate
            independent_audit_quality_gate = $Triple.quality_gate
            release_status = "HOLD_FOR_MANUAL_SCIENTIFIC_REVIEW_NO_STRUCTURE_HANDOFF"
            content_file_count = $ReviewContents.Count
            contents = $ReviewContents
        }
        $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText(
            (Join-Path $ReviewStaging "review_bundle_manifest.json"),
            (($ReviewManifest | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
            $Utf8NoBom
        )
        Compress-PortableArchive `
            -PythonPath $ResolvedPython `
            -SourceDirectory $ReviewStaging `
            -DestinationPath $ReviewBundle `
            -Stage "Portable V6 review ZIP packaging"
    } finally {
        if (Test-Path -LiteralPath $ReviewStaging) {
            Remove-Item -LiteralPath $ReviewStaging -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "V6 AUTOMATED GATES PASSED; MANUAL REVIEW IS STILL REQUIRED" -ForegroundColor Green
Write-Host "Representation audit: $RepresentationAuditReport"
Write-Host "Final candidates:     $(Join-Path $GenerationOut 'methylated_new_candidates.csv')"
Write-Host "Manual-review bundle: $ReviewBundle"
Write-Host "Shang-ge handoff:     NOT CREATED" -ForegroundColor Yellow
Write-Host "Next step:             upload the V6 review ZIP here; do not send V5 or V6 candidates yet" -ForegroundColor Yellow
