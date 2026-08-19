param(
    [string]$Python = "",
    [string]$CondaEnvironment = "wain",
    [int]$TrainingBatchSize = 8,
    [int]$AuditBatchSize = 8,
    [int]$ScoringBatchSize = 8,
    [switch]$AllowCpu,
    [switch]$ReviewOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$V3Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_order_balanced_v3"
$V6Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_cyclic_representation_v6"
$V7Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_serine_only_cyclic_v7"
$ModelOut = Join-Path $V7Root "model"
$RepresentationOut = Join-Path $V7Root "representation_audit"
$GenerationOut = Join-Path $V7Root "generation"
$TripleOut = Join-Path $V7Root "triple_audit"
$ReviewBundle = Join-Path $V7Root "serine_qc_serine_only_cyclic_v7_review_bundle.zip"

$ParentCheckpoint = Join-Path $RepoRoot "frankenstein_v28.pt"
$Checkpoint = Join-Path $ModelOut "frankenstein_v28_serine_only_qc.pt"
$ExpertManifest = Join-Path $ModelOut "expert_heads_retrain_manifest.json"
$RepresentationReport = Join-Path $RepresentationOut "cyclic_representation_audit.json"
$GenerationManifest = Join-Path $GenerationOut "generation_manifest.json"
$TripleReport = Join-Path $TripleOut "three_pass_generation_audit.json"
$V6Generation = Join-Path $V6Root "generation"
$V6All = Join-Path $V6Generation "all_candidates.csv"
$V6Manifest = Join-Path $V6Generation "generation_manifest.json"
$V6FormalAudit = Join-Path $V6Generation "formal_target_abstention_audit.json"
$V6ExpertManifest = Join-Path $V6Root "model\expert_heads_retrain_manifest.json"
$V6RepresentationReport = Join-Path $V6Root "representation_audit\cyclic_representation_audit.json"

$TrainJsonl = Join-Path $V3Root "data\train_serine_provenance_corrected.jsonl"
$TestJsonl = Join-Path $V3Root "data\test_serine_provenance_corrected.jsonl"
$NativeJsonl = Join-Path $RepoRoot "17_complexes_native.jsonl"
$BestCsv = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\best_designs.csv"
$HistoricalCsv = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\all_designs.csv"
$PriorHandoff = Join-Path $RepoRoot "paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\methylated_new_candidates.csv"
$Plan = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\target_plan_cyclic_representation_v6.json"
$Trainer = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\02_retrain_canonical_expert_heads.py"
$RepresentationAuditor = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\07_audit_cyclic_representation_equivariance.py"
$Reannotator = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\10_reannotate_v6_pool_serine_only_v7.py"
$TripleAuditor = Join-Path $RepoRoot "paper_clean_v28\serine_qc_retrain\11_triple_audit_serine_only_v7.py"

$V7ExpertProtocol = "canonical_clean_v28_serine_only_corrected_labels_cyclic_representation_augmented_v7"
$V7RepresentationProtocol = "cyclic_representation_equivariance_heldout_gate_v2_serine_only"
$V7RepresentationAuthorization = "SERINE_ONLY_REPAIR_VALIDATED_FOR_ISOLATED_V7_REANNOTATION"
$V7GenerationProtocol = "temperature_0.5_serine_only_cyclic_v7_reannotation_of_preserved_v6_pool"
$ExpectedV6AllSha = "1ab4791c09a1b2428b1a84894d13bb8c4049ba580df05bebd93c263a2e4e634c"
$ExpectedV6ManifestSha = "067a22a2175c97cf483e64967168eefc676389e302c9acc79a66c70e8290711f"

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

function Assert-ExitCode {
    param([string]$Stage)
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
}

function Assert-JsonPass {
    param(
        [string]$Path,
        [string]$Protocol,
        [string]$Stage
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    $Payload = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ([string]$Payload.quality_gate -ne "PASS") {
        throw "$Stage exists but is not PASS. It was preserved for diagnosis: $Path"
    }
    if ([string]$Payload.protocol -ne $Protocol) {
        throw "$Stage uses a stale/wrong protocol: $($Payload.protocol)"
    }
    return $true
}

function Invoke-PythonProgram {
    param(
        [string]$PythonPath,
        [string]$Program,
        [string]$Stage,
        [string[]]$Arguments = @()
    )
    $Stem = "proteinmpnn_serine_v7_$([Guid]::NewGuid().ToString('N'))"
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
        [string]$DestinationPath
    )
    $Program = @'
import sys
import zipfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
files = sorted(path for path in source.rglob("*") if path.is_file())
if not files:
    raise SystemExit("review staging directory is empty")
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix(destination.suffix + ".tmp")
if temporary.exists():
    temporary.unlink()
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in files:
        archive.write(path, arcname=path.relative_to(source).as_posix())
with zipfile.ZipFile(temporary, "r") as archive:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise SystemExit("duplicate ZIP member names")
    if any("\\" in name for name in names):
        raise SystemExit("non-portable ZIP member path")
    bad = archive.testzip()
    if bad is not None:
        raise SystemExit(f"ZIP CRC failure: {bad}")
temporary.replace(destination)
'@
    Invoke-PythonProgram `
        -PythonPath $PythonPath `
        -Program $Program `
        -Stage "Portable V7 review ZIP packaging" `
        -Arguments @($SourceDirectory, $DestinationPath)
}

if ($TrainingBatchSize -le 0 -or $AuditBatchSize -le 0 -or $ScoringBatchSize -le 0) {
    throw "TrainingBatchSize, AuditBatchSize, and ScoringBatchSize must be positive"
}

$RequiredInputs = @(
    $ParentCheckpoint,
    $TrainJsonl,
    $TestJsonl,
    $NativeJsonl,
    $BestCsv,
    $HistoricalCsv,
    $PriorHandoff,
    $Plan,
    $Trainer,
    $RepresentationAuditor,
    $Reannotator,
    $TripleAuditor,
    $V6All,
    $V6Manifest,
    $V6FormalAudit,
    $V6ExpertManifest,
    $V6RepresentationReport
)
foreach ($Required in $RequiredInputs) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required input is missing: $Required"
    }
}

$ParentHashBefore = (Get-FileHash -LiteralPath $ParentCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
$ObservedV6AllSha = (Get-FileHash -LiteralPath $V6All -Algorithm SHA256).Hash.ToLowerInvariant()
$ObservedV6ManifestSha = (Get-FileHash -LiteralPath $V6Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ObservedV6AllSha -ne $ExpectedV6AllSha) {
    throw "V6 all_candidates.csv is not the uploaded audited 31,500-row pool: $ObservedV6AllSha"
}
if ($ObservedV6ManifestSha -ne $ExpectedV6ManifestSha) {
    throw "V6 generation_manifest.json is not the uploaded audited result: $ObservedV6ManifestSha"
}

$ResolvedPython = Resolve-PythonExecutable
Write-Host "============================================================"
Write-Host "SERINE QC PROVENANCE-SCOPED RECOVERY V7"
Write-Host "Repository: $RepoRoot"
Write-Host "Python:     $ResolvedPython"
Write-Host "Training:   Ser expert only; all other 19 experts and shared tensors frozen"
Write-Host "Sampling:   none; retain and reannotate the audited 31,500 V6 natural rows"
Write-Host "3ZGC:       no formal abstention; all 17 targets must yield a strict >0.6 candidate"
Write-Host "Position:   cyclic starts map back to physical residues; 3AV homolog site audited"
Write-Host "Release:    manual-review ZIP only; no Shang-ge handoff or permeability run"
Write-Host "============================================================"

Push-Location $RepoRoot
try {
    $ModelReady = Assert-JsonPass $ExpertManifest $V7ExpertProtocol "V7 Ser-only model"
    if (-not $ModelReady) {
        if ($ReviewOnly) {
            throw "-ReviewOnly requires a completed V7 model: $ExpertManifest"
        }
        if ((Test-Path -LiteralPath $ModelOut) -and @(Get-ChildItem -LiteralPath $ModelOut -Force).Count -gt 0) {
            throw "Partial V7 model output exists and was preserved. Inspect it before retrying: $ModelOut"
        }
        $Probe = 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'
        Invoke-PythonProgram $ResolvedPython $Probe "Python/PyTorch preflight"
        if (-not $AllowCpu) {
            Invoke-PythonProgram $ResolvedPython 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)' "CUDA preflight"
        }
        $TrainingArguments = @(
            $Trainer,
            "--model-path", $ParentCheckpoint,
            "--train-jsonl", $TrainJsonl,
            "--test-jsonl", $TestJsonl,
            "--out-dir", $ModelOut,
            "--epochs", 80,
            "--batch-size", $TrainingBatchSize,
            "--learning-rate", 0.001,
            "--validation-fraction", 0.20,
            "--early-stopping-patience", 12,
            "--threshold", 0.6,
            "--deployment-temperature", 0.5,
            "--seed", 42,
            "--expert-scope", "serine-only",
            "--cyclic-representation-augmentation"
        )
        if ($AllowCpu) { $TrainingArguments += "--allow-cpu" }
        & $ResolvedPython @TrainingArguments
        Assert-ExitCode "Ser-only cyclic V7 retraining"
        $ModelReady = Assert-JsonPass $ExpertManifest $V7ExpertProtocol "V7 Ser-only model"
    } else {
        Write-Host "Model step: reused passed V7 Ser-only checkpoint"
    }

    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Passed V7 manifest has no promoted checkpoint: $Checkpoint"
    }
    $Expert = Get-Content -LiteralPath $ExpertManifest -Raw | ConvertFrom-Json
    $ExpectedChanged = @("experts.15.bias", "experts.15.weight") | Sort-Object
    $ObservedChanged = @($Expert.changed_state_keys | ForEach-Object { [string]$_ }) | Sort-Object
    if (($ObservedChanged -join "|") -ne ($ExpectedChanged -join "|")) {
        throw "V7 changed tensors are not exactly the Ser weight+bias: $($ObservedChanged -join ', ')"
    }
    if ([double]$Expert.maximum_non_ser_probability_difference_from_parent -ne 0.0) {
        throw "V7 changed a non-Ser held-out probability"
    }

    $RepresentationReady = Assert-JsonPass $RepresentationReport $V7RepresentationProtocol "V7 representation audit"
    if (-not $RepresentationReady) {
        if ($ReviewOnly) {
            throw "-ReviewOnly requires a completed representation audit: $RepresentationReport"
        }
        $AuditArguments = @(
            $RepresentationAuditor,
            "--model-path", $Checkpoint,
            "--required-expert-protocol", $V7ExpertProtocol,
            "--test-jsonl", $TestJsonl,
            "--native-jsonl", $NativeJsonl,
            "--best-csv", $BestCsv,
            "--plan", $Plan,
            "--out-dir", $RepresentationOut,
            "--batch-size", $AuditBatchSize,
            "--temperature", 0.5,
            "--threshold", 0.6,
            "--overwrite"
        )
        if ($AllowCpu) { $AuditArguments += @("--device", "auto", "--allow-cpu") }
        else { $AuditArguments += @("--device", "cuda") }
        & $ResolvedPython @AuditArguments
        Assert-ExitCode "V7 held-out cyclic-representation audit"
        $RepresentationReady = Assert-JsonPass $RepresentationReport $V7RepresentationProtocol "V7 representation audit"
    } else {
        Write-Host "Representation step: reused passed V7 audit"
    }
    $Representation = Get-Content -LiteralPath $RepresentationReport -Raw | ConvertFrom-Json
    if ([string]$Representation.release_authorization -ne $V7RepresentationAuthorization) {
        throw "V7 representation audit has no reannotation authorization"
    }

    $GenerationReady = Assert-JsonPass $GenerationManifest $V7GenerationProtocol "V7 reannotation"
    if (-not $GenerationReady) {
        if ($ReviewOnly) {
            throw "-ReviewOnly requires completed V7 generation: $GenerationManifest"
        }
        if ((Test-Path -LiteralPath $GenerationManifest -PathType Leaf)) {
            throw "A failed V7 generation manifest was preserved. Do not rerun blindly: $GenerationManifest"
        }
        $ScoringArguments = @(
            $Reannotator,
            "--plan", $Plan,
            "--model-path", $Checkpoint,
            "--expert-manifest", $ExpertManifest,
            "--representation-audit-json", $RepresentationReport,
            "--source-run-dir", $V6Generation,
            "--out-dir", $GenerationOut,
            "--native-jsonl", $NativeJsonl,
            "--old-designs-csv", $HistoricalCsv,
            "--prior-designs-csv", $PriorHandoff,
            "--batch-size", $ScoringBatchSize,
            "--overwrite"
        )
        if ($AllowCpu) { $ScoringArguments += @("--device", "auto", "--allow-cpu") }
        else { $ScoringArguments += @("--device", "cuda") }
        & $ResolvedPython @ScoringArguments
        Assert-ExitCode "V7 preserved-pool reannotation"
        $GenerationReady = Assert-JsonPass $GenerationManifest $V7GenerationProtocol "V7 reannotation"
    } else {
        Write-Host "Scoring step: reused passed V7 reannotation"
    }

    & $ResolvedPython $TripleAuditor `
        --run-dir $GenerationOut `
        --plan $Plan `
        --native-jsonl $NativeJsonl `
        --historical-designs-csv $HistoricalCsv `
        --prior-handoff-csv $PriorHandoff `
        --out-dir $TripleOut
    Assert-ExitCode "Independent V7 three-pass audit"

    $Triple = Get-Content -LiteralPath $TripleReport -Raw | ConvertFrom-Json
    if ([string]$Triple.quality_gate -ne "PASS") {
        throw "Independent V7 three-pass audit is not PASS"
    }
    $Generation = Get-Content -LiteralPath $GenerationManifest -Raw | ConvertFrom-Json
    if ([int]$Generation.targets_with_signature_candidate -ne 17) {
        throw "V7 does not cover all 17 targets"
    }
    if (@($Generation.targets_formally_abstained).Count -ne 0) {
        throw "V7 incorrectly contains formal target abstention"
    }

    $ReviewStaging = Join-Path $V7Root "review_bundle_staging"
    if (Test-Path -LiteralPath $ReviewStaging) {
        Remove-Item -LiteralPath $ReviewStaging -Recurse -Force
    }
    New-Item -ItemType Directory -Path $ReviewStaging -Force | Out-Null
    try {
        $BundleFileMap = [ordered]@{
            "v7_expert_manifest.json" = $ExpertManifest
            "v7_training_history.csv" = (Join-Path $ModelOut "training_history.csv")
            "v7_test_metrics_by_residue.csv" = (Join-Path $ModelOut "test_metrics_by_residue.csv")
            "v7_test_position_probabilities.csv" = (Join-Path $ModelOut "test_position_probabilities.csv")
            "v7_representation_audit.json" = $RepresentationReport
            "v7_heldout_position_probabilities.csv" = (Join-Path $RepresentationOut "heldout_position_probabilities.csv")
            "v7_native_target_representation_summary.csv" = (Join-Path $RepresentationOut "native_target_representation_summary.csv")
            "v7_generation_manifest.json" = $GenerationManifest
            "v7_target_manifest.csv" = (Join-Path $GenerationOut "target_manifest.csv")
            "v7_generation_summary_by_target.csv" = (Join-Path $GenerationOut "generation_summary_by_target.csv")
            "v7_all_candidates.csv" = (Join-Path $GenerationOut "all_candidates.csv")
            "v7_unique_candidates.csv" = (Join-Path $GenerationOut "unique_candidates.csv")
            "v7_methylated_new_candidates.csv" = (Join-Path $GenerationOut "methylated_new_candidates.csv")
            "v7_three_pass_generation_audit.json" = $TripleReport
            "v7_three_pass_concentration_by_target.csv" = (Join-Path $TripleOut "three_pass_concentration_by_target.csv")
            "v7_av_family_physical_position_support.json" = (Join-Path $TripleOut "av_family_physical_position_support.json")
            "source_v6_formal_target_abstention_audit.json" = $V6FormalAudit
            "source_v6_expert_manifest.json" = $V6ExpertManifest
            "source_v6_representation_audit.json" = $V6RepresentationReport
            "serine_provenance_corrected_train.jsonl" = $TrainJsonl
            "serine_provenance_corrected_test.jsonl" = $TestJsonl
            "native_17_complexes.jsonl" = $NativeJsonl
        }
        $Contents = @()
        foreach ($Entry in $BundleFileMap.GetEnumerator()) {
            $ArchiveName = [string]$Entry.Key
            $SourcePath = [string]$Entry.Value
            if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
                throw "Review bundle source is missing: $SourcePath"
            }
            Copy-Item -LiteralPath $SourcePath -Destination (Join-Path $ReviewStaging $ArchiveName) -Force
            $Contents += [ordered]@{
                archive_name = $ArchiveName
                source_path = (Resolve-Path -LiteralPath $SourcePath).Path
                sha256 = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
                bytes = (Get-Item -LiteralPath $SourcePath).Length
            }
        }
        $ReviewManifest = [ordered]@{
            protocol = "serine_qc_serine_only_cyclic_v7_manual_review_bundle_v1"
            expert_quality_gate = $Expert.quality_gate
            representation_quality_gate = $Representation.quality_gate
            generation_quality_gate = $Generation.quality_gate
            independent_three_pass_quality_gate = $Triple.quality_gate
            target_coverage = "$($Generation.targets_with_signature_candidate)/17"
            targets_formally_abstained = @()
            release_status = "HOLD_FOR_MANUAL_SCIENTIFIC_REVIEW_NO_STRUCTURE_HANDOFF"
            content_file_count = $Contents.Count
            contents = $Contents
        }
        $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText(
            (Join-Path $ReviewStaging "review_bundle_manifest.json"),
            (($ReviewManifest | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
            $Utf8NoBom
        )
        Compress-PortableArchive $ResolvedPython $ReviewStaging $ReviewBundle
    } finally {
        if (Test-Path -LiteralPath $ReviewStaging) {
            Remove-Item -LiteralPath $ReviewStaging -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    $ParentHashAfter = (Get-FileHash -LiteralPath $ParentCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ParentHashAfter -ne $ParentHashBefore) {
        throw "Canonical frankenstein_v28.pt changed during V7; release is blocked"
    }
    $V6AllHashAfter = (Get-FileHash -LiteralPath $V6All -Algorithm SHA256).Hash.ToLowerInvariant()
    $V6ManifestHashAfter = (Get-FileHash -LiteralPath $V6Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($V6AllHashAfter -ne $ObservedV6AllSha -or $V6ManifestHashAfter -ne $ObservedV6ManifestSha) {
        throw "The read-only V6 source pool changed during V7; release is blocked"
    }
    if (-not (Test-Path -LiteralPath $ReviewBundle -PathType Leaf)) {
        throw "V7 review bundle was not created"
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "V7 ALL AUTOMATED GATES PASSED; MANUAL SCIENTIFIC REVIEW IS NEXT" -ForegroundColor Green
Write-Host "Target coverage:       17/17; no formal abstention"
Write-Host "Final candidates:      $(Join-Path $GenerationOut 'methylated_new_candidates.csv')"
Write-Host "Three-pass audit:      $TripleReport"
Write-Host "Manual-review bundle:  $ReviewBundle"
Write-Host "Shang-ge handoff:      NOT CREATED" -ForegroundColor Yellow
Write-Host "Permeability:           DEFERRED until returned structures pass both RMSD gates" -ForegroundColor Yellow
