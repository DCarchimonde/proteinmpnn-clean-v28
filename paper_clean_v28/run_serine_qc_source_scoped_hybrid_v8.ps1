param(
    [string]$Python = "",
    [string]$CondaEnvironment = "wain",
    [int]$AuditBatchSize = 8,
    [int]$ScoringBatchSize = 8,
    [int]$SearchBatchSize = 64,
    [int]$BaseBatchSize = 32,
    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"
$LauncherProgram = $MyInvocation.MyCommand.Path
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptRoot

$V3Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_order_balanced_v3"
$V6Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_cyclic_representation_v6"
$V7Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_serine_only_cyclic_v7"
$V8Root = Join-Path $RepoRoot "paper_clean_v28_outputs\serine_qc_source_scoped_hybrid_v8"

$ModelOut = Join-Path $V8Root "model"
$RepresentationOut = Join-Path $V8Root "representation_audit"
$BaselineOut = Join-Path $V8Root "generation_baseline"
$SearchOut = Join-Path $V8Root "directed_search"
$RecoveredOut = Join-Path $V8Root "generation_recovered"
$RecoveredAuditOut = Join-Path $V8Root "triple_audit_recovered"
$ReviewBundle = Join-Path $V8Root "serine_qc_source_scoped_hybrid_v8_review_bundle.zip"

$CanonicalCheckpoint = Join-Path $RepoRoot "frankenstein_v28.pt"
$V6Checkpoint = Join-Path $V6Root "model\frankenstein_v28_expert_heads_qc.pt"
$V6ExpertManifest = Join-Path $V6Root "model\expert_heads_retrain_manifest.json"
$V6Representation = Join-Path $V6Root "representation_audit\cyclic_representation_audit.json"
$V6Generation = Join-Path $V6Root "generation"
$V6AllCandidates = Join-Path $V6Generation "all_candidates.csv"
$V6GenerationManifest = Join-Path $V6Generation "generation_manifest.json"
$V6FormalAbstention = Join-Path $V6Generation "formal_target_abstention_audit.json"

$V7Checkpoint = Join-Path $V7Root "model\frankenstein_v28_serine_only_qc.pt"
$V7ExpertManifest = Join-Path $V7Root "model\expert_heads_retrain_manifest.json"
$V7Representation = Join-Path $V7Root "representation_audit\cyclic_representation_audit.json"
$V7Generation = Join-Path $V7Root "generation"
$V7GenerationManifest = Join-Path $V7Generation "generation_manifest.json"

$V8Checkpoint = Join-Path $ModelOut "frankenstein_v28_source_scoped_hybrid_v8.pt"
$V8ExpertManifest = Join-Path $ModelOut "expert_source_composition_manifest.json"
$V8Representation = Join-Path $RepresentationOut "cyclic_representation_audit.json"
$V8BaselineManifest = Join-Path $BaselineOut "generation_manifest.json"
$V8SearchManifest = Join-Path $SearchOut "directed_search_manifest.json"
$V8RecoveredManifest = Join-Path $RecoveredOut "generation_manifest.json"
$V8RecoveredAudit = Join-Path $RecoveredAuditOut "three_pass_generation_audit.json"

$TestJsonl = Join-Path $V3Root "data\test_serine_provenance_corrected.jsonl"
$NativeJsonl = Join-Path $RepoRoot "17_complexes_native.jsonl"
$BestCsv = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\best_designs.csv"
$HistoricalCsv = Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\all_designs.csv"
$PriorHandoff = Join-Path $RepoRoot "paper_clean_v28_outputs\rerun_temperature_0.5_multiseed\methylated_new_candidates.csv"
$Plan = Join-Path $ScriptRoot "serine_qc_retrain\target_plan_cyclic_representation_v6.json"

$Composer = Join-Path $ScriptRoot "serine_qc_retrain\12_compose_source_scoped_hybrid_v8.py"
$RepresentationAuditor = Join-Path $ScriptRoot "serine_qc_retrain\13_audit_source_scoped_hybrid_v8.py"
$Reannotator = Join-Path $ScriptRoot "serine_qc_retrain\10_reannotate_v6_pool_serine_only_v7.py"
$DirectedSearch = Join-Path $ScriptRoot "serine_qc_retrain\14_directed_recovery_search_v8.py"
$RecoveryFinalizer = Join-Path $ScriptRoot "serine_qc_retrain\15_finalize_and_audit_recovery_v8.py"
$PositionAuditor = Join-Path $ScriptRoot "serine_qc_retrain\11_triple_audit_serine_only_v7.py"
$TrainerProgram = Join-Path $ScriptRoot "serine_qc_retrain\02_retrain_canonical_expert_heads.py"
$EquivarianceAuditor = Join-Path $ScriptRoot "serine_qc_retrain\07_audit_cyclic_representation_equivariance.py"
$GeneratorProgram = Join-Path $ScriptRoot "rerun_t05\01_generate_t05_multiseed.py"
$CommonProgram = Join-Path $ScriptRoot "clean_v28_common.py"
$ModelUtilsProgram = Join-Path $RepoRoot "model_utils.py"
$NmethylConfigProgram = Join-Path $RepoRoot "nmethyl\utils\nmethyl_config.py"

$V6ExpertProtocol = "canonical_clean_v28_all_expert_heads_corrected_labels_cyclic_representation_augmented_v6"
$V7ExpertProtocol = "canonical_clean_v28_serine_only_corrected_labels_cyclic_representation_augmented_v7"
$V8ExpertProtocol = "canonical_shared_v6_non_ser_v7_ser_cyclic_representation_hybrid_v8"
$V6RepresentationProtocol = "cyclic_representation_equivariance_heldout_gate_v1"
$V7RepresentationProtocol = "cyclic_representation_equivariance_heldout_gate_v2_serine_only"
$V8RepresentationProtocol = "cyclic_representation_frozen_audit_source_scoped_hybrid_v8"
$V6RepresentationAuthorization = "REPRESENTATION_ENSEMBLE_VALIDATED_FOR_ISOLATED_V6_REGENERATION"
$V7RepresentationAuthorization = "SERINE_ONLY_REPAIR_VALIDATED_FOR_ISOLATED_V7_REANNOTATION"
$V8RepresentationAuthorization = "SOURCE_SCOPED_HYBRID_V8_AUTHORIZED_FOR_DIRECTED_RECOVERY"
$V7GenerationProtocol = "temperature_0.5_serine_only_cyclic_v7_reannotation_of_preserved_v6_pool"
$V8BaselineProtocol = "temperature_0.5_source_scoped_hybrid_v8_reannotation_of_preserved_v6_pool"
$V8SearchProtocol = "deterministic_missing_target_directed_recovery_v8"
$V8RecoveredProtocol = "immutable_baseline_plus_directed_recovery_overlay_v8"
$V8RecoveredAuditProtocol = "independent_three_pass_source_scoped_recovery_v8"
$NaturalExpertTokensJson = '["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]'
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

function Get-Sha256 {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Cannot hash missing file: $Path"
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required JSON is missing: $Path"
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Invoke-PythonStage {
    param(
        [string]$PythonPath,
        [string]$Stage,
        [string[]]$Arguments
    )
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
}

function Assert-EmptyOrAbsentDirectory {
    param(
        [string]$Path,
        [string]$Stage
    )
    if ((Test-Path -LiteralPath $Path -PathType Container) -and
        @(Get-ChildItem -LiteralPath $Path -ErrorAction SilentlyContinue).Count -gt 0) {
        throw "$Stage has a partial directory. It was preserved for inspection: $Path"
    }
}

function Assert-SameStringSet {
    param(
        [object[]]$Observed,
        [string[]]$Expected,
        [string]$Stage
    )
    $ObservedSorted = @($Observed | ForEach-Object { [string]$_ } | Sort-Object -Unique)
    $ExpectedSorted = @($Expected | ForEach-Object { [string]$_ } | Sort-Object -Unique)
    if (($ObservedSorted -join "|") -ne ($ExpectedSorted -join "|")) {
        throw "$Stage set mismatch. Observed=[$($ObservedSorted -join ', ')] Expected=[$($ExpectedSorted -join ', ')]"
    }
}

function Assert-PassedManifest {
    param(
        [string]$Path,
        [string]$Protocol,
        [string]$Stage
    )
    $Payload = Read-JsonFile $Path
    if ([string]$Payload.quality_gate -ne "PASS") {
        throw "$Stage is not PASS: $Path"
    }
    if ([string]$Payload.protocol -ne $Protocol) {
        throw "$Stage uses stale/wrong protocol: $($Payload.protocol)"
    }
    return $Payload
}

function Assert-ArtifactNode {
    param(
        [object]$Node,
        [string]$ArtifactRoot,
        [string]$Stage,
        [string]$LogicalName
    )
    if ($null -eq $Node) {
        throw "$Stage has a null artifact entry: $LogicalName"
    }
    if ($Node -isnot [pscustomobject] -and
        $Node -isnot [System.Collections.IDictionary]) {
        throw "$Stage artifact node is not an object: $LogicalName"
    }
    $Properties = @($Node.PSObject.Properties)
    if ($Properties.Count -eq 0) {
        throw "$Stage artifact node is empty: $LogicalName"
    }
    $PathProperty = $Node.PSObject.Properties["path"]
    $HashProperty = $Node.PSObject.Properties["sha256"]
    if (($null -eq $PathProperty) -xor ($null -eq $HashProperty)) {
        throw "$Stage artifact leaf has only path or sha256: $LogicalName"
    }
    if ($null -ne $PathProperty) {
        if ($Properties.Count -ne 2) {
            throw "$Stage artifact leaf has unexpected fields: $LogicalName"
        }
        $DeclaredPath = [System.IO.Path]::GetFullPath([string]$PathProperty.Value)
        $ResolvedRoot = [System.IO.Path]::GetFullPath($ArtifactRoot).TrimEnd(
            [System.IO.Path]::DirectorySeparatorChar,
            [System.IO.Path]::AltDirectorySeparatorChar
        )
        $RootPrefix = $ResolvedRoot + [System.IO.Path]::DirectorySeparatorChar
        if (-not $DeclaredPath.StartsWith(
            $RootPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "$Stage artifact escapes its stage root: $LogicalName ($DeclaredPath)"
        }
        if (-not (Test-Path -LiteralPath $DeclaredPath -PathType Leaf)) {
            throw "$Stage artifact is missing: $LogicalName ($DeclaredPath)"
        }
        $ExpectedHash = ([string]$HashProperty.Value).ToLowerInvariant()
        if ($ExpectedHash -notmatch "^[0-9a-f]{64}$") {
            throw "$Stage artifact has an invalid SHA-256: $LogicalName"
        }
        $Observed = Get-Sha256 $DeclaredPath
        if ($Observed -ne $ExpectedHash) {
            throw "$Stage artifact hash mismatch: $LogicalName ($DeclaredPath)"
        }
        return 1
    }
    $LeafCount = 0
    foreach ($Property in $Properties) {
        $ChildCount = [int](Assert-ArtifactNode $Property.Value $ArtifactRoot $Stage "$LogicalName/$($Property.Name)")
        if ($ChildCount -lt 1) {
            throw "$Stage artifact subtree has no hash leaf: $LogicalName/$($Property.Name)"
        }
        $LeafCount += $ChildCount
    }
    if ($LeafCount -lt 1) {
        throw "$Stage artifact subtree has no hash leaf: $LogicalName"
    }
    return $LeafCount
}

function Assert-ArtifactHashes {
    param(
        [object]$Artifacts,
        [string]$ArtifactRoot,
        [string]$Stage
    )
    if ($null -eq $Artifacts) {
        throw "$Stage has no artifact hash map"
    }
    $Properties = @($Artifacts.PSObject.Properties)
    if ($Properties.Count -eq 0) {
        throw "$Stage has an empty artifact hash map"
    }
    $LeafTotal = 0
    foreach ($Property in $Properties) {
        $SubtreeLeafCount = [int](Assert-ArtifactNode $Property.Value $ArtifactRoot $Stage ([string]$Property.Name))
        if ($SubtreeLeafCount -lt 1) {
            throw "$Stage artifact subtree is empty: $($Property.Name)"
        }
        $LeafTotal += $SubtreeLeafCount
    }
    if ($LeafTotal -lt 1) {
        throw "$Stage has no artifact hash leaves"
    }
}

function Assert-ArtifactLeafExact {
    param(
        [object]$Leaf,
        [string]$ExpectedPath,
        [string]$Stage,
        [string]$LogicalName
    )
    if ($null -eq $Leaf -or
        $Leaf -isnot [pscustomobject] -or
        @($Leaf.PSObject.Properties).Count -ne 2 -or
        $null -eq $Leaf.PSObject.Properties["path"] -or
        $null -eq $Leaf.PSObject.Properties["sha256"]) {
        throw "$Stage has a malformed artifact leaf: $LogicalName"
    }
    $DeclaredPath = [System.IO.Path]::GetFullPath([string]$Leaf.path)
    $ResolvedExpected = [System.IO.Path]::GetFullPath($ExpectedPath)
    if (-not $DeclaredPath.Equals(
        $ResolvedExpected,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Stage artifact path mismatch: $LogicalName ($DeclaredPath != $ResolvedExpected)"
    }
    if (-not (Test-Path -LiteralPath $ResolvedExpected -PathType Leaf)) {
        throw "$Stage artifact is missing: $LogicalName ($ResolvedExpected)"
    }
    $ExpectedHash = ([string]$Leaf.sha256).ToLowerInvariant()
    if ($ExpectedHash -notmatch "^[0-9a-f]{64}$" -or
        (Get-Sha256 $ResolvedExpected) -ne $ExpectedHash) {
        throw "$Stage artifact hash mismatch: $LogicalName ($ResolvedExpected)"
    }
}

function Assert-SourceModel {
    param(
        [string]$Checkpoint,
        [string]$ManifestPath,
        [string]$Protocol,
        [string]$Scope,
        [string[]]$ActiveTokens,
        [string]$Stage
    )
    $Manifest = Assert-PassedManifest $ManifestPath $Protocol $Stage
    if ([string]$Manifest.expert_scope -ne $Scope) {
        throw "$Stage has wrong expert scope: $($Manifest.expert_scope)"
    }
    Assert-SameStringSet -Observed @($Manifest.active_expert_tokens) -Expected $ActiveTokens -Stage "$Stage active experts"
    if ((Get-Sha256 $Checkpoint) -ne [string]$Manifest.checkpoint_artifact_sha256) {
        throw "$Stage checkpoint hash does not match its manifest"
    }
    if ((Get-Sha256 $CanonicalCheckpoint) -ne [string]$Manifest.parent_checkpoint_sha256) {
        throw "$Stage was not derived from the pinned canonical checkpoint"
    }
    return $Manifest
}

function Assert-SourceRepresentation {
    param(
        [string]$Path,
        [string]$Protocol,
        [string]$Authorization,
        [string]$Checkpoint,
        [string]$Stage
    )
    $Report = Assert-PassedManifest $Path $Protocol $Stage
    if ([string]$Report.release_authorization -ne $Authorization) {
        throw "$Stage has no expected authorization"
    }
    if ([string]$Report.model_sha256 -ne (Get-Sha256 $Checkpoint)) {
        throw "$Stage belongs to another checkpoint"
    }
    return $Report
}

function Assert-V7FailureDiagnostic {
    $Manifest = Read-JsonFile $V7GenerationManifest
    if ([string]$Manifest.quality_gate -ne "FAIL" -or
        [string]$Manifest.protocol -ne $V7GenerationProtocol -or
        [int]$Manifest.raw_candidates_generated -ne 31500 -or
        [int]$Manifest.targets_with_signature_candidate -ne 15) {
        throw "V7 generation is not the preserved 31,500-row, 15/17 diagnostic result"
    }
    Assert-SameStringSet -Observed @($Manifest.targets_without_signature_candidate | ForEach-Object { ([string]$_).ToUpperInvariant() }) -Expected @("3WNE", "3ZGC") -Stage "V7 missing targets"
    if (@($Manifest.targets_formally_abstained).Count -ne 0) {
        throw "V7 diagnostic unexpectedly contains a formal abstention"
    }
    $FalseChecks = @(
        $Manifest.quality_checks.PSObject.Properties |
            Where-Object { -not [bool]$_.Value } |
            ForEach-Object { [string]$_.Name }
    )
    Assert-SameStringSet -Observed $FalseChecks -Expected @("every_target_has_at_least_one_novel_methylated_signature_candidate") -Stage "V7 failed checks"
    if ([string]$Manifest.model_sha256 -ne (Get-Sha256 $V7Checkpoint)) {
        throw "V7 diagnostic belongs to another checkpoint"
    }
    if ([string]$Manifest.source_v6_all_candidates_sha256 -ne $ExpectedV6AllSha -or
        [string]$Manifest.source_v6_generation_manifest_sha256 -ne $ExpectedV6ManifestSha) {
        throw "V7 diagnostic is not derived from the pinned V6 source pool"
    }
    Assert-ArtifactHashes $Manifest.candidate_artifacts $V7Generation "V7 diagnostic"
    Assert-SameStringSet -Observed @($Manifest.candidate_artifacts.PSObject.Properties | ForEach-Object { $_.Name }) -Expected @("all", "unique", "eligible", "target_manifest", "target_summary") -Stage "V7 diagnostic artifacts"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.all (Join-Path $V7Generation "all_candidates.csv") "V7 diagnostic" "all"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.unique (Join-Path $V7Generation "unique_candidates.csv") "V7 diagnostic" "unique"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.eligible (Join-Path $V7Generation "methylated_new_candidates.csv") "V7 diagnostic" "eligible"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.target_manifest (Join-Path $V7Generation "target_manifest.csv") "V7 diagnostic" "target_manifest"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.target_summary (Join-Path $V7Generation "generation_summary_by_target.csv") "V7 diagnostic" "target_summary"
    return $Manifest
}

function Assert-V8Baseline {
    $Manifest = Read-JsonFile $V8BaselineManifest
    if ([string]$Manifest.protocol -ne $V8BaselineProtocol -or
        [int]$Manifest.raw_candidates_generated -ne 31500 -or
        [string]$Manifest.model_sha256 -ne (Get-Sha256 $V8Checkpoint) -or
        [string]$Manifest.deterministic_runtime.cublas_workspace_config -ne ":4096:8" -or
        -not [bool]$Manifest.deterministic_runtime.deterministic_algorithms_enabled -or
        -not [bool]$Manifest.deterministic_runtime.cudnn_deterministic -or
        [bool]$Manifest.deterministic_runtime.cudnn_benchmark -or
        [string]$Manifest.reannotator_program_sha256 -ne (Get-Sha256 $Reannotator) -or
        [string]$Manifest.generator_program_sha256 -ne (Get-Sha256 $GeneratorProgram) -or
        [string]$Manifest.common_program_sha256 -ne (Get-Sha256 $CommonProgram) -or
        [string]$Manifest.model_utils_program_sha256 -ne (Get-Sha256 $ModelUtilsProgram) -or
        [string]$Manifest.nmethyl_config_program_sha256 -ne (Get-Sha256 $NmethylConfigProgram) -or
        [string]$Manifest.plan_sha256 -ne (Get-Sha256 $Plan) -or
        [string]$Manifest.native_jsonl_sha256 -ne (Get-Sha256 $NativeJsonl) -or
        [string]$Manifest.historical_design_csv_sha256 -ne (Get-Sha256 $HistoricalCsv) -or
        [string]$Manifest.prior_handoff_csv_sha256 -ne (Get-Sha256 $PriorHandoff) -or
        [int]$Manifest.scoring_batch_size -ne 8 -or
        [double]$Manifest.temperature -ne 0.5 -or
        [double]$Manifest.methyl_threshold -ne 0.6 -or
        [string]$Manifest.strict_threshold_operator -ne ">" -or
        [string]$Manifest.summary_score_label -ne "v8" -or
        [string]$Manifest.expert_scope -ne "residue-source-scoped-hybrid" -or
        [string]$Manifest.model_expert_qc_protocol -ne $V8ExpertProtocol -or
        [string]$Manifest.cyclic_representation_heldout_audit.sha256 -ne (Get-Sha256 $V8Representation)) {
        throw "V8 baseline protocol, row count, or checkpoint pin is invalid"
    }
    if (@($Manifest.targets_formally_abstained).Count -ne 0) {
        throw "V8 baseline contains a forbidden formal abstention"
    }
    $Missing = @(
        $Manifest.targets_without_signature_candidate |
            ForEach-Object { ([string]$_).ToUpperInvariant() } |
            Sort-Object -Unique
    )
    foreach ($Target in $Missing) {
        if ($Target -notin @("3WNE", "3ZGC")) {
            throw "V8 baseline has a non-recoverable missing target: $Target"
        }
    }
    if ([int]$Manifest.targets_with_signature_candidate -ne (17 - $Missing.Count)) {
        throw "V8 baseline coverage count is internally inconsistent"
    }
    $FalseChecks = @(
        $Manifest.quality_checks.PSObject.Properties |
            Where-Object { -not [bool]$_.Value } |
            ForEach-Object { [string]$_.Name }
    )
    if ($Missing.Count -eq 0) {
        if ([string]$Manifest.quality_gate -ne "PASS" -or $FalseChecks.Count -ne 0) {
            throw "A complete V8 baseline must be PASS with no failed checks"
        }
    } else {
        if ([string]$Manifest.quality_gate -ne "FAIL" -or
            -not [bool]$Manifest.directed_recovery_eligible) {
            throw "Incomplete V8 baseline is not explicitly eligible for directed recovery"
        }
        Assert-SameStringSet -Observed $FalseChecks -Expected @("every_target_has_at_least_one_novel_methylated_signature_candidate") -Stage "V8 baseline failed checks"
    }
    if ([string]$Manifest.source_v6_all_candidates_sha256 -ne $ExpectedV6AllSha -or
        [string]$Manifest.source_v6_generation_manifest_sha256 -ne $ExpectedV6ManifestSha) {
        throw "V8 baseline is not derived from the pinned V6 pool"
    }
    Assert-ArtifactHashes $Manifest.candidate_artifacts $BaselineOut "V8 baseline"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.all (Join-Path $BaselineOut "all_candidates.csv") "V8 baseline" "all"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.unique (Join-Path $BaselineOut "unique_candidates.csv") "V8 baseline" "unique"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.eligible (Join-Path $BaselineOut "methylated_new_candidates.csv") "V8 baseline" "eligible"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.target_manifest (Join-Path $BaselineOut "target_manifest.csv") "V8 baseline" "target_manifest"
    Assert-ArtifactLeafExact $Manifest.candidate_artifacts.target_summary (Join-Path $BaselineOut "generation_summary_by_target.csv") "V8 baseline" "target_summary"
    return $Manifest
}

function Get-SourceHashSnapshot {
    return [pscustomobject][ordered]@{
        canonical_checkpoint = Get-Sha256 $CanonicalCheckpoint
        v6_checkpoint = Get-Sha256 $V6Checkpoint
        v6_expert_manifest = Get-Sha256 $V6ExpertManifest
        v6_representation_audit = Get-Sha256 $V6Representation
        v6_generation_all = Get-Sha256 $V6AllCandidates
        v6_generation_manifest = Get-Sha256 $V6GenerationManifest
        v7_checkpoint = Get-Sha256 $V7Checkpoint
        v7_expert_manifest = Get-Sha256 $V7ExpertManifest
        v7_training_history = Get-Sha256 (Join-Path $V7Root "model\training_history.csv")
        v7_test_metrics_by_residue = Get-Sha256 (Join-Path $V7Root "model\test_metrics_by_residue.csv")
        v7_test_position_probabilities = Get-Sha256 (Join-Path $V7Root "model\test_position_probabilities.csv")
        v7_representation_audit = Get-Sha256 $V7Representation
        v7_generation_all = Get-Sha256 (Join-Path $V7Generation "all_candidates.csv")
        v7_generation_manifest = Get-Sha256 $V7GenerationManifest
        frozen_test_jsonl = Get-Sha256 $TestJsonl
        native_jsonl = Get-Sha256 $NativeJsonl
        best_designs_csv = Get-Sha256 $BestCsv
        historical_designs_csv = Get-Sha256 $HistoricalCsv
        prior_handoff_csv = Get-Sha256 $PriorHandoff
        target_plan = Get-Sha256 $Plan
        launcher_program = Get-Sha256 $LauncherProgram
        composer_program = Get-Sha256 $Composer
        representation_auditor_program = Get-Sha256 $RepresentationAuditor
        reannotator_program = Get-Sha256 $Reannotator
        directed_search_program = Get-Sha256 $DirectedSearch
        recovery_finalizer_program = Get-Sha256 $RecoveryFinalizer
        position_auditor_program = Get-Sha256 $PositionAuditor
        trainer_program = Get-Sha256 $TrainerProgram
        equivariance_auditor_program = Get-Sha256 $EquivarianceAuditor
        generator_program = Get-Sha256 $GeneratorProgram
        common_program = Get-Sha256 $CommonProgram
        model_utils_program = Get-Sha256 $ModelUtilsProgram
        nmethyl_config_program = Get-Sha256 $NmethylConfigProgram
    }
}

function Get-BaselineHashSnapshot {
    return [pscustomobject][ordered]@{
        all_candidates = Get-Sha256 (Join-Path $BaselineOut "all_candidates.csv")
        unique_candidates = Get-Sha256 (Join-Path $BaselineOut "unique_candidates.csv")
        eligible_candidates = Get-Sha256 (Join-Path $BaselineOut "methylated_new_candidates.csv")
        target_manifest = Get-Sha256 (Join-Path $BaselineOut "target_manifest.csv")
        target_summary = Get-Sha256 (Join-Path $BaselineOut "generation_summary_by_target.csv")
        generation_manifest = Get-Sha256 $V8BaselineManifest
    }
}

function Get-V8RecoveryInputHashSnapshot {
    $Payload = [ordered]@{}
    foreach ($Root in @($ModelOut, $RepresentationOut, $BaselineOut, $SearchOut)) {
        if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
            throw "V8 recovery input directory is missing: $Root"
        }
        foreach ($File in @(Get-ChildItem -LiteralPath $Root -Recurse -File | Sort-Object FullName)) {
            $Relative = $File.FullName.Substring($V8Root.Length).TrimStart(
                [System.IO.Path]::DirectorySeparatorChar,
                [System.IO.Path]::AltDirectorySeparatorChar
            ).Replace("\", "/")
            if ($Payload.Contains($Relative)) {
                throw "Duplicate V8 recovery input path: $Relative"
            }
            $Payload[$Relative] = Get-Sha256 $File.FullName
        }
    }
    if ($Payload.Count -eq 0) {
        throw "V8 recovery input snapshot is empty"
    }
    return [pscustomobject]$Payload
}

function Assert-HashSnapshotUnchanged {
    param(
        [object]$Before,
        [object]$After,
        [string]$Stage
    )
    foreach ($Property in @($Before.PSObject.Properties)) {
        $Name = [string]$Property.Name
        if ([string]$Property.Value -ne [string]$After.$Name) {
            throw "$Stage changed immutable input: $Name"
        }
    }
}

function Write-JsonNoBom {
    param(
        [string]$Path,
        [object]$Payload
    )
    $Parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $Parent -PathType Container)) {
        New-Item -ItemType Directory -Path $Parent | Out-Null
    }
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        $Path,
        (($Payload | ConvertTo-Json -Depth 16) + [Environment]::NewLine),
        $Utf8NoBom
    )
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
    & $PythonPath -c $Program $SourceDirectory $DestinationPath
    if ($LASTEXITCODE -ne 0) {
        throw "Portable V8 review ZIP packaging failed with exit code $LASTEXITCODE"
    }
}

if ($AuditBatchSize -le 0 -or $ScoringBatchSize -le 0 -or
    $SearchBatchSize -le 0 -or $BaseBatchSize -le 0) {
    throw "All batch sizes must be positive"
}
if ($SearchBatchSize -ne 64 -or $BaseBatchSize -ne 32) {
    throw "V8 search is frozen to SearchBatchSize=64 and BaseBatchSize=32"
}
if ($AuditBatchSize -ne 8 -or $ScoringBatchSize -ne 8) {
    throw "V8 audit/reannotation is frozen to AuditBatchSize=8 and ScoringBatchSize=8"
}

$RequiredInputs = @(
    $CanonicalCheckpoint,
    $V6Checkpoint,
    $V6ExpertManifest,
    $V6Representation,
    $V6AllCandidates,
    $V6GenerationManifest,
    $V7Checkpoint,
    $V7ExpertManifest,
    (Join-Path $V7Root "model\training_history.csv"),
    (Join-Path $V7Root "model\test_metrics_by_residue.csv"),
    (Join-Path $V7Root "model\test_position_probabilities.csv"),
    $V7Representation,
    $V7GenerationManifest,
    (Join-Path $V7Generation "all_candidates.csv"),
    (Join-Path $V7Generation "unique_candidates.csv"),
    (Join-Path $V7Generation "methylated_new_candidates.csv"),
    (Join-Path $V7Generation "target_manifest.csv"),
    (Join-Path $V7Generation "generation_summary_by_target.csv"),
    $TestJsonl,
    $NativeJsonl,
    $BestCsv,
    $HistoricalCsv,
    $PriorHandoff,
    $Plan,
    $LauncherProgram,
    $Composer,
    $RepresentationAuditor,
    $Reannotator,
    $DirectedSearch,
    $RecoveryFinalizer,
    $PositionAuditor,
    $TrainerProgram,
    $EquivarianceAuditor,
    $GeneratorProgram,
    $CommonProgram,
    $ModelUtilsProgram,
    $NmethylConfigProgram
)
foreach ($Required in $RequiredInputs) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required immutable input is missing: $Required"
    }
}
if ((Get-Sha256 $V6AllCandidates) -ne $ExpectedV6AllSha -or
    (Get-Sha256 $V6GenerationManifest) -ne $ExpectedV6ManifestSha) {
    throw "V6 source pool is not the uploaded audited 31,500-row result"
}

$V6Source = Assert-SourceModel $V6Checkpoint $V6ExpertManifest $V6ExpertProtocol "all" @("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y") "V6 source model"
$V7Source = Assert-SourceModel $V7Checkpoint $V7ExpertManifest $V7ExpertProtocol "serine-only" @("S") "V7 source model"
$V6SourceRepresentation = Assert-SourceRepresentation $V6Representation $V6RepresentationProtocol $V6RepresentationAuthorization $V6Checkpoint "V6 representation audit"
$V7SourceRepresentation = Assert-SourceRepresentation $V7Representation $V7RepresentationProtocol $V7RepresentationAuthorization $V7Checkpoint "V7 representation audit"
Assert-SameStringSet -Observed @($V7Source.changed_state_keys) -Expected @("experts.15.bias", "experts.15.weight") -Stage "V7 changed tensors"
if ($null -eq $V7Source.maximum_non_ser_probability_difference_from_parent -or
    [double]$V7Source.maximum_non_ser_probability_difference_from_parent -ne 0.0) {
    throw "V7 source model changed a non-Ser held-out probability"
}
$V7Diagnostic = Assert-V7FailureDiagnostic
$SourceHashesBefore = Get-SourceHashSnapshot

$ResolvedPython = Resolve-PythonExecutable
$Probe = 'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'
Invoke-PythonStage $ResolvedPython "Python/PyTorch preflight" @("-c", $Probe)
if (-not $AllowCpu) {
    Invoke-PythonStage $ResolvedPython "CUDA preflight" @("-c", "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)")
}
$DeviceArguments = if ($AllowCpu) { @("--device", "auto", "--allow-cpu") } else { @("--device", "cuda") }

Write-Host "============================================================"
Write-Host "SERINE QC SOURCE-SCOPED HYBRID RECOVERY V8"
Write-Host "Repository: $RepoRoot"
Write-Host "Python:     $ResolvedPython"
Write-Host "Model:      canonical shared + V6 non-Ser experts + V7 Ser expert"
Write-Host "Training:   NONE; passed canonical/V6/V7 artifacts are immutable inputs"
Write-Host "Baseline:   reannotate the hash-pinned 31,500 V6 natural rows"
Write-Host "Search:     deterministic fixed budget for actual missing 3WNE/3ZGC targets"
Write-Host "Release:    manual-review ZIP only; no structure handoff or permeability input"
Write-Host "============================================================"

Push-Location $RepoRoot
try {
    if (Test-Path -LiteralPath $V8ExpertManifest -PathType Leaf) {
        $V8Model = Assert-PassedManifest $V8ExpertManifest $V8ExpertProtocol "V8 source-scoped model"
        Write-Host "Model step: reused passed V8 source-scoped checkpoint"
    } else {
        Assert-EmptyOrAbsentDirectory $ModelOut "V8 model composition"
        $Arguments = @(
            $Composer,
            "--canonical-model", $CanonicalCheckpoint,
            "--v6-model", $V6Checkpoint,
            "--v6-manifest", $V6ExpertManifest,
            "--v7-model", $V7Checkpoint,
            "--v7-manifest", $V7ExpertManifest,
            "--test-jsonl", $TestJsonl,
            "--out-dir", $ModelOut,
            "--batch-size", $AuditBatchSize,
            "--temperature", 0.5,
            "--threshold", 0.6
        ) + $DeviceArguments
        Invoke-PythonStage $ResolvedPython "V8 source-scoped checkpoint composition and paired audit" $Arguments
        $V8Model = Assert-PassedManifest $V8ExpertManifest $V8ExpertProtocol "V8 source-scoped model"
    }
    if ((Get-Sha256 $V8Checkpoint) -ne [string]$V8Model.checkpoint_artifact_sha256) {
        throw "V8 checkpoint hash does not match its composition manifest"
    }
    if ([int]$V8Model.audit_batch_size -ne 8) {
        throw "V8 model paired audit used a non-frozen batch size"
    }
    Assert-ArtifactHashes $V8Model.artifacts $ModelOut "V8 source-scoped model"
    Assert-SameStringSet -Observed @($V8Model.artifacts.PSObject.Properties | ForEach-Object { $_.Name }) -Expected @("metric_comparison", "metrics_by_residue", "position_probabilities") -Stage "V8 model artifacts"
    Assert-ArtifactLeafExact $V8Model.artifacts.metric_comparison (Join-Path $ModelOut "v6_v7_v8_metric_comparison.csv") "V8 source-scoped model" "metric_comparison"
    Assert-ArtifactLeafExact $V8Model.artifacts.metrics_by_residue (Join-Path $ModelOut "test_metrics_by_residue.csv") "V8 source-scoped model" "metrics_by_residue"
    Assert-ArtifactLeafExact $V8Model.artifacts.position_probabilities (Join-Path $ModelOut "test_position_probabilities.csv") "V8 source-scoped model" "position_probabilities"
    if ([string]$V8Model.canonical_checkpoint_sha256 -ne (Get-Sha256 $CanonicalCheckpoint) -or
        [string]$V8Model.v6_checkpoint_sha256 -ne (Get-Sha256 $V6Checkpoint) -or
        [string]$V8Model.v6_manifest_sha256 -ne (Get-Sha256 $V6ExpertManifest) -or
        [string]$V8Model.v7_checkpoint_sha256 -ne (Get-Sha256 $V7Checkpoint) -or
        [string]$V8Model.v7_manifest_sha256 -ne (Get-Sha256 $V7ExpertManifest) -or
        [string]$V8Model.test_jsonl_sha256 -ne (Get-Sha256 $TestJsonl) -or
        [string]$V8Model.composer_program_sha256 -ne (Get-Sha256 $Composer) -or
        [string]$V8Model.trainer_program_sha256 -ne (Get-Sha256 $TrainerProgram) -or
        [string]$V8Model.common_program_sha256 -ne (Get-Sha256 $CommonProgram) -or
        [string]$V8Model.model_utils_program_sha256 -ne (Get-Sha256 $ModelUtilsProgram) -or
        [string]$V8Model.nmethyl_config_program_sha256 -ne (Get-Sha256 $NmethylConfigProgram) -or
        [string]$V8Model.deterministic_runtime.cublas_workspace_config -ne ":4096:8" -or
        -not [bool]$V8Model.deterministic_runtime.deterministic_algorithms_enabled -or
        -not [bool]$V8Model.deterministic_runtime.cudnn_deterministic -or
        [bool]$V8Model.deterministic_runtime.cudnn_benchmark) {
        throw "V8 composition manifest is stale against an immutable source"
    }
    if (-not [bool]$V8Model.quality_checks.every_non_ser_probability_is_inherited_from_v6 -or
        -not [bool]$V8Model.quality_checks.every_ser_probability_is_inherited_from_v7 -or
        -not [bool]$V8Model.quality_checks.recall_at_0_6_is_non_inferior_to_v6 -or
        -not [bool]$V8Model.quality_checks.f1_at_0_6_is_non_inferior_to_v6 -or
        -not [bool]$V8Model.quality_checks.non_ser_recall_at_0_6_is_non_inferior_to_v6 -or
        -not [bool]$V8Model.quality_checks.non_ser_f1_at_0_6_is_non_inferior_to_v6) {
        throw "V8 source/probability inheritance or non-inferiority gate is not proven"
    }

    if (Test-Path -LiteralPath $V8Representation -PathType Leaf) {
        $V8RepresentationReport = Assert-PassedManifest $V8Representation $V8RepresentationProtocol "V8 representation audit"
        Write-Host "Representation step: reused passed V8 audit"
    } else {
        Assert-EmptyOrAbsentDirectory $RepresentationOut "V8 representation audit"
        $Arguments = @(
            $RepresentationAuditor,
            "--model-path", $V8Checkpoint,
            "--model-manifest", $V8ExpertManifest,
            "--test-jsonl", $TestJsonl,
            "--native-jsonl", $NativeJsonl,
            "--best-csv", $BestCsv,
            "--plan", $Plan,
            "--out-dir", $RepresentationOut,
            "--batch-size", $AuditBatchSize,
            "--temperature", 0.5,
            "--threshold", 0.6
        ) + $DeviceArguments
        Invoke-PythonStage $ResolvedPython "V8 frozen cyclic-representation audit" $Arguments
        $V8RepresentationReport = Assert-PassedManifest $V8Representation $V8RepresentationProtocol "V8 representation audit"
    }
    if ([string]$V8RepresentationReport.release_authorization -ne $V8RepresentationAuthorization -or
        [string]$V8RepresentationReport.model_sha256 -ne (Get-Sha256 $V8Checkpoint) -or
        [string]$V8RepresentationReport.model_manifest_sha256 -ne (Get-Sha256 $V8ExpertManifest) -or
        [string]$V8RepresentationReport.test_jsonl_sha256 -ne (Get-Sha256 $TestJsonl) -or
        [string]$V8RepresentationReport.best_csv_sha256 -ne (Get-Sha256 $BestCsv) -or
        [string]$V8RepresentationReport.plan_sha256 -ne (Get-Sha256 $Plan) -or
        [int]$V8RepresentationReport.audit_batch_size -ne 8 -or
        [string]$V8RepresentationReport.native_jsonl_sha256 -ne (Get-Sha256 $NativeJsonl) -or
        [string]$V8RepresentationReport.representation_auditor_program_sha256 -ne (Get-Sha256 $RepresentationAuditor) -or
        [string]$V8RepresentationReport.equivariance_auditor_program_sha256 -ne (Get-Sha256 $EquivarianceAuditor) -or
        [string]$V8RepresentationReport.common_program_sha256 -ne (Get-Sha256 $CommonProgram) -or
        [string]$V8RepresentationReport.model_utils_program_sha256 -ne (Get-Sha256 $ModelUtilsProgram) -or
        [string]$V8RepresentationReport.nmethyl_config_program_sha256 -ne (Get-Sha256 $NmethylConfigProgram) -or
        [string]$V8RepresentationReport.deterministic_runtime.cublas_workspace_config -ne ":4096:8" -or
        -not [bool]$V8RepresentationReport.deterministic_runtime.deterministic_algorithms_enabled -or
        -not [bool]$V8RepresentationReport.deterministic_runtime.cudnn_deterministic -or
        [bool]$V8RepresentationReport.deterministic_runtime.cudnn_benchmark) {
        throw "V8 representation authorization or source hash gate failed"
    }
    Assert-ArtifactHashes $V8RepresentationReport.artifacts $RepresentationOut "V8 representation audit"
    Assert-SameStringSet -Observed @($V8RepresentationReport.artifacts.PSObject.Properties | ForEach-Object { $_.Name }) -Expected @("frozen_test_positions", "length_metrics", "native_probabilities", "native_summary") -Stage "V8 representation artifacts"
    Assert-ArtifactLeafExact $V8RepresentationReport.artifacts.frozen_test_positions (Join-Path $RepresentationOut "frozen_test_position_probabilities.csv") "V8 representation audit" "frozen_test_positions"
    Assert-ArtifactLeafExact $V8RepresentationReport.artifacts.length_metrics (Join-Path $RepresentationOut "frozen_test_metrics_by_length.csv") "V8 representation audit" "length_metrics"
    Assert-ArtifactLeafExact $V8RepresentationReport.artifacts.native_probabilities (Join-Path $RepresentationOut "native_target_representation_probabilities.csv") "V8 representation audit" "native_probabilities"
    Assert-ArtifactLeafExact $V8RepresentationReport.artifacts.native_summary (Join-Path $RepresentationOut "native_target_representation_summary.csv") "V8 representation audit" "native_summary"

    if (Test-Path -LiteralPath $V8BaselineManifest -PathType Leaf) {
        $V8Baseline = Assert-V8Baseline
        Write-Host "Baseline step: reused hash-valid V8 reannotation"
    } else {
        Assert-EmptyOrAbsentDirectory $BaselineOut "V8 31,500-row baseline reannotation"
        $ScientificReason = "Ser provenance repair and cyclic representation retraining are distinct interventions. V8 uses V7 only for the Ser expert and V6 for all non-Ser experts under a canonical shared network."
        $Arguments = @(
            $Reannotator,
            "--plan", $Plan,
            "--model-path", $V8Checkpoint,
            "--expert-manifest", $V8ExpertManifest,
            "--representation-audit-json", $V8Representation,
            "--source-run-dir", $V6Generation,
            "--out-dir", $BaselineOut,
            "--native-jsonl", $NativeJsonl,
            "--old-designs-csv", $HistoricalCsv,
            "--prior-designs-csv", $PriorHandoff,
            "--batch-size", $ScoringBatchSize,
            "--expected-source-all-sha256", $ExpectedV6AllSha,
            "--expected-source-manifest-sha256", $ExpectedV6ManifestSha,
            "--expected-expert-protocol", $V8ExpertProtocol,
            "--expected-expert-scope", "residue-source-scoped-hybrid",
            "--expected-active-expert-tokens-json", $NaturalExpertTokensJson,
            "--expected-representation-protocol", $V8RepresentationProtocol,
            "--expected-representation-authorization", $V8RepresentationAuthorization,
            "--output-protocol", $V8BaselineProtocol,
            "--recovery-mode", "SOURCE_SCOPED_HYBRID_V8_REANNOTATE_PRESERVED_V6_NATURAL_POOL_NO_RESAMPLING_NO_ABSTENTION",
            "--run-label", "SOURCE-SCOPED HYBRID V8",
            "--summary-score-label", "v8",
            "--scientific-reason", $ScientificReason,
            "--permit-missing-targets-for-recovery"
        ) + $DeviceArguments
        Invoke-PythonStage $ResolvedPython "V8 immutable-pool baseline reannotation" $Arguments
        $V8Baseline = Assert-V8Baseline
    }
    $BaselineHashesBeforeRecovery = Get-BaselineHashSnapshot

    $SearchReady = $false
    if (Test-Path -LiteralPath $V8SearchManifest -PathType Leaf) {
        $ExistingSearch = Read-JsonFile $V8SearchManifest
        if ([string]$ExistingSearch.quality_gate -eq "PASS") {
            $V8Search = Assert-PassedManifest $V8SearchManifest $V8SearchProtocol "V8 deterministic search"
            $SearchReady = $true
            Write-Host "Search step: reused passed deterministic result"
        }
    }
    if (-not $SearchReady) {
        $UnmanifestedSearchPartial = (
            (Test-Path -LiteralPath $SearchOut -PathType Container) -and
            @(Get-ChildItem -LiteralPath $SearchOut -ErrorAction SilentlyContinue).Count -gt 0 -and
            -not (Test-Path -LiteralPath $V8SearchManifest -PathType Leaf)
        )
        $ResumeCheckpoints = @(
            Get-ChildItem -LiteralPath (Join-Path $SearchOut "checkpoints") -File -Filter "3zgc_round_*.json.gz" -ErrorAction SilentlyContinue
        )
        if ($UnmanifestedSearchPartial -and $ResumeCheckpoints.Count -eq 0) {
            throw "Unmanifested directed-search partial output has no round checkpoint and was preserved: $SearchOut"
        }
        $Arguments = @(
            $DirectedSearch,
            "--model-path", $V8Checkpoint,
            "--model-manifest", $V8ExpertManifest,
            "--representation-audit", $V8Representation,
            "--baseline-run-dir", $BaselineOut,
            "--plan", $Plan,
            "--native-jsonl", $NativeJsonl,
            "--historical-designs-csv", $HistoricalCsv,
            "--prior-handoff-csv", $PriorHandoff,
            "--out-dir", $SearchOut,
            "--batch-size", $SearchBatchSize,
            "--base-batch-size", $BaseBatchSize,
            "--wne-radius", 2,
            "--zgc-rounds", 6,
            "--zgc-beam-width", 512,
            "--zgc-offspring-per-round", 4096,
            "--max-release-per-target", 200,
            "--resume"
        ) + $DeviceArguments
        Invoke-PythonStage $ResolvedPython "V8 deterministic missing-target search" $Arguments
        $V8Search = Assert-PassedManifest $V8SearchManifest $V8SearchProtocol "V8 deterministic search"
    }
    if ([string]$V8Search.model_sha256 -ne (Get-Sha256 $V8Checkpoint) -or
        [string]$V8Search.baseline_manifest_sha256 -ne (Get-Sha256 $V8BaselineManifest)) {
        throw "V8 directed search is stale against the model or immutable baseline"
    }
    $SearchConfig = $V8Search.config
    if ([double]$SearchConfig.temperature -ne 0.5 -or
        [double]$SearchConfig.threshold -ne 0.6 -or
        [string]$SearchConfig.strict_operator -ne ">" -or
        [string]$SearchConfig.alphabet -ne "ACDEFGHIKLMNPQRSTVWY" -or
        [int]$SearchConfig.'3wne_radius' -ne 2 -or
        [int]$SearchConfig.'3zgc_rounds' -ne 6 -or
        [int]$SearchConfig.'3zgc_beam_width' -ne 512 -or
        [int]$SearchConfig.'3zgc_offspring_per_round' -ne 4096 -or
        [int]$SearchConfig.methyl_batch_size -ne 64 -or
        [int]$SearchConfig.base_plausibility_batch_size -ne 32 -or
        [int]$SearchConfig.maximum_released_candidates_per_target -ne 200 -or
        [int]$SearchConfig.probability_persistence_decimal_places -ne 8 -or
        [string]$SearchConfig.cublas_workspace_config -ne ":4096:8" -or
        -not [bool]$SearchConfig.deterministic_algorithms_enabled -or
        -not [bool]$SearchConfig.cudnn_deterministic -or
        [bool]$SearchConfig.cudnn_benchmark -or
        -not [bool]$SearchConfig.full_budget_no_early_stop) {
        throw "V8 directed search does not use the frozen deterministic budget"
    }
    $SearchInputHashes = $SearchConfig.input_hashes
    if ([string]$SearchInputHashes.model -ne (Get-Sha256 $V8Checkpoint) -or
        [string]$SearchInputHashes.model_manifest -ne (Get-Sha256 $V8ExpertManifest) -or
        [string]$SearchInputHashes.representation_audit -ne (Get-Sha256 $V8Representation) -or
        [string]$SearchInputHashes.baseline_manifest -ne (Get-Sha256 $V8BaselineManifest) -or
        [string]$SearchInputHashes.baseline_all -ne (Get-Sha256 (Join-Path $BaselineOut "all_candidates.csv")) -or
        [string]$SearchInputHashes.baseline_unique -ne (Get-Sha256 (Join-Path $BaselineOut "unique_candidates.csv")) -or
        [string]$SearchInputHashes.baseline_eligible -ne (Get-Sha256 (Join-Path $BaselineOut "methylated_new_candidates.csv")) -or
        [string]$SearchInputHashes.plan -ne (Get-Sha256 $Plan) -or
        [string]$SearchInputHashes.native -ne (Get-Sha256 $NativeJsonl) -or
        [string]$SearchInputHashes.historical -ne (Get-Sha256 $HistoricalCsv) -or
        [string]$SearchInputHashes.prior -ne (Get-Sha256 $PriorHandoff) -or
        [string]$SearchInputHashes.search_program -ne (Get-Sha256 $DirectedSearch) -or
        [string]$SearchInputHashes.reannotator_program -ne (Get-Sha256 $Reannotator) -or
        [string]$SearchInputHashes.generator_program -ne (Get-Sha256 $GeneratorProgram) -or
        [string]$SearchInputHashes.common_program -ne (Get-Sha256 $CommonProgram) -or
        [string]$SearchInputHashes.model_utils_program -ne (Get-Sha256 $ModelUtilsProgram) -or
        [string]$SearchInputHashes.nmethyl_config_program -ne (Get-Sha256 $NmethylConfigProgram)) {
        throw "V8 directed search input hash map is incomplete or stale"
    }
    Assert-SameStringSet -Observed @($V8Search.missing_targets_before_search) -Expected @($V8Baseline.targets_without_signature_candidate) -Stage "V8 search input targets"
    Assert-SameStringSet -Observed @($V8Search.missing_targets_after_search) -Expected @() -Stage "V8 search unresolved targets"
    Assert-ArtifactHashes $V8Search.artifacts $SearchOut "V8 deterministic search"
    $ExpectedSearchArtifactKeys = @("controls", "plausibility", "directed_candidates", "trace")
    if (@($V8Search.missing_targets_before_search).Count -gt 0) {
        $ExpectedSearchArtifactKeys += "search_ledgers"
    }
    if (@($V8Search.missing_targets_before_search | Where-Object { ([string]$_).ToUpperInvariant() -eq "3ZGC" }).Count -gt 0) {
        $ExpectedSearchArtifactKeys += "checkpoints"
    }
    Assert-SameStringSet -Observed @($V8Search.artifacts.PSObject.Properties | ForEach-Object { $_.Name }) -Expected $ExpectedSearchArtifactKeys -Stage "V8 search artifacts"
    Assert-ArtifactLeafExact $V8Search.artifacts.controls (Join-Path $SearchOut "mandatory_length_6_7_controls.csv") "V8 deterministic search" "controls"
    Assert-ArtifactLeafExact $V8Search.artifacts.plausibility (Join-Path $SearchOut "qualified_candidate_plausibility_and_novelty.csv") "V8 deterministic search" "plausibility"
    Assert-ArtifactLeafExact $V8Search.artifacts.directed_candidates (Join-Path $SearchOut "directed_candidates.csv") "V8 deterministic search" "directed_candidates"
    Assert-ArtifactLeafExact $V8Search.artifacts.trace (Join-Path $SearchOut "search_trace_by_round.csv") "V8 deterministic search" "trace"
    if ($null -ne $V8Search.artifacts.PSObject.Properties["search_ledgers"]) {
        foreach ($Property in @($V8Search.artifacts.search_ledgers.PSObject.Properties)) {
            Assert-ArtifactLeafExact $Property.Value (Join-Path $SearchOut ([string]$Property.Name)) "V8 deterministic search" "search_ledgers/$($Property.Name)"
        }
    }
    if ($null -ne $V8Search.artifacts.PSObject.Properties["checkpoints"]) {
        foreach ($Property in @($V8Search.artifacts.checkpoints.PSObject.Properties)) {
            Assert-ArtifactLeafExact $Property.Value (Join-Path (Join-Path $SearchOut "checkpoints") ([string]$Property.Name)) "V8 deterministic search" "checkpoints/$($Property.Name)"
        }
    }
    $ControlRows = @(Import-Csv -LiteralPath (Join-Path $SearchOut "mandatory_length_6_7_controls.csv"))
    Assert-SameStringSet -Observed @($ControlRows | ForEach-Object { ([string]$_.target_name).ToUpperInvariant() }) -Expected @("3WNE", "3ZGC") -Stage "V8 mandatory short-peptide controls"
    if (@($V8Baseline.targets_without_signature_candidate).Count -eq 0 -and [int]$V8Search.released_candidates -ne 0) {
        throw "A complete V8 baseline must not produce directed release candidates"
    }

    $V8RecoveryInputsBefore = Get-V8RecoveryInputHashSnapshot
    $FinalReady = $false
    if ((Test-Path -LiteralPath $V8RecoveredManifest -PathType Leaf) -and
        (Test-Path -LiteralPath $V8RecoveredAudit -PathType Leaf)) {
        $V8Recovered = Assert-PassedManifest $V8RecoveredManifest $V8RecoveredProtocol "V8 recovered generation"
        $V8Audit = Assert-PassedManifest $V8RecoveredAudit $V8RecoveredAuditProtocol "V8 recovered three-pass audit"
        $FinalReady = $true
        Write-Host "Final audit step: reused passed recovery overlay"
    }
    if (-not $FinalReady) {
        Assert-EmptyOrAbsentDirectory $RecoveredOut "V8 recovered generation"
        Assert-EmptyOrAbsentDirectory $RecoveredAuditOut "V8 recovered three-pass audit"
        $Arguments = @(
            $RecoveryFinalizer,
            "--model-path", $V8Checkpoint,
            "--model-manifest", $V8ExpertManifest,
            "--representation-audit", $V8Representation,
            "--baseline-run-dir", $BaselineOut,
            "--search-dir", $SearchOut,
            "--plan", $Plan,
            "--native-jsonl", $NativeJsonl,
            "--historical-designs-csv", $HistoricalCsv,
            "--prior-handoff-csv", $PriorHandoff,
            "--out-dir", $RecoveredOut,
            "--audit-out-dir", $RecoveredAuditOut
        ) + $DeviceArguments
        Invoke-PythonStage $ResolvedPython "V8 immutable recovery overlay and independent three-pass audit" $Arguments
        $V8Recovered = Assert-PassedManifest $V8RecoveredManifest $V8RecoveredProtocol "V8 recovered generation"
        $V8Audit = Assert-PassedManifest $V8RecoveredAudit $V8RecoveredAuditProtocol "V8 recovered three-pass audit"
    }
    if ([int]$V8Recovered.targets_with_signature_candidate -ne 17 -or
        @($V8Recovered.targets_without_signature_candidate).Count -ne 0 -or
        @($V8Recovered.targets_formally_abstained).Count -ne 0) {
        throw "V8 final recovery is not 17/17 with zero formal abstentions"
    }
    if ([string]$V8Recovered.finalizer_program_sha256 -ne (Get-Sha256 $RecoveryFinalizer) -or
        [string]$V8Recovered.position_auditor_program_sha256 -ne (Get-Sha256 $PositionAuditor) -or
        [string]$V8Audit.finalizer_program_sha256 -ne (Get-Sha256 $RecoveryFinalizer) -or
        [string]$V8Audit.position_auditor_program_sha256 -ne (Get-Sha256 $PositionAuditor)) {
        throw "V8 recovered result was produced by stale final-audit code and was preserved"
    }
    if ([string]$V8Recovered.model_sha256 -ne (Get-Sha256 $V8Checkpoint) -or
        [string]$V8Recovered.search_manifest_sha256 -ne (Get-Sha256 $V8SearchManifest)) {
        throw "V8 recovered generation is stale against model or search"
    }
    if ([string]$V8Recovered.baseline_artifact_sha256.all -ne (Get-Sha256 (Join-Path $BaselineOut "all_candidates.csv")) -or
        [string]$V8Recovered.baseline_artifact_sha256.unique -ne (Get-Sha256 (Join-Path $BaselineOut "unique_candidates.csv")) -or
        [string]$V8Recovered.baseline_artifact_sha256.eligible -ne (Get-Sha256 (Join-Path $BaselineOut "methylated_new_candidates.csv")) -or
        [string]$V8Recovered.baseline_artifact_sha256.target_manifest -ne (Get-Sha256 (Join-Path $BaselineOut "target_manifest.csv")) -or
        [string]$V8Recovered.baseline_artifact_sha256.summary -ne (Get-Sha256 (Join-Path $BaselineOut "generation_summary_by_target.csv")) -or
        [string]$V8Recovered.baseline_artifact_sha256.manifest -ne (Get-Sha256 $V8BaselineManifest)) {
        throw "V8 recovered generation does not pin every immutable baseline artifact"
    }
    Assert-ArtifactHashes $V8Recovered.artifacts $RecoveredOut "V8 recovered generation"
    Assert-SameStringSet -Observed @($V8Recovered.artifacts.PSObject.Properties | ForEach-Object { $_.Name }) -Expected @("all_candidates", "unique_candidates", "final_candidates", "target_summary", "target_manifest") -Stage "V8 recovered artifacts"
    Assert-ArtifactLeafExact $V8Recovered.artifacts.all_candidates (Join-Path $RecoveredOut "all_candidates.csv") "V8 recovered generation" "all_candidates"
    Assert-ArtifactLeafExact $V8Recovered.artifacts.unique_candidates (Join-Path $RecoveredOut "unique_candidates.csv") "V8 recovered generation" "unique_candidates"
    Assert-ArtifactLeafExact $V8Recovered.artifacts.final_candidates (Join-Path $RecoveredOut "methylated_new_candidates.csv") "V8 recovered generation" "final_candidates"
    Assert-ArtifactLeafExact $V8Recovered.artifacts.target_summary (Join-Path $RecoveredOut "generation_summary_by_target.csv") "V8 recovered generation" "target_summary"
    Assert-ArtifactLeafExact $V8Recovered.artifacts.target_manifest (Join-Path $RecoveredOut "target_manifest.csv") "V8 recovered generation" "target_manifest"
    if ([string]$V8Audit.artifacts.final_manifest.sha256 -ne (Get-Sha256 $V8RecoveredManifest) -or
        [string]$V8Audit.artifacts.search_manifest.sha256 -ne (Get-Sha256 $V8SearchManifest)) {
        throw "V8 recovered audit artifact hashes are stale"
    }
    Assert-SameStringSet -Observed @($V8Audit.artifacts.PSObject.Properties | ForEach-Object { $_.Name }) -Expected @("final_manifest", "final_candidates", "final_all_candidates", "final_unique_candidates", "final_target_manifest", "final_target_summary", "search_manifest", "position_concentration", "av_family_physical_support") -Stage "V8 recovered audit artifacts"
    Assert-ArtifactLeafExact $V8Audit.artifacts.final_manifest $V8RecoveredManifest "V8 recovered audit" "final_manifest"
    Assert-ArtifactLeafExact $V8Audit.artifacts.final_candidates (Join-Path $RecoveredOut "methylated_new_candidates.csv") "V8 recovered audit" "final_candidates"
    Assert-ArtifactLeafExact $V8Audit.artifacts.final_all_candidates (Join-Path $RecoveredOut "all_candidates.csv") "V8 recovered audit" "final_all_candidates"
    Assert-ArtifactLeafExact $V8Audit.artifacts.final_unique_candidates (Join-Path $RecoveredOut "unique_candidates.csv") "V8 recovered audit" "final_unique_candidates"
    Assert-ArtifactLeafExact $V8Audit.artifacts.final_target_manifest (Join-Path $RecoveredOut "target_manifest.csv") "V8 recovered audit" "final_target_manifest"
    Assert-ArtifactLeafExact $V8Audit.artifacts.final_target_summary (Join-Path $RecoveredOut "generation_summary_by_target.csv") "V8 recovered audit" "final_target_summary"
    Assert-ArtifactLeafExact $V8Audit.artifacts.search_manifest $V8SearchManifest "V8 recovered audit" "search_manifest"
    Assert-ArtifactLeafExact $V8Audit.artifacts.position_concentration (Join-Path $RecoveredAuditOut "three_pass_concentration_by_target.csv") "V8 recovered audit" "position_concentration"
    Assert-ArtifactLeafExact $V8Audit.artifacts.av_family_physical_support (Join-Path $RecoveredAuditOut "av_family_physical_position_support.json") "V8 recovered audit" "av_family_physical_support"
    if ([string]$V8Audit.pass_1_integrity_and_rescore.quality_gate -ne "PASS" -or
        [string]$V8Audit.pass_2_position_and_representation.quality_gate -ne "PASS" -or
        [string]$V8Audit.pass_3_novelty_coverage_workflow.quality_gate -ne "PASS") {
        throw "V8 recovered audit does not pass all three independent passes"
    }

    $BaselineHashesAfterRecovery = Get-BaselineHashSnapshot
    Assert-HashSnapshotUnchanged $BaselineHashesBeforeRecovery $BaselineHashesAfterRecovery "V8 recovery"
    $V8RecoveryInputsAfter = Get-V8RecoveryInputHashSnapshot
    Assert-HashSnapshotUnchanged $V8RecoveryInputsBefore $V8RecoveryInputsAfter "V8 final audit"
    $SourceHashesAfter = Get-SourceHashSnapshot
    Assert-HashSnapshotUnchanged $SourceHashesBefore $SourceHashesAfter "V8 workflow"

    $ForbiddenOutputs = @(
        Get-ChildItem -LiteralPath $V8Root -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match "(?i)handoff|permeability" }
    )
    if ($ForbiddenOutputs.Count -ne 0) {
        throw "V8 created a forbidden handoff/permeability artifact: $($ForbiddenOutputs.FullName -join ', ')"
    }

    $Staging = Join-Path ([System.IO.Path]::GetTempPath()) ("proteinmpnn_v8_review_" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $Staging | Out-Null
    try {
        $BundleFileMap = [ordered]@{
            "source/v6_expert_manifest.json" = $V6ExpertManifest
            "source/v6_representation_audit.json" = $V6Representation
            "source/v6_generation_manifest.json" = $V6GenerationManifest
            "source/v7_expert_manifest.json" = $V7ExpertManifest
            "source/v7_training_history.csv" = (Join-Path $V7Root "model\training_history.csv")
            "source/v7_test_metrics_by_residue.csv" = (Join-Path $V7Root "model\test_metrics_by_residue.csv")
            "source/v7_test_position_probabilities.csv" = (Join-Path $V7Root "model\test_position_probabilities.csv")
            "source/v7_representation_audit.json" = $V7Representation
            "source/v7_15_of_17_generation_manifest.json" = $V7GenerationManifest
            "source/v7_15_of_17_target_manifest.csv" = (Join-Path $V7Generation "target_manifest.csv")
            "source/v7_15_of_17_target_summary.csv" = (Join-Path $V7Generation "generation_summary_by_target.csv")
            "source/v7_15_of_17_all_candidates.csv" = (Join-Path $V7Generation "all_candidates.csv")
            "source/v7_15_of_17_unique_candidates.csv" = (Join-Path $V7Generation "unique_candidates.csv")
            "source/v7_15_of_17_methylated_candidates.csv" = (Join-Path $V7Generation "methylated_new_candidates.csv")
            "inputs/frozen_test_serine_provenance_corrected.jsonl" = $TestJsonl
            "inputs/native_17_complexes.jsonl" = $NativeJsonl
            "inputs/target_plan.json" = $Plan
            "inputs/best_designs.csv" = $BestCsv
            "inputs/historical_all_designs.csv" = $HistoricalCsv
            "inputs/prior_handoff.csv" = $PriorHandoff
            "programs/run_serine_qc_source_scoped_hybrid_v8.ps1" = $LauncherProgram
            "programs/02_retrain_canonical_expert_heads.py" = $TrainerProgram
            "programs/07_audit_cyclic_representation_equivariance.py" = $EquivarianceAuditor
            "programs/10_reannotate_v6_pool_serine_only_v7.py" = $Reannotator
            "programs/11_triple_audit_serine_only_v7.py" = $PositionAuditor
            "programs/12_compose_source_scoped_hybrid_v8.py" = $Composer
            "programs/13_audit_source_scoped_hybrid_v8.py" = $RepresentationAuditor
            "programs/14_directed_recovery_search_v8.py" = $DirectedSearch
            "programs/15_finalize_and_audit_recovery_v8.py" = $RecoveryFinalizer
            "programs/01_generate_t05_multiseed.py" = $GeneratorProgram
            "programs/clean_v28_common.py" = $CommonProgram
            "programs/model_utils.py" = $ModelUtilsProgram
            "programs/nmethyl_config.py" = $NmethylConfigProgram
            "model/expert_source_composition_manifest.json" = $V8ExpertManifest
            "model/frankenstein_v28_source_scoped_hybrid_v8.pt" = $V8Checkpoint
            "model/v6_v7_v8_metric_comparison.csv" = (Join-Path $ModelOut "v6_v7_v8_metric_comparison.csv")
            "model/test_metrics_by_residue.csv" = (Join-Path $ModelOut "test_metrics_by_residue.csv")
            "model/test_position_probabilities.csv" = (Join-Path $ModelOut "test_position_probabilities.csv")
            "representation/cyclic_representation_audit.json" = $V8Representation
            "representation/frozen_test_position_probabilities.csv" = (Join-Path $RepresentationOut "frozen_test_position_probabilities.csv")
            "representation/frozen_test_metrics_by_length.csv" = (Join-Path $RepresentationOut "frozen_test_metrics_by_length.csv")
            "representation/native_target_representation_summary.csv" = (Join-Path $RepresentationOut "native_target_representation_summary.csv")
            "representation/native_target_representation_probabilities.csv" = (Join-Path $RepresentationOut "native_target_representation_probabilities.csv")
            "baseline/generation_manifest.json" = $V8BaselineManifest
            "baseline/target_manifest.csv" = (Join-Path $BaselineOut "target_manifest.csv")
            "baseline/generation_summary_by_target.csv" = (Join-Path $BaselineOut "generation_summary_by_target.csv")
            "baseline/all_candidates.csv" = (Join-Path $BaselineOut "all_candidates.csv")
            "baseline/unique_candidates.csv" = (Join-Path $BaselineOut "unique_candidates.csv")
            "baseline/methylated_new_candidates.csv" = (Join-Path $BaselineOut "methylated_new_candidates.csv")
            "search/directed_search_manifest.json" = $V8SearchManifest
            "search/mandatory_length_6_7_controls.csv" = (Join-Path $SearchOut "mandatory_length_6_7_controls.csv")
            "search/search_trace_by_round.csv" = (Join-Path $SearchOut "search_trace_by_round.csv")
            "search/qualified_candidate_plausibility_and_novelty.csv" = (Join-Path $SearchOut "qualified_candidate_plausibility_and_novelty.csv")
            "search/directed_candidates.csv" = (Join-Path $SearchOut "directed_candidates.csv")
            "final/generation_manifest.json" = $V8RecoveredManifest
            "final/all_candidates.csv" = (Join-Path $RecoveredOut "all_candidates.csv")
            "final/unique_candidates.csv" = (Join-Path $RecoveredOut "unique_candidates.csv")
            "final/methylated_new_candidates.csv" = (Join-Path $RecoveredOut "methylated_new_candidates.csv")
            "final/target_manifest.csv" = (Join-Path $RecoveredOut "target_manifest.csv")
            "final/generation_summary_by_target.csv" = (Join-Path $RecoveredOut "generation_summary_by_target.csv")
            "final/three_pass_generation_audit.json" = $V8RecoveredAudit
            "final/three_pass_concentration_by_target.csv" = (Join-Path $RecoveredAuditOut "three_pass_concentration_by_target.csv")
            "final/av_family_physical_position_support.json" = (Join-Path $RecoveredAuditOut "av_family_physical_position_support.json")
        }
        if (Test-Path -LiteralPath $V6FormalAbstention -PathType Leaf) {
            $BundleFileMap["source/v6_formal_target_abstention_audit.json"] = $V6FormalAbstention
        }
        foreach ($Ledger in @(Get-ChildItem -LiteralPath $SearchOut -File -Filter "*.csv.gz" -ErrorAction SilentlyContinue)) {
            $BundleFileMap["search/ledger/$($Ledger.Name)"] = $Ledger.FullName
        }
        $CheckpointDirectory = Join-Path $SearchOut "checkpoints"
        if (Test-Path -LiteralPath $CheckpointDirectory -PathType Container) {
            foreach ($CheckpointFile in @(Get-ChildItem -LiteralPath $CheckpointDirectory -File -Filter "*.json.gz" -ErrorAction SilentlyContinue)) {
                $BundleFileMap["search/checkpoints/$($CheckpointFile.Name)"] = $CheckpointFile.FullName
            }
        }

        $Contents = @()
        foreach ($Entry in $BundleFileMap.GetEnumerator()) {
            $ArchiveName = [string]$Entry.Key
            $SourcePath = [string]$Entry.Value
            if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
                throw "Review bundle source is missing: $SourcePath"
            }
            $Destination = Join-Path $Staging $ArchiveName
            $DestinationParent = Split-Path -Parent $Destination
            if (-not (Test-Path -LiteralPath $DestinationParent -PathType Container)) {
                New-Item -ItemType Directory -Path $DestinationParent | Out-Null
            }
            Copy-Item -LiteralPath $SourcePath -Destination $Destination
            $Contents += [ordered]@{
                archive_name = $ArchiveName.Replace("\", "/")
                source_path = (Resolve-Path -LiteralPath $SourcePath).Path
                sha256 = Get-Sha256 $SourcePath
                bytes = (Get-Item -LiteralPath $SourcePath).Length
            }
        }

        $ImmutableAuditPath = Join-Path $Staging "checksums\immutable_input_hashes.json"
        $ImmutableAudit = [ordered]@{
            quality_gate = "PASS"
            protocol = "source_scoped_hybrid_v8_immutable_input_hash_audit_v1"
            source_before = $SourceHashesBefore
            source_after = $SourceHashesAfter
            v8_baseline_before_search = $BaselineHashesBeforeRecovery
            v8_baseline_after_final_audit = $BaselineHashesAfterRecovery
            v8_recovery_inputs_before_final_audit = $V8RecoveryInputsBefore
            v8_recovery_inputs_after_final_audit = $V8RecoveryInputsAfter
        }
        Write-JsonNoBom $ImmutableAuditPath $ImmutableAudit
        $Contents += [ordered]@{
            archive_name = "checksums/immutable_input_hashes.json"
            source_path = "generated_during_review_packaging"
            sha256 = Get-Sha256 $ImmutableAuditPath
            bytes = (Get-Item -LiteralPath $ImmutableAuditPath).Length
        }

        $ReviewManifest = [ordered]@{
            quality_gate = "PASS"
            protocol = "serine_qc_source_scoped_hybrid_v8_manual_review_bundle_v1"
            source_v7_failure_diagnostic_coverage = "15/17"
            v8_baseline_coverage = "$($V8Baseline.targets_with_signature_candidate)/17"
            v8_baseline_missing_targets = @($V8Baseline.targets_without_signature_candidate)
            directed_search_missing_after = @($V8Search.missing_targets_after_search)
            final_coverage = "$($V8Recovered.targets_with_signature_candidate)/17"
            targets_formally_abstained = @()
            frozen_test_reuse_limitation = [string]$V8Model.test_reuse_limitation
            v6_v7_v8_recall_at_0_6 = [ordered]@{
                v6 = [double]$V8Model.v6_test.overall_at_threshold.recall
                v7 = [double]$V8Model.v7_test.overall_at_threshold.recall
                v8 = [double]$V8Model.v8_test.overall_at_threshold.recall
            }
            release_status = "HOLD_FOR_MANUAL_SCIENTIFIC_REVIEW_NO_STRUCTURE_HANDOFF"
            structure_requirement = "GLOBAL_COMPLEX_CA_RMSD_LT_3A_AND_COMPLETE_CYCLIC_PEPTIDE_CA_RMSD_LT_3A"
            permeability_status = "DEFERRED_UNTIL_RETURNED_STRUCTURES_PASS_BOTH_RMSD_GATES"
            content_file_count = $Contents.Count
            contents = $Contents
        }
        Write-JsonNoBom (Join-Path $Staging "review_bundle_manifest.json") $ReviewManifest
        Compress-PortableArchive $ResolvedPython $Staging $ReviewBundle
    } finally {
        if (Test-Path -LiteralPath $Staging -PathType Container) {
            Remove-Item -LiteralPath $Staging -Recurse -ErrorAction SilentlyContinue
        }
    }

    if (-not (Test-Path -LiteralPath $ReviewBundle -PathType Leaf)) {
        throw "V8 manual-review ZIP was not created"
    }
    $SourceHashesFinal = Get-SourceHashSnapshot
    $BaselineHashesFinal = Get-BaselineHashSnapshot
    Assert-HashSnapshotUnchanged $SourceHashesBefore $SourceHashesFinal "V8 review packaging"
    Assert-HashSnapshotUnchanged $BaselineHashesBeforeRecovery $BaselineHashesFinal "V8 review packaging"
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "V8 ALL AUTOMATED GATES PASSED; MANUAL SCIENTIFIC REVIEW IS NEXT" -ForegroundColor Green
Write-Host "Source V7 diagnostic: 15/17 (preserved in review ZIP)"
Write-Host "V8 baseline coverage: $($V8Baseline.targets_with_signature_candidate)/17"
Write-Host "Final coverage:       $($V8Recovered.targets_with_signature_candidate)/17; no formal abstention"
Write-Host "Final candidates:     $(Join-Path $RecoveredOut 'methylated_new_candidates.csv')"
Write-Host "Three-pass audit:     $V8RecoveredAudit"
Write-Host "Manual-review bundle: $ReviewBundle"
Write-Host "Shang-ge handoff:     NOT CREATED" -ForegroundColor Yellow
Write-Host "Permeability:          DEFERRED until returned structures pass both RMSD gates" -ForegroundColor Yellow
