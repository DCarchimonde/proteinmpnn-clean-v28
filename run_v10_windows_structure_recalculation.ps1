[CmdletBinding()]
param(
    [string]$RepoRoot = "E:\ProteinMPNN_work\proteinmpnn-clean-v28",
    [string]$V10Manifest = "",
    [string]$AutoDlMonomerManifest = "",
    [string]$MonomerPdbDir = "",
    [string]$MonomerPermeabilityRoot = "",
    [string]$RunDir = "",
    [string]$WindowsCondaEnv = "wain",
    [string]$TmCondaEnv = "tmdiv",
    [string]$WslDistribution = "Ubuntu",
    [string]$WslCondaRoot = "/home/aaron/miniconda3",
    [string]$PyRosettaEnv = "pyrosetta_eval"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
if (-not (Test-Path $RepoRoot -PathType Container)) {
    throw "Windows repository root does not exist: $RepoRoot"
}
if ([string]::IsNullOrWhiteSpace($V10Manifest)) {
    $V10Manifest = Join-Path $RepoRoot (
        "paper_clean_v28_outputs\rmsd_aware_v10_1700_monomer\" +
        "monomer_final\monomer_v10_design_manifest_151.csv"
    )
}
if ([string]::IsNullOrWhiteSpace($MonomerPdbDir)) {
    $MonomerPdbDir = Join-Path $RepoRoot (
        "raw_external\pdb_permeability_v20260624\pdb_monomer\" +
        "pdb_monomer_hf4"
    )
}
if ([string]::IsNullOrWhiteSpace($MonomerPermeabilityRoot)) {
    $MonomerPermeabilityRoot = Join-Path $RepoRoot (
        "raw_external\pdb_permeability_v20260624"
    )
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunDir = Join-Path $RepoRoot (
        "paper_clean_v28_outputs\rmsd_aware_v10_1700_monomer\" +
        "windows_structure_recalculation"
    )
}

$V10Manifest = [System.IO.Path]::GetFullPath($V10Manifest)
if ([string]::IsNullOrWhiteSpace($AutoDlMonomerManifest)) {
    $AutoDlMonomerManifest = Join-Path (
        Split-Path -Parent $V10Manifest
    ) "monomer_v10_manifest.json"
}
$AutoDlMonomerManifest = [System.IO.Path]::GetFullPath($AutoDlMonomerManifest)
$MonomerPdbDir = [System.IO.Path]::GetFullPath($MonomerPdbDir)
$MonomerPermeabilityRoot = [System.IO.Path]::GetFullPath($MonomerPermeabilityRoot)
$RunDir = [System.IO.Path]::GetFullPath($RunDir)
$AuditScript = Join-Path $RepoRoot (
    "paper_clean_v28\structure_metrics\22_audit_monomer_v10_pdb_reuse.py"
)
$Controller = Join-Path $RepoRoot (
    "paper_clean_v28\structure_metrics\run_temperature05_best17_all.ps1"
)
$AuditOutDir = Join-Path $RunDir "pdb_reuse_audit"
$AuditJson = Join-Path $AuditOutDir "monomer_v10_pdb_reuse_audit.json"

Write-Host "===== V10 WINDOWS-LOCAL MONOMER STRUCTURE RECALCULATION =====" -ForegroundColor Cyan
Write-Host "Windows-local repository: $RepoRoot"
Write-Host "Windows-local old PDB directory: $MonomerPdbDir"
Write-Host "AutoDL does not contain these old PDB files."
Write-Host (
    "Required AutoDL artifacts: download monomer_v10_design_manifest_151.csv " +
    "and monomer_v10_manifest.json together into this repository before running."
)
Write-Host "Downloaded AutoDL design CSV: $V10Manifest"
Write-Host "Downloaded AutoDL monomer manifest: $AutoDlMonomerManifest"

foreach ($RequiredFile in @(
    $V10Manifest,
    $AutoDlMonomerManifest,
    $AuditScript,
    $Controller
)) {
    if (-not (Test-Path $RequiredFile -PathType Leaf)) {
        throw (
            "Missing required file: $RequiredFile. If this is the V10 manifest, " +
            "download the completed AutoDL output into the Windows repository first."
        )
    }
}
foreach ($RequiredDirectory in @($MonomerPdbDir, $MonomerPermeabilityRoot)) {
    if (-not (Test-Path $RequiredDirectory -PathType Container)) {
        throw "Missing Windows-local directory: $RequiredDirectory"
    }
}

$ExpectedAutoDlProtocol = "corrected_monomer_cyclic_stability_and_base_freeze_audit_v10"
$ExpectedAutoDlManifestName = "monomer_v10_manifest.json"
if ((Split-Path -Leaf $AutoDlMonomerManifest) -cne $ExpectedAutoDlManifestName) {
    throw "AutoDL monomer manifest must be named $ExpectedAutoDlManifestName"
}
$AutoDlFilesAreColocated = [System.StringComparer]::OrdinalIgnoreCase.Equals(
    (Split-Path -Parent $V10Manifest),
    (Split-Path -Parent $AutoDlMonomerManifest)
)
if (-not $AutoDlFilesAreColocated) {
    throw (
        "The downloaded V10 design CSV and monomer_v10_manifest.json must remain " +
        "in the same monomer_final directory."
    )
}

$AutoDlManifestSha256 = (
    Get-FileHash -LiteralPath $AutoDlMonomerManifest -Algorithm SHA256
).Hash.ToLowerInvariant()
$WindowsDesignSha256 = (
    Get-FileHash -LiteralPath $V10Manifest -Algorithm SHA256
).Hash.ToLowerInvariant()
$AutoDlManifestPayload = Get-Content -LiteralPath $AutoDlMonomerManifest -Raw |
    ConvertFrom-Json
if (-not ($AutoDlManifestPayload.PSObject.Properties.Name -contains "protocol")) {
    throw "AutoDL monomer manifest is missing protocol"
}
if ([string]$AutoDlManifestPayload.protocol -cne $ExpectedAutoDlProtocol) {
    throw (
        "AutoDL monomer manifest protocol mismatch: expected " +
        "$ExpectedAutoDlProtocol, observed $($AutoDlManifestPayload.protocol)"
    )
}
if (-not ($AutoDlManifestPayload.PSObject.Properties.Name -contains "quality_gate") -or
    [string]$AutoDlManifestPayload.quality_gate -cne "PASS") {
    throw "AutoDL monomer manifest quality_gate is not PASS"
}
if (-not ($AutoDlManifestPayload.PSObject.Properties.Name -contains "quality_checks")) {
    throw "AutoDL monomer manifest is missing quality_checks"
}
$AutoDlQualityChecks = @(
    $AutoDlManifestPayload.quality_checks.PSObject.Properties
)
$FailedAutoDlChecks = @(
    $AutoDlQualityChecks |
        Where-Object { $_.Value -ne $true } |
        ForEach-Object { $_.Name }
)
if ($AutoDlQualityChecks.Count -eq 0 -or $FailedAutoDlChecks.Count -ne 0) {
    throw (
        "AutoDL monomer manifest quality_checks are empty or not all PASS: " +
        ($FailedAutoDlChecks -join ", ")
    )
}
if (-not ($AutoDlManifestPayload.PSObject.Properties.Name -contains "artifacts") -or
    -not ($AutoDlManifestPayload.artifacts.PSObject.Properties.Name -contains "design_manifest")) {
    throw "AutoDL monomer manifest is missing artifacts.design_manifest"
}
$RecordedDesignArtifact = $AutoDlManifestPayload.artifacts.design_manifest
if (-not ($RecordedDesignArtifact.PSObject.Properties.Name -contains "sha256")) {
    throw "AutoDL monomer manifest design artifact is missing sha256"
}
$RecordedDesignSha256 = ([string]$RecordedDesignArtifact.sha256).ToLowerInvariant()
if ($RecordedDesignSha256 -notmatch "^[0-9a-f]{64}$") {
    throw "AutoDL monomer manifest design artifact sha256 is invalid"
}
if ($WindowsDesignSha256 -cne $RecordedDesignSha256) {
    throw (
        "Downloaded V10 design CSV bytes do not match the AutoDL monomer " +
        "manifest: expected=$RecordedDesignSha256 observed=$WindowsDesignSha256"
    )
}
Write-Host "AutoDL monomer manifest contract: PASS"
Write-Host "AutoDL monomer manifest SHA-256: $AutoDlManifestSha256"
Write-Host "V10 design CSV SHA-256: $WindowsDesignSha256"

New-Item -ItemType Directory -Force -Path $AuditOutDir | Out-Null
Write-Host ""
Write-Host "===== 1/2 READ-ONLY PDB REUSE AUDIT =====" -ForegroundColor Cyan
$global:LASTEXITCODE = 0
conda run --no-capture-output -n $WindowsCondaEnv python $AuditScript `
    --design-manifest $V10Manifest `
    --autodl-monomer-manifest $AutoDlMonomerManifest `
    --autodl-monomer-manifest-sha256 $AutoDlManifestSha256 `
    --pdb-dir $MonomerPdbDir `
    --out-dir $AuditOutDir
if ($LASTEXITCODE -ne 0) {
    throw "V10 PDB reuse audit failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $AuditJson -PathType Leaf)) {
    throw "V10 PDB reuse audit JSON was not created: $AuditJson"
}
$AuditReport = Get-Content -LiteralPath $AuditJson -Raw | ConvertFrom-Json
if (
    $AuditReport.quality_gate -ne "PASS" -or
    -not [bool]$AuditReport.naturalized_reuse_authorized -or
    [int]$AuditReport.reuse_authorization.variant2_reference_naturalized.authorized_count -ne 151 -or
    [int]$AuditReport.reuse_authorization.variant4_v10_e2e_naturalized.authorized_count -ne 151 -or
    [string]$AuditReport.inputs.autodl_monomer_manifest.sha256 -cne $AutoDlManifestSha256 -or
    [string]$AuditReport.inputs.autodl_monomer_manifest.design_manifest_artifact.sha256 -cne $WindowsDesignSha256
) {
    throw "Audit JSON did not authorize all 151 variant2/variant4 naturalized pairs."
}

Write-Host ""
Write-Host "===== 2/2 MONOMER STRUCTURE + ENERGY RECALCULATION =====" -ForegroundColor Cyan
& $Controller `
    -MonomerOnly `
    -StartStep 10 `
    -RunDir $RunDir `
    -MonomerDesignManifest $V10Manifest `
    -MonomerPdbDir $MonomerPdbDir `
    -MonomerPermeabilityRoot $MonomerPermeabilityRoot `
    -PdbReuseAuditJson $AuditJson `
    -WindowsCondaEnv $WindowsCondaEnv `
    -TmCondaEnv $TmCondaEnv `
    -WslDistribution $WslDistribution `
    -WslCondaRoot $WslCondaRoot `
    -PyRosettaEnv $PyRosettaEnv
if (-not $?) {
    throw "Existing monomer structure/energy stages returned failure."
}

Write-Host ""
Write-Host "===== V10 WINDOWS RECALCULATION COMPLETE =====" -ForegroundColor Green
Write-Host "Reuse audit JSON: $AuditJson"
Write-Host "Structure/energy output: $(Join-Path $RunDir 'monomer')"
