param(
    [string]$Python = "python",
    [int]$BatchSize = 16,
    [int[]]$Seeds = @(101, 202, 303, 404, 505),
    [string]$OutputDir = "",
    [string]$PermeabilityCsv = "",
    [switch]$SkipGeneration,
    [switch]$Force,
    [switch]$AllowCpu,
    [switch]$AllowPartialPredictions
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Generator = Join-Path $RepoRoot "paper_clean_v28\rerun_t05\01_generate_t05_multiseed.py"
$Selector = Join-Path $RepoRoot "paper_clean_v28\rerun_t05\02_select_after_permeability.py"
$Plan = Join-Path $RepoRoot "paper_clean_v28\rerun_t05\target_plan.json"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "paper_clean_v28_outputs\rerun_temperature_0.5_multiseed"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputDir))
}

function Assert-LastExitCode {
    param([string]$Stage)
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
}

Write-Host "============================================================"
Write-Host "T=0.5 MULTISEED LOCAL PRESCREEN"
Write-Host "Repository: $RepoRoot"
Write-Host "Output:     $OutputDir"
Write-Host "============================================================"

foreach ($RequiredPath in @($Generator, $Selector, $Plan)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required file is missing: $RequiredPath"
    }
}

Push-Location $RepoRoot
try {
    & $Python $Generator --plan $Plan --seeds $Seeds --plan-only
    Assert-LastExitCode "Protocol preflight"

    if (-not $SkipGeneration) {
        & $Python -c "import numpy, torch; print('Python/Torch preflight PASS'); print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
        Assert-LastExitCode "Python/Torch preflight"

        $GenerateArguments = @(
            $Generator,
            "--plan", $Plan,
            "--model_path", (Join-Path $RepoRoot "frankenstein_v28.pt"),
            "--native_jsonl", (Join-Path $RepoRoot "17_complexes_native.jsonl"),
            "--best_csv", (Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\best_designs.csv"),
            "--old_designs_csv", (Join-Path $RepoRoot "paper_clean_v28_outputs\generated_fasta_clean_auto_single\all_designs.csv"),
            "--out_dir", $OutputDir,
            "--batch_size", $BatchSize,
            "--seeds"
        )
        foreach ($Seed in $Seeds) {
            $GenerateArguments += [string]$Seed
        }
        if ($Force) {
            $GenerateArguments += "--overwrite"
        }
        if ($AllowCpu) {
            $GenerateArguments += "--allow-cpu"
        }

        & $Python @GenerateArguments
        Assert-LastExitCode "Sequence generation"
    } else {
        $GenerationManifest = Join-Path $OutputDir "generation_manifest.json"
        if (-not (Test-Path -LiteralPath $GenerationManifest -PathType Leaf)) {
            throw "-SkipGeneration was used but generation_manifest.json is missing: $GenerationManifest"
        }
        Write-Host "Generation skipped; using the completed isolated run."
    }

    if ([string]::IsNullOrWhiteSpace($PermeabilityCsv)) {
        $DefaultPrediction = Join-Path $OutputDir "permeability_predictions.csv"
        if (Test-Path -LiteralPath $DefaultPrediction -PathType Leaf) {
            $PermeabilityCsv = $DefaultPrediction
        }
    } elseif (-not [System.IO.Path]::IsPathRooted($PermeabilityCsv)) {
        $PermeabilityCsv = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PermeabilityCsv))
    }

    if ([string]::IsNullOrWhiteSpace($PermeabilityCsv) -or -not (Test-Path -LiteralPath $PermeabilityCsv -PathType Leaf)) {
        Write-Host ""
        Write-Host "===== LOCAL GENERATION COMPLETE =====" -ForegroundColor Green
        Write-Host "Run the SAME permeability model on this file:"
        Write-Host (Join-Path $OutputDir "permeability_input.csv") -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Save its output as:"
        Write-Host (Join-Path $OutputDir "permeability_predictions.csv") -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Then rerun this same controller with:"
        Write-Host "powershell -ExecutionPolicy Bypass -File .\run_t05_rerun.ps1 -SkipGeneration" -ForegroundColor Cyan
        Write-Host "No structure task has been sent yet; this prevents blind company-machine runs."
        exit 0
    }

    $SelectArguments = @(
        $Selector,
        "--run_dir", $OutputDir,
        "--plan", $Plan,
        "--permeability_csv", $PermeabilityCsv
    )
    if ($AllowPartialPredictions) {
        $SelectArguments += "--allow-partial-predictions"
    }
    & $Python @SelectArguments
    Assert-LastExitCode "Permeability prescreen"

    Write-Host ""
    Write-Host "===== ALL AVAILABLE LOCAL STAGES DONE =====" -ForegroundColor Green
    Write-Host "Send Shang-ge this manifest:"
    Write-Host (Join-Path $OutputDir "selected_for_structure\structure_tasks_for_shangge.csv") -ForegroundColor Yellow
    Write-Host "Individual FASTA files:"
    Write-Host (Join-Path $OutputDir "selected_for_structure\structure_inputs_for_shangge") -ForegroundColor Yellow
} finally {
    Pop-Location
}
