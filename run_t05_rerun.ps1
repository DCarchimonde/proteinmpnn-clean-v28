param(
    [string]$Python = "",
    [string]$CondaEnvironment = "wain",
    [int]$BatchSize = 16,
    [int[]]$Seeds = @(101, 202, 303, 404, 505),
    [string]$OutputDir = "",
    [string]$PermeabilityCsv = "",
    [switch]$SkipGeneration,
    [switch]$Force,
    [switch]$AllowCpu,
    [switch]$InstallTorchIfMissing,
    [switch]$AllowPartialPredictions
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Generator = Join-Path $RepoRoot "paper_clean_v28\rerun_t05\01_generate_t05_multiseed.py"
$Selector = Join-Path $RepoRoot "paper_clean_v28\rerun_t05\02_select_after_permeability.py"
$Plan = Join-Path $RepoRoot "paper_clean_v28\rerun_t05\target_plan.json"
$FrozenTorchVersion = "2.5.1"
$FrozenTorchIndex = "https://download.pytorch.org/whl/cu124"
$FrozenNumpyVersion = "2.0.1"

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

function Add-PythonCandidate {
    param(
        [System.Collections.ArrayList]$Candidates,
        [hashtable]$Seen,
        [string]$CandidatePath,
        [string]$Label,
        [bool]$InstallTarget = $false
    )

    if ([string]::IsNullOrWhiteSpace($CandidatePath)) {
        return
    }

    $ResolvedPath = $CandidatePath
    if (-not [System.IO.Path]::IsPathRooted($ResolvedPath)) {
        $Command = Get-Command -Name $ResolvedPath -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $Command) {
            return
        }
        $ResolvedPath = $Command.Source
    }
    if (-not (Test-Path -LiteralPath $ResolvedPath -PathType Leaf)) {
        return
    }

    $ResolvedPath = [System.IO.Path]::GetFullPath($ResolvedPath)
    $Key = $ResolvedPath.ToLowerInvariant()
    if ($Seen.ContainsKey($Key)) {
        if ($InstallTarget) {
            $Seen[$Key].InstallTarget = $true
        }
        return
    }

    $Candidate = [PSCustomObject]@{
        Path = $ResolvedPath
        Label = $Label
        InstallTarget = $InstallTarget
    }
    [void]$Candidates.Add($Candidate)
    $Seen[$Key] = $Candidate
}

function Get-CondaEnvironmentPaths {
    $CondaCommand = ""
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE) -and
        (Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
        $CondaCommand = $env:CONDA_EXE
    } else {
        $CondaApplication = Get-Command -Name "conda.exe" -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $CondaApplication) {
            $CondaCommand = $CondaApplication.Source
        }
    }

    if ([string]::IsNullOrWhiteSpace($CondaCommand)) {
        return @()
    }

    try {
        $RawJson = (& $CondaCommand env list --json 2>$null | Out-String)
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($RawJson)) {
            return @()
        }
        $Parsed = $RawJson | ConvertFrom-Json
        return @($Parsed.envs | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    } catch {
        Write-Host "Conda environment discovery warning: $($_.Exception.Message)" -ForegroundColor DarkYellow
        return @()
    }
}

function Invoke-PythonProbe {
    param(
        [PSCustomObject]$Candidate,
        [bool]$RequireTorch
    )

    if ($RequireTorch) {
        $ProbeCode = 'import json, sys; import numpy, torch; print("__T05_PYTHON__" + json.dumps({"executable": sys.executable, "python": sys.version.split()[0], "numpy": numpy.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda": bool(torch.cuda.is_available()), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}))'
    } else {
        $ProbeCode = 'import json, sys; print("__T05_PYTHON__" + json.dumps({"executable": sys.executable, "python": sys.version.split()[0], "numpy": "not required", "torch": "not required", "torch_cuda": None, "cuda": False, "device": "not required"}))'
    }

    # Windows PowerShell 5.1 can corrupt nested quotes when a Python program is
    # passed through `python -c $ProbeCode`.  It can also promote the first line
    # written to native stderr into a terminating PowerShell error, hiding the
    # useful remainder of a Python traceback.  Execute a temporary .py file and
    # capture stdout/stderr in files so the probe is both quote-safe and fully
    # diagnosable on the user's Windows setup.
    $ProbeStem = "proteinmpnn_t05_probe_$([Guid]::NewGuid().ToString('N'))"
    $ProbePath = Join-Path ([System.IO.Path]::GetTempPath()) ($ProbeStem + ".py")
    $StdoutPath = Join-Path ([System.IO.Path]::GetTempPath()) ($ProbeStem + ".stdout.txt")
    $StderrPath = Join-Path ([System.IO.Path]::GetTempPath()) ($ProbeStem + ".stderr.txt")
    try {
        $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($ProbePath, $ProbeCode, $Utf8NoBom)
        & $Candidate.Path $ProbePath 1> $StdoutPath 2> $StderrPath
        $ProbeExitCode = $LASTEXITCODE
        $ProbeOutput = @()
        if (Test-Path -LiteralPath $StdoutPath -PathType Leaf) {
            $ProbeOutput += @(Get-Content -LiteralPath $StdoutPath)
        }
        if (Test-Path -LiteralPath $StderrPath -PathType Leaf) {
            $ProbeOutput += @(Get-Content -LiteralPath $StderrPath)
        }
    } catch {
        $ProbeOutput = @($_.Exception.Message)
        $ProbeExitCode = 1
    } finally {
        foreach ($TemporaryPath in @($ProbePath, $StdoutPath, $StderrPath)) {
            if (Test-Path -LiteralPath $TemporaryPath) {
                Remove-Item -LiteralPath $TemporaryPath -Force -ErrorAction SilentlyContinue
            }
        }
    }

    $TextLines = @($ProbeOutput | ForEach-Object { [string]$_ })
    $MarkerLine = $TextLines |
        Where-Object { $_.StartsWith("__T05_PYTHON__") } |
        Select-Object -Last 1
    if ($ProbeExitCode -eq 0 -and $null -ne $MarkerLine) {
        try {
            $Info = $MarkerLine.Substring("__T05_PYTHON__".Length) | ConvertFrom-Json
            return [PSCustomObject]@{
                Candidate = $Candidate
                Success = $true
                Info = $Info
                Error = ""
            }
        } catch {
            $TextLines += $_.Exception.Message
        }
    }

    $ErrorSummary = ($TextLines | Select-Object -Last 4) -join " | "
    return [PSCustomObject]@{
        Candidate = $Candidate
        Success = $false
        Info = $null
        Error = $ErrorSummary
    }
}

function Install-FrozenTorch {
    param([string]$PythonPath)

    Write-Host ""
    Write-Host "No importable PyTorch was found in the requested '$CondaEnvironment' environment." -ForegroundColor Yellow
    Write-Host "Installing the frozen project dependency into:" -ForegroundColor Yellow
    Write-Host "  $PythonPath" -ForegroundColor Cyan
    Write-Host "  numpy=$FrozenNumpyVersion, torch=$FrozenTorchVersion, CUDA wheel=cu124" -ForegroundColor Cyan

    & $PythonPath -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        & $PythonPath -m ensurepip --upgrade 2>&1 | ForEach-Object { Write-Host $_ }
        Assert-LastExitCode "pip bootstrap"
    }

    & $PythonPath -c "import numpy" *> $null
    if ($LASTEXITCODE -ne 0) {
        & $PythonPath -m pip install "numpy==$FrozenNumpyVersion" 2>&1 |
            ForEach-Object { Write-Host $_ }
        Assert-LastExitCode "NumPy installation"
    }

    & $PythonPath -m pip install --upgrade "torch==$FrozenTorchVersion" --index-url $FrozenTorchIndex 2>&1 |
        ForEach-Object { Write-Host $_ }
    Assert-LastExitCode "PyTorch installation"
}

function Resolve-ProjectPython {
    param([bool]$RequireTorch)

    $Candidates = New-Object System.Collections.ArrayList
    $Seen = @{}

    if (-not [string]::IsNullOrWhiteSpace($Python)) {
        Add-PythonCandidate $Candidates $Seen $Python "-Python argument" $true
    }

    $CondaPaths = @(Get-CondaEnvironmentPaths)
    foreach ($EnvironmentPath in $CondaPaths) {
        $EnvironmentName = Split-Path -Leaf ([string]$EnvironmentPath)
        if ($EnvironmentName -ieq $CondaEnvironment) {
            Add-PythonCandidate $Candidates $Seen (Join-Path $EnvironmentPath "python.exe") "Conda environment '$CondaEnvironment'" $true
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $CurrentIsRequested = ($env:CONDA_DEFAULT_ENV -ieq $CondaEnvironment) -or
            ((Split-Path -Leaf $env:CONDA_PREFIX) -ieq $CondaEnvironment)
        Add-PythonCandidate $Candidates $Seen (Join-Path $env:CONDA_PREFIX "python.exe") "current CONDA_PREFIX ($env:CONDA_DEFAULT_ENV)" $CurrentIsRequested
    }

    foreach ($EnvironmentPath in $CondaPaths) {
        $EnvironmentName = Split-Path -Leaf ([string]$EnvironmentPath)
        Add-PythonCandidate $Candidates $Seen (Join-Path $EnvironmentPath "python.exe") "Conda environment '$EnvironmentName'" ($EnvironmentName -ieq $CondaEnvironment)
    }
    Add-PythonCandidate $Candidates $Seen "python" "python on PATH" $false

    if ($Candidates.Count -eq 0) {
        throw "No Python executable was found. Activate Conda environment '$CondaEnvironment' or pass -Python with the full path to python.exe."
    }

    $ProbeResults = New-Object System.Collections.ArrayList
    $Selected = $null
    foreach ($Candidate in $Candidates) {
        $Probe = Invoke-PythonProbe $Candidate $RequireTorch
        [void]$ProbeResults.Add($Probe)
        if (-not $Probe.Success) {
            Write-Host "Python candidate rejected: $($Candidate.Label) [$($Candidate.Path)]" -ForegroundColor DarkYellow
            continue
        }
        if ($RequireTorch -and -not [bool]$Probe.Info.cuda -and -not $AllowCpu) {
            Write-Host "Python candidate has PyTorch but no available CUDA: $($Candidate.Path)" -ForegroundColor DarkYellow
            continue
        }
        $Selected = $Probe
        break
    }

    if ($null -eq $Selected -and $RequireTorch -and $InstallTorchIfMissing) {
        $InstallCandidate = $Candidates |
            Where-Object { $_.InstallTarget } |
            Select-Object -First 1
        if ($null -eq $InstallCandidate) {
            throw "The Conda environment '$CondaEnvironment' could not be located, so PyTorch was not installed into an unrelated environment. Activate it or pass -Python with its full python.exe path."
        }

        $ExistingProbe = $ProbeResults |
            Where-Object { $_.Candidate.Path -ieq $InstallCandidate.Path } |
            Select-Object -First 1
        if ($null -ne $ExistingProbe -and $ExistingProbe.Success) {
            throw "PyTorch imports in '$($InstallCandidate.Path)', but CUDA is unavailable. Reinstalling cannot safely diagnose a GPU-driver or CPU-wheel problem."
        }

        Install-FrozenTorch $InstallCandidate.Path
        $InstalledProbe = Invoke-PythonProbe $InstallCandidate $true
        if (-not $InstalledProbe.Success) {
            throw "PyTorch installation finished, but the import probe still failed: $($InstalledProbe.Error)"
        }
        if (-not [bool]$InstalledProbe.Info.cuda -and -not $AllowCpu) {
            throw "PyTorch $($InstalledProbe.Info.torch) imports, but CUDA is unavailable. Check the NVIDIA driver instead of generating 13,500 sequences on CPU."
        }
        $Selected = $InstalledProbe
    }

    if ($null -eq $Selected) {
        $FailureLines = @($ProbeResults | ForEach-Object {
            if ($_.Success) {
                "- $($_.Candidate.Path): torch imports but CUDA=$($_.Info.cuda)"
            } else {
                "- $($_.Candidate.Path): $($_.Error)"
            }
        })
        $InstallHint = ""
        if ($RequireTorch) {
            $InstallHint = "`nRerun with -InstallTorchIfMissing to install torch=$FrozenTorchVersion+cu124 only into '$CondaEnvironment'."
        }
        throw ("No usable Python environment was found.`n" + ($FailureLines -join "`n") + $InstallHint)
    }

    Write-Host "Selected Python: $($Selected.Info.executable)" -ForegroundColor Green
    Write-Host "Python version:  $($Selected.Info.python)"
    if ($RequireTorch) {
        Write-Host "NumPy version:   $($Selected.Info.numpy)"
        Write-Host "PyTorch version: $($Selected.Info.torch)"
        Write-Host "Torch CUDA:      $($Selected.Info.torch_cuda)"
        Write-Host "GPU:             $($Selected.Info.device)"
    }
    return [string]$Selected.Candidate.Path
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

$ResolvedPython = Resolve-ProjectPython (-not [bool]$SkipGeneration)

Push-Location $RepoRoot
try {
    & $ResolvedPython $Generator --plan $Plan --seeds $Seeds --plan-only
    Assert-LastExitCode "Protocol preflight"

    if (-not $SkipGeneration) {
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

        & $ResolvedPython @GenerateArguments
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
        Write-Host "powershell -NoProfile -ExecutionPolicy Bypass -File .\run_t05_rerun.ps1 -SkipGeneration" -ForegroundColor Cyan
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
    & $ResolvedPython @SelectArguments
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
