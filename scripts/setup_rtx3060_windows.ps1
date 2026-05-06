param(
    [string]$CudaIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"

function Invoke-PythonCandidate {
    param(
        [string]$File,
        [string[]]$Args
    )
    & $File @Args -c "import sys; ok=(3,10) <= sys.version_info[:2] < (3,14); print(sys.executable); print(sys.version.split()[0]); raise SystemExit(0 if ok else 2)"
}

$Candidates = @(
    @{ File = "py"; Args = @("-3.12") },
    @{ File = (Join-Path $Root "..\.old\.python312\python.exe"); Args = @() },
    @{ File = "python"; Args = @() }
)

$Python = $null
$PythonArgs = @()
foreach ($Candidate in $Candidates) {
    try {
        $Output = Invoke-PythonCandidate -File $Candidate.File -Args $Candidate.Args 2>$null
        if ($LASTEXITCODE -eq 0 -and $Output) {
            $Python = $Candidate.File
            $PythonArgs = $Candidate.Args
            Write-Host "Using Python: $Output"
            break
        }
    } catch {
        continue
    }
}

if (-not $Python) {
    throw "Compatible Python not found. Use Python 3.10-3.13; Python 3.14 is intentionally skipped for CUDA wheel stability."
}

& $Python @PythonArgs -m venv $Venv
$VenvPython = Join-Path $Venv "Scripts\python.exe"

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install torch --index-url $CudaIndexUrl
& $VenvPython -m pip install tqdm numpy pytest tokenizers datasets rich

& $VenvPython (Join-Path $Root "scripts\check_accelerator.py")

Write-Host ""
Write-Host "RTX 3060 profile command:"
Write-Host ".\.venv\Scripts\python.exe .\scripts\train_pretrain.py"
