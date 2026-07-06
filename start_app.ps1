param(
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

function Write-Step($Message) {
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Write-Fail($Message) {
    Write-Host ""
    Write-Host "FAILED: $Message" -ForegroundColor Red
}

function Test-CommandExists($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-PythonExecutable($Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    try {
        & $Path --version *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

Write-Step "Project"
Write-Host $ProjectRoot

if ($Rebuild -and (Test-Path -LiteralPath ".\venv")) {
    Write-Step "Removing old virtual environment"
    Remove-Item -LiteralPath ".\venv" -Recurse -Force
}

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$VenvPip = Join-Path $ProjectRoot "venv\Scripts\pip.exe"

if (-not (Test-PythonExecutable $VenvPython)) {
    Write-Step "Virtual environment is missing or broken"

    if (Test-Path -LiteralPath ".\venv\pyvenv.cfg") {
        Write-Host "Current venv config:"
        Get-Content -LiteralPath ".\venv\pyvenv.cfg"
        Write-Host ""
    }

    if (-not (Test-CommandExists "python")) {
        Write-Fail "Windows cannot find python on PATH."
        Write-Host "Install Python 3.12 from python.org, check 'Add python.exe to PATH', then open a NEW PowerShell window."
        Write-Host "Also turn off Windows Store app execution aliases for python.exe/python3.exe if they are enabled."
        exit 1
    }

    Write-Host "Creating a fresh virtual environment..."
    if (Test-Path -LiteralPath ".\venv") {
        Remove-Item -LiteralPath ".\venv" -Recurse -Force
    }
    python -m venv venv

    if (-not (Test-PythonExecutable $VenvPython)) {
        Write-Fail "A fresh venv was created, but its Python still cannot run."
        Write-Host "This usually means the Python install or Windows app alias is still broken."
        exit 1
    }
}

Write-Step "Python"
& $VenvPython --version

Write-Step "Installing dependencies"
& $VenvPython -m pip install --upgrade pip
& $VenvPip install -r requirements.txt

if (-not (Test-Path -LiteralPath ".\.env")) {
    Write-Step "Creating .env"
    Copy-Item -LiteralPath ".\.env.example" -Destination ".\.env"
    Write-Host "Created .env from .env.example. Add your real OpenAI or Google key before using AI Insight."
}

Write-Step "Backend import check"
& $VenvPython -c "from backend.main import app; print(app.title)"

$PortInUse = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($PortInUse) {
    Write-Fail "Port 8000 is already in use."
    Write-Host "Close the old terminal/server using port 8000, or run:"
    Write-Host "uvicorn backend.main:app --reload --port 8001"
    exit 1
}

Write-Step "Starting app"
Write-Host "Open this after startup: http://127.0.0.1:8000/"
Write-Host "API docs: http://127.0.0.1:8000/docs"
Write-Host ""
& $VenvPython -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
