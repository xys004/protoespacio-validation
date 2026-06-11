# Build politico Windows-nativo: tests antes que PDF.
# Activa el env conda 'protoespacio' de C:\Users\Nelson\anaconda3 y usa TeX Live 2025.
#
# Equivalente bash en WSL: scripts/build_book.sh
$ErrorActionPreference = 'Stop'

$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root

# --- Localiza env 'protoespacio'. conda 4.10 lo crea en %USERPROFILE%\.conda\envs.
$envPython = 'C:\Users\Nelson\.conda\envs\protoespacio\python.exe'
if (-not (Test-Path $envPython)) {
    $alt = 'C:\Users\Nelson\anaconda3\envs\protoespacio\python.exe'
    if (Test-Path $alt) {
        $envPython = $alt
    } else {
        throw "Env 'protoespacio' no encontrado. Crear con:`n  & 'C:\Users\Nelson\anaconda3\_conda.exe' create -y -n protoespacio -c conda-forge --override-channels python=3.12 sympy z3-solver pytest numpy scipy matplotlib"
    }
}

# --- Localiza latexmk (TeX Live 2025)
$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
if (-not $latexmk) {
    $candidate = 'C:\texlive\2025\bin\windows\latexmk.exe'
    if (Test-Path $candidate) {
        $env:Path = "C:\texlive\2025\bin\windows;$env:Path"
        $latexmk = Get-Command latexmk
    } else {
        throw 'latexmk no encontrado. Verificar instalacion de TeX Live 2025.'
    }
}

Write-Host '==> pytest' -ForegroundColor Cyan
& $envPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'pytest fallo; PDF no se produce' }

Write-Host '==> latexmk book/main.tex' -ForegroundColor Cyan
Set-Location (Join-Path $root 'book')
& $latexmk.Path -pdf -interaction=nonstopmode -halt-on-error main.tex
if ($LASTEXITCODE -ne 0) { throw 'latexmk fallo' }

Write-Host "==> OK: $root\book\main.pdf" -ForegroundColor Green
