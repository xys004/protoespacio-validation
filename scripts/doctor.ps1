# Diagnostico del setup: verifica conda env, TeX Live, validators, build.
# Uso: powershell -File scripts\doctor.ps1
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path "$PSScriptRoot\..").Path
$ok = $true

function Check($label, $script) {
    Write-Host "[ .. ] $label" -NoNewline
    try {
        $res = & $script
        Write-Host "`r[ OK ] $label  $res"
    } catch {
        Write-Host "`r[FAIL] $label  $_" -ForegroundColor Red
        $script:ok = $false
    }
}

Check 'Windows Anaconda' { (& 'C:\Users\Nelson\anaconda3\_conda.exe' --version) }
Check "env 'protoespacio'" {
    $py = 'C:\Users\Nelson\.conda\envs\protoespacio\python.exe'
    if (-not (Test-Path $py)) { throw "missing $py" }
    & $py --version
}
Check 'sympy + z3 + pytest' {
    $py = 'C:\Users\Nelson\.conda\envs\protoespacio\python.exe'
    & $py -c "import sympy, z3, pytest; print('sympy', sympy.__version__, 'z3', z3.get_version_string(), 'pytest', pytest.__version__)"
}
Check 'TeX Live latexmk' {
    if (-not (Test-Path 'C:\texlive\2025\bin\windows\latexmk.exe')) { throw 'missing latexmk' }
    (& 'C:\texlive\2025\bin\windows\latexmk.exe' -v 2>&1 | Select-Object -First 1)
}
Check 'TeXstudio' {
    if (Test-Path 'C:\Program Files\texstudio\texstudio.exe') { 'installed' } else { throw 'not installed' }
}
Check 'pytest pasa' {
    Set-Location $root
    $py = 'C:\Users\Nelson\.conda\envs\protoespacio\python.exe'
    $output = & $py -m pytest -q 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { throw $output.Trim() }
    ($output -split "`n" | Select-Object -Last 2 | Out-String).Trim()
}

if ($ok) {
    Write-Host "`nDoctor: setup OK" -ForegroundColor Green
} else {
    Write-Host "`nDoctor: hay errores" -ForegroundColor Red
    exit 1
}
