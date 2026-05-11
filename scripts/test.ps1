# Solo tests (Windows-nativo).
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root
$envPython = 'C:\Users\Nelson\.conda\envs\protoespacio\python.exe'
if (-not (Test-Path $envPython)) { $envPython = 'C:\Users\Nelson\anaconda3\envs\protoespacio\python.exe' }
& $envPython -m pytest @args
