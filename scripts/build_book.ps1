# Wrapper Windows que delega al script bash en WSL.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..").Path
$wslPath = $root -replace '^([A-Za-z]):','/mnt/$1' -replace '\\','/'
$wslPath = $wslPath.ToLower() -replace '/mnt/([a-z])','/mnt/$1'
wsl.exe -d Ubuntu -- bash -lc "cd '$wslPath' && bash scripts/build_book.sh"
