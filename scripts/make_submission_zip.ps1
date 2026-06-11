# Builds a self-contained submission zip for Quantum and arXiv.
# Includes the quantum-class manuscript, sections, figures, refs.bib,
# and the local quantumarticle.cls + quantum.bst so reviewers / arXiv
# can compile without depending on TeX Live having the package.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..").Path
$paperDir = Join-Path $root 'paper'
$out = Join-Path $root 'submission_quantum.zip'
$staging = Join-Path $root '_staging_submission'

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging -Force | Out-Null

Set-Location $paperDir
$mainTargets = @('main_quantum.tex','refs.bib','quantumarticle.cls','quantum.bst','cover_letter.tex')
foreach ($f in $mainTargets) {
    if (-not (Test-Path $f)) { throw "Missing required file: $paperDir\$f" }
    Copy-Item $f $staging
}
Copy-Item 'sections' (Join-Path $staging 'sections') -Recurse
Copy-Item 'figures'  (Join-Path $staging 'figures')  -Recurse

# Optional: include compiled PDFs for reviewer convenience
if (Test-Path 'main_quantum.pdf') { Copy-Item 'main_quantum.pdf' $staging }
if (Test-Path 'cover_letter.pdf') { Copy-Item 'cover_letter.pdf' $staging }

if (Test-Path $out) { Remove-Item $out -Force }
Compress-Archive -Path "$staging\*" -DestinationPath $out -Force
Remove-Item $staging -Recurse -Force

$size = [math]::Round((Get-Item $out).Length / 1KB, 1)
Write-Host "==> Wrote $out  ($size KB)" -ForegroundColor Green
