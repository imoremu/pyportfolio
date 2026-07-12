<#
.SYNOPSIS
    Recreates the .claude -> .agents junctions for NereoSkills on Windows.

.DESCRIPTION
    Windows (without Developer Mode / admin) cannot create the POSIX symlinks the
    NereoSkills README assumes, so we use directory junctions instead. They are
    functionally equivalent for reading skills/agents and need no admin rights.

    The junctions are per-machine and .gitignored; the real content lives in the
    .agents submodule. Run this once after cloning on a new machine:

        git clone --recurse-submodules <repo>
        # or, on an existing clone:
        git submodule update --init --recursive
        powershell -ExecutionPolicy Bypass -File scripts/setup-nereolinks.ps1
#>

$ErrorActionPreference = "Stop"

$root   = Split-Path -Parent $PSScriptRoot   # repo root (this script lives in scripts/)
$claude = Join-Path $root ".claude"
$agents = Join-Path $root ".agents"

if (-not (Test-Path (Join-Path $agents "skills"))) {
    throw "No existe '$agents\skills'. Ejecuta primero: git submodule update --init --recursive"
}

if (-not (Test-Path $claude)) { New-Item -ItemType Directory -Path $claude | Out-Null }

foreach ($name in @("skills", "agents")) {
    $link   = Join-Path $claude $name
    $target = Join-Path $agents $name
    if (Test-Path $link) { (Get-Item $link).Delete() }
    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    Write-Host ("OK: .claude\{0} -> .agents\{0}" -f $name)
}

Write-Host "Junctions de NereoSkills listas."
