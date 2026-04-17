$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root "preprint\build.ps1"

if (-not (Test-Path $script)) {
    throw "Missing build script: $script"
}

& $script
exit $LASTEXITCODE
