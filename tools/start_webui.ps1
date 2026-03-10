param(
    [string]$PythonPath = ".\conda_env\python.exe",
    [string]$Host = "0.0.0.0",
    [int]$Port = 7860,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path $PythonPath)) {
    throw "Python executable not found: $PythonPath"
}

$env:WEBUI_HOST = $Host
$env:WEBUI_PORT = "$Port"
$env:WEBUI_INBROWSER = if ($NoBrowser) { "0" } else { "1" }

Write-Host "Starting Higgs Audio Web UI on http://$Host`:$Port"
& $PythonPath "apps\gradio_app.py"
