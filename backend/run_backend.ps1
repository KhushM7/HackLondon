param(
  [string]$BindHost = "0.0.0.0",
  [int]$Port = 8000,
  [switch]$NoRefresh
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

if (-not (Test-Path (Join-Path $scriptDir ".env"))) {
  Write-Warning "No backend/.env file found. MongoDB connection settings may be missing."
}

Write-Host "Starting OrbitGuard backend on http://$BindHost`:$Port" -ForegroundColor Cyan
Write-Host "Python: $python" -ForegroundColor DarkGray

$refreshJob = $null
if (-not $NoRefresh) {
  $refreshScript = @"
`$ProgressPreference = 'SilentlyContinue'
for (`$i = 0; `$i -lt 90; `$i++) {
  try {
    Invoke-WebRequest -Uri 'http://127.0.0.1:$Port/' -Method Get -TimeoutSec 2 | Out-Null
    break
  } catch {
    Start-Sleep -Milliseconds 500
  }
}

try {
  Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:$Port/ingest/refresh' -Headers @{
    'Accept' = 'application/json'
    'Content-Type' = 'application/json'
    'Origin' = 'http://localhost:3000'
  } -TimeoutSec 120 | Out-Null
} catch {
  # Keep launcher resilient even if upstream ingestion fails.
}
"@
  $refreshJob = Start-Job -ScriptBlock ([scriptblock]::Create($refreshScript))
}

try {
  & $python -m uvicorn app.main:app --reload --host $BindHost --port $Port
} finally {
  if ($refreshJob -ne $null) {
    try {
      if ($refreshJob.State -eq "Running") {
        Stop-Job $refreshJob -Force | Out-Null
      }
    } catch {}
    Remove-Job $refreshJob -Force -ErrorAction SilentlyContinue | Out-Null
  }
}
