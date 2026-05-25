param(
  [int]$Port = 9223,
  [string]$Items = "all",
  [string]$ChromeProfile = "$env:TEMP\codex-linkedin-chrome-profile",
  [string]$BrowserPath = "",
  [string]$ProfileDirectory = "Profile 1",
  [int]$ExpandIterations = 6,
  [double]$SleepSeconds = 1.0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).ProviderPath
$PackDir = Join-Path $ProjectRoot "reports\linkedin_credit_risk_andrija_djurovic\logged_in_review"
$CaptureScript = Join-Path $ProjectRoot "scripts\research\capture_linkedin_logged_in_cdp.py"
$AnalyzeScript = Join-Path $ProjectRoot "scripts\research\analyze_linkedin_logged_in_review.py"
$QueueScript = Join-Path $ProjectRoot "scripts\research\build_andrija_logged_in_review_queue.py"
$IntakeScript = Join-Path $ProjectRoot "scripts\research\build_andrija_logged_in_project_intake.py"
$TempWork = Join-Path $env:TEMP "codex-uv-playwright"
$CdpUrl = "http://127.0.0.1:$Port"

function Test-Cdp {
  param([string]$Url)
  try {
    $null = Invoke-WebRequest -UseBasicParsing "$Url/json/version" -TimeoutSec 3
    return $true
  } catch {
    return $false
  }
}

if (-not (Test-Cdp $CdpUrl)) {
  if (-not $BrowserPath) {
    $BrowserPath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
    if (-not (Test-Path $BrowserPath)) {
      $BrowserPath = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    }
  }
  if (-not (Test-Path $BrowserPath)) {
    throw "Browser executable not found: $BrowserPath"
  }
  if (-not (Test-Path $ChromeProfile)) {
    throw "Browser profile not found: $ChromeProfile"
  }

  $browserArgs = @(
    "--remote-debugging-port=$Port",
    "--user-data-dir=$ChromeProfile",
    "about:blank"
  )
  if ($ProfileDirectory) {
    $browserArgs = @(
      "--remote-debugging-port=$Port",
      "--user-data-dir=$ChromeProfile",
      "--profile-directory=$ProfileDirectory",
      "about:blank"
    )
  }

  Start-Process -FilePath $BrowserPath -ArgumentList $browserArgs
  Start-Sleep -Seconds 3
}

if (-not (Test-Cdp $CdpUrl)) {
  throw "CDP endpoint unavailable at $CdpUrl after launching the browser."
}

New-Item -ItemType Directory -Force -Path $TempWork | Out-Null
Set-Location $ProjectRoot

uv run --no-project --with playwright python $QueueScript
uv run --no-project --with playwright python $CaptureScript `
  --pack-dir $PackDir `
  --cdp-url $CdpUrl `
  --items $Items `
  --sleep-seconds $SleepSeconds `
  --expand-iterations $ExpandIterations

if ($Items -eq "all") {
  uv run --no-project --with playwright python $AnalyzeScript `
    --pack-dir $PackDir `
    --resolve
  uv run --no-project --with playwright python $IntakeScript
}
