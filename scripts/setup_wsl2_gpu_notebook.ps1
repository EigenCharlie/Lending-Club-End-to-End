[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Get-VirtualizationState {
    $cpu = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
    return [bool]$cpu.VirtualizationFirmwareEnabled
}

Write-Step "Checking virtualization firmware state"
$virtEnabled = Get-VirtualizationState
if (-not $virtEnabled) {
    Write-Host "Virtualization is disabled in BIOS/UEFI. Enable SVM/AMD-V/VT-x first." -ForegroundColor Yellow
    Write-Host "After enabling it, reboot and run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Step "Checking WSL status"
$statusOutput = & wsl --status 2>&1 | Out-String
Write-Host $statusOutput

if ($statusOutput -match "se aplicarán una vez que se reinicie el sistema" -or
    $statusOutput -match "changes will not take effect until the system is rebooted") {
    Write-Host "A system reboot is still pending for WSL feature activation." -ForegroundColor Yellow
    Write-Host "Reboot Windows and rerun this script." -ForegroundColor Yellow
    exit 1
}

Write-Step "Ensuring default WSL version is 2"
& wsl --set-default-version 2 | Out-Null

Write-Step "Ensuring distro '$Distro' is installed"
$distros = (& wsl -l -q 2>$null | ForEach-Object { $_.Trim() }) -ne ""
if (-not ($distros -contains $Distro)) {
    & wsl --install -d $Distro
    Write-Host "Distro install requested. If prompted, complete initialization then rerun this script." -ForegroundColor Yellow
}

Write-Step "Running Linux setup + notebook execution"
$projectWin = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectWsl = "/mnt/$($projectWin[0].ToLower())/" + ($projectWin.Substring(3) -replace "\\", "/")
$scriptWsl = "$projectWsl/scripts/setup_gpu_notebook_wsl.sh"

& wsl -d $Distro -u root -- bash -lc "chmod +x '$scriptWsl' && '$scriptWsl'"

Write-Step "Done"
Write-Host "Notebook and benchmark artifacts should be under reports/gpu_benchmark/" -ForegroundColor Green
