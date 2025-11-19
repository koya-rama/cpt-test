# PowerShell script to enable long paths on Windows
# Must be run as Administrator

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Enable Windows Long Paths" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To run as Administrator:" -ForegroundColor Yellow
    Write-Host "1. Right-click PowerShell" -ForegroundColor Yellow
    Write-Host "2. Select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host "3. Run: .\scripts\enable_long_paths.ps1" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

Write-Host "[1/3] Enabling long paths in Windows Registry..." -ForegroundColor Green

try {
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                     -Name "LongPathsEnabled" `
                     -Value 1 `
                     -PropertyType DWORD `
                     -Force | Out-Null
    Write-Host "✓ Long paths enabled in registry" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to enable long paths in registry" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Write-Host "[2/3] Enabling long paths in Git (if available)..." -ForegroundColor Green

if (Get-Command git -ErrorAction SilentlyContinue) {
    try {
        git config --system core.longpaths true
        Write-Host "✓ Long paths enabled in Git" -ForegroundColor Green
    } catch {
        Write-Host "✗ Failed to enable long paths in Git" -ForegroundColor Yellow
    }
} else {
    Write-Host "⊘ Git not installed, skipping" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[3/3] Creating short temp directory..." -ForegroundColor Green

try {
    New-Item -ItemType Directory -Force -Path "C:\Temp" | Out-Null
    Write-Host "✓ Created C:\Temp for short paths" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to create C:\Temp" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Configuration Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: You must RESTART your computer for changes to take effect!" -ForegroundColor Yellow
Write-Host ""
Write-Host "After restart, try installing Flash Attention:" -ForegroundColor White
Write-Host "  pip install flash-attn --no-build-isolation" -ForegroundColor Cyan
Write-Host ""

pause
