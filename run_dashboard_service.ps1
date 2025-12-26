# PowerShell script to run Streamlit dashboard as a service
# Run this script as Administrator for best results

param(
    [string]$Port = "8503",
    [string]$Host = "localhost"
)

$DashboardPath = Join-Path $PSScriptRoot "smart_dashboard.py"
$LogPath = Join-Path $PSScriptRoot "dashboard.log"

Write-Host "Starting SMART Goals Dashboard Service..." -ForegroundColor Green
Write-Host "Dashboard will be available at: http://$Host`:$Port" -ForegroundColor Cyan
Write-Host "Log file: $LogPath" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the service" -ForegroundColor Red
Write-Host ""

# Function to start dashboard
function Start-Dashboard {
    try {
        Write-Host "Starting Streamlit dashboard..." -ForegroundColor Green
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path $LogPath -Value "[$timestamp] Starting dashboard on port $Port"
        
        # Start Streamlit with specific configuration
        streamlit run $DashboardPath --server.port $Port --server.headless true --server.runOnSave false --browser.gatherUsageStats false
        
    } catch {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path $LogPath -Value "[$timestamp] Error starting dashboard: $($_.Exception.Message)"
        Write-Host "Error starting dashboard: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Main loop
try {
    while ($true) {
        Start-Dashboard
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path $LogPath -Value "[$timestamp] Dashboard stopped, restarting in 5 seconds..."
        Write-Host "Dashboard stopped, restarting in 5 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
} catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "[$timestamp] Service stopped by user"
    Write-Host "Service stopped." -ForegroundColor Green
} 