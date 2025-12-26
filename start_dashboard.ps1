# PowerShell script to start SMART Goals Dashboard
# Right-click this file and select "Run with PowerShell"

Write-Host "🚀 Starting SMART Goals Dashboard..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Check current directory
$currentDir = Get-Location
Write-Host "Current directory: $currentDir" -ForegroundColor Yellow

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if Streamlit is available
try {
    $streamlitVersion = streamlit --version 2>&1
    Write-Host "✅ Streamlit found: $streamlitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Streamlit not found. Installing..." -ForegroundColor Yellow
    try {
        pip install streamlit
        Write-Host "✅ Streamlit installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to install Streamlit" -ForegroundColor Red
        Write-Host "Try running: pip install streamlit --user" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if dashboard file exists
$dashboardFile = "smart_dashboard.py"
if (Test-Path $dashboardFile) {
    Write-Host "✅ Dashboard file found: $dashboardFile" -ForegroundColor Green
} else {
    Write-Host "❌ Dashboard file not found: $dashboardFile" -ForegroundColor Red
    Write-Host "Please make sure you're in the correct directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🎯 Starting dashboard..." -ForegroundColor Green
Write-Host "Dashboard will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "To stop: Press Ctrl+C" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    # Start the dashboard
    streamlit run $dashboardFile --server.port 8501 --server.headless true
} catch {
    Write-Host "❌ Error starting dashboard: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Try running manually: streamlit run $dashboardFile" -ForegroundColor Yellow
} finally {
    Write-Host ""
    Write-Host "Dashboard stopped." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}


