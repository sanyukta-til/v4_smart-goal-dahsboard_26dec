@echo off
title SMART Goals Dashboard - Modern Colors
color 0A

echo ========================================
echo    SMART Goals Dashboard - Modern UI
echo ========================================
echo.
echo 🎨 Enhanced with modern colors and styling
echo 🚀 Starting your beautiful dashboard...
echo.

REM Change to the dashboard directory
cd /d "%~dp0"

REM Check if Streamlit is installed
streamlit --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Streamlit not found. Installing...
    pip install streamlit
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to install Streamlit
        echo Please install manually: pip install streamlit
        pause
        exit /b 1
    )
)

echo ✅ Streamlit found. Starting dashboard...
echo.
echo 🌈 Dashboard will open with modern colors at: http://localhost:8503
echo.
echo ✨ Features:
echo    • Modern gradient backgrounds
echo    • Enhanced KPI cards with hover effects
echo    • Beautiful table styling
echo    • Smooth animations and transitions
echo.
echo To stop: Press Ctrl+C in this window
echo.

REM Start the dashboard
streamlit run smart_dashboard.py --server.port 8503

echo.
echo Dashboard stopped.
pause
