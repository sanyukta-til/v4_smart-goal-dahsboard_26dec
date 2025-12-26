@echo off
echo Starting SMART Goals Dashboard...
echo.
echo Dashboard will be available at: http://localhost:8503
echo.
echo Press Ctrl+C to stop the dashboard
echo.
streamlit run smart_dashboard.py --server.port 8503
pause
