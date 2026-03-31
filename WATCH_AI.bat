@echo off
cd /d "%~dp0"

echo Installing dependencies...
pip install pygame numpy torch -q

echo.
echo ==========================================
echo   WATCH AI - ThinkerAgent Demo
echo ==========================================
echo   Controls:
echo   Space       - Pause / Resume AI
echo   S           - Slow down
echo   F           - Speed up
echo   N           - Next level
echo   ESC         - Quit
echo ==========================================
echo.

python WATCH_AI.py
if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong. Check the output above.
    pause
)
