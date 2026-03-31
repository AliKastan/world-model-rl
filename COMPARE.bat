@echo off
title PPO vs ThinkerAgent - Live Comparison
cd /d "%~dp0"

echo Installing dependencies...
pip install pygame numpy torch -q 2>nul

echo.
echo ==========================================
echo   PPO vs ThinkerAgent - Side by Side
echo ==========================================
echo   Watch both agents learn simultaneously!
echo   PPO (red) learns by trial and error.
echo   Thinker (blue) learns a world model
echo   and plans ahead - much faster!
echo.
echo   Controls:
echo     Space = Pause/Resume
echo     F     = Fast mode
echo     S     = Slow mode
echo     V     = Visual mode
echo     1-3   = Change difficulty
echo     ESC   = Quit
echo ==========================================
echo.

python COMPARE.py
if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong. Check the output above.
    pause
)
