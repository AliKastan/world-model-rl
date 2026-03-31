@echo off
title AI Learning - ThinkerAgent
cd /d "%~dp0"

echo Installing dependencies...
pip install pygame numpy torch -q 2>nul

echo.
echo ==========================================
echo   ThinkerAgent - World Model + Planning
echo ==========================================
echo   Controls:
echo     Space = Pause/Resume
echo     F     = Fast mode (skip rendering)
echo     S     = Slow mode (watch every step)
echo     V     = Visual mode (normal speed)
echo     1-5   = Change difficulty
echo     ESC   = Quit
echo ==========================================
echo.

python TRAIN_AND_WATCH_THINKER.py
if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong. Check the output above.
    pause
)
