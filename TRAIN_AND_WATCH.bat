@echo off
title RL Training - Watch AI Learn Sokoban
cd /d "%~dp0"

echo Installing dependencies...
pip install pygame numpy torch -q 2>nul

echo.
echo ==========================================
echo   Watch AI Learn Sokoban from Scratch!
echo ==========================================
echo   The agent starts RANDOM and gets smarter.
echo.
echo   Controls:
echo     Space = Pause/Resume
echo     F     = Fast mode (skip visuals)
echo     S     = Slow mode (watch every step)
echo     V     = Visual mode (normal)
echo     ESC   = Quit
echo ==========================================
echo.

python TRAIN_AND_WATCH.py
if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong. Check the output above.
    pause
)
