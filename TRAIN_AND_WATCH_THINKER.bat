@echo off
title RL Training - ThinkerAgent with World Model
cd /d "%~dp0"

echo Installing dependencies...
pip install pygame numpy torch -q 2>nul

echo.
echo ==========================================
echo   ThinkerAgent - World Model + Planning
echo ==========================================
echo   Learns a world model, then thinks ahead.
echo   Should learn FASTER than pure PPO!
echo.
echo   Controls:
echo     Space = Pause / Resume
echo     F     = Fast mode
echo     S     = Slow mode
echo     V     = Visual mode
echo     ESC   = Quit
echo ==========================================
echo.

python TRAIN_AND_WATCH_THINKER.py
if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong. Check the output above.
    pause
)
