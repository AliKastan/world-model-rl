@echo off
cd /d "%~dp0"

echo Installing dependencies...
pip install pygame numpy -q 2>nul

echo.
echo ==========================================
echo   SOKOBAN - 60 Levels
echo ==========================================
echo   Controls:
echo     Arrow Keys - Move
echo     U          - Undo
echo     R          - Restart
echo     N          - Next Level
echo     P          - Previous Level
echo     H          - Hint
echo     ESC        - Level Select
echo     Q          - Quit
echo ==========================================
echo.

python play_game.py
if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong. Check the output above.
    pause
)
