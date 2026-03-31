@echo off
cd /d "%~dp0"

echo Installing dependencies...
pip install pygame numpy -q

echo.
echo ==========================================
echo   Controls:
echo   Arrow Keys  - Move
echo   U           - Undo
echo   R           - Restart
echo   N           - Next Level
echo   H           - Hint
echo   ESC         - Quit
echo ==========================================
echo.

python play_game.py
if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong. Check the output above.
    pause
)
