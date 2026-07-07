@echo off
REM ============================================================
REM  Computer Use Agent - one-shot launcher
REM  Starts the Flask backend and the Vite frontend, each in its
REM  own window, then opens the web UI. Close either window to
REM  stop that service.
REM ============================================================

REM %~dp0 = folder this .bat lives in, so it works from any CWD.
set "ROOT=%~dp0"

echo Starting backend (Flask, port 5000) ...
start "Agent Backend"  cmd /k "call conda activate visualagent && cd /d "%ROOT%backend" && python app.py"

echo Starting frontend (Vite, port 3000) ...
start "Agent Frontend" cmd /k "call conda activate visualagent && cd /d "%ROOT%frontend" && npm run dev"

REM Give the Vite dev server a moment to come up before opening the page.
echo Waiting for the dev server to start ...
timeout /t 6 /nobreak >nul

start "" http://localhost:3000/

echo.
echo Backend  -> http://localhost:5000
echo Frontend -> http://localhost:3000
echo Two service windows opened. Close them (or press Ctrl+C in each) to stop.
