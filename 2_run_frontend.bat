@echo off
REM Activate the 'visualagent' conda environment
CALL conda activate visualagent

REM Navigate to the backend folder
cd /d frontend

REM Run the backend app
npm run dev

pause
