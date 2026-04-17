@echo off
REM Activate the 'visualagent' conda environment
CALL conda activate visualagent

REM Navigate to the backend folder
cd /d backend

REM Run the backend app
python app.py

pause
