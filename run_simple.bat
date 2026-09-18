@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM Use local virtual environment python
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON_EXE=%VENV_PYTHON%"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" main.py --skip-fpga-check --gui-only
pause