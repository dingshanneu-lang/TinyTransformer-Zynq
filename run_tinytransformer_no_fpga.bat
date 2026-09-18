@echo off
chcp 65001 >nul
title TinyTransformer-Zynq Cat Dog Classification (No FPGA)

echo =============================================
echo  TinyTransformer-Zynq Cat Dog Classification
echo =============================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Use local virtual environment python
set "VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON_EXE=%VENV_PYTHON%"
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [ERROR] python.exe not found
        echo Please ensure .venv exists or python is in PATH
        pause
        exit /b 1
    )
)

echo [INFO] Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo.

echo [START] Skipping FPGA check, launching...
echo.

"%PYTHON_EXE%" main.py --skip-fpga-check %*

echo.
echo [INFO] Program exited (code: %errorlevel%)
pause