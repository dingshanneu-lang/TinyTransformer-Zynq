@echo off
chcp 65001 >nul
title TinyTransformer-Zynq Cat Dog Classification Launcher

echo =============================================
echo  TinyTransformer-Zynq Cat Dog Classification
echo =============================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo [INFO] Working directory: %SCRIPT_DIR%
echo.

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

REM Check dependencies (simplified, one per line)
echo [CHECK] Dependencies...
"%PYTHON_EXE%" -c "import torch; print('  torch:', torch.__version__)" 2>&1
if errorlevel 1 (echo [ERROR] torch not installed & pause & exit /b 1)

"%PYTHON_EXE%" -c "import torchvision; print('  torchvision:', torchvision.__version__)" 2>&1
if errorlevel 1 (echo [ERROR] torchvision not installed & pause & exit /b 1)

"%PYTHON_EXE%" -c "from PySide6.QtWidgets import QApplication; print('  PySide6: OK')" 2>&1
if errorlevel 1 (
    "%PYTHON_EXE%" -c "from PyQt5.QtWidgets import QApplication; print('  PyQt5: OK')" 2>&1
    if errorlevel 1 (echo [ERROR] PySide6/PyQt5 not installed & pause & exit /b 1)
)

"%PYTHON_EXE%" -c "import numpy; print('  numpy:', numpy.__version__)" 2>&1
"%PYTHON_EXE%" -c "from PIL import Image; print('  Pillow: OK')" 2>&1
echo.

REM Check main.py
if not exist "main.py" (
    echo [ERROR] main.py not found
    pause
    exit /b 1
)

echo [START] Launching main program...
echo.

REM Run main program (pass all arguments)
"%PYTHON_EXE%" main.py %*

echo.
echo [INFO] Program exited (code: %errorlevel%)
pause