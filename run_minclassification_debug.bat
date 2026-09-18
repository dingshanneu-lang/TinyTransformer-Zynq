@echo off
chcp 65001 >nul
title TinyTransformer-Zynq 猫狗分类启动器 - 调试版

echo =============================================
echo  TinyTransformer-Zynq 猫狗分类 - 调试启动
echo =============================================
echo.

REM 获取脚本所在目录
set "SCRIPT_DIR=%~dp0"
echo [调试] 脚本目录: %SCRIPT_DIR%
cd /d "%SCRIPT_DIR%"
echo [调试] 当前目录: %CD%
echo.

REM 使用本地虚拟环境 python
set "VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON_EXE=%VENV_PYTHON%"
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [错误] 找不到 python.exe
        echo 请确保 .venv 存在或 python 在 PATH 中
        echo.
        pause
        exit /b 1
    )
)

echo [调试] Python 路径: %PYTHON_EXE%
echo [调试] Python 版本:
"%PYTHON_EXE%" --version
echo.

REM 检查主程序
if not exist "main.py" (
    echo [错误] 未找到 main.py
    echo 当前目录文件列表:
    dir /b
    echo.
    pause
    exit /b 1
)

echo [启动] 正在启动主程序 (--skip-fpga-check --gui-only)...
echo.

REM 运行主程序，显示所有输出
"%PYTHON_EXE%" main.py --skip-fpga-check --gui-only

echo.
echo [调试] 程序退出码: %errorlevel%
echo [调试] 如果看到此行，说明批处理正常结束
pause