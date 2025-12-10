@echo off
REM Windows 打包脚本
REM 使用方法: scripts\build_nuitka.bat [--onefile]

cd /d "%~dp0\.."
echo 当前目录: %CD%

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请确保 Python 已安装并添加到 PATH
    pause
    exit /b 1
)

REM 执行打包脚本
python scripts\build_nuitka.py %*

pause
