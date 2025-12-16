#!/bin/bash
# Linux 打包脚本
# 使用方法: bash scripts/build_nuitka.sh [--onefile]

set -e

cd "$(dirname "$0")/.."
echo "当前目录: $(pwd)"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 检查 GCC
if ! command -v gcc &> /dev/null; then
    echo "错误: 未找到 GCC，请安装: sudo apt install build-essential"
    exit 1
fi

if ! command -v patchelf &> /dev/null; then
    echo "警告: 未找到 patchelf，Linux 下 Nuitka standalone 可能无法正确修补 ELF 依赖。"
    echo "  Debian/Ubuntu: sudo apt install patchelf"
    echo "  CentOS/RHEL:   sudo yum install patchelf   (或 sudo dnf install patchelf)"
    echo "  Arch:          sudo pacman -S patchelf"
fi

# 执行打包脚本
python3 scripts/build_nuitka.py "$@"
