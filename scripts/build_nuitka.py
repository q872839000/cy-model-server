#!/usr/bin/env python
"""
Nuitka 打包脚本 - 将 cy-model-server 编译为独立可执行文件
使用方法：python scripts/build_nuitka.py [--onefile]
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
OUTPUT_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def check_nuitka():
    """检查 Nuitka 是否已安装"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nuitka", "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print(f"✓ Nuitka 已安装: {version}")
            return True
        raise ImportError()
    except (ImportError, FileNotFoundError):
        print("✗ Nuitka 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nuitka", "ordered-set", "zstandard"])
        return True


def find_vcvarsall():
    """查找 vcvarsall.bat 路径"""
    import glob
    
    # 自定义路径 + 常见安装路径
    vs_paths = [
        # 自定义安装路径
        r"D:\software\C\VC\Auxiliary\Build\vcvarsall.bat",
        # 默认安装路径
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2019\*\VC\Auxiliary\Build\vcvarsall.bat",
    ]
    
    for pattern in vs_paths:
        matches = sorted(glob.glob(pattern), reverse=True)
        for path in matches:
            if os.path.exists(path):
                return path
        # 如果不是 glob 模式，直接检查
        if os.path.exists(pattern):
            return pattern
    return None


def setup_msvc_env():
    """通过 vcvarsall.bat 设置 MSVC 环境变量"""
    vcvarsall = find_vcvarsall()
    if not vcvarsall:
        return False
    
    print(f"✓ 找到 vcvarsall.bat: {vcvarsall}")
    print("  正在配置 MSVC 编译环境...")
    
    # 调用 vcvarsall.bat 并捕获环境变量
    cmd = f'"{vcvarsall}" x64 && set'
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            print(f"  警告: vcvarsall.bat 执行失败")
            return False
        
        # 解析并设置环境变量
        for line in result.stdout.splitlines():
            if '=' in line:
                key, _, value = line.partition('=')
                os.environ[key] = value
        
        print("  ✓ MSVC 环境配置完成")
        return True
    except Exception as e:
        print(f"  警告: 配置 MSVC 环境失败: {e}")
        return False


def find_msvc():
    """查找 MSVC 编译器路径"""
    import glob
    
    # 自定义路径 + 常见安装路径
    vs_paths = [
        # 自定义安装路径
        r"D:\software\C\VC\Tools\MSVC\*\bin\Hostx64\x64",
        # 默认安装路径
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
        r"C:\Program Files\Microsoft Visual Studio\2019\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
    ]
    
    for pattern in vs_paths:
        matches = sorted(glob.glob(pattern), reverse=True)  # 优先使用最新版本
        for path in matches:
            cl_path = os.path.join(path, "cl.exe")
            if os.path.exists(cl_path):
                return path
    return None


def check_compiler():
    """检查 C 编译器"""
    if sys.platform == "win32":
        # 先尝试通过 vcvarsall.bat 配置完整的 MSVC 环境
        if setup_msvc_env():
            # 验证 cl.exe 是否可用
            result = subprocess.run(["where", "cl.exe"], capture_output=True)
            if result.returncode == 0:
                return "msvc"
        
        # 检查 PATH 中是否有 cl.exe
        result = subprocess.run(["where", "cl.exe"], capture_output=True)
        if result.returncode == 0:
            print("✓ 检测到 MSVC 编译器 (PATH)")
            return "msvc"
        
        # 自动查找 MSVC 安装路径（仅添加 cl.exe 路径，可能缺少 SDK）
        msvc_path = find_msvc()
        if msvc_path:
            print(f"✓ 检测到 MSVC 编译器: {msvc_path}")
            print("  警告: 未能通过 vcvarsall.bat 配置环境，可能缺少 Windows SDK")
            os.environ["PATH"] = msvc_path + os.pathsep + os.environ.get("PATH", "")
            return "msvc"
        
        # 检查 MinGW
        result = subprocess.run(["where", "gcc.exe"], capture_output=True)
        if result.returncode == 0:
            print("✓ 检测到 MinGW 编译器")
            return "mingw"
        
        print("✗ 未检测到 C 编译器，请安装 Visual Studio Build Tools 或 MinGW")
        print("  下载地址: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        return None
    else:
        # Linux/Mac
        result = subprocess.run(["which", "gcc"], capture_output=True)
        if result.returncode == 0:
            print("✓ 检测到 GCC 编译器")
            return "gcc"
        print("✗ 未检测到 GCC，请安装: sudo apt install build-essential")
        return None


def build(onefile: bool = False):
    """执行 Nuitka 编译"""
    
    print("\n" + "=" * 60)
    print("CY-Model-Server Nuitka 打包")
    print("=" * 60)
    
    # 检查依赖
    if not check_nuitka():
        return False
    
    if not check_compiler():
        return False
    
    # 清理旧构建
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    
    print(f"\n项目路径: {PROJECT_ROOT}")
    print(f"入口文件: {MAIN_SCRIPT}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"打包模式: {'单文件' if onefile else '目录'}")
    
    # Nuitka 命令参数
    cmd = [
        sys.executable, "-m", "nuitka",
        
        # 基础选项
        "--standalone",                      # 独立部署，包含所有依赖
        f"--output-dir={OUTPUT_DIR}",        # 输出目录
        
        # 包含所有项目模块
        "--include-package=api",
        "--include-package=core",
        "--include-package=engines",
        "--include-package=models",
        "--include-package=observability",
        "--include-package=services",
        "--include-package=strategies",
        "--include-package=workers",
        
        # 包含配置文件
        f"--include-data-dir={PROJECT_ROOT / 'configs'}=configs",
        
        # 包含必要的第三方库（显式指定以确保完整）
        "--include-package=transformers",
        "--include-package=torch",
        "--include-package=accelerate",
        "--include-package=tokenizers",
        "--include-package=safetensors",
        "--include-package=sentence_transformers",
        "--include-package=sentencepiece",
        "--include-package=fastapi",
        "--include-package=uvicorn",
        "--include-package=pydantic",
        "--include-package=pydantic_settings",
        "--include-package=yaml",
        "--include-package=loguru",
        "--include-package=orjson",
        "--include-package=httpx",
        "--include-package=prometheus_client",
        "--include-package=prometheus_fastapi_instrumentator",
        "--include-package=numpy",
        "--include-package=huggingface_hub",
        "--include-package=filelock",
        "--include-package=tqdm",
        "--include-package=regex",
        
        # 包含数据文件（transformers 需要）
        "--include-package-data=transformers",
        "--include-package-data=tokenizers",
        "--include-package-data=sentencepiece",
        
        # 注意：numpy 和 torch 插件在 Nuitka 2.x 已弃用，无需显式启用
        
        # 优化选项
        "--assume-yes-for-downloads",        # 自动下载依赖
        "--remove-output",                   # 覆盖旧输出
        
        # 禁用控制台（可选，生产环境可启用）
        # "--disable-console",
        
        # 编译信息
        "--company-name=CY",
        "--product-name=CY-Model-Server",
        "--product-version=1.0.0",
        
        # 入口脚本
        str(MAIN_SCRIPT),
    ]
    
    # 单文件模式（体积更大，启动更慢，但更便携）
    if onefile:
        cmd.insert(3, "--onefile")
    
    print("\n开始编译（这可能需要 10-30 分钟）...\n")
    print("执行命令:")
    print(" ".join(cmd[:10]) + " ...")
    print()
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="")
        
        process.wait()
        
        if process.returncode == 0:
            print("\n" + "=" * 60)
            print("✓ 编译成功！")
            print("=" * 60)
            
            # 输出结果位置
            if onefile:
                exe_name = "main.exe" if sys.platform == "win32" else "main.bin"
                print(f"\n可执行文件: {OUTPUT_DIR / exe_name}")
            else:
                dist_folder = OUTPUT_DIR / "main.dist"
                exe_name = "main.exe" if sys.platform == "win32" else "main.bin"
                print(f"\n部署目录: {dist_folder}")
                print(f"可执行文件: {dist_folder / exe_name}")
            
            print("\n使用方法:")
            print("  1. 将整个 dist 目录复制到目标机器")
            print("  2. 运行可执行文件即可启动服务")
            print("  3. 配置文件位于 configs/config.yaml")
            
            return True
        else:
            print(f"\n✗ 编译失败，返回码: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n✗ 编译出错: {e}")
        return False


def main():
    onefile = "--onefile" in sys.argv or "-o" in sys.argv
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\n选项:")
        print("  --onefile, -o    打包为单个可执行文件（体积更大，启动更慢）")
        print("  --help, -h       显示帮助信息")
        return
    
    success = build(onefile=onefile)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
