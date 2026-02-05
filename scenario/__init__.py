"""
想定生成模块

提供AI辅助的军事仿真想定生成功能。
采用Agent架构，支持意图理解、动态规划、反思调整。

主要组件：
- agent: Agent核心（意图解析、规划、执行、反思）
- constants: 常量定义
- generators: 生成器（名称生成）
- memory: 轻量级记忆管理（中间状态，带TTL）
- tools: Tool定义和执行器（调用外部API）
- services: 服务入口
"""

from .constants import (
    DialogPhase,
    ValidationConfig,
    MemoryConfig,
    Classify,
)

__all__ = [
    "DialogPhase",
    "ValidationConfig",
    "MemoryConfig",
    "Classify",
]
