"""
想定记忆管理模块

提供轻量级的中间状态管理，用于跨请求保持想定生成的上下文。
"""

from .memory_manager import MemoryManager, ScenarioMemory, MEMORY_MANAGER

__all__ = [
    "MemoryManager",
    "ScenarioMemory",
    "MEMORY_MANAGER",
]
