"""
想定生成常量定义

Agent模式下仅保留必要的常量。
"""
from enum import Enum


class DialogPhase(str, Enum):
    """对话阶段（Agent模式下主要用于状态标记）"""
    PROCESSING = "processing"   # 处理中
    COMPLETED = "completed"     # 完成


class Classify(str, Enum):
    """阵营"""
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    WHITE = "white"


class ValidationConfig:
    """校验配置"""
    MAX_FIX_ATTEMPTS = 5


class MemoryConfig:
    """记忆配置"""
    TTL_SECONDS = 1800  # 30分钟
    CLEANUP_INTERVAL = 300  # 5分钟
