"""上下文管理模块

提供 LLM 对话的上下文窗口管理能力：
- 消息截断（滑动窗口）
- Token 预算计算
"""

from core.context.trimmer import trim_messages, TokenCounter

__all__ = ["trim_messages", "TokenCounter"]
