"""Qwen 系列模型基础策略"""

from typing import List, Dict
from strategies.base import LLMStrategy


class QwenBaseStrategy(LLMStrategy):
    """
    Qwen 系列基础策略。
    
    适用于 Qwen 1.x 及通用 Qwen 模型。
    """

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """将消息列表转换为 Qwen 模型输入的 prompt"""
        IM_START = "<" + "|im_start|" + ">"
        IM_END = "<" + "|im_end|" + ">"
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            parts.append(f"{IM_START}{role}\n{content}{IM_END}")
        parts.append(f"{IM_START}assistant\n")
        return "\n".join(parts)

    def get_default_stop_words(self) -> List[str]:
        IM_START = "<" + "|im_start|" + ">"
        IM_END = "<" + "|im_end|" + ">"
        EOF = "<" + "|endoftext|" + ">"
        return [IM_END, IM_START, EOF]
