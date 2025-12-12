"""Qwen 系列模型基础策略"""

from typing import List, Dict, Iterator, Union
from strategies.base import LLMStrategy


class QwenBaseStrategy(LLMStrategy):
    """
    Qwen 系列基础策略。
    
    适用于 Qwen 1.x 及通用 Qwen 模型。
    使用 <|im_start|>role\ncontent<|im_end|> 格式。
    """

    def apply_chat_template(self, messages: List[Dict]) -> str:
        """将消息列表转换为 Qwen 模型输入的 prompt"""
        parts = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
