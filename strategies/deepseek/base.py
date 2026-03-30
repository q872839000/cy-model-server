"""Deepseek 系列模型基础策略"""

from typing import List, Dict
from strategies.base import LLMStrategy, ToolPromptMode


class DeepseekBaseStrategy(LLMStrategy):
    """
    Deepseek 系列基础策略。
    
    使用标准的 ChatML 格式。
    """

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """将消息列表转换为 Deepseek 模型输入的 prompt"""
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            parts.append(f"<{role}>\n{content}</{role}>")
        parts.append("<assistant>\n")
        return "\n".join(parts)

    def get_default_stop_words(self) -> List[str]:
        return ["</assistant>", "<user>", "</user>"]

    def supports_native_tools(self) -> bool:
        return False

    def tool_prompt_mode(self) -> ToolPromptMode:
        return ToolPromptMode.INJECTED
