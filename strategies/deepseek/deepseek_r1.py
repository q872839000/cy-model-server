"""Deepseek-R1 系列模型策略（支持深度思考）"""

from typing import List, Dict
from strategies.deepseek.base import DeepseekBaseStrategy
from strategies.protocol import StrategyInput, PromptOutput


class DeepseekR1Strategy(DeepseekBaseStrategy):
    """
    Deepseek-R1 系列策略。
    
    R1 模型支持深度思考模式，通过 <think> 标签引导模型输出思考过程。
    """

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """应用对话模板，可选启用深度思考模式。"""
        enable_thinking = kwargs.get("enable_thinking", True)
        
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            parts.append(f"<{role}>\n{content}</{role}>")
        
        if enable_thinking:
            parts.append("<assistant>\n<think>")
        else:
            parts.append("<assistant>\n")
        
        return "\n".join(parts)

    def build_prompt(self, input: StrategyInput) -> PromptOutput:
        """
        构建 Prompt，支持深度思考模式。
        """
        prompt = self.apply_chat_template(
            input.messages,
            enable_thinking=input.enable_thinking,
            **input.extra
        )
        
        thinking_prefix = "<think>" if input.enable_thinking else None
        
        return PromptOutput(
            prompt=prompt,
            thinking_prefix=thinking_prefix,
            stop_words=input.stop or self.get_default_stop_words(),
        )
