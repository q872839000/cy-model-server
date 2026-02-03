"""GLM4-Z1 系列模型策略（强制深度思考）"""

from typing import List, Dict
from strategies.glm.base import GLMBaseStrategy
from strategies.protocol import StrategyInput, PromptOutput


class GLM4Z1Strategy(GLMBaseStrategy):
    """
    GLM4-Z1 系列策略（强制深度思考）。
    
    Z1 模型专为深度推理设计，默认启用思考模式。
    """

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """应用对话模板，可选启用深度思考模式。"""
        enable_thinking = kwargs.get("enable_thinking", True)
        prompt = super().apply_chat_template(messages, **kwargs)
        if enable_thinking:
            prompt = prompt.rstrip("\n") + "\n<think>"
        return prompt

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
