"""GLM4-0414 系列模型策略（支持深度思考）"""

from typing import List, Dict
from strategies.glm.base import GLMBaseStrategy
from strategies.protocol import StrategyInput, PromptOutput


class GLM4_0414Strategy(GLMBaseStrategy):
    """
    GLM4-0414 系列策略。
    
    通过在 prompt 中注入 <think> 标签来引导模型输出思考过程。
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
        
        当 enable_thinking=True 时，设置 thinking_prefix 为 "<think>"，
        由基类 execute() 统一处理前缀输出。
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
