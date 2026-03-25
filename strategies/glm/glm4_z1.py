"""GLM4-Z1 系列模型策略（强制深度思考）

优先使用 HuggingFace tokenizer 原生 apply_chat_template 构建 prompt，
确保 [gMASK]<sop> 等特殊标记正确映射到模型词表，避免 token ID 越界。
"""

from typing import List, Dict
from dataclasses import replace as dataclass_replace
from strategies.glm.base import GLMBaseStrategy
from strategies.protocol import StrategyInput, PromptOutput


class GLM4Z1Strategy(GLMBaseStrategy):
    """
    GLM4-Z1 系列策略（强制深度思考）。

    Z1 模型专为深度推理设计，始终启用思考模式。
    优先使用 tokenizer 原生模板，仅在不可用时回退到手动模板。
    """

    def prefer_native_template(self) -> bool:
        """GLM-Z1 系列必须使用原生模板，避免特殊标记与词表不匹配"""
        return True

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """应用对话模板，始终启用深度思考模式（回退路径）。"""
        prompt = super().apply_chat_template(messages, **kwargs)
        # Z1 模型强制开启思考
        if not prompt.endswith("\n"):
            prompt += "\n"
        prompt += "<think>"
        return prompt

    def build_prompt(self, input: StrategyInput) -> PromptOutput:
        """
        构建 Prompt，Z1 强制启用深度思考模式。
        """
        # Z1 系列强制开启 thinking
        effective_input = input
        if not input.enable_thinking:
            effective_input = dataclass_replace(input, enable_thinking=True)

        prompt = self.apply_chat_template(
            effective_input.messages,
            enable_thinking=True,
            **effective_input.extra
        )

        return PromptOutput(
            prompt=prompt,
            thinking_prefix="<think>",
            stop_words=effective_input.stop or self.get_default_stop_words(),
        )
