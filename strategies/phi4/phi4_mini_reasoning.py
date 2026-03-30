"""Phi-4-mini-reasoning 模型策略（3.8B，支持深度思考）

Phi-4-mini-reasoning 基于 Phi-4-mini 架构，
使用 system / user / assistant / end 角色标签格式，
支持 think / /think 深度思考标记。

适用模型：
- microsoft/Phi-4-mini-reasoning (3.8B)
- microsoft/Phi-4-mini-flash-reasoning (3.8B)

参考：https://huggingface.co/microsoft/Phi-4-mini-reasoning
"""

from typing import List, Dict
from strategies.phi4.phi4_mini import Phi4MiniStrategy
from strategies.protocol import StrategyInput, PromptOutput


class Phi4MiniReasoningStrategy(Phi4MiniStrategy):
    """
    Phi-4-mini-reasoning (3.8B) 策略，支持深度思考模式。

    继承 Phi4MiniStrategy 的对话模板格式，增加 think 思考标记支持。
    """

    _THINK_START = "<" + "think" + ">"
    _THINK_END = "<" + "/think" + ">"

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """应用对话模板，可选启用深度思考模式"""
        enable_thinking = kwargs.get("enable_thinking", True)

        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            if role == "system":
                parts.append(f"{self._SYSTEM}{content}{self._END}")
            elif role == "user":
                parts.append(f"{self._USER}{content}{self._END}")
            elif role == "assistant":
                parts.append(f"{self._ASSISTANT}{content}{self._END}")
            elif role in ("tool", "observation"):
                parts.append(f"{self._USER}{content}{self._END}")
            else:
                parts.append(f"{self._USER}{content}{self._END}")

        if enable_thinking:
            parts.append(f"{self._ASSISTANT}{self._THINK_START}\n")
        else:
            parts.append(self._ASSISTANT)

        return "".join(parts)

    def build_prompt(self, input: StrategyInput) -> PromptOutput:
        """构建 Prompt，支持深度思考模式"""
        prompt = self.apply_chat_template(
            input.messages,
            enable_thinking=input.enable_thinking,
            **input.extra
        )

        thinking_prefix = self._THINK_START if input.enable_thinking else None

        return PromptOutput(
            prompt=prompt,
            thinking_prefix=thinking_prefix,
            stop_words=input.stop or self.get_default_stop_words(),
        )
