"""Phi-4-reasoning 系列模型策略（14B，支持深度思考）

Phi-4-reasoning / Phi-4-reasoning-plus 基于 Phi-4 (14B) 架构，
使用相同的 im_start / im_sep / im_end 对话模板格式，
并内置推理系统提示，支持 think / /think 深度思考标记。

适用模型：
- microsoft/Phi-4-reasoning (14B)
- microsoft/Phi-4-reasoning-plus (14B)

参考：https://huggingface.co/microsoft/Phi-4-reasoning-plus
"""

from typing import List, Dict
from strategies.phi4.base import Phi4BaseStrategy
from strategies.protocol import StrategyInput, PromptOutput


# Phi-4-reasoning 官方推荐系统提示
_THINK_START = "<" + "think" + ">"
_THINK_END = "<" + "/think" + ">"

_REASONING_SYSTEM_PROMPT = (
    "You are Phi, a language model trained by Microsoft to help users. "
    "Your role as an assistant involves thoroughly exploring questions through a systematic "
    "thinking process before providing the final precise and accurate solutions. This requires "
    "engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, "
    "reflection, backtracking, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thinking and Answer. "
    "In the Thinking section, detail your reasoning process using the specified format: "
    + _THINK_START + "\n{put your thinking here}\n" + _THINK_END + "\n "
    "Finally, provide your definitive response in the Answer section, formatted as: "
    "\n**Answer**\n{put your final answer here}"
)


class Phi4ReasoningStrategy(Phi4BaseStrategy):
    """
    Phi-4-reasoning (14B) 策略，支持深度思考模式。

    继承 Phi4BaseStrategy 的对话模板格式，增加：
    1. 内置推理系统提示（官方推荐）
    2. enable_thinking=True 时通过 think 标记引导思考输出
    3. enable_thinking=False 时跳过思考引导
    """

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """应用对话模板，可选启用深度思考模式

        如果消息列表中没有 system 消息，自动注入官方推荐的推理系统提示。
        """
        enable_thinking = kwargs.get("enable_thinking", True)

        # 检查是否已有 system 消息
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": _REASONING_SYSTEM_PROMPT}] + list(messages)

        # 构建基础 prompt
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            parts.append(
                f"{self._IM_START}{role}{self._IM_SEP}{content}{self._IM_END}"
            )

        if enable_thinking:
            parts.append(f"{self._IM_START}assistant{self._IM_SEP}{_THINK_START}\n")
        else:
            parts.append(f"{self._IM_START}assistant{self._IM_SEP}")

        return "".join(parts)

    def build_prompt(self, input: StrategyInput) -> PromptOutput:
        """构建 Prompt，支持深度思考模式

        当 enable_thinking=True 时，设置 thinking_prefix，
        由基类 execute() 统一处理前缀输出。
        """
        prompt = self.apply_chat_template(
            input.messages,
            enable_thinking=input.enable_thinking,
            **input.extra
        )

        thinking_prefix = _THINK_START if input.enable_thinking else None

        return PromptOutput(
            prompt=prompt,
            thinking_prefix=thinking_prefix,
            stop_words=input.stop or self.get_default_stop_words(),
        )
