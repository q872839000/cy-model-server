"""Phi-4 系列模型基础策略（14B）

Phi-4 (14B) 使用 ChatML 变体格式，采用 im_start / im_sep / im_end 标记。
与 Qwen 系列的 im_start + role + 换行 + content + im_end 格式不同，
Phi-4 使用 im_start + role + im_sep + content + im_end 格式。

适用模型：
- microsoft/phi-4（14B 基础指令模型）

参考：https://huggingface.co/microsoft/phi-4
"""

from typing import List, Dict
from strategies.base import LLMStrategy


class Phi4BaseStrategy(LLMStrategy):
    """
    Phi-4 (14B) 基础策略。

    对话模板格式：
        im_start + system + im_sep + 系统提示 + im_end
        im_start + user + im_sep + 用户消息 + im_end
        im_start + assistant + im_sep

    推荐参数：temperature=0.8, top_p=0.95
    """

    # 特殊标记定义（通过拼接避免被 tokenizer 解析）
    _IM_START = "<" + "|im_start|" + ">"
    _IM_SEP = "<" + "|im_sep|" + ">"
    _IM_END = "<" + "|im_end|" + ">"

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """将消息列表转换为 Phi-4 模型输入的 prompt"""
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            parts.append(
                f"{self._IM_START}{role}{self._IM_SEP}{content}{self._IM_END}"
            )
        parts.append(f"{self._IM_START}assistant{self._IM_SEP}")
        return "".join(parts)

    def get_default_stop_words(self) -> List[str]:
        return [self._IM_END, self._IM_START]

    def supports_native_tools(self) -> bool:
        """Phi-4 (14B) 基础模型未经过 function calling 训练，
        tokenizer 的 Jinja 模板不支持全局 tools 参数。
        返回 False 强制走 system prompt 注入路径（尽力而为）。
        """
        return False
