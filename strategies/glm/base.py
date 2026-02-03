"""GLM 系列模型基础策略"""

from typing import List, Dict
from strategies.base import LLMStrategy


class GLMBaseStrategy(LLMStrategy):
    """GLM 系列基础策略，支持 observation 角色。"""

    # 定义特殊标记，避免直接在字符串中使用导致解析问题
    _SYSTEM_TAG = "<" + "|system|" + ">"
    _USER_TAG = "<" + "|user|" + ">"
    _ASSISTANT_TAG = "<" + "|assistant|" + ">"
    _OBSERVATION_TAG = "<" + "|observation|" + ">"

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """将消息列表转换为 GLM 模型输入的 prompt"""
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            if role == "system":
                parts.append(f"{self._SYSTEM_TAG}\n{content}")
            elif role == "user":
                parts.append(f"{self._USER_TAG}\n{content}")
            elif role == "assistant":
                parts.append(f"{self._ASSISTANT_TAG}\n{content}")
            elif role in ("observation", "tool"):
                parts.append(f"{self._OBSERVATION_TAG}\n{content}")
            else:
                parts.append(f"{self._USER_TAG}\n{content}")
        parts.append(f"{self._ASSISTANT_TAG}\n")
        return "[gMASK]<sop>" + "\n".join(parts)

    def get_default_stop_words(self) -> List[str]:
        return [self._USER_TAG, self._ASSISTANT_TAG, self._SYSTEM_TAG, self._OBSERVATION_TAG]
