from typing import List, Dict
from strategies.base import LLMStrategy


class DeepseekChatStrategy(LLMStrategy):
	"""Deepseek 系列聊天策略。"""

	def apply_chat_template(self, messages: List[Dict]) -> str:
		# 简化模板：使用ROLE: content形式并加入特殊前缀
		parts = ["<deepseek>"]
		for m in messages:
			role = m.get("role")
			content = m.get("content", "")
			parts.append(f"{role.upper()}: {content}")
		parts.append("ASSISTANT:")
		return "\n".join(parts)
