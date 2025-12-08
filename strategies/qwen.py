from typing import List, Dict
from strategies.base import LLMStrategy


class QwenChatStrategy(LLMStrategy):
	"""Qwen 系列聊天策略。"""

	def apply_chat_template(self, messages: List[Dict]) -> str:
		# 简化模板：Qwen 常见使用 <|im_start|>role\ncontent<|im_end|>
		parts = []
		for m in messages:
			role = m.get("role")
			content = m.get("content", "")
			parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
		parts.append("<|im_start|>assistant\n")
		return "\n".join(parts)
