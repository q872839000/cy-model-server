from typing import List, Dict
from strategies.base import LLMStrategy


class GLMChatStrategy(LLMStrategy):
	"""GLM 系列聊天策略。注意GLM使用特殊模板，这里以简化模板示例。"""

	def apply_chat_template(self, messages: List[Dict]) -> str:
		# 简化示例：GLM 常见格式包含 <|system|>, <|user|>, <|assistant|>
		parts = []
		for m in messages:
			role = m.get("role")
			content = m.get("content", "")
			if role == "system":
				parts.append(f"<|system|>{content}")
			elif role == "user":
				parts.append(f"<|user|>{content}")
			elif role == "assistant":
				parts.append(f"<|assistant|>{content}")
		parts.append("<|assistant|>")
		return "\n".join(parts)

	def generate(self, engine, messages: List[Dict], **kwargs) -> str:
		prompt = self.apply_chat_template(messages)
		return engine.generate(prompt, **kwargs)
