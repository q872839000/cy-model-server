from abc import ABC, abstractmethod
from typing import List, Dict, Iterator


class LLMStrategy(ABC):
	"""LLM 策略接口，负责对话模板与生成调用。"""

	@abstractmethod
	def apply_chat_template(self, messages: List[Dict]) -> str:
		"""将消息列表转换为模型输入的prompt"""
		...

	@abstractmethod
	def generate(self, engine, messages: List[Dict], **kwargs) -> str:
		"""同步生成完整响应"""
		...
	
	def generate_stream(self, engine, messages: List[Dict], **kwargs) -> Iterator[str]:
		"""流式生成响应（默认实现）"""
		prompt = self.apply_chat_template(messages)
		yield from engine.generate_stream(prompt, **kwargs)


class GenericChatStrategy(LLMStrategy):
	"""通用聊天策略：简单拼接user与system上下文。适用于大多数兼容模型。"""

	def apply_chat_template(self, messages: List[Dict]) -> str:
		parts = []
		for m in messages:
			role = m.get("role", "user")
			content = m.get("content", "")
			parts.append(f"{role}: {content}")
		parts.append("assistant:")
		return "\n".join(parts)

	def generate(self, engine, messages: List[Dict], **kwargs) -> str:
		prompt = self.apply_chat_template(messages)
		return engine.generate(prompt, **kwargs)
