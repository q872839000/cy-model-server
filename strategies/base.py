from abc import ABC, abstractmethod
from typing import List, Dict, Iterator, Union


class LLMStrategy(ABC):
	"""LLM 策略接口，负责对话模板与生成调用。"""

	@abstractmethod
	def apply_chat_template(self, messages: List[Dict]) -> str:
		"""将消息列表转换为模型输入的prompt"""
		...

	def generate(self, engine, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
		"""
		生成响应（统一接口，符合 OpenAI 范式）
		
		Args:
			engine: LLM引擎实例
			messages: 消息列表
			stream: 是否流式输出
			**kwargs: 其他生成参数
		
		Returns:
			stream=False: 返回完整文本
			stream=True: 返回文本片段迭代器
		"""
		prompt = self.apply_chat_template(messages)
		return engine.generate(prompt, stream=stream, **kwargs)


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
