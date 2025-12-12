"""Deepseek R1 系列模型策略（支持深度思考模式）"""

from typing import List, Dict, Iterator, Union
from strategies.deepseek.base import DeepseekBaseStrategy


class DeepseekR1Strategy(DeepseekBaseStrategy):
	"""
	Deepseek R1 系列策略。
	
	Deepseek R1 支持深度思考模式，模型会在回答前进行推理。
	思考内容会被包含在 <think>...</think> 标签中。
	"""

	def apply_chat_template(self, messages: List[Dict], enable_thinking: bool = True) -> str:
		"""
		将消息列表转换为 Deepseek R1 模型输入的 prompt。
		
		Args:
			messages: 消息列表
			enable_thinking: 是否启用深度思考模式，默认 True
		"""
		parts = ["<deepseek>"]
		for m in messages:
			role = m.get("role")
			content = m.get("content", "")
			parts.append(f"{role.upper()}: {content}")
		
		if enable_thinking:
			parts.append("ASSISTANT: <think>")
		else:
			parts.append("ASSISTANT:")
		
		return "\n".join(parts)

	def generate(self, engine, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
		"""
		生成响应（支持 enable_thinking 参数）
		"""
		enable_thinking = kwargs.pop("enable_thinking", True)
		prompt = self.apply_chat_template(messages, enable_thinking=enable_thinking)
		return engine.generate(prompt, stream=stream, **kwargs)
