from typing import List, Dict
from strategies.base import LLMStrategy


class GLMBaseStrategy(LLMStrategy):
	"""GLM 系列基础策略，支持 observation 角色。"""

	def apply_chat_template(self, messages: List[Dict]) -> str:
		"""将消息列表转换为 GLM 模型输入的 prompt"""
		parts = []
		for m in messages:
			role = m.get("role")
			content = m.get("content", "")
			if role == "system":
				parts.append(f"<|system|>\n{content}")
			elif role == "user":
				parts.append(f"<|user|>\n{content}")
			elif role == "assistant":
				parts.append(f"<|assistant|>\n{content}")
			elif role == "observation":
				parts.append(f"<|observation|>\n{content}")
		parts.append("<|assistant|>")
		return "[gMASK]<sop>" + "\n".join(parts)
