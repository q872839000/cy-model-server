from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator


class LLMEngine(ABC):
	"""LLM 引擎统一接口。负责具体推理实现（transformers/vLLM）。"""

	def generate(self, prompt: str, stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
		"""
		生成文本（统一接口，符合 OpenAI 范式）
		
		Args:
			prompt: 输入提示词
			stream: 是否流式输出
			**kwargs: 其他生成参数（max_tokens, temperature, top_p 等）
		
		Returns:
			stream=False: 返回完整文本字符串
			stream=True: 返回文本片段的迭代器
		"""
		if stream:
			return self._generate_stream(prompt, **kwargs)
		return self._generate(prompt, **kwargs)
	
	@abstractmethod
	def _generate(self, prompt: str, **kwargs) -> str:
		"""内部非流式生成实现"""
		...
	
	def _generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
		"""内部流式生成实现（默认回退到非流式）"""
		yield self._generate(prompt, **kwargs)


class EmbeddingEngine(ABC):
	"""Embedding 引擎统一接口。"""

	@abstractmethod
	def embed(self, texts: List[str]) -> List[List[float]]:
		...


class RerankerEngine(ABC):
	"""Reranker 引擎统一接口。"""

	@abstractmethod
	def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[float]:
		...
