from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Iterator


class LLMEngine(ABC):
	"""LLM 引擎统一接口。负责具体推理实现（transformers/vLLM）。"""

	@abstractmethod
	def generate(self, prompt: str, **kwargs) -> str:
		"""同步生成完整文本"""
		...
	
	def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
		"""流式生成文本（默认实现：回退到非流式）"""
		# 默认实现：子类可以覆盖此方法以支持真正的流式
		yield self.generate(prompt, **kwargs)


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
