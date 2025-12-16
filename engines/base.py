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
		stop = kwargs.pop("stop", None)
		if stream:
			it = self._generate_stream(prompt, **kwargs)
			if stop:
				return self._apply_stop_stream(it, stop)
			return it
		text = self._generate(prompt, **kwargs)
		if stop:
			return self._apply_stop_text(text, stop)
		return text

	@staticmethod
	def _normalize_stop(stop: Any) -> List[str]:
		if stop is None:
			return []
		if isinstance(stop, str):
			return [stop] if stop else []
		try:
			stops = [s for s in stop if isinstance(s, str) and s]
		except TypeError:
			return []
		return stops

	@classmethod
	def _apply_stop_text(cls, text: str, stop: Any) -> str:
		stops = cls._normalize_stop(stop)
		if not stops:
			return text
		cut = None
		for s in stops:
			idx = text.find(s)
			if idx != -1 and (cut is None or idx < cut):
				cut = idx
		return text if cut is None else text[:cut]

	@classmethod
	def _apply_stop_stream(cls, it: Iterator[str], stop: Any) -> Iterator[str]:
		stops = cls._normalize_stop(stop)
		if not stops:
			yield from it
			return
		max_len = max(len(s) for s in stops)
		buffer = ""
		for chunk in it:
			if not chunk:
				continue
			combined = buffer + chunk
			cut = None
			for s in stops:
				idx = combined.find(s)
				if idx != -1 and (cut is None or idx < cut):
					cut = idx
			if cut is None:
				keep = max_len - 1
				if keep <= 0:
					yield combined
					buffer = ""
					continue
				if len(combined) > keep:
					emit_len = len(combined) - keep
					yield combined[:emit_len]
					buffer = combined[emit_len:]
				else:
					buffer = combined
				continue
			if cut > 0:
				yield combined[:cut]
			for _ in it:
				pass
			return
		if buffer:
			yield buffer
	
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
