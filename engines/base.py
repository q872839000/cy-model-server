from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator


class LLMEngine(ABC):
	"""LLM 引擎统一接口。负责具体推理实现（transformers/vLLM）。"""

	def apply_chat_template(
		self,
		messages: List[Dict[str, Any]],
		tools: Optional[List[Dict[str, Any]]] = None,
		**kwargs,
	) -> Optional[str]:
		"""使用模型原生 tokenizer 构建 prompt（支持 tools/function calling）

		子类应在 tokenizer 支持 apply_chat_template 时覆盖此方法。
		返回 None 表示不支持，调用方应回退到 Strategy 层的手动模板。

		Args:
			messages: OpenAI 格式的对话消息列表（含 tool_calls/tool_call_id 等字段）
			tools: OpenAI 格式的工具定义列表
			**kwargs: 其他参数（如 enable_thinking、tool_choice）

		Returns:
			构建好的 prompt 字符串，或 None（不支持时）
		"""
		return None

	@staticmethod
	def _build_template_kwargs(
		tools: Optional[List[Dict[str, Any]]] = None,
		**kwargs,
	) -> Dict[str, Any]:
		"""构建 tokenizer.apply_chat_template 的通用参数字典

		所有引擎子类共享此方法，避免重复的 kwargs 构建逻辑。
		仅包含 tokenizer 实际接受的参数（tokenize, add_generation_prompt, tools, enable_thinking）。
		tool_choice 等不被 HF tokenizer 原生支持的参数会被过滤掉。

		Args:
			tools: OpenAI 格式的工具定义列表
			**kwargs: 其他参数（enable_thinking, tool_choice 等）

		Returns:
			可直接传给 tokenizer.apply_chat_template 的参数字典
		"""
		template_kwargs: Dict[str, Any] = {
			"tokenize": False,
			"add_generation_prompt": True,
		}
		if tools:
			template_kwargs["tools"] = tools
		# 部分 tokenizer 支持 enable_thinking（如 Qwen3）
		enable_thinking = kwargs.get("enable_thinking")
		if enable_thinking is not None:
			template_kwargs["enable_thinking"] = enable_thinking
		return template_kwargs

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
