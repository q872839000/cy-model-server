from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Iterator, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from engines.cancel import CancelToken


@dataclass(frozen=True)
class EngineCapabilities:
    """引擎能力声明，供调度层决策

    Attributes:
        supports_concurrent_requests: 是否支持同引擎实例内多请求并发（如 vLLM 连续批处理）
        supports_cancel: 是否支持协作式取消（中途中止推理释放 GPU）
        preferred_max_concurrency: 建议的每副本最大并发数（0 = 由配置决定）
    """
    supports_concurrent_requests: bool = False
    supports_cancel: bool = False
    preferred_max_concurrency: int = 0


class _LoadLockMixin:
	"""为引擎子类自动注入 _load_lock 的 Mixin。

	通过 __init_subclass__ 包装子类的 __init__，确保锁在子类构造前就存在。
	LLMEngine 额外提供 _inference_lock（GPU 推理互斥）。
	"""

	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		original_init = cls.__init__

		def _wrapped_init(self, *args, **kw):
			if not hasattr(self, '_load_lock'):
				self._load_lock = threading.Lock()
			# LLMEngine 子类额外注入 _inference_lock
			if isinstance(self, LLMEngine) and not hasattr(self, '_inference_lock'):
				self._inference_lock = threading.Lock()
			original_init(self, *args, **kw)

		cls.__init__ = _wrapped_init


class LLMEngine(_LoadLockMixin, ABC):
	"""LLM 引擎统一接口。负责具体推理实现（transformers/vLLM）。

	并发安全机制（所有子类自动继承）：
	- _load_lock: 保护模型加载过程，防止并发 _ensure_loaded() 导致双重加载 / OOM
	- _inference_lock: 保护 GPU 推理过程，防止多线程同时调用 model.generate() 导致 CUDA 错误

	子类应覆盖 capabilities() 以声明自身并发特性。
	"""

	@classmethod
	def capabilities(cls) -> EngineCapabilities:
		"""声明引擎并发能力，供调度层决策。子类应覆盖此方法。"""
		return EngineCapabilities()

	def get_context_window(self) -> Optional[int]:
		"""获取模型的上下文窗口大小（token 数）

		自动从已加载模型的 config 中探测，按优先级尝试以下字段：
		- max_position_embeddings（最通用，Qwen/LLaMA/Mistral/GPT 等均有）
		- seq_length（GLM 系列）
		- n_positions（GPT-2 系列）
		- max_sequence_length（部分旧模型）

		子类可覆盖以实现引擎特定的探测逻辑（如 vLLM 的 max_model_len）。

		Returns:
			上下文窗口 token 数，无法探测时返回 None
		"""
		model = getattr(self, "_model", None)
		if model is None:
			# vLLM fallback 场景
			fallback = getattr(self, "_fallback", None)
			if fallback is not None:
				return fallback.get_context_window()
			return None
		config = getattr(model, "config", None)
		if config is None:
			return None
		# 按优先级尝试多个常见字段名
		for attr in ("max_position_embeddings", "seq_length",
					 "n_positions", "max_sequence_length"):
			val = getattr(config, attr, None)
			if val is not None and isinstance(val, int) and val > 0:
				return val
		return None

	def apply_chat_template(
		self,
		messages: List[Dict[str, Any]],
		tools: Optional[List[Dict[str, Any]]] = None,
		**kwargs,
	) -> Optional[str]:
		"""使用模型原生 tokenizer 构建 prompt（支持 tools/function calling）

		子类应在 tokenizer 支持 apply_chat_template 时覆盖此方法。
		返回 None 表示不支持，调用方应回退到 Strategy 层的手动模板。
		始终返回字符串（tokenize=False），由 _build_inputs 统一编码。

		Args:
			messages: OpenAI 格式的对话消息列表（含 tool_calls/tool_call_id 等字段）
			tools: OpenAI 格式的工具定义列表
			**kwargs: 其他参数（如 enable_thinking）

		Returns:
			prompt 字符串，或 None
		"""
		return None

	@staticmethod
	def _build_template_kwargs(
		tools: Optional[List[Dict[str, Any]]] = None,
		tokenize: bool = False,
		**kwargs,
	) -> Dict[str, Any]:
		"""构建 tokenizer.apply_chat_template 的通用参数字典

		所有引擎子类共享此方法，避免重复的 kwargs 构建逻辑。
		仅包含 tokenizer 实际接受的参数（tokenize, add_generation_prompt, tools, enable_thinking）。
		tool_choice 等不被 HF tokenizer 原生支持的参数会被过滤掉。

		Args:
			tools: OpenAI 格式的工具定义列表
			tokenize: 是否返回 token ID（True）或文本（False）
			**kwargs: 其他参数（enable_thinking, tool_choice 等）

		Returns:
			可直接传给 tokenizer.apply_chat_template 的参数字典
		"""
		template_kwargs: Dict[str, Any] = {
			"tokenize": tokenize,
			"add_generation_prompt": True,
		}
		if tools:
			template_kwargs["tools"] = tools
		# 部分 tokenizer 支持 enable_thinking（如 Qwen3）
		enable_thinking = kwargs.get("enable_thinking")
		if enable_thinking is not None:
			template_kwargs["enable_thinking"] = enable_thinking
		return template_kwargs

	def generate(
		self, prompt: str, stream: bool = False, **kwargs,
	) -> Union[str, Iterator[str]]:
		"""
		生成文本（统一接口，符合 OpenAI 范式）

		Args:
			prompt: 输入提示词字符串
			stream: 是否流式输出
			**kwargs: 其他生成参数（max_tokens, temperature, top_p, cancel_token 等）

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


class EmbeddingEngine(_LoadLockMixin, ABC):
	"""Embedding 引擎统一接口。"""

	@abstractmethod
	def embed(self, texts: List[str]) -> List[List[float]]:
		...


class RerankerEngine(_LoadLockMixin, ABC):
	"""Reranker 引擎统一接口。"""

	@abstractmethod
	def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[float]:
		...
