from typing import Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

from core.config import load_yaml, AppSettings
from engines.base import LLMEngine, EmbeddingEngine, RerankerEngine
from engines.transformers_engine import TransformersLLMEngine, TransformersEmbeddingEngine, TransformersRerankerEngine
from engines.vllm_engine import VLLMLLMEngine


@dataclass
class LLMConfig:
	name: str
	engine: str
	path: str
	chat_strategy: str  # glm|qwen|deepseek|generic
	dtype: Optional[str] = None
	device: Optional[str] = None
	gen_params: Optional[dict] = None
	model_features: Optional[dict] = None  # 模型特性参数，如 enable_thinking


@dataclass
class EmbeddingConfig:
	name: str
	engine: str
	path: str
	device: Optional[str] = None


@dataclass
class RerankerConfig:
	name: str
	engine: str
	path: str
	device: Optional[str] = None


class ModelRegistry:
	"""模型注册表，管理所有模型引擎实例。"""

	def __init__(self) -> None:
		self._llms: Dict[str, LLMEngine] = {}  # 模型名称 -> 引擎实例
		self._embeddings: Dict[str, EmbeddingEngine] = {}  # 模型名称 -> 引擎实例
		self._rerankers: Dict[str, RerankerEngine] = {}  # 模型名称 -> 引擎实例
		# 默认模型（优先使用第一个加载的模型）
		self._default_llm: Optional[str] = None
		self._default_embedding: Optional[str] = None
		self._default_reranker: Optional[str] = None
		# LLM对话策略
		self._llm_strategies: Dict[str, str] = {}
		# LLM模型特性参数
		self._llm_features: Dict[str, dict] = {}

	def get_llm(self, name: Optional[str]) -> Optional[LLMEngine]:
		"""获取指定名称的LLM引擎实例。"""
		if name is None:
			name = self._default_llm
		return self._llms.get(name)

	def get_llm_strategy_key(self, name: Optional[str]) -> Optional[str]:
		"""获取指定名称的LLM对话策略。"""
		if name is None:
			name = self._default_llm
		return self._llm_strategies.get(name)

	def get_llm_features(self, name: Optional[str]) -> dict:
		"""获取指定名称的LLM模型特性参数。"""
		if name is None:
			name = self._default_llm
		return self._llm_features.get(name, {})

	def get_embedding(self, name: Optional[str]) -> Optional[EmbeddingEngine]:
		"""获取指定名称的Embedding引擎实例。"""
		if name is None:
			name = self._default_embedding
		return self._embeddings.get(name)

	def get_reranker(self, name: Optional[str]) -> Optional[RerankerEngine]:
		"""获取指定名称的Reranker引擎实例。"""
		if name is None:
			name = self._default_reranker
		return self._rerankers.get(name)

	def has_any_llm(self) -> bool:
		"""是否加载了任何LLM模型。"""
		return len(self._llms) > 0

	def has_any_embedding(self) -> bool:
		"""是否加载了任何Embedding模型。"""
		return len(self._embeddings) > 0

	def has_any_reranker(self) -> bool:
		"""是否加载了任何Reranker模型。"""
		return len(self._rerankers) > 0

	# === 公开的计数方法 ===
	def llm_count(self) -> int:
		"""返回已加载的LLM模型数量。"""
		return len(self._llms)

	def embedding_count(self) -> int:
		"""返回已加载的Embedding模型数量。"""
		return len(self._embeddings)

	def reranker_count(self) -> int:
		"""返回已加载的Reranker模型数量。"""
		return len(self._rerankers)

	def clear(self) -> None:
		"""清空所有模型。"""
		# 清空注册表（让Python GC自然回收模型对象）
		self._llms.clear()
		self._embeddings.clear()
		self._rerankers.clear()
		self._llm_strategies.clear()
		self._llm_features.clear()
		self._default_llm = None
		self._default_embedding = None
		self._default_reranker = None

	def _validate_model_path(self, model_path: str, model_name: str) -> bool:
		"""验证模型路径是否存在。"""
		path = Path(model_path)
		if not path.exists():
			logger.error('模型路径不存在: {} -> {}', model_name, model_path)
			return False
		return True

	def load_from_config(self, settings: AppSettings) -> None:
		"""从 YAML 配置加载所有模型。严禁硬编码。"""
		# 如果已经加载过模型，跳过重复加载
		if self._llms or self._embeddings or self._rerankers:
			logger.info("模型已经加载过，跳过重复加载")
			return
		
		cfg = load_yaml(settings.models_config_path) or {}
		engine_defaults = cfg.get("engine_defaults", {})
		default_device = engine_defaults.get("device")
		default_dtype = engine_defaults.get("dtype")

		# LLMs
		llms_cfg = cfg.get("llms", [])
		for i, item in enumerate(llms_cfg):
				# 提取模型特性参数
			model_features = {}
			if "enable_thinking" in item:
				model_features["enable_thinking"] = item["enable_thinking"]
			
			llm_conf = LLMConfig(
				name=item["name"],
				engine=item.get("engine") or engine_defaults.get("llm_engine", "transformers"),
				path=item["path"],
				chat_strategy=item.get("chat_strategy", "generic"),
				dtype=item.get("dtype") or default_dtype,
				device=item.get("device") or default_device,
				gen_params=item.get("gen_params", {}),
				model_features=model_features if model_features else None,
			)
			# 验证模型路径
			if not self._validate_model_path(llm_conf.path, llm_conf.name):
				continue
			
			engine = self._build_llm_engine(llm_conf)
			try:
				if hasattr(engine, '_ensure_loaded'):
					engine._ensure_loaded()
					# 只有加载成功才添加到注册表
				self._llms[llm_conf.name] = engine
				self._llm_strategies[llm_conf.name] = llm_conf.chat_strategy
				if llm_conf.model_features:
					self._llm_features[llm_conf.name] = llm_conf.model_features
				if self._default_llm is None:
					self._default_llm = llm_conf.name
				logger.info("LLM loaded: {} via {} (strategy={})", llm_conf.name, llm_conf.engine, llm_conf.chat_strategy)
			except Exception as e:
				logger.error('加载 LLM 失败: {} -> {}', llm_conf.name, e)
				# 加载失败时不添加到注册表，不设置为默认模型

		# Embeddings
		embs_cfg = cfg.get("embeddings", [])
		for i, item in enumerate(embs_cfg):
			emb_conf = EmbeddingConfig(
				name=item["name"],
				engine=item.get("engine") or engine_defaults.get("embedding_engine", "transformers"),
				path=item["path"],
				device=item.get("device") or default_device,
			)
			# 验证模型路径
			if not self._validate_model_path(emb_conf.path, emb_conf.name):
				continue
			
			engine = self._build_embedding_engine(emb_conf)
			try:
				if hasattr(engine, '_ensure_loaded'):
					engine._ensure_loaded()
				# 只有加载成功才添加到注册表
				self._embeddings[emb_conf.name] = engine
				if self._default_embedding is None:
					self._default_embedding = emb_conf.name
				logger.info("Embedding loaded: {} via {}", emb_conf.name, emb_conf.engine)
			except Exception as e:
				logger.error('加载 Embedding 失败: {} -> {}', emb_conf.name, e)
				# 加载失败时不添加到注册表，不设置为默认模型

		# Rerankers
		rerank_cfg = cfg.get("rerankers", [])
		for i, item in enumerate(rerank_cfg):
			r_conf = RerankerConfig(
				name=item["name"],
				engine=item.get("engine") or engine_defaults.get("reranker_engine", "transformers"),
				path=item["path"],
				device=item.get("device") or default_device,
			)
			# 验证模型路径
			if not self._validate_model_path(r_conf.path, r_conf.name):
				continue
			
			engine = self._build_reranker_engine(r_conf)
			try:
				if hasattr(engine, '_ensure_loaded'):
					engine._ensure_loaded()
				# 只有加载成功才添加到注册表
				self._rerankers[r_conf.name] = engine
				if self._default_reranker is None:
					self._default_reranker = r_conf.name
				logger.info("Reranker loaded: {} via {}", r_conf.name, r_conf.engine)
			except Exception as e:
				logger.error('加载 Reranker 失败: {} -> {}', r_conf.name, e)
				# 加载失败时不添加到注册表，不设置为默认模型

	def _build_llm_engine(self, conf: LLMConfig) -> LLMEngine:
		if conf.engine == "transformers":
			return TransformersLLMEngine(model_path=conf.path, dtype=conf.dtype, device=conf.device, gen_params=conf.gen_params or {})
		elif conf.engine == "vllm":
			return VLLMLLMEngine(model_path=conf.path, dtype=conf.dtype, device=conf.device, gen_params=conf.gen_params or {})
		else:
			logger.warning(f"未知的 LLM 引擎类型: {conf.engine}，默认使用 transformers")
			return TransformersLLMEngine(model_path=conf.path, dtype=conf.dtype, device=conf.device, gen_params=conf.gen_params or {})

	def _build_embedding_engine(self, conf: EmbeddingConfig) -> EmbeddingEngine:
		if conf.engine == "transformers":
			return TransformersEmbeddingEngine(model_path=conf.path, device=conf.device)
		else:
			logger.warning(f"未知的 Embedding 引擎类型: {conf.engine}，默认使用 transformers")
			return TransformersEmbeddingEngine(model_path=conf.path, device=conf.device)

	def _build_reranker_engine(self, conf: RerankerConfig) -> RerankerEngine:
		if conf.engine == "transformers":
			return TransformersRerankerEngine(model_path=conf.path, device=conf.device)
		else:
			logger.warning(f"未知的 Reranker 引擎类型: {conf.engine}，默认使用 transformers")
			return TransformersRerankerEngine(model_path=conf.path, device=conf.device)


# 全局注册表实例
REGISTRY = ModelRegistry()