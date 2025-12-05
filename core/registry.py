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
	"""全局模型注册表：负责按配置加载与管理 LLM/Embedding/Reranker 引擎实例。"""

	def __init__(self) -> None:
		self._llms: Dict[str, LLMEngine] = {}
		self._embeddings: Dict[str, EmbeddingEngine] = {}
		self._rerankers: Dict[str, RerankerEngine] = {}
		self._llm_strategies: Dict[str, str] = {}  # model_name -> strategy key
		self._default_llm: Optional[str] = None
		self._default_embedding: Optional[str] = None
		self._default_reranker: Optional[str] = None

	def get_llm(self, name: Optional[str]) -> Optional[LLMEngine]:
		key = name or self._default_llm
		return self._llms.get(key) if key else None

	def get_llm_strategy_key(self, name: Optional[str]) -> Optional[str]:
		key = name or self._default_llm
		return self._llm_strategies.get(key) if key else None

	def get_embedding(self, name: Optional[str]) -> Optional[EmbeddingEngine]:
		key = name or self._default_embedding
		return self._embeddings.get(key) if key else None

	def get_reranker(self, name: Optional[str]) -> Optional[RerankerEngine]:
		key = name or self._default_reranker
		return self._rerankers.get(key) if key else None

	def has_any_llm(self) -> bool:
		return bool(self._llms)

	def has_any_embedding(self) -> bool:
		return bool(self._embeddings)

	def has_any_reranker(self) -> bool:
		return bool(self._rerankers)

	def clear(self) -> None:
		self._llms.clear()
		self._embeddings.clear()
		self._rerankers.clear()
		self._llm_strategies.clear()
		self._default_llm = None
		self._default_embedding = None
		self._default_reranker = None

	def _validate_model_path(self, model_path: str, model_name: str) -> bool:
		"""验证模型路径是否存在"""
		path = Path(model_path)
		if not path.exists():
			logger.error("模型路径不存在: {} -> {}", model_name, model_path)
			return False
		if not path.is_dir():
			logger.error("模型路径不是目录: {} -> {}", model_name, model_path)
			return False
		# 检查是否有config.json文件
		config_file = path / "config.json"
		if not config_file.exists():
			logger.error("模型配置文件不存在: {} -> {}", model_name, config_file)
			return False
		return True

	def load_from_config(self, settings: AppSettings) -> None:
		"""从 YAML 配置加载所有模型。严禁硬编码。"""
		cfg = load_yaml(settings.models_config_path) or {}
		engine_defaults = cfg.get("engine_defaults", {})

		# LLMs
		llms_cfg = cfg.get("llms", [])
		for i, item in enumerate(llms_cfg):
			llm_conf = LLMConfig(
				name=item["name"],
				engine=item.get("engine") or engine_defaults.get("llm_engine", "transformers"),
				path=item["path"],
				chat_strategy=item.get("chat_strategy", "generic"),
				dtype=item.get("dtype"),
				device=item.get("device"),
				gen_params=item.get("gen_params", {}),
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
				if self._default_llm is None:
					self._default_llm = llm_conf.name
				logger.info("LLM loaded: {} via {}", llm_conf.name, llm_conf.engine)
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
				device=item.get("device"),
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
				device=item.get("device"),
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
		if conf.engine == "vllm":
			return VLLMLLMEngine(model_path=conf.path, dtype=conf.dtype, device=conf.device, gen_params=conf.gen_params or {})
		raise ValueError(f"Unsupported LLM engine: {conf.engine}")

	def _build_embedding_engine(self, conf: EmbeddingConfig) -> EmbeddingEngine:
		if conf.engine == "transformers":
			return TransformersEmbeddingEngine(model_path=conf.path, device=conf.device)
		raise ValueError(f"Unsupported Embedding engine: {conf.engine}")

	def _build_reranker_engine(self, conf: RerankerConfig) -> RerankerEngine:
		if conf.engine == "transformers":
			return TransformersRerankerEngine(model_path=conf.path, device=conf.device)
		raise ValueError(f"Unsupported Reranker engine: {conf.engine}")


# 全局注册表实例（简单单例）
REGISTRY = ModelRegistry()
