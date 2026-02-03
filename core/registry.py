from typing import Dict, Optional, Any
from loguru import logger
from pathlib import Path

from core.config.schemas import LLMModelConfig, EmbeddingModelConfig, RerankerModelConfig
from engines.base import LLMEngine, EmbeddingEngine, RerankerEngine
from engines.transformers_engine import TransformersLLMEngine, TransformersEmbeddingEngine, TransformersRerankerEngine
from engines.vllm_engine import VLLMLLMEngine


class ModelRegistry:
	"""模型注册表，管理所有模型引擎实例。"""

	def __init__(self) -> None:
		# 直接存储配置对象和引擎实例
		self._llms: Dict[str, LLMEngine] = {}  # 模型名称 -> 引擎实例
		self._llm_configs: Dict[str, LLMModelConfig] = {}  # 模型名称 -> 配置对象
		self._embeddings: Dict[str, EmbeddingEngine] = {}  # 模型名称 -> 引擎实例
		self._embedding_configs: Dict[str, EmbeddingModelConfig] = {}  # 模型名称 -> 配置对象
		self._rerankers: Dict[str, RerankerEngine] = {}  # 模型名称 -> 引擎实例
		self._reranker_configs: Dict[str, RerankerModelConfig] = {}  # 模型名称 -> 配置对象
		# 规范化名称索引（用于忽略大小写匹配）
		self._llm_name_index: Dict[str, str] = {}
		self._embedding_name_index: Dict[str, str] = {}
		self._reranker_name_index: Dict[str, str] = {}
		# 默认模型（优先使用第一个加载的模型）
		self._default_llm: Optional[str] = None
		self._default_embedding: Optional[str] = None
		self._default_reranker: Optional[str] = None

	@staticmethod
	def _normalize_name(name: Optional[str]) -> Optional[str]:
		if name is None:
			return None
		return str(name).strip().lower()

	def _resolve_llm_name(self, name: Optional[str]) -> Optional[str]:
		if name is None:
			return None
		if name in self._llms:
			return name
		return self._llm_name_index.get(self._normalize_name(name) or "")

	def _resolve_embedding_name(self, name: Optional[str]) -> Optional[str]:
		if name is None:
			return None
		if name in self._embeddings:
			return name
		return self._embedding_name_index.get(self._normalize_name(name) or "")

	def _resolve_reranker_name(self, name: Optional[str]) -> Optional[str]:
		if name is None:
			return None
		if name in self._rerankers:
			return name
		return self._reranker_name_index.get(self._normalize_name(name) or "")

	def get_llm(self, name: Optional[str]) -> Optional[LLMEngine]:
		"""获取指定名称的LLM引擎实例。"""
		if name is None:
			name = self._default_llm
		resolved = self._resolve_llm_name(name)
		if resolved is None:
			return None
		return self._llms.get(resolved)

	def get_llm_config(self, name: Optional[str]) -> Optional[LLMModelConfig]:
		"""获取指定名称的LLM配置对象。"""
		if name is None:
			name = self._default_llm
		resolved = self._resolve_llm_name(name)
		if resolved is None:
			return None
		return self._llm_configs.get(resolved)

	def get_llm_strategy_key(self, name: Optional[str]) -> Optional[str]:
		"""获取指定名称的LLM对话策略。"""
		config = self.get_llm_config(name)
		return config.chat_strategy if config else None

	def get_embedding_config(self, name: Optional[str]) -> Optional[EmbeddingModelConfig]:
		"""获取指定名称的Embedding配置对象。"""
		if name is None:
			name = self._default_embedding
		resolved = self._resolve_embedding_name(name)
		if resolved is None:
			return None
		return self._embedding_configs.get(resolved)

	def get_reranker_config(self, name: Optional[str]) -> Optional[RerankerModelConfig]:
		"""获取指定名称的Reranker配置对象。"""
		if name is None:
			name = self._default_reranker
		resolved = self._resolve_reranker_name(name)
		if resolved is None:
			return None
		return self._reranker_configs.get(resolved)

	def get_embedding(self, name: Optional[str]) -> Optional[EmbeddingEngine]:
		"""获取指定名称的Embedding引擎实例。"""
		if name is None:
			name = self._default_embedding
		resolved = self._resolve_embedding_name(name)
		if resolved is None:
			return None
		return self._embeddings.get(resolved)

	def get_reranker(self, name: Optional[str]) -> Optional[RerankerEngine]:
		"""获取指定名称的Reranker引擎实例。"""
		if name is None:
			name = self._default_reranker
		resolved = self._resolve_reranker_name(name)
		if resolved is None:
			return None
		return self._rerankers.get(resolved)

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

	# === 公开的列表方法 ===
	def list_llm_names(self) -> list[str]:
		"""返回所有已加载的LLM模型名称列表。"""
		return list(self._llms.keys())

	def list_embedding_names(self) -> list[str]:
		"""返回所有已加载的Embedding模型名称列表。"""
		return list(self._embeddings.keys())

	def list_reranker_names(self) -> list[str]:
		"""返回所有已加载的Reranker模型名称列表。"""
		return list(self._rerankers.keys())

	def get_default_llm_name(self) -> Optional[str]:
		"""返回默认LLM模型名称。"""
		return self._default_llm

	def get_default_embedding_name(self) -> Optional[str]:
		"""返回默认Embedding模型名称。"""
		return self._default_embedding

	def get_default_reranker_name(self) -> Optional[str]:
		"""返回默认Reranker模型名称。"""
		return self._default_reranker

	def clear(self) -> None:
		"""清空所有模型。"""
		# 清空注册表（让Python GC自然回收模型对象）
		self._llms.clear()
		self._llm_configs.clear()
		self._embeddings.clear()
		self._embedding_configs.clear()
		self._rerankers.clear()
		self._reranker_configs.clear()
		self._llm_name_index.clear()
		self._embedding_name_index.clear()
		self._reranker_name_index.clear()
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

	def load_from_config(self) -> None:
		"""从配置系统加载所有模，避免重复转换。"""
		# 如果已经加载过模型，跳过重复加载
		if self._llms or self._embeddings or self._rerankers:
			logger.info("模型已经加载过，跳过重复加载")
			return
		
		# 使用统一配置系统获取已加载的配置对象
		from core.config import Config
		models_cfg = Config.models
		engine_defaults = models_cfg.engine_defaults

		# LLMs
		for llm_cfg in models_cfg.llms:
			# 应用默认值
			engine = llm_cfg.engine or engine_defaults.llm_engine
			device = llm_cfg.device or engine_defaults.device
			dtype = llm_cfg.dtype or engine_defaults.dtype
			
			# 验证模型路径
			if not self._validate_model_path(llm_cfg.path, llm_cfg.name):
				continue
			
			# 构建引擎（直接使用配置参数）
			engine_instance = self._build_llm_engine(llm_cfg, engine, device, dtype)
			try:
				if hasattr(engine_instance, '_ensure_loaded'):
					engine_instance._ensure_loaded()
				
				# 存储引擎和完整配置对象
				self._llms[llm_cfg.name] = engine_instance
				self._llm_configs[llm_cfg.name] = llm_cfg
				normalized_name = self._normalize_name(llm_cfg.name)
				if normalized_name:
					self._llm_name_index[normalized_name] = llm_cfg.name
				if self._default_llm is None:
					self._default_llm = llm_cfg.name
				logger.info("LLM loaded: {} via {} (strategy={})", llm_cfg.name, engine, llm_cfg.chat_strategy)
			except Exception as e:
				logger.error('加载 LLM 失败: {} -> {}', llm_cfg.name, e)

		# Embeddings
		for emb_cfg in models_cfg.embeddings:
			# 应用默认值
			engine = emb_cfg.engine or engine_defaults.embedding_engine
			device = emb_cfg.device or engine_defaults.device
			
			# 验证模型路径
			if not self._validate_model_path(emb_cfg.path, emb_cfg.name):
				continue
			
			# 构建引擎
			engine_instance = self._build_embedding_engine(emb_cfg, engine, device)
			try:
				if hasattr(engine_instance, '_ensure_loaded'):
					engine_instance._ensure_loaded()
				
				# 存储引擎和完整配置对象
				self._embeddings[emb_cfg.name] = engine_instance
				self._embedding_configs[emb_cfg.name] = emb_cfg
				normalized_name = self._normalize_name(emb_cfg.name)
				if normalized_name:
					self._embedding_name_index[normalized_name] = emb_cfg.name
				if self._default_embedding is None:
					self._default_embedding = emb_cfg.name
				logger.info("Embedding loaded: {} via {}", emb_cfg.name, engine)
			except Exception as e:
				logger.error('加载 Embedding 失败: {} -> {}', emb_cfg.name, e)

		# Rerankers
		for rerank_cfg in models_cfg.rerankers:
			# 应用默认值
			engine = rerank_cfg.engine or engine_defaults.reranker_engine
			device = rerank_cfg.device or engine_defaults.device
			
			# 验证模型路径
			if not self._validate_model_path(rerank_cfg.path, rerank_cfg.name):
				continue
			
			# 构建引擎
			engine_instance = self._build_reranker_engine(rerank_cfg, engine, device)
			try:
				if hasattr(engine_instance, '_ensure_loaded'):
					engine_instance._ensure_loaded()
				
				# 存储引擎和完整配置对象
				self._rerankers[rerank_cfg.name] = engine_instance
				self._reranker_configs[rerank_cfg.name] = rerank_cfg
				normalized_name = self._normalize_name(rerank_cfg.name)
				if normalized_name:
					self._reranker_name_index[normalized_name] = rerank_cfg.name
				if self._default_reranker is None:
					self._default_reranker = rerank_cfg.name
				logger.info("Reranker loaded: {} via {}", rerank_cfg.name, engine)
			except Exception as e:
				logger.error('加载 Reranker 失败: {} -> {}', rerank_cfg.name, e)

	def _build_llm_engine(self, cfg: LLMModelConfig, engine: str, device: Optional[str], dtype: Optional[str]) -> LLMEngine:
		"""构建 LLM 引擎实例。"""
		if engine == "transformers":
			return TransformersLLMEngine(
				model_path=cfg.path,
				dtype=dtype,
				device=device,
				gen_params=cfg.gen_params or {}
			)
		elif engine == "vllm":
			return VLLMLLMEngine(
				model_path=cfg.path,
				dtype=dtype,
				device=device,
				gen_params=cfg.gen_params or {}
			)
		else:
			logger.warning(f"未知的 LLM 引擎类型: {engine}，默认使用 transformers")
			return TransformersLLMEngine(
				model_path=cfg.path,
				dtype=dtype,
				device=device,
				gen_params=cfg.gen_params or {}
			)

	def _build_embedding_engine(self, cfg: EmbeddingModelConfig, engine: str, device: Optional[str]) -> EmbeddingEngine:
		"""构建 Embedding 引擎实例。"""
		if engine == "transformers":
			return TransformersEmbeddingEngine(model_path=cfg.path, device=device)
		else:
			logger.warning(f"未知的 Embedding 引擎类型: {engine}，默认使用 transformers")
			return TransformersEmbeddingEngine(model_path=cfg.path, device=device)

	def _build_reranker_engine(self, cfg: RerankerModelConfig, engine: str, device: Optional[str]) -> RerankerEngine:
		"""构建 Reranker 引擎实例。"""
		if engine == "transformers":
			return TransformersRerankerEngine(model_path=cfg.path, device=device)
		else:
			logger.warning(f"未知的 Reranker 引擎类型: {engine}，默认使用 transformers")
			return TransformersRerankerEngine(model_path=cfg.path, device=device)


# 全局注册表实例
REGISTRY = ModelRegistry()