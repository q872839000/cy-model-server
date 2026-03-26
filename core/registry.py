from typing import Dict, Optional, Any, List
from loguru import logger
from pathlib import Path

from core.config.schemas import LLMModelConfig, EmbeddingModelConfig, RerankerModelConfig
from core.deployment.manager import DeploymentManager
from core.deployment.models import ModelDeployment
from engines.base import LLMEngine, EmbeddingEngine, RerankerEngine
from engines.transformers_engine import TransformersLLMEngine, TransformersEmbeddingEngine, TransformersRerankerEngine
from engines.vllm_engine import VLLMLLMEngine


class ModelRegistry:
	"""模型注册表，管理所有模型引擎实例。

	LLM 模型通过 DeploymentManager 管理（支持多副本、调度、并发控制）。
	Embedding/Reranker 保持单实例模式。
	"""

	def __init__(self) -> None:
		# LLM 由 DeploymentManager 管理
		self._deployment_manager = DeploymentManager()
		self._llm_configs: Dict[str, LLMModelConfig] = {}  # 模型名称 -> 配置对象
		# Embedding / Reranker 保持单实例
		self._embeddings: Dict[str, EmbeddingEngine] = {}
		self._embedding_configs: Dict[str, EmbeddingModelConfig] = {}
		self._rerankers: Dict[str, RerankerEngine] = {}
		self._reranker_configs: Dict[str, RerankerModelConfig] = {}
		# 规范化名称索引
		self._embedding_name_index: Dict[str, str] = {}
		self._reranker_name_index: Dict[str, str] = {}
		# 默认模型
		self._default_embedding: Optional[str] = None
		self._default_reranker: Optional[str] = None

	@staticmethod
	def _normalize_name(name: Optional[str]) -> Optional[str]:
		if name is None:
			return None
		return str(name).strip().lower()

	def _resolve_llm_name(self, name: Optional[str]) -> Optional[str]:
		return self._deployment_manager._resolve_name(name)

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

	def get_deployment(self, name: Optional[str]) -> Optional[ModelDeployment]:
		"""获取指定名称的LLM模型部署（含多副本和调度）。"""
		return self._deployment_manager.get_deployment(name)

	def get_llm(self, name: Optional[str]) -> Optional[LLMEngine]:
		"""获取指定名称的LLM引擎实例（向后兼容）。

		返回部署中第一个副本的引擎实例。
		新代码应优先使用 get_deployment() 获取部署级访问。
		"""
		deployment = self._deployment_manager.get_deployment(name)
		if deployment is None:
			return None
		replicas = deployment.replicas
		if not replicas:
			return None
		return replicas[0].engine

	def get_llm_config(self, name: Optional[str]) -> Optional[LLMModelConfig]:
		"""获取指定名称的LLM配置对象。"""
		deployment = self._deployment_manager.get_deployment(name)
		if deployment is None:
			return None
		return self._llm_configs.get(deployment.model_name)

	def get_llm_strategy_key(self, name: Optional[str]) -> Optional[str]:
		"""获取指定名称的LLM对话策略。"""
		deployment = self._deployment_manager.get_deployment(name)
		if deployment is not None:
			return deployment.strategy_key
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
		return self._deployment_manager.count() > 0

	def has_any_embedding(self) -> bool:
		"""是否加载了任何Embedding模型。"""
		return len(self._embeddings) > 0

	def has_any_reranker(self) -> bool:
		"""是否加载了任何Reranker模型。"""
		return len(self._rerankers) > 0

	# === 公开的计数方法 ===
	def llm_count(self) -> int:
		"""返回已加载的LLM模型数量。"""
		return self._deployment_manager.count()

	def embedding_count(self) -> int:
		"""返回已加载的Embedding模型数量。"""
		return len(self._embeddings)

	def reranker_count(self) -> int:
		"""返回已加载的Reranker模型数量。"""
		return len(self._rerankers)

	# === 公开的列表方法 ===
	def list_llm_names(self) -> list[str]:
		"""返回所有已加载的LLM模型名称列表。"""
		return self._deployment_manager.list_names()

	def list_embedding_names(self) -> list[str]:
		"""返回所有已加载的Embedding模型名称列表。"""
		return list(self._embeddings.keys())

	def list_reranker_names(self) -> list[str]:
		"""返回所有已加载的Reranker模型名称列表。"""
		return list(self._rerankers.keys())

	def get_default_llm_name(self) -> Optional[str]:
		"""返回默认LLM模型名称。"""
		return self._deployment_manager.get_default_name()

	def get_default_embedding_name(self) -> Optional[str]:
		"""返回默认Embedding模型名称。"""
		return self._default_embedding

	def get_default_reranker_name(self) -> Optional[str]:
		"""返回默认Reranker模型名称。"""
		return self._default_reranker

	def clear(self) -> None:
		"""清空所有模型。"""
		self._deployment_manager.clear()
		self._llm_configs.clear()
		self._embeddings.clear()
		self._embedding_configs.clear()
		self._rerankers.clear()
		self._reranker_configs.clear()
		self._embedding_name_index.clear()
		self._reranker_name_index.clear()
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
		"""从配置系统加载所有模型。

		LLM 模型通过 DeploymentManager 构建部署（支持多副本）。
		Embedding/Reranker 保持单实例模式。
		"""
		if self._deployment_manager.count() > 0 or self._embeddings or self._rerankers:
			logger.info("模型已经加载过，跳过重复加载")
			return

		from core.config import Config
		models_cfg = Config.models
		engine_defaults = models_cfg.engine_defaults

		# LLMs — 通过 DeploymentManager 构建部署
		llm_success = 0
		for llm_cfg in models_cfg.llms:
			self._llm_configs[llm_cfg.name] = llm_cfg
			ok = self._deployment_manager.build_and_register(
				llm_cfg=llm_cfg,
				engine_builder=self._build_llm_engine,
				engine_defaults=engine_defaults,
			)
			if ok:
				llm_success += 1

		# Embeddings（保持单实例模式）
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

		# 加载结果汇总
		total_configured = len(models_cfg.llms) + len(models_cfg.embeddings) + len(models_cfg.rerankers)
		total_loaded = llm_success + len(self._embeddings) + len(self._rerankers)
		failed_count = total_configured - total_loaded

		logger.info(
			"模型加载完成: LLM={}/{}, Embedding={}/{}, Reranker={}/{}",
			llm_success, len(models_cfg.llms),
			len(self._embeddings), len(models_cfg.embeddings),
			len(self._rerankers), len(models_cfg.rerankers),
		)

		# 打印部署级详情
		for name in self._deployment_manager.list_names():
			dep = self._deployment_manager.get_deployment(name)
			if dep:
				logger.info(
					"  部署 {}: replicas={} capacity={} strategy={}",
					name, len(dep.replicas), dep.total_capacity, dep.strategy_key,
				)

		if failed_count > 0:
			logger.warning("共 {} 个模型加载失败，请检查上方错误日志", failed_count)

		if not self._deployment_manager.count() and models_cfg.llms:
			logger.error("所有 LLM 模型加载失败，聊天功能将不可用")

	@staticmethod
	def _build_llm_engine(cfg: LLMModelConfig, engine: str, device: Optional[str], dtype: Optional[str]) -> LLMEngine:
		"""构建 LLM 引擎实例。

		作为构建函数传递给 DeploymentManager.build_and_register()，
		每个副本调用一次以创建独立的引擎实例。
		"""
		if engine == "transformers":
			return TransformersLLMEngine(
				model_path=cfg.path,
				dtype=dtype,
				device=device,
				gen_params=cfg.gen_params or {},
			)
		elif engine == "vllm":
			return VLLMLLMEngine(
				model_path=cfg.path,
				dtype=dtype,
				device=device,
				gen_params=cfg.gen_params or {},
				tensor_parallel_size=cfg.tensor_parallel_size,
			)
		else:
			logger.warning("未知的 LLM 引擎类型: {}，默认使用 transformers", engine)
			return TransformersLLMEngine(
				model_path=cfg.path,
				dtype=dtype,
				device=device,
				gen_params=cfg.gen_params or {},
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