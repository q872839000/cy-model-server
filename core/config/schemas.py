"""
配置数据结构定义

本模块统一定义项目所有的配置数据结构，包括：
- 应用级配置
- 知识库对话配置
- Milvus 数据库配置
- 模型配置
- 检索器配置
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any, List


# ==================== 应用级配置 ====================

class AppSettings(BaseSettings):
	"""
	应用级配置（环境变量前缀：UMS_）
	
	配置项：
	- env: 运行环境标识（dev/test/prod）
	- host/port: 服务监听地址与端口
	- log_level/log_format: 日志级别与格式
	- models_config_path: 模型配置文件（YAML）路径，严禁在代码中硬编码模型参数
	"""
	env: str = Field(default="dev")
	host: str = Field(default="0.0.0.0")
	port: int = Field(default=8000)
	log_level: str = Field(default="INFO")
	log_format: Optional[str] = Field(default=None)
	models_config_path: str = Field(default="configs/config.yaml")

	model_config = SettingsConfigDict(env_prefix="UMS_")
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
		"""从字典创建配置实例，只传递模型已定义的字段"""
		# 获取模型定义的字段名
		valid_fields = set(cls.model_fields.keys())
		# 只传递模型已定义且不为 None 的字段
		filtered_data = {k: v for k, v in data.items() if k in valid_fields and v is not None}
		return cls(**filtered_data)


class KBChatSettings(BaseSettings):
	"""
	知识库对话配置（环境变量前缀：KB_CHAT_）
	
	配置项：
	- 意图识别相关配置
	- Query 改写相关配置
	- 检索参数配置
	- Prompt 组装配置
	- 闲聊关键词配置
	"""
	# 意图识别
	intent_enabled: bool = Field(default=True, description="是否启用意图识别")
	intent_use_llm: bool = Field(default=False, description="是否使用LLM识别意图")
	
	# Query 改写
	query_rewrite_mode: str = Field(default="auto", description="改写模式: auto/always/never")
	rewrite_max_history_turns: int = Field(default=2, description="改写时参考的历史轮数")
	rewrite_timeout_ms: int = Field(default=10000, description="改写超时(毫秒)")
	
	# 检索参数
	search_top_k: int = Field(default=5, description="检索数量")
	search_rerank: bool = Field(default=True, description="是否重排序")
	search_score_threshold: float = Field(default=0.3, description="分数阈值")
	search_mode: str = Field(default="hybrid", description="检索模式")
	
	# Prompt 组装
	prompt_max_history_turns: int = Field(default=3, description="传给LLM的历史轮数")
	prompt_max_context_chars: int = Field(default=6000, description="检索内容最大字符")
	
	# 闲聊关键词
	chitchat_keywords: List[str] = Field(
		default_factory=lambda: [
			"好的", "谢谢", "明白了", "知道了", "了解", "OK", "ok",
			"嗯", "哦", "行", "可以", "没问题", "感谢", "辛苦",
			"太棒了", "很好", "不错", "懂了", "收到",
		],
		description="闲聊关键词"
	)
	
	model_config = SettingsConfigDict(env_prefix="KB_CHAT_")
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> "KBChatSettings":
		"""从字典创建配置实例，只传递模型已定义的字段"""
		# 获取模型定义的字段名
		valid_fields = set(cls.model_fields.keys())
		# 只传递模型已定义且不为 None 的字段
		filtered_data = {k: v for k, v in data.items() if k in valid_fields and v is not None}
		return cls(**filtered_data)


# ==================== Milvus 数据库配置 ====================

class MilvusConfig(BaseSettings):
    """
    Milvus 连接配置（环境变量前缀：MILVUS_）
    
    定义 Milvus 数据库的连接参数，支持认证、TLS等高级配置。
    
    属性:
        host: Milvus 服务地址
        port: Milvus 服务端口
        user: 用户名（可选，用于认证）
        password: 密码（可选，用于认证）
        database: 数据库名称
        secure: 是否使用 TLS 连接
        timeout: 连接超时时间（秒）
    """
    host: str = Field(default="localhost", description="Milvus 服务地址")
    port: int = Field(default=19530, description="Milvus 服务端口")
    user: Optional[str] = Field(default=None, description="用户名")
    password: Optional[str] = Field(default=None, description="密码")
    database: str = Field(default="default", description="数据库名称")
    secure: bool = Field(default=False, description="是否使用 TLS")
    timeout: float = Field(default=30.0, description="连接超时时间（秒）")
    
    model_config = SettingsConfigDict(env_prefix="MILVUS_")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MilvusConfig":
        """从字典创建配置实例"""
        return cls(**{k: v for k, v in data.items() if v is not None})


class KBCollectionConfig(BaseSettings):
    """
    知识库 Collection 配置（环境变量前缀：KB_）
    
    定义知识库 Collection 的索引参数和检索权重配置。
    Collection 名称由业务系统指定，本配置不做命名转换。
    
    属性:
        dense_dim: 稠密向量维度（需与 embedding 模型匹配）
        dense_index_type: 稠密向量索引类型
        dense_metric_type: 稠密向量距离度量类型
        hnsw_m: HNSW 索引参数 M
        hnsw_ef_construction: HNSW 索引构建时的 ef 参数
        bm25_k1: BM25 参数 k1
        bm25_b: BM25 参数 b
        dense_weight: Dense 向量在混合检索中的权重
        content_sparse_weight: Content Sparse (BM25) 在混合检索中的权重
        title_sparse_weight: Title Sparse (BM25) 在混合检索中的权重
    """
    # Dense Vector 配置
    dense_dim: int = Field(
        default=1024, 
        description="稠密向量维度，需与 embedding 模型输出维度匹配"
    )
    dense_index_type: str = Field(
        default="HNSW", 
        description="稠密向量索引类型: HNSW/IVF_FLAT/IVF_SQ8"
    )
    dense_metric_type: str = Field(
        default="COSINE", 
        description="稠密向量距离度量: COSINE/L2/IP"
    )
    
    # HNSW 索引参数
    hnsw_m: int = Field(
        default=16, 
        description="HNSW 参数 M，影响索引质量和内存占用"
    )
    hnsw_ef_construction: int = Field(
        default=256, 
        description="HNSW 构建时的 ef 参数，越大索引质量越高"
    )
    
    # BM25 参数
    bm25_k1: float = Field(
        default=1.2, 
        description="BM25 参数 k1，控制词频饱和度"
    )
    bm25_b: float = Field(
        default=0.75, 
        description="BM25 参数 b，控制文档长度归一化"
    )
    
    # 三路融合权重（用于 WeightedRanker）
    dense_weight: float = Field(
        default=1.0,
        description="Dense 向量在混合检索中的权重"
    )
    content_sparse_weight: float = Field(
        default=0.8,
        description="Content Sparse (BM25) 在混合检索中的权重"
    )
    title_sparse_weight: float = Field(
        default=0.5,
        description="Title Sparse (BM25) 在混合检索中的权重"
    )
    
    model_config = SettingsConfigDict(env_prefix="KB_", extra="ignore")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KBCollectionConfig":
        """从字典创建配置实例，忽略未定义的字段"""
        return cls(**{k: v for k, v in data.items() if v is not None})


# ==================== 模型配置 ====================

class EngineDefaults(BaseModel):
    """
    引擎默认配置
    
    定义各类型模型的默认引擎和设备配置。
    """
    llm_engine: str = Field(default="transformers", description="LLM默认引擎")
    embedding_engine: str = Field(default="transformers", description="Embedding默认引擎")
    reranker_engine: str = Field(default="transformers", description="Reranker默认引擎")
    device: Optional[str] = Field(default=None, description="默认设备")
    dtype: Optional[str] = Field(default=None, description="默认数据类型")


class LLMModelConfig(BaseModel):
    """
    LLM模型配置
    
    定义单个LLM模型的所有配置参数，配置结构明确直观。
    """
    name: str = Field(description="模型名称")
    engine: Optional[str] = Field(default=None, description="引擎类型")
    path: str = Field(description="模型路径")
    chat_strategy: str = Field(default="generic", description="对话策略")
    dtype: Optional[str] = Field(default=None, description="数据类型")
    device: Optional[str] = Field(default=None, description="设备")
    gen_params: Optional[Dict[str, Any]] = Field(default=None, description="生成参数")
    enable_thinking: bool = Field(default=False, description="是否启用思维模式")
    context_window: Optional[int] = Field(
        default=None,
        description="模型上下文窗口大小（token 数）。未配置时自动从模型 config 探测",
    )


class EmbeddingModelConfig(BaseModel):
    """
    Embedding模型配置
    
    定义单个Embedding模型的所有配置参数。
    """
    name: str = Field(description="模型名称")
    engine: Optional[str] = Field(default=None, description="引擎类型")
    path: str = Field(description="模型路径")
    device: Optional[str] = Field(default=None, description="设备")


class RerankerModelConfig(BaseModel):
    """
    Reranker模型配置
    
    定义单个Reranker模型的所有配置参数。
    """
    name: str = Field(description="模型名称")
    engine: Optional[str] = Field(default=None, description="引擎类型")
    path: str = Field(description="模型路径")
    device: Optional[str] = Field(default=None, description="设备")


class ModelsConfig(BaseSettings):
    """
    模型配置（环境变量前缀：MODELS_）
    
    管理LLM、Embedding、Reranker模型的配置信息。
    使用明确定义的配置模型，不使用通用Dict。
    """
    # 引擎默认配置
    engine_defaults: EngineDefaults = Field(default_factory=EngineDefaults, description="引擎默认配置")
    
    # 模型配置列表，使用明确的配置模型
    llms: List[LLMModelConfig] = Field(default_factory=list, description="LLM模型配置列表")
    embeddings: List[EmbeddingModelConfig] = Field(default_factory=list, description="Embedding模型配置列表")
    rerankers: List[RerankerModelConfig] = Field(default_factory=list, description="Reranker模型配置列表")
    
    model_config = SettingsConfigDict(env_prefix="MODELS_")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelsConfig":
        """从字典创建配置实例"""
        # 处理引擎默认配置
        engine_defaults_data = data.get("engine_defaults", {})
        engine_defaults = EngineDefaults(**engine_defaults_data)
        
        # 处理LLM配置列表，直接映射配置字段
        llms_data = data.get("llms", [])
        llms = [LLMModelConfig(**llm_data) for llm_data in llms_data]
        
        # 处理Embedding配置列表
        embeddings_data = data.get("embeddings", [])
        embeddings = [EmbeddingModelConfig(**emb_data) for emb_data in embeddings_data]
        
        # 处理Reranker配置列表
        rerankers_data = data.get("rerankers", [])
        rerankers = [RerankerModelConfig(**rerank_data) for rerank_data in rerankers_data]
        
        return cls(
            engine_defaults=engine_defaults,
            llms=llms,
            embeddings=embeddings,
            rerankers=rerankers
        )


# ==================== 检索器配置 ====================

class RetrieverConfig(BaseSettings):
    """
    检索器配置（环境变量前缀：RETRIEVER_）
    
    定义混合检索器的各项参数，支持从配置文件和环境变量加载。
    
    Attributes:
        default_top_k: 默认返回结果数量
        dense_weight: 混合检索中稠密向量的权重
        content_sparse_weight: 混合检索中内容稀疏向量的权重
        title_sparse_weight: 混合检索中标题稀疏向量的权重
        rrf_k: RRF 融合算法的 k 参数
        use_rrf: 是否使用 RRF 融合（否则使用加权融合）
        ef_search: HNSW 搜索时的 ef 参数
    """
    default_top_k: int = Field(default=10, description="默认返回结果数量")
    dense_weight: float = Field(default=1.0, description="混合检索中稠密向量的权重")
    content_sparse_weight: float = Field(default=0.8, description="混合检索中内容稀疏向量的权重")
    title_sparse_weight: float = Field(default=0.5, description="混合检索中标题稀疏向量的权重")
    rrf_k: int = Field(default=60, description="RRF 融合算法的 k 参数")
    use_rrf: bool = Field(default=False, description="是否使用 RRF 融合")
    ef_search: int = Field(default=64, description="HNSW 搜索时的 ef 参数")
    
    model_config = SettingsConfigDict(env_prefix="RETRIEVER_")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrieverConfig":
        """从字典创建配置实例，忽略未定义的字段"""
        return cls(**{k: v for k, v in data.items() if v is not None})
