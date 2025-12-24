"""
Milvus 配置模块

本模块定义 Milvus 连接和 Collection 的配置参数。
配置优先级：环境变量 > config.yaml > 默认值
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any


class MilvusConfig(BaseSettings):
    """
    Milvus 连接配置。
    
    支持通过环境变量覆盖，环境变量前缀为 MILVUS_。
    例如：MILVUS_HOST=localhost 会覆盖 host 字段。
    
    Attributes:
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
    Collection 全局配置。
    
    定义 Collection 的索引参数。
    Collection 名称由业务系统指定，本模块不做命名转换。
    
    支持通过环境变量覆盖，前缀为 KB_。
    
    Attributes:
        dense_dim: 稠密向量维度（需与 embedding 模型匹配）
        dense_index_type: 稠密向量索引类型
        dense_metric_type: 稠密向量距离度量类型
        hnsw_m: HNSW 索引参数 M
        hnsw_ef_construction: HNSW 索引构建时的 ef 参数
        bm25_k1: BM25 参数 k1
        bm25_b: BM25 参数 b
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


# 全局配置缓存
_milvus_config: Optional[MilvusConfig] = None
_kb_collection_config: Optional[KBCollectionConfig] = None


def load_milvus_config(yaml_config: Optional[Dict[str, Any]] = None) -> MilvusConfig:
    """
    加载 Milvus 连接配置。
    
    优先级：环境变量 > yaml_config > 默认值
    
    Args:
        yaml_config: 从 config.yaml 读取的 milvus 配置段
        
    Returns:
        MilvusConfig: Milvus 连接配置实例
    """
    global _milvus_config
    
    if _milvus_config is not None:
        return _milvus_config
    
    if yaml_config:
        _milvus_config = MilvusConfig.from_dict(yaml_config)
    else:
        _milvus_config = MilvusConfig()
    
    return _milvus_config


def load_kb_collection_config(yaml_config: Optional[Dict[str, Any]] = None) -> KBCollectionConfig:
    """
    加载知识库 Collection 配置。
    
    优先级：环境变量 > yaml_config > 默认值
    
    Args:
        yaml_config: 从 config.yaml 读取的 knowledge_base 配置段
        
    Returns:
        KBCollectionConfig: Collection 配置实例
    """
    global _kb_collection_config
    
    if _kb_collection_config is not None:
        return _kb_collection_config
    
    if yaml_config:
        _kb_collection_config = KBCollectionConfig.from_dict(yaml_config)
    else:
        _kb_collection_config = KBCollectionConfig()
    
    return _kb_collection_config


def init_configs_from_yaml(config_path: str = "configs/config.yaml") -> None:
    """
    从 YAML 配置文件初始化所有 Milvus 相关配置。
    
    应在应用启动时调用一次。
    
    Args:
        config_path: 配置文件路径
    """
    from core.config import load_yaml
    
    cfg = load_yaml(config_path) or {}
    
    # 加载 Milvus 连接配置
    milvus_cfg = cfg.get("milvus") or {}
    load_milvus_config(milvus_cfg)
    
    # 加载知识库 Collection 配置
    kb_cfg = cfg.get("knowledge_base") or {}
    load_kb_collection_config(kb_cfg)
