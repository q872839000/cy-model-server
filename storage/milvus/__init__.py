"""
Milvus 存储模块

本模块提供 Milvus 向量数据库的封装，包括：
- MilvusConfig: Milvus 连接配置
- MilvusClient: Milvus 客户端封装
- KBCollectionManager: 知识库 Collection 管理器

支持 Milvus 2.5+ 的混合检索能力（Dense + Sparse/BM25）。

全局连接管理：
- init_milvus(): 服务启动时调用，建立连接
- shutdown_milvus(): 服务关闭时调用，断开连接
- get_milvus_client(): 获取已连接的客户端实例
"""

from typing import Optional
from loguru import logger

from core.config import MilvusConfig, KBCollectionConfig
from storage.milvus.client import MilvusClient
from storage.milvus.collections import KBCollectionManager

__all__ = [
    "MilvusConfig",
    "KBCollectionConfig", 
    "MilvusClient",
    "KBCollectionManager",
    "init_milvus",
    "shutdown_milvus",
    "get_milvus_client",
]

# 全局 Milvus 客户端实例
_milvus_client: Optional[MilvusClient] = None


def init_milvus() -> bool:
    """
    初始化 Milvus 服务。
    
    应在应用启动时调用，使用全局配置系统加载配置，
    然后建立连接。连接失败时不会导致程序崩溃，而是优雅处理。
    
    Returns:
        bool: 初始化是否成功
    """
    global _milvus_client
    
    try:
        # 使用全局配置系统创建客户端
        _milvus_client = MilvusClient()
        _milvus_client.connect()
        
        logger.info("Milvus 服务初始化成功")
        return True
        
    except Exception as e:
        logger.warning("Milvus 服务初始化失败，RAG功能不可用: {}", str(e))
        _milvus_client = None
        return False


def shutdown_milvus() -> None:
    """
    关闭 Milvus 连接。
    
    在服务关闭时调用，断开与 Milvus 的连接。
    """
    global _milvus_client
    
    if _milvus_client is not None:
        try:
            _milvus_client.disconnect()
            logger.info("Milvus 连接已关闭")
        except Exception as e:
            logger.warning("Milvus 断开连接失败: {}", str(e))
        _milvus_client = None


def get_milvus_client() -> Optional[MilvusClient]:
    """
    获取 Milvus 客户端实例。
    
    Returns:
        Optional[MilvusClient]: 已连接的客户端实例，未初始化时返回 None
    """
    return _milvus_client
