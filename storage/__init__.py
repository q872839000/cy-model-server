"""
存储层模块

本模块提供各类存储后端的抽象与实现，包括：
- milvus: Milvus 向量数据库存储（支持混合检索）

存储层负责数据的持久化与检索，与业务逻辑解耦。
"""

from storage.milvus import MilvusClient, MilvusConfig, KBCollectionManager

__all__ = [
    "MilvusClient",
    "MilvusConfig", 
    "KBCollectionManager",
]
