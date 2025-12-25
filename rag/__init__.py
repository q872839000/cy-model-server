"""
RAG（检索增强生成）业务模块

本模块提供 RAG 相关的业务能力，包括：
- retriever: 混合检索器（Dense + Sparse/BM25）
- service: RAG 业务服务（整合检索、重排序）

模块依赖：
- storage: 存储层（Milvus）
- models.kb_schemas: 知识库数据结构
"""

from rag.retriever import HybridRetriever
from core.config import RetrieverConfig
from rag.service import RAGService

__all__ = [
    "HybridRetriever", 
    "RetrieverConfig",
    "RAGService",
]
