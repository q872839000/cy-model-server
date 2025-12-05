"""
高级检索服务：提供多种召回策略和重排选项

功能：
- 多路召回：向量召回、关键词召回、混合召回
- 召回后处理：去重、过滤、融合
- 多阶段重排：粗排(向量) -> 精排(reranker) -> 多样性调整
- 可配置的召回链路
"""

from typing import List, Dict, Optional, Any, Tuple
from loguru import logger
from dataclasses import dataclass
from enum import Enum

from models.schemas import Document, RetrievalItem
from core.registry import REGISTRY
from services.milvus_client import MilvusClient
from services.vector_store_adapter import get_adapter
from services.document_content_service import get_document_content_service


class RecallStrategy(str, Enum):
    """召回策略枚举"""
    VECTOR_ONLY = "vector_only"  # 纯向量召回
    HYBRID = "hybrid"  # 混合召回（向量+关键词）
    MULTI_VECTOR = "multi_vector"  # 多向量召回（使用不同的embedding模型）


class RerankStrategy(str, Enum):
    """重排策略枚举"""
    NONE = "none"  # 不重排
    RERANKER = "reranker"  # 使用Reranker模型
    MMR = "mmr"  # 最大边际相关性（多样性）
    RERANKER_MMR = "reranker_mmr"  # Reranker + MMR


@dataclass
class RecallConfig:
    """
    召回配置
    
    属性:
        strategy: 召回策略（vector_only/hybrid/multi_vector）
        top_k: 召回数量
        similarity_threshold: COSINE相似度阈值，范围[-1,1]，1表示完全相同，0表示正交，-1表示完全相反
                             实际常用范围0.7-1.0，例如0.8表示高相似度，只保留相似度>=0.8的结果
        enable_dedup: 是否去重
    """
    strategy: RecallStrategy = RecallStrategy.VECTOR_ONLY
    top_k: int = 20  # 召回数量
    similarity_threshold: Optional[float] = None  # COSINE相似度阈值（-1到1）
    enable_dedup: bool = True  # 是否去重


@dataclass
class RerankConfig:
    """重排配置"""
    strategy: RerankStrategy = RerankStrategy.RERANKER
    final_top_k: int = 10  # 最终返回数量
    mmr_lambda: float = 0.5  # MMR多样性参数（0-1，越大越注重多样性）


class AdvancedRetrievalService:
    """高级检索服务"""

    def __init__(self):
        self._milvus_client = MilvusClient()
        self._adapter = get_adapter()
        self._content_service = get_document_content_service()

    def recall(
        self,
        query: str,
        collection_name: str,
        embedding_model: str,
        config: RecallConfig
    ) -> List[RetrievalItem]:
        """
        多策略召回
        
        Args:
            query: 查询文本
            collection_name: 集合名称
            embedding_model: Embedding模型名称
            config: 召回配置
        
        Returns:
            召回结果列表
        """
        if config.strategy == RecallStrategy.VECTOR_ONLY:
            return self._vector_recall(query, collection_name, embedding_model, config)
        elif config.strategy == RecallStrategy.HYBRID:
            return self._hybrid_recall(query, collection_name, embedding_model, config)
        elif config.strategy == RecallStrategy.MULTI_VECTOR:
            return self._multi_vector_recall(query, collection_name, config)
        else:
            raise ValueError(f"不支持的召回策略: {config.strategy}")

    def _vector_recall(
        self,
        query: str,
        collection_name: str,
        embedding_model: str,
        config: RecallConfig
    ) -> List[RetrievalItem]:
        """纯向量召回"""
        engine = REGISTRY.get_embedding(embedding_model)
        if not engine:
            raise ValueError(f"未找到embedding模型: {embedding_model}")
        
        # 生成查询向量
        query_vector = engine.embed([query])[0]
        
        # 向量检索（返回 DocChunk 和 score）
        chunk_groups = self._milvus_client.search_doc_chunks(
            collection_name=collection_name,
            query_vectors=[query_vector],
            top_k=config.top_k
        )
        
        # 获取第一组结果
        chunk_results = chunk_groups[0] if chunk_groups else []
        
        # 批量查询 MySQL 获取真实文档内容
        chunk_ids = [result['chunk'].chunk_id for result in chunk_results]
        content_map = {}
        if chunk_ids:
            try:
                content_map = self._content_service.batch_get_chunk_contents(chunk_ids)
                logger.info(f"从 MySQL 批量查询到 {len(content_map)} 条文档内容")
            except Exception as e:
                logger.warning(f"从 MySQL 查询文档内容失败，将使用占位符: {e}")
        
        # 转换为RetrievalItem
        items = []
        for result in chunk_results:
            chunk = result['chunk']
            score = result['score']
            
            # 应用相似度阈值
            if config.similarity_threshold and score < config.similarity_threshold:
                continue
            
            # 从 content_map 中获取真实内容
            content = content_map.get(chunk.chunk_id)
            
            # 转换为 Document
            doc = self._adapter.doc_chunk_to_document(chunk, content=content)
            
            # 创建 RetrievalItem
            items.append(RetrievalItem(
                document=doc,
                score=score
            ))
        
        logger.info(f"向量召回: {len(items)} 条结果")
        return items

    def _hybrid_recall(
        self,
        query: str,
        collection_name: str,
        embedding_model: str,
        config: RecallConfig
    ) -> List[RetrievalItem]:
        """
        混合召回：向量召回 + 关键词过滤
        注：Milvus支持标量过滤，这里演示如何结合使用
        """
        # 先做向量召回
        items = self._vector_recall(query, collection_name, embedding_model, config)
        
        # TODO: 可以在这里添加关键词匹配、BM25等传统检索方法
        # 目前简单返回向量召回结果
        
        logger.info(f"混合召回: {len(items)} 条结果")
        return items

    def _multi_vector_recall(
        self,
        query: str,
        collection_name: str,
        config: RecallConfig
    ) -> List[RetrievalItem]:
        """
        多向量召回：使用不同的embedding模型进行召回，然后融合结果
        """
        all_items: Dict[str, RetrievalItem] = {}  # 使用doc_id去重
        
        # 获取所有可用的embedding模型
        embedding_models = list(REGISTRY._embeddings.keys())
        if not embedding_models:
            raise ValueError("没有可用的embedding模型")
        
        logger.info(f"使用 {len(embedding_models)} 个embedding模型进行多路召回")
        
        # 使用每个模型进行召回
        for model_name in embedding_models:
            try:
                items = self._vector_recall(
                    query,
                    collection_name,
                    model_name,
                    RecallConfig(
                        strategy=RecallStrategy.VECTOR_ONLY,
                        top_k=config.top_k,
                        similarity_threshold=config.similarity_threshold
                    )
                )
                
                # 融合结果（使用RRF - Reciprocal Rank Fusion）
                for rank, item in enumerate(items, 1):
                    doc_id = item.document.id
                    if doc_id in all_items:
                        # 已存在，累加分数（RRF公式）
                        all_items[doc_id].score += 1.0 / (60 + rank)
                    else:
                        # 新文档
                        item.score = 1.0 / (60 + rank)
                        all_items[doc_id] = item
            except Exception as e:
                logger.warning(f"使用模型 {model_name} 召回失败: {e}")
                continue
        
        # 按融合后的分数排序
        result_items = sorted(all_items.values(), key=lambda x: x.score, reverse=True)
        result_items = result_items[:config.top_k]
        
        logger.info(f"多向量召回融合后: {len(result_items)} 条结果")
        return result_items

    def rerank(
        self,
        query: str,
        items: List[RetrievalItem],
        config: RerankConfig
    ) -> List[RetrievalItem]:
        """
        多策略重排
        
        Args:
            query: 查询文本
            items: 待重排的结果列表
            config: 重排配置
        
        Returns:
            重排后的结果列表
        """
        if not items:
            return items
        
        if config.strategy == RerankStrategy.NONE:
            return items[:config.final_top_k]
        elif config.strategy == RerankStrategy.RERANKER:
            return self._reranker_rerank(query, items, config.final_top_k)
        elif config.strategy == RerankStrategy.MMR:
            return self._mmr_rerank(query, items, config.final_top_k, config.mmr_lambda)
        elif config.strategy == RerankStrategy.RERANKER_MMR:
            # 先用reranker重排，再用MMR增加多样性
            reranked = self._reranker_rerank(query, items, config.final_top_k * 2)
            return self._mmr_rerank(query, reranked, config.final_top_k, config.mmr_lambda)
        else:
            raise ValueError(f"不支持的重排策略: {config.strategy}")

    def _reranker_rerank(
        self,
        query: str,
        items: List[RetrievalItem],
        top_k: int
    ) -> List[RetrievalItem]:
        """使用Reranker模型重排"""
        if not REGISTRY.has_any_reranker():
            logger.warning("没有可用的reranker模型，返回原始结果")
            return items[:top_k]
        
        reranker = REGISTRY.get_reranker(None)
        if not reranker:
            return items[:top_k]
        
        # 提取文档内容
        documents = [item.document.content for item in items]
        
        # Reranker打分
        scores = reranker.rerank(query, documents)
        
        # 更新分数
        for item, score in zip(items, scores):
            item.score = float(score)
        
        # 按新分数排序
        items.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Reranker重排: {len(items)} -> {min(top_k, len(items))} 条结果")
        return items[:top_k]

    def _mmr_rerank(
        self,
        query: str,
        items: List[RetrievalItem],
        top_k: int,
        lambda_param: float = 0.5
    ) -> List[RetrievalItem]:
        """
        MMR (Maximal Marginal Relevance) 重排：在相关性和多样性之间平衡
        
        Args:
            query: 查询文本
            items: 待重排的结果
            top_k: 返回数量
            lambda_param: 平衡参数，0-1之间，越大越注重多样性
        """
        if len(items) <= top_k:
            return items
        
        # 需要embedding模型计算相似度
        if not REGISTRY.has_any_embedding():
            logger.warning("没有embedding模型，无法执行MMR，返回原始结果")
            return items[:top_k]
        
        embedding_model = REGISTRY.get_embedding(None)
        if not embedding_model:
            return items[:top_k]
        
        # 获取所有文档的embedding
        docs_text = [item.document.content for item in items]
        docs_embeddings = embedding_model.embed(docs_text)
        query_embedding = embedding_model.embed([query])[0]
        
        # 计算查询与文档的相似度
        from numpy import dot
        from numpy.linalg import norm
        
        def cosine_similarity(a, b):
            return dot(a, b) / (norm(a) * norm(b))
        
        # 初始化
        selected_indices = []
        remaining_indices = list(range(len(items)))
        
        # 迭代选择
        for _ in range(min(top_k, len(items))):
            if not remaining_indices:
                break
            
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # 与查询的相似度（相关性）
                relevance = cosine_similarity(query_embedding, docs_embeddings[idx])
                
                # 与已选文档的最大相似度（多样性惩罚）
                if selected_indices:
                    max_sim = max(
                        cosine_similarity(docs_embeddings[idx], docs_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim = 0
                
                # MMR分数
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # 按选择顺序返回结果
        result = [items[idx] for idx in selected_indices]
        
        logger.info(f"MMR重排(λ={lambda_param}): {len(items)} -> {len(result)} 条结果")
        return result

    def retrieve(
        self,
        query: str,
        collection_name: str,
        embedding_model: str,
        recall_config: Optional[RecallConfig] = None,
        rerank_config: Optional[RerankConfig] = None
    ) -> List[RetrievalItem]:
        """
        完整的检索流程：召回 -> 重排
        
        Args:
            query: 查询文本
            collection_name: 集合名称
            embedding_model: Embedding模型名称
            recall_config: 召回配置
            rerank_config: 重排配置
        
        Returns:
            最终检索结果
        """
        # 使用默认配置
        if recall_config is None:
            recall_config = RecallConfig()
        if rerank_config is None:
            rerank_config = RerankConfig()
        
        # 第一阶段：召回
        logger.info(f"开始检索: query='{query[:50]}...', collection={collection_name}")
        recalled_items = self.recall(query, collection_name, embedding_model, recall_config)
        logger.info(f"召回阶段完成: {len(recalled_items)} 条结果")
        
        if not recalled_items:
            return []
        
        # 第二阶段：重排
        reranked_items = self.rerank(query, recalled_items, rerank_config)
        logger.info(f"重排阶段完成: {len(reranked_items)} 条最终结果")
        
        return reranked_items


# 全局实例
ADVANCED_RETRIEVAL_SERVICE = AdvancedRetrievalService()

