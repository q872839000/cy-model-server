"""
RAG 业务服务模块

本模块提供 RAG 的核心业务服务，整合：
- 混合检索（Dense + Sparse/BM25）
- 重排序（Reranker）
- 结果后处理

作为 RAG 业务的统一入口，对外屏蔽底层实现细节。
"""

import time
from typing import Optional, List, Dict, Any, Callable
from loguru import logger

from models.kb_schemas import (
    KBSearchRequest,
    KBSearchResponse,
    KBSearchHit,
    SearchMode,
)
from storage.milvus.client import MilvusClient
from storage.milvus.collections import KBCollectionManager
from rag.retriever import HybridRetriever, RetrieverConfig


class RAGService:
    """
    RAG 业务服务。
    
    提供知识库检索的完整业务流程：
    1. 接收用户查询
    2. 执行混合检索（Dense + Sparse）
    3. 可选：重排序优化结果
    4. 返回格式化结果
    
    Attributes:
        retriever: 混合检索器
        reranker_fn: 重排序函数（可选）
        
    Usage:
        >>> service = RAGService(
        ...     embedding_fn=get_embedding,
        ...     reranker_fn=rerank_documents
        ... )
        >>> response = service.search(KBSearchRequest(query="如何配置"))
    """
    
    def __init__(
        self,
        embedding_fn: Callable[[str], List[float]],
        reranker_fn: Optional[Callable[[str, List[str]], List[Dict[str, Any]]]] = None,
        milvus_client: Optional[MilvusClient] = None,
        collection_manager: Optional[KBCollectionManager] = None,
        retriever_config: Optional[RetrieverConfig] = None,
    ) -> None:
        """
        初始化 RAG 服务。
        
        Args:
            embedding_fn: 向量化函数
                签名: (text: str) -> List[float]
            reranker_fn: 重排序函数（可选）
                签名: (query: str, documents: List[str]) -> List[Dict[str, Any]]
                返回: [{"index": int, "score": float}, ...]
            milvus_client: Milvus 客户端
            collection_manager: Collection 管理器
            retriever_config: 检索器配置
        """
        self._milvus_client = milvus_client or MilvusClient()
        self._collection_manager = collection_manager or KBCollectionManager(
            client=self._milvus_client
        )
        self._retriever = HybridRetriever(
            embedding_fn=embedding_fn,
            collection_manager=self._collection_manager,
            config=retriever_config,
        )
        self._embedding_fn = embedding_fn
        self._reranker_fn = reranker_fn
    
    @property
    def retriever(self) -> HybridRetriever:
        """获取检索器实例"""
        return self._retriever
    
    @property
    def collection_manager(self) -> KBCollectionManager:
        """获取 Collection 管理器"""
        return self._collection_manager
    
    def initialize(self) -> None:
        """
        初始化服务。
        
        验证 Milvus 连接状态。
        Milvus 连接在 main.py 启动时已建立。
        """
        if not self._milvus_client.is_connected:
            raise RuntimeError("Milvus 未连接，请检查服务配置")
        logger.info("RAG 服务就绪")
    
    def search(self, request: KBSearchRequest) -> KBSearchResponse:
        """
        执行知识库检索。
        
        流程：
        1. 调用混合检索器获取候选结果
        2. 如果启用重排序且有 reranker，执行重排序
        3. 应用分数阈值过滤
        4. 返回最终结果
        
        Args:
            request: 检索请求
            
        Returns:
            KBSearchResponse: 检索响应
        """
        start_time = time.time()
        
        # 计算实际需要检索的数量（重排序时需要更多候选）
        retrieve_top_k = request.top_k
        if request.rerank and self._reranker_fn:
            # 重排序时检索更多候选
            retrieve_top_k = min(request.top_k * 3, 100)
        
        # 创建检索请求（调整 top_k）
        retrieve_request = KBSearchRequest(
            collection_name=request.collection_name,
            query=request.query,
            mode=request.mode,
            top_k=retrieve_top_k,
            doc_name=request.doc_name,
            chapter_filter=request.chapter_filter,
            rerank=False,  # 检索阶段不重排
            score_threshold=None,  # 检索阶段不过滤
            metadata_filter=request.metadata_filter,
        )
        
        # 执行检索
        response = self._retriever.search(retrieve_request)
        hits = response.hits
        
        # 调试：显示检索阶段的结果
        logger.info("检索阶段返回 {} 条结果:", len(hits))
        for i, hit in enumerate(hits[:5]):
            logger.info("  [{}] {:.4f} | {}", i, hit.score, hit.chunk.chapter[:30])
        
        # 重排序
        if request.rerank and self._reranker_fn and hits:
            hits = self._rerank_hits(request.query, hits)
        
        # 截取 top_k
        final_top_k = request.rerank_top_k or request.top_k
        hits = hits[:final_top_k]
        
        # 应用分数阈值
        if request.score_threshold is not None:
            hits = [h for h in hits if h.score >= request.score_threshold]
        
        took_ms = (time.time() - start_time) * 1000
        
        return KBSearchResponse(
            hits=hits,
            total=len(hits),
            query=request.query,
            mode=request.mode,
            took_ms=took_ms,
        )
    
    def _rerank_hits(
        self, 
        query: str, 
        hits: List[KBSearchHit]
    ) -> List[KBSearchHit]:
        """
        对检索结果进行重排序。
        
        Args:
            query: 查询文本
            hits: 检索结果
            
        Returns:
            List[KBSearchHit]: 重排序后的结果
        """
        if not self._reranker_fn or not hits:
            return hits
        
        try:
            # 提取文档内容（拼接标题增强匹配）
            documents = []
            for hit in hits:
                # 将章节标题拼接到内容前面，增强关键词和语义匹配
                title = hit.chunk.chapter or ""
                content = hit.chunk.content or ""
                if title:
                    documents.append(f"【{title}】\n{content}")
                else:
                    documents.append(content)
            
            # 调用重排序
            rerank_results = self._reranker_fn(query, documents)
            
            # 调试日志：显示重排序前后的分数
            logger.debug("重排序 query: {}", query[:50])
            for i, (hit, result) in enumerate(zip(hits, rerank_results)):
                logger.debug("  [{}] {} -> {:.4f} | {}", 
                           i, hit.chunk.chapter[:20], result.get("score", 0), hit.chunk.content[:30])
            
            # 按重排序分数重新排列
            reranked_hits: List[KBSearchHit] = []
            for result in rerank_results:
                idx = result.get("index", 0)
                score = result.get("score", 0.0)
                
                if 0 <= idx < len(hits):
                    hit = hits[idx]
                    # 更新分数
                    reranked_hit = KBSearchHit(
                        chunk=hit.chunk,
                        score=score,
                        dense_score=hit.dense_score,
                        sparse_score=hit.sparse_score,
                        rerank_score=score,
                    )
                    reranked_hits.append(reranked_hit)
            
            # 按分数降序排列
            reranked_hits.sort(key=lambda x: x.score, reverse=True)
            
            logger.info("重排序完成: {} 条结果，最高分: {:.4f}", 
                       len(reranked_hits), 
                       reranked_hits[0].score if reranked_hits else 0)
            return reranked_hits
            
        except Exception as e:
            logger.warning("重排序失败，返回原始结果: {}", str(e))
            return hits
    
