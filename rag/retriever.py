"""
混合检索器模块

本模块提供知识库的混合检索能力，支持：
- Dense（稠密向量）语义检索
- Sparse（BM25）关键词检索
- Hybrid（混合）检索，融合语义和关键词结果

基于 Milvus 2.5+ 的原生混合检索能力实现。
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from loguru import logger

from models.kb_schemas import (
    KBChunk,
    KBSearchRequest,
    KBSearchResponse,
    KBSearchHit,
    SearchMode,
    ChunkPosition,
    ChunkOverlapInfo,
)
from storage.milvus.client import MilvusClient
from storage.milvus.collections import KBCollectionManager

# 延迟导入 pymilvus
_pymilvus_available = False
try:
    from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker
    _pymilvus_available = True
except ImportError:
    pass


@dataclass
class RetrieverConfig:
    """
    检索器配置。
    
    Attributes:
        default_top_k: 默认返回结果数量
        dense_weight: 混合检索中稠密向量的权重
        sparse_weight: 混合检索中稀疏向量的权重
        rrf_k: RRF 融合算法的 k 参数
        use_rrf: 是否使用 RRF 融合（否则使用加权融合）
        ef_search: HNSW 搜索时的 ef 参数
    """
    default_top_k: int = 10
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60
    use_rrf: bool = True
    ef_search: int = 64


class HybridRetriever:
    """
    混合检索器。
    
    提供知识库的多模式检索能力：
    - Dense: 基于稠密向量的语义检索，适合理解查询意图
    - Sparse: 基于 BM25 的关键词检索，适合精确匹配
    - Hybrid: 融合两种检索结果，兼顾语义理解和关键词匹配
    
    Usage:
        >>> retriever = HybridRetriever(embedding_fn=get_embedding)
        >>> response = retriever.search(KBSearchRequest(
        ...     query="如何配置网络",
        ...     mode=SearchMode.HYBRID,
        ...     top_k=10
        ... ))
    """
    
    def __init__(
        self,
        embedding_fn,
        collection_manager: Optional[KBCollectionManager] = None,
        config: Optional[RetrieverConfig] = None,
    ) -> None:
        """
        初始化混合检索器。
        
        Args:
            embedding_fn: 向量化函数，接收文本返回向量
                         签名: (text: str) -> List[float]
            collection_manager: Collection 管理器，为 None 时自动创建
            config: 检索器配置
        """
        self._embedding_fn = embedding_fn
        self._collection_manager = collection_manager or KBCollectionManager()
        self._config = config or RetrieverConfig()
    
    @property
    def config(self) -> RetrieverConfig:
        """获取检索器配置"""
        return self._config
    
    def search(self, request: KBSearchRequest) -> KBSearchResponse:
        """
        执行检索。
        
        根据请求中的 mode 参数选择检索策略：
        - DENSE: 仅语义检索
        - SPARSE: 仅关键词检索
        - HYBRID: 混合检索
        
        Args:
            request: 检索请求
            
        Returns:
            KBSearchResponse: 检索响应
        """
        start_time = time.time()
        
        try:
            if request.mode == SearchMode.DENSE:
                hits = self._dense_search(request)
            elif request.mode == SearchMode.SPARSE:
                hits = self._sparse_search(request)
            else:  # HYBRID
                hits = self._hybrid_search(request)
            
            took_ms = (time.time() - start_time) * 1000
            
            return KBSearchResponse(
                hits=hits,
                total=len(hits),
                query=request.query,
                mode=request.mode,
                took_ms=took_ms,
            )
            
        except Exception as e:
            logger.error("检索失败: {}", str(e))
            raise
    
    def _dense_search(self, request: KBSearchRequest) -> List[KBSearchHit]:
        """
        执行稠密向量检索。
        
        Args:
            request: 检索请求
            
        Returns:
            List[KBSearchHit]: 检索结果
        """
        # 获取查询向量
        query_vector = self._embedding_fn(request.query)
        
        # 构建过滤表达式
        filter_expr = self._build_filter_expr(request)
        
        # 获取 Collection（根据知识库名称）
        collection = self._collection_manager.get_collection(request.collection_name)
        
        # 执行搜索
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": self._config.ef_search}
        }
        
        results = collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=request.top_k,
            expr=filter_expr if filter_expr else None,
            output_fields=self._get_output_fields(),
        )
        
        return self._convert_results(results[0], score_field="dense_score")
    
    def _sparse_search(self, request: KBSearchRequest) -> List[KBSearchHit]:
        """
        执行稀疏向量（BM25）检索。
        
        Args:
            request: 检索请求
            
        Returns:
            List[KBSearchHit]: 检索结果
        """
        # 构建过滤表达式
        filter_expr = self._build_filter_expr(request)
        
        # 获取 Collection（根据知识库名称）
        collection = self._collection_manager.get_collection(request.collection_name)
        
        # BM25 搜索参数
        search_params = {
            "metric_type": "BM25",
            "params": {}
        }
        
        # 使用文本查询进行 BM25 搜索
        results = collection.search(
            data=[request.query],  # BM25 直接使用文本
            anns_field="sparse_vector",
            param=search_params,
            limit=request.top_k,
            expr=filter_expr if filter_expr else None,
            output_fields=self._get_output_fields(),
        )
        
        return self._convert_results(results[0], score_field="sparse_score")
    
    def _hybrid_search(self, request: KBSearchRequest) -> List[KBSearchHit]:
        """
        执行混合检索。
        
        同时执行 Dense 和 Sparse 检索，然后融合结果。
        
        Args:
            request: 检索请求
            
        Returns:
            List[KBSearchHit]: 检索结果
        """
        if not _pymilvus_available:
            raise RuntimeError("pymilvus 未安装，无法执行混合检索")
        
        # 获取查询向量
        query_vector = self._embedding_fn(request.query)
        
        # 构建过滤表达式
        filter_expr = self._build_filter_expr(request)
        
        # 获取 Collection（根据知识库名称）
        collection = self._collection_manager.get_collection(request.collection_name)
        
        # 构建 Dense 搜索请求
        dense_search_params = {
            "metric_type": "COSINE",
            "params": {"ef": self._config.ef_search}
        }
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=request.top_k,
            expr=filter_expr if filter_expr else None,
        )
        
        # 构建 Sparse 搜索请求
        sparse_search_params = {
            "metric_type": "BM25",
            "params": {}
        }
        sparse_req = AnnSearchRequest(
            data=[request.query],
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=request.top_k,
            expr=filter_expr if filter_expr else None,
        )
        
        # 选择融合策略
        if self._config.use_rrf:
            ranker = RRFRanker(k=self._config.rrf_k)
        else:
            ranker = WeightedRanker(
                self._config.dense_weight,
                self._config.sparse_weight
            )
        
        # 执行混合搜索
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=request.top_k,
            output_fields=self._get_output_fields(),
        )
        
        return self._convert_results(results[0])
    
    def _build_filter_expr(self, request: KBSearchRequest) -> Optional[str]:
        """
        构建 Milvus 过滤表达式。
        
        Args:
            request: 检索请求
            
        Returns:
            Optional[str]: 过滤表达式，无过滤条件时返回 None
        """
        conditions: List[str] = []
        
        # 按文档名称过滤
        if request.doc_name:
            conditions.append(f'doc_name == "{request.doc_name}"')
        
        # 按章节路径过滤（前缀匹配）
        if request.chapter_filter:
            conditions.append(f'chapter_path like "{request.chapter_filter}%"')
        
        if not conditions:
            return None
        
        return " and ".join(conditions)
    
    def _get_output_fields(self) -> List[str]:
        """获取需要返回的字段列表"""
        return [
            "id", "doc_id", "doc_name", "chapter", "chapter_path",
            "chunk_idx", "content", "start_pos", "end_pos",
            "overlap_prev", "overlap_next", "metadata", "created_at"
        ]
    
    def _convert_results(
        self, 
        results, 
        score_field: Optional[str] = None
    ) -> List[KBSearchHit]:
        """
        将 Milvus 搜索结果转换为 KBSearchHit 列表。
        
        Args:
            results: Milvus 搜索结果
            score_field: 分数字段名（dense_score/sparse_score）
            
        Returns:
            List[KBSearchHit]: 转换后的结果
        """
        hits: List[KBSearchHit] = []
        
        for hit in results:
            entity = hit.entity
            
            # 构建 KBChunk
            chunk = KBChunk(
                id=entity.get("id"),
                doc_id=entity.get("doc_id"),
                doc_name=entity.get("doc_name"),
                chapter=entity.get("chapter", ""),
                chapter_path=entity.get("chapter_path", ""),
                chunk_idx=entity.get("chunk_idx"),
                content=entity.get("content"),
                position=ChunkPosition(
                    start_pos=entity.get("start_pos", 0),
                    end_pos=entity.get("end_pos", 0)
                ),
                overlap=ChunkOverlapInfo(
                    prev_chars=entity.get("overlap_prev", 0),
                    next_chars=entity.get("overlap_next", 0)
                ),
                metadata=entity.get("metadata", {}),
                created_at=entity.get("created_at", 0),
            )
            
            # 构建 KBSearchHit
            search_hit = KBSearchHit(
                chunk=chunk,
                score=hit.score,
            )
            
            # 设置具体的分数字段
            if score_field == "dense_score":
                search_hit.dense_score = hit.score
            elif score_field == "sparse_score":
                search_hit.sparse_score = hit.score
            
            hits.append(search_hit)
        
        return hits
    
    def search_by_chapter(
        self,
        collection_name: str,
        doc_name: str,
        chapter_path: str,
        merge_content: bool = True,
    ) -> Dict[str, Any]:
        """
        按章节查询内容（内部方法）。
        
        用于处理类似 "请告诉我《产品手册》第三章的内容" 的查询。
        
        Args:
            collection_name: Milvus Collection 名称
            doc_name: 文档名称
            chapter_path: 章节路径（支持前缀匹配）
            merge_content: 是否合并切片内容
            
        Returns:
            Dict: 包含章节内容和切片信息
        """
        # 查询切片
        chunks_data = self._collection_manager.get_chunks_by_chapter(
            collection_name=collection_name,
            doc_name=doc_name,
            chapter_path=chapter_path,
        )
        
        if not chunks_data:
            return {
                "doc_name": doc_name,
                "chapter_path": chapter_path,
                "content": "",
                "chunks": [],
                "chunk_count": 0,
            }
        
        # 转换为 KBChunk 对象
        chunks = [
            KBChunk(
                id=data.get("id"),
                doc_id=data.get("doc_id"),
                doc_name=data.get("doc_name"),
                chapter=data.get("chapter", ""),
                chapter_path=data.get("chapter_path", ""),
                chunk_idx=data.get("chunk_idx"),
                content=data.get("content"),
                position=ChunkPosition(
                    start_pos=data.get("start_pos", 0),
                    end_pos=data.get("end_pos", 0)
                ),
                overlap=ChunkOverlapInfo(
                    prev_chars=data.get("overlap_prev", 0),
                    next_chars=data.get("overlap_next", 0)
                ),
                metadata=data.get("metadata", {}),
                created_at=data.get("created_at", 0),
            )
            for data in chunks_data
        ]
        
        # 合并内容（去除重叠部分）
        content = ""
        if merge_content and chunks:
            content = self._merge_chunks_content(chunks)
        
        return {
            "doc_name": doc_name,
            "chapter_path": chapter_path,
            "content": content,
            "chunks": chunks,
            "chunk_count": len(chunks),
        }
    
    @staticmethod
    def _merge_chunks_content(chunks: List[KBChunk]) -> str:
        """
        合并多个切片的内容，去除重叠部分。
        
        Args:
            chunks: 切片列表（需按 chunk_idx 排序）
            
        Returns:
            str: 合并后的内容
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0].content
        
        # 按 chunk_idx 排序
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_idx)
        
        result_parts: List[str] = []
        
        for i, chunk in enumerate(sorted_chunks):
            content = chunk.content
            
            if i > 0:
                # 去除与前一个切片的重叠部分
                overlap_chars = chunk.overlap.prev_chars
                if overlap_chars > 0 and overlap_chars < len(content):
                    content = content[overlap_chars:]
            
            result_parts.append(content)
        
        return "".join(result_parts)
