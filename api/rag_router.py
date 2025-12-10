"""
RAG 检索 API 路由

本模块提供检索 API：
- POST /v1/kb/search - 检索（支持混合检索）
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from models.kb_schemas import (
    KBSearchRequest,
    SearchMode,
    SearchRequest,
    SearchHitResponse,
    SearchResponse,
)
from services.container import CONTAINER

router = APIRouter(prefix="/v1/kb", tags=["Knowledge Base"])


def _get_rag_service():
    """获取 RAG 服务实例"""
    rag_service = CONTAINER.get_rag_service()
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 服务不可用，请检查 Milvus 连接和配置"
        )
    return rag_service


@router.post("/search", response_model=SearchResponse, summary="检索")
async def search(request: SearchRequest):
    """
    知识库检索。
    
    支持三种检索模式：
    - dense: 语义检索（基于向量相似度）
    - sparse: 关键词检索（基于 BM25）
    - hybrid: 混合检索（融合语义和关键词）
    
    Args:
        request: 检索请求
        
    Returns:
        SearchResponse: 检索结果
    """
    rag_service = _get_rag_service()
    
    try:
        # 转换检索模式
        mode_map = {
            "dense": SearchMode.DENSE,
            "sparse": SearchMode.SPARSE,
            "hybrid": SearchMode.HYBRID,
        }
        mode = mode_map.get(request.mode.lower(), SearchMode.HYBRID)
        
        # 构建内部请求
        search_request = KBSearchRequest(
            collection_name=request.collection_name,
            query=request.query,
            mode=mode,
            top_k=request.top_k,
            doc_name=request.doc_name,
            chapter_filter=request.chapter_filter,
            rerank=request.rerank,
            score_threshold=request.score_threshold,
        )
        
        # 执行检索
        response = rag_service.search(search_request)
        
        # 转换响应
        hits = [
            SearchHitResponse(
                id=hit.chunk.id,
                doc_name=hit.chunk.doc_name,
                chapter=hit.chunk.chapter,
                content=hit.chunk.content,
                score=hit.score,
            )
            for hit in response.hits
        ]
        
        return SearchResponse(
            hits=hits,
            total=response.total,
            query=response.query,
            mode=response.mode.value,
            took_ms=response.took_ms,
        )
        
    except Exception as e:
        logger.error("检索失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")
