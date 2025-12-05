"""
高级检索API：提供多策略召回和重排功能

本模块提供企业级的高级检索能力，支持：
1. 多种召回策略：纯向量、混合召回、多向量召回
2. 多种重排策略：Reranker模型、MMR多样性、组合策略
3. 灵活的配置参数：召回数量、相似度阈值、多样性参数等

API端点：
- POST /v1/advanced-retrieval/search - 完整的高级检索（召回+重排）
- POST /v1/advanced-retrieval/recall-only - 仅召回，不重排
- POST /v1/advanced-retrieval/rerank-only - 仅重排，对已有结果重排

适用场景：
- 需要精确控制召回和重排流程的应用
- 需要平衡相关性和多样性的搜索场景
- 需要对比不同策略效果的实验场景
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

from models.schemas import RetrievalItem
from services.advanced_retrieval_service import (
    ADVANCED_RETRIEVAL_SERVICE,
    RecallStrategy,
    RerankStrategy,
    RecallConfig,
    RerankConfig
)


router = APIRouter(prefix="/v1/advanced-retrieval", tags=["advanced-retrieval"])


class RecallStrategyEnum(str, Enum):
    """
    召回策略枚举（用于API）
    
    - vector_only: 纯向量召回，使用单一embedding模型
    - hybrid: 混合召回，结合向量检索和关键词检索
    - multi_vector: 多向量召回，使用多个embedding模型融合结果
    """
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    MULTI_VECTOR = "multi_vector"


class RerankStrategyEnum(str, Enum):
    """
    重排策略枚举（用于API）
    
    - none: 不重排，直接返回召回结果
    - reranker: 使用Reranker模型重排，提升相关性
    - mmr: 使用MMR算法增加结果多样性，避免重复
    - reranker_mmr: 先用Reranker重排提升相关性，再用MMR增加多样性
    """
    NONE = "none"
    RERANKER = "reranker"
    MMR = "mmr"
    RERANKER_MMR = "reranker_mmr"


class AdvancedRetrievalRequest(BaseModel):
    """
    高级检索请求
    
    包含召回配置、重排配置和生成配置的完整参数
    """
    query: str = Field(..., description="查询文本")
    collection: str = Field(..., description="集合名称")
    embedding_model: str = Field(..., description="Embedding模型名称")
    
    # 召回配置
    recall_strategy: RecallStrategyEnum = Field(
        default=RecallStrategyEnum.VECTOR_ONLY,
        description="召回策略"
    )
    recall_top_k: int = Field(default=20, description="召回数量", ge=1, le=200)
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="COSINE相似度阈值（-1到1，1=完全相同，0=正交，-1=完全相反；实际常用0.7-1.0表示高相似度）",
        ge=-1.0,
        le=1.0
    )
    
    # 重排配置
    rerank_strategy: RerankStrategyEnum = Field(
        default=RerankStrategyEnum.RERANKER,
        description="重排策略"
    )
    final_top_k: int = Field(default=10, description="最终返回数量", ge=1, le=100)
    mmr_lambda: float = Field(
        default=0.5,
        description="MMR多样性参数(0-1，越大越注重多样性)",
        ge=0.0,
        le=1.0
    )


class AdvancedRetrievalResponse(BaseModel):
    """
    高级检索响应
    
    属性:
        query: 原始查询文本
        collection: 检索的集合名称
        embedding_model: 使用的embedding模型
        recall_strategy: 使用的召回策略
        rerank_strategy: 使用的重排策略
        items: 检索结果列表
        total_count: 结果总数
    """
    query: str
    collection: str
    embedding_model: str
    recall_strategy: str
    rerank_strategy: str
    items: List[RetrievalItem]
    total_count: int


@router.post("/search", response_model=AdvancedRetrievalResponse)
async def advanced_search(req: AdvancedRetrievalRequest):
    """
    高级检索：支持多种召回和重排策略的完整检索流程
    
    这是最强大的检索接口，提供了完整的召回+重排流程控制。
    
    召回策略说明：
    - vector_only: 纯向量召回，使用单一embedding模型进行语义检索
    - hybrid: 混合召回，结合向量检索和关键词检索（需要额外配置）
    - multi_vector: 多向量召回，使用多个embedding模型融合结果，提升召回率
    
    重排策略说明：
    - none: 不重排，直接返回召回结果
    - reranker: 使用Reranker模型对召回结果重排，提升精准度
    - mmr: 使用MMR(Maximal Marginal Relevance)算法增加结果多样性，减少冗余
    - reranker_mmr: 组合策略，先用Reranker提升相关性，再用MMR增加多样性
    
    参数:
        req: 高级检索请求，包含召回和重排的详细配置
        
    返回:
        AdvancedRetrievalResponse: 包含检索结果和元信息
        
    异常:
        400: 参数错误（无效的策略、模型不存在等）
        500: 检索失败
    """
    try:
        # 构建配置
        recall_config = RecallConfig(
            strategy=RecallStrategy(req.recall_strategy.value),
            top_k=req.recall_top_k,
            similarity_threshold=req.similarity_threshold,
            enable_dedup=True
        )
        
        rerank_config = RerankConfig(
            strategy=RerankStrategy(req.rerank_strategy.value),
            final_top_k=req.final_top_k,
            mmr_lambda=req.mmr_lambda
        )
        
        # 执行检索
        items = ADVANCED_RETRIEVAL_SERVICE.retrieve(
            query=req.query,
            collection_name=req.collection,
            embedding_model=req.embedding_model,
            recall_config=recall_config,
            rerank_config=rerank_config
        )
        
        return AdvancedRetrievalResponse(
            query=req.query,
            collection=req.collection,
            embedding_model=req.embedding_model,
            recall_strategy=req.recall_strategy.value,
            rerank_strategy=req.rerank_strategy.value,
            items=items,
            total_count=len(items)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@router.post("/recall-only", response_model=AdvancedRetrievalResponse)
async def recall_only(req: AdvancedRetrievalRequest):
    """
    仅召回：不进行重排，直接返回召回结果
    
    适用场景：
    - 需要获取原始召回结果进行分析
    - 自行实现重排逻辑
    - 对比不同召回策略的效果
    
    参数:
        req: 高级检索请求（仅使用召回相关参数）
        
    返回:
        AdvancedRetrievalResponse: 召回结果，rerank_strategy固定为"none"
        
    异常:
        400: 参数错误
        500: 召回失败
    """
    try:
        recall_config = RecallConfig(
            strategy=RecallStrategy(req.recall_strategy.value),
            top_k=req.recall_top_k,
            similarity_threshold=req.similarity_threshold,
            enable_dedup=True
        )
        
        items = ADVANCED_RETRIEVAL_SERVICE.recall(
            query=req.query,
            collection_name=req.collection,
            embedding_model=req.embedding_model,
            config=recall_config
        )
        
        return AdvancedRetrievalResponse(
            query=req.query,
            collection=req.collection,
            embedding_model=req.embedding_model,
            recall_strategy=req.recall_strategy.value,
            rerank_strategy="none",
            items=items,
            total_count=len(items)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"召回失败: {str(e)}")


class RerankOnlyRequest(BaseModel):
    """
    仅重排请求
    
    用于对已有的检索结果进行重排序，不执行召回过程。
    
    属性:
        query: 查询文本（用于计算相关性）
        items: 待重排的检索结果列表
        rerank_strategy: 重排策略（reranker/mmr/reranker_mmr）
        final_top_k: 重排后保留的结果数量
        mmr_lambda: MMR多样性参数(0-1)，越大越注重多样性
    """
    query: str = Field(..., description="查询文本")
    items: List[RetrievalItem] = Field(..., description="待重排的结果列表")
    rerank_strategy: RerankStrategyEnum = Field(
        default=RerankStrategyEnum.RERANKER,
        description="重排策略"
    )
    final_top_k: int = Field(default=10, description="最终返回数量", ge=1, le=100)
    mmr_lambda: float = Field(
        default=0.5,
        description="MMR多样性参数",
        ge=0.0,
        le=1.0
    )


@router.post("/rerank-only")
async def rerank_only(req: RerankOnlyRequest):
    """
    仅重排：对已有的结果进行重排序
    
    适用场景：
    - 已有召回结果，需要优化排序
    - 对比不同重排策略的效果
    - 实现分阶段的检索流程（先召回，后重排）
    
    工作流程：
    1. 接收已有的检索结果列表
    2. 根据指定的重排策略重新排序
    3. 返回top_k个结果
    
    参数:
        req: 仅重排请求，包含待重排的结果和重排配置
        
    返回:
        dict: 包含重排后的结果列表和元信息
        
    异常:
        400: 参数错误（无效的策略等）
        500: 重排失败
    """
    try:
        rerank_config = RerankConfig(
            strategy=RerankStrategy(req.rerank_strategy.value),
            final_top_k=req.final_top_k,
            mmr_lambda=req.mmr_lambda
        )
        
        reranked_items = ADVANCED_RETRIEVAL_SERVICE.rerank(
            query=req.query,
            items=req.items,
            config=rerank_config
        )
        
        return {
            "query": req.query,
            "rerank_strategy": req.rerank_strategy.value,
            "items": reranked_items,
            "total_count": len(reranked_items)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重排失败: {str(e)}")

