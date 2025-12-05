"""
RAG API：检索增强生成接口（Retrieval-Augmented Generation）

本模块提供完整的RAG流程，将检索和生成无缝结合：
1. 自动检索相关知识
2. 构建包含上下文的提示词
3. 使用LLM生成基于知识的回答
4. 返回答案和引用来源

主要特性：
- 多种RAG策略：简单问答、带引用、对话式、逐步精炼
- 灵活的提示词模板：专业、友好、简洁、详细等风格
- 可配置的检索参数：召回策略、重排策略、相似度阈值等
- 支持流式输出：实时返回生成的内容
- 自动引用管理：追踪答案的知识来源

API端点：
- POST /v1/rag/generate - RAG生成（非流式）
- POST /v1/rag/generate-stream - RAG生成（流式）
- GET /v1/rag/health - 健康检查

适用场景：
- 知识库问答系统
- 文档智能助手
- 客服机器人
- 企业知识管理
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import orjson

from services.rag_service import (
    RAG_SERVICE,
    RAGStrategy,
    PromptTemplate,
    RAGConfig
)
from services.advanced_retrieval_service import (
    RecallStrategy,
    RerankStrategy,
    RecallConfig,
    RerankConfig
)


router = APIRouter(prefix="/v1/rag", tags=["rag"])


class RAGStrategyEnum(str, Enum):
    """
    RAG策略枚举
    
    - simple: 简单问答，直接基于检索结果生成答案
    - citation: 带引用问答，答案中标注知识来源
    - conversational: 对话式问答，考虑上下文的多轮对话
    - refine: 逐步精炼，多次迭代优化答案质量
    """
    SIMPLE = "simple"
    CITATION = "citation"
    CONVERSATIONAL = "conversational"
    REFINE = "refine"


class PromptTemplateEnum(str, Enum):
    """
    提示词模板枚举
    
    - default: 默认模板，平衡专业性和可读性
    - professional: 专业模板，正式、严谨的表达
    - friendly: 友好模板，亲切、易懂的表达
    - concise: 简洁模板，简明扼要的回答
    - detailed: 详细模板，全面、深入的解释
    """
    DEFAULT = "default"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CONCISE = "concise"
    DETAILED = "detailed"


class RecallStrategyEnum(str, Enum):
    """
    召回策略枚举
    
    - vector_only: 纯向量召回
    - hybrid: 混合召回（向量+关键词）
    - multi_vector: 多向量召回（多模型融合）
    """
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    MULTI_VECTOR = "multi_vector"


class RerankStrategyEnum(str, Enum):
    """
    重排策略枚举
    
    - none: 不重排
    - reranker: 使用Reranker模型重排
    - mmr: 使用MMR算法增加多样性
    - reranker_mmr: Reranker + MMR组合策略
    """
    NONE = "none"
    RERANKER = "reranker"
    MMR = "mmr"
    RERANKER_MMR = "reranker_mmr"


class RAGRequest(BaseModel):
    """
    RAG请求
    
    包含完整的RAG流程配置：检索、生成、提示词等
    """
    # 必填参数
    query: str = Field(..., description="用户问题")
    collection: str = Field(..., description="知识库集合名称")
    model: str = Field(..., description="LLM模型名称")
    embedding_model: str = Field(..., description="Embedding模型名称")
    
    # RAG配置
    rag_strategy: RAGStrategyEnum = Field(
        default=RAGStrategyEnum.SIMPLE,
        description="RAG策略"
    )
    prompt_template: PromptTemplateEnum = Field(
        default=PromptTemplateEnum.DEFAULT,
        description="提示词模板"
    )
    max_context_length: int = Field(
        default=3000,
        description="最大上下文长度",
        ge=500,
        le=8000
    )
    include_citations: bool = Field(
        default=True,
        description="是否包含引用来源"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="自定义系统提示（可选）"
    )
    
    # 召回配置
    recall_strategy: RecallStrategyEnum = Field(
        default=RecallStrategyEnum.VECTOR_ONLY,
        description="召回策略"
    )
    recall_top_k: int = Field(
        default=10,
        description="召回数量",
        ge=1,
        le=50
    )
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
    rerank_top_k: int = Field(
        default=5,
        description="重排后保留数量",
        ge=1,
        le=20
    )
    mmr_lambda: float = Field(
        default=0.5,
        description="MMR多样性参数",
        ge=0.0,
        le=1.0
    )
    
    # 生成配置
    max_tokens: int = Field(default=512, description="最大生成token数", ge=1, le=4096)
    temperature: float = Field(default=0.7, description="生成温度", ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, description="Top-p采样", ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="是否流式输出")


class SourceInfo(BaseModel):
    """
    来源信息
    
    记录答案的知识来源，用于引用和溯源
    
    属性:
        index: 来源序号
        content: 来源文档内容
        score: 相关性分数
        metadata: 文档元数据（标题、作者、时间等）
        document_id: 文档唯一标识
    """
    index: int
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    document_id: str


class RAGResponse(BaseModel):
    """
    RAG响应
    
    属性:
        id: 请求唯一标识
        query: 原始问题
        answer: 生成的答案
        sources: 引用的知识来源列表
        retrieved_count: 检索到的文档数量
        model: 使用的LLM模型
        collection: 检索的知识库集合
    """
    id: str
    query: str
    answer: str
    sources: List[SourceInfo]
    retrieved_count: int
    model: str
    collection: str


@router.post("/generate", response_model=RAGResponse)
async def rag_generate(req: RAGRequest):
    """
    RAG生成：基于知识库回答问题（非流式）
    
    这是最常用的RAG接口，提供完整的检索增强生成流程。
    
    工作流程：
    1. 使用embedding模型对问题进行向量化
    2. 在指定集合中检索相关文档（支持多种召回策略）
    3. 对检索结果进行重排序（可选）
    4. 构建包含上下文的提示词（根据选择的模板和策略）
    5. 使用LLM生成答案
    6. 返回答案和来源引用
    
    与直接使用LLM的区别：
    - 答案基于知识库的实际内容，而非模型的训练数据
    - 可以追溯答案来源，增强可信度
    - 支持私有知识库，不泄露给公共模型
    - 可以实时更新知识，无需重新训练模型
    
    参数:
        req: RAG请求，包含问题、知识库、模型选择和各种配置
        
    返回:
        RAGResponse: 包含答案、来源引用和元信息
        
    异常:
        400: 参数错误（模型不存在、集合不存在、流式标志设置错误等）
        500: 生成失败
    """
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="流式输出请使用 /v1/rag/generate-stream 端点"
        )
    
    try:
        # 构建配置
        rag_config = RAGConfig(
            strategy=RAGStrategy(req.rag_strategy.value),
            prompt_template=PromptTemplate(req.prompt_template.value),
            max_context_length=req.max_context_length,
            include_citations=req.include_citations,
            system_prompt=req.system_prompt
        )
        
        recall_config = RecallConfig(
            strategy=RecallStrategy(req.recall_strategy.value),
            top_k=req.recall_top_k,
            similarity_threshold=req.similarity_threshold
        )
        
        rerank_config = RerankConfig(
            strategy=RerankStrategy(req.rerank_strategy.value),
            final_top_k=req.rerank_top_k,
            mmr_lambda=req.mmr_lambda
        )
        
        generation_params = {
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p
        }
        
        # 执行RAG生成
        result = RAG_SERVICE.generate(
            query=req.query,
            collection_name=req.collection,
            model_name=req.model,
            embedding_model=req.embedding_model,
            rag_config=rag_config,
            recall_config=recall_config,
            rerank_config=rerank_config,
            generation_params=generation_params
        )
        
        # 构建响应
        return RAGResponse(
            id=f"rag-{uuid.uuid4().hex[:8]}",
            query=result["query"],
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            retrieved_count=result["retrieved_count"],
            model=req.model,
            collection=req.collection
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG生成失败: {str(e)}")


@router.post("/generate-stream")
async def rag_generate_stream(req: RAGRequest):
    """
    RAG流式生成：基于知识库回答问题（流式输出）
    
    与非流式版本功能相同，但采用流式输出，用户可以实时看到生成过程。
    
    适用场景：
    - 需要快速响应的交互式应用
    - 生成较长答案时提升用户体验
    - 实时展示生成进度
    
    返回格式：Server-Sent Events (SSE)
    事件类型：
    - start: 开始事件，包含请求元信息
    - chunk: 内容块，包含生成的文本片段
    - done: 结束事件
    - error: 错误事件（如果生成失败）
    
    参数:
        req: RAG请求
        
    返回:
        StreamingResponse: SSE格式的流式响应
        
    异常:
        400: 参数错误
        500: 生成失败（会在流中发送error事件）
    """
    try:
        # 构建配置
        rag_config = RAGConfig(
            strategy=RAGStrategy(req.rag_strategy.value),
            prompt_template=PromptTemplate(req.prompt_template.value),
            max_context_length=req.max_context_length,
            include_citations=req.include_citations,
            system_prompt=req.system_prompt
        )
        
        recall_config = RecallConfig(
            strategy=RecallStrategy(req.recall_strategy.value),
            top_k=req.recall_top_k,
            similarity_threshold=req.similarity_threshold
        )
        
        rerank_config = RerankConfig(
            strategy=RerankStrategy(req.rerank_strategy.value),
            final_top_k=req.rerank_top_k,
            mmr_lambda=req.mmr_lambda
        )
        
        generation_params = {
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p
        }
        
        # 流式生成
        async def event_generator():
            chunk_id = f"rag-{uuid.uuid4().hex[:8]}"
            created = int(__import__('time').time())
            
            # 发送开始事件
            start_event = {
                "id": chunk_id,
                "type": "start",
                "created": created,
                "query": req.query,
                "model": req.model,
                "collection": req.collection
            }
            yield f"data: {orjson.dumps(start_event).decode()}\n\n"
            
            # 流式生成答案
            try:
                for text_chunk in RAG_SERVICE.generate_stream(
                    query=req.query,
                    collection_name=req.collection,
                    model_name=req.model,
                    embedding_model=req.embedding_model,
                    rag_config=rag_config,
                    recall_config=recall_config,
                    rerank_config=rerank_config,
                    generation_params=generation_params
                ):
                    chunk_event = {
                        "id": chunk_id,
                        "type": "chunk",
                        "content": text_chunk
                    }
                    yield f"data: {orjson.dumps(chunk_event).decode()}\n\n"
            except Exception as e:
                error_event = {
                    "id": chunk_id,
                    "type": "error",
                    "error": str(e)
                }
                yield f"data: {orjson.dumps(error_event).decode()}\n\n"
                raise
            
            # 发送结束事件
            done_event = {
                "id": chunk_id,
                "type": "done"
            }
            yield f"data: {orjson.dumps(done_event).decode()}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG流式生成失败: {str(e)}")


@router.get("/health")
async def rag_health():
    """
    检查RAG服务健康状态
    
    检查RAG服务所需的各个组件是否就绪：
    - LLM模型（生成答案）
    - Embedding模型（向量化查询）
    - Reranker模型（重排序，可选）
    
    返回:
        dict: 包含健康状态和已加载的模型列表
    """
    from core.registry import REGISTRY
    
    return {
        "status": "ok",
        "has_llm": REGISTRY.has_any_llm(),
        "has_embedding": REGISTRY.has_any_embedding(),
        "has_reranker": REGISTRY.has_any_reranker(),
        "llm_models": list(REGISTRY._llms.keys()),
        "embedding_models": list(REGISTRY._embeddings.keys()),
        "reranker_models": list(REGISTRY._rerankers.keys())
    }

