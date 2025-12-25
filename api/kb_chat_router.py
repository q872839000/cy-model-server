"""
知识库对话 API 路由

本模块提供知识库对话 API：
- POST /v1/kb/chat/completions - 知识库对话（支持流式）

兼容 OpenAI Chat Completions API 格式，增加知识库特有字段。
"""

import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from models import (
    KBChatRequest,
    KBChatResponse,
    KBChatChoice,
    KBChatMessageResponse,
    KBChatInfo,
    KBSourceReference,
    KBChatDelta,
    KBChatChunkChoice,
    KBChatChunkResponse,
)
from core.exceptions import ModelServerException
from rag.kb_chat import KBChatService
from utils.message_filter import clean_messages_for_history
from core.container import CONTAINER
from workers.model_worker import WORKER


router = APIRouter(prefix="/v1/kb", tags=["Knowledge Base Chat"])


# ==================== 服务获取 ====================

# 缓存的 KBChatService 实例
_kb_chat_service: Optional[KBChatService] = None


def _get_kb_chat_service() -> KBChatService:
    """
    获取知识库对话服务实例
    
    使用全局缓存避免重复初始化服务。如果RAG服务不可用，
    则抛出HTTP 503错误。
    
    返回:
        KBChatService: 知识库对话服务实例
        
    异常:
        HTTPException: 当RAG服务不可用时抛出503错误
    """
    global _kb_chat_service
    
    if _kb_chat_service is not None:
        return _kb_chat_service
    
    # 获取 RAG 服务
    rag_service = CONTAINER.get_rag_service()
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG服务不可用，请检查Milvus连接和配置"
        )
    
    # 创建 LLM 调用函数
    async def llm_fn(model: str, messages: List[dict], max_tokens: int, temperature: float, top_p: float = 0.95, enable_thinking: bool = True, **kwargs) -> str:
        """
        非流式LLM调用函数
        
        封装WORKER.generate_chat为知识库对话服务提供统一的LLM接口
        
        参数:
            model: 模型名称
            messages: 对话消息列表
            max_tokens: 最大生成token数
            temperature: 生成温度
            top_p: 核采样参数
            enable_thinking: 是否启用深度思考模式
            
        返回:
            生成的文本内容
        """
        return await WORKER.generate_chat(
            model_name=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            enable_thinking=enable_thinking,
        )
    
    async def llm_stream_fn(model: str, messages: List[dict], max_tokens: int, temperature: float, top_p: float = 0.95, enable_thinking: bool = True, **kwargs):
        """
        流式LLM调用函数
        
        封装WORKER.generate_chat的流式调用为知识库对话服务提供统一接口
        
        参数:
            model: 模型名称
            messages: 对话消息列表
            max_tokens: 最大生成token数
            temperature: 生成温度
            top_p: 核采样参数
            enable_thinking: 是否启用深度思考模式
            
        生成:
            文本块迭代器
        """
        async for chunk in WORKER.generate_chat(
            model_name=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            enable_thinking=enable_thinking,
        ):
            yield chunk
    
    # 创建服务（使用全局配置管理器）
    _kb_chat_service = KBChatService(
        rag_service=rag_service,
        llm_fn=llm_fn,
        llm_stream_fn=llm_stream_fn,
        # config 参数为 None，服务内部会自动使用全局配置管理器
    )
    
    logger.info("知识库对话服务初始化完成")
    return _kb_chat_service


# ==================== API 端点 ====================

@router.post(
    "/chat/completions",
    response_model=KBChatResponse,
    summary="知识库对话",
    description="基于知识库的智能对话，支持多轮对话、Query改写、引用溯源",
)
async def kb_chat_completions(request: KBChatRequest) -> KBChatResponse | StreamingResponse:
    """
    知识库对话 API（兼容OpenAI格式）
    
    基于知识库的智能对话，集成RAG检索和大模型生成。
    支持意图识别、查询改写、混合检索、重排序等高级功能。
    
    处理流程：
    1. 意图识别：判断是否需要检索知识库
    2. Query改写：多轮对话下优化检索查询
    3. RAG检索：混合检索策略 + 重排序优化
    4. Prompt组装：根据模型类型智能组装上下文
    5. LLM生成：调用大模型生成回答
    
    参数:
        request: 知识库对话请求对象，包含模型、消息、检索参数等
        
    返回:
        非流式: KBChatResponse - JSON格式的完整响应，包含回答和知识库信息
        流式: StreamingResponse - SSE格式的流式响应，逐步返回生成内容
        
    异常:
        HTTPException: 当对话失败时抛出500错误
    """
    try:
        # 获取知识库对话服务实例
        service = _get_kb_chat_service()
        
        # 转换消息格式为Python字典，并过滤历史消息中的思考内容
        messages = clean_messages_for_history([msg.model_dump() for msg in request.messages])
        
        if not request.stream:
            # 非流式
            result = await service.chat(
                model=request.model,
                collection_name=request.collection_name,
                messages=messages,
                search_enabled=request.search_enabled,
                search_mode=request.search_mode,
                search_top_k=request.search_top_k,
                rerank=request.rerank,
                score_threshold=request.score_threshold,
                query_rewrite=request.query_rewrite,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                session_id=request.session_id,
                return_sources=request.return_sources,
                enable_thinking=request.enable_thinking,
            )
            
            # 构建响应
            return KBChatResponse(
                id=f"kbchat-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    KBChatChoice(
                        message=KBChatMessageResponse(content=result.content)
                    )
                ],
                kb_info=KBChatInfo(
                    search_performed=result.kb_info.search_performed,
                    search_query=result.kb_info.search_query,
                    query_rewritten=result.kb_info.query_rewritten,
                    original_query=result.kb_info.original_query,
                    intent=result.kb_info.intent,
                    sources=[
                        KBSourceReference(
                            index=s.index,
                            doc_name=s.doc_name,
                            chapter=s.chapter,
                            content_preview=s.content_preview,
                            score=s.score,
                        )
                        for s in result.kb_info.sources
                    ],
                    search_took_ms=result.kb_info.search_took_ms,
                ),
            )
        
        # 流式：返回SSE流式响应
        return StreamingResponse(
            _generate_stream(service, request, messages),
            media_type="text/event-stream",
        )
        
    except ModelServerException:
        # 让全局异常处理器处理
        raise
    except Exception as e:
        logger.error("知识库对话失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")


async def _generate_stream(
    service: KBChatService,
    request: KBChatRequest,
    messages: List[dict],
):
    """
    生成知识库对话的流式响应
    
    使用Server-Sent Events (SSE)格式输出，兼容OpenAI流式格式。
    支持搜索状态推送、内容流式生成和错误处理。
    
    参数:
        service: 知识库对话服务实例
        request: 知识库对话请求对象
        messages: 已转换的消息列表
        
    生成:
        SSE数据流，包含搜索状态、内容块和结束标记
    """
    import orjson
    
    chat_id = f"kbchat-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    def make_chunk(delta: KBChatDelta, finish_reason: Optional[str] = None, kb_info: Optional[KBChatInfo] = None) -> str:
        """
        构造SSE数据块
        
        使用Pydantic模型构造标准化的流式响应块，确保数据结构的一致性
        
        参数:
            delta: 增量内容对象
            finish_reason: 结束原因（可选）
            kb_info: 知识库相关信息（可选）
            
        返回:
            JSON格式的SSE数据块字符串
        """
        chunk = KBChatChunkResponse(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[KBChatChunkChoice(
                delta=delta,
                finish_reason=finish_reason,
            )],
            kb_info=kb_info,
        )
        return orjson.dumps(chunk.model_dump(exclude_none=True)).decode()
    
    try:
        async for event in service.chat_stream(
            model=request.model,
            collection_name=request.collection_name,
            messages=messages,
            search_enabled=request.search_enabled,
            search_mode=request.search_mode,
            search_top_k=request.search_top_k,
            rerank=request.rerank,
            score_threshold=request.score_threshold,
            query_rewrite=request.query_rewrite,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            session_id=request.session_id,
            return_sources=request.return_sources,
            enable_thinking=request.enable_thinking,
        ):
            event_type = event.get("type")
            event_data = event.get("data", {})
            
            if event_type == "search_start":
                # 可选：通知客户端开始检索
                pass
            
            elif event_type == "search_done":
                # 发送检索结果信息，包含初始角色标识
                kb_info_obj = None
                if event_data:
                    try:
                        # 为缺失的必填字段提供默认值
                        kb_data = {
                            "search_performed": True,  # 既然有search_done事件，说明进行了检索
                            "intent": "search",  # 默认意图为检索
                            **event_data  # 使用event_data的其他字段
                        }
                        kb_info_obj = KBChatInfo(**kb_data)
                    except Exception as e:
                        logger.warning("KBChatInfo构造失败: {}, event_data: {}", str(e), event_data)
                        # 构造失败时不传递kb_info
                        kb_info_obj = None
                yield f"data: {make_chunk(KBChatDelta(role='assistant'), kb_info=kb_info_obj)}\n\n"
            
            elif event_type == "content":
                # 发送内容片段，跳过空内容避免无意义的空包
                text = event_data.get("text", "")
                if text:  # 只有非空内容才发送
                    yield f"data: {make_chunk(KBChatDelta(content=text))}\n\n"
            
            elif event_type == "done":
                # 发送结束标记
                yield f"data: {make_chunk(KBChatDelta(), 'stop')}\n\n"
                yield "data: [DONE]\n\n"
            
            elif event_type == "error":
                # 发送错误信息
                error_msg = event_data.get("message", "Unknown error")
                yield f"data: {make_chunk(KBChatDelta(content=f'[错误: {error_msg}]'), 'error')}\n\n"
                yield "data: [DONE]\n\n"
                
    except Exception as e:
        logger.error("流式对话失败: {}", str(e))
        # 对Pydantic验证错误提供更友好的错误信息
        if "validation error" in str(e).lower():
            error_msg = "数据校验失败，请检查请求参数"
        else:
            error_msg = str(e)
        yield f"data: {make_chunk(KBChatDelta(content=f'[错误: {error_msg}]'), 'error')}\n\n"
        yield "data: [DONE]\n\n"
