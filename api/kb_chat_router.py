"""
知识库对话 API 路由

本模块提供知识库对话 API：
- POST /v1/kb/chat/completions - 知识库对话（支持流式）

兼容 OpenAI Chat Completions API 格式，增加知识库特有字段。
"""

import time
import uuid
from typing import Any, List, Optional

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
from utils.thinking_parser import parse_thinking_content, ThinkingStreamSplitter
from utils.chat_logger import save_chat_log
from core.container import CONTAINER


router = APIRouter(prefix="/v1/kb", tags=["Knowledge Base Chat"])


# ==================== 服务获取 ====================


def _get_kb_chat_service() -> KBChatService:
    """
    获取知识库对话服务实例
    
    通过 CONTAINER 统一管理服务生命周期。
    
    返回:
        KBChatService: 知识库对话服务实例
        
    异常:
        HTTPException: 当服务不可用时抛出503错误
    """
    service = CONTAINER.get_kb_chat_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="知识库对话服务不可用，请检查Milvus连接和配置"
        )
    return service


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
        raw_request = request.model_dump(mode="json", exclude_none=False)
        
        # 转换消息格式为Python字典，并过滤历史消息中的思考内容
        messages = clean_messages_for_history([msg.model_dump() for msg in request.messages])

        request_payload = {
            "protocol": "kb_chat_completions",
            "raw": raw_request,
            "normalized": {
                "messages": messages,
                "params": {
                    "collection_name": request.collection_name,
                    "search_enabled": request.search_enabled,
                    "search_mode": request.search_mode,
                    "search_top_k": request.search_top_k,
                    "rerank": request.rerank,
                    "score_threshold": request.score_threshold,
                    "query_rewrite": request.query_rewrite,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "session_id": request.session_id,
                    "return_sources": request.return_sources,
                    "enable_thinking": request.enable_thinking,
                    "stream": request.stream,
                },
            },
        }
        
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
            
            # 拆分 <think> 标签：reasoning_content 和 content 分离
            reasoning_content, content = parse_thinking_content(result.content)

            # 构建响应
            response = KBChatResponse(
                id=f"kbchat-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    KBChatChoice(
                        message=KBChatMessageResponse(
                            content=content,
                            reasoning_content=reasoning_content,
                        )
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

            await save_chat_log(
                model=request.model,
                params={
                    "collection_name": request.collection_name,
                    "search_enabled": request.search_enabled,
                    "search_mode": request.search_mode,
                    "search_top_k": request.search_top_k,
                    "rerank": request.rerank,
                    "score_threshold": request.score_threshold,
                    "query_rewrite": request.query_rewrite,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "session_id": request.session_id,
                    "return_sources": request.return_sources,
                    "enable_thinking": request.enable_thinking,
                    "stream": False,
                    "api": "kb_chat",
                },
                messages=messages,
                assistant_content=result.content,
                request_payload=request_payload,
                response_payload=response.model_dump(mode="json", exclude_none=False),
            )

            return response
        
        # 流式：返回SSE流式响应
        return StreamingResponse(
            _generate_stream(service, request, messages, request_payload),
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
    request_payload: dict[str, Any],
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
    stream_events: list[dict[str, Any]] = []
    full_text = ""
    kb_info_payload: dict[str, Any] | None = None
    
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
        payload = chunk.model_dump(exclude_none=True)
        stream_events.append(payload)
        return orjson.dumps(payload).decode()
    
    # 流式 <think> 标签拆分器
    splitter = ThinkingStreamSplitter()

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
                if kb_info_obj is not None:
                    kb_info_payload = kb_info_obj.model_dump(exclude_none=True)
                yield f"data: {make_chunk(KBChatDelta(role='assistant'), kb_info=kb_info_obj)}\n\n"
            
            elif event_type == "content":
                # 通过状态机拆分：thinking 阶段→reasoning_content，正文阶段→content
                text = event_data.get("text", "")
                if text:
                    full_text += text
                    for field, split_text in splitter.feed(text):
                        delta = KBChatDelta(
                            reasoning_content=split_text if field == "reasoning_content" else None,
                            content=split_text if field == "content" else None,
                        )
                        yield f"data: {make_chunk(delta)}\n\n"
            
            elif event_type == "done":
                # 刷新 splitter 缓冲区中的剩余内容
                for field, split_text in splitter.flush():
                    delta = KBChatDelta(
                        reasoning_content=split_text if field == "reasoning_content" else None,
                        content=split_text if field == "content" else None,
                    )
                    yield f"data: {make_chunk(delta)}\n\n"
                # 发送结束标记
                yield f"data: {make_chunk(KBChatDelta(), 'stop')}\n\n"
                stream_events.append({"data": "[DONE]"})
                yield "data: [DONE]\n\n"

                reasoning_content, content = parse_thinking_content(full_text)

                await save_chat_log(
                    model=request.model,
                    params={
                        "collection_name": request.collection_name,
                        "search_enabled": request.search_enabled,
                        "search_mode": request.search_mode,
                        "search_top_k": request.search_top_k,
                        "rerank": request.rerank,
                        "score_threshold": request.score_threshold,
                        "query_rewrite": request.query_rewrite,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "session_id": request.session_id,
                        "return_sources": request.return_sources,
                        "enable_thinking": request.enable_thinking,
                        "stream": True,
                        "api": "kb_chat",
                    },
                    messages=messages,
                    assistant_content=full_text,
                    request_payload=request_payload,
                    response_payload={
                        "protocol": "kb_chat_completions_stream",
                        "stream": True,
                        "finish_reason": "stop",
                        "assistant": {
                            "content": content,
                            "reasoning_content": reasoning_content,
                        },
                        "kb_info": kb_info_payload,
                        "events": stream_events,
                    },
                )
                break
            
            elif event_type == "error":
                # 发送错误信息
                error_msg = event_data.get("message", "Unknown error")
                yield f"data: {make_chunk(KBChatDelta(content=f'[错误: {error_msg}]'), 'error')}\n\n"
                stream_events.append({"data": "[DONE]"})
                yield "data: [DONE]\n\n"

                reasoning_content, content = parse_thinking_content(full_text)

                await save_chat_log(
                    model=request.model,
                    params={
                        "collection_name": request.collection_name,
                        "search_enabled": request.search_enabled,
                        "search_mode": request.search_mode,
                        "search_top_k": request.search_top_k,
                        "rerank": request.rerank,
                        "score_threshold": request.score_threshold,
                        "query_rewrite": request.query_rewrite,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "session_id": request.session_id,
                        "return_sources": request.return_sources,
                        "enable_thinking": request.enable_thinking,
                        "stream": True,
                        "api": "kb_chat",
                    },
                    messages=messages,
                    assistant_content=full_text,
                    request_payload=request_payload,
                    response_payload={
                        "protocol": "kb_chat_completions_stream",
                        "stream": True,
                        "finish_reason": "error",
                        "assistant": {
                            "content": content,
                            "reasoning_content": reasoning_content,
                        },
                        "kb_info": kb_info_payload,
                        "events": stream_events,
                    },
                    error={"message": error_msg},
                )
                break
                
    except Exception as e:
        logger.error("流式对话失败: {}", str(e))
        # 对Pydantic验证错误提供更友好的错误信息
        if "validation error" in str(e).lower():
            error_msg = "数据校验失败，请检查请求参数"
        else:
            error_msg = str(e)
        yield f"data: {make_chunk(KBChatDelta(content=f'[错误: {error_msg}]'), 'error')}\n\n"
        stream_events.append({"data": "[DONE]"})
        yield "data: [DONE]\n\n"

        reasoning_content, content = parse_thinking_content(full_text)

        await save_chat_log(
            model=request.model,
            params={
                "collection_name": request.collection_name,
                "search_enabled": request.search_enabled,
                "search_mode": request.search_mode,
                "search_top_k": request.search_top_k,
                "rerank": request.rerank,
                "score_threshold": request.score_threshold,
                "query_rewrite": request.query_rewrite,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "session_id": request.session_id,
                "return_sources": request.return_sources,
                "enable_thinking": request.enable_thinking,
                "stream": True,
                "api": "kb_chat",
            },
            messages=messages,
            assistant_content=full_text,
            request_payload=request_payload,
            response_payload={
                "protocol": "kb_chat_completions_stream",
                "stream": True,
                "finish_reason": "error",
                "assistant": {
                    "content": content,
                    "reasoning_content": reasoning_content,
                },
                "kb_info": kb_info_payload,
                "events": stream_events,
            },
            error={"message": error_msg},
        )
