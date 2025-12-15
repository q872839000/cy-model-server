"""
知识库对话 API 路由

本模块提供知识库对话 API：
- POST /v1/kb/chat/completions - 知识库对话（支持流式）

兼容 OpenAI Chat Completions API 格式，增加知识库特有字段。
"""

import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from models.kb_schemas import (
    KBChatRequest,
    KBChatResponse,
    KBChatChoice,
    KBChatMessageResponse,
    KBChatInfo,
    KBSourceReference,
)
from rag.kb_chat import KBChatService, KBChatConfig
from services.container import CONTAINER
from workers.model_worker import WORKER


router = APIRouter(prefix="/v1/kb", tags=["Knowledge Base Chat"])


# ==================== 服务获取 ====================

# 缓存的 KBChatService 实例
_kb_chat_service: Optional[KBChatService] = None


def _get_kb_chat_service() -> KBChatService:
    """获取知识库对话服务实例"""
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
    async def llm_fn(model, messages, max_tokens, temperature, top_p=0.95, enable_thinking=True, **kwargs):
        """非流式 LLM 调用"""
        return await WORKER.generate_chat(
            model_name=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            enable_thinking=enable_thinking,
        )
    
    async def llm_stream_fn(model, messages, max_tokens, temperature, top_p=0.95, enable_thinking=True, **kwargs):
        """流式 LLM 调用"""
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
    
    # 创建配置
    config = KBChatConfig()
    
    # 创建服务
    _kb_chat_service = KBChatService(
        rag_service=rag_service,
        llm_fn=llm_fn,
        llm_stream_fn=llm_stream_fn,
        config=config,
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
async def kb_chat_completions(request: KBChatRequest):
    """
    知识库对话 API
    
    流程：
    1. 意图识别：判断是否需要检索
    2. Query 改写：多轮对话时优化检索 query
    3. RAG 检索：混合检索 + 重排序
    4. Prompt 组装：根据模型类型组装消息
    5. LLM 生成：生成回答
    
    支持流式和非流式两种模式。
    """
    service = _get_kb_chat_service()
    
    try:
        # 转换消息格式
        messages = [msg.model_dump() for msg in request.messages]
        
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
        
        # 流式
        return StreamingResponse(
            _generate_stream(service, request, messages),
            media_type="text/event-stream",
        )
        
    except Exception as e:
        logger.error("知识库对话失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")


async def _generate_stream(
    service: KBChatService,
    request: KBChatRequest,
    messages: List[dict],
):
    """
    生成流式响应
    
    SSE 格式输出，兼容 OpenAI 流式格式。
    """
    import orjson
    
    chat_id = f"kbchat-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    def make_chunk(delta: dict, finish_reason: str = None, kb_info: dict = None) -> str:
        """构造 SSE chunk"""
        data = {
            "id": chat_id,
            "object": "kb.chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        if kb_info:
            data["kb_info"] = kb_info
        return orjson.dumps(data).decode()
    
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
                # 发送检索结果信息
                yield f"data: {make_chunk({'role': 'assistant'}, kb_info=event_data)}\n\n"
            
            elif event_type == "content":
                # 发送内容片段
                text = event_data.get("text", "")
                yield f"data: {make_chunk({'content': text})}\n\n"
            
            elif event_type == "done":
                # 发送结束标记
                yield f"data: {make_chunk({}, 'stop')}\n\n"
                yield "data: [DONE]\n\n"
            
            elif event_type == "error":
                # 发送错误
                error_msg = event_data.get("message", "Unknown error")
                yield f"data: {make_chunk({'content': f'[错误: {error_msg}]'}, 'error')}\n\n"
                yield "data: [DONE]\n\n"
                
    except Exception as e:
        logger.error("流式对话失败: {}", str(e))
        yield f"data: {make_chunk({'content': f'[错误: {str(e)}]'}, 'error')}\n\n"
        yield "data: [DONE]\n\n"
