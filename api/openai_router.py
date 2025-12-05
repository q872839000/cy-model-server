"""
OpenAI兼容API：提供与OpenAI API格式兼容的接口

本模块提供以下功能：
1. 聊天补全（Chat Completions）：支持流式和非流式对话生成
2. 嵌入生成（Embeddings）：文本向量化
3. 模型列表（Models）：列出所有可用的模型

API端点：
- POST /v1/chat/completions - 聊天补全
- POST /v1/embeddings - 生成文本嵌入向量
- GET /v1/models - 列出可用模型
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import List, Any, Dict
import asyncio
from workers.model_worker import WORKER
from core.exceptions import ModelServerException
from starlette.responses import StreamingResponse
import uuid

router = APIRouter()


def _get_model_manager(request):
    """
    获取模型管理器实例
    
    返回:
        REGISTRY: 全局模型注册表，管理所有已加载的模型
    """
    # 使用 core.registry.REGISTRY 作为模型管理器
    try:
        from core.registry import REGISTRY
        return REGISTRY
    except Exception:
        return None


class ChatMessage(BaseModel):
    """
    聊天消息
    
    属性:
        role: 消息角色，可选值：system(系统)、user(用户)、assistant(助手)
        content: 消息内容文本
    """
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """
    聊天补全请求
    
    属性:
        model: 使用的LLM模型名称
        messages: 对话历史消息列表
        max_tokens: 最大生成token数，默认256
        temperature: 生成温度(0-2)，越高越随机，默认0.0
        top_p: 核采样参数(0-1)，默认1.0
        stream: 是否使用流式输出，默认False
    """
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False


class EmbeddingRequest(BaseModel):
    """
    嵌入生成请求
    
    属性:
        model: 使用的Embedding模型名称
        input: 待向量化的文本列表
    """
    model: str
    input: List[str]

def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    将消息列表转换为提示词字符串
    
    参数:
        messages: 消息字典列表
        
    返回:
        str: 格式化的提示词文本
    """
    parts = []
    for m in messages:
        role = m.get('role', '')
        content = m.get('content', '') or m.get('text', '')
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


@router.post('/v1/chat/completions')
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """
    聊天补全API（兼容OpenAI格式）
    
    支持两种模式：
    1. 非流式：一次性返回完整响应
    2. 流式：逐token实时返回，使用Server-Sent Events (SSE)格式
    
    参数:
        req: 聊天补全请求对象
        request: FastAPI请求对象
        
    返回:
        非流式: JSON格式的聊天补全响应
        流式: StreamingResponse，包含多个data chunk
        
    异常:
        HTTPException: 当推理失败时抛出500错误
    """
    try:
        if not req.stream:
            # 非流式：一次性返回
            text = await WORKER.generate_chat(
                model_name=req.model,
                messages=[m.dict() for m in req.messages],
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
            return {
                'id': f'chatcmpl-{uuid.uuid4().hex[:8]}',
                'object': 'chat.completion',
                'created': int(__import__('time').time()),
                'model': req.model,
                'choices': [
                    {
                        'index': 0,
                        'message': {'role': 'assistant', 'content': text},
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            }

        # 流式：真正的token级别流式输出
        async def event_generator():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(__import__('time').time())
            
            # 发送role首块
            first_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }
                ]
            }
            import orjson as _oj
            yield f"data: {_oj.dumps(first_chunk).decode()}\n\n"
            
            # 真正的流式生成：逐token输出
            try:
                async for text_chunk in WORKER.generate_chat_stream(
                    model_name=req.model,
                    messages=[m.dict() for m in req.messages],
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                ):
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text_chunk},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {_oj.dumps(chunk).decode()}\n\n"
            except Exception as e:
                # 如果生成过程中出错，发送错误信息
                error_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error"
                        }
                    ]
                }
                yield f"data: {_oj.dumps(error_chunk).decode()}\n\n"
                raise
            
            # 发送结束块
            done_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {_oj.dumps(done_chunk).decode()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except ModelServerException:
        # 让全局异常处理器处理
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'推理失败: {str(e)}')

@router.post('/v1/embeddings')
async def embeddings(req: EmbeddingRequest, request: Request):
    """
    嵌入生成API（兼容OpenAI格式）
    
    将输入文本列表转换为向量表示，用于语义检索、相似度计算等场景
    
    参数:
        req: 嵌入生成请求对象
        request: FastAPI请求对象
        
    返回:
        JSON格式的嵌入响应，包含：
        - object: 固定值'list'
        - data: 向量列表，每个元素包含embedding和index
        
    异常:
        HTTPException: 当嵌入生成失败时抛出500错误
    """
    try:
        # 使用worker进行异步推理
        vecs = await WORKER.generate_embeddings(
            model_name=req.model,
            texts=req.input
        )
        
        return {
            'object': 'list',
            'data': [{'embedding': v, 'index': i} for i, v in enumerate(vecs)]
        }
    except ModelServerException:
        # 让全局异常处理器处理
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'嵌入生成失败: {str(e)}')


@router.get('/v1/models')
async def list_models(request: Request):
    """
    列出所有可用的模型（兼容OpenAI格式）
    
    返回系统中已加载的所有模型，包括：
    - LLM模型（聊天补全）
    - Embedding模型（文本向量化）
    - Reranker模型（重排序）
    
    参数:
        request: FastAPI请求对象
        
    返回:
        JSON格式的模型列表响应，包含：
        - object: 固定值'list'
        - data: 模型列表，每个模型包含id、object、created、owned_by等字段
        
    异常:
        HTTPException: 当获取模型列表失败时抛出500错误
    """
    try:
        model_info = WORKER.get_model_info()
        return {
            'object': 'list',
            'data': [
                {
                    'id': model,
                    'object': 'model',
                    'created': 1234567890,
                    'owned_by': 'cy-model-server'
                }
                for model in model_info['llm_models'] + model_info['embedding_models'] + model_info['reranker_models']
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'获取模型列表失败: {str(e)}')