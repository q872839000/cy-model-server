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

from fastapi import APIRouter, HTTPException, Request
from workers.model_worker import WORKER
from core.exceptions import ModelServerException
from starlette.responses import StreamingResponse
from utils.message_filter import clean_messages_for_history
from models import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionUsage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionDelta,
    ChatCompletionChunkChoice,
    ChatCompletionChunkResponse,
    EmbeddingRequest,
    EmbeddingData,
    EmbeddingResponse,
    ModelInfo,
    ModelListResponse
)
import uuid
import time

from core.services.chat_service import CHAT_SERVICE

router = APIRouter()


def _get_tokenizer_for_model(model_name: str):
    """
    获取指定模型的tokenizer实例
    
    通过 CHAT_SERVICE 获取 tokenizer，遵循分层架构。
    
    参数:
        model_name: 模型名称
        
    返回:
        tokenizer实例或None（获取失败时）
    """
    return CHAT_SERVICE.get_tokenizer(model_name)


def _count_tokens(tokenizer, text: str) -> int:
    """
    统计文本的token数量
    
    用于usage统计，尽量排除special tokens以更接近OpenAI的计算方式
    注意：不同tokenizer的计数规则可能略有差异，此处仅做近似估算
    
    参数:
        tokenizer: tokenizer实例
        text: 待统计的文本
        
    返回:
        token数量（统计失败时返回0）
    """
    # 统计 token 数（仅用于 usage 估算；不同 tokenizer 的计数规则可能略有差异）
    if tokenizer is None or not text:
        return 0
    try:
        # 尽量不把 special tokens 计入（更接近 OpenAI usage 语义）
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))
    except Exception:
        return 0


def _infer_finish_reason(completion_tokens: int, max_tokens: int | None) -> str:
    """
    推断生成结束的原因
    
    OpenAI 兼容：根据生成的token数量与最大限制判断结束原因
    - 达到max_tokens限制：返回"length"
    - 自然结束或其他原因：返回"stop"
    
    参数:
        completion_tokens: 实际生成的token数量
        max_tokens: 最大token限制
        
    返回:
        结束原因字符串："stop" 或 "length"
    """
    if max_tokens is None or max_tokens <= 0 or completion_tokens < max_tokens:
        return "stop"
    return "length"


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
        # 过滤历史消息中的思考内容，避免干扰和冗余
        cleaned_messages = clean_messages_for_history([m.model_dump() for m in req.messages])
        
        # 预先构造 prompt 并统计 prompt_tokens（流式/非流式共用，避免重复计算）
        tokenizer = _get_tokenizer_for_model(req.model)
        prompt = CHAT_SERVICE.build_prompt(req.model, cleaned_messages, req.enable_thinking)
        prompt_tokens = _count_tokens(tokenizer, prompt)
        if not req.stream:
            # 非流式：一次性返回
            text = await WORKER.generate_chat(
                model_name=req.model,
                messages=cleaned_messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                enable_thinking=req.enable_thinking,
            )

            # completion_tokens 统计基于最终输出文本
            completion_tokens = _count_tokens(tokenizer, text)
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=req.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=text),
                        finish_reason=_infer_finish_reason(completion_tokens, req.max_tokens),
                    )
                ],
                usage=usage,
            )

        # 流式：SSE 格式输出
        import orjson as _oj

        async def event_generator():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())

            # OpenAI 兼容：stream_options.include_usage=true 时，在最后追加一个仅含 usage 的 chunk
            include_usage = bool(req.stream_options and req.stream_options.include_usage)

            def dump_chunk(chunk: ChatCompletionChunkResponse) -> str:
                return _oj.dumps(chunk.model_dump(exclude_none=True)).decode()

            yield (
                "data: "
                + dump_chunk(
                    ChatCompletionChunkResponse(
                        id=chunk_id,
                        created=created,
                        model=req.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionDelta(role="assistant"),
                                finish_reason=None,
                            )
                        ],
                    )
                )
                + "\n\n"
            )

            # 流式生成：使用统一的 generate_chat 接口
            full_text = ""
            try:
                async for text_chunk in WORKER.generate_chat(
                    model_name=req.model,
                    messages=cleaned_messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stream=True,
                    enable_thinking=req.enable_thinking,
                ):
                    # 为了在流式结束后计算 completion_tokens，需要拼接完整输出
                    if text_chunk:
                        full_text += text_chunk
                        # 只有非空chunk才发送，避免无意义的空包
                        yield (
                            "data: "
                            + dump_chunk(
                                ChatCompletionChunkResponse(
                                    id=chunk_id,
                                    created=created,
                                    model=req.model,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            index=0,
                                            delta=ChatCompletionDelta(content=text_chunk),
                                            finish_reason=None,
                                        )
                                    ],
                                )
                            )
                            + "\n\n"
                        )
            except Exception:
                raise

            # 结束后统一计算 completion_tokens，并决定 finish_reason
            completion_tokens = _count_tokens(tokenizer, full_text)

            # 结束块
            yield (
                "data: "
                + dump_chunk(
                    ChatCompletionChunkResponse(
                        id=chunk_id,
                        created=created,
                        model=req.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionDelta(),
                                finish_reason=_infer_finish_reason(completion_tokens, req.max_tokens),
                            )
                        ],
                    )
                )
                + "\n\n"
            )

            if include_usage:
                # OpenAI 兼容：最后额外发一个 chunk，choices=[]，usage 有值
                usage = ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                yield (
                    "data: "
                    + dump_chunk(
                        ChatCompletionChunkResponse(
                            id=chunk_id,
                            created=created,
                            model=req.model,
                            choices=[],
                            usage=usage,
                        )
                    )
                    + "\n\n"
                )

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

        return EmbeddingResponse(
            object='list',
            data=[EmbeddingData(index=i, embedding=v) for i, v in enumerate(vecs)],
            model=req.model,
            dimension=(len(vecs[0]) if vecs else None)
        )
    except ModelServerException:
        # 让全局异常处理器处理
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'嵌入生成失败: {str(e)}')


@router.get('/v1/models')
async def list_models(request: Request) -> ModelListResponse:
    """
    列出所有可用的模型（兼容OpenAI格式）
    
    返回系统中已加载的所有模型，包括：
    - LLM模型
    - Embedding模型（文本向量化）
    - Reranker模型（重排序）
    
    参数:
        request: FastAPI请求对象
        
    返回:
        ModelListResponse: 标准格式的模型列表响应
        
    异常:
        HTTPException: 当获取模型列表失败时抛出500错误
    """
    try:
        model_info = WORKER.get_model_info()
        current_timestamp = int(time.time())
        
        # 合并所有类型的模型并构造响应
        all_models = (
            model_info.get('llm_models', []) + 
            model_info.get('embedding_models', []) + 
            model_info.get('reranker_models', [])
        )
        
        model_list = [
            ModelInfo(
                id=model,
                created=current_timestamp,
                owned_by='cy-model-server'
            )
            for model in all_models
        ]
        
        return ModelListResponse(data=model_list)
        
    except ModelServerException:
        # 让全局异常处理器处理
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'获取模型列表失败: {str(e)}')