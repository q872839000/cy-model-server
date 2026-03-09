"""
OpenAI兼容API：提供与OpenAI API格式兼容的接口

本模块提供以下功能：
1. 聊天补全（Chat Completions）：支持流式和非流式对话生成
2. 嵌入生成（Embeddings）：文本向量化
3. 模型列表（Models）：列出/查询可用模型

API端点：
- POST /v1/chat/completions - 聊天补全
- POST /v1/embeddings - 生成文本嵌入向量
- GET /v1/models - 列出可用模型
- GET /v1/models/{model_id} - 查询单个模型
"""

import uuid
import time
from typing import List

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse
from loguru import logger

from workers.model_worker import WORKER
from core.exceptions import ModelServerException
from core.services.chat_service import CHAT_SERVICE
from utils.message_filter import clean_messages_for_history
from tests.chat_logger import save_chat_log
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
    EmbeddingUsage,
    EmbeddingResponse,
    ModelInfo,
    ModelListResponse,
)

router = APIRouter()


def _normalize_message_role(role: str) -> str:
    """将消息角色标准化为服务端支持的角色名

    OpenAI 兼容处理：
    - "developer" → "system"（OpenAI 新版 API 的 developer 角色等价于 system）
    - "tool" → "user"（本服务暂不支持工具调用，将 tool 消息作为 user 输入处理）
    - 其他角色保持原样，空值兜底为 "user"

    参数:
        role: 原始消息角色字符串

    返回:
        标准化后的角色字符串
    """
    normalized = (role or "").strip().lower()
    # OpenAI 新版 API 使用 developer 替代 system
    if normalized == "developer":
        return "system"
    # 本服务暂不支持 tool calling，将 tool 消息作为 user 输入处理
    if normalized == "tool":
        return "user"
    return normalized or "user"


def _extract_text_from_content(content) -> str:
    """从消息 content 中提取纯文本

    OpenAI 的 content 字段支持多种格式：
    - None: assistant 带 tool_calls 时 content 可为 null
    - str: 最常见的纯文本格式
    - list: 多模态内容块列表（ChatContentBlock 或 dict）

    参数:
        content: 消息的 content 字段，类型不确定

    返回:
        提取后的纯文本字符串，null/空时返回空字符串
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            # 同时支持 dict（原始 JSON）和 Pydantic ChatContentBlock 对象
            if isinstance(block, dict):
                t = block.get("type")
                text_val = block.get("text")
            elif hasattr(block, "type"):
                # Pydantic 模型通过属性访问
                t = getattr(block, "type", None)
                text_val = getattr(block, "text", None)
            else:
                # 未知类型，兜底转字符串
                parts.append(str(block))
                continue
            # 提取文本类内容块
            if t in ("text", "input_text"):
                if isinstance(text_val, str) and text_val:
                    parts.append(text_val)
                continue
            # 图片类内容块用占位符替代
            if t == "image_url":
                parts.append("[image]")
                continue
            # 其他未知类型兜底
            parts.append(str(block))
        return "\n".join([p for p in parts if p])
    # 兜底：非预期类型强制转字符串
    return str(content)


def _normalize_messages(messages: list[ChatMessage]) -> list[dict]:
    """将 ChatMessage 列表标准化为纯文本字典列表

    遍历所有消息，统一角色名并提取纯文本内容，
    生成下游 WORKER/CHAT_SERVICE 可直接消费的格式。

    参数:
        messages: 原始 ChatMessage 对象列表

    返回:
        标准化后的消息字典列表，每项包含 "role" 和 "content" 键
    """
    normalized: list[dict] = []
    for m in messages:
        role = _normalize_message_role(getattr(m, "role", "user"))
        content = _extract_text_from_content(getattr(m, "content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def _resolve_max_tokens(req: ChatCompletionRequest) -> int:
    """解析最大生成 token 数，优先使用 max_completion_tokens"""
    if req.max_completion_tokens is not None:
        return int(req.max_completion_tokens)
    if req.max_tokens is not None:
        return int(req.max_tokens)
    return 2048


def _safe_float(val, default: float) -> float:
    """安全获取 float 参数

    部分 OpenAI 客户端可能显式发送 null（如 temperature: null），
    需要兜底为默认值以防下游 TypeError。

    参数:
        val: 原始参数值，可能为 None
        default: 当 val 为 None 时的默认值

    返回:
        有效的 float 值
    """
    return float(val) if val is not None else default


def _safe_bool(val, default: bool) -> bool:
    """安全获取 bool 参数

    部分 OpenAI 客户端可能显式发送 null（如 stream: null），
    需要兜底为默认值以防下游异常。

    参数:
        val: 原始参数值，可能为 None
        default: 当 val 为 None 时的默认值

    返回:
        有效的 bool 值
    """
    return bool(val) if val is not None else default


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
        # 兼容 OpenAI content blocks + role 扩展（developer/tool 等），统一成纯文本
        normalized_messages = _normalize_messages(req.messages)

        # 过滤历史消息中的思考内容，避免干扰和冗余
        cleaned_messages = clean_messages_for_history(normalized_messages)

        max_tokens = _resolve_max_tokens(req)
        temperature = _safe_float(req.temperature, 0.0)
        top_p = _safe_float(req.top_p, 1.0)
        enable_thinking = _safe_bool(req.enable_thinking, True)
        
        # 预先构造 prompt 并统计 prompt_tokens（流式/非流式共用，避免重复计算）
        tokenizer = _get_tokenizer_for_model(req.model)
        prompt = CHAT_SERVICE.build_prompt(req.model, cleaned_messages, enable_thinking)
        prompt_tokens = _count_tokens(tokenizer, prompt)
        if not req.stream:
            # 非流式：一次性返回
            text = await WORKER.generate_chat(
                model_name=req.model,
                messages=cleaned_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )

            # completion_tokens 统计基于最终输出文本
            completion_tokens = _count_tokens(tokenizer, text)
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            # 测试记录内容
            await save_chat_log(
                model=req.model,
                params={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "enable_thinking": enable_thinking,
                    "stream": req.stream,
                },
                messages=cleaned_messages,
                assistant_content=text,
                usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
            )

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=req.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=text),
                        finish_reason=_infer_finish_reason(completion_tokens, max_tokens),
                    )
                ],
                usage=usage,
            )

        # ====== 流式模式：SSE (Server-Sent Events) 格式输出 ======
        # 延迟导入 orjson，避免在不需要流式时加载
        import orjson as _oj

        async def event_generator():
            """SSE 事件生成器：逐 chunk 推送流式响应"""
            # 同一次对话的所有 chunk 共享相同的 id 和 created
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())

            # OpenAI 兼容：stream_options.include_usage=true 时，在最后追加一个仅含 usage 的 chunk
            include_usage = bool(req.stream_options and req.stream_options.include_usage)

            def dump_chunk(chunk: ChatCompletionChunkResponse) -> str:
                return _oj.dumps(chunk.model_dump(exclude_none=True)).decode()

            # OpenAI 规范：首个 chunk 仅包含 role="assistant"，不含 content
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

            # 流式生成：使用统一的 generate_chat 接口，逐 chunk 获取文本增量
            full_text = ""
            try:
                async for text_chunk in WORKER.generate_chat(
                    model_name=req.model,
                    messages=cleaned_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    enable_thinking=enable_thinking,
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

            # 测试记录内容
            await save_chat_log(
                model=req.model,
                params={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "enable_thinking": enable_thinking,
                    "stream": req.stream,
                },
                messages=cleaned_messages,
                assistant_content=full_text,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

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
                                finish_reason=_infer_finish_reason(completion_tokens, max_tokens),
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

            # OpenAI 规范：流式结束时发送 [DONE] 标记
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except ModelServerException:
        # 让全局异常处理器处理
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'推理失败: {str(e)}')

@router.post('/v1/embeddings')
async def embeddings(req: EmbeddingRequest, request: Request):
    """
    嵌入生成API（兼容 OpenAI POST /v1/embeddings 端点）

    将输入文本转换为向量表示，用于语义检索、相似度计算等场景。
    input 支持单个字符串、字符串列表或 token ID 数组（token ID 暂不支持解码）。

    参数:
        req: 嵌入生成请求对象（包含 model、input 等字段）
        request: FastAPI 请求对象

    返回:
        EmbeddingResponse: 包含向量结果列表和 usage 统计

    异常:
        HTTPException 500: 嵌入生成失败时
    """
    try:
        # 归一化 input：OpenAI 允许单字符串，统一转为 List[str]
        texts = _normalize_embedding_input(req.input)

        # 使用worker进行异步推理
        vecs = await WORKER.generate_embeddings(
            model_name=req.model,
            texts=texts
        )

        # 估算 token 用量
        tokenizer = _get_tokenizer_for_model(req.model)
        prompt_tokens = sum(_count_tokens(tokenizer, t) for t in texts)

        return EmbeddingResponse(
            data=[EmbeddingData(index=i, embedding=v) for i, v in enumerate(vecs)],
            model=req.model,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )
    except ModelServerException:
        # 让全局异常处理器处理
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'嵌入生成失败: {str(e)}')


def _normalize_embedding_input(raw_input) -> List[str]:
    """将 OpenAI Embeddings API 的各种 input 格式归一化为 List[str]

    OpenAI 的 input 字段支持多种格式，本函数统一转为 List[str] 供下游处理：
    - str → [str]              单字符串（最常见，很多客户端默认发送此格式）
    - List[str] → 原样返回     多条文本批量向量化
    - List[int] → [""]         token ID 数组（本服务暂不支持 token 解码）
    - List[List[int]] → [""]*N 多条 token ID（本服务暂不支持 token 解码）

    参数:
        raw_input: 原始 input 字段值

    返回:
        归一化后的字符串列表
    """
    # 最常见的情况：单个字符串
    if isinstance(raw_input, str):
        return [raw_input]
    if isinstance(raw_input, list):
        if len(raw_input) == 0:
            return []
        # List[str] — 字符串列表，直接返回
        if isinstance(raw_input[0], str):
            return raw_input
        # List[int] — token IDs，本服务不支持 token 解码，回退为空字符串
        if isinstance(raw_input[0], int):
            logger.warning("Embedding input 为 token ID 数组，本服务暂不支持 token 解码")
            return [""]
        # List[List[int]] — 多条 token IDs，回退为对应数量的空字符串
        if isinstance(raw_input[0], list):
            logger.warning("Embedding input 为 token ID 二维数组，本服务暂不支持 token 解码")
            return [""] * len(raw_input)
    # 兜底：未知类型强制转字符串
    return [str(raw_input)]


def _get_all_model_infos() -> list[ModelInfo]:
    """获取所有已注册模型的信息列表

    合并 LLM、Embedding、Reranker 三种类型的模型，
    构造 OpenAI 兼容的 ModelInfo 列表。复用于 list 和 retrieve 端点。

    返回:
        ModelInfo 对象列表，包含所有已加载的模型
    """
    model_info = WORKER.get_model_info()
    current_timestamp = int(time.time())
    # 合并所有类型的模型：LLM + Embedding + Reranker
    all_models = (
        model_info.get('llm_models', []) +
        model_info.get('embedding_models', []) +
        model_info.get('reranker_models', [])
    )
    return [
        ModelInfo(id=model, created=current_timestamp)
        for model in all_models
    ]


@router.get('/v1/models')
async def list_models(request: Request) -> ModelListResponse:
    """
    列出所有可用的模型（兼容 OpenAI GET /v1/models 端点）

    返回系统中已加载的所有模型，包括：
    - LLM模型（聊天/对话生成）
    - Embedding模型（文本向量化）
    - Reranker模型（文档重排序）

    参数:
        request: FastAPI 请求对象

    返回:
        ModelListResponse: 包含 object="list" 和 data=[ModelInfo...] 的响应

    异常:
        HTTPException 500: 获取模型列表失败时
    """
    try:
        return ModelListResponse(data=_get_all_model_infos())
    except ModelServerException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'获取模型列表失败: {str(e)}')


@router.get('/v1/models/{model_id:path}')
async def retrieve_model(model_id: str, request: Request) -> ModelInfo:
    """
    查询单个模型信息（兼容 OpenAI GET /v1/models/{model} 端点）

    参数:
        model_id: 模型名称/标识符（路径参数，支持含 / 的模型名）
        request: FastAPI 请求对象

    返回:
        ModelInfo: 匹配的模型信息

    异常:
        HTTPException 404: 模型不存在时
        HTTPException 500: 获取模型信息失败时
    """
    try:
        # 遍历所有已注册模型，精确匹配 model_id
        for m in _get_all_model_infos():
            if m.id == model_id:
                return m
        # 未找到匹配模型，返回 404（符合 OpenAI 规范）
        raise HTTPException(
            status_code=404,
            detail=f"The model '{model_id}' does not exist",
        )
    except HTTPException:
        raise
    except ModelServerException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'获取模型信息失败: {str(e)}')