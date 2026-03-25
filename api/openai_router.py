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
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse
from loguru import logger

from workers.model_worker import WORKER
from core.exceptions import ModelServerException, UnsupportedParameterError
from core.services.chat_service import CHAT_SERVICE
from utils.message_filter import clean_messages_for_history
from utils.thinking_parser import parse_thinking_content, ThinkingStreamSplitter
from utils.chat_logger import save_chat_log
from models import (
    ChatCompletionRequest,
    ChatMessage,
    ToolCall,
    ToolCallFunction,
    ChatCompletionUsage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionDelta,
    ChatCompletionChunkChoice,
    ChatCompletionChunkResponse,
    ToolCallChunk,
    ToolCallChunkFunction,
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
    - "tool" 保持原样（用于 function calling 的工具返回消息）
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
    """将 ChatMessage 列表标准化为下游可消费的字典列表

    遍历所有消息，统一角色名并提取文本内容。
    保留 function calling 相关字段（tool_calls、tool_call_id、name），
    确保完整的工具调用上下文可传递给 Engine/Strategy 层。

    参数:
        messages: 原始 ChatMessage 对象列表

    返回:
        标准化后的消息字典列表
    """
    normalized: list[dict] = []
    for m in messages:
        role = _normalize_message_role(getattr(m, "role", "user"))
        content = _extract_text_from_content(getattr(m, "content", ""))
        msg: dict = {"role": role, "content": content}

        # 保留 assistant 消息中的 tool_calls（工具调用历史）
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = [
                tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else tc
                for tc in tool_calls
            ]
            # OpenAI 规范：assistant 带 tool_calls 时 content 可为 null
            if not content:
                msg["content"] = None

        # 保留 tool 消息中的 tool_call_id 和 name
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        name = getattr(m, "name", None)
        if name:
            msg["name"] = name

        normalized.append(msg)
    return normalized


def _get_client_max_tokens(req: ChatCompletionRequest) -> Optional[int]:
    """提取客户端显式传入的 max_tokens 原始值

    OpenAI 兼容：优先使用 max_completion_tokens，其次 max_tokens。
    返回 None 表示客户端未显式指定（由动态预算决定）。
    """
    if req.max_completion_tokens is not None:
        return int(req.max_completion_tokens)
    if req.max_tokens is not None:
        return int(req.max_tokens)
    return None


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


def _validate_unsupported_params(req: ChatCompletionRequest) -> None:
    """校验当前不支持的 OpenAI 参数，显式拒绝而非静默忽略

    OpenAI 兼容策略：对于已在 schema 中声明但尚未实现的参数，
    当客户端显式传入非默认值时，主动返回错误，避免行为不符预期。
    """
    # n > 1：多候选生成尚未实现
    if req.n is not None and req.n > 1:
        raise UnsupportedParameterError(
            "n", "当前仅支持 n=1，多候选生成尚未实现"
        )
    # response_format：结构化输出尚未实现
    if req.response_format is not None:
        fmt_type = getattr(req.response_format, "type", None)
        if fmt_type and fmt_type != "text":
            raise UnsupportedParameterError(
                "response_format",
                f"当前仅支持 response_format.type='text'，不支持 '{fmt_type}'",
            )
    # parallel_tool_calls=false：本服务始终按单次工具调用语义处理，
    # 显式传入 false 时拒绝，避免客户端误认为并行调用已受限
    if req.parallel_tool_calls is not None and req.parallel_tool_calls is False:
        raise UnsupportedParameterError(
            "parallel_tool_calls",
            "当前不支持显式禁用并行工具调用（parallel_tool_calls=false）",
        )

    # ---- 以下参数已声明但尚未实现，传入非默认值时记录警告 ----
    # 策略说明：这些参数对生成结果有实质影响，但当前引擎层未消费。
    # 采用 warn 而非 reject，原因是多数 OpenAI 客户端/SDK 会默认携带这些字段，
    # 硬拒绝会导致大量客户端无法正常对接。
    _ignored_params: list[tuple[str, Any, Any]] = [
        ("presence_penalty", req.presence_penalty, 0),
        ("frequency_penalty", req.frequency_penalty, 0),
        ("logit_bias", req.logit_bias, None),
        ("seed", req.seed, None),
        ("logprobs", req.logprobs, None),
        ("top_logprobs", req.top_logprobs, None),
    ]
    for param_name, value, default in _ignored_params:
        if value is not None and value != default:
            logger.warning(
                "参数 '{}' 已传入(值={})但当前未实现，将被忽略",
                param_name, value,
            )


def _resolve_tools_and_choice(req: ChatCompletionRequest):
    """统一解析 tools/tool_choice，兼容旧版 functions/function_call

    OpenAI 旧版使用 functions + function_call，新版使用 tools + tool_choice。
    当请求中仅包含旧版字段时，自动映射为新版格式，确保下游统一处理。

    Returns:
        (tools, tool_choice) 元组
    """
    tools = req.tools if req.tools else None
    tool_choice = req.tool_choice

    # 旧版 functions/function_call → 新版 tools/tool_choice 兼容映射
    if not tools and req.functions:
        tools = [
            {"type": "function", "function": fn}
            for fn in req.functions
        ]
        # 映射 function_call → tool_choice
        if req.function_call is not None and tool_choice is None:
            fc = req.function_call
            if isinstance(fc, str):
                # "auto" / "none" 直接映射
                tool_choice = fc
            elif isinstance(fc, dict) and "name" in fc:
                # {"name": "xxx"} → {"type": "function", "function": {"name": "xxx"}}
                tool_choice = {
                    "type": "function",
                    "function": {"name": fc["name"]},
                }

    return tools, tool_choice


def _normalize_stop(req_stop) -> list[str] | None:
    """将 OpenAI stop 参数归一化为 List[str] 或 None

    OpenAI stop 支持 str | List[str] | None。
    """
    if req_stop is None:
        return None
    if isinstance(req_stop, str):
        return [req_stop] if req_stop else None
    if isinstance(req_stop, list):
        filtered = [s for s in req_stop if isinstance(s, str) and s]
        return filtered if filtered else None
    return None


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
        HTTPException 400: 参数不支持
        HTTPException 500: 推理失败
    """
    try:
        # ---- 参数校验：显式拒绝尚未支持的参数 ----
        _validate_unsupported_params(req)

        # 兼容 OpenAI content blocks + role 扩展（developer/tool 等），统一成纯文本
        normalized_messages = _normalize_messages(req.messages)

        # 过滤历史消息中的思考内容，避免干扰和冗余
        cleaned_messages = clean_messages_for_history(normalized_messages)

        client_max_tokens = _get_client_max_tokens(req)
        temperature = _safe_float(req.temperature, 0.0)
        top_p = _safe_float(req.top_p, 1.0)
        enable_thinking = req.enable_thinking  # None 时由 ChatService 从模型配置读取默认值
        stop = _normalize_stop(req.stop)

        # 统一 tools/tool_choice（兼容旧版 functions/function_call）
        tools, tool_choice = _resolve_tools_and_choice(req)

        # 预先构造 prompt 并统计 prompt_tokens（流式/非流式共用，避免重复计算）
        # 此处传入 tools 以使 prompt 统计与真实执行路径尽量一致
        tokenizer = _get_tokenizer_for_model(req.model)
        prompt = CHAT_SERVICE.build_prompt(
            req.model, cleaned_messages, enable_thinking,
            tools=tools, tool_choice=tool_choice,
        )
        prompt_tokens = _count_tokens(tokenizer, prompt)

        # 动态预算：根据 context_window 和 prompt_tokens 计算安全的 max_tokens
        max_tokens = CHAT_SERVICE.resolve_max_tokens(
            req.model, prompt_tokens, client_max_tokens,
        )

        if not req.stream:
            # 非流式：一次性返回
            chat_result = await WORKER.generate_chat(
                model_name=req.model,
                messages=cleaned_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                enable_thinking=enable_thinking,
                tools=tools,
                tool_choice=tool_choice,
            )

            text = chat_result.get_text()

            # 拆分 <think> 标签：reasoning_content 和 content 分离
            reasoning_content, content = parse_thinking_content(text)

            # completion_tokens 统计基于完整输出文本（含思考内容）
            completion_tokens = _count_tokens(tokenizer, text)
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            # 对话日志记录（调试/监控用）
            await save_chat_log(
                model=req.model,
                params={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "enable_thinking": enable_thinking,
                    "stream": req.stream,
                    "tools": bool(tools),
                },
                messages=cleaned_messages,
                assistant_content=text,
                usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
            )

            # 构建 assistant 消息
            assistant_msg = ChatMessage(
                role="assistant",
                content=content or None,
                reasoning_content=reasoning_content,
            )

            # 确定 finish_reason：策略层已设置 "tool_calls"/"stop"，
            # 路由层仅需补充 "length" 判断（策略层无 token 计数信息）
            if chat_result.has_tool_calls:
                # 将工具调用转换为 OpenAI ToolCall 模型
                assistant_msg.tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type=tc.get("type", "function"),
                        function=ToolCallFunction(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in chat_result.tool_calls
                ]
                finish_reason = "tool_calls"
            else:
                finish_reason = _infer_finish_reason(completion_tokens, max_tokens)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=req.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=assistant_msg,
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage,
            )

        # ====== 流式模式：SSE (Server-Sent Events) 格式输出 ======
        import orjson as _oj

        # 当 tools 存在且 tool_choice != "none" 时，需要缓冲全部输出再决定发送方式。
        # 原因：本地模型无法提前声明是否会调用工具，必须等生成结束后解析。
        # OpenAI 规范要求 tool_calls 通过 delta.tool_calls 推送，不能混在 content 中。
        should_buffer_for_tools = bool(tools and tool_choice != "none")

        async def event_generator():
            """SSE 事件生成器：逐 chunk 推送流式响应"""
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            include_usage = bool(req.stream_options and req.stream_options.include_usage)

            # ---- SSE 格式化辅助函数 ----

            def _sse(chunk: ChatCompletionChunkResponse) -> str:
                """将 chunk 对象序列化为 SSE data 行"""
                return "data: " + _oj.dumps(chunk.model_dump(exclude_none=True)).decode() + "\n\n"

            def _delta_sse(
                delta: ChatCompletionDelta,
                finish_reason: str | None = None,
            ) -> str:
                """构建包含单个 delta 的 SSE data 行"""
                return _sse(ChatCompletionChunkResponse(
                    id=chunk_id,
                    created=created,
                    model=req.model,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=delta,
                        finish_reason=finish_reason,
                    )],
                ))

            def _thinking_delta_sse(field: str, text: str) -> str:
                """将 ThinkingStreamSplitter 输出转换为 SSE data 行"""
                return _delta_sse(ChatCompletionDelta(
                    reasoning_content=text if field == "reasoning_content" else None,
                    content=text if field == "content" else None,
                ))

            # ---- OpenAI 规范：首个 chunk 仅包含 role="assistant" ----
            yield _delta_sse(ChatCompletionDelta(role="assistant"))

            # ---- 收集生成文本 ----
            full_text = ""
            finish_reason = "stop"
            splitter = ThinkingStreamSplitter()

            async for text_chunk in WORKER.generate_chat(
                model_name=req.model,
                messages=cleaned_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=True,
                enable_thinking=enable_thinking,
                tools=tools,
                tool_choice=tool_choice,
            ):
                if not text_chunk:
                    continue
                full_text += text_chunk

                # 普通模式：实时推送 content/reasoning_content 增量
                if not should_buffer_for_tools:
                    for field, text in splitter.feed(text_chunk):
                        yield _thinking_delta_sse(field, text)

            # ---- 生成结束后的处理 ----

            if should_buffer_for_tools:
                # 工具缓冲模式：通过 Service 层解析工具调用（保持分层架构）
                parsed_calls = CHAT_SERVICE.parse_tool_calls(req.model, full_text)

                if parsed_calls:
                    # 按 OpenAI 规范推送 tool_calls 增量
                    # 每个工具调用分两个 chunk：首个含 id/type/name，后续含 arguments
                    for idx, tc in enumerate(parsed_calls):
                        fn = tc["function"]
                        # 首个 chunk：id + type + function.name
                        yield _delta_sse(ChatCompletionDelta(
                            tool_calls=[ToolCallChunk(
                                index=idx,
                                id=tc["id"],
                                type="function",
                                function=ToolCallChunkFunction(
                                    name=fn["name"],
                                    arguments="",
                                ),
                            )],
                        ))
                        # 后续 chunk：function.arguments（完整发送）
                        yield _delta_sse(ChatCompletionDelta(
                            tool_calls=[ToolCallChunk(
                                index=idx,
                                function=ToolCallChunkFunction(
                                    arguments=fn["arguments"],
                                ),
                            )],
                        ))
                    finish_reason = "tool_calls"
                else:
                    # 无工具调用：回放缓冲文本（经过 thinking splitter 处理）
                    replay_splitter = ThinkingStreamSplitter()
                    for field, text in replay_splitter.feed(full_text):
                        yield _thinking_delta_sse(field, text)
                    for field, text in replay_splitter.flush():
                        yield _thinking_delta_sse(field, text)
            else:
                # 普通模式：刷新 splitter 缓冲区中的剩余内容
                for field, text in splitter.flush():
                    yield _thinking_delta_sse(field, text)

            # ---- 统计与日志 ----
            completion_tokens = _count_tokens(tokenizer, full_text)

            if finish_reason == "stop":
                finish_reason = _infer_finish_reason(completion_tokens, max_tokens)

            await save_chat_log(
                model=req.model,
                params={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "enable_thinking": enable_thinking,
                    "stream": req.stream,
                    "tools": bool(tools),
                },
                messages=cleaned_messages,
                assistant_content=full_text,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

            # ---- 结束 chunk（含 finish_reason）----
            yield _delta_sse(ChatCompletionDelta(), finish_reason=finish_reason)

            # ---- OpenAI 兼容：include_usage 的额外 usage chunk ----
            if include_usage:
                usage = ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                yield _sse(ChatCompletionChunkResponse(
                    id=chunk_id,
                    created=created,
                    model=req.model,
                    choices=[],
                    usage=usage,
                ))

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