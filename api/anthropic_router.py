"""
Anthropic Messages API 兼容端点

本模块提供与 Anthropic Messages API 格式兼容的接口，用于对接 Claude Code 等
仅支持 Anthropic API 规范的客户端工具。

核心职责：
1. 接收 Anthropic 格式的请求，转换为内部格式
2. 调用 WORKER / CHAT_SERVICE 执行推理（与 OpenAI Router 共享同一套基础设施）
3. 将内部格式的响应转换为 Anthropic 格式返回

API端点：
- POST /v1/messages - Anthropic Messages API（支持流式和非流式）

设计原则：
- 纯协议翻译层，不引入新的业务逻辑
- 底层完全复用 WORKER / CHAT_SERVICE / Strategy / Engine 全链路
"""

import json
import uuid
import time
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse
from loguru import logger

from workers.model_worker import WORKER
from core.registry import REGISTRY
from core.exceptions import ModelServerException
from core.services.chat_service import CHAT_SERVICE
from utils.message_filter import clean_messages_for_history
from utils.thinking_parser import parse_thinking_content, ThinkingStreamSplitter
from utils.chat_logger import save_chat_log
from models.anthropic_schemas import (
    AnthropicMessagesRequest,
    AnthropicMessage,
    AnthropicContentBlock,
    AnthropicToolDef,
    AnthropicToolChoice,
    AnthropicUsage,
    AnthropicResponseContentBlock,
    AnthropicMessagesResponse,
    AnthropicErrorDetail,
    AnthropicErrorResponse,
)

router = APIRouter()


# ==================== 请求转换：Anthropic → 内部格式 ====================


def _resolve_model(requested_model: str) -> str:
    """解析模型名称，不存在时兜底到默认模型

    Claude Code 默认请求 claude-sonnet-4-20250514 等 Anthropic 模型名，
    本地服务需要映射到实际加载的模型。

    参数:
        requested_model: 客户端请求的模型名称

    返回:
        实际可用的模型名称

    异常:
        HTTPException 404: 无任何可用模型
    """
    if REGISTRY.get_llm(requested_model):
        return requested_model
    default = REGISTRY.get_default_llm_name()
    if default:
        logger.info("模型 '{}' 不存在，使用默认模型 '{}'", requested_model, default)
        return default
    raise HTTPException(status_code=404, detail=f"Model '{requested_model}' not found")


def _extract_system_text(system: Any) -> str:
    """从 Anthropic system 字段提取纯文本

    Anthropic system 支持两种格式：
    - str: 直接文本
    - List[Dict]: 内容块数组，提取 type=text 的 text 字段

    参数:
        system: Anthropic 请求的 system 字段

    返回:
        纯文本字符串
    """
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        texts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)
    return str(system)


def _convert_assistant_message(msg: AnthropicMessage) -> dict:
    """转换 Anthropic assistant 消息为内部格式

    Anthropic assistant 消息可包含 text + tool_use + thinking 混合 blocks，
    需要拆分为 content（文本）和 tool_calls（工具调用）。
    thinking blocks 在历史消息中跳过。

    参数:
        msg: Anthropic 格式的 assistant 消息

    返回:
        内部格式的消息字典
    """
    if isinstance(msg.content, str):
        return {"role": "assistant", "content": msg.content}

    texts: list[str] = []
    tool_calls: list[dict] = []
    for block in msg.content:
        if block.type == "text":
            texts.append(block.text or "")
        elif block.type == "tool_use":
            # Anthropic tool_use → OpenAI tool_calls 格式
            arguments = block.input
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False) if arguments else "{}"
            tool_calls.append({
                "id": block.id or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": block.name or "",
                    "arguments": arguments,
                },
            })
        # thinking blocks 在历史上下文中跳过

    result: dict = {"role": "assistant", "content": "\n".join(texts) if texts else None}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _convert_user_message(msg: AnthropicMessage) -> list[dict]:
    """转换 Anthropic user 消息为内部格式

    Anthropic 的 tool_result 放在 user 消息的 content blocks 中，
    需要拆分为独立的 role="tool" 消息（OpenAI 格式）。
    普通文本合并为一条 role="user" 消息。

    参数:
        msg: Anthropic 格式的 user 消息

    返回:
        内部格式消息字典的列表（可能拆分为多条）
    """
    if isinstance(msg.content, str):
        return [{"role": "user", "content": msg.content}]

    results: list[dict] = []
    text_parts: list[str] = []

    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text or "")
        elif block.type == "tool_result":
            # 先输出之前积累的文本
            if text_parts:
                results.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts = []
            # tool_result → role: "tool"
            if isinstance(block.content, str):
                tool_content = block.content
            elif block.content is not None:
                tool_content = json.dumps(block.content, ensure_ascii=False)
            else:
                tool_content = ""
            results.append({
                "role": "tool",
                "tool_call_id": block.tool_use_id or "",
                "content": tool_content,
            })
        elif block.type == "image":
            # 图片占位符（依赖底层模型多模态能力）
            text_parts.append("[image]")

    if text_parts:
        results.append({"role": "user", "content": "\n".join(text_parts)})

    return results if results else [{"role": "user", "content": ""}]


def _convert_messages(req: AnthropicMessagesRequest) -> list[dict]:
    """将 Anthropic 消息格式转换为内部消息格式

    转换要点：
    1. system (顶层字段) → {"role": "system", "content": "..."}
    2. assistant 消息中的 tool_use blocks → tool_calls 字段
    3. user 消息中的 tool_result blocks → role: "tool" 独立消息
    4. thinking blocks → 跳过

    参数:
        req: Anthropic 请求体

    返回:
        内部格式的消息字典列表
    """
    messages: list[dict] = []

    # system prompt → system message
    system_text = _extract_system_text(req.system)
    if system_text:
        messages.append({"role": "system", "content": system_text})

    # 逐条转换对话消息
    for msg in req.messages:
        if msg.role == "assistant":
            messages.append(_convert_assistant_message(msg))
        elif msg.role == "user":
            messages.extend(_convert_user_message(msg))
        else:
            # 未知角色兜底
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            messages.append({"role": msg.role, "content": content})

    return messages


def _convert_tools(anthropic_tools: list[AnthropicToolDef]) -> list[dict]:
    """将 Anthropic 工具定义转换为 OpenAI 格式

    核心差异：input_schema → parameters

    参数:
        anthropic_tools: Anthropic 格式的工具定义列表

    返回:
        OpenAI 格式的工具定义列表
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema,
            },
        }
        for t in anthropic_tools
    ]


def _convert_tool_choice(tc: AnthropicToolChoice) -> Any:
    """将 Anthropic tool_choice 转换为 OpenAI 格式

    映射关系：
    - {"type": "auto"} → "auto"
    - {"type": "any"} → "required"
    - {"type": "tool", "name": "xxx"} → {"type": "function", "function": {"name": "xxx"}}

    参数:
        tc: Anthropic 格式的 tool_choice

    返回:
        OpenAI 格式的 tool_choice
    """
    if tc.type == "auto":
        return "auto"
    elif tc.type == "any":
        return "required"
    elif tc.type == "tool" and tc.name:
        return {"type": "function", "function": {"name": tc.name}}
    return "auto"


def _resolve_enable_thinking(thinking: Any) -> Optional[bool]:
    """将 Anthropic thinking 配置转换为内部 enable_thinking

    参数:
        thinking: Anthropic ThinkingConfig 对象

    返回:
        True/False/None（None 表示使用模型配置默认值）
    """
    if thinking is None:
        return None
    if hasattr(thinking, "type"):
        if thinking.type == "enabled":
            if hasattr(thinking, "budget_tokens") and thinking.budget_tokens:
                logger.warning(
                    "Anthropic thinking.budget_tokens={} 在本地模型中无法精确控制，将被忽略",
                    thinking.budget_tokens,
                )
            return True
        elif thinking.type == "disabled":
            return False
    return None


# ==================== 响应转换：内部格式 → Anthropic ====================


def _map_stop_reason(finish_reason: str) -> str:
    """将内部 finish_reason 映射为 Anthropic stop_reason

    映射关系：
    - "stop" → "end_turn"
    - "length" → "max_tokens"
    - "tool_calls" → "tool_use"

    参数:
        finish_reason: 内部的 finish_reason 字符串

    返回:
        Anthropic 格式的 stop_reason
    """
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_reason, "end_turn")


def _build_content_blocks(
    text: str,
    reasoning_content: Optional[str],
    tool_calls: Optional[list],
) -> list[AnthropicResponseContentBlock]:
    """构建 Anthropic 响应的 content 内容块列表

    按照 Anthropic 规范的顺序排列：thinking → text → tool_use

    参数:
        text: 正文文本
        reasoning_content: 思考内容（可选）
        tool_calls: OpenAI 格式的 tool_calls 列表（可选）

    返回:
        Anthropic 格式的内容块列表
    """
    content: list[AnthropicResponseContentBlock] = []

    # 1. thinking block
    if reasoning_content:
        content.append(AnthropicResponseContentBlock(
            type="thinking", thinking=reasoning_content,
        ))

    # 2. text block
    if text:
        content.append(AnthropicResponseContentBlock(type="text", text=text))

    # 3. tool_use blocks
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            # 解析 arguments JSON 字符串为 dict
            try:
                input_obj = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                input_obj = {}
            # tool_use id：Anthropic 使用 toolu_ 前缀
            raw_id = tc.get("id", "")
            tool_id = raw_id.replace("call_", "toolu_", 1) if raw_id.startswith("call_") else raw_id
            if not tool_id:
                tool_id = f"toolu_{uuid.uuid4().hex[:12]}"
            content.append(AnthropicResponseContentBlock(
                type="tool_use",
                id=tool_id,
                name=fn.get("name", ""),
                input=input_obj,
            ))

    # 确保 content 不为空
    if not content:
        content.append(AnthropicResponseContentBlock(type="text", text=""))

    return content


# ==================== Token 统计辅助 ====================


def _get_tokenizer_for_model(model_name: str):
    """获取指定模型的 tokenizer 实例"""
    return CHAT_SERVICE.get_tokenizer(model_name)


def _count_tokens(tokenizer, text: str) -> int:
    """统计文本的 token 数量"""
    if tokenizer is None or not text:
        return 0
    try:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))
    except Exception:
        return 0


def _infer_finish_reason(completion_tokens: int, max_tokens: int) -> str:
    """推断生成结束原因"""
    if max_tokens <= 0 or completion_tokens < max_tokens:
        return "stop"
    return "length"


# ==================== Anthropic 错误响应构建 ====================


def _anthropic_error_response(status_code: int, error_type: str, message: str):
    """构建 Anthropic 格式的错误 JSON 响应

    参数:
        status_code: HTTP 状态码
        error_type: Anthropic 错误类型
        message: 错误描述信息

    返回:
        JSONResponse 对象
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}},
        status_code=status_code,
    )


# ==================== SSE 事件格式化 ====================


def _sse(event_type: str, data: dict) -> str:
    """格式化 Anthropic SSE 事件

    Anthropic SSE 格式与 OpenAI 不同：每个事件有 event: 行和 data: 行

    参数:
        event_type: SSE 事件类型名称
        data: 事件数据字典

    返回:
        格式化后的 SSE 字符串
    """
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==================== API 端点 ====================


@router.post("/v1/messages")
async def create_message(req: AnthropicMessagesRequest, request: Request):
    """Anthropic Messages API 兼容端点

    支持两种模式：
    1. 非流式：一次性返回完整响应（AnthropicMessagesResponse）
    2. 流式：逐事件实时返回，使用 Anthropic SSE 格式

    参数:
        req: Anthropic Messages 请求对象
        request: FastAPI 请求对象

    返回:
        非流式: AnthropicMessagesResponse JSON
        流式: StreamingResponse，包含 Anthropic SSE 事件
    """
    try:
        raw_request = req.model_dump(mode="json", exclude_none=False)

        # ---- 1. 模型解析 ----
        model = _resolve_model(req.model)

        # ---- 2. 请求转换：Anthropic → 内部格式 ----
        messages = _convert_messages(req)
        tools = _convert_tools(req.tools) if req.tools else None
        tool_choice = _convert_tool_choice(req.tool_choice) if req.tool_choice else None
        stop = req.stop_sequences
        enable_thinking = _resolve_enable_thinking(req.thinking)
        temperature = req.temperature if req.temperature is not None else 0.0
        top_p = req.top_p if req.top_p is not None else 1.0
        top_k = req.top_k  # 传透到引擎层
        client_max_tokens = req.max_tokens  # 客户端原始值，None 表示未传

        # 过滤历史消息中的思考内容
        cleaned_messages = clean_messages_for_history(messages)

        # ---- 3. Token 统计准备 ----
        tokenizer = _get_tokenizer_for_model(model)
        prompt = CHAT_SERVICE.build_prompt(
            model, cleaned_messages, enable_thinking,
            tools=tools, tool_choice=tool_choice,
        )
        input_tokens = _count_tokens(tokenizer, prompt)

        # 动态预算：根据 context_window 和 input_tokens 计算安全的 max_tokens
        max_tokens = CHAT_SERVICE.resolve_max_tokens(
            model, input_tokens, client_max_tokens,
        )

        request_payload = {
            "protocol": "anthropic_messages",
            "raw": raw_request,
            "normalized": {
                "requested_model": req.model,
                "resolved_model": model,
                "messages": cleaned_messages,
                "prompt": prompt,
                "params": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stop": stop,
                    "enable_thinking": enable_thinking,
                    "stream": req.stream,
                    "tools": tools,
                    "tool_choice": tool_choice,
                },
            },
        }

        if req.stream:
            # ====== 流式模式 ======
            return StreamingResponse(
                _stream_anthropic_events(
                    req=req,
                    model=model,
                    messages=cleaned_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    stop=stop,
                    enable_thinking=enable_thinking,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    tokenizer=tokenizer,
                    input_tokens=input_tokens,
                    request_payload=request_payload,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # ====== 非流式模式 ======
        chat_result = await WORKER.generate_chat(
            model_name=model,
            messages=cleaned_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            enable_thinking=enable_thinking,
            tools=tools,
            tool_choice=tool_choice,
        )

        text = chat_result.get_text()
        reasoning_content, content = parse_thinking_content(text)
        output_tokens = _count_tokens(tokenizer, text)

        # 确定 finish_reason
        if chat_result.has_tool_calls:
            finish_reason = "tool_calls"
            tool_calls_data = chat_result.tool_calls
        else:
            finish_reason = _infer_finish_reason(output_tokens, max_tokens)
            tool_calls_data = None

        # 构建 Anthropic 响应
        content_blocks = _build_content_blocks(content, reasoning_content, tool_calls_data)
        stop_reason = _map_stop_reason(finish_reason)

        response = AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            model=model,
            content=content_blocks,
            stop_reason=stop_reason,
            usage=AnthropicUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

        await save_chat_log(
            model=model,
            params={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "stop": stop,
                "enable_thinking": enable_thinking,
                "stream": False,
                "tools": tools,
                "tool_choice": tool_choice,
                "api": "anthropic",
            },
            messages=cleaned_messages,
            assistant_content=text,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            request_payload=request_payload,
            response_payload=response.model_dump(mode="json", exclude_none=False),
        )

        return response

    except ModelServerException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Anthropic Messages API 推理失败: {}", str(e))
        return _anthropic_error_response(
            500, "api_error", f"Internal server error: {str(e)}"
        )


# ==================== 流式事件生成器 ====================


async def _stream_anthropic_events(
    req: AnthropicMessagesRequest,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    tool_choice: Any,
    stop: Optional[list[str]],
    enable_thinking: Optional[bool],
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    max_tokens: int,
    tokenizer: Any,
    input_tokens: int,
    request_payload: dict[str, Any],
):
    """生成 Anthropic 格式的 SSE 事件流

    事件序列：
    1. message_start    → 初始消息元数据（id, model, usage.input_tokens）
    2. ping             → 心跳
    3. content_block_start → 内容块开始（index, type）
    4. content_block_delta → 内容增量（text_delta / thinking_delta / input_json_delta）
    5. content_block_stop  → 内容块结束
    6. message_delta     → 结束元数据（stop_reason, usage.output_tokens）
    7. message_stop      → 流结束

    参数:
        req: 原始 Anthropic 请求（用于读取配置）
        model: 解析后的模型名称
        messages: 转换后的内部消息列表
        tools: OpenAI 格式的工具定义（可选）
        tool_choice: OpenAI 格式的工具选择策略
        stop: 停止序列
        enable_thinking: 是否启用深度思考
        temperature: 生成温度
        top_p: 核采样参数
        top_k: Top-K 采样参数
        max_tokens: 最大生成 token 数
        tokenizer: tokenizer 实例
        input_tokens: 输入 token 数量
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    stream_events: list[dict[str, Any]] = []

    def _sse(event_type: str, data: dict) -> str:
        stream_events.append({"event": event_type, "data": data})
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    # ---- 1. message_start ----
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })

    # ---- 2. ping ----
    yield _sse("ping", {"type": "ping"})

    # ---- 判断是否需要缓冲（工具调用场景） ----
    should_buffer_for_tools = bool(tools and tool_choice != "none")

    # ---- 3. 流式生成 ----
    full_text = ""
    block_index = 0
    current_field: Optional[str] = None  # "reasoning_content" | "content"
    splitter = ThinkingStreamSplitter()
    parsed_calls = None

    async for text_chunk in WORKER.generate_chat(
        model_name=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop,
        stream=True,
        enable_thinking=enable_thinking,
        tools=tools,
        tool_choice=tool_choice,
    ):
        if not text_chunk:
            continue
        full_text += text_chunk

        if not should_buffer_for_tools:
            for field, text in splitter.feed(text_chunk):
                if field != current_field:
                    # 关闭上一个 block
                    if current_field is not None:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": block_index,
                        })
                        block_index += 1

                    # 开启新 block
                    current_field = field
                    if field == "reasoning_content":
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    else:
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        })

                # 发送增量
                if field == "reasoning_content":
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": text},
                    })
                else:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })

    # ---- 4. 生成结束后的处理 ----
    finish_reason = "stop"

    if should_buffer_for_tools:
        # 工具缓冲模式：解析工具调用
        parsed_calls = CHAT_SERVICE.parse_tool_calls(model, full_text)

        if parsed_calls:
            # 解析 thinking + content
            reasoning_content, content_text = parse_thinking_content(full_text)

            # 回放 thinking block
            if reasoning_content:
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                })
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning_content},
                })
                yield _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block_index,
                })
                block_index += 1

            # 回放 text block（如果有非工具文本）
            # 从 content_text 中移除已解析的工具调用文本
            clean_text = CHAT_SERVICE.strip_tool_call_text(content_text, parsed_calls) if content_text else ""
            if clean_text:
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": clean_text},
                })
                yield _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block_index,
                })
                block_index += 1

            # 发送 tool_use blocks
            for tc in parsed_calls:
                fn = tc.get("function", {})
                try:
                    input_obj = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    input_obj = {}
                raw_id = tc.get("id", "")
                tool_id = raw_id.replace("call_", "toolu_", 1) if raw_id.startswith("call_") else raw_id
                if not tool_id:
                    tool_id = f"toolu_{uuid.uuid4().hex[:12]}"

                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": fn.get("name", ""),
                        "input": {},
                    },
                })
                # 以 input_json_delta 发送完整参数
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(input_obj, ensure_ascii=False),
                    },
                })
                yield _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block_index,
                })
                block_index += 1

            finish_reason = "tool_calls"
        else:
            # 无工具调用：回放缓冲文本
            replay_splitter = ThinkingStreamSplitter()
            for field, text in replay_splitter.feed(full_text):
                if field != current_field:
                    if current_field is not None:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": block_index,
                        })
                        block_index += 1
                    current_field = field
                    if field == "reasoning_content":
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    else:
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                if field == "reasoning_content":
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": text},
                    })
                else:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })
            # flush
            for field, text in replay_splitter.flush():
                if field != current_field:
                    if current_field is not None:
                        yield _sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": block_index,
                        })
                        block_index += 1
                    current_field = field
                    if field == "reasoning_content":
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    else:
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                if field == "reasoning_content":
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": text},
                    })
                else:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })
    else:
        # 非缓冲模式：刷新 splitter 缓冲区
        for field, text in splitter.flush():
            if field != current_field:
                if current_field is not None:
                    yield _sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": block_index,
                    })
                    block_index += 1
                current_field = field
                if field == "reasoning_content":
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })
                else:
                    yield _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
            if field == "reasoning_content":
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "thinking_delta", "thinking": text},
                })
            else:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": text},
                })

    # ---- 如果从未开启过任何 block，发送一个空 text block ----
    if current_field is None and not should_buffer_for_tools:
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "text", "text": ""},
        })
        current_field = "content"

    # ---- 关闭最后一个 block ----
    if current_field is not None:
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
        })

    # ---- 统计与日志 ----
    output_tokens = _count_tokens(tokenizer, full_text)

    if finish_reason == "stop":
        finish_reason = _infer_finish_reason(output_tokens, max_tokens)

    # ---- message_delta（结束信息）----
    stop_reason = _map_stop_reason(finish_reason)
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })

    # ---- message_stop ----
    yield _sse("message_stop", {"type": "message_stop"})

    reasoning_content, content = parse_thinking_content(full_text)

    await save_chat_log(
        model=model,
        params={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop,
            "enable_thinking": enable_thinking,
            "stream": True,
            "tools": tools,
            "tool_choice": tool_choice,
            "api": "anthropic",
        },
        messages=messages,
        assistant_content=full_text,
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        request_payload=request_payload,
        response_payload={
            "protocol": "anthropic_messages_stream",
            "stream": True,
            "stop_reason": stop_reason,
            "assistant": {
                "content": content,
                "reasoning_content": reasoning_content,
                "tool_calls": parsed_calls,
            },
            "events": stream_events,
        },
    )
