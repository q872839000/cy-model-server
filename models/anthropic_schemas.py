"""Anthropic Messages API 数据结构定义

本模块定义 Anthropic Messages API 的请求/响应模型，用于兼容 Claude Code 等
仅支持 Anthropic API 规范的客户端工具。

参考: https://docs.anthropic.com/en/api/messages

与 OpenAI API 的核心差异：
- system prompt 是顶层字段而非消息角色
- content 使用 type-discriminated 内容块（text/tool_use/tool_result/thinking）
- tools 使用 input_schema 而非 parameters
- 流式采用命名事件（message_start/content_block_delta 等）而非统一 data 行
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional, Union


# ==================== 请求：内容块 ====================


class AnthropicContentBlock(BaseModel):
    """Anthropic 消息内容块

    支持多种内容类型，通过 type 字段区分：
    - text: 文本内容
    - image: 图片内容（base64）
    - tool_use: 工具调用（assistant 消息中）
    - tool_result: 工具返回结果（user 消息中）
    - thinking: 深度思考内容（assistant 消息中）

    属性:
        type: 内容块类型标识
        text: 文本内容（type=text 时有效）
        source: 图片来源信息（type=image 时有效）
        id: 工具调用唯一标识（type=tool_use 时有效）
        name: 工具名称（type=tool_use 时有效）
        input: 工具输入参数（type=tool_use 时有效）
        tool_use_id: 关联的工具调用 ID（type=tool_result 时有效）
        content: 工具返回内容（type=tool_result 时有效）
        is_error: 工具调用是否出错（type=tool_result 时有效）
        thinking: 思考内容（type=thinking 时有效）
    """
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="内容块类型: text/image/tool_use/tool_result/thinking")
    # text block
    text: Optional[str] = Field(default=None, description="文本内容")
    # image block
    source: Optional[Dict[str, Any]] = Field(default=None, description="图片来源信息")
    # tool_use block（assistant 消息中的工具调用）
    id: Optional[str] = Field(default=None, description="工具调用唯一标识")
    name: Optional[str] = Field(default=None, description="工具名称")
    input: Optional[Any] = Field(default=None, description="工具输入参数")
    # tool_result block（user 消息中的工具返回）
    tool_use_id: Optional[str] = Field(default=None, description="关联的工具调用 ID")
    content: Optional[Union[str, List[Any]]] = Field(default=None, description="工具返回内容")
    is_error: Optional[bool] = Field(default=None, description="工具调用是否出错")
    # thinking block
    thinking: Optional[str] = Field(default=None, description="思考内容")


# ==================== 请求：消息与工具定义 ====================


class AnthropicMessage(BaseModel):
    """Anthropic 对话消息

    与 OpenAI 的核心差异：
    - role 仅支持 "user" 和 "assistant"（system 在顶层）
    - content 可以是字符串或内容块数组（混合 text/tool_use/tool_result）

    属性:
        role: 消息角色，仅支持 user/assistant
        content: 消息内容，字符串或内容块数组
    """
    model_config = ConfigDict(extra="allow")

    role: str = Field(..., description="消息角色: user/assistant")
    content: Union[str, List[AnthropicContentBlock]] = Field(
        ..., description="消息内容，可为文本或内容块数组"
    )


class AnthropicToolDef(BaseModel):
    """Anthropic 工具定义

    与 OpenAI 的核心差异：使用 input_schema 而非 parameters

    属性:
        name: 工具名称
        description: 工具功能描述
        input_schema: 输入参数的 JSON Schema 定义
    """
    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="工具名称")
    description: Optional[str] = Field(default=None, description="工具功能描述")
    input_schema: Dict[str, Any] = Field(..., description="输入参数的 JSON Schema 定义")


class AnthropicThinkingConfig(BaseModel):
    """Anthropic 深度思考配置

    属性:
        type: 启用状态，"enabled" 或 "disabled"
        budget_tokens: 思考过程的 token 预算（本地模型中仅记录警告，无法精确控制）
    """
    type: str = Field(default="enabled", description="启用状态: enabled/disabled")
    budget_tokens: Optional[int] = Field(
        default=None, description="思考 token 预算（本地模型中仅记录警告）"
    )


class AnthropicToolChoice(BaseModel):
    """Anthropic 工具选择策略

    与 OpenAI 的映射关系：
    - {"type": "auto"} → "auto"
    - {"type": "any"} → "required"
    - {"type": "tool", "name": "xxx"} → {"type": "function", "function": {"name": "xxx"}}

    属性:
        type: 选择策略类型: auto/any/tool
        name: 指定工具名称（type=tool 时必填）
        disable_parallel_tool_use: 是否禁用并行工具调用
    """
    model_config = ConfigDict(extra="allow")

    type: str = Field(default="auto", description="工具选择策略: auto/any/tool")
    name: Optional[str] = Field(default=None, description="指定工具名称（type=tool 时）")
    disable_parallel_tool_use: Optional[bool] = Field(
        default=None, description="是否禁用并行工具调用"
    )


# ==================== 请求体 ====================


class AnthropicMessagesRequest(BaseModel):
    """Anthropic POST /v1/messages 请求体

    与 OpenAI ChatCompletionRequest 的核心差异：
    - max_tokens 为必填（OpenAI 可选）
    - system 是顶层字段（OpenAI 在 messages 中）
    - tools 使用 input_schema（OpenAI 使用 parameters）
    - stop_sequences 替代 stop
    - thinking 替代 enable_thinking
    - 无 n/response_format/logprobs 等字段

    属性:
        model: 模型名称
        max_tokens: 最大生成 token 数（必填）
        messages: 对话消息列表
        system: 系统提示，字符串或内容块数组
        temperature: 生成温度
        top_p: 核采样参数
        top_k: Top-K 采样参数
        stop_sequences: 停止序列列表
        stream: 是否流式输出
        tools: 可用工具列表
        tool_choice: 工具选择策略
        metadata: 请求元数据
        thinking: 深度思考配置
    """
    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="模型名称")
    max_tokens: Optional[int] = Field(default=None, description="最大生成 token 数（未传时使用模型 gen_params.max_tokens，仍未配置则兜底 8192）")
    messages: List[AnthropicMessage] = Field(..., description="对话消息列表")
    system: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        default=None, description="系统提示，字符串或内容块数组"
    )
    temperature: Optional[float] = Field(default=None, description="生成温度")
    top_p: Optional[float] = Field(default=None, description="核采样参数")
    top_k: Optional[int] = Field(default=None, description="Top-K 采样参数")
    stop_sequences: Optional[List[str]] = Field(default=None, description="停止序列列表")
    stream: Optional[bool] = Field(default=False, description="是否流式输出")
    tools: Optional[List[AnthropicToolDef]] = Field(default=None, description="可用工具列表")
    tool_choice: Optional[AnthropicToolChoice] = Field(
        default=None, description="工具选择策略"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="请求元数据")
    thinking: Optional[AnthropicThinkingConfig] = Field(
        default=None, description="深度思考配置"
    )


# ==================== 响应：Usage ====================


class AnthropicUsage(BaseModel):
    """Anthropic token 使用量统计

    与 OpenAI 的差异：
    - input_tokens 替代 prompt_tokens
    - output_tokens 替代 completion_tokens
    - 无 total_tokens 字段

    属性:
        input_tokens: 输入 token 数量
        output_tokens: 输出 token 数量
    """
    input_tokens: int = Field(default=0, description="输入 token 数量")
    output_tokens: int = Field(default=0, description="输出 token 数量")


# ==================== 响应：内容块 ====================


class AnthropicResponseContentBlock(BaseModel):
    """Anthropic 响应内容块

    非流式响应的 content 数组元素，通过 type 区分：
    - text: 文本回复
    - tool_use: 工具调用请求
    - thinking: 深度思考过程

    属性:
        type: 内容块类型: text/tool_use/thinking
        text: 文本内容（type=text）
        id: 工具调用 ID（type=tool_use）
        name: 工具名称（type=tool_use）
        input: 工具输入参数（type=tool_use）
        thinking: 思考内容（type=thinking）
    """
    type: str = Field(..., description="内容块类型: text/tool_use/thinking")
    # text block
    text: Optional[str] = Field(default=None, description="文本内容")
    # tool_use block
    id: Optional[str] = Field(default=None, description="工具调用 ID")
    name: Optional[str] = Field(default=None, description="工具名称")
    input: Optional[Any] = Field(default=None, description="工具输入参数")
    # thinking block
    thinking: Optional[str] = Field(default=None, description="思考内容")


# ==================== 响应体 ====================


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API 非流式响应

    与 OpenAI ChatCompletionResponse 的核心差异：
    - 无 choices 数组，直接返回 content 内容块列表
    - stop_reason 替代 finish_reason（值也不同）
    - usage 使用 input_tokens/output_tokens

    stop_reason 值映射：
    - "end_turn" ← OpenAI "stop"
    - "max_tokens" ← OpenAI "length"
    - "tool_use" ← OpenAI "tool_calls"
    - "stop_sequence" ← 命中停止序列时

    属性:
        id: 消息唯一标识，格式: "msg_{hex}"
        type: 对象类型，固定为 "message"
        role: 角色，固定为 "assistant"
        model: 执行推理的模型名称
        content: 响应内容块列表
        stop_reason: 生成停止原因
        stop_sequence: 触发停止的序列（stop_reason=stop_sequence 时有值）
        usage: token 使用量统计
    """
    id: str = Field(..., description="消息唯一标识，格式: msg_{hex}")
    type: str = Field(default="message", description="对象类型，固定为 message")
    role: str = Field(default="assistant", description="角色，固定为 assistant")
    model: str = Field(..., description="执行推理的模型名称")
    content: List[AnthropicResponseContentBlock] = Field(
        ..., description="响应内容块列表"
    )
    stop_reason: Optional[str] = Field(
        default=None,
        description="生成停止原因: end_turn/max_tokens/stop_sequence/tool_use",
    )
    stop_sequence: Optional[str] = Field(
        default=None, description="触发停止的序列"
    )
    usage: AnthropicUsage = Field(..., description="token 使用量统计")


# ==================== 错误响应 ====================


class AnthropicErrorDetail(BaseModel):
    """Anthropic 错误详情

    属性:
        type: 错误类型，如 invalid_request_error/authentication_error/not_found_error 等
        message: 错误描述信息
    """
    type: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误描述信息")


class AnthropicErrorResponse(BaseModel):
    """Anthropic 错误响应

    与 OpenAI 错误格式的差异：
    - OpenAI: {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}
    - Anthropic: {"type": "error", "error": {"type": ..., "message": ...}}

    属性:
        type: 固定为 "error"
        error: 错误详情
    """
    type: str = Field(default="error", description="固定为 error")
    error: AnthropicErrorDetail = Field(..., description="错误详情")
