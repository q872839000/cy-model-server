"""聊天 API 数据结构定义

本模块定义聊天相关的请求/响应模型，完全兼容 OpenAI Chat Completions API 规范。
参考: https://platform.openai.com/docs/api-reference/chat/create
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional, Union


# ==================== 内容块与工具调用 ====================


class ChatContentBlock(BaseModel):
    """消息内容块（多模态支持）

    用于在 messages.content 中传递多模态信息，如文本、图片、音频等。
    当 content 为列表时，每个元素即为一个 ChatContentBlock。

    属性:
        type: 内容块类型标识。常见值: "text"(文本)、"image_url"(图片)、"input_text"(输入文本)
        text: 文本内容，仅当 type 为 "text" 或 "input_text" 时有效
        image_url: 图片信息，仅当 type 为 "image_url" 时有效，格式: {"url": "..."}
    """
    # extra="allow" 确保不会因为新的内容块类型而报 422
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="内容块类型，如 text/image_url/input_text")
    text: Optional[str] = Field(default=None, description="文本内容（type=text/input_text 时有效）")
    image_url: Optional[Dict[str, Any]] = Field(default=None, description="图片信息，格式: {url: str, detail?: str}")


class ToolCallFunction(BaseModel):
    """工具调用中的函数信息

    属性:
        name: 被调用的函数名称
        arguments: 函数参数，为 JSON 格式的字符串（由模型生成）
    """
    name: str = Field(..., description="被调用的函数名称")
    arguments: str = Field(..., description="函数参数，JSON 格式字符串")


class ToolCall(BaseModel):
    """工具调用对象

    当模型决定调用工具时，assistant 消息的 tool_calls 列表中的元素。

    属性:
        id: 工具调用的唯一标识，tool 角色消息需通过 tool_call_id 引用此 ID
        type: 工具类型，目前 OpenAI 仅支持 "function"
        function: 具体的函数调用信息（名称+参数）
    """
    id: str = Field(..., description="工具调用唯一标识，tool 消息通过 tool_call_id 引用")
    type: str = Field(default="function", description="工具类型，固定为 function")
    function: ToolCallFunction = Field(..., description="函数调用详情（名称+参数）")


class FunctionCall(BaseModel):
    """函数调用信息（已废弃，保留向后兼容）

    此模型用于兼容旧版 OpenAI API 的 function_call 字段。
    新代码应使用 ToolCall 模型代替。

    属性:
        name: 被调用的函数名称
        arguments: 函数参数，JSON 格式字符串
    """
    name: str = Field(..., description="被调用的函数名称")
    arguments: str = Field(..., description="函数参数，JSON 格式字符串")


# ==================== 消息模型 ====================


class ChatMessage(BaseModel):
    """聊天消息单元（完全兼容 OpenAI 规范）

    支持所有 OpenAI 消息角色和字段：
    - system/user/assistant/tool/function/developer
    - content 允许 null（如 assistant 带 tool_calls 时）
    - 支持多模态内容块
    """
    model_config = ConfigDict(extra="allow")

    role: str = Field(..., description="消息角色: system/user/assistant/tool/function/developer")
    content: Optional[Union[str, List[ChatContentBlock]]] = Field(
        default=None, description="消息内容，可为文本、内容块列表或 null"
    )
    name: Optional[str] = Field(default=None, description="发送者名称（可选）")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="assistant 请求的工具调用列表"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="tool 角色消息对应的工具调用 ID"
    )
    function_call: Optional[FunctionCall] = Field(
        default=None, description="（已废弃）函数调用信息"
    )
    refusal: Optional[str] = Field(
        default=None, description="模型拒绝回答的原因"
    )
    reasoning_content: Optional[str] = Field(
        default=None, description="推理/思考过程内容（深度思考模式时返回）"
    )


# ==================== 请求模型 ====================


class ChatCompletionStreamOptions(BaseModel):
    """流式输出选项配置

    OpenAI 兼容：用于控制流式返回时的额外行为
    """
    model_config = ConfigDict(extra="allow")

    include_usage: bool = Field(
        default=False,
        description="是否在流式输出最后一个 chunk 中包含 usage 统计信息",
    )


class ResponseFormat(BaseModel):
    """响应格式控制

    用于指定模型输出的格式要求。

    属性:
        type: 响应格式类型。
            - "text": 普通文本输出（默认）
            - "json_object": 强制输出合法 JSON
            - "json_schema": 按指定 JSON Schema 输出
        json_schema: 当 type="json_schema" 时，指定输出应遵循的 JSON Schema
    """
    model_config = ConfigDict(extra="allow")

    type: str = Field(default="text", description="响应格式类型: text/json_object/json_schema")
    json_schema: Optional[Dict[str, Any]] = Field(default=None, description="当 type=json_schema 时的 Schema 定义")


class ChatCompletionRequest(BaseModel):
    """聊天补全请求（完全兼容 OpenAI API 规范）

    所有 OpenAI 标准字段均已声明，确保任何兼容 OpenAI 的客户端/插件
    都不会因为多传字段而收到 422。

    参数处理策略：
    - 已实现：model, messages, max_tokens, temperature, top_p, stop, stream,
      stream_options, tools, tool_choice, functions, function_call, enable_thinking
    - 显式拒绝（传入非默认值时返回 400）：n>1, response_format(非text),
      parallel_tool_calls=false
    - 接受但忽略（传入非默认值时记录警告）：presence_penalty, frequency_penalty,
      logit_bias, seed, logprobs, top_logprobs
    - 透传忽略（不影响行为）：user, service_tier, store, metadata
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # ---- 必填字段 ----
    model: str = Field(..., description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")

    # ---- 生成参数 ----
    max_tokens: Optional[int] = Field(default=2048, description="最大生成 token 数")
    max_completion_tokens: Optional[int] = Field(
        default=None, description="兼容字段：优先生效时会覆盖 max_tokens"
    )
    temperature: Optional[float] = Field(default=0.0, ge=0, le=2, description="生成温度")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="核采样参数")
    n: Optional[int] = Field(default=1, ge=1, description="生成候选数量")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="停止序列")
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2, description="存在惩罚")
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2, description="频率惩罚")
    logit_bias: Optional[Dict[str, float]] = Field(default=None, description="token 偏置")
    logprobs: Optional[bool] = Field(default=None, description="是否返回 logprobs")
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20, description="返回的 top logprobs 数量")
    seed: Optional[int] = Field(default=None, description="随机种子（用于可复现生成）")

    # ---- 流式控制 ----
    stream: Optional[bool] = Field(default=False, description="是否流式输出")
    stream_options: Optional[ChatCompletionStreamOptions] = Field(
        default=None, description="流式输出选项"
    )

    # ---- 工具与函数调用 ----
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="可用工具列表")
    tool_choice: Optional[Any] = Field(default=None, description="工具选择策略")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="是否允许并行工具调用")
    functions: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="（已废弃）可用函数列表"
    )
    function_call: Optional[Any] = Field(
        default=None, description="（已废弃）函数调用策略"
    )

    # ---- 输出格式 ----
    response_format: Optional[ResponseFormat] = Field(default=None, description="响应格式")

    # ---- 其他标准字段 ----
    user: Optional[str] = Field(default=None, description="终端用户标识")
    service_tier: Optional[str] = Field(default=None, description="服务层级")
    store: Optional[bool] = Field(default=None, description="是否存储对话")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="请求元数据")

    # ---- 本服务扩展字段 ----
    enable_thinking: Optional[bool] = Field(default=None, description="是否启用深度思考模式（None 时使用模型配置默认值）")


# ==================== Usage ====================


class CompletionTokensDetails(BaseModel):
    """生成 token 的细分统计

    属性:
        reasoning_tokens: 推理过程消耗的 token 数（如 CoT 思维链部分）
    """
    reasoning_tokens: int = Field(default=0, description="推理过程消耗的 token 数")


class PromptTokensDetails(BaseModel):
    """提示 token 的细分统计

    属性:
        cached_tokens: 命中 KV Cache 的 token 数量（减少实际计算量）
    """
    cached_tokens: int = Field(default=0, description="命中 KV Cache 的 token 数")


class ChatCompletionUsage(BaseModel):
    """Token 使用量统计（兼容 OpenAI 规范）

    用于记录本次对话的 token 消耗情况，可用于计费和监控。

    属性:
        prompt_tokens: 输入提示消耗的 token 数量
        completion_tokens: 模型生成回复消耗的 token 数量
        total_tokens: 总 token 数量 (prompt_tokens + completion_tokens)
        completion_tokens_details: 生成 token 的细分统计（推理 token 等）
        prompt_tokens_details: 提示 token 的细分统计（缓存命中等）
    """
    prompt_tokens: int = Field(default=0, description="输入提示消耗的 token 数")
    completion_tokens: int = Field(default=0, description="模型生成回复消耗的 token 数")
    total_tokens: int = Field(default=0, description="总 token 数 (prompt + completion)")
    completion_tokens_details: Optional[CompletionTokensDetails] = Field(
        default=None, description="生成 token 细分统计"
    )
    prompt_tokens_details: Optional[PromptTokensDetails] = Field(
        default=None, description="提示 token 细分统计"
    )


# ==================== 非流式响应 ====================


class ChatCompletionChoice(BaseModel):
    """非流式响应中的单条生成结果

    属性:
        index: 结果在 choices 数组中的序号（从 0 开始）
        message: 模型生成的完整消息体（含 role 和 content）
        finish_reason: 生成结束原因:
            - "stop": 自然结束或命中停止序列
            - "length": 达到 max_tokens 限制
            - "tool_calls": 模型请求调用工具
            - "content_filter": 内容被安全过滤
        logprobs: token 级别的对数概率信息（当请求中 logprobs=true 时返回）
    """
    index: int = Field(default=0, description="结果在 choices 数组中的序号")
    message: ChatMessage = Field(..., description="模型生成的完整消息体")
    finish_reason: Optional[str] = Field(default="stop", description="生成结束原因: stop/length/tool_calls/content_filter")
    logprobs: Optional[Any] = Field(default=None, description="token 级别的对数概率信息")


class ChatCompletionResponse(BaseModel):
    """聊天补全响应（完全兼容 OpenAI 规范）

    非流式模式下返回的完整响应结构。

    属性:
        id: 本次响应的唯一标识，格式: "chatcmpl-{hex}"
        object: 对象类型标识，固定为 "chat.completion"
        created: 响应创建的 Unix 时间戳（秒）
        model: 实际执行推理的模型名称
        choices: 生成结果列表，每个元素对应一个候选回复
        usage: Token 使用量统计信息
        system_fingerprint: 系统指纹，用于追踪后端配置变化（可复现性）
        service_tier: 服务层级标识
    """
    id: str = Field(..., description="响应唯一标识，格式: chatcmpl-{hex}")
    object: str = Field(default="chat.completion", description="对象类型，固定为 chat.completion")
    created: int = Field(..., description="响应创建的 Unix 时间戳（秒）")
    model: str = Field(..., description="实际执行推理的模型名称")
    choices: List[ChatCompletionChoice] = Field(..., description="生成结果列表")
    usage: Optional[ChatCompletionUsage] = Field(
        default_factory=ChatCompletionUsage, description="Token 使用量统计"
    )
    system_fingerprint: Optional[str] = Field(
        default=None, description="系统指纹，用于追踪后端配置变化"
    )
    service_tier: Optional[str] = Field(default=None, description="服务层级标识")


# ==================== 流式响应 ====================


class ToolCallChunkFunction(BaseModel):
    """流式工具调用中的函数增量

    在流式输出中，函数名称和参数会被拆分为多个增量 chunk 逐步推送。

    属性:
        name: 函数名称（通常仅在首个 chunk 中出现）
        arguments: 函数参数的增量片段（逐步拼接为完整 JSON）
    """
    name: Optional[str] = Field(default=None, description="函数名称（首个 chunk 中出现）")
    arguments: Optional[str] = Field(default=None, description="函数参数增量片段")


class ToolCallChunk(BaseModel):
    """流式工具调用增量

    流式输出中 delta.tool_calls 数组的元素，逐步推送工具调用信息。

    属性:
        index: 工具调用在 tool_calls 数组中的序号
        id: 工具调用唯一标识（仅在首个 chunk 中出现）
        type: 工具类型（仅在首个 chunk 中出现，固定为 "function"）
        function: 函数调用的增量信息
    """
    index: int = Field(..., description="工具调用在 tool_calls 数组中的序号")
    id: Optional[str] = Field(default=None, description="工具调用唯一标识（首个 chunk）")
    type: Optional[str] = Field(default=None, description="工具类型（首个 chunk，固定为 function）")
    function: Optional[ToolCallChunkFunction] = Field(default=None, description="函数调用增量信息")


class ChatCompletionDelta(BaseModel):
    """流式输出的增量内容

    SSE 流式输出中每个 chunk 的 choices[i].delta 字段。
    首个 chunk 通常只包含 role，后续 chunk 包含 content 增量。

    属性:
        role: 消息角色，通常仅在首个 chunk 中出现（"assistant"）
        content: 增量文本内容，每个 chunk 推送部分生成的文本
        tool_calls: 增量工具调用列表（当模型请求调用工具时）
        function_call: （已废弃）增量函数调用信息
        refusal: 模型拒绝回答的原因
    """
    role: Optional[str] = Field(default=None, description="消息角色（仅首个 chunk 出现）")
    content: Optional[str] = Field(default=None, description="增量文本内容")
    tool_calls: Optional[List[ToolCallChunk]] = Field(
        default=None, description="增量工具调用列表"
    )
    function_call: Optional[FunctionCall] = Field(
        default=None, description="（已废弃）增量函数调用信息"
    )
    refusal: Optional[str] = Field(default=None, description="模型拒绝回答的原因")
    reasoning_content: Optional[str] = Field(
        default=None, description="增量推理/思考内容（深度思考模式的流式输出）"
    )


class ChatCompletionChunkChoice(BaseModel):
    """流式输出中的单个选择项

    属性:
        index: 结果在 choices 数组中的序号
        delta: 本次 chunk 的增量内容
        finish_reason: 生成结束原因，仅在最后一个内容 chunk 中出现
            ("stop"/"length"/"tool_calls"/"content_filter")，其余 chunk 为 null
        logprobs: token 级别的对数概率信息
    """
    index: int = Field(default=0, description="结果在 choices 数组中的序号")
    delta: ChatCompletionDelta = Field(..., description="本次 chunk 的增量内容")
    finish_reason: Optional[str] = Field(default=None, description="生成结束原因（仅最后一个内容 chunk）")
    logprobs: Optional[Any] = Field(default=None, description="token 级别的对数概率信息")


class ChatCompletionChunkResponse(BaseModel):
    """流式聊天补全响应的单个 chunk（完全兼容 OpenAI 规范）

    SSE 流式输出中每个 "data: {...}" 行的数据结构。
    同一次对话的所有 chunk 共享相同的 id 和 created。

    属性:
        id: 响应唯一标识，同一次对话的所有 chunk 使用相同 ID
        object: 对象类型，固定为 "chat.completion.chunk"
        created: 响应创建的 Unix 时间戳（秒）
        model: 实际执行推理的模型名称
        choices: 生成结果列表（流式结束时的 usage chunk 中为空列表）
        usage: Token 使用量统计，仅在 stream_options.include_usage=true 时的最终 chunk 中出现
        system_fingerprint: 系统指纹，用于追踪后端配置变化
        service_tier: 服务层级标识
    """
    id: str = Field(..., description="响应唯一标识（同一对话的所有 chunk 共享）")
    object: str = Field(default="chat.completion.chunk", description="对象类型，固定为 chat.completion.chunk")
    created: int = Field(..., description="响应创建的 Unix 时间戳（秒）")
    model: str = Field(..., description="实际执行推理的模型名称")
    choices: List[ChatCompletionChunkChoice] = Field(..., description="生成结果列表（usage chunk 中为空列表）")
    usage: Optional[ChatCompletionUsage] = Field(default=None, description="Token 使用量（仅 include_usage=true 的最终 chunk）")
    system_fingerprint: Optional[str] = Field(
        default=None, description="系统指纹，用于追踪后端配置变化"
    )
    service_tier: Optional[str] = Field(default=None, description="服务层级标识")


# ==================== 模型列表 ====================


class ModelInfo(BaseModel):
    """单个模型信息（兼容 OpenAI /v1/models 规范）

    属性:
        id: 模型唯一标识符（即模型名称）
        object: 对象类型，固定为 "model"
        created: 模型注册的 Unix 时间戳（秒）
        owned_by: 模型所有者/提供方标识
    """
    id: str = Field(..., description="模型唯一标识符")
    object: str = Field(default="model", description="对象类型，固定为 model")
    created: int = Field(..., description="模型注册的 Unix 时间戳（秒）")
    owned_by: str = Field(default="cy-model-server", description="模型所有者/提供方标识")


class ModelListResponse(BaseModel):
    """模型列表响应

    GET /v1/models 接口的返回结构。

    属性:
        object: 对象类型，固定为 "list"
        data: 可用模型信息列表
    """
    object: str = Field(default="list", description="对象类型，固定为 list")
    data: List[ModelInfo] = Field(..., description="可用模型信息列表")


# ==================== 前向引用重建 ====================

ChatCompletionRequest.model_rebuild()
