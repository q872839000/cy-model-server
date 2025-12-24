"""
聊天 API 数据结构定义

本模块定义聊天相关的请求/响应模型，用于 OpenAI 兼容的 Chat Completions API。
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    """聊天消息单元。
    - role: 角色标识，支持 "system" | "user" | "assistant"
    - content: 消息文本内容
    """
    role: str = Field(..., description="消息角色，system/user/assistant 其中之一")
    content: str = Field(..., description="消息正文内容")


class ChatCompletionRequest(BaseModel):
    """
    聊天补全请求

    属性:
        model: 使用的LLM模型名称
        messages: 对话历史消息列表
        max_tokens: 最大生成token数，默认2048
        temperature: 生成温度(0-2)，越高越随机，默认0.0
        top_p: 核采样参数(0-1)，默认1.0
        stream: 是否使用流式输出，默认False
        enable_thinking: 是否启用深度思考模式，默认True
    """
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    # OpenAI 兼容：用于控制流式返回的附加行为（如 include_usage）
    stream_options: Optional["ChatCompletionStreamOptions"] = None
    enable_thinking: bool = True


class ChatCompletionStreamOptions(BaseModel):
    """
    流式输出选项配置
    
    OpenAI 兼容：用于控制流式返回时的额外行为
    
    属性:
        include_usage: 是否在流式输出结束时额外发送包含usage统计的chunk
    """
    include_usage: bool = False


class ChatCompletionUsage(BaseModel):
    """
    Token 使用量统计
    
    OpenAI 兼容：记录本次对话的token消耗情况，用于计费和监控
    
    属性:
        prompt_tokens: 输入提示消耗的token数量
        completion_tokens: 生成回复消耗的token数量  
        total_tokens: 总token数量 (prompt_tokens + completion_tokens)
    """
    prompt_tokens: int = Field(default=0, description="提示token数")
    completion_tokens: int = Field(default=0, description="生成token数")
    total_tokens: int = Field(default=0, description="总token数")


class ChatCompletionChoice(BaseModel):
    """单条生成结果项。
    - index: 结果序号
    - message: 生成的消息
    - finish_reason: 结束原因（如 stop、length 等）
    """
    index: int = Field(..., description="结果序号")
    message: ChatMessage = Field(..., description="生成的消息体")
    finish_reason: str = Field(..., description="结束原因，示例：stop/length")


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="响应ID")
    # OpenAI 兼容：非流式返回 object 固定为 chat.completion
    object: str = Field(default="chat.completion", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="实际使用的模型名称")
    choices: List[ChatCompletionChoice] = Field(..., description="生成结果列表")
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage, description="Token使用量")


class ChatCompletionDelta(BaseModel):
    """
    流式输出的增量内容
    
    OpenAI 兼容：流式chunk中的delta字段，包含本次推送的增量信息
    
    属性:
        role: 角色标识，通常只在第一个chunk中出现
        content: 增量文本内容，每个chunk推送部分生成的文本
    """
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """
    流式输出中的单个选择项
    
    OpenAI 兼容：流式chunk中choices数组的元素，包含增量内容和状态
    
    属性:
        index: 选择项序号，通常为0
        delta: 本次chunk的增量内容
        finish_reason: 结束原因，仅在最后一个content chunk中出现（如stop/length）
    """
    index: int = Field(..., description="结果序号")
    delta: ChatCompletionDelta = Field(..., description="增量内容")
    finish_reason: Optional[str] = Field(default=None, description="结束原因")


class ChatCompletionChunkResponse(BaseModel):
    """
    流式聊天补全响应的单个chunk
    
    OpenAI 兼容：Server-Sent Events (SSE) 流式输出的数据结构
    
    属性:
        id: 响应会话ID，同一次对话的所有chunk使用相同ID
        object: 对象类型标识，流式输出固定为"chat.completion.chunk"
        created: 响应创建的Unix时间戳
        model: 实际执行推理的模型名称
        choices: 生成选择列表，通常只包含一个元素
        usage: Token使用统计，仅在stream_options.include_usage=true且为最后一个chunk时出现
    """
    id: str = Field(..., description="响应ID")
    object: str = Field(default="chat.completion.chunk", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="实际使用的模型名称")
    choices: List[ChatCompletionChunkChoice] = Field(..., description="生成结果列表")
    usage: Optional[ChatCompletionUsage] = Field(default=None, description="Token使用量")


class ModelInfo(BaseModel):
    """
    单个模型信息
    
    OpenAI 兼容：/v1/models 接口返回的单个模型描述
    
    属性:
        id: 模型唯一标识符
        object: 对象类型，固定为"model"
        created: 模型创建时间戳
        owned_by: 模型所有者/提供方
    """
    id: str = Field(..., description="模型ID")
    object: str = Field(default="model", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    owned_by: str = Field(..., description="模型所有者")


class ModelListResponse(BaseModel):
    """
    模型列表响应
    
    OpenAI 兼容：/v1/models 接口的完整响应结构
    
    属性:
        object: 对象类型，固定为"list"
        data: 模型信息列表
    """
    object: str = Field(default="list", description="对象类型")
    data: List[ModelInfo] = Field(..., description="模型列表")


ChatCompletionRequest.model_rebuild()
