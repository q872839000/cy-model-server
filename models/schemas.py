from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class ChatMessage(BaseModel):
    """聊天消息单元。
    - role: 角色标识，支持 "system" | "user" | "assistant"
    - content: 消息文本内容
    """
    role: str = Field(..., description="消息角色，system/user/assistant 其中之一")
    content: str = Field(..., description="消息正文内容")


class ChatRequest(BaseModel):
    """聊天生成请求。
    - model: 指定使用的对话模型名（留空则使用默认）
    - messages: 聊天历史消息，按顺序排列
    - max_tokens: 生成的最大新词元数
    - temperature: 采样温度，越高越发散
    - top_p: nucleus sampling 截断阈值
    """
    model: str = Field(..., description="模型名称；为空则使用默认模型")
    messages: List[ChatMessage] = Field(..., description="聊天历史消息列表")
    max_tokens: int = Field(256, description="最大生成词元数")
    temperature: float = Field(0.7, description="采样温度，0-1 之间")
    top_p: float = Field(0.95, description="Top-P 截断阈值")


class Choice(BaseModel):
    """单条生成结果项。
    - index: 结果序号
    - message: 生成的消息
    - finish_reason: 结束原因（如 stop、length 等）
    """
    index: int = Field(..., description="结果序号")
    message: ChatMessage = Field(..., description="生成的消息体")
    finish_reason: str = Field(..., description="结束原因，示例：stop/length")


class ChatResponse(BaseModel):
    """聊天生成响应。"""
    id: str = Field(..., description="请求/会话唯一标识")
    model: str = Field(..., description="实际使用的模型名称")
    choices: List[Choice] = Field(..., description="生成结果列表")


class EmbeddingInput(BaseModel):
    """向量化请求。
    - model: 指定使用的 embedding 模型名
    - input: 待向量化的文本列表
    """
    model: str = Field(..., description="Embedding 模型名称")
    input: List[str] = Field(..., description="待向量化的文本列表")


class EmbeddingVector(BaseModel):
    """单条向量结果。"""
    index: int = Field(..., description="输入文本在请求中的序号")
    embedding: List[float] = Field(..., description="向量数值数组")


class EmbeddingResponse(BaseModel):
    """向量化响应。"""
    data: List[EmbeddingVector] = Field(..., description="向量结果列表")
    model: str = Field(..., description="实际使用的 embedding 模型")
    dimension: Optional[int] = Field(None, description="向量维度（由模型决定）")


class RerankQuery(BaseModel):
    """重排序请求。
    - model: 指定使用的 reranker 模型名（为空则默认）
    - query: 查询语句
    - documents: 候选文档列表（纯文本）
    - top_k: 可选，返回前 K 条
    """
    model: str = Field(..., description="Reranker 模型名称；为空则使用默认")
    query: str = Field(..., description="查询语句")
    documents: List[str] = Field(..., description="候选文档文本列表")
    top_k: Optional[int] = Field(None, description="返回前 K 条（可选）")


class RerankItem(BaseModel):
    """重排序后的单条结果。"""
    index: int = Field(..., description="原始文档序号")
    document: str = Field(..., description="文档内容")
    score: float = Field(..., description="相关性分数，越大越相关")


class RerankResponse(BaseModel):
    """重排序响应。"""
    items: List[RerankItem] = Field(..., description="重排序后的结果集合")
    model: str = Field(..., description="实际使用的 reranker 模型")