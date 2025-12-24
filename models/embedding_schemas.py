"""
向量化 API 数据结构定义

本模块定义 Embedding 相关的请求/响应模型，用于 OpenAI 兼容的 Embeddings API。
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class EmbeddingRequest(BaseModel):
    """向量化请求。
    - model: 指定使用的 embedding 模型名
    - input: 待向量化的文本列表
    """
    model: str = Field(..., description="Embedding 模型名称")
    input: List[str] = Field(..., description="待向量化的文本列表")


class EmbeddingData(BaseModel):
    """单条向量结果。"""
    index: int = Field(..., description="输入文本在请求中的序号")
    embedding: List[float] = Field(..., description="向量数值数组")


class EmbeddingResponse(BaseModel):
    """向量化响应。"""
    object: str = Field('list', description="固定字段")
    data: List[EmbeddingData] = Field(..., description="向量结果列表")
    model: str = Field(..., description="实际使用的 embedding 模型")
    dimension: Optional[int] = Field(None, description="向量维度（由模型决定）")
