"""
向量化 API 数据结构定义

本模块定义 Embedding 相关的请求/响应模型，完全兼容 OpenAI Embeddings API 规范。
参考: https://platform.openai.com/docs/api-reference/embeddings
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Union


class EmbeddingRequest(BaseModel):
    """向量化请求（完全兼容 OpenAI Embeddings API 规范）

    input 支持 OpenAI 规范的所有格式：
    - 单个字符串: "hello world"
    - 字符串列表: ["hello", "world"]
    - token ID 数组: [15339, 1917]
    - token ID 二维数组: [[15339, 1917]]

    属性:
        model: Embedding 模型名称
        input: 待向量化的输入，支持多种格式
        encoding_format: 向量输出编码格式，"float"(默认) 或 "base64"
        dimensions: 输出向量维度，仅部分模型支持维度截断
        user: 终端用户标识，用于监控和滥用检测
    """
    # extra="allow" 确保不会因为未知字段而报 422
    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Embedding 模型名称")
    input: Union[str, List[str], List[int], List[List[int]]] = Field(
        ..., description="待向量化的输入，支持字符串/字符串列表/token ID 数组"
    )
    encoding_format: Optional[str] = Field(
        default=None, description="向量输出编码格式: float(默认)/base64"
    )
    dimensions: Optional[int] = Field(
        default=None, description="输出向量维度（仅部分模型支持维度截断）"
    )
    user: Optional[str] = Field(default=None, description="终端用户标识，用于监控和滥用检测")


class EmbeddingData(BaseModel):
    """单条向量结果（兼容 OpenAI 规范）

    属性:
        object: 对象类型，固定为 "embedding"
        index: 该向量对应的输入文本在请求 input 中的序号（从 0 开始）
        embedding: 向量数值数组，维度由模型决定
    """
    object: str = Field(default="embedding", description="对象类型，固定为 embedding")
    index: int = Field(..., description="对应输入文本在 input 中的序号")
    embedding: List[float] = Field(..., description="向量数值数组，维度由模型决定")


class EmbeddingUsage(BaseModel):
    """Embedding 接口的 token 使用量统计

    注意: Embedding 接口的 usage 不包含 completion_tokens，
    因为 embedding 不涉及文本生成。

    属性:
        prompt_tokens: 输入文本消耗的 token 数量
        total_tokens: 总 token 数量（等于 prompt_tokens）
    """
    prompt_tokens: int = Field(default=0, description="输入文本消耗的 token 数")
    total_tokens: int = Field(default=0, description="总 token 数（等于 prompt_tokens）")


class EmbeddingResponse(BaseModel):
    """向量化响应（兼容 OpenAI 规范）

    POST /v1/embeddings 接口的返回结构。

    属性:
        object: 对象类型，固定为 "list"
        data: 向量结果列表，与输入 input 一一对应
        model: 实际执行向量化的模型名称
        usage: Token 使用量统计
    """
    object: str = Field(default="list", description="对象类型，固定为 list")
    data: List[EmbeddingData] = Field(..., description="向量结果列表，与输入 input 一一对应")
    model: str = Field(..., description="实际执行向量化的模型名称")
    usage: EmbeddingUsage = Field(
        default_factory=EmbeddingUsage, description="Token 使用量统计"
    )
