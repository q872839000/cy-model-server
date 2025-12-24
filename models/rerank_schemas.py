"""
重排序 API 数据结构定义

本模块定义 Reranker 相关的请求/响应模型，用于文档重排序 API。
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class RerankRequest(BaseModel):
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
