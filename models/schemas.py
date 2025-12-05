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


# === Retrieval / Collection Schemas ===

class Document(BaseModel):
    """检索库内的文档对象。
    - id: 文档唯一标识
    - content: 文档正文
    - metadata: 元数据（来源、标签等）
    """
    id: str = Field(..., description="文档唯一标识")
    content: str = Field(..., description="文档正文内容")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据，可选")


class CreateCollectionRequest(BaseModel):
    """
    创建向量集合请求
    
    属性:
        name: 集合名称（唯一标识）
        embedding_model: 集合绑定的embedding模型名称
        description: 集合描述信息，可选
            特殊格式：在描述末尾添加 [schema_type=vector_resource] 可指定使用VectorResource schema
            示例："知识库集合 [schema_type=vector_resource]"
    """
    name: str = Field(..., description="集合名称")
    embedding_model: str = Field(..., description="集合绑定的 embedding 模型")
    description: Optional[str] = Field(None, description="集合描述信息，可在末尾添加 [schema_type=xxx] 指定schema类型")


class CollectionInfo(BaseModel):
    """向量集合信息。"""
    name: str = Field(..., description="集合名称")
    document_count: int = Field(..., description="文档数量")
    embedding_model: str = Field(..., description="集合绑定的 embedding 模型")
    dimension: Optional[int] = Field(None, description="向量维度")


class AddDocumentsRequest(BaseModel):
    """批量添加文档请求。"""
    collection: str = Field(..., description="目标集合名称")
    documents: List[Document] = Field(..., description="待入库文档列表")
    embedding_model: str = Field(..., description="用于生成向量的 embedding 模型")


class RetrievalQuery(BaseModel):
    """向量检索请求。
    - query: 查询语句
    - embedding_model: 用于查询向量化的 embedding 模型
    - collection: 目标集合名
    - top_k: 返回前 K 条
    - similarity_threshold: 相似度阈值（可选，COSINE相似度，范围[-1,1]，实际常用[0,1]）
    """
    query: str = Field(..., description="检索查询文本")
    embedding_model: str = Field(..., description="用于向量化查询的 embedding 模型")
    collection: str = Field(..., description="检索的目标集合名称")
    top_k: int = Field(10, description="返回前 K 条候选")
    similarity_threshold: Optional[float] = Field(
        None, 
        description="COSINE相似度阈值（-1到1，1表示完全相同，0表示正交，-1表示完全相反，实际常用范围0-1）",
        ge=-1.0,
        le=1.0
    )


class RetrievalItem(BaseModel):
    """检索结果项。"""
    document: Document = Field(..., description="命中文档对象")
    score: float = Field(..., description="检索分数/相似度，越大越相关")


class RetrievalResponse(BaseModel):
    """向量检索响应。"""
    items: List[RetrievalItem] = Field(..., description="检索结果列表")
    query: str = Field(..., description="原始查询文本")
    embedding_model: str = Field(..., description="实际使用的 embedding 模型")
    collection: str = Field(..., description="检索的集合名称")
