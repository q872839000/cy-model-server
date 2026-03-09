# Models package
"""
数据模型包

本包提供 API 请求/响应的数据结构定义：
- chat_schemas: 聊天 API 数据结构
- embedding_schemas: 向量化 API 数据结构
- rerank_schemas: 重排序 API 数据结构
- kb_schemas: 知识库相关数据结构
"""

from models.chat_schemas import (
    ChatContentBlock,
    ToolCallFunction,
    ToolCall,
    FunctionCall,
    ChatMessage,
    ChatCompletionStreamOptions,
    ResponseFormat,
    ChatCompletionRequest,
    CompletionTokensDetails,
    PromptTokensDetails,
    ChatCompletionUsage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ToolCallChunkFunction,
    ToolCallChunk,
    ChatCompletionDelta,
    ChatCompletionChunkChoice,
    ChatCompletionChunkResponse,
    ModelInfo,
    ModelListResponse,
)
from models.embedding_schemas import (
    EmbeddingRequest,
    EmbeddingData,
    EmbeddingUsage,
    EmbeddingResponse,
)
from models.rerank_schemas import (
    RerankRequest,
    RerankItem,
    RerankResponse,
)
from models.kb_schemas import (
    # 核心数据结构
    KBChunk,
    KBDocument,
    ChunkPosition,
    ChunkOverlapInfo,
    # 检索相关
    KBSearchRequest,
    KBSearchResponse,
    KBSearchHit,
    SearchMode,
    # API 层
    SearchRequest,
    SearchResponse,
    SearchHitResponse,
    # 对话相关
    KBChatRequest,
    KBChatResponse,
    KBChatChoice,
    KBChatMessageResponse,
    KBChatInfo,
    KBSourceReference,
    KBChatDelta,
    KBChatChunkChoice,
    KBChatChunkResponse,
)
from core.config import (
    MilvusConfig,
    KBCollectionConfig,
)

__all__ = [
    # Chat - 内容块与工具调用
    "ChatContentBlock",
    "ToolCallFunction",
    "ToolCall",
    "FunctionCall",
    # Chat - 消息与请求
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionStreamOptions",
    "ResponseFormat",
    # Chat - Usage
    "CompletionTokensDetails",
    "PromptTokensDetails",
    "ChatCompletionUsage",
    # Chat - 非流式响应
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    # Chat - 流式响应
    "ToolCallChunkFunction",
    "ToolCallChunk",
    "ChatCompletionDelta",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkResponse",
    # Chat - 模型列表
    "ModelInfo",
    "ModelListResponse",
    # KB - 核心数据结构
    "KBChunk",
    "KBDocument", 
    "ChunkPosition",
    "ChunkOverlapInfo",
    # KB - 检索相关
    "KBSearchRequest",
    "KBSearchResponse",
    "KBSearchHit",
    "SearchMode",
    # KB - API 层
    "SearchRequest",
    "SearchResponse",
    "SearchHitResponse",
    # KB - 对话相关
    "KBChatRequest",
    "KBChatResponse",
    "KBChatChoice",
    "KBChatMessageResponse",
    "KBChatInfo",
    "KBSourceReference",
    "KBChatDelta",
    "KBChatChunkChoice",
    "KBChatChunkResponse",
    # Milvus
    "MilvusConfig",
    "KBCollectionConfig",
    # Embedding
    "EmbeddingRequest",
    "EmbeddingData",
    "EmbeddingUsage",
    "EmbeddingResponse",
    # Rerank
    "RerankRequest",
    "RerankItem",
    "RerankResponse",
]
