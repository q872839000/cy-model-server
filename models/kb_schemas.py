"""
知识库数据结构定义

本模块定义知识库系统的公共数据结构，包括：
- 文档切片（KBChunk）
- 文档元信息（KBDocument）
- 检索请求/响应（KBSearchRequest/KBSearchResult）

这些结构体被 storage 层和 rag 层共同使用。
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class ChunkOverlapInfo(BaseModel):
    """
    切片重叠信息。
    
    用于记录切片与相邻切片的重叠情况，支持：
    1. 合并相邻切片时精确裁剪重叠部分
    2. 去重展示，避免重复内容
    """
    prev_chars: int = Field(default=0, description="与前一个切片重叠的字符数")
    next_chars: int = Field(default=0, description="与后一个切片重叠的字符数")


class ChunkPosition(BaseModel):
    """
    切片在原文中的位置信息。
    
    用于精确定位切片在原始文档中的位置，支持：
    1. 原文高亮展示
    2. 上下文扩展
    """
    start_pos: int = Field(..., description="切片在原文中的起始位置（字符偏移，0-based）")
    end_pos: int = Field(..., description="切片在原文中的结束位置（字符偏移，不包含）")


class KBChunk(BaseModel):
    """
    知识库切片。
    
    表示文档经过切片处理后的最小检索单元。每个切片包含：
    - 唯一标识与归属信息
    - 文本内容
    - 章节定位信息（支持按章节查询）
    - 位置与重叠信息（支持精确定位与去重）
    - 扩展元数据
    
    Attributes:
        id: 切片唯一ID，格式为 "{doc_id}_{chunk_idx}"
        doc_id: 所属文档的唯一标识
        doc_name: 文档名称，用于用户友好的展示和查询
        chapter: 章节标题
        chapter_path: 章节层级路径，如 "第一章/1.2节/概述"
        chunk_idx: 切片在文档中的序号（0-based）
        content: 切片文本内容
        position: 切片在原文中的位置信息
        overlap: 与相邻切片的重叠信息
        metadata: 扩展元数据（如作者、标签等）
        created_at: 入库时间戳（Unix 毫秒）
    """
    id: str = Field(..., description="切片唯一ID，格式: {doc_id}_{chunk_idx}")
    doc_id: str = Field(..., description="所属文档ID")
    doc_name: str = Field(..., description="文档名称")
    chapter: str = Field(default="", description="章节标题")
    chapter_path: str = Field(default="", description="章节层级路径，如 '第一章/1.2节/概述'")
    chunk_idx: int = Field(..., description="切片在文档中的序号（0-based）")
    content: str = Field(..., description="切片文本内容")
    position: ChunkPosition = Field(..., description="切片在原文中的位置信息")
    overlap: ChunkOverlapInfo = Field(default_factory=ChunkOverlapInfo, description="重叠信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")
    created_at: int = Field(default_factory=lambda: int(__import__('time').time() * 1000), description="入库时间戳（Unix 毫秒）")


class ChapterNode(BaseModel):
    """
    章节树节点。
    
    用于表示文档的层级章节结构。
    """
    title: str = Field(..., description="章节标题")
    level: int = Field(..., description="章节层级（1 为顶级章节）")
    path: str = Field(..., description="章节完整路径")
    start_chunk_idx: int = Field(..., description="章节起始切片索引")
    end_chunk_idx: int = Field(..., description="章节结束切片索引（不包含）")
    children: List["ChapterNode"] = Field(default_factory=list, description="子章节列表")


class KBDocument(BaseModel):
    """
    知识库文档元信息。
    
    存储文档级别的元数据，与切片分离存储以避免冗余。
    
    Attributes:
        doc_id: 文档唯一标识
        doc_name: 文档名称（用于用户查询，如 "产品手册.pdf"）
        doc_type: 文档类型（pdf/docx/md/txt/html）
        total_chunks: 文档切片总数
        chapter_tree: 章节结构树
        file_hash: 文件内容哈希（SHA256），用于去重和变更检测
        file_size: 文件大小（字节）
        source_path: 原始文件路径或URL
        created_at: 入库时间戳（Unix 毫秒）
        updated_at: 更新时间戳（Unix 毫秒）
        metadata: 扩展元数据
    """
    doc_id: str = Field(..., description="文档唯一ID")
    doc_name: str = Field(..., description="文档名称")
    doc_type: str = Field(..., description="文档类型: pdf/docx/md/txt/html")
    total_chunks: int = Field(default=0, description="切片总数")
    chapter_tree: List[ChapterNode] = Field(default_factory=list, description="章节结构树")
    file_hash: str = Field(..., description="文件SHA256哈希")
    file_size: int = Field(..., description="文件大小（字节）")
    source_path: str = Field(default="", description="原始文件路径或URL")
    created_at: int = Field(..., description="入库时间戳（Unix 毫秒）")
    updated_at: int = Field(..., description="更新时间戳（Unix 毫秒）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")


class SearchMode(str, Enum):
    """
    检索模式枚举。
    
    - DENSE: 仅使用稠密向量进行语义检索
    - SPARSE: 仅使用稀疏向量（BM25）进行关键词检索
    - HYBRID: 混合检索，融合语义和关键词结果
    """
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class KBSearchRequest(BaseModel):
    """
    检索请求。
    
    支持多种检索模式和过滤条件。
    
    Attributes:
        collection_name: Milvus Collection 名称（必填，由业务系统指定）
        query: 用户查询文本
        mode: 检索模式（dense/sparse/hybrid）
        top_k: 返回结果数量
        doc_name: 可选，按文档名称过滤
        chapter_filter: 可选，按章节路径过滤（支持前缀匹配）
        rerank: 是否启用重排序
        rerank_top_k: 重排序后返回的结果数量
        score_threshold: 可选，最低相关性分数阈值
        metadata_filter: 可选，元数据过滤条件
    """
    collection_name: str = Field(..., description="Milvus Collection 名称")
    query: str = Field(..., description="用户查询文本")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="检索模式")
    top_k: int = Field(default=10, description="返回结果数量")
    doc_name: Optional[str] = Field(default=None, description="按文档名称过滤")
    chapter_filter: Optional[str] = Field(default=None, description="按章节路径过滤（前缀匹配）")
    rerank: bool = Field(default=True, description="是否启用重排序")
    rerank_top_k: Optional[int] = Field(default=None, description="重排序后返回数量，默认等于 top_k")
    score_threshold: Optional[float] = Field(default=None, description="最低相关性分数阈值")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="元数据过滤条件")


class KBSearchHit(BaseModel):
    """
    单条检索结果。
    
    Attributes:
        chunk: 命中的切片
        score: 相关性分数（归一化到 0-1）
        dense_score: 稠密向量检索分数（可选）
        sparse_score: 稀疏向量检索分数（可选）
        rerank_score: 重排序分数（可选）
    """
    chunk: KBChunk = Field(..., description="命中的切片")
    score: float = Field(..., description="最终相关性分数")
    dense_score: Optional[float] = Field(default=None, description="稠密向量检索分数")
    sparse_score: Optional[float] = Field(default=None, description="稀疏向量检索分数")
    rerank_score: Optional[float] = Field(default=None, description="重排序分数")


class KBSearchResponse(BaseModel):
    """
    知识库检索响应。
    
    Attributes:
        hits: 检索结果列表，按相关性降序排列
        total: 命中总数（在 top_k 截断前）
        query: 原始查询文本
        mode: 使用的检索模式
        took_ms: 检索耗时（毫秒）
    """
    hits: List[KBSearchHit] = Field(default_factory=list, description="检索结果列表")
    total: int = Field(default=0, description="命中总数")
    query: str = Field(..., description="原始查询文本")
    mode: SearchMode = Field(..., description="使用的检索模式")
    took_ms: float = Field(..., description="检索耗时（毫秒）")


# ==================== API 层请求/响应模型 ====================

class SearchRequest(BaseModel):
    """
    检索请求（API 层）。
    
    Attributes:
        collection_name: Milvus Collection 名称（由业务系统指定）
        query: 查询文本（自然语言）
        mode: 检索模式 (dense/sparse/hybrid)
        top_k: 返回结果数量
        doc_name: 按文档名称过滤（可选）
        chapter_filter: 按章节路径过滤（可选）
        rerank: 是否启用重排序
        score_threshold: 最低分数阈值
    """
    collection_name: str = Field(..., description="Milvus Collection 名称", min_length=1)
    query: str = Field(..., description="查询文本", min_length=1)
    mode: str = Field(default="hybrid", description="检索模式: dense/sparse/hybrid")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    doc_name: Optional[str] = Field(default=None, description="按文档名称过滤")
    chapter_filter: Optional[str] = Field(default=None, description="按章节路径过滤")
    rerank: bool = Field(default=True, description="是否启用重排序")
    score_threshold: Optional[float] = Field(default=None, ge=0, le=1, description="最低分数阈值")


class SearchHitResponse(BaseModel):
    """单条检索结果（API 层）"""
    id: str = Field(..., description="切片ID")
    doc_name: str = Field(..., description="文档名称")
    chapter: str = Field(default="", description="章节标题")
    content: str = Field(..., description="切片内容")
    score: float = Field(..., description="相关性分数")


class SearchResponse(BaseModel):
    """检索响应（API 层）"""
    hits: List[SearchHitResponse] = Field(default_factory=list, description="检索结果")
    total: int = Field(default=0, description="结果总数")
    query: str = Field(..., description="原始查询")
    mode: str = Field(..., description="检索模式")
    took_ms: float = Field(..., description="耗时(毫秒)")


# 支持 ChapterNode 的自引用
ChapterNode.model_rebuild()
