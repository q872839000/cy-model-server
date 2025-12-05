from __future__ import annotations

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"


class Coordinates(BaseModel):
    page: int = Field(..., description="所在页码，从 1 开始")
    x: Optional[float] = Field(None, description="左上角 X 坐标")
    y: Optional[float] = Field(None, description="左上角 Y 坐标")
    w: Optional[float] = Field(None, description="宽度")
    h: Optional[float] = Field(None, description="高度")


class Styles(BaseModel):
    bold: bool = Field(False, description="是否加粗")
    italic: bool = Field(False, description="是否斜体")
    font_size: Optional[float] = Field(None, description="字体大小")
    font_family: Optional[str] = Field(None, description="字体名称")
    indent: Optional[float] = Field(None, description="缩进")


class TableData(BaseModel):
    rows: List[List[str]] = Field(default_factory=list, description="表格行数据")


class Block(BaseModel):
    block_id: str = Field(..., description="内容块唯一ID")
    type: BlockType = Field(..., description="内容块类型")
    order: int = Field(..., description="在章节中的顺序")

    text: Optional[str] = Field(None, description="段落文本 / OCR 文本")
    html: Optional[str] = Field(None, description="原格式 HTML（用于渲染）")
    ocr_text: Optional[str] = Field(None, description="图片 OCR 提取的文本")

    table: Optional[TableData] = Field(None, description="表格结构，仅当 type=table 时有效")
    coordinates: Optional[Coordinates] = Field(None, description="位置信息")
    styles: Optional[Styles] = Field(None, description="格式信息")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="可扩展的解析信息")


class SectionNode(BaseModel):
    section_id: str = Field(..., description="章节唯一ID")
    parent_id: Optional[str] = Field(None, description="父节点ID")
    level: int = Field(..., description="层级：1=H1 / 2=H2 / 3=H3")
    title: str = Field(..., description="章节标题")
    order: int = Field(..., description="在文档中的顺序")

    page_start: Optional[int] = Field(None, description="内容开始页")
    page_end: Optional[int] = Field(None, description="内容结束页")

    content_blocks: List[Block] = Field(default_factory=list, description="该章节下的内容块")
    children: List["SectionNode"] = Field(default_factory=list, description="子节点")


class DocumentMetadata(BaseModel):
    keywords: Optional[str] = Field(None, description="关键词")
    title: Optional[str] = Field(None, description="标题")


class ChunkType(str, Enum):
    SECTION_CHUNK = "section_chunk"
    PARAGRAPH_CHUNK = "paragraph_chunk"
    SLIDING_WINDOW = "sliding_window"


class ChunkMeta(BaseModel):
    title: Optional[str] = Field(None, description="所属标题")
    headings: Optional[List[str]] = Field(None, description="所有上级标题链")
    file_name: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件类型")
    page_start: int = Field(..., description="起始页码")
    page_end: int = Field(..., description="结束页码")


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Chunk ID")
    doc_id: str = Field(..., description="所属文档ID")
    section_id: str = Field(..., description="所属章节")
    block_ids: List[str] = Field(default_factory=list, description="由哪些 Block 拼装而来")
    
    chunk_type: ChunkType = Field(..., description="Chunk类型")
    order: int = Field(..., description="在文档中的顺序")
    content: str = Field(..., description="Chunk 文本（用于 embedding）")
    raw_html: str = Field(..., description="原格式 HTML（用于展示）")
    token_count: int = Field(..., description="字符/Token 数量")
    
    meta: ChunkMeta = Field(..., description="元数据")
    embedding: Optional[List[float]] = Field(None, description="向量（入库后填充）")


class ParsedDocument(BaseModel):
    original_filename: str = Field(..., description="文件名称")
    content_type: str = Field(..., description="文件类型 MIME")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="文档元数据")
    text: str = Field(..., description="文档原文内容（纯文本）")
    structure_tree: List[SectionNode] = Field(default_factory=list, description="文档结构树（章节树）")
    chunks: List[Chunk] = Field(default_factory=list, description="语义分块结果")
