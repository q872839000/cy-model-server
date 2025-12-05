"""
DocChunk 统一结构体

这是 Milvus 向量数据库的唯一数据载体，所有向量读写、召回、管理操作都基于此结构。
彻底消除硬编码字段和 if-else 分支，实现统一的数据模型。

字段说明：
- id: 主键ID，自动增长（INT64）
- docId: 所属文档的唯一标识符（INT64）
- chunkId: 文本块的唯一标识符（INT64）
- contentVector: 文本内容的向量化表示，维度为1024（FLOAT_VECTOR）
- chunkIndex: 文本块在文档中的顺序编号（INT32）
- createTime: 记录创建的时间戳（INT64，毫秒）
- updateTime: 记录最后更新的时间戳（INT64，毫秒）
- version: 数据版本号，用于并发控制（INT32）
- meta_data: JSON结构，动态字段（JSON，可选）

索引配置：
- 使用 HNSW 索引类型对 contentVector 进行索引
- 采用 COSINE 余弦相似度
- 索引参数设置为 M=16, efConstruction=200
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import time


class DocChunk(BaseModel):
    """
    DocChunk 统一结构体（Pydantic 版本，用于 API 校验）
    
    所有字段顺序、类型、tag 必须与定义严格一致，不得增减任何成员。
    字段名使用下划线命名（snake_case），与 Milvus 数据库字段一致。
    
    注意：content_vector 在搜索结果中可能不存在（Milvus search 不返回向量），
    因此设为可选字段，默认为空列表。
    """
    id: Optional[int] = Field(None, description="主键ID，自动增长")
    doc_id: int = Field(..., description="所属文档的唯一标识符")
    chunk_id: int = Field(..., description="文本块的唯一标识符")
    content_vector: List[float] = Field(default_factory=list, description="文本内容的向量化表示，维度为1024")
    chunk_index: int = Field(..., description="文本块在文档中的顺序编号")
    create_time: Optional[int] = Field(None, description="记录创建的时间戳（毫秒）")
    update_time: Optional[int] = Field(None, description="记录最后更新的时间戳（毫秒）")
    version: int = Field(1, description="数据版本号，用于并发控制")
    
    def to_milvus_insert_data(self) -> Dict[str, Any]:
        """
        转换为 Milvus 插入格式
        
        注意：主键 id 是自动生成的，不包含在插入数据中
        """
        current_time = int(time.time() * 1000)
        
        data = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content_vector": self.content_vector,
            "chunk_index": self.chunk_index,
            "create_time": self.create_time or current_time,
            "update_time": self.update_time or current_time,
            "version": self.version,
        }
        
        return data
    
    @classmethod
    def from_milvus_result(cls, data: Dict[str, Any]) -> "DocChunk":
        """
        从 Milvus 查询结果构造 DocChunk
        
        Args:
            data: Milvus 查询返回的数据字典
            
        Returns:
            DocChunk 实例
            
        Raises:
            ValueError: 如果缺少必需字段 doc_id 或 chunk_id
        """
        doc_id = data.get("doc_id")
        chunk_id = data.get("chunk_id")
        
        # 必需字段检查
        if doc_id is None:
            raise ValueError(f"Milvus result missing required field 'doc_id': {data}")
        if chunk_id is None:
            raise ValueError(f"Milvus result missing required field 'chunk_id': {data}")
        
        return cls(
            id=data.get("id"),
            doc_id=doc_id,
            chunk_id=chunk_id,
            content_vector=data.get("content_vector", []),
            chunk_index=data.get("chunk_index", 0),
            create_time=data.get("create_time"),
            update_time=data.get("update_time"),
            version=data.get("version", 1)
        )


@dataclass
class DocChunkSchema:
    """
    DocChunk Schema 配置
    
    定义 DocChunk 在 Milvus 中的字段类型、索引配置等。
    这是唯一的 Schema 定义，所有集合必须使用此 Schema。
    """
    vector_dim: int = 1024
    
    # 索引配置（固定值，不可修改）
    INDEX_TYPE: str = "HNSW"
    METRIC_TYPE: str = "COSINE"
    INDEX_M: int = 16
    INDEX_EF_CONSTRUCTION: int = 200
    
    # 字段名（固定值，不可修改，使用下划线命名）
    FIELD_ID: str = "id"
    FIELD_DOC_ID: str = "doc_id"
    FIELD_CHUNK_ID: str = "chunk_id"
    FIELD_CONTENT_VECTOR: str = "content_vector"
    FIELD_CHUNK_INDEX: str = "chunk_index"
    FIELD_CREATE_TIME: str = "create_time"
    FIELD_UPDATE_TIME: str = "update_time"
    FIELD_VERSION: str = "version"
    
    def get_field_configs(self) -> List[Dict[str, Any]]:
        """
        获取字段配置列表
        
        Returns:
            字段配置列表，每个字段包含 name、dtype、is_primary、auto_id、dim 等属性
        """
        from pymilvus import DataType
        
        return [
            {
                "name": self.FIELD_ID,
                "dtype": DataType.INT64,
                "is_primary": True,
                "auto_id": True,
                "description": "主键ID，自动增长"
            },
            {
                "name": self.FIELD_DOC_ID,
                "dtype": DataType.INT64,
                "description": "所属文档的唯一标识符"
            },
            {
                "name": self.FIELD_CHUNK_ID,
                "dtype": DataType.INT64,
                "description": "文本块的唯一标识符"
            },
            {
                "name": self.FIELD_CONTENT_VECTOR,
                "dtype": DataType.FLOAT_VECTOR,
                "dim": self.vector_dim,
                "description": "文本内容的向量化表示"
            },
            {
                "name": self.FIELD_CHUNK_INDEX,
                "dtype": DataType.INT32,
                "description": "文本块在文档中的顺序编号"
            },
            {
                "name": self.FIELD_CREATE_TIME,
                "dtype": DataType.INT64,
                "description": "记录创建的时间戳（毫秒）"
            },
            {
                "name": self.FIELD_UPDATE_TIME,
                "dtype": DataType.INT64,
                "description": "记录最后更新的时间戳（毫秒）"
            },
            {
                "name": self.FIELD_VERSION,
                "dtype": DataType.INT32,
                "description": "数据版本号，用于并发控制"
            },
        ]
    
    def get_vector_index_config(self) -> Dict[str, Any]:
        """
        获取向量索引配置
        
        Returns:
            索引配置字典，包含 field_name、index_type、metric_type、params
        """
        return {
            "field_name": self.FIELD_CONTENT_VECTOR,
            "index_type": self.INDEX_TYPE,
            "metric_type": self.METRIC_TYPE,
            "params": {
                "M": self.INDEX_M,
                "efConstruction": self.INDEX_EF_CONSTRUCTION
            }
        }
    
    def get_search_params(self, ef: int = 64) -> Dict[str, Any]:
        """
        获取搜索参数
        
        Args:
            ef: HNSW 搜索参数，控制搜索深度
            
        Returns:
            搜索参数字典
        """
        return {
            "metric_type": self.METRIC_TYPE,
            "params": {"ef": ef}
        }
    
    def get_collection_description(self) -> str:
        """
        获取集合描述
        
        Returns:
            集合描述文本
        """
        return f"DocChunk Collection (vector_dim={self.vector_dim}, index={self.INDEX_TYPE}, metric={self.METRIC_TYPE})"


class DocChunkConverter:
    """
    DocChunk 转换器
    
    提供 DocChunk 与其他数据格式之间的转换方法。
    """
    
    @staticmethod
    def from_document_content(
        doc_id: int,
        chunk_id: int,
        content: str,
        embedding: List[float],
        chunk_index: int = 0,
        version: int = 1
    ) -> DocChunk:
        """
        从文档内容创建 DocChunk
        
        Args:
            doc_id: 文档ID
            chunk_id: 文本块ID
            content: 文本内容（不存储，仅用于生成向量）
            embedding: 向量表示
            chunk_index: 文本块索引
            version: 版本号
            
        Returns:
            DocChunk 实例
        """
        current_time = int(time.time() * 1000)
        
        return DocChunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            content_vector=embedding,
            chunk_index=chunk_index,
            create_time=current_time,
            update_time=current_time,
            version=version
        )
    
    @staticmethod
    def to_retrieval_result(
        chunk: DocChunk,
        score: float,
        content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        转换为检索结果格式
        
        Args:
            chunk: DocChunk 实例
            score: 相似度得分
            content: 文本内容（可选，需要外部提供）
            
        Returns:
            检索结果字典
        """
        return {
            "id": f"{chunk.doc_id}_{chunk.chunk_id}",
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "score": score,
            "content": content or f"Document {chunk.doc_id}, Chunk {chunk.chunk_id}",
            "metadata": {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "create_time": chunk.create_time,
                "update_time": chunk.update_time,
                "version": chunk.version
            }
        }
    
    @staticmethod
    def batch_to_milvus_insert_format(chunks: List[DocChunk]) -> List[List[Any]]:
        """
        批量转换为 Milvus 插入格式（列式存储）
        
        Args:
            chunks: DocChunk 列表
            
        Returns:
            Milvus 插入数据（列式）：[doc_ids, chunk_ids, vectors, chunk_indexes, ...]
        """
        doc_ids = []
        chunk_ids = []
        vectors = []
        chunk_indexes = []
        create_times = []
        update_times = []
        versions = []
        
        current_time = int(time.time() * 1000)
        
        for chunk in chunks:
            doc_ids.append(chunk.doc_id)
            chunk_ids.append(chunk.chunk_id)
            vectors.append(chunk.content_vector)
            chunk_indexes.append(chunk.chunk_index)
            create_times.append(chunk.create_time or current_time)
            update_times.append(chunk.update_time or current_time)
            versions.append(chunk.version)
        
        return [
            doc_ids,
            chunk_ids,
            vectors,
            chunk_indexes,
            create_times,
            update_times,
            versions,
        ]

