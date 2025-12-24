"""
Milvus Collection 管理模块

本模块提供 Collection 的定义与管理，支持：
- 动态创建：按需创建 Collection
- 混合检索：Dense + Sparse/BM25

支持 Milvus 2.5+ 的 Sparse Vector（BM25）能力。

注意：本模块不包含"知识库"概念，直接使用 collection_name 操作。
知识库的管理由业务系统负责。
"""

from typing import Optional, List, Dict, Any
from loguru import logger

from storage.milvus.config import KBCollectionConfig, load_kb_collection_config
from storage.milvus.client import MilvusClient, MilvusConnectionError
from models import KBChunk, KBDocument

# 延迟导入 pymilvus
_pymilvus_available = False
try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
        Function,
        FunctionType,
    )
    _pymilvus_available = True
except ImportError:
    pass


class CollectionNotFoundError(Exception):
    """Collection 不存在异常"""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        super().__init__(f"Collection '{collection_name}' 不存在")


class KBCollectionManager:
    """
    Milvus Collection 管理器。
    
    提供 Collection 的创建、管理和数据操作。
    Collection 名称由业务系统传入，本模块不做命名转换。
    
    主要功能：
    - 动态创建/删除 Collection
    - Schema 定义（支持 Dense + Sparse 混合检索）
    - 索引管理
    - 数据的增删改查
    
    Attributes:
        client: Milvus 客户端实例
        config: Collection 全局配置
        
    Usage:
        >>> manager = KBCollectionManager()
        >>> manager.create_collection("my_collection")
        >>> manager.insert_chunks("my_collection", chunks, vectors)
    """
    
    def __init__(
        self, 
        client: Optional[MilvusClient] = None,
        config: Optional[KBCollectionConfig] = None
    ) -> None:
        """
        初始化 Collection 管理器。
        
        Args:
            client: Milvus 客户端，为 None 时使用默认单例
            config: Collection 配置，为 None 时从配置文件加载
        """
        self._client = client or MilvusClient()
        self._config = config or load_kb_collection_config()
        # 缓存已加载的 Collection 实例
        self._collections: Dict[str, "Collection"] = {}
        
    @property
    def config(self) -> KBCollectionConfig:
        """获取 Collection 配置"""
        return self._config
    
    def _check_connection(self) -> None:
        """检查 Milvus 连接状态"""
        if not self._client.is_connected:
            raise MilvusConnectionError("未连接到 Milvus，请先调用 client.connect()")
    
    def _check_pymilvus(self) -> None:
        """检查 pymilvus 是否可用"""
        if not _pymilvus_available:
            raise MilvusConnectionError(
                "pymilvus 未安装，请执行: pip install pymilvus>=2.5.0"
            )
    
    # ==================== Collection 管理 ====================
    
    def create_collection(self, collection_name: str) -> str:
        """
        创建 Collection。
        
        如果 Collection 已存在则跳过。
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            str: Collection 名称
        """
        self._check_connection()
        self._check_pymilvus()
        
        if utility.has_collection(collection_name):
            logger.debug("Collection '{}' 已存在", collection_name)
            return collection_name
        
        # 创建 Schema
        schema = self._create_kb_schema()
        
        # 创建 Collection
        collection = Collection(
            name=collection_name,
            schema=schema,
            consistency_level="Strong"
        )
        
        # 创建索引
        self._create_indexes(collection)
        
        # 加载到内存
        collection.load()
        
        # 缓存
        self._collections[collection_name] = collection
        
        logger.info("Collection '{}' 创建成功", collection_name)
        return collection_name
    
    def delete_collection(self, collection_name: str, confirm: bool = False) -> bool:
        """
        删除 Collection。
        
        Args:
            collection_name: Collection 名称
            confirm: 确认删除，必须为 True
            
        Returns:
            bool: 是否删除成功
            
        Warning:
            此操作不可逆，将删除所有数据！
        """
        if not confirm:
            logger.warning("删除操作未确认，跳过")
            return False
        
        self._check_connection()
        self._check_pymilvus()
        
        if not utility.has_collection(collection_name):
            logger.warning("Collection '{}' 不存在", collection_name)
            return False
        
        utility.drop_collection(collection_name)
        
        # 清除缓存
        if collection_name in self._collections:
            del self._collections[collection_name]
        
        logger.info("Collection '{}' 已删除", collection_name)
        return True
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        检查 Collection 是否存在。
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            bool: 是否存在
        """
        self._check_connection()
        self._check_pymilvus()
        
        return utility.has_collection(collection_name)
    
    def list_collections(self) -> List[str]:
        """
        列出所有 Collection。
        
        Returns:
            List[str]: Collection 名称列表
        """
        self._check_connection()
        self._check_pymilvus()
        
        return utility.list_collections()
    
    def get_collection(self, collection_name: str) -> "Collection":
        """
        获取 Collection 实例。
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            Collection: Milvus Collection 实例
            
        Raises:
            CollectionNotFoundError: Collection 不存在
        """
        self._check_connection()
        self._check_pymilvus()
        
        # 检查缓存
        if collection_name in self._collections:
            return self._collections[collection_name]
        
        # 检查是否存在
        if not utility.has_collection(collection_name):
            raise CollectionNotFoundError(collection_name)
        
        # 加载并缓存
        collection = Collection(collection_name)
        collection.load()
        self._collections[collection_name] = collection
        
        return collection
    
    # ==================== Schema 定义 ====================
    
    def _create_kb_schema(self) -> "CollectionSchema":
        """
        创建知识库 Collection 的 Schema。
        
        Schema 包含：
        - 基础字段：id, doc_id, doc_name, chapter, chapter_path, chunk_idx, content, title
        - 位置字段：start_pos, end_pos, overlap_prev, overlap_next
        - 向量字段：dense_vector (稠密), sparse_vector (内容BM25), title_sparse_vector (标题BM25)
        - 元数据字段：metadata (JSON), created_at
        
        Returns:
            CollectionSchema: Collection Schema 定义
        """
        fields = [
            # 主键
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=128,
                is_primary=True,
                description="切片唯一ID: {doc_id}_{chunk_idx}"
            ),
            
            # 文档归属
            FieldSchema(
                name="doc_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="所属文档ID"
            ),
            FieldSchema(
                name="doc_name",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="文档名称"
            ),
            
            # 章节信息
            FieldSchema(
                name="chapter",
                dtype=DataType.VARCHAR,
                max_length=256,
                description="章节标题"
            ),
            FieldSchema(
                name="chapter_path",
                dtype=DataType.VARCHAR,
                max_length=1024,
                description="章节层级路径"
            ),
            
            # 切片信息
            FieldSchema(
                name="chunk_idx",
                dtype=DataType.INT32,
                description="切片序号"
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                enable_match=True,
                analyzer_params={"type": "chinese"},
                description="切片文本内容"
            ),
            
            # 位置信息
            FieldSchema(
                name="start_pos",
                dtype=DataType.INT32,
                description="原文起始位置"
            ),
            FieldSchema(
                name="end_pos",
                dtype=DataType.INT32,
                description="原文结束位置"
            ),
            FieldSchema(
                name="overlap_prev",
                dtype=DataType.INT32,
                description="与前一切片重叠字符数"
            ),
            FieldSchema(
                name="overlap_next",
                dtype=DataType.INT32,
                description="与后一切片重叠字符数"
            ),
            
            # 稠密向量（语义检索）
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._config.dense_dim,
                description="稠密向量（embedding）"
            ),
            
            # 标题字段（用于 BM25 检索，由 doc_name + chapter 拼接）
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=1024,
                enable_analyzer=True,
                enable_match=True,
                analyzer_params={"type": "chinese"},
                description="标题（doc_name + chapter 拼接，用于 BM25 检索）"
            ),
            
            # 内容稀疏向量（BM25 检索）
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="内容稀疏向量（BM25）"
            ),
            
            # 标题稀疏向量（BM25 检索）
            FieldSchema(
                name="title_sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="标题稀疏向量（BM25）"
            ),
            
            # 元数据
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="扩展元数据"
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
                description="入库时间戳（毫秒）"
            ),
        ]
        
        # BM25 函数：从 content 自动生成 sparse_vector
        content_bm25_function = Function(
            name="content_bm25",
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
        
        # BM25 函数：从 title 自动生成 title_sparse_vector
        title_bm25_function = Function(
            name="title_bm25",
            input_field_names=["title"],
            output_field_names=["title_sparse_vector"],
            function_type=FunctionType.BM25,
        )
        
        schema = CollectionSchema(
            fields=fields,
            description="知识库切片，支持三路混合检索（Dense + Content BM25 + Title BM25）",
            functions=[content_bm25_function, title_bm25_function]
        )
        
        return schema
    
    def _create_indexes(self, collection: "Collection") -> None:
        """创建 Collection 索引"""
        # Dense Vector 索引
        dense_index_params = {
            "index_type": self._config.dense_index_type,
            "metric_type": self._config.dense_metric_type,
            "params": {
                "M": self._config.hnsw_m,
                "efConstruction": self._config.hnsw_ef_construction
            }
        }
        collection.create_index(
            field_name="dense_vector",
            index_params=dense_index_params,
            index_name="idx_dense_vector"
        )
        
        # Content Sparse Vector 索引（BM25）
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {
                "bm25_k1": self._config.bm25_k1,
                "bm25_b": self._config.bm25_b,
            }
        }
        collection.create_index(
            field_name="sparse_vector",
            index_params=sparse_index_params,
            index_name="idx_sparse_vector"
        )
        
        # Title Sparse Vector 索引（BM25）
        title_sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {
                "bm25_k1": self._config.bm25_k1,
                "bm25_b": self._config.bm25_b,
            }
        }
        collection.create_index(
            field_name="title_sparse_vector",
            index_params=title_sparse_index_params,
            index_name="idx_title_sparse_vector"
        )
        
        # 标量字段索引
        collection.create_index(field_name="doc_name", index_name="idx_doc_name")
        collection.create_index(field_name="doc_id", index_name="idx_doc_id")
        
        logger.debug("索引创建完成（Dense + Content Sparse + Title Sparse）")
    
    @staticmethod
    def _build_title(doc_name: str, chapter: str) -> str:
        """
        构建用于 BM25 检索的标题字段。
        
        将 doc_name 和 chapter 拼接，中间用空格分隔，便于分词器切分。
        
        Args:
            doc_name: 文档名称
            chapter: 章节标题
            
        Returns:
            str: 拼接后的标题
        """
        parts = []
        if doc_name:
            parts.append(doc_name.strip())
        if chapter:
            parts.append(chapter.strip())
        return " ".join(parts)
    
    # ==================== 数据操作 ====================
    
    def insert_chunks(
        self,
        collection_name: str,
        chunks: List[KBChunk],
        dense_vectors: List[List[float]]
    ) -> List[str]:
        """
        批量插入切片数据。
        
        Args:
            collection_name: Collection 名称
            chunks: 切片列表
            dense_vectors: 对应的稠密向量列表
            
        Returns:
            List[str]: 插入的切片 ID 列表
            
        Raises:
            ValueError: chunks 和 dense_vectors 长度不匹配
            CollectionNotFoundError: Collection 不存在
        """
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks 和 dense_vectors 长度不匹配: {len(chunks)} vs {len(dense_vectors)}"
            )
        
        if not chunks:
            return []
        
        collection = self.get_collection(collection_name)
        
        # 构建插入数据
        data = [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "chapter": chunk.chapter,
                "chapter_path": chunk.chapter_path,
                "chunk_idx": chunk.chunk_idx,
                "content": chunk.content,
                "title": self._build_title(chunk.doc_name, chunk.chapter),
                "start_pos": chunk.position.start_pos,
                "end_pos": chunk.position.end_pos,
                "overlap_prev": chunk.overlap.prev_chars,
                "overlap_next": chunk.overlap.next_chars,
                "dense_vector": dense_vectors[i],
                "metadata": chunk.metadata,
                "created_at": chunk.created_at,
            }
            for i, chunk in enumerate(chunks)
        ]
        
        collection.insert(data)
        logger.info("Collection '{}': 插入 {} 个切片", collection_name, len(chunks))
        
        return [chunk.id for chunk in chunks]
    
    def delete_chunks_by_doc(self, collection_name: str, doc_id: str) -> None:
        """
        删除指定文档的所有切片。
        
        Args:
            collection_name: Collection 名称
            doc_id: 文档 ID
        """
        collection = self.get_collection(collection_name)
        collection.delete(expr=f'doc_id == "{doc_id}"')
        logger.info("Collection '{}': 删除文档 '{}' 的所有切片", collection_name, doc_id)
    
    def get_chunks_by_chapter(
        self,
        collection_name: str,
        doc_name: str,
        chapter_path: str,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        根据文档名称和章节路径查询切片。
        
        Args:
            collection_name: Collection 名称
            doc_name: 文档名称
            chapter_path: 章节路径（支持前缀匹配）
            output_fields: 返回字段列表
            
        Returns:
            List[Dict]: 切片列表，按 chunk_idx 排序
        """
        collection = self.get_collection(collection_name)
        
        expr = f'doc_name == "{doc_name}" and chapter_path like "{chapter_path}%"'
        
        results = collection.query(
            expr=expr,
            output_fields=output_fields or ["*"],
            limit=10000
        )
        
        results.sort(key=lambda x: x.get("chunk_idx", 0))
        return results
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取 Collection 统计信息。
        
        Args:
            collection_name: Collection 名称
            
        Returns:
            Dict: 统计信息
        """
        collection = self.get_collection(collection_name)
        
        return {
            "collection_name": collection_name,
            "num_entities": collection.num_entities,
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        获取所有 Collection 的统计信息。
        
        Returns:
            Dict: 统计信息
        """
        self._check_connection()
        self._check_pymilvus()
        
        collections = self.list_collections()
        stats = []
        
        for col_name in collections:
            try:
                collection = Collection(col_name)
                stats.append({
                    "collection_name": col_name,
                    "num_entities": collection.num_entities,
                })
            except Exception as e:
                logger.warning("获取 Collection '{}' 信息失败: {}", col_name, str(e))
        
        return {
            "total_collections": len(stats),
            "collections": stats,
        }
