"""
Milvus向量库Schema配置

基于统一的 DocChunk 结构体重构，彻底消除多种 schema 类型和硬编码字段。
所有 Milvus 集合必须使用统一的 DocChunk schema。

DocChunk 字段：
- id: 主键ID（自动增长）
- docId: 文档唯一标识符
- chunkId: 文本块唯一标识符
- contentVector: 向量表示（1024维）
- chunkIndex: 文本块序号
- createTime: 创建时间戳
- updateTime: 更新时间戳
- version: 版本号
- meta_data: JSON结构（动态字段）

索引配置：
- HNSW 索引
- COSINE 余弦相似度
- M=16, efConstruction=200
"""

from typing import Dict, Any, List
from pymilvus import DataType, FieldSchema, CollectionSchema, Collection
from loguru import logger

# 导入 DocChunk Schema 配置
from models.doc_chunk import DocChunkSchema


class MilvusSchemaManager:
    """
    Milvus Schema 管理器
    
    基于 DocChunk 统一管理所有 Milvus 集合的 Schema。
    不再支持多种 schema 类型，所有集合必须使用 DocChunk schema。
    """
    
    def __init__(self, vector_dim: int = 1024):
        """
        初始化 Schema 管理器
        
        Args:
            vector_dim: 向量维度，默认 1024
        """
        self.schema_config = DocChunkSchema(vector_dim=vector_dim)
        logger.info(f"MilvusSchemaManager 初始化完成，vector_dim={vector_dim}")
    
    def get_schema_config(self) -> DocChunkSchema:
        """
        获取 DocChunk Schema 配置
        
        Returns:
            DocChunkSchema 实例
        """
        return self.schema_config
    
    def create_collection_schema(self, enable_dynamic_field: bool = True) -> CollectionSchema:
        """
        创建 Milvus CollectionSchema
        
        Args:
            enable_dynamic_field: 是否启用动态字段（支持 meta_data）
            
        Returns:
            CollectionSchema 实例
        """
        fields = []
        
        for field_config in self.schema_config.get_field_configs():
            kwargs = {
                "name": field_config["name"],
                "dtype": field_config["dtype"],
            }
            
            if field_config.get("is_primary"):
                kwargs["is_primary"] = True
                kwargs["auto_id"] = field_config.get("auto_id", False)
            
            if field_config.get("max_length"):
                kwargs["max_length"] = field_config["max_length"]
            
            if field_config.get("dim"):
                kwargs["dim"] = field_config["dim"]
            
            if field_config.get("description"):
                kwargs["description"] = field_config["description"]
            
            fields.append(FieldSchema(**kwargs))
        
        schema = CollectionSchema(
            fields=fields,
            description=self.schema_config.get_collection_description(),
            enable_dynamic_field=enable_dynamic_field
        )
        
        logger.info(f"创建 CollectionSchema: {len(fields)} 个字段")
        return schema
    
    def create_vector_index(self, collection: Collection) -> None:
        """
        为集合创建向量索引
        
        Args:
            collection: Milvus Collection 实例
        """
        index_config = self.schema_config.get_vector_index_config()
        
        index_params = {
            "index_type": index_config["index_type"],
            "metric_type": index_config["metric_type"],
            "params": index_config["params"]
        }
        
        collection.create_index(
            field_name=index_config["field_name"],
            index_params=index_params
        )
        
        logger.info(f"为集合 '{collection.name}' 创建向量索引: {index_params}")
    
    def get_vector_field_name(self) -> str:
        """
        获取向量字段名
        
        Returns:
            向量字段名（固定为 contentVector）
        """
        return self.schema_config.FIELD_CONTENT_VECTOR
    
    def get_all_field_names(self) -> List[str]:
        """
        获取所有字段名
        
        Returns:
            字段名列表
        """
        return [
            self.schema_config.FIELD_ID,
            self.schema_config.FIELD_DOC_ID,
            self.schema_config.FIELD_CHUNK_ID,
            self.schema_config.FIELD_CONTENT_VECTOR,
            self.schema_config.FIELD_CHUNK_INDEX,
            self.schema_config.FIELD_CREATE_TIME,
            self.schema_config.FIELD_UPDATE_TIME,
            self.schema_config.FIELD_VERSION,
        ]
    
    def get_output_fields_for_search(self, include_vector: bool = False) -> List[str]:
        """
        获取搜索时需要返回的字段
        
        Args:
            include_vector: 是否包含向量字段
            
        Returns:
            字段名列表（使用下划线命名）
        """
        fields = [
            self.schema_config.FIELD_DOC_ID,       # "doc_id"
            self.schema_config.FIELD_CHUNK_ID,     # "chunk_id"
            self.schema_config.FIELD_CHUNK_INDEX,  # "chunk_index"
            self.schema_config.FIELD_CREATE_TIME,  # "create_time"
            self.schema_config.FIELD_UPDATE_TIME,  # "update_time"
            self.schema_config.FIELD_VERSION,      # "version"
        ]
        
        if include_vector:
            fields.append(self.schema_config.FIELD_CONTENT_VECTOR)  # "content_vector"
        
        return fields
    
    def get_search_params(self, ef: int = 64) -> Dict[str, Any]:
        """
        获取搜索参数
        
        Args:
            ef: HNSW 搜索参数
            
        Returns:
            搜索参数字典
        """
        return self.schema_config.get_search_params(ef=ef)


# 全局 Schema 管理器实例（单例）
_global_schema_manager: MilvusSchemaManager = None


def get_schema_manager(vector_dim: int = 1024) -> MilvusSchemaManager:
    """
    获取全局 Schema 管理器实例
    
    Args:
        vector_dim: 向量维度
        
    Returns:
        MilvusSchemaManager 实例
    """
    global _global_schema_manager
    
    if _global_schema_manager is None:
        _global_schema_manager = MilvusSchemaManager(vector_dim=vector_dim)
    elif _global_schema_manager.schema_config.vector_dim != vector_dim:
        # 如果维度不同，重新创建
        logger.warning(f"向量维度变化：{_global_schema_manager.schema_config.vector_dim} -> {vector_dim}")
        _global_schema_manager = MilvusSchemaManager(vector_dim=vector_dim)
    
    return _global_schema_manager

