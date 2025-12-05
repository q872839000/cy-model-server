"""
Milvus 客户端（基于 DocChunk 重构）

完全基于 DocChunk 结构体的 Milvus 操作封装，彻底消除硬编码字段和 schema 类型判断。
所有集合操作、数据操作、向量检索都使用统一的 DocChunk 接口。

主要功能：
1. 连接管理：建立、复用 Milvus 连接
2. 集合管理：创建、删除、查询集合
3. 数据操作：插入、删除、更新、查询
4. 向量检索：基于 DocChunk 的向量相似度搜索
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from pymilvus import connections, utility, Collection, MilvusException
from pymilvus import db as milvus_db
import time
import math

from models.doc_chunk import DocChunk, DocChunkConverter
from configs.milvus_schema import get_schema_manager, MilvusSchemaManager
from core.config import load_settings


class MilvusClient:
    """
    Milvus 客户端（基于 DocChunk）
    
    所有操作都基于 DocChunk 结构体，不再支持多种 schema 类型。
    """
    
    def __init__(self, vector_dim: int = 1024):
        """
        初始化 Milvus 客户端
        
        Args:
            vector_dim: 向量维度，默认 1024
        """
        # 加载配置
        settings = load_settings()
        import yaml
        cfg = yaml.safe_load(open(settings.models_config_path, 'r', encoding='utf-8')) or {}
        vs = (cfg.get('vector_store') or {}).get('milvus') or {}
        
        # 连接配置
        self.uri = vs.get('uri', 'http://127.0.0.1:19530')
        self.user = vs.get('user')
        self.password = vs.get('password')
        self.secure = vs.get('secure', False)
        self.database = vs.get('database', 'default')
        self.collection_prefix = vs.get('collection_prefix', '')
        self.timeout = int(vs.get('timeout', 30))
        
        # 集合配置
        self.shard_num = int(vs.get('shard_num', 1))
        self.consistency_level = vs.get('consistency_level', 'Session')
        
        # 搜索配置
        search_cfg = vs.get('search') or {}
        self.max_top_k = int(search_cfg.get('max_top_k', 100))
        self.hnsw_ef = int(search_cfg.get('ef', 64))
        self.fetch_payload_in_search = bool(search_cfg.get('fetch_payload_in_search', False))
        
        # Schema 管理器
        self.schema_manager = get_schema_manager(vector_dim=vector_dim)
        self.converter = DocChunkConverter()
        
        # 建立连接
        self._ensure_connected()
        
        logger.info(f"MilvusClient 初始化完成: uri={self.uri}, database={self.database}, vector_dim={vector_dim}")
    
    def _ensure_connected(self) -> None:
        """建立或复用 Milvus 连接"""
        if not connections.has_connection('default'):
            logger.info(f"正在连接 Milvus: uri={self.uri}, secure={self.secure}")
            connections.connect(
                alias='default',
                uri=self.uri,
                user=self.user,
                password=self.password,
                secure=self.secure,
                timeout=self.timeout,
            )
        else:
            logger.info("复用现有 Milvus 连接")
        
        # 切换数据库
        try:
            logger.info(f"切换到数据库: {self.database}")
            milvus_db.using_database(self.database)
        except Exception as e:
            logger.error(f"切换数据库失败: {e}")
            raise
    
    def _collection_name(self, name: str) -> str:
        """获取带前缀的集合名"""
        return f"{self.collection_prefix}{name}"
    
    # ==================== 集合管理 ====================
    
    def create_collection(self, name: str) -> Collection:
        """
        创建新集合（基于 DocChunk Schema）
        
        Args:
            name: 集合名称
            
        Returns:
            Collection 实例
            
        Raises:
            ValueError: 如果集合已存在
        """
        cname = self._collection_name(name)
        if utility.has_collection(cname):
            raise ValueError(f"集合 '{name}' 已存在")
        
        logger.info(f"创建集合: {cname}")
        
        # 创建 Schema
        schema = self.schema_manager.create_collection_schema(enable_dynamic_field=True)
        
        # 创建集合
        collection = Collection(
            name=cname,
            schema=schema,
            shards_num=self.shard_num,
            consistency_level=self.consistency_level
        )
        
        # 创建向量索引
        self.schema_manager.create_vector_index(collection)
        
        # 加载集合
        collection.load()
        
        logger.info(f"集合 '{name}' 创建成功")
        return collection
    
    def delete_collection(self, name: str) -> None:
        """
        删除集合
        
        Args:
            name: 集合名称
            
        Raises:
            ValueError: 如果集合不存在
        """
        cname = self._collection_name(name)
        if not utility.has_collection(cname):
            raise ValueError(f"集合 '{name}' 不存在")
        
        utility.drop_collection(cname)
        logger.info(f"集合 '{name}' 已删除")
    
    def has_collection(self, name: str) -> bool:
        """
        检查集合是否存在
        
        Args:
            name: 集合名称
            
        Returns:
            是否存在
        """
        cname = self._collection_name(name)
        return utility.has_collection(cname)
    
    def get_collection(self, name: str) -> Collection:
        """
        获取集合对象
        
        Args:
            name: 集合名称
            
        Returns:
            Collection 实例
            
        Raises:
            ValueError: 如果集合不存在
        """
        cname = self._collection_name(name)
        if not utility.has_collection(cname):
            raise ValueError(f"集合 '{name}' 不存在")
        
        col = Collection(name=cname)
        col.load()
        return col
    
    def count(self, collection_name: str) -> int:
        """
        获取集合中的文档数量
        
        Args:
            collection_name: 集合名称
            
        Returns:
            文档数量
        """
        col = self.get_collection(collection_name)
        return col.num_entities
    
    # ==================== 数据操作 ====================
    
    def insert(self, collection_name: str, chunks: List[DocChunk]) -> int:
        """
        插入 DocChunk 数据
        
        Args:
            collection_name: 集合名称
            chunks: DocChunk 列表
            
        Returns:
            插入的文档数量
        """
        if not chunks:
            return 0
        
        col = self.get_collection(collection_name)
        
        logger.info(f"向集合 '{collection_name}' 插入 {len(chunks)} 条 DocChunk 数据")
        
        # 转换为 Milvus 列式格式
        data = self.converter.batch_to_milvus_insert_format(chunks)
        
        # 插入数据
        mr = col.insert(data)
        col.flush()
        
        logger.info(f"成功插入 {mr.insert_count} 条数据")
        return mr.insert_count
    
    def delete_by_doc_chunk_ids(
        self,
        collection_name: str,
        doc_ids: Optional[List[int]] = None,
        chunk_ids: Optional[List[int]] = None
    ) -> int:
        """
        根据 docId 或 chunkId 删除数据
        
        Args:
            collection_name: 集合名称
            doc_ids: 文档ID列表（可选）
            chunk_ids: 文本块ID列表（可选）
            
        Returns:
            删除的数据条数
        """
        col = self.get_collection(collection_name)
        
        # 构建删除表达式
        conditions = []
        if doc_ids:
            conditions.append(f"doc_id in {doc_ids}")
        if chunk_ids:
            conditions.append(f"chunk_id in {chunk_ids}")
        
        if not conditions:
            raise ValueError("必须提供 doc_ids 或 chunk_ids")
        
        expr = " or ".join(conditions)
        
        logger.info(f"从集合 '{collection_name}' 删除数据: {expr}")
        col.delete(expr)
        col.flush()
        
        # 注意：Milvus 的 delete 不返回删除数量，这里返回请求删除的ID数量
        count = (len(doc_ids) if doc_ids else 0) + (len(chunk_ids) if chunk_ids else 0)
        logger.info(f"删除操作完成，请求删除 {count} 个ID")
        return count
    
    def query_by_doc_chunk_ids(
        self,
        collection_name: str,
        doc_ids: Optional[List[int]] = None,
        chunk_ids: Optional[List[int]] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[DocChunk]:
        """
        根据 docId 或 chunkId 查询数据
        
        Args:
            collection_name: 集合名称
            doc_ids: 文档ID列表（可选）
            chunk_ids: 文本块ID列表（可选）
            output_fields: 输出字段列表（可选）
            
        Returns:
            DocChunk 列表
        """
        col = self.get_collection(collection_name)
        
        # 构建查询表达式
        conditions = []
        if doc_ids:
            conditions.append(f"doc_id in {doc_ids}")
        if chunk_ids:
            conditions.append(f"chunk_id in {chunk_ids}")
        
        if not conditions:
            raise ValueError("必须提供 doc_ids 或 chunk_ids")
        
        expr = " or ".join(conditions)
        
        # 确定输出字段
        if output_fields is None:
            output_fields = self.schema_manager.get_output_fields_for_search(include_vector=True)
        
        logger.info(f"从集合 '{collection_name}' 查询数据: {expr}")
        results = col.query(expr=expr, output_fields=output_fields)
        
        # 转换为 DocChunk
        chunks = [DocChunk.from_milvus_result(r) for r in results]
        logger.info(f"查询到 {len(chunks)} 条 DocChunk 数据")
        return chunks
    
    # ==================== 向量检索 ====================
    
    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        向量检索
        
        Args:
            collection_name: 集合名称
            query_vectors: 查询向量列表
            top_k: 返回 top K 个结果
            expr: 过滤表达式（可选）
            output_fields: 输出字段列表（可选）
            
        Returns:
            检索结果，每个查询向量对应一组结果
        """
        col = self.get_collection(collection_name)
        
        # 限制 top_k
        limit = min(max(1, top_k), self.max_top_k)
        
        # 获取搜索参数
        search_params = self.schema_manager.get_search_params(ef=self.hnsw_ef)
        
        # 确定输出字段
        if output_fields is None:
            output_fields = self.schema_manager.get_output_fields_for_search(
                include_vector=False
            ) if self.fetch_payload_in_search else []
        
        # 向量字段名
        vector_field = self.schema_manager.get_vector_field_name()
        
        logger.info(
            f"向量检索: collection={collection_name}, top_k={limit}, "
            f"vector_field={vector_field}, params={search_params}"
        )
        
        # 执行搜索
        res = col.search(
            data=query_vectors,
            anns_field=vector_field,
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=output_fields,
            consistency_level=self.consistency_level,
        )
        
        # 解析结果
        results: List[List[Dict[str, Any]]] = []
        
        for hits in res:
            group = []
            for h in hits:
                # COSINE 距离转换为相似度分数
                # Milvus 返回的 COSINE distance = 1 - cosine_similarity
                # 所以 similarity_score = 1 - distance
                # distance 范围: [0, 2]，0表示完全相同，2表示完全相反
                # similarity_score 范围: [-1, 1]，1表示完全相同，-1表示完全相反
                distance = float(h.distance)
                similarity_score = 1.0 - distance
                
                item = {
                    'score': similarity_score,
                    'distance': distance,  # 保留原始距离用于调试
                    'id': h.id if hasattr(h, 'id') else None,
                }
                
                # 提取字段
                if hasattr(h, 'entity'):
                    item['doc_id'] = h.entity.get('doc_id')
                    item['chunk_id'] = h.entity.get('chunk_id')
                    item['chunk_index'] = h.entity.get('chunk_index')
                    item['create_time'] = h.entity.get('create_time')
                    item['update_time'] = h.entity.get('update_time')
                    item['version'] = h.entity.get('version')
                else:
                    # 如果没有获取到 entity，后续需要再次查询
                    item['doc_id'] = None
                    item['chunk_id'] = None
                    item['chunk_index'] = None
                    item['create_time'] = None
                    item['update_time'] = None
                    item['version'] = None
                
                group.append(item)
            results.append(group)
        
        # 如果没有在搜索时获取 payload，需要根据 ID 查询
        if not self.fetch_payload_in_search:
            for group in results:
                ids_to_fetch = [item['id'] for item in group if item['id'] is not None]
                if ids_to_fetch:
                    # 根据主键 ID 查询
                    expr_query = f"id in {ids_to_fetch}"
                    docs = col.query(
                        expr=expr_query,
                        output_fields=self.schema_manager.get_output_fields_for_search(include_vector=False)
                    )
                    doc_map = {d['id']: d for d in docs}
                    
                    for item in group:
                        if item['id'] in doc_map:
                            doc = doc_map[item['id']]
                            item['doc_id'] = doc.get('doc_id')
                            item['chunk_id'] = doc.get('chunk_id')
                            item['chunk_index'] = doc.get('chunk_index')
                            item['create_time'] = doc.get('create_time')
                            item['update_time'] = doc.get('update_time')
                            item['version'] = doc.get('version')
        
        logger.info(f"检索完成，返回 {len(results)} 组结果")
        return results
    
    def search_doc_chunks(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int,
        expr: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        向量检索（返回 DocChunk 和 score）
        
        Args:
            collection_name: 集合名称
            query_vectors: 查询向量列表
            top_k: 返回 top K 个结果
            expr: 过滤表达式（可选）
            
        Returns:
            结果列表，每个元素包含 'chunk' (DocChunk) 和 'score' (float)
        """
        # 执行搜索（不包含向量字段，Milvus search 不支持返回向量）
        search_results = self.search(
            collection_name=collection_name,
            query_vectors=query_vectors,
            top_k=top_k,
            expr=expr,
            output_fields=self.schema_manager.get_output_fields_for_search(include_vector=False)
        )
        
        # 转换为 DocChunk 和 score
        chunk_results = []
        for group in search_results:
            chunks = []
            for item in group:
                chunk = DocChunk.from_milvus_result(item)
                chunks.append({
                    'chunk': chunk,
                    'score': item.get('score', 0.0)
                })
            chunk_results.append(chunks)
        
        return chunk_results

