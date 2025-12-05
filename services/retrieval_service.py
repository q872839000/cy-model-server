"""
检索服务（基于 DocChunk 重构）

完全基于 DocChunk 结构体的检索服务，提供：
1. 集合管理：创建、删除、查询、列表
2. 文档管理：添加、删除、更新、查询（基于 DocChunk）
3. 集合元数据持久化
4. 向量检索与重排序
5. 与现有 API 的兼容性

所有 Milvus 操作都通过 DocChunk 结构体进行，不再有 schema 类型判断。
"""

from typing import List, Dict, Optional, Any
from loguru import logger
from pathlib import Path
import json
from datetime import datetime

from models.schemas import (
    Document, RetrievalQuery, RetrievalResponse, RetrievalItem,
    CreateCollectionRequest, AddDocumentsRequest, CollectionInfo
)
from models.doc_chunk import DocChunk
from core.config import load_settings
from core.registry import REGISTRY
from services.milvus_client import MilvusClient
from services.vector_store_adapter import get_adapter
from services.document_content_service import get_document_content_service


class CollectionMetadataManager:
    """集合元数据管理器：持久化存储集合信息"""
    
    def __init__(self, metadata_path: str = "data/collections_metadata.json"):
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata: Dict[str, Dict[str, Any]] = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """从文件加载元数据"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载集合元数据失败: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """保存元数据到文件"""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存集合元数据失败: {e}")
    
    def add_collection(self, name: str, embedding_model: str, dimension: int, description: str = None) -> None:
        """添加集合元数据（基于 DocChunk，不再需要 schema_type）"""
        self._metadata[name] = {
            "name": name,
            "embedding_model": embedding_model,
            "dimension": dimension,
            "description": description,
            "schema_type": "doc_chunk",  # 统一使用 DocChunk schema
            "created_at": datetime.now().isoformat(),
            "document_count": 0
        }
        self._save_metadata()
        logger.info(f"已添加集合元数据: {name}, schema_type=doc_chunk")
    
    def remove_collection(self, name: str) -> None:
        """删除集合元数据"""
        if name in self._metadata:
            del self._metadata[name]
            self._save_metadata()
            logger.info(f"已删除集合元数据: {name}")
    
    def get_collection(self, name: str) -> Optional[Dict[str, Any]]:
        """获取集合元数据"""
        return self._metadata.get(name)
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """列出所有集合"""
        return list(self._metadata.values())
    
    def update_document_count(self, name: str, count: int) -> None:
        """更新文档数量"""
        if name in self._metadata:
            self._metadata[name]["document_count"] = count
            self._save_metadata()


class RetrievalService:
    """检索服务（基于 DocChunk）"""

    def __init__(self, vector_dim: int = 1024):
        """
        初始化检索服务
        
        Args:
            vector_dim: 向量维度，默认 1024
        """
        self._client = MilvusClient(vector_dim=vector_dim)
        self._metadata_manager = CollectionMetadataManager()
        self._adapter = get_adapter()
        self._content_service = get_document_content_service()
        logger.info(f"RetrievalService 初始化完成，vector_dim={vector_dim}")

    def has_any_embedding_model(self) -> bool:
        """检查是否有可用的embedding模型"""
        return REGISTRY.has_any_embedding()

    def create_collection(self, req: CreateCollectionRequest) -> CollectionInfo:
        """
        创建新集合（基于 DocChunk）
        
        Args:
            req: 创建集合请求
        
        Returns:
            CollectionInfo: 集合信息
        """
        # 检查 embedding 模型是否存在
        engine = REGISTRY.get_embedding(req.embedding_model)
        if engine is None:
            raise ValueError(f"未找到embedding模型: {req.embedding_model}")
        
        # 检查集合是否已存在
        if self._client.has_collection(req.name):
            raise ValueError(f"集合 '{req.name}' 已存在")
        
        # 通过一次编码确定维度
        vec = engine.embed(["dimension probe"])[0]
        dim = len(vec)
        
        # 创建集合（使用 DocChunk schema）
        self._client.create_collection(req.name)
        
        # 保存元数据
        self._metadata_manager.add_collection(
            req.name,
            req.embedding_model,
            dim,
            req.description
        )
        
        logger.info(f"成功创建集合: {req.name}, 维度: {dim}, schema: DocChunk")
        return CollectionInfo(
            name=req.name,
            document_count=0,
            embedding_model=req.embedding_model,
            dimension=dim
        )

    def delete_collection(self, collection_name: str) -> Dict[str, str]:
        """删除集合"""
        self._client.delete_collection(collection_name)
        self._metadata_manager.remove_collection(collection_name)
        logger.info(f"成功删除集合: {collection_name}")
        return {"message": f"集合 '{collection_name}' 已删除", "collection": collection_name}

    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """获取集合信息"""
        if not self._client.has_collection(collection_name):
            raise ValueError(f"集合 '{collection_name}' 不存在")
        
        metadata = self._metadata_manager.get_collection(collection_name)
        if not metadata:
            # 如果元数据不存在，尝试从 Milvus 获取基本信息
            count = self._client.count(collection_name)
            return CollectionInfo(
                name=collection_name,
                document_count=count,
                embedding_model="unknown",
                dimension=None
            )
        
        count = self._client.count(collection_name)
        self._metadata_manager.update_document_count(collection_name, count)
        
        return CollectionInfo(
            name=metadata['name'],
            document_count=count,
            embedding_model=metadata['embedding_model'],
            dimension=metadata.get('dimension')
        )

    def list_collections(self) -> List[CollectionInfo]:
        """列出所有集合"""
        collections = []
        for metadata in self._metadata_manager.list_collections():
            try:
                count = self._client.count(metadata['name'])
                collections.append(CollectionInfo(
                    name=metadata['name'],
                    document_count=count,
                    embedding_model=metadata['embedding_model'],
                    dimension=metadata.get('dimension')
                ))
            except Exception as e:
                logger.warning(f"获取集合 '{metadata['name']}' 信息失败: {e}")
        return collections

    def add_documents(self, req: AddDocumentsRequest) -> Dict[str, Any]:
        """
        添加文档到集合（基于 DocChunk）
        
        Args:
            req: 添加文档请求
            
        Returns:
            添加结果字典
        """
        engine = REGISTRY.get_embedding(req.embedding_model)
        if engine is None:
            raise ValueError(f"未找到embedding模型: {req.embedding_model}")
        
        if not req.documents:
            return {"collection": req.collection, "added_count": 0, "total_count": 0}
        
        # 生成向量
        texts = [d.content for d in req.documents]
        embeddings = engine.embed(texts)
        
        # 转换为 DocChunk
        chunks = self._adapter.batch_documents_to_doc_chunks(req.documents, embeddings)
        
        # 插入数据
        inserted = self._client.insert(req.collection, chunks)
        total = self._client.count(req.collection)
        
        # 更新元数据
        self._metadata_manager.update_document_count(req.collection, total)
        
        logger.info(f"成功向集合 '{req.collection}' 添加 {inserted} 条文档（DocChunk）")
        return {
            "collection": req.collection,
            "added_count": inserted,
            "total_count": total
        }

    def delete_documents(self, collection_name: str, doc_ids: List[str]) -> Dict[str, Any]:
        """
        从集合中删除文档（基于 DocChunk）
        
        Args:
            collection_name: 集合名称
            doc_ids: 文档ID列表（格式："docId_chunkId" 或 "docId"）
            
        Returns:
            删除结果字典
        """
        # 解析 doc_ids，提取 docId 和 chunkId
        doc_id_set = set()
        chunk_id_set = set()
        
        for doc_id in doc_ids:
            parts = doc_id.split("_")
            try:
                doc_id_int = int(parts[0])
                doc_id_set.add(doc_id_int)
                
                if len(parts) > 1:
                    chunk_id_int = int(parts[1])
                    chunk_id_set.add(chunk_id_int)
            except ValueError:
                logger.warning(f"无效的文档ID格式: {doc_id}")
                continue
        
        # 删除数据
        deleted = self._client.delete_by_doc_chunk_ids(
            collection_name,
            doc_ids=list(doc_id_set) if doc_id_set else None,
            chunk_ids=list(chunk_id_set) if chunk_id_set else None
        )
        
        total = self._client.count(collection_name)
        
        # 更新元数据
        self._metadata_manager.update_document_count(collection_name, total)
        
        return {
            "collection": collection_name,
            "deleted_count": deleted,
            "total_count": total
        }

    def update_document(
        self,
        collection_name: str,
        doc_id: str,
        new_doc: Document,
        embedding_model: str
    ) -> Dict[str, str]:
        """
        更新单个文档（基于 DocChunk）
        
        Args:
            collection_name: 集合名称
            doc_id: 文档ID
            new_doc: 新文档对象
            embedding_model: Embedding 模型名称
            
        Returns:
            更新结果字典
        """
        engine = REGISTRY.get_embedding(embedding_model)
        if engine is None:
            raise ValueError(f"未找到embedding模型: {embedding_model}")
        
        # 生成新向量
        new_embedding = engine.embed([new_doc.content])[0]
        
        # 转换为 DocChunk
        chunk = self._adapter.document_to_doc_chunk(new_doc, new_embedding)
        
        # 先删除旧数据
        self.delete_documents(collection_name, [doc_id])
        
        # 插入新数据
        self._client.insert(collection_name, [chunk])
        
        return {
            "message": "文档更新成功",
            "collection": collection_name,
            "document_id": doc_id
        }

    def query_documents(self, collection_name: str, doc_ids: List[str]) -> List[Document]:
        """
        根据ID查询文档（基于 DocChunk）
        
        Args:
            collection_name: 集合名称
            doc_ids: 文档ID列表
            
        Returns:
            Document 列表
        """
        # 解析 doc_ids
        doc_id_set = set()
        chunk_id_set = set()
        
        for doc_id in doc_ids:
            parts = doc_id.split("_")
            try:
                doc_id_int = int(parts[0])
                doc_id_set.add(doc_id_int)
                
                if len(parts) > 1:
                    chunk_id_int = int(parts[1])
                    chunk_id_set.add(chunk_id_int)
            except ValueError:
                continue
        
        # 查询 DocChunk
        chunks = self._client.query_by_doc_chunk_ids(
            collection_name,
            doc_ids=list(doc_id_set) if doc_id_set else None,
            chunk_ids=list(chunk_id_set) if chunk_id_set else None
        )
        
        # 转换为 Document
        documents = []
        for chunk in chunks:
            doc = self._adapter.doc_chunk_to_document(chunk)
            documents.append(doc)
        
        return documents

    def search(self, req: RetrievalQuery) -> RetrievalResponse:
        """
        向量检索（基于 DocChunk，可选重排序）
        
        Args:
            req: 检索请求
            
        Returns:
            RetrievalResponse: 检索结果
        """
        engine = REGISTRY.get_embedding(req.embedding_model)
        if engine is None:
            raise ValueError(f"未找到embedding模型: {req.embedding_model}")
        
        # 生成查询向量
        qv = engine.embed([req.query])[0]
        
        # 向量检索（返回 DocChunk 和 score）
        chunk_groups = self._client.search_doc_chunks(
            collection_name=req.collection,
            query_vectors=[qv],
            top_k=req.top_k
        )
        
        # 获取第一组结果
        chunk_results = chunk_groups[0] if chunk_groups else []
        
        # 批量查询 MySQL 获取真实文档内容
        chunk_ids = [result['chunk'].chunk_id for result in chunk_results]
        content_map = {}
        if chunk_ids:
            try:
                content_map = self._content_service.batch_get_chunk_contents(chunk_ids)
                logger.info(f"从 MySQL 批量查询到 {len(content_map)} 条文档内容")
            except Exception as e:
                logger.warning(f"从 MySQL 查询文档内容失败，将使用占位符: {e}")
        
        # 构建检索结果
        items: List[RetrievalItem] = []
        for result in chunk_results:
            chunk = result['chunk']
            score = result['score']
            
            # 从 content_map 中获取真实内容
            content = content_map.get(chunk.chunk_id)
            
            # 转换为 Document
            doc = self._adapter.doc_chunk_to_document(chunk, content=content)
            
            # 创建 RetrievalItem
            items.append(RetrievalItem(
                document=doc,
                score=score
            ))
        
        # 应用相似度阈值
        if req.similarity_threshold is not None:
            items = [it for it in items if it.score >= req.similarity_threshold]
        
        # 重排序（如果有 reranker）
        if REGISTRY.has_any_reranker() and len(items) > 1:
            try:
                reranker_engine = REGISTRY.get_reranker(None)
                if reranker_engine is not None:
                    documents = [it.document.content for it in items]
                    rerank_scores = reranker_engine.rerank(req.query, documents)
                    
                    # 使用 reranker 分数
                    for it, sc in zip(items, rerank_scores):
                        it.score = float(sc)
                    
                    # 重新排序
                    items.sort(key=lambda x: x.score, reverse=True)
                    
                    # 应用 top_k 和阈值
                    items = items[:req.top_k]
                    if req.similarity_threshold is not None:
                        items = [it for it in items if it.score >= req.similarity_threshold]
                    
                    logger.info(f"已使用reranker重排序，最终返回 {len(items)} 条结果")
            except Exception as e:
                logger.warning(f"Reranker重排序失败，使用原始检索结果: {e}")

        return RetrievalResponse(
            items=items,
            query=req.query,
            embedding_model=req.embedding_model,
            collection=req.collection
        )
