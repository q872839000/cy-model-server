"""
向量存储适配器（基于 DocChunk 重构）

提供统一的向量存储接口，彻底消除 schema 类型判断和 if-else 分支。
所有数据转换、格式化操作都基于 DocChunk 结构体。

主要功能：
1. Legacy Document 与 DocChunk 之间的转换
2. 批量数据插入格式化
3. 搜索结果解析
4. 统一的数据访问接口
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from models.doc_chunk import DocChunk, DocChunkConverter
from models.schemas import Document


class DocChunkAdapter:
    """
    DocChunk 适配器
    
    提供 DocChunk 与 Document 之间的转换，以及各种数据格式化功能。
    不再依赖 schema 类型判断，统一使用 DocChunk 作为数据载体。
    """
    
    def __init__(self):
        """初始化适配器"""
        self.converter = DocChunkConverter()
        logger.info("DocChunkAdapter 初始化完成")
    
    def document_to_doc_chunk(
        self,
        doc: Document,
        embedding: List[float],
        chunk_index: int = 0
    ) -> DocChunk:
        """
        将 Document 转换为 DocChunk
        
        Args:
            doc: Document 对象
            embedding: 向量表示
            chunk_index: 文本块索引
            
        Returns:
            DocChunk 实例
        """
        metadata = doc.metadata or {}
        
        # 从 metadata 中提取 doc_id 和 chunk_id
        doc_id = metadata.get("doc_id") or metadata.get("docId")
        chunk_id = metadata.get("chunk_id") or metadata.get("chunkId")
        
        # 如果 metadata 中没有，尝试从 doc.id 解析
        if doc_id is None or chunk_id is None:
            # 假设 id 格式为 "123" 或 "123_456"
            parts = doc.id.split("_")
            try:
                if doc_id is None:
                    doc_id = int(parts[0])
                if chunk_id is None:
                    chunk_id = int(parts[1]) if len(parts) > 1 else chunk_index
            except (ValueError, IndexError):
                # 如果解析失败，使用 hash 和索引
                if doc_id is None:
                    doc_id = hash(doc.id) % (2**31)
                if chunk_id is None:
                    chunk_id = chunk_index
        
        # 获取其他字段
        chunk_idx = metadata.get("chunk_index") or metadata.get("chunkIndex") or chunk_index
        version = metadata.get("version", 1)
        
        return self.converter.from_document_content(
            doc_id=doc_id,
            chunk_id=chunk_id,
            content=doc.content,
            embedding=embedding,
            chunk_index=chunk_idx,
            version=version
        )
    
    def doc_chunk_to_document(
        self,
        chunk: DocChunk,
        content: Optional[str] = None
    ) -> Document:
        """
        将 DocChunk 转换为 Document
        
        Args:
            chunk: DocChunk 实例
            content: 文本内容（可选，DocChunk 本身不存储文本）
                    如果为 None，将使用占位符
            
        Returns:
            Document 对象
        """
        # 构造 ID
        combined_id = f"{chunk.doc_id}_{chunk.chunk_id}"
        
        # 构造 metadata
        metadata = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "create_time": chunk.create_time,
            "update_time": chunk.update_time,
            "version": chunk.version,
        }
        
        # 添加原始 ID（如果有）
        if chunk.id is not None:
            metadata["original_id"] = chunk.id
        
        return Document(
            id=combined_id,
            content=content or f"Document {chunk.doc_id}, Chunk {chunk.chunk_id}",
            metadata=metadata
        )
    
    def batch_doc_chunks_to_documents(
        self,
        chunks: List[DocChunk],
        content_map: Optional[Dict[int, str]] = None
    ) -> List[Document]:
        """
        批量将 DocChunk 转换为 Document
        
        Args:
            chunks: DocChunk 列表
            content_map: chunk_id -> content 的映射字典（可选）
            
        Returns:
            Document 列表
        """
        documents = []
        for chunk in chunks:
            # 从 content_map 中获取真实内容
            content = None
            if content_map:
                content = content_map.get(chunk.chunk_id)
            
            doc = self.doc_chunk_to_document(chunk, content)
            documents.append(doc)
        
        logger.info(f"批量转换完成：{len(chunks)} 个 DocChunk -> Document")
        return documents
    
    def batch_documents_to_doc_chunks(
        self,
        docs: List[Document],
        embeddings: List[List[float]]
    ) -> List[DocChunk]:
        """
        批量将 Document 转换为 DocChunk
        
        Args:
            docs: Document 列表
            embeddings: 对应的向量列表
            
        Returns:
            DocChunk 列表
        """
        if len(docs) != len(embeddings):
            raise ValueError(f"文档数量 ({len(docs)}) 与向量数量 ({len(embeddings)}) 不匹配")
        
        chunks = []
        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            chunk = self.document_to_doc_chunk(doc, emb, chunk_index=i)
            chunks.append(chunk)
        
        logger.info(f"批量转换完成：{len(chunks)} 个 Document -> DocChunk")
        return chunks
    
    def prepare_insert_data(
        self,
        docs: List[Document],
        embeddings: List[List[float]]
    ) -> List[List[Any]]:
        """
        准备插入数据（Milvus 列式格式）
        
        Args:
            docs: Document 列表
            embeddings: 向量列表
            
        Returns:
            Milvus 插入数据（列式）
        """
        chunks = self.batch_documents_to_doc_chunks(docs, embeddings)
        return self.converter.batch_to_milvus_insert_format(chunks)
    
    def parse_search_result(
        self,
        result_data: Dict[str, Any],
        score: float,
        content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        解析单个搜索结果
        
        Args:
            result_data: Milvus 搜索结果数据
            score: 相似度得分
            content: 文本内容（可选）
            
        Returns:
            解析后的结果字典
        """
        # 从结果中构造 DocChunk
        chunk = DocChunk.from_milvus_result(result_data)
        
        # 转换为检索结果格式
        return self.converter.to_retrieval_result(chunk, score, content)
    
    def parse_search_results(
        self,
        results: List[Dict[str, Any]],
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量解析搜索结果
        
        Args:
            results: Milvus 搜索结果列表
            include_vectors: 是否包含向量
            
        Returns:
            解析后的结果列表
        """
        parsed = []
        
        for item in results:
            score = item.get("score", 0.0)
            content = item.get("content")
            
            parsed_item = self.parse_search_result(item, score, content)
            
            # 如果需要包含向量
            if include_vectors and "contentVector" in item:
                parsed_item["vector"] = item["contentVector"]
            
            parsed.append(parsed_item)
        
        logger.info(f"批量解析完成：{len(parsed)} 条搜索结果")
        return parsed
    
    def create_doc_chunk_from_raw_data(
        self,
        doc_id: int,
        chunk_id: int,
        content: str,
        embedding: List[float],
        chunk_index: int = 0,
        meta_data: Optional[Dict[str, Any]] = None,
        version: int = 1
    ) -> DocChunk:
        """
        从原始数据创建 DocChunk
        
        Args:
            doc_id: 文档ID
            chunk_id: 文本块ID
            content: 文本内容（不存储，仅用于生成向量）
            embedding: 向量表示
            chunk_index: 文本块索引
            meta_data: 元数据
            version: 版本号
            
        Returns:
            DocChunk 实例
        """
        return self.converter.from_document_content(
            doc_id=doc_id,
            chunk_id=chunk_id,
            content=content,
            embedding=embedding,
            chunk_index=chunk_index,
            meta_data=meta_data,
            version=version
        )


# 全局适配器实例（单例）
_global_adapter: DocChunkAdapter = None


def get_adapter() -> DocChunkAdapter:
    """
    获取全局适配器实例
    
    Returns:
        DocChunkAdapter 实例
    """
    global _global_adapter
    
    if _global_adapter is None:
        _global_adapter = DocChunkAdapter()
    
    return _global_adapter


def create_adapter() -> DocChunkAdapter:
    """
    创建新的适配器实例（兼容旧接口）
    
    Returns:
        DocChunkAdapter 实例
    """
    return DocChunkAdapter()
