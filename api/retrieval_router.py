"""
基础检索API：提供向量集合管理和文档检索功能

本模块提供以下功能：
1. 集合管理：创建、查询、列表、删除向量集合
2. 文档管理：添加、更新、删除、查询文档
3. 向量检索：基于相似度的文档检索，支持重排序

API端点：
- POST /v1/retrieval/collections - 创建新的向量集合
- GET /v1/retrieval/collections - 列出所有集合
- GET /v1/retrieval/collections/{collection_name} - 获取集合信息
- DELETE /v1/retrieval/collections/{collection_name} - 删除集合
- POST /v1/retrieval/collections/{collection_name}/documents - 添加文档
- DELETE /v1/retrieval/collections/{collection_name}/documents - 删除文档
- PUT /v1/retrieval/collections/{collection_name}/documents/{doc_id} - 更新文档
- POST /v1/retrieval/collections/{collection_name}/query - 根据ID查询文档
- POST /v1/retrieval/search - 向量检索
"""

from fastapi import APIRouter, HTTPException, Request
from typing import List
from pydantic import BaseModel

from models.schemas import (
    RetrievalQuery, RetrievalResponse, CollectionInfo,
    CreateCollectionRequest, AddDocumentsRequest, Document
)
from services.container import CONTAINER


router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])


@router.post("/collections", response_model=CollectionInfo)
async def create_collection(req: CreateCollectionRequest, request: Request):
    """
    创建新的向量集合
    
    根据指定的embedding模型创建向量集合，系统会自动检测向量维度并创建相应的索引。
    
    参数:
        req: 创建集合请求，包含：
            - name: 集合名称（唯一标识）
            - embedding_model: 绑定的embedding模型名称
            - description: 集合描述（可选）
        request: FastAPI请求对象
        
    返回:
        CollectionInfo: 创建成功的集合信息，包含名称、文档数、模型、维度等
        
    异常:
        400: 集合已存在或embedding模型不存在
        500: 创建失败（数据库错误等）
    """
    try:
        svc = CONTAINER.get_retrieval_service()
        return svc.create_collection(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建集合失败: {str(e)}")


@router.get("/collections", response_model=List[CollectionInfo])
async def list_collections(request: Request):
    """
    列出所有向量集合
    
    返回系统中所有已创建的向量集合及其统计信息。
    
    参数:
        request: FastAPI请求对象
        
    返回:
        List[CollectionInfo]: 集合信息列表
        
    异常:
        500: 获取失败
    """
    try:
        svc = CONTAINER.get_retrieval_service()
        return svc.list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取集合列表失败: {str(e)}")


@router.get("/collections/{collection_name}", response_model=CollectionInfo)
async def get_collection(collection_name: str, request: Request):
    """
    获取指定集合的详细信息
    
    查询特定集合的元数据，包括文档数量、向量维度、绑定的模型等。
    
    参数:
        collection_name: 集合名称
        request: FastAPI请求对象
        
    返回:
        CollectionInfo: 集合详细信息
        
    异常:
        404: 集合不存在
        500: 查询失败
    """
    try:
        svc = CONTAINER.get_retrieval_service()
        return svc.get_collection_info(collection_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取集合信息失败: {str(e)}")


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str, request: Request):
    """
    删除指定的向量集合
    
    永久删除集合及其包含的所有文档和向量数据，此操作不可逆。
    
    参数:
        collection_name: 要删除的集合名称
        request: FastAPI请求对象
        
    返回:
        dict: 删除确认消息
        
    异常:
        404: 集合不存在
        500: 删除失败
    """
    try:
        svc = CONTAINER.get_retrieval_service()
        return svc.delete_collection(collection_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除集合失败: {str(e)}")


@router.post("/collections/{collection_name}/documents")
async def add_documents(collection_name: str, req: AddDocumentsRequest, request: Request):
    """
    向集合中批量添加文档
    
    将文档列表添加到指定集合，系统会自动：
    1. 使用指定的embedding模型生成向量
    2. 存储文档内容、元数据和向量
    3. 更新集合统计信息
    
    参数:
        collection_name: 目标集合名称
        req: 添加文档请求，包含：
            - collection: 集合名称（需与URL一致）
            - documents: 文档列表（id、content、metadata）
            - embedding_model: 用于生成向量的模型
        request: FastAPI请求对象
        
    返回:
        dict: 包含添加的文档数量和集合总数
        
    异常:
        400: URL与请求体中的集合名不一致或参数错误
        500: 添加失败（向量生成错误、数据库错误等）
    """
    try:
        if req.collection != collection_name:
            raise HTTPException(status_code=400, detail="URL 与请求体中的集合名不一致")
        svc = CONTAINER.get_retrieval_service()
        return svc.add_documents(req)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加文档失败: {str(e)}")


class DeleteDocumentsRequest(BaseModel):
    """
    删除文档请求
    
    属性:
        doc_ids: 要删除的文档ID列表
    """
    doc_ids: List[str]


@router.delete("/collections/{collection_name}/documents")
async def delete_documents(collection_name: str, req: DeleteDocumentsRequest, request: Request):
    """
    从集合中批量删除文档
    
    根据文档ID列表删除指定的文档及其向量数据。
    
    参数:
        collection_name: 集合名称
        req: 删除请求，包含要删除的文档ID列表
        request: FastAPI请求对象
        
    返回:
        dict: 包含删除的文档数量和剩余总数
        
    异常:
        400: 参数错误
        500: 删除失败
    """
    try:
        svc = CONTAINER.get_retrieval_service()
        return svc.delete_documents(collection_name, req.doc_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


class UpdateDocumentRequest(BaseModel):
    """
    更新文档请求
    
    属性:
        document: 新的文档对象（包含更新后的内容和元数据）
        embedding_model: 用于重新生成向量的模型
    """
    document: Document
    embedding_model: str


@router.put("/collections/{collection_name}/documents/{doc_id}")
async def update_document(collection_name: str, doc_id: str, req: UpdateDocumentRequest, request: Request):
    """
    更新集合中的单个文档
    
    更新文档的内容、元数据，并重新生成向量。采用删除后重新插入的方式实现。
    
    参数:
        collection_name: 集合名称
        doc_id: 要更新的文档ID
        req: 更新请求，包含新文档对象和embedding模型
        request: FastAPI请求对象
        
    返回:
        dict: 更新确认消息
        
    异常:
        400: 参数错误
        500: 更新失败
    """
    try:
        svc = CONTAINER.get_retrieval_service()
        # 确保文档ID与URL中的一致
        req.document.id = doc_id
        return svc.update_document(collection_name, doc_id, req.document, req.embedding_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新文档失败: {str(e)}")


class QueryDocumentsRequest(BaseModel):
    """
    查询文档请求
    
    属性:
        doc_ids: 要查询的文档ID列表
    """
    doc_ids: List[str]


@router.post("/collections/{collection_name}/query", response_model=List[Document])
async def query_documents(collection_name: str, req: QueryDocumentsRequest, request: Request):
    """
    根据ID精确查询文档
    
    通过文档ID列表批量查询文档的完整信息（不进行向量检索）。
    
    参数:
        collection_name: 集合名称
        req: 查询请求，包含文档ID列表
        request: FastAPI请求对象
        
    返回:
        List[Document]: 查询到的文档列表
        
    异常:
        400: 参数错误
        500: 查询失败
    """
    try:
        svc = CONTAINER.get_retrieval_service()
        return svc.query_documents(collection_name, req.doc_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询文档失败: {str(e)}")


@router.post("/search", response_model=RetrievalResponse)
async def search_documents(req: RetrievalQuery, request: Request):
    """
    向量相似度检索
    
    基于查询文本进行向量检索，返回最相似的文档列表。可选支持：
    - 相似度阈值过滤
    - Reranker模型重排序（如果系统中有可用的reranker模型）
    
    工作流程：
    1. 使用embedding模型将查询文本向量化
    2. 在向量数据库中进行相似度检索
    3. （可选）使用reranker模型对结果重排序
    4. 返回top_k个最相关的文档
    
    参数:
        req: 检索请求，包含：
            - query: 查询文本
            - collection: 集合名称
            - embedding_model: 用于向量化的模型
            - top_k: 返回结果数量（默认10）
            - similarity_threshold: 相似度阈值（可选）
        request: FastAPI请求对象
        
    返回:
        RetrievalResponse: 检索结果，包含：
            - items: 检索到的文档列表（按相似度降序）
            - query: 原始查询文本
            - embedding_model: 使用的模型
            - collection: 检索的集合
            
    异常:
        400: 参数错误（集合不存在、模型不存在等）
        500: 检索失败
    """
    try:
        svc = CONTAINER.get_retrieval_service()
        return svc.search(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
