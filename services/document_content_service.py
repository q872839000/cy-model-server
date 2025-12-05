"""
文档内容服务

从 MySQL 数据库查询文档原文内容，支持：
1. 根据 chunk_id 批量查询
2. 缓存优化
3. 异常处理和降级
"""

from typing import List, Dict, Optional
from loguru import logger

from services.mysql_client import get_mysql_client


class DocumentContentService:
    """
    文档内容服务
    
    从 MySQL 的 document_chunk 表中查询文档原文内容。
    """
    
    def __init__(self):
        """初始化文档内容服务"""
        self.mysql_client = get_mysql_client()
        self.table_name = "document_chunk"
        logger.info("DocumentContentService 初始化完成")
    
    def get_chunk_content_by_id(self, chunk_id: int) -> Optional[str]:
        """
        根据 chunk_id 查询单个文档块的内容
        
        Args:
            chunk_id: 文档块 ID
            
        Returns:
            文档内容文本，如果未找到返回 None
        """
        try:
            query = f"""
                SELECT chunk_text
                FROM {self.table_name}
                WHERE id = :chunk_id
                LIMIT 1
            """
            
            results = self.mysql_client.execute_query(
                query,
                {"chunk_id": chunk_id}
            )
            
            if results:
                content = results[0].get("chunk_text")
                return content
            else:
                logger.warning(f"未找到 chunk_id={chunk_id} 的文档内容")
                return None
                
        except Exception as e:
            logger.error(f"查询 chunk_id={chunk_id} 内容失败: {e}")
            return None
    
    def batch_get_chunk_contents(
        self,
        chunk_ids: List[int]
    ) -> Dict[int, str]:
        """
        批量查询文档块内容
        
        Args:
            chunk_ids: 文档块 ID 列表
            
        Returns:
            字典，key 为 chunk_id，value 为文档内容
        """
        if not chunk_ids:
            return {}
        
        try:
            # 使用 IN 查询批量获取
            # 注意：SQLAlchemy 的 text() 不支持列表参数，需要手动构建
            placeholders = ','.join([f':id{i}' for i in range(len(chunk_ids))])
            query = f"""
                SELECT id, chunk_text
                FROM {self.table_name}
                WHERE id IN ({placeholders})
            """
            
            # 构建参数字典
            params = {f'id{i}': chunk_id for i, chunk_id in enumerate(chunk_ids)}
            
            results = self.mysql_client.execute_query(query, params)
            
            # 转换为字典
            content_map = {
                row['id']: row['chunk_text']
                for row in results
                if row.get('chunk_text')
            }
            
            # 日志统计
            found_count = len(content_map)
            missing_count = len(chunk_ids) - found_count
            
            if missing_count > 0:
                missing_ids = set(chunk_ids) - set(content_map.keys())
                logger.warning(
                    f"批量查询完成：找到 {found_count} 条，缺失 {missing_count} 条。"
                    f"缺失 ID: {list(missing_ids)[:5]}{'...' if missing_count > 5 else ''}"
                )
            else:
                logger.info(f"批量查询成功：找到 {found_count} 条文档内容")
            
            return content_map
            
        except Exception as e:
            logger.error(f"批量查询文档内容失败: {e}, chunk_ids={chunk_ids[:10]}...")
            # 降级：返回空字典
            return {}
    
    def get_chunk_full_info(self, chunk_id: int) -> Optional[Dict]:
        """
        获取文档块的完整信息（包括关联的知识库ID、文档ID等）
        
        Args:
            chunk_id: 文档块 ID
            
        Returns:
            完整信息字典，包含所有字段
        """
        try:
            query = f"""
                SELECT 
                    id,
                    knowledgeAppId,
                    docId,
                    chunk_text,
                    chunkIndex
                FROM {self.table_name}
                WHERE id = :chunk_id
                LIMIT 1
            """
            
            results = self.mysql_client.execute_query(
                query,
                {"chunk_id": chunk_id}
            )
            
            if results:
                return results[0]
            else:
                logger.warning(f"未找到 chunk_id={chunk_id} 的文档信息")
                return None
                
        except Exception as e:
            logger.error(f"查询 chunk_id={chunk_id} 完整信息失败: {e}")
            return None
    
    def batch_get_chunk_full_info(
        self,
        chunk_ids: List[int]
    ) -> Dict[int, Dict]:
        """
        批量获取文档块的完整信息
        
        Args:
            chunk_ids: 文档块 ID 列表
            
        Returns:
            字典，key 为 chunk_id，value 为完整信息字典
        """
        if not chunk_ids:
            return {}
        
        try:
            placeholders = ','.join([f':id{i}' for i in range(len(chunk_ids))])
            query = f"""
                SELECT 
                    id,
                    knowledgeAppId,
                    docId,
                    chunk_text,
                    chunkIndex
                FROM {self.table_name}
                WHERE id IN ({placeholders})
            """
            
            params = {f'id{i}': chunk_id for i, chunk_id in enumerate(chunk_ids)}
            results = self.mysql_client.execute_query(query, params)
            
            # 转换为字典
            info_map = {
                row['id']: row
                for row in results
            }
            
            logger.info(f"批量查询完整信息成功：找到 {len(info_map)} 条")
            return info_map
            
        except Exception as e:
            logger.error(f"批量查询完整信息失败: {e}")
            return {}
    
    def health_check(self) -> bool:
        """
        健康检查：测试 MySQL 连接和表是否存在
        
        Returns:
            True 表示正常，False 表示异常
        """
        try:
            query = f"SELECT COUNT(*) as count FROM {self.table_name} LIMIT 1"
            results = self.mysql_client.execute_query(query)
            logger.info(f"DocumentContentService 健康检查通过，表 {self.table_name} 存在")
            return True
        except Exception as e:
            logger.error(f"DocumentContentService 健康检查失败: {e}")
            return False


# 全局单例服务
_global_content_service: Optional[DocumentContentService] = None


def get_document_content_service() -> DocumentContentService:
    """
    获取全局文档内容服务实例（单例模式）
    
    Returns:
        DocumentContentService 实例
    """
    global _global_content_service
    
    if _global_content_service is None:
        _global_content_service = DocumentContentService()
    
    return _global_content_service

