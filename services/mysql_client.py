"""
MySQL 数据库客户端

提供生产级的 MySQL 连接管理和查询封装，支持：
1. 连接池管理
2. 自动重连
3. 异常处理
4. 查询日志
"""

from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from loguru import logger
import yaml

from sqlalchemy import create_engine, text, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from core.config import load_settings


class MySQLClient:
    """
    MySQL 客户端（基于 SQLAlchemy）
    
    提供数据库连接池管理和基础查询操作，线程安全。
    """
    
    def __init__(self):
        """初始化 MySQL 客户端"""
        # 加载配置
        settings = load_settings()
        cfg = yaml.safe_load(open(settings.models_config_path, 'r', encoding='utf-8')) or {}
        mysql_cfg = cfg.get('mysql') or {}
        
        # 数据库连接参数
        self.host = mysql_cfg.get('host', '127.0.0.1')
        self.port = int(mysql_cfg.get('port', 3306))
        self.user = mysql_cfg.get('user', 'root')
        self.password = mysql_cfg.get('password', '')
        self.database = mysql_cfg.get('database', 'knowledge')
        self.charset = mysql_cfg.get('charset', 'utf8mb4')
        
        # 连接池参数
        self.pool_size = int(mysql_cfg.get('pool_size', 10))
        self.max_overflow = int(mysql_cfg.get('max_overflow', 20))
        self.pool_timeout = int(mysql_cfg.get('pool_timeout', 30))
        self.pool_recycle = int(mysql_cfg.get('pool_recycle', 3600))
        
        # 创建数据库引擎
        self._engine = None
        self._session_factory = None
        self._init_engine()
        
        logger.info(
            f"MySQLClient 初始化完成: {self.user}@{self.host}:{self.port}/{self.database}"
        )
    
    def _init_engine(self):
        """初始化数据库引擎和会话工厂"""
        # 构建连接字符串
        connection_url = (
            f"mysql+pymysql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?charset={self.charset}"
        )
        
        # 创建引擎（使用连接池）
        self._engine = create_engine(
            connection_url,
            poolclass=pool.QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            pool_pre_ping=True,  # 使用前检查连接是否有效
            echo=False,  # 生产环境关闭 SQL 日志
        )
        
        # 创建会话工厂
        self._session_factory = sessionmaker(bind=self._engine)
        
        logger.info("MySQL 数据库引擎创建成功")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        获取数据库会话（上下文管理器）
        
        使用方式：
            with client.get_session() as session:
                result = session.execute(...)
        
        Yields:
            Session: SQLAlchemy 会话对象
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库事务失败，已回滚: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行查询语句并返回结果
        
        Args:
            query: SQL 查询语句（使用 :param 格式的参数占位符）
            params: 查询参数字典
            
        Returns:
            查询结果列表，每个元素是一个字典
            
        Example:
            results = client.execute_query(
                "SELECT * FROM users WHERE id = :user_id",
                {"user_id": 123}
            )
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                # 转换为字典列表
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
        except SQLAlchemyError as e:
            logger.error(f"查询执行失败: {e}\nSQL: {query}\n参数: {params}")
            raise
    
    def execute_batch_query(
        self,
        query: str,
        params_list: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        批量执行查询（每组参数执行一次）
        
        Args:
            query: SQL 查询语句
            params_list: 参数字典列表
            
        Returns:
            结果列表的列表
        """
        results = []
        for params in params_list:
            result = self.execute_query(query, params)
            results.append(result)
        return results
    
    def health_check(self) -> bool:
        """
        健康检查：测试数据库连接是否正常
        
        Returns:
            True 表示连接正常，False 表示连接失败
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("MySQL 健康检查通过")
            return True
        except Exception as e:
            logger.error(f"MySQL 健康检查失败: {e}")
            return False
    
    def close(self):
        """关闭数据库连接池"""
        if self._engine:
            self._engine.dispose()
            logger.info("MySQL 连接池已关闭")


# 全局单例客户端
_global_mysql_client: Optional[MySQLClient] = None


def get_mysql_client() -> MySQLClient:
    """
    获取全局 MySQL 客户端实例（单例模式）
    
    Returns:
        MySQLClient 实例
    """
    global _global_mysql_client
    
    if _global_mysql_client is None:
        _global_mysql_client = MySQLClient()
    
    return _global_mysql_client

