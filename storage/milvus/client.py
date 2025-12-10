"""
Milvus 客户端封装

本模块提供 Milvus 客户端的封装，包括：
- 连接管理（支持连接池复用）
- 健康检查
- 资源清理

设计原则：
1. 单例模式，避免重复创建连接
2. 线程安全
3. 优雅的错误处理
"""

from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger

from storage.milvus.config import MilvusConfig, load_milvus_config

# 延迟导入 pymilvus，避免未安装时启动失败
_pymilvus_available = False
try:
    from pymilvus import connections, utility
    _pymilvus_available = True
except ImportError:
    pass


class MilvusConnectionError(Exception):
    """Milvus 连接异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class MilvusClient:
    """
    Milvus 客户端封装。
    
    提供 Milvus 连接的统一管理，支持：
    - 延迟连接（首次使用时建立）
    - 连接复用（单例模式）
    - 健康检查
    - 优雅断开
    
    Usage:
        >>> client = MilvusClient()
        >>> client.connect()
        >>> if client.is_healthy():
        ...     # 执行操作
        >>> client.disconnect()
    
    或使用上下文管理器：
        >>> with MilvusClient() as client:
        ...     # 执行操作
    """
    
    _instance: Optional["MilvusClient"] = None
    _lock: Lock = Lock()
    
    def __new__(cls, config: Optional[MilvusConfig] = None) -> "MilvusClient":
        """
        单例模式实现。
        
        Args:
            config: Milvus 配置，仅在首次创建时生效
            
        Returns:
            MilvusClient: 客户端单例实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
    
    def __init__(self, config: Optional[MilvusConfig] = None) -> None:
        """
        初始化 Milvus 客户端。
        
        Args:
            config: Milvus 配置，为 None 时从环境变量加载
        """
        if self._initialized:
            return
            
        self._config = config or load_milvus_config()
        self._alias = "default"
        self._connected = False
        self._initialized = True
        
    @property
    def config(self) -> MilvusConfig:
        """获取当前配置"""
        return self._config
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
    
    def _check_pymilvus(self) -> None:
        """
        检查 pymilvus 是否可用。
        
        Raises:
            MilvusConnectionError: pymilvus 未安装
        """
        if not _pymilvus_available:
            raise MilvusConnectionError(
                "pymilvus 未安装，请执行: pip install pymilvus>=2.5.0",
                {"package": "pymilvus", "min_version": "2.5.0"}
            )
    
    def connect(self) -> None:
        """
        建立 Milvus 连接。
        
        如果已连接则跳过。连接失败时抛出 MilvusConnectionError。
        
        Raises:
            MilvusConnectionError: 连接失败
        """
        if self._connected:
            logger.debug("Milvus 已连接，跳过重复连接")
            return
            
        self._check_pymilvus()
        
        try:
            connect_params = {
                "alias": self._alias,
                "host": self._config.host,
                "port": self._config.port,
                "db_name": self._config.database,
                "timeout": self._config.timeout,
            }
            
            # 添加认证信息（如果配置了）
            if self._config.user and self._config.password:
                connect_params["user"] = self._config.user
                connect_params["password"] = self._config.password
                
            # 添加 TLS 配置
            if self._config.secure:
                connect_params["secure"] = True
                
            connections.connect(**connect_params)
            self._connected = True
            
            logger.info(
                "Milvus 连接成功: {}:{}/{}",
                self._config.host,
                self._config.port,
                self._config.database
            )
            
        except Exception as e:
            raise MilvusConnectionError(
                f"Milvus 连接失败: {str(e)}",
                {
                    "host": self._config.host,
                    "port": self._config.port,
                    "database": self._config.database,
                    "error": str(e)
                }
            )
    
    def disconnect(self) -> None:
        """
        断开 Milvus 连接。
        
        如果未连接则跳过。
        """
        if not self._connected:
            return
            
        try:
            connections.disconnect(self._alias)
            self._connected = False
            logger.info("Milvus 连接已断开")
        except Exception as e:
            logger.warning("Milvus 断开连接时出错: {}", str(e))
    
    def is_healthy(self) -> bool:
        """
        检查 Milvus 服务健康状态。
        
        Returns:
            bool: 服务是否健康
        """
        if not self._connected:
            return False
            
        try:
            # 使用 utility.get_server_version() 作为健康检查
            version = utility.get_server_version()
            logger.debug("Milvus 服务版本: {}", version)
            return True
        except Exception as e:
            logger.warning("Milvus 健康检查失败: {}", str(e))
            return False
    
    def get_server_version(self) -> str:
        """
        获取 Milvus 服务版本。
        
        Returns:
            str: 服务版本号
            
        Raises:
            MilvusConnectionError: 未连接或获取失败
        """
        if not self._connected:
            raise MilvusConnectionError("未连接到 Milvus")
            
        try:
            return utility.get_server_version()
        except Exception as e:
            raise MilvusConnectionError(f"获取服务版本失败: {str(e)}")
    
    def __enter__(self) -> "MilvusClient":
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.disconnect()
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        重置单例实例（仅用于测试）。
        
        Warning:
            此方法仅应在测试环境中使用。
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.disconnect()
                cls._instance = None
