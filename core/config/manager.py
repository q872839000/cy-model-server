"""
配置管理器

本模块提供的配置管理方案，特点：
- 元类单例模式确保全局唯一实例
- 属性代理模式支持直接访问 (Config.app, Config.milvus)
- 延迟加载和智能缓存机制
- 类型安全的配置访问
- 统一的配置段映射，消除硬编码
"""

from typing import Optional, ClassVar, Dict, Any, Union
from threading import Lock
from loguru import logger

from .schemas import (
    AppSettings,
    KBChatSettings,
    MilvusConfig,
    KBCollectionConfig,
    RetrieverConfig,
    ModelsConfig,
)
from .loaders import (
    load_app_settings,
    load_kb_chat_settings,
    load_milvus_config,
    load_kb_collection_config,
    load_retriever_config,
    load_models_config,
    load_yaml,
)


class ConfigManagerMeta(type):
    """
    配置管理器元类 - 实现单例模式和属性代理
    """
    _instances: Dict[str, 'ConfigManager'] = {}
    _lock: Lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls.__name__ not in cls._instances:
            with cls._lock:
                if cls.__name__ not in cls._instances:
                    cls._instances[cls.__name__] = super().__call__(*args, **kwargs)
        return cls._instances[cls.__name__]


class ConfigManager(metaclass=ConfigManagerMeta):
    """
    配置管理器 - 元类单例 + 属性代理模式
    
    特性：
    1. 元类单例模式，确保全局唯一实例
    2. 属性代理模式，支持直接属性访问 (Config.app, Config.milvus)
    3. 延迟加载和缓存机制
    4. 类型安全的配置访问
    5. 统一的配置段映射，消除硬编码
    
    Usage:
        Config.app.host          # 直接访问应用配置
        Config.milvus.host       # 直接访问Milvus配置  
        Config.kb_chat.search_top_k  # 直接访问知识库对话配置
    """
    
    # 配置段映射：属性名 -> (配置类, 加载器函数, YAML段名)
    _CONFIG_MAPPING = {
        'app': (AppSettings, load_app_settings, 'app'),
        'kb_chat': (KBChatSettings, load_kb_chat_settings, 'kb_chat'), 
        'milvus': (MilvusConfig, load_milvus_config, 'milvus'),
        'kb_collection': (KBCollectionConfig, load_kb_collection_config, 'knowledge_base'),
        'retriever': (RetrieverConfig, load_retriever_config, 'retriever'),
        'models': (ModelsConfig, load_models_config, 'models'),
    }
    
    _initialized: bool = False
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化配置管理器"""
        if self._initialized:
            return
        
        # 配置文件路径
        self._config_path = config_path
        
        # 配置实例缓存 - 使用字典统一管理
        self._config_cache: Dict[str, Any] = {}
        
        # YAML原始数据缓存
        self._yaml_cache: Optional[Dict[str, Any]] = None
        
        self._initialized = True
        logger.debug(f"配置管理器已创建: {config_path}")
    
    def __getattr__(self, name: str) -> Any:
        """
        属性代理 - 支持 Config.app, Config.milvus 等直接访问
        
        Args:
            name: 配置属性名
            
        Returns:
            配置实例
            
        Raises:
            AttributeError: 未知的配置属性
        """
        if name in self._CONFIG_MAPPING:
            return self._get_config(name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def _get_config(self, config_name: str) -> Any:
        """
        获取配置实例（带缓存）- 使用统一YAML缓存避免重复文件读取
        
        Args:
            config_name: 配置名称
            
        Returns:
            配置实例
        """
        if config_name not in self._config_cache:
            config_class, loader_func, yaml_section = self._CONFIG_MAPPING[config_name]
            
            # 确保YAML数据已加载
            if self._yaml_cache is None:
                self._yaml_cache = load_yaml(self._config_path) or {}
            
            # 调用loader函数，传入预加载的YAML数据
            self._config_cache[config_name] = loader_func(self._config_path, self._yaml_cache)
            logger.debug(f"配置已加载: {config_name} -> {config_class.__name__}")
        
        return self._config_cache[config_name]
    
    def get_yaml_section(self, section: str) -> Optional[Dict[str, Any]]:
        """
        获取YAML配置文件的指定段落
        
        Args:
            section: 配置段名
            
        Returns:
            配置段内容，不存在时返回None
        """
        if self._yaml_cache is None:
            self._yaml_cache = load_yaml(self._config_path) or {}
        
        return self._yaml_cache.get(section)
    
    def reload_configs(self) -> None:
        """
        重新加载所有配置（清空缓存）
        """
        self._config_cache.clear()
        self._yaml_cache = None
        logger.info("配置缓存已清空，下次访问时重新加载")
    
    def preload_all(self) -> None:
        """
        预加载所有配置（用于启动时验证）
        """
        logger.debug("开始预加载所有配置...")
        
        for config_name in self._CONFIG_MAPPING.keys():
            self._get_config(config_name)
        
        logger.debug("所有配置预加载完成")
    
    def reload_config(self, config_path: Optional[str] = None) -> None:
        """
        重新加载配置（用于配置热更新）
        
        Args:
            config_path: 新的配置文件路径，为 None 时使用当前路径
        """
        if config_path is not None:
            self._config_path = config_path
        
        logger.info("开始重新加载配置: {}", self._config_path)
        self.reload_configs()
        logger.info("配置重新加载完成")


# 全局配置实例
# 直接通过 Config.app, Config.milvus 等属性访问
Config = ConfigManager()


def init_config(config_path: str = "configs/config.yaml") -> ConfigManager:
    """
    初始化配置系统

    由于 ConfigManager 使用元类单例，直接操作全局实例即可。
    更新配置路径、清空缓存并预加载所有配置以验证正确性。

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigManager: 配置管理器实例
    """
    Config.reload_config(config_path)
    Config.preload_all()
    logger.info("配置系统初始化完成: {}", config_path)
    return Config
