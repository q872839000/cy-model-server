"""
统一配置管理模块

本模块提供项目的统一配置管理入口，包括：
- 全局配置管理器
- 配置结构体定义
- 配置加载器
"""

from .manager import (
    ConfigManager,
    Config,
    init_config,
)
from .schemas import (
    AppSettings,
    KBChatSettings,
    MilvusConfig,
    KBCollectionConfig,
    RetrieverConfig,
)

__all__ = [
    "Config",
    "ConfigManager", 
    "init_config",
    # 配置结构体
    "AppSettings",
    "KBChatSettings", 
    "MilvusConfig",
    "KBCollectionConfig",
    "RetrieverConfig",
]
