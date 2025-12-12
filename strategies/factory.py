"""
策略工厂模块

本模块提供策略工厂，根据 chat_strategy 配置值自动匹配最合适的策略实例。
支持精确匹配、系列匹配和通用回退。
"""

from typing import Dict, Optional, Type
from loguru import logger

from strategies.base import LLMStrategy, GenericChatStrategy

# Qwen 系列
from strategies.qwen import QwenBaseStrategy, Qwen2Strategy, Qwen3Strategy

# GLM 系列
from strategies.glm import GLMBaseStrategy, GLM4Strategy, GLM4_0414Strategy, GLM4Z1Strategy

# Deepseek 系列
from strategies.deepseek import DeepseekBaseStrategy, DeepseekR1Strategy


class StrategyFactory:
    """
    策略工厂：根据 chat_strategy 配置值创建对应的策略实例。
    
    匹配优先级：
    1. 精确匹配（如 qwen3 -> Qwen3Strategy）
    2. 系列匹配（如 qwen -> QwenBaseStrategy）
    3. 通用回退（generic -> GenericChatStrategy）
    """

    # 策略注册表：chat_strategy 值 -> 策略类
    _registry: Dict[str, Type[LLMStrategy]] = {
        # Qwen 系列（精确匹配）
        "qwen3": Qwen3Strategy,
        "qwen2": Qwen2Strategy,
        
        # Qwen 系列（系列匹配）
        "qwen": QwenBaseStrategy,
        
        # GLM 系列（精确匹配）
        "glm4-0414": GLM4_0414Strategy,
        "glm4-z1": GLM4Z1Strategy,
        "glm4": GLM4Strategy,
        
        # GLM 系列（系列匹配）
        "glm": GLMBaseStrategy,
        
        # Deepseek 系列（精确匹配）
        "deepseek-r1": DeepseekR1Strategy,
        
        # Deepseek 系列（系列匹配）
        "deepseek": DeepseekBaseStrategy,
        
        # 通用策略
        "generic": GenericChatStrategy,
    }

    # 策略实例缓存（单例模式）
    _instances: Dict[str, LLMStrategy] = {}

    @classmethod
    def register(cls, key: str, strategy_class: Type[LLMStrategy]) -> None:
        """
        注册新的策略类。
        
        Args:
            key: 策略标识符（用于配置文件中的 chat_strategy 字段）
            strategy_class: 策略类
        """
        cls._registry[key] = strategy_class
        logger.debug("注册策略: {} -> {}", key, strategy_class.__name__)

    @classmethod
    def get(cls, key: Optional[str]) -> LLMStrategy:
        """
        获取策略实例。
        
        Args:
            key: 策略标识符，为 None 时返回通用策略
            
        Returns:
            LLMStrategy: 策略实例
        """
        if key is None:
            key = "generic"
        
        # 检查缓存
        if key in cls._instances:
            return cls._instances[key]
        
        # 查找策略类
        strategy_class = cls._registry.get(key)
        
        if strategy_class is None:
            logger.warning("未找到策略 '{}', 回退到通用策略", key)
            strategy_class = GenericChatStrategy
        
        # 创建实例并缓存
        instance = strategy_class()
        cls._instances[key] = instance
        logger.info("加载策略: {} -> {}", key, strategy_class.__name__)
        
        return instance

    @classmethod
    def list_strategies(cls) -> Dict[str, str]:
        """
        列出所有已注册的策略。
        
        Returns:
            Dict[str, str]: 策略标识符 -> 策略类名
        """
        return {k: v.__name__ for k, v in cls._registry.items()}

    @classmethod
    def clear_cache(cls) -> None:
        """清空策略实例缓存"""
        cls._instances.clear()
