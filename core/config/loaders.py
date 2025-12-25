"""
配置加载器

本模块提供配置加载的工具函数，包括：
- YAML 文件读取
- 环境变量与 YAML 配置合并
- 配置验证和转换
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from .schemas import (
    AppSettings,
    KBChatSettings,
    MilvusConfig,
    KBCollectionConfig,
    RetrieverConfig,
    ModelsConfig,
)


def load_yaml(path: str) -> Optional[Dict[str, Any]]:
	"""
	读取 YAML 文件内容为 dict；若文件不存在或为空，返回 None。
	
	Args:
		path: YAML 文件路径
		
	Returns:
		Dict[str, Any] | None: 配置字典或 None
	"""
	p = Path(path)
	if not p.exists():
		logger.warning("配置文件不存在: {}", path)
		return None
	
	try:
		with p.open("r", encoding="utf-8") as f:
			data = yaml.safe_load(f)
			return data if isinstance(data, dict) else None
	except Exception as e:
		logger.error("读取配置文件失败: {} -> {}", path, e)
		return None


def load_app_settings(config_path: str, yaml_data: Optional[Dict[str, Any]] = None) -> AppSettings:
	"""
	加载应用配置：优先级 环境变量 > YAML(app段) > 默认值
	
	Args:
		config_path: YAML 配置文件路径
		yaml_data: 预加载的YAML数据（可选，避免重复读取文件）
		
	Returns:
		AppSettings: 应用配置实例
	"""
	# 使用预加载的YAML数据或读取文件
	cfg = yaml_data if yaml_data is not None else (load_yaml(config_path) or {})
	app_cfg = cfg.get("app") or {}
	
	# 使用统一的 from_dict 方式，同时设置 models_config_path
	app_cfg["models_config_path"] = config_path
	
	# AppSettings 继承了 BaseSettings，会自动应用环境变量
	# 再使用 from_dict 填充 YAML 配置
	return AppSettings.from_dict(app_cfg)


def load_kb_chat_settings(config_path: str, yaml_data: Optional[Dict[str, Any]] = None) -> KBChatSettings:
	"""
	加载知识库对话配置：优先级 环境变量 > YAML(kb_chat段) > 默认值
	
	Args:
		config_path: YAML 配置文件路径
		yaml_data: 预加载的YAML数据（可选，避免重复读取文件）
		
	Returns:
		KBChatSettings: 知识库对话配置实例
	"""
	# 使用预加载的YAML数据或读取文件
	cfg = yaml_data if yaml_data is not None else (load_yaml(config_path) or {})
	kb_chat_cfg = cfg.get("kb_chat") or {}
	
	# KBChatSettings 继承了 BaseSettings，会自动应用环境变量
	# 再使用 from_dict 填充 YAML 配置
	return KBChatSettings.from_dict(kb_chat_cfg)


def load_milvus_config(config_path: str, yaml_data: Optional[Dict[str, Any]] = None) -> MilvusConfig:
	"""
	加载 Milvus 配置：优先级 环境变量 > YAML(milvus段) > 默认值
	
	Args:
		config_path: YAML 配置文件路径
		yaml_data: 预加载的YAML数据（可选，避免重复读取文件）
		
	Returns:
		MilvusConfig: Milvus 连接配置实例
	"""
	cfg = yaml_data if yaml_data is not None else (load_yaml(config_path) or {})
	milvus_cfg = cfg.get("milvus") or {}
	
	# MilvusConfig 继承了 BaseSettings，会自动应用环境变量
	# 再使用 from_dict 填充 YAML 配置
	return MilvusConfig.from_dict(milvus_cfg)


def load_kb_collection_config(config_path: str, yaml_data: Optional[Dict[str, Any]] = None) -> KBCollectionConfig:
	"""
	加载知识库 Collection 配置：优先级 环境变量 > YAML(knowledge_base段) > 默认值
	
	Args:
		config_path: YAML 配置文件路径
		yaml_data: 预加载的YAML数据（可选，避免重复读取文件）
		
	Returns:
		KBCollectionConfig: Collection 配置实例
	"""
	cfg = yaml_data if yaml_data is not None else (load_yaml(config_path) or {})
	kb_cfg = cfg.get("knowledge_base") or {}
	
	# KBCollectionConfig 继承了 BaseSettings，会自动应用环境变量
	# 再使用 from_dict 填充 YAML 配置
	return KBCollectionConfig.from_dict(kb_cfg)


def load_retriever_config(config_path: str, yaml_data: Optional[Dict[str, Any]] = None) -> RetrieverConfig:
	"""
	加载检索器配置：优先级 环境变量 > YAML(retriever段) > 默认值
	
	Args:
		config_path: YAML 配置文件路径
		yaml_data: 预加载的YAML数据（可选，避免重复读取文件）
		
	Returns:
		RetrieverConfig: 检索器配置实例
	"""
	cfg = yaml_data if yaml_data is not None else (load_yaml(config_path) or {})
	retriever_cfg = cfg.get("retriever") or {}
	
	# RetrieverConfig 继承了 BaseSettings，会自动应用环境变量
	# 再使用 from_dict 填充 YAML 配置
	return RetrieverConfig.from_dict(retriever_cfg)


def load_models_config(config_path: str, yaml_data: Optional[Dict[str, Any]] = None) -> ModelsConfig:
	"""
	加载模型配置：优先级 环境变量 > YAML(整个文件) > 默认值
	
	Args:
		config_path: YAML 配置文件路径
		yaml_data: 预加载的YAML数据（可选，避免重复读取文件）
		
	Returns:
		ModelsConfig: 模型配置实例
	"""
	cfg = yaml_data if yaml_data is not None else (load_yaml(config_path) or {})
	
	# ModelsConfig 继承了 BaseSettings，会自动应用环境变量
	# 再使用 from_dict 填充 YAML 配置
	return ModelsConfig.from_dict(cfg)
