from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, Any, Dict, List
from pathlib import Path
import os
import yaml


class AppSettings(BaseSettings):
	"""
	应用级通用配置（环境变量前缀：UMS_）：
	- env: 运行环境标识（dev/test/prod）
	- host/port: 服务监听地址与端口
	- log_level/log_format: 日志级别与格式
	- models_config_path: 模型配置文件（YAML）路径，严禁在代码中硬编码模型参数
	"""
	env: str = Field(default="dev")
	host: str = Field(default="0.0.0.0")
	port: int = Field(default=8000)
	log_level: str = Field(default="INFO")
	log_format: Optional[str] = Field(default=None)
	models_config_path: str = Field(default="configs/config.yaml")

	model_config = SettingsConfigDict(env_prefix="UMS_")  # 例如 UMS_MODELS_CONFIG_PATH 覆盖 models_config_path


class KBChatSettings(BaseSettings):
	"""
	知识库对话配置（环境变量前缀：KB_CHAT_）：
	- 意图识别相关配置
	- Query 改写相关配置
	- 检索参数配置
	- Prompt 组装配置
	- 闲聊关键词配置
	"""
	# 意图识别
	intent_enabled: bool = Field(default=True, description="是否启用意图识别")
	intent_use_llm: bool = Field(default=False, description="是否使用LLM识别意图")
	
	# Query 改写
	query_rewrite_mode: str = Field(default="auto", description="改写模式: auto/always/never")
	rewrite_max_history_turns: int = Field(default=2, description="改写时参考的历史轮数")
	rewrite_timeout_ms: int = Field(default=10000, description="改写超时(毫秒)")
	
	# 检索参数
	search_top_k: int = Field(default=5, description="检索数量")
	search_rerank: bool = Field(default=True, description="是否重排序")
	search_score_threshold: float = Field(default=0.3, description="分数阈值")
	search_mode: str = Field(default="hybrid", description="检索模式")
	
	# Prompt 组装
	prompt_max_history_turns: int = Field(default=3, description="传给LLM的历史轮数")
	prompt_max_context_chars: int = Field(default=6000, description="检索内容最大字符")
	
	# 闲聊关键词
	chitchat_keywords: List[str] = Field(
		default_factory=lambda: [
			"好的", "谢谢", "明白了", "知道了", "了解", "OK", "ok",
			"嗯", "哦", "行", "可以", "没问题", "感谢", "辛苦",
			"太棒了", "很好", "不错", "懂了", "收到",
		],
		description="闲聊关键词"
	)
	
	model_config = SettingsConfigDict(env_prefix="KB_CHAT_")  # 例如 KB_CHAT_SEARCH_TOP_K 覆盖 search_top_k


def load_settings() -> AppSettings:
	"""
	读取应用配置：优先级 环境变量 > YAML(app段) > 默认值。
	- 先读取 BaseSettings（自动应用环境变量）
	- 再读取 models_config_path 指向的 YAML，若存在 app 段则填充未被环境变量覆盖的字段
	"""
	settings = AppSettings()
	cfg = load_yaml(settings.models_config_path) or {}
	app_cfg = cfg.get("app") or {}
	# 若环境变量未覆盖，则从 YAML app 段提取到 settings
	mapping = {
		"host": "UMS_HOST",
		"port": "UMS_PORT",
		"log_level": "UMS_LOG_LEVEL",
		"log_format": "UMS_LOG_FORMAT",
	}
	for key, env_name in mapping.items():
		if os.getenv(env_name) is None and key in app_cfg and app_cfg[key] is not None:
			setattr(settings, key, app_cfg[key])
	return settings


def load_kb_chat_settings() -> KBChatSettings:
	"""
	读取知识库对话配置：优先级 环境变量 > YAML(kb_chat段) > 默认值。
	- 先读取 BaseSettings（自动应用环境变量）
	- 再读取 YAML 文件中的 kb_chat 段，填充未被环境变量覆盖的字段
	"""
	app_settings = AppSettings()  # 获取配置文件路径
	settings = KBChatSettings()
	cfg = load_yaml(app_settings.models_config_path) or {}
	kb_chat_cfg = cfg.get("kb_chat") or {}
	
	# 若环境变量未覆盖，则从 YAML kb_chat 段提取到 settings
	mapping = {
		"intent_enabled": "KB_CHAT_INTENT_ENABLED",
		"intent_use_llm": "KB_CHAT_INTENT_USE_LLM",
		"query_rewrite_mode": "KB_CHAT_QUERY_REWRITE_MODE",
		"rewrite_max_history_turns": "KB_CHAT_REWRITE_MAX_HISTORY_TURNS",
		"rewrite_timeout_ms": "KB_CHAT_REWRITE_TIMEOUT_MS",
		"search_top_k": "KB_CHAT_SEARCH_TOP_K",
		"search_rerank": "KB_CHAT_SEARCH_RERANK",
		"search_score_threshold": "KB_CHAT_SEARCH_SCORE_THRESHOLD",
		"search_mode": "KB_CHAT_SEARCH_MODE",
		"prompt_max_history_turns": "KB_CHAT_PROMPT_MAX_HISTORY_TURNS",
		"prompt_max_context_chars": "KB_CHAT_PROMPT_MAX_CONTEXT_CHARS",
		"chitchat_keywords": "KB_CHAT_CHITCHAT_KEYWORDS",
	}
	
	for key, env_name in mapping.items():
		if os.getenv(env_name) is None and key in kb_chat_cfg and kb_chat_cfg[key] is not None:
			setattr(settings, key, kb_chat_cfg[key])
	
	return settings


class KBChatConfigManager:
	"""
	知识库对话配置管理器。
	
	实现配置的集中管理和依赖注入，避免在各个组件中重复创建配置实例。
	支持：
	- 单例模式：全局唯一的配置实例
	- 懒加载：首次访问时才加载配置
	- 配置覆盖：支持测试时注入自定义配置
	- 多种配置源：环境变量 > YAML文件 > 默认值
	"""
	_instance: Optional['KBChatConfigManager'] = None
	_settings: Optional[KBChatSettings] = None
	
	def __new__(cls) -> 'KBChatConfigManager':
		"""单例模式：确保全局只有一个配置管理器实例"""
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance
	
	@classmethod
	def get_config(cls) -> KBChatSettings:
		"""
		获取知识库对话配置实例。
		
		采用懒加载模式，首次调用时从配置源加载。
		后续调用直接返回缓存的配置实例。
		
		Returns:
			KBChatSettings: 知识库对话配置实例
		"""
		if cls._settings is None:
			cls._settings = load_kb_chat_settings()
		return cls._settings
	
	@classmethod
	def set_config(cls, settings: KBChatSettings) -> None:
		"""
		设置知识库对话配置实例。
		
		主要用于测试场景，允许注入自定义配置覆盖默认配置。
		注意：此操作会影响全局配置，请谨慎使用。
		
		Args:
			settings: 自定义的知识库对话配置实例
		"""
		cls._settings = settings
	
	@classmethod
	def reset_config(cls) -> None:
		"""
		重置配置缓存。
		
		清除缓存的配置实例，下次调用 get_config() 时会重新加载配置。
		主要用于测试场景或配置热更新。
		"""
		cls._settings = None


def load_yaml(path: str) -> Optional[Dict[str, Any]]:
	"""
	读取 YAML 文件内容为 dict；若文件不存在或为空，返回 None。
	"""
	p = Path(path)
	if not p.exists():
		return None
	with p.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
		return data if isinstance(data, dict) else None
