from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Any, Dict
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

	class Config:
		env_prefix = "UMS_"  # 例如 UMS_MODELS_CONFIG_PATH 覆盖 models_config_path


def load_settings() -> AppSettings:
	"""
	读取配置：优先级 环境变量 > YAML(app段) > 默认值。
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
