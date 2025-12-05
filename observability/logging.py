from typing import Optional
from loguru import logger
import sys
import os
from pathlib import Path


LEVEL_MAP = {
	"TRACE": "TRACE",
	"DEBUG": "DEBUG",
	"INFO": "INFO",
	"WARNING": "WARNING",
	"ERROR": "ERROR",
	"CRITICAL": "CRITICAL",
}


def setup_logging(level: str = "INFO", log_format: Optional[str] = None) -> None:
	"""
	根据配置初始化日志：
	- level: 日志级别（TRACE/DEBUG/INFO/WARNING/ERROR/CRITICAL）
	- log_format: 可选的自定义格式；为空则使用默认格式
	"""
	fmt = log_format or "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {process} | {name}:{function}:{line} - {message}"
	resolved_level = LEVEL_MAP.get(level.upper(), "INFO")
	
	# 移除默认处理器
	logger.remove()
	
	# 控制台输出
	logger.add(
		sys.stdout, 
		level=resolved_level, 
		enqueue=True, 
		backtrace=True, 
		diagnose=True, 
		format=fmt,
		colorize=True
	)
	
	# 生产环境文件输出
	if os.getenv('ENV', 'dev') == 'prod':
		# 创建日志目录
		log_dir = Path("logs")
		log_dir.mkdir(exist_ok=True)
		
		# 应用日志
		logger.add(
			log_dir / "app.log",
			format=fmt,
			level=resolved_level,
			rotation="100 MB",
			retention="30 days",
			compression="zip",
			enqueue=True,
		)
		
		# 错误日志
		logger.add(
			log_dir / "error.log",
			format=fmt,
			level="ERROR",
			rotation="50 MB",
			retention="90 days",
			compression="zip",
			enqueue=True,
		)
