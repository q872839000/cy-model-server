from contextlib import asynccontextmanager

from api.openai_router import router as openai_router
from api.rag_router import router as rag_router
from api.kb_chat_router import router as kb_chat_router
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
import orjson
import time

from core.config import AppSettings, load_settings
from core.registry import REGISTRY
from core.exceptions import ModelServerException
from observability.logging import setup_logging


def _print_startup_banner(settings: AppSettings) -> None:
	"""打印启动完成横幅"""
	host = settings.host
	port = settings.port
	llm = REGISTRY.llm_count()
	emb = REGISTRY.embedding_count()
	rerank = REGISTRY.reranker_count()
	
	print("\n" + "=" * 60)
	print("  ____ __   __  __  __  ___  ___  ___ _    ")
	print(" / ___|\\ \\ / / |  \\/  |/ _ \\|   \\| __| |   ")
	print("| |     \\ V /  | |\\/| | | | | |) | _|| |__ ")
	print("| |___   | |   | |  | | |_| |   /| __|____|")
	print(" \\____|  |_|   |_|  |_|\\___/|___/|___|_____|")
	print("=" * 60)
	print(f"  [OK] Server Ready!")
	print(f"  API:     http://{host}:{port}")
	print(f"  Docs:    http://{host}:{port}/docs")
	print(f"  Metrics: http://{host}:{port}/metrics")
	print(f"  Health:  http://{host}:{port}/healthz")
	print("-" * 60)
	print(f"  Models: LLM={llm} Embedding={emb} Reranker={rerank}")
	print("=" * 60 + "\n")


class ORJSONResponse(JSONResponse):
	"""
	自定义 ORJSON 响应类：
	- 使用 orjson 提升 JSON 序列化性能；
	- 设置默认的 numpy 序列化选项。
	"""
	media_type = "application/json"

	def render(self, content: object) -> bytes:
		return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""
	FastAPI 生命周期管理：
	- startup: 加载模型注册表、连接 Milvus
	- shutdown: 清理资源、断开连接
	"""
	# Startup: 加载模型
	settings = load_settings()
	try:
		REGISTRY.load_from_config(settings)
		logger.info("模型注册表已加载（LLM:{} Emb:{} Rerank:{})",
				REGISTRY.llm_count(), REGISTRY.embedding_count(), REGISTRY.reranker_count())
	except Exception as e:
		logger.error("加载模型配置失败: {}", e)
	
	# Startup: 初始化 Milvus 连接
	try:
		from storage.milvus import init_milvus, shutdown_milvus
		init_milvus()
	except ImportError:
		logger.info("Milvus 模块不可用，跳过初始化")
		shutdown_milvus = None
	
	# 启动完成标识
	_print_startup_banner(settings)
	
	yield  # 应用运行中
	
	# Shutdown: 清理资源
	logger.info("服务关闭，清理资源...")
	if shutdown_milvus:
		shutdown_milvus()
	REGISTRY.clear()


def create_app(settings: AppSettings | None = None) -> FastAPI:
	"""
	应用工厂：创建并返回 FastAPI 实例。
	- 初始化日志（使用可配置级别与格式）
	- 读取环境配置（优先级：环境变量 > YAML(app段) > 默认值）
	- 模型加载在 lifespan 中完成（只在工作进程启动时执行）
	- 暴露 Prometheus 指标
	- 挂载 API 路由（若存在）
	- 注册健康检查与全局异常处理
	"""
	settings = settings or load_settings()
	setup_logging(level=settings.log_level, log_format=settings.log_format)

	app = FastAPI(
		title="CY Model Server",
		version="1.0.0",
		description="生产级大模型对话服务",
		default_response_class=ORJSONResponse,
		docs_url="/docs" if settings.env == "dev" else None,
		redoc_url="/redoc" if settings.env == "dev" else None,
		lifespan=lifespan,
	)

	# 生产级中间件
	_setup_middleware(app, settings)

	# Prometheus 指标
	Instrumentator().instrument(app).expose(app, endpoint="/metrics")

	# 健康检查：用于存活/就绪探针
	@app.get("/healthz")
	async def healthz():
		return {
			"status": "ok",
			"timestamp": time.time(),
			"models": {
				"llm_count": REGISTRY.llm_count(),
				"embedding_count": REGISTRY.embedding_count(),
				"reranker_count": REGISTRY.reranker_count(),
			}
		}

	# 自定义异常处理
	_setup_exception_handlers(app)

	logger.info(
		"Service started. env={} host={} port={} models_config={}",
		settings.env,
		settings.host,
		settings.port,
		settings.models_config_path,
	)
	return app


def _setup_middleware(app: FastAPI, settings: AppSettings) -> None:
	"""设置生产级中间件"""
	
	# CORS中间件
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"] if settings.env == "dev" else [],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)
	
	# 信任主机中间件（生产环境）
	if settings.env == "prod":
		app.add_middleware(
			TrustedHostMiddleware,
			allowed_hosts=["*"]  # 根据实际需求配置
		)
	
	# 请求日志中间件
	@app.middleware("http")
	async def log_requests(request: Request, call_next):
		start_time = time.time()
		response = await call_next(request)
		process_time = time.time() - start_time
		
		logger.info(
			"{} {} {} {}ms",
			request.method,
			request.url.path,
			response.status_code,
			round(process_time * 1000, 2)
		)
		
		return response


def _setup_exception_handlers(app: FastAPI) -> None:
	"""设置异常处理器"""
	
	@app.exception_handler(ModelServerException)
	async def model_server_exception_handler(request: Request, exc: ModelServerException):
		logger.error("Model server error: {} - {}", exc.error_code, exc.message)
		return ORJSONResponse(
			{
				"error": {
					"code": exc.error_code,
					"message": exc.message,
					"details": exc.details
				}
			},
			status_code=400
		)
	
	@app.exception_handler(HTTPException)
	async def http_exception_handler(request: Request, exc: HTTPException):
		logger.warning("HTTP error: {} - {}", exc.status_code, exc.detail)
		return ORJSONResponse(
			{
				"error": {
					"code": "HTTP_ERROR",
					"message": exc.detail,
					"status_code": exc.status_code
				}
			},
			status_code=exc.status_code
		)
	
	@app.exception_handler(Exception)
	async def unhandled_exception_handler(request: Request, exc: Exception):
		logger.exception("Unhandled error: {} {}", request.method, request.url)
		return ORJSONResponse(
			{
				"error": {
					"code": "INTERNAL_ERROR",
					"message": "内部服务器错误"
				}
			},
			status_code=500
		)


# ASGI 应用对象，供 Uvicorn/Gunicorn 加载
app = create_app()

# 注册路由
app.include_router(openai_router)
app.include_router(rag_router)
app.include_router(kb_chat_router)



if __name__ == "__main__":
	"""
	开发环境启动入口
	生产环境请使用: uvicorn main:app --host 0.0.0.0 --port 8000
	"""
	settings = load_settings()
	import uvicorn
	# 直接传递 app 对象而非字符串，兼容 Nuitka 编译
	uvicorn.run(app, host=settings.host, port=settings.port)
