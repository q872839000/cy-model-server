from contextlib import asynccontextmanager

from api.openai_router import router as openai_router
from api.rag_router import router as rag_router
from api.kb_chat_router import router as kb_chat_router
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
import orjson
import time

from core.config import AppSettings, Config, init_config
from core.registry import REGISTRY
from core.exceptions import ModelServerException
from core.logging import setup_logging


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
	# Startup: 初始化配置系统
	init_config()
	settings = Config.app  # 直接属性访问应用配置
	try:
		REGISTRY.load_from_config()
		logger.info("模型注册表已加载（LLM:{} Emb:{} Rerank:{})",
				REGISTRY.llm_count(), REGISTRY.embedding_count(), REGISTRY.reranker_count())
	except Exception as e:
		logger.error("加载模型配置失败: {}", e)

	# Startup: 初始化 Milvus 连接
	try:
		from storage.milvus import init_milvus, shutdown_milvus
		milvus_available = init_milvus()
		if not milvus_available:
			logger.info("Milvus 服务不可用，RAG功能将受限")
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
	if settings is None:
		# 如果未传入设置，直接从全局Config获取
		settings = Config.app
	setup_logging(level=settings.log_level, log_format=settings.log_format)

	app = FastAPI(
		title="CY Model Server",
		version="1.0.0",
		description="大模型对话服务",
		default_response_class=ORJSONResponse,
		docs_url="/docs" if settings.env == "dev" else None,
		redoc_url="/redoc" if settings.env == "dev" else None,
		lifespan=lifespan,
	)

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
	"""设置中间件"""

	# CORS中间件
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"] if settings.env == "dev" else [],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
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

	@app.exception_handler(RequestValidationError)
	async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
		try:
			raw_body = await request.body()
			if raw_body:
				logger.warning(
					"422 validation error on {} {}: body={} errors={}",
					request.method,
					request.url.path,
					raw_body.decode("utf-8", errors="replace"),
					exc.errors(),
				)
			else:
				logger.warning(
					"422 validation error on {} {}: empty body errors={}",
					request.method,
					request.url.path,
					exc.errors(),
				)
		except Exception as log_err:
			logger.warning(
				"422 validation error on {} {} (failed to read body: {}) errors={}",
				request.method,
				request.url.path,
				str(log_err),
				exc.errors(),
			)

		# OpenAI 兼容错误格式：拼接可读的验证错误信息
		error_messages = []
		for err in exc.errors():
			loc = " -> ".join(str(l) for l in err.get("loc", []))
			msg = err.get("msg", "")
			error_messages.append(f"{loc}: {msg}" if loc else msg)
		return ORJSONResponse(
			{
				"error": {
					"message": "; ".join(error_messages) or "Invalid request parameters",
					"type": "invalid_request_error",
					"param": None,
					"code": None,
				}
			},
			status_code=422,
		)

	@app.exception_handler(ModelServerException)
	async def model_server_exception_handler(request: Request, exc: ModelServerException):
		logger.error("Model server error: {} - {}", exc.error_code, exc.message)
		# 根据错误类型推断 HTTP 状态码（与 OpenAI 错误规范对齐）
		status_map = {
			"MODEL_NOT_FOUND": 404,
			"CONFIGURATION_ERROR": 400,
			"UNSUPPORTED_PARAMETER": 400,
			"RESOURCE_LIMIT_ERROR": 429,
			"INFERENCE_ERROR": 500,
			"MODEL_LOAD_ERROR": 503,
		}
		status_code = status_map.get(exc.error_code, 500)
		# OpenAI 错误类型：4xx 为 invalid_request_error，5xx 为 server_error
		error_type = "invalid_request_error" if status_code < 500 else "server_error"
		# 额外提取 param 信息（UnsupportedParameterError 携带）
		param = exc.details.get("param") if exc.details else None
		return ORJSONResponse(
			{
				"error": {
					"message": exc.message,
					"type": error_type,
					"param": param,
					"code": exc.error_code,
				}
			},
			status_code=status_code,
		)

	@app.exception_handler(HTTPException)
	async def http_exception_handler(request: Request, exc: HTTPException):
		logger.warning("HTTP error: {} - {}", exc.status_code, exc.detail)
		# 根据状态码推断 OpenAI 规范的 error type
		if exc.status_code == 401:
			error_type = "authentication_error"
		elif exc.status_code == 403:
			error_type = "permission_error"
		elif exc.status_code == 404:
			error_type = "not_found_error"
		elif exc.status_code == 429:
			error_type = "rate_limit_error"
		elif exc.status_code >= 500:
			error_type = "server_error"
		else:
			error_type = "invalid_request_error"
		return ORJSONResponse(
			{
				"error": {
					"message": exc.detail,
					"type": error_type,
					"param": None,
					"code": None,
				}
			},
			status_code=exc.status_code,
		)

	@app.exception_handler(Exception)
	async def unhandled_exception_handler(request: Request, exc: Exception):
		logger.exception("Unhandled error: {} {}", request.method, request.url)
		return ORJSONResponse(
			{
				"error": {
					"message": "Internal server error",
					"type": "server_error",
					"param": None,
					"code": None,
				}
			},
			status_code=500,
		)


# ASGI 应用对象，供 Uvicorn/Gunicorn 加载
app = create_app()

# 注册路由
app.include_router(openai_router)
app.include_router(rag_router)
app.include_router(kb_chat_router)



if __name__ == "__main__":
	settings = Config.app
	import uvicorn
	# 直接传递 app 对象而非字符串，兼容 Nuitka 编译
	uvicorn.run(app, host=settings.host, port=settings.port)
