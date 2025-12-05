from api.openai_router import router as openai_router
from api.retrieval_router import router as retrieval_router
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


class ORJSONResponse(JSONResponse):
	"""
	自定义 ORJSON 响应类：
	- 使用 orjson 提升 JSON 序列化性能；
	- 设置默认的 numpy 序列化选项。
	"""
	media_type = "application/json"

	def render(self, content: object) -> bytes:
		return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)




def _init_registry(settings: AppSettings) -> None:
	"""
	按配置文件加载全局模型注册表：
	- 清空旧注册；
	- 从 YAML 路径读取引擎与模型配置；
	- 严禁在此处硬编码任何模型参数。
	"""
	REGISTRY.clear()
	REGISTRY.load_from_config(settings)
	logger.info("模型注册表已加载（LLM:{} Emb:{} Rerank:{}）",
				REGISTRY.has_any_llm(), REGISTRY.has_any_embedding(), REGISTRY.has_any_reranker())


def create_app(settings: AppSettings | None = None) -> FastAPI:
	"""
	应用工厂：创建并返回 FastAPI 实例。
	- 初始化日志（使用可配置级别与格式）
	- 读取环境配置（优先级：环境变量 > YAML(app段) > 默认值）
	- 尝试加载模型注册表
	- 暴露 Prometheus 指标
	- 挂载 API 路由（若存在）
	- 注册健康检查与全局异常处理
	"""
	settings = settings or load_settings()
	setup_logging(level=settings.log_level, log_format=settings.log_format)

	# 加载模型注册表（按配置）
	try:
		_init_registry(settings)
	except Exception as e:
		# 注册表加载失败不阻断服务启动（便于先起服务再排查配置），但会记录错误
		logger.error("加载模型配置失败: {}", e)

	app = FastAPI(
		title="CY Model Server",
		version="1.0.0",
		description="生产级大模型对话服务",
		default_response_class=ORJSONResponse,
		docs_url="/docs" if settings.env == "dev" else None,
		redoc_url="/redoc" if settings.env == "dev" else None,
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
				"llm_count": len(REGISTRY._llms),
				"embedding_count": len(REGISTRY._embeddings),
				"reranker_count": len(REGISTRY._rerankers),
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
app.include_router(retrieval_router)

# 注册高级检索路由
from api.advanced_retrieval_router import router as advanced_retrieval_router
app.include_router(advanced_retrieval_router)

# 注册RAG路由
from api.rag_router import router as rag_router
app.include_router(rag_router)


if __name__ == "__main__":
	"""
	开发环境启动入口
	生产环境请使用: uvicorn main:app --host 0.0.0.0 --port 8000
	"""
	settings = load_settings()
	from uvicorn import run
	run("main:app", host=settings.host, port=settings.port, reload=True)
