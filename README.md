# CY Model Server

一个基于 FastAPI 的大模型服务，提供 OpenAI 兼容接口、Anthropic Messages 兼容接口，以及知识库检索与 RAG 对话能力。适合本地部署大语言模型、嵌入模型和重排序模型，并通过统一 API 服务对外暴露。

## 项目简介

该项目主要解决以下问题：

- 统一封装大模型推理服务，支持多种模型加载方式
- 提供 OpenAI Chat Completions / Embeddings / Models 兼容接口
- 提供 Anthropic Messages API 兼容接口，方便 Claude Code 等客户端接入
- 支持知识库问答与混合检索（Dense + Sparse + Rerank）
- 支持本地模型部署、服务监控、健康检查和 Prometheus 指标

## 主要特性

- OpenAI API 兼容
  - `POST /v1/chat/completions`
  - `POST /v1/embeddings`
  - `GET /v1/models`
  - `GET /v1/models/{model_id}`
- Anthropic API 兼容
  - `POST /v1/messages`
- 知识库与 RAG
  - `POST /v1/kb/search`
  - `POST /v1/kb/chat/completions`
- 支持多种推理引擎
  - Transformers
  - vLLM（可选）
- 支持多模型配置
  - LLM
  - Embedding
  - Reranker
- 运维能力
  - `/healthz` 健康检查
  - `/metrics` Prometheus 指标
  - CORS、中间件日志、统一异常处理

## 技术栈

- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic / Pydantic Settings
- Transformers
- Sentence Transformers
- PyMilvus（RAG / Milvus 可选）
- Loguru
- Prometheus FastAPI Instrumentator

## 项目结构

```text
.
├── api/                     # API 路由层
│   ├── anthropic_router.py
│   ├── kb_chat_router.py
│   ├── openai_router.py
│   └── rag_router.py
├── configs/
│   └── config.yaml         # 模型与服务配置
├── core/                   # 核心配置、注册表和服务管理
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── engines/                # 推理引擎
├── models/                 # 数据模型及 schema
├── rag/                    # RAG / 知识库相关实现
├── scripts/                # 脚本工具
├── storage/                # 向量数据库/存储层
├── strategies/              # 对话模板与生成策略
├── tests/                  # 测试
├── utils/                  # 工具函数
├── workers/                # 异步工作线程与推理 worker
├── main.py                 # 应用入口
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始

### 1. 环境准备

建议使用 Python 3.10 及以上版本，并确保具备可用的 CUDA 环境（若使用 GPU 推理）。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置模型

编辑 `configs/config.yaml` 文件，配置你的 LLM、Embedding、Reranker 模型路径和参数。

示例：

```yaml
llms:
  - name: Qwen3-1___7B
    path: E:\Ai\models\Qwen\Qwen3-1___7B
    chat_strategy: qwen3
    enable_thinking: false
    engine: vllm
    max_concurrent: 32
    gen_params:
      max_tokens: 2048
```

如需启用 RAG，请确保：

- Milvus 服务已启动
- `configs/config.yaml` 中 `milvus` 与 `knowledge_base` 配置正确
- `embedding_model` 和 `reranker_model` 已正确声明

### 4. 启动服务

直接运行：

```bash
python main.py
```

或者：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动后可访问：

- 服务地址：`http://localhost:8000`
- OpenAPI 文档：`http://localhost:8000/docs`
- ReDoc 文档：`http://localhost:8000/redoc`
- 健康检查：`http://localhost:8000/healthz`
- 指标：`http://localhost:8000/metrics`

## Docker 部署

仓库已提供 Docker 相关配置。

### 构建并启动

```bash
docker compose -f docker/docker-compose.yml up --build
```

### 说明

- 镜像入口：`docker/Dockerfile`
- 端口：`8000`
- 配置挂载：`configs/`
- 模型目录挂载：`models/`

## 配置说明

核心配置文件：`configs/config.yaml`

常见配置项包括：

- `app.host` / `app.port`
- `app.log_level` / `app.log_format`
- `engine_defaults`
- `llms`
- `embeddings`
- `rerankers`
- `milvus`
- `knowledge_base`

注意：

- `engine_defaults.device` 默认值通常为 `cuda`
- `llm_engine` / `embedding_engine` / `reranker_engine` 可配置为 `transformers` 或 `vllm`
- 若不开启 RAG，Milvus 可视为可选依赖

## API 使用示例

### 1. 获取模型列表

```bash
curl http://localhost:8000/v1/models
```

### 2. 聊天补全（OpenAI 兼容）

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-1___7B",
    "messages": [
      {"role": "user", "content": "你好，请简单介绍一下这个项目。"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### 3. Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-large-zh",
    "input": "这是一个用于大模型服务的文本向量化请求。"
  }'
```

### 4. Anthropic Messages API

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-1___7B",
    "messages": [
      {"role": "user", "content": "请总结一下这个项目的能力。"}
    ],
    "max_tokens": 512
  }'
```

### 5. RAG 检索

```bash
curl -X POST http://localhost:8000/v1/kb/search \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "demo_collection",
    "query": "什么是大模型服务",
    "mode": "hybrid",
    "top_k": 5
  }'
```

## 注意事项

- 该服务主要面向本地部署或企业内部大模型场景，不依赖第三方云端推理服务。
- 如果使用 `vLLM`，请确保环境中已正确安装并配置 GPU 运行条件。
- 如未配置 embedding / reranker 模型，RAG 功能会受到限制，部分接口可能返回 503 或不可用。
- 生产环境建议：
  - 使用更稳定的模型路径和资源隔离
  - 配置 `workers` 和 `max_concurrent` 以控制并发压力
  - 设置严格的日志与监控策略

## 常见问题

### Q: 服务启动后接口 404？
A: 请确认是否已正确配置路由和模型，并检查 `main.py` 中是否已注册对应 router。

### Q: `v1/models` 里没有模型？
A: 检查 `configs/config.yaml` 中是否正确加载 `llms`、`embeddings`、`rerankers`，并确认模型路径有效。

### Q: RAG 相关接口不可用？
A: 确认 Milvus 服务正在运行，并检查 `milvus` 和 `knowledge_base` 配置。

## 许可证

本项目未在仓库中显式声明许可证文件，具体使用权限请以仓库中实际 License 信息为准。

## 贡献

欢迎提交 Issues、PR 和改进建议。若你希望扩展新的模型支持、添加更多 Router，或增强 RAG 能力，欢迎参与项目开发。

## 结语

CY Model Server 是一个面向大模型服务化部署的 Python 后端，适合做统一的模型推理网关，兼容主流 OpenAI/Anthropic 协议，并具备知识库问答能力。该项目适合用于本地模型托管、企业内部知识库问答、大模型统一接入等场景。
