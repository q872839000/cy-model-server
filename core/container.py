"""
服务容器模块

本模块提供服务容器，统一管理各类业务服务的实例化和获取，包括：
- LLM 对话服务（含策略）
- Embedding 向量化服务
- Reranker 重排序服务
- RAG 知识库检索服务
"""

from typing import Dict, Optional, List, Any
from threading import Lock
from loguru import logger

from core.registry import REGISTRY
from strategies.base import LLMStrategy
from strategies.factory import StrategyFactory


class ServiceContainer:
    """
    服务容器：负责提供面向业务的引用（引擎与策略）。
    
    职责：
    1. 通过策略工厂获取 LLM 策略实例
    2. 提供 Embedding/Reranker 引擎获取
    3. 提供 RAG 服务（延迟初始化）
    """

    def __init__(self) -> None:
        """初始化服务容器"""
        self._rag_service: Optional["RAGService"] = None
        self._rag_initialized: bool = False
        self._kb_chat_service: Optional["KBChatService"] = None
        self._rag_lock = Lock()
        self._kb_chat_lock = Lock()

    def get_strategy(self, strategy_key: Optional[str]) -> LLMStrategy:
        """
        获取 LLM 策略实例。
        
        Args:
            strategy_key: 策略标识符
            
        Returns:
            LLMStrategy: 策略实例
        """
        return StrategyFactory.get(strategy_key)

    def get_llm_and_strategy(self, model_name: str | None):
        """
        获取 LLM 引擎和对应的策略。
        
        Args:
            model_name: 模型名称，为 None 时使用默认模型
            
        Returns:
            Tuple[LLMEngine, LLMStrategy]: 引擎和策略
        """
        engine = REGISTRY.get_llm(model_name)
        strategy_key = REGISTRY.get_llm_strategy_key(model_name) or "generic"
        strategy = StrategyFactory.get(strategy_key)
        return engine, strategy

    def get_embedding(self, model_name: str | None):
        """
        获取 Embedding 引擎。
        
        Args:
            model_name: 模型名称，为 None 时使用默认模型
            
        Returns:
            EmbeddingEngine: Embedding 引擎实例
        """
        return REGISTRY.get_embedding(model_name)

    def get_reranker(self, model_name: str | None):
        """
        获取 Reranker 引擎。
        
        Args:
            model_name: 模型名称，为 None 时使用默认模型
            
        Returns:
            RerankerEngine: Reranker 引擎实例
        """
        return REGISTRY.get_reranker(model_name)
    
    # ==================== RAG 服务 ====================
    
    def get_rag_service(self) -> Optional["RAGService"]:
        """
        获取 RAG 服务实例。

        首次调用时自动初始化。使用 storage.milvus 模块中已建立的 Milvus 连接。

        Returns:
            Optional[RAGService]: RAG 服务实例
        """
        if self._rag_service is not None:
            return self._rag_service

        with self._rag_lock:
            # 双重检查：获取锁后再次确认
            if self._rag_service is not None:
                return self._rag_service

            # 获取全局 Milvus 客户端（在服务启动时已连接）
            from storage.milvus import get_milvus_client
            milvus_client = get_milvus_client()

            if milvus_client is None or not milvus_client.is_connected:
                logger.warning("Milvus 未连接，RAG 服务不可用")
                return None

            try:
                # 从配置获取模型名称
                from core.config import Config
                kb_cfg = Config.get_yaml_section("knowledge_base") or {}

                embedding_model = kb_cfg.get("embedding_model")
                reranker_model = kb_cfg.get("reranker_model")

                # 延迟导入
                from rag.service import RAGService
                from storage.milvus.collections import KBCollectionManager

                # 获取 Embedding 引擎并包装为函数
                embedding_engine = self.get_embedding(embedding_model)

                def embedding_fn(text: str) -> List[float]:
                    """向量化函数包装（使用 embed 方法）"""
                    result = embedding_engine.embed([text])
                    return result[0] if isinstance(result[0], list) else list(result[0])

                # 获取 Reranker 引擎并包装为函数（可选）
                reranker_fn = None
                try:
                    reranker_engine = self.get_reranker(reranker_model)
                    if reranker_engine:
                        def reranker_fn(query: str, documents: List[str]) -> List[Dict[str, Any]]:
                            """重排序函数包装（使用 rerank 方法）"""
                            scores = reranker_engine.rerank(query, documents)
                            return [
                                {"index": i, "score": float(score)}
                                for i, score in enumerate(scores)
                            ]
                except Exception as e:
                    logger.warning("Reranker 不可用: {}", str(e))

                # 创建 Collection 管理器（使用已连接的客户端）
                collection_manager = KBCollectionManager(client=milvus_client)

                # 创建 RAG 服务（传入已连接的客户端和管理器）
                self._rag_service = RAGService(
                    embedding_fn=embedding_fn,
                    reranker_fn=reranker_fn,
                    milvus_client=milvus_client,
                    collection_manager=collection_manager,
                )

                # 初始化（不再需要连接 Milvus）
                self._rag_service.initialize()
                self._rag_initialized = True

                logger.info("RAG 服务初始化完成")
                return self._rag_service

            except Exception as e:
                logger.warning("RAG 服务初始化失败: {}", str(e))
                return None

    # ==================== 知识库对话服务 ====================
    
    def get_kb_chat_service(self) -> Optional["KBChatService"]:
        """
        获取知识库对话服务实例。

        首次调用时自动初始化。依赖 RAG 服务和 WORKER。

        Returns:
            Optional[KBChatService]: 知识库对话服务实例
        """
        if self._kb_chat_service is not None:
            return self._kb_chat_service

        with self._kb_chat_lock:
            # 双重检查：获取锁后再次确认
            if self._kb_chat_service is not None:
                return self._kb_chat_service

            # 获取 RAG 服务
            rag_service = self.get_rag_service()
            if rag_service is None:
                logger.warning("RAG 服务不可用，知识库对话服务无法初始化")
                return None

            try:
                from rag.kb_chat import KBChatService
                from workers.model_worker import WORKER

                # 创建 LLM 调用函数
                async def llm_fn(
                    model: str,
                    messages: List[dict],
                    max_tokens: int,
                    temperature: float,
                    top_p: float = 0.95,
                    enable_thinking: bool = True,
                    **kwargs
                ) -> str:
                    """非流式 LLM 调用函数"""
                    result = await WORKER.generate_chat(
                        model_name=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=False,
                        enable_thinking=enable_thinking,
                    )
                    # KB chat 只需要文本，从 StrategyOutput 中提取
                    return result.text

                async def llm_stream_fn(
                    model: str,
                    messages: List[dict],
                    max_tokens: int,
                    temperature: float,
                    top_p: float = 0.95,
                    enable_thinking: bool = True,
                    **kwargs
                ):
                    """流式 LLM 调用函数"""
                    async for chunk in WORKER.generate_chat(
                        model_name=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=True,
                        enable_thinking=enable_thinking,
                    ):
                        yield chunk

                # 创建服务
                self._kb_chat_service = KBChatService(
                    rag_service=rag_service,
                    llm_fn=llm_fn,
                    llm_stream_fn=llm_stream_fn,
                )

                logger.info("知识库对话服务初始化完成")
                return self._kb_chat_service

            except Exception as e:
                logger.warning("知识库对话服务初始化失败: {}", str(e))
                return None


# 全局服务容器单例
CONTAINER = ServiceContainer()
