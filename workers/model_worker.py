"""
模型工作器：提供异步模型推理接口

重构后的职责：
1. 对话生成委托给 ChatService（业务逻辑）
2. Embedding/Reranker 使用 AsyncWorker（通用执行）
3. 保持原有 API 兼容性

设计原则：
- Worker 层不感知 Strategy
- Worker 层不处理业务参数（如 enable_thinking）
- 业务逻辑集中在 Service 层
"""

from typing import Dict, Any, Optional, List, Union, AsyncIterator, Coroutine
from loguru import logger

from core.registry import REGISTRY
from core.exceptions import ModelNotFoundError, InferenceError
from core.services.chat_service import CHAT_SERVICE
from workers.async_worker import ASYNC_WORKER


class ModelWorker:
    """
    模型工作器：提供异步模型推理接口
    
    对话生成委托给 ChatService，
    Embedding/Reranker 使用 AsyncWorker 执行。
    """

    def generate_chat(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs
    ) -> Union[Coroutine[Any, Any, str], AsyncIterator[str]]:
        """
        对话生成（统一接口，符合 OpenAI 范式）
        
        Args:
            model_name: 模型名称
            messages: 消息列表
            stream: 是否流式输出
            **kwargs: 其他生成参数
        
        Returns:
            stream=False: 返回协程，await 后得到完整文本
            stream=True: 返回异步迭代器，可直接 async for 迭代
        """
        if stream:
            return CHAT_SERVICE.generate_stream(
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
        return CHAT_SERVICE.generate(
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
    
    async def generate_embeddings(
        self, 
        model_name: str, 
        texts: List[str]
    ) -> List[List[float]]:
        """
        异步生成嵌入向量
        
        Args:
            model_name: 模型名称
            texts: 输入文本列表
            
        Returns:
            嵌入向量列表
        """
        engine = REGISTRY.get_embedding(model_name)
        if not engine:
            raise ModelNotFoundError(model_name)
        
        # 在线程池中执行同步的embed调用
        try:
            result = await ASYNC_WORKER.run(engine.embed, texts)
            return result
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            raise InferenceError(model_name, str(e))
    
    async def rerank_documents(
        self, 
        model_name: str, 
        query: str, 
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[float]:
        """
        异步文档重排序
        
        Args:
            model_name: 模型名称
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k个结果
            
        Returns:
            重排序分数列表
        """
        engine = REGISTRY.get_reranker(model_name)
        if not engine:
            raise ModelNotFoundError(model_name)
        
        # 在线程池中执行同步的rerank调用
        try:
            result = await ASYNC_WORKER.run(engine.rerank, query, documents, top_k)
            return result
        except Exception as e:
            logger.error(f"文档重排序失败: {e}")
            raise InferenceError(model_name, str(e))
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "llm_models": REGISTRY.list_llm_names(),
            "embedding_models": REGISTRY.list_embedding_names(),
            "reranker_models": REGISTRY.list_reranker_names(),
            "default_llm": REGISTRY.get_default_llm_name(),
            "default_embedding": REGISTRY.get_default_embedding_name(),
            "default_reranker": REGISTRY.get_default_reranker_name(),
        }
    
    def shutdown(self):
        """关闭工作器"""
        ASYNC_WORKER.shutdown(wait=True)
        logger.info("模型工作器已关闭")


# 全局工作器实例
WORKER = ModelWorker()
