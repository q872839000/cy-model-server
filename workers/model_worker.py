"""
模型工作器：负责模型的加载、管理和推理任务调度。
提供异步模型推理接口，支持并发请求处理和流式输出。
"""

import asyncio
import threading
from typing import Dict, Any, Optional, List, Union, Iterator, AsyncIterator, Coroutine
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from core.registry import REGISTRY
from core.exceptions import ModelNotFoundError, InferenceError
from services.container import CONTAINER


class ModelWorker:
    """
    模型工作器：管理模型推理任务
    - 支持异步推理
    - 线程池执行同步模型调用
    - 任务队列管理
    """

    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._task_counter = 0
        self._lock = threading.Lock()
        
    def _get_next_task_id(self) -> str:
        """获取下一个任务ID"""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter}"
    
    async def generate_text(
        self, 
        model_name: str, 
        prompt: str, 
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        异步文本生成
        
        Args:
            model_name: 模型名称
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本
        """
        engine = REGISTRY.get_llm(model_name)
        if not engine:
            raise ModelNotFoundError(model_name)
        
        # 在线程池中执行同步的generate调用
        loop = asyncio.get_running_loop()
        try:
            # run_in_executor 不支持关键字参数，这里用 partial 包装
            from functools import partial
            bound = partial(engine.generate, prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **kwargs)
            result = await loop.run_in_executor(self.executor, bound)
            return result
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            raise InferenceError(model_name, str(e))

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
            
        Usage:
            # 非流式
            text = await worker.generate_chat(model, messages, stream=False)
            
            # 流式
            async for chunk in worker.generate_chat(model, messages, stream=True):
                print(chunk)
        """
        if stream:
            return self._generate_chat_stream(
                model_name, messages, max_tokens, temperature, top_p, **kwargs
            )
        return self._generate_chat(
            model_name, messages, max_tokens, temperature, top_p, **kwargs
        )
    
    async def _generate_chat(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """内部非流式对话生成实现"""
        engine = REGISTRY.get_llm(model_name)
        if not engine:
            raise ModelNotFoundError(model_name)
        strategy_key = REGISTRY.get_llm_strategy_key(model_name) or "generic"
        strategy = CONTAINER.get_strategy(strategy_key)
        
        # 合并模型特性参数到 kwargs
        model_features = REGISTRY.get_llm_features(model_name)
        merged_kwargs = {**model_features, **kwargs}

        loop = asyncio.get_running_loop()
        try:
            from functools import partial
            bound = partial(
                strategy.generate,
                engine,
                messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **merged_kwargs,
            )
            result = await loop.run_in_executor(self.executor, bound)
            return result
        except Exception as e:
            logger.error(f"对话生成失败: {e}")
            raise InferenceError(model_name, str(e))
    
    async def _generate_chat_stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> AsyncIterator[str]:
        """内部流式对话生成实现"""
        engine = REGISTRY.get_llm(model_name)
        if not engine:
            raise ModelNotFoundError(model_name)
        strategy_key = REGISTRY.get_llm_strategy_key(model_name) or "generic"
        strategy = CONTAINER.get_strategy(strategy_key)
        
        # 合并模型特性参数到 kwargs
        model_features = REGISTRY.get_llm_features(model_name)
        merged_kwargs = {**model_features, **kwargs}

        try:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            
            def _stream_worker():
                try:
                    for chunk in strategy.generate(
                        engine,
                        messages,
                        stream=True,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        **merged_kwargs,
                    ):
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(e), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            
            self.executor.submit(_stream_worker)
            
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise InferenceError(model_name, str(chunk))
                yield chunk
                
        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"流式对话生成失败: {e}")
            raise InferenceError(model_name, str(e))
    
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
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                engine.embed,
                texts
            )
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
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self.executor,
                engine.rerank,
                query,
                documents,
                top_k
            )
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
            "llm_models": list(REGISTRY._llms.keys()),
            "embedding_models": list(REGISTRY._embeddings.keys()),
            "reranker_models": list(REGISTRY._rerankers.keys()),
            "default_llm": REGISTRY._default_llm,
            "default_embedding": REGISTRY._default_embedding,
            "default_reranker": REGISTRY._default_reranker,
        }
    
    def shutdown(self):
        """关闭工作器"""
        self.executor.shutdown(wait=True)
        logger.info("模型工作器已关闭")


# 全局工作器实例
WORKER = ModelWorker()
