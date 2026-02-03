"""
通用异步工作器：纯粹的任务执行器

职责：
1. 并发控制（线程池）
2. 超时管理
3. 任务取消

设计原则：
- 不感知业务类型（模型、策略、RAG 等）
- 不导入任何业务模块
- 只执行传入的 callable
"""

import asyncio
import threading
from typing import (
    Any,
    Callable,
    TypeVar,
    Iterator,
    AsyncIterator,
    Optional,
)
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from loguru import logger


T = TypeVar("T")


@dataclass
class TaskResult:
    """任务执行结果"""
    success: bool
    value: Any = None
    error: Optional[Exception] = None


class AsyncWorker:
    """
    通用异步工作器
    
    提供将同步 callable 异步执行的能力，支持：
    - 非流式调用：返回完整结果
    - 流式调用：返回异步迭代器
    - 超时控制
    - 任务取消
    
    使用示例:
        worker = AsyncWorker(max_workers=4)
        
        # 非流式
        result = await worker.run(my_sync_function, arg1, arg2, kwarg1=value1)
        
        # 流式
        async for chunk in worker.run_stream(my_sync_generator, arg1):
            print(chunk)
    """

    def __init__(self, max_workers: int = 4):
        """
        初始化工作器
        
        Args:
            max_workers: 线程池最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._task_counter = 0
        self._lock = threading.Lock()
        self._active_tasks: dict[str, Future] = {}

    def _get_next_task_id(self) -> str:
        """获取下一个任务ID"""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter}"

    async def run(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> T:
        """
        异步执行同步函数
        
        Args:
            func: 要执行的同步函数
            *args: 位置参数
            timeout: 超时时间（秒），None 表示不超时
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            asyncio.TimeoutError: 超时
            Exception: 函数执行异常
        """
        loop = asyncio.get_running_loop()
        task_id = self._get_next_task_id()
        
        future = loop.run_in_executor(
            self.executor,
            lambda: func(*args, **kwargs)
        )
        
        try:
            if timeout is not None:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            return result
        except asyncio.TimeoutError:
            logger.warning(f"任务 {task_id} 超时 (timeout={timeout}s)")
            raise
        except Exception as e:
            logger.error(f"任务 {task_id} 执行失败: {e}")
            raise

    async def run_stream(
        self,
        func: Callable[..., Iterator[T]],
        *args,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[T]:
        """
        异步执行同步生成器函数，返回异步迭代器
        
        Args:
            func: 要执行的同步生成器函数
            *args: 位置参数
            timeout: 单个 chunk 的超时时间（秒）
            **kwargs: 关键字参数
            
        Yields:
            生成器产出的每个元素
            
        Raises:
            asyncio.TimeoutError: 超时
            Exception: 函数执行异常
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        task_id = self._get_next_task_id()
        error_holder: list = []

        def _stream_worker():
            try:
                for chunk in func(*args, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as e:
                error_holder.append(e)
                asyncio.run_coroutine_threadsafe(queue.put(e), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        self.executor.submit(_stream_worker)

        while True:
            try:
                if timeout is not None:
                    chunk = await asyncio.wait_for(queue.get(), timeout=timeout)
                else:
                    chunk = await queue.get()
            except asyncio.TimeoutError:
                logger.warning(f"流式任务 {task_id} 超时")
                raise

            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务（尽力而为）
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        with self._lock:
            future = self._active_tasks.get(task_id)
            if future is not None:
                return future.cancel()
        return False

    def shutdown(self, wait: bool = True):
        """
        关闭工作器
        
        Args:
            wait: 是否等待所有任务完成
        """
        self.executor.shutdown(wait=wait)
        logger.info("AsyncWorker 已关闭")


# 全局工作器实例
ASYNC_WORKER = AsyncWorker()
