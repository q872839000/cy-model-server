"""
通用异步工作器：纯粹的任务执行器

职责：
1. 并发控制（线程池）
2. 超时管理
3. 流式传输（含背压和取消机制）

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
from concurrent.futures import ThreadPoolExecutor
from loguru import logger


T = TypeVar("T")

# 流式队列的默认上限：防止快生产慢消费导致内存无限增长
# 设为 64 足够缓冲 GPU 输出，同时限制内存占用
_DEFAULT_STREAM_QUEUE_MAXSIZE = 64


class _StreamHandle:
    """流式传输句柄：管理生产线程与消费端之间的生命周期。

    核心职责：
    1. cancel_event: 消费端停止迭代时通知生产线程停止
    2. queue: 带 maxsize 的有界队列，实现背压
    3. 确保生产线程异常能正确传播到消费端

    协议：
    - 生产线程每次 yield 前检查 cancel_event
    - 消费端通过 async for 消费，正常结束或异常时调用 cancel()
    - queue 中放入 None 表示结束，放入 Exception 实例表示错误
    """

    def __init__(self, loop: asyncio.AbstractEventLoop,
                 maxsize: int = _DEFAULT_STREAM_QUEUE_MAXSIZE):
        self.cancel_event = threading.Event()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._loop = loop

    def cancel(self):
        """通知生产线程停止"""
        self.cancel_event.set()

    def put_threadsafe(self, item: Any) -> bool:
        """从生产线程向队列投递（线程安全，支持背压）。

        当队列满时阻塞等待，同时每 0.5s 检查 cancel_event，
        避免消费端已取消但生产端永久阻塞在 put 上。

        Returns:
            True 表示成功投递，False 表示已取消
        """
        if self.cancel_event.is_set():
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.queue.put(item), self._loop)
        except Exception:
            return False
        # 等待同一个 future 完成，周期性检查取消状态
        while True:
            try:
                future.result(timeout=0.5)
                return True
            except TimeoutError:
                if self.cancel_event.is_set():
                    future.cancel()
                    return False
                continue
            except Exception:
                return False

    def put_end_threadsafe(self):
        """从生产线程发送结束标记"""
        try:
            asyncio.run_coroutine_threadsafe(
                self.queue.put(None), self._loop
            ).result(timeout=5.0)
        except Exception:
            pass

    def put_error_threadsafe(self, exc: Exception):
        """从生产线程发送错误"""
        try:
            asyncio.run_coroutine_threadsafe(
                self.queue.put(exc), self._loop
            ).result(timeout=5.0)
        except Exception:
            pass


class AsyncWorker:
    """
    通用异步工作器

    提供将同步 callable 异步执行的能力，支持：
    - 非流式调用：返回完整结果
    - 流式调用：返回异步迭代器（含背压和取消机制）
    - 超时控制

    使用示例:
        worker = AsyncWorker(max_workers=4)

        # 非流式
        result = await worker.run(my_sync_function, arg1, arg2, kwarg1=value1)

        # 流式
        async for chunk in worker.run_stream(my_sync_generator, arg1):
            print(chunk)
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._task_counter = 0
        self._lock = threading.Lock()

    def _get_next_task_id(self) -> str:
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
        异步执行同步生成器函数，返回异步迭代器。

        具有背压和取消机制：
        - 队列满时生产线程阻塞，防止慢消费导致内存无限增长
        - 消费端停止迭代时通知生产线程尽早停止，避免 GPU 空跑

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
        handle = _StreamHandle(loop)
        task_id = self._get_next_task_id()

        def _stream_worker():
            try:
                for chunk in func(*args, **kwargs):
                    if handle.cancel_event.is_set():
                        logger.debug("流式任务 {} 已取消，停止生产", task_id)
                        break
                    if not handle.put_threadsafe(chunk):
                        break
            except Exception as e:
                handle.put_error_threadsafe(e)
            finally:
                handle.put_end_threadsafe()

        self.executor.submit(_stream_worker)

        try:
            while True:
                try:
                    if timeout is not None:
                        chunk = await asyncio.wait_for(
                            handle.queue.get(), timeout=timeout)
                    else:
                        chunk = await handle.queue.get()
                except asyncio.TimeoutError:
                    logger.warning(f"流式任务 {task_id} 超时")
                    raise

                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        finally:
            # 消费端退出时（正常结束、异常、客户端断开），通知生产线程停止
            handle.cancel()

    def shutdown(self, wait: bool = True):
        """关闭工作器"""
        self.executor.shutdown(wait=wait)
        logger.info("AsyncWorker 已关闭")


# 全局工作器实例
ASYNC_WORKER = AsyncWorker()
