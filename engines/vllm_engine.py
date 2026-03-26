"""vLLM 引擎实现

并发模型：单副本多请求并发（连续批处理），通过内部 step-loop 线程统一调度。
- 每个请求通过 add_request() 提交到 vLLM 内部调度器
- 专用 step-loop 线程持续执行 step()，将输出分发到各请求的结果队列
- 支持 abort(request_id) 实现协作式取消
- 不使用 _inference_lock（并发由 vLLM 内部调度器管理）
- 若 vLLM 不可用，自动回退到 TransformersLLMEngine

注意：vLLM 对 CUDA 驱动、显存、环境有较高要求，Windows 不支持 vLLM。
"""

import queue
import threading
import uuid
from typing import Optional, Dict, Any, Iterator, Union
from loguru import logger

from engines.base import LLMEngine, EngineCapabilities

# vLLM 是否可用的标志
_VLLM_AVAILABLE = False
_LLM = None
_LLMEngine = None
_SamplingParams = None
_EngineArgs = None

try:
    from vllm import LLM as _LLM, SamplingParams as _SamplingParams
    from vllm.engine.llm_engine import LLMEngine as _LLMEngine
    from vllm.engine.arg_utils import EngineArgs as _EngineArgs
    _VLLM_AVAILABLE = True
except ImportError:
    pass


# 队列哨兵值
_SENTINEL_DONE = object()


class _RequestHandle:
    """单个推理请求的句柄，承载结果队列和取消信号"""

    __slots__ = ("request_id", "result_queue", "cancelled", "finished")

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.result_queue: queue.Queue = queue.Queue(maxsize=128)
        self.cancelled = False
        self.finished = False

    def put_delta(self, delta: str) -> None:
        """投递增量文本"""
        if not self.cancelled:
            try:
                self.result_queue.put_nowait(delta)
            except queue.Full:
                pass  # 背压：消费端太慢，丢弃增量

    def put_done(self) -> None:
        """标记请求完成"""
        self.finished = True
        try:
            self.result_queue.put_nowait(_SENTINEL_DONE)
        except queue.Full:
            pass

    def put_error(self, exc: Exception) -> None:
        """投递错误"""
        self.finished = True
        try:
            self.result_queue.put_nowait(exc)
        except queue.Full:
            pass

    def cancel(self) -> None:
        """标记取消"""
        self.cancelled = True


class VLLMLLMEngine(LLMEngine):
    """vLLM 引擎封装，支持高并发推理和流式输出。

    并发模型：
    - 单引擎实例支持多请求并发（连续批处理）
    - 内部维护一个 step-loop 线程，统一调度所有活跃请求
    - 每个请求通过 _RequestHandle 获取结果
    - 支持 abort(request_id) 取消正在执行的请求

    回退：若 vLLM 不可用，自动回退到 TransformersLLMEngine（单并发）。
    """

    @classmethod
    def capabilities(cls) -> EngineCapabilities:
        if _VLLM_AVAILABLE:
            return EngineCapabilities(
                supports_concurrent_requests=True,
                supports_cancel=True,
                preferred_max_concurrency=32,
            )
        # 回退到 transformers 时
        return EngineCapabilities(
            supports_concurrent_requests=False,
            supports_cancel=True,
            preferred_max_concurrency=1,
        )

    def __init__(self, model_path: str, dtype: Optional[str] = None,
                 device: Optional[str] = None,
                 gen_params: Optional[Dict[str, Any]] = None,
                 tensor_parallel_size: int = 1) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.gen_params = gen_params or {}
        self.tensor_parallel_size = tensor_parallel_size
        self._llm_engine = None    # 底层 vLLM LLMEngine
        self._fallback = None      # 回退到 transformers

        # step-loop 线程管理
        self._step_thread: Optional[threading.Thread] = None
        self._active_handles: Dict[str, _RequestHandle] = {}
        self._handles_lock = threading.Lock()
        self._has_work = threading.Event()  # 通知 step-loop 有新请求
        self._shutdown = threading.Event()

    # ==================== 能力探测 ====================

    @property
    def is_vllm_active(self) -> bool:
        return self._llm_engine is not None

    # ==================== Chat Template ====================

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Optional[str]:
        """委托给底层引擎或 fallback 的 apply_chat_template

        始终返回字符串（tokenize=False）。
        """
        self._ensure_loaded()
        if self._fallback is not None:
            return self._fallback.apply_chat_template(
                messages, tools=tools, **kwargs)
        if self._llm_engine is not None:
            try:
                tokenizer = self._llm_engine.get_tokenizer()
                if tokenizer is None:
                    return None
                template_kwargs = self._build_template_kwargs(
                    tools=tools, tokenize=False, **kwargs)
                return tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as e:
                logger.warning("vLLM tokenizer.apply_chat_template 失败: {}", e)
                return None
        return None

    # ==================== generate 入口 ====================

    def generate(
        self, prompt: str, stream: bool = False, **kwargs,
    ) -> Union[str, Iterator[str]]:
        """当使用 TransformersLLMEngine 回退时，完整委托给 fallback.generate()。

        避免 base class generate() 先 pop stop 再调用 _generate → fallback.generate()
        导致 stop words 丢失的问题。
        """
        self._ensure_loaded()
        if self._fallback is not None:
            return self._fallback.generate(prompt, stream=stream, **kwargs)
        return super().generate(prompt, stream=stream, **kwargs)

    # ==================== 模型加载 ====================

    def _ensure_loaded(self) -> None:
        if self._llm_engine is not None or self._fallback is not None:
            return
        with self._load_lock:
            if self._llm_engine is not None or self._fallback is not None:
                return
            self._do_load()

    def _do_load(self) -> None:
        """实际执行模型加载（由 _ensure_loaded 在锁内调用）"""
        if _VLLM_AVAILABLE:
            try:
                logger.info("Loading vLLM model {}", self.model_path)
                engine_kwargs: Dict[str, Any] = {
                    "model": self.model_path,
                    "trust_remote_code": True,
                }
                if self.dtype:
                    engine_kwargs["dtype"] = self.dtype
                if self.tensor_parallel_size > 1:
                    engine_kwargs["tensor_parallel_size"] = self.tensor_parallel_size

                engine_args = _EngineArgs(**engine_kwargs)
                self._llm_engine = _LLMEngine.from_engine_args(engine_args)

                # 启动 step-loop 线程
                self._start_step_loop()
                logger.info(
                    "vLLM LLMEngine loaded successfully (tp={})",
                    self.tensor_parallel_size,
                )
                return
            except Exception as e:
                if "float16" in str(e) and self.dtype == "float16":
                    logger.warning("vLLM 不支持 float16，尝试 bfloat16...")
                    try:
                        engine_kwargs["dtype"] = "bfloat16"
                        engine_args = _EngineArgs(**engine_kwargs)
                        self._llm_engine = _LLMEngine.from_engine_args(engine_args)
                        self._start_step_loop()
                        return
                    except Exception as e2:
                        logger.warning("vLLM bfloat16 也失败，回退到 transformers：{}", e2)
                else:
                    logger.warning("vLLM 加载失败（会回退到 transformers）：{}", e)
        else:
            logger.warning("vLLM 未安装，回退到 transformers")

        # 回退：使用 transformers
        try:
            from engines.transformers_engine import TransformersLLMEngine
            self._fallback = TransformersLLMEngine(
                self.model_path, dtype=self.dtype, device=self.device,
                gen_params=self.gen_params,
            )
        except Exception as e2:
            raise RuntimeError("vLLM 与 transformers 均不可用，请检查依赖") from e2

    # ==================== Step-Loop 线程 ====================

    def _start_step_loop(self) -> None:
        """启动后台 step-loop 线程"""
        self._shutdown.clear()
        self._step_thread = threading.Thread(
            target=self._step_loop, daemon=True, name="vllm-step-loop",
        )
        self._step_thread.start()
        logger.info("vLLM step-loop 线程已启动")

    def _step_loop(self) -> None:
        """后台 step-loop：持续调用 engine.step() 处理所有活跃请求

        设计要点：
        - 有活跃请求时持续 step（CPU busy-loop，vLLM 标准用法）
        - 无活跃请求时等待 _has_work 信号（避免空转）
        - 每次 step 输出分发到各请求的 result_queue
        """
        while not self._shutdown.is_set():
            # 无活跃请求时休眠
            if not self._llm_engine.has_unfinished_requests():
                self._has_work.wait(timeout=1.0)
                self._has_work.clear()
                continue

            try:
                step_outputs = self._llm_engine.step()
            except Exception as e:
                logger.error("vLLM step() 异常: {}", e)
                # 通知所有活跃请求出错
                with self._handles_lock:
                    for handle in self._active_handles.values():
                        handle.put_error(RuntimeError(f"vLLM step error: {e}"))
                    self._active_handles.clear()
                continue

            # 分发输出到各请求
            for output in step_outputs:
                rid = output.request_id
                with self._handles_lock:
                    handle = self._active_handles.get(rid)
                if handle is None:
                    continue

                # 检查取消
                if handle.cancelled:
                    self._abort_request(rid)
                    continue

                if output.outputs:
                    current_text = output.outputs[0].text
                    # 计算增量（vLLM 返回累积文本）
                    prev_len = getattr(handle, '_prev_len', 0)
                    if len(current_text) > prev_len:
                        delta = current_text[prev_len:]
                        handle.put_delta(delta)
                        handle._prev_len = len(current_text)

                if output.finished:
                    handle.put_done()
                    with self._handles_lock:
                        self._active_handles.pop(rid, None)

        logger.info("vLLM step-loop 线程已退出")

    def _abort_request(self, request_id: str) -> None:
        """中止一个请求"""
        try:
            self._llm_engine.abort_request(request_id)
        except Exception as e:
            logger.debug("abort_request({}) 失败: {}", request_id, e)
        with self._handles_lock:
            handle = self._active_handles.pop(request_id, None)
        if handle and not handle.finished:
            handle.put_done()

    # ==================== 请求提交 ====================

    def _submit_request(self, prompt: str, **kwargs) -> _RequestHandle:
        """向 vLLM 引擎提交一个推理请求

        Returns:
            _RequestHandle: 请求句柄，可从中读取流式结果
        """
        sampling_params = self._get_sampling_params(**kwargs)
        request_id = str(uuid.uuid4())
        handle = _RequestHandle(request_id)
        handle._prev_len = 0  # 用于计算增量文本

        with self._handles_lock:
            self._active_handles[request_id] = handle

        # 提交到 vLLM 引擎
        self._llm_engine.add_request(request_id, prompt, sampling_params)

        # 唤醒 step-loop
        self._has_work.set()

        return handle

    def _get_sampling_params(self, **kwargs):
        """构建 vLLM SamplingParams"""
        max_tokens = kwargs.get('max_tokens', self.gen_params.get('max_tokens', 128))
        temperature = kwargs.get('temperature', self.gen_params.get('temperature', 0.0))
        top_p = kwargs.get('top_p', self.gen_params.get('top_p', 1.0))
        top_k = kwargs.get('top_k', self.gen_params.get('top_k', None))

        sp_kwargs: Dict[str, Any] = dict(
            temperature=temperature if temperature > 0.0 else 0.0,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if top_k is not None:
            sp_kwargs["top_k"] = top_k

        return _SamplingParams(**sp_kwargs)

    # ==================== 生成实现 ====================

    def _generate(self, prompt: str, **kwargs) -> str:
        """非流式生成：提交请求并等待完整结果"""
        self._ensure_loaded()
        cancel_token = kwargs.pop('cancel_token', None)

        if self._llm_engine is not None:
            handle = self._submit_request(prompt, **kwargs)

            # 收集完整文本
            chunks = []
            while True:
                try:
                    item = handle.result_queue.get(timeout=1.0)
                except queue.Empty:
                    if cancel_token and cancel_token.is_cancelled:
                        handle.cancel()
                        self._abort_request(handle.request_id)
                        break
                    continue

                if item is _SENTINEL_DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                chunks.append(item)

            return "".join(chunks)

        elif self._fallback is not None:
            return self._fallback.generate(prompt, **kwargs)
        else:
            raise RuntimeError("No underlying engine available")

    def _generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成：提交请求并逐步 yield 增量文本"""
        self._ensure_loaded()
        cancel_token = kwargs.pop('cancel_token', None)

        if self._llm_engine is not None:
            handle = self._submit_request(prompt, **kwargs)

            try:
                while True:
                    try:
                        item = handle.result_queue.get(timeout=1.0)
                    except queue.Empty:
                        if cancel_token and cancel_token.is_cancelled:
                            handle.cancel()
                            self._abort_request(handle.request_id)
                            return
                        continue

                    if item is _SENTINEL_DONE:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item
            except GeneratorExit:
                # 消费端停止迭代
                handle.cancel()
                self._abort_request(handle.request_id)

        elif self._fallback is not None:
            yield from self._fallback.generate(prompt, stream=True, **kwargs)
        else:
            raise RuntimeError("No underlying engine available")

    # ==================== 生命周期 ====================

    def shutdown(self) -> None:
        """关闭引擎，停止 step-loop"""
        self._shutdown.set()
        self._has_work.set()  # 唤醒以检查 shutdown
        if self._step_thread and self._step_thread.is_alive():
            self._step_thread.join(timeout=10)
        # 通知所有活跃请求
        with self._handles_lock:
            for handle in self._active_handles.values():
                handle.put_done()
            self._active_handles.clear()
        logger.info("VLLMLLMEngine 已关闭")