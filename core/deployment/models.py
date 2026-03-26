"""部署模型定义

核心概念：
- ModelReplica: 一个引擎实例 + 设备绑定 + 活跃请求计数
- ModelDeployment: 一个模型的部署定义（含多个 replica + 调度器）
- RequestSlot: 一次推理请求获得的"入场券"，持有期间占用 replica 并发槽位
"""

import threading
import time
import uuid
from contextlib import contextmanager
from typing import Optional, Iterator, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from engines.base import LLMEngine
    from engines.cancel import CancelToken
    from core.config.schemas import LLMModelConfig


class ModelReplica:
    """模型副本：持有一个独立的引擎实例

    每个 replica 绑定到固定设备，维护活跃请求计数，
    供调度器做 least-loaded 路由决策。

    Attributes:
        replica_id: 副本唯一标识
        engine: 引擎实例
        device: 绑定的设备标识
        max_concurrent: 允许的最大并发推理数
    """

    __slots__ = (
        "replica_id", "engine", "device", "max_concurrent",
        "_active_count", "_lock",
    )

    def __init__(
        self,
        engine: "LLMEngine",
        device: Optional[str] = None,
        max_concurrent: int = 1,
        replica_id: Optional[str] = None,
    ) -> None:
        self.replica_id = replica_id or uuid.uuid4().hex[:8]
        self.engine = engine
        self.device = device
        self.max_concurrent = max_concurrent
        self._active_count = 0
        self._lock = threading.Lock()

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def available_slots(self) -> int:
        return max(0, self.max_concurrent - self._active_count)

    @property
    def is_available(self) -> bool:
        return self._active_count < self.max_concurrent

    def acquire(self) -> bool:
        """尝试占用一个并发槽位，成功返回 True"""
        with self._lock:
            if self._active_count < self.max_concurrent:
                self._active_count += 1
                return True
            return False

    def release(self) -> None:
        """释放一个并发槽位"""
        with self._lock:
            self._active_count = max(0, self._active_count - 1)

    def __repr__(self) -> str:
        return (
            f"<ModelReplica {self.replica_id} "
            f"device={self.device} "
            f"active={self._active_count}/{self.max_concurrent}>"
        )


class RequestSlot:
    """推理请求入场券

    通过 with 语句使用，离开作用域时自动释放 replica 槽位。
    携带 cancel_token 支持取消传播。

    用法::

        slot = deployment.acquire_slot(timeout=30)
        if slot is None:
            raise ServiceOverloaded()
        with slot:
            result = strategy.execute(slot.engine, input)
    """

    __slots__ = (
        "request_id", "replica", "cancel_token",
        "_released", "_acquired_at",
    )

    def __init__(
        self,
        replica: ModelReplica,
        cancel_token: Optional["CancelToken"] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self.request_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        self.replica = replica
        self.cancel_token = cancel_token
        self._released = False
        self._acquired_at = time.monotonic()

    @property
    def engine(self) -> "LLMEngine":
        return self.replica.engine

    @property
    def elapsed(self) -> float:
        """获取自分配以来经过的时间（秒）"""
        return time.monotonic() - self._acquired_at

    def release(self) -> None:
        """释放 replica 槽位（幂等）"""
        if not self._released:
            self._released = True
            self.replica.release()
            logger.debug(
                "请求槽位已释放: request={} replica={} elapsed={:.2f}s",
                self.request_id, self.replica.replica_id, self.elapsed,
            )

    def __enter__(self) -> "RequestSlot":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"<RequestSlot {self.request_id} "
            f"replica={self.replica.replica_id} "
            f"released={self._released}>"
        )


class ModelDeployment:
    """模型部署：管理一个模型的所有副本和调度

    职责：
    1. 持有 replica 列表
    2. 维护等待队列（Semaphore + 排队计数）
    3. 选择最优 replica 并分配 RequestSlot
    4. 提供部署级指标

    Attributes:
        model_name: 模型名称
        strategy_key: 对话策略标识
        config: 原始模型配置
    """

    def __init__(
        self,
        model_name: str,
        strategy_key: str,
        replicas: list[ModelReplica],
        max_queue_size: int = 64,
        request_timeout: float = 300.0,
        config: Optional["LLMModelConfig"] = None,
    ) -> None:
        self.model_name = model_name
        self.strategy_key = strategy_key
        self.config = config
        self._replicas = list(replicas)
        self._max_queue_size = max_queue_size
        self._request_timeout = request_timeout

        # 总并发容量 = 所有 replica 的 max_concurrent 之和
        total_capacity = sum(r.max_concurrent for r in self._replicas)
        # Semaphore 容量 = 总并发容量 + 排队缓冲
        self._semaphore = threading.Semaphore(total_capacity + max_queue_size)
        self._queue_count = 0
        self._queue_lock = threading.Lock()

        # 指标
        self._total_requests = 0
        self._total_rejected = 0
        self._total_timeouts = 0

    @property
    def replicas(self) -> list[ModelReplica]:
        return list(self._replicas)

    @property
    def total_active(self) -> int:
        return sum(r.active_count for r in self._replicas)

    @property
    def total_capacity(self) -> int:
        return sum(r.max_concurrent for r in self._replicas)

    @property
    def queue_count(self) -> int:
        return self._queue_count

    def acquire_slot(
        self,
        cancel_token: Optional["CancelToken"] = None,
        timeout: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> Optional[RequestSlot]:
        """获取推理槽位

        流程：
        1. 通过 Semaphore 做入场限流（含排队等待）
        2. 在所有 replica 中选择 least-loaded 的可用副本
        3. 占用 replica 槽位并返回 RequestSlot

        Args:
            cancel_token: 取消令牌，排队期间会检查
            timeout: 最大等待时间（秒），None 使用部署默认值
            request_id: 请求标识

        Returns:
            RequestSlot 或 None（超时/过载/取消）
        """
        effective_timeout = timeout if timeout is not None else self._request_timeout
        self._total_requests += 1

        # 入场限流
        with self._queue_lock:
            self._queue_count += 1

        try:
            acquired = self._semaphore.acquire(timeout=effective_timeout)
            if not acquired:
                self._total_timeouts += 1
                logger.warning(
                    "部署 {} 排队超时: timeout={:.1f}s queue={}",
                    self.model_name, effective_timeout, self._queue_count,
                )
                return None

            # 检查取消
            if cancel_token and cancel_token.is_cancelled:
                self._semaphore.release()
                return None

            # 选择 least-loaded replica
            replica = self._select_replica()
            if replica is None:
                # 所有 replica 都满了（不应发生，Semaphore 已控制）
                self._semaphore.release()
                self._total_rejected += 1
                logger.error("部署 {} 无可用副本（Semaphore 泄漏？）", self.model_name)
                return None

            slot = RequestSlot(
                replica=replica,
                cancel_token=cancel_token,
                request_id=request_id,
            )
            logger.debug(
                "已分配推理槽位: model={} request={} replica={} active={}/{}",
                self.model_name, slot.request_id,
                replica.replica_id, replica.active_count, replica.max_concurrent,
            )
            return slot

        finally:
            with self._queue_lock:
                self._queue_count = max(0, self._queue_count - 1)

    def release_slot(self, slot: RequestSlot) -> None:
        """释放槽位（由 RequestSlot.__exit__ 调用后，再释放 Semaphore）"""
        slot.release()
        self._semaphore.release()

    @contextmanager
    def slot_context(
        self,
        cancel_token: Optional["CancelToken"] = None,
        timeout: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> Iterator[RequestSlot]:
        """上下文管理器方式获取和释放槽位

        用法::

            with deployment.slot_context() as slot:
                result = strategy.execute(slot.engine, input)

        Raises:
            RuntimeError: 无法获取槽位（超时/过载）
        """
        slot = self.acquire_slot(
            cancel_token=cancel_token,
            timeout=timeout,
            request_id=request_id,
        )
        if slot is None:
            raise RuntimeError(
                f"模型 {self.model_name} 过载，无法分配推理槽位 "
                f"(active={self.total_active}/{self.total_capacity}, "
                f"queue={self.queue_count})"
            )
        try:
            yield slot
        finally:
            self.release_slot(slot)

    def _select_replica(self) -> Optional[ModelReplica]:
        """选择最优副本（least-loaded 策略）

        先按 active_count 排序找到最优候选，再尝试 acquire。
        如果 acquire 失败（并发竞争），尝试下一个候选。
        """
        # 按当前负载排序（快照，不持锁）
        candidates = sorted(self._replicas, key=lambda r: r.active_count)
        for r in candidates:
            if r.acquire():
                return r
        return None

    def get_metrics(self) -> dict:
        """获取部署级指标"""
        return {
            "model_name": self.model_name,
            "replicas": len(self._replicas),
            "total_capacity": self.total_capacity,
            "total_active": self.total_active,
            "queue_count": self.queue_count,
            "total_requests": self._total_requests,
            "total_rejected": self._total_rejected,
            "total_timeouts": self._total_timeouts,
            "replica_details": [
                {
                    "replica_id": r.replica_id,
                    "device": r.device,
                    "active": r.active_count,
                    "max_concurrent": r.max_concurrent,
                }
                for r in self._replicas
            ],
        }

    def shutdown(self) -> None:
        """关闭部署，释放所有资源"""
        logger.info("关闭部署: {} ({} replicas)", self.model_name, len(self._replicas))
        self._replicas.clear()

    def __repr__(self) -> str:
        return (
            f"<ModelDeployment {self.model_name} "
            f"replicas={len(self._replicas)} "
            f"active={self.total_active}/{self.total_capacity}>"
        )
