"""副本调度器

提供 per-deployment 的请求调度与入场控制。
当前实现为 least-loaded 策略，后续可扩展为 token-aware、优先级队列等。
"""

from typing import Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from core.deployment.models import ModelDeployment, RequestSlot
    from engines.cancel import CancelToken


class ReplicaScheduler:
    """副本调度器：为 ModelDeployment 提供调度决策

    当前调度策略：
    - least-loaded: 选择活跃请求最少的可用副本
    - 入场控制: 基于 Semaphore + max_queue_size 限流

    调度器不持有 replica 状态，所有状态由 ModelDeployment 管理。
    调度器的价值在于：
    1. 将调度策略从 deployment 中解耦，便于替换
    2. 提供统一的指标采集点
    3. 后续可扩展 token-aware 调度
    """

    def acquire(
        self,
        deployment: "ModelDeployment",
        cancel_token: Optional["CancelToken"] = None,
        timeout: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> Optional["RequestSlot"]:
        """通过部署获取推理槽位

        当前直接委托给 deployment.acquire_slot()，
        后续可在此层加入更复杂的调度逻辑（如跨部署路由、优先级）。

        Args:
            deployment: 目标部署
            cancel_token: 取消令牌
            timeout: 最大等待时间（秒）
            request_id: 请求标识

        Returns:
            RequestSlot 或 None
        """
        return deployment.acquire_slot(
            cancel_token=cancel_token,
            timeout=timeout,
            request_id=request_id,
        )

    def release(
        self,
        deployment: "ModelDeployment",
        slot: "RequestSlot",
    ) -> None:
        """释放槽位"""
        deployment.release_slot(slot)
