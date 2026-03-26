"""模型部署管理模块

提供从「模型名 → 单引擎实例」到「模型名 → 部署 → 副本池 → 调度」的升级，
支持同模型多副本并行推理、per-model 排队与限流、协作式取消。
"""

from core.deployment.models import ModelReplica, ModelDeployment, RequestSlot
from core.deployment.scheduler import ReplicaScheduler
from core.deployment.manager import DeploymentManager

__all__ = [
    "ModelReplica",
    "ModelDeployment",
    "RequestSlot",
    "ReplicaScheduler",
    "DeploymentManager",
]
