"""部署管理器

从配置构建 ModelDeployment 实例，管理所有模型部署的生命周期。
替代原有 ModelRegistry 中「模型名 → 单引擎实例」的模式，
升级为「模型名 → 部署（含多副本 + 调度）」。
"""

from typing import Dict, Optional, Any, List
from pathlib import Path

from loguru import logger

from core.config.schemas import LLMModelConfig
from core.deployment.models import ModelReplica, ModelDeployment
from engines.base import LLMEngine, EngineCapabilities


class DeploymentManager:
    """部署管理器：管理所有 LLM 模型部署

    职责：
    1. 从配置构建 ModelDeployment（含多副本引擎实例）
    2. 按模型名称检索部署
    3. 提供全局部署状态和指标
    4. 管理部署生命周期（启动/关闭）
    """

    def __init__(self) -> None:
        self._deployments: Dict[str, ModelDeployment] = {}
        self._name_index: Dict[str, str] = {}  # 规范化名称 -> 原始名称
        self._default_name: Optional[str] = None

    @staticmethod
    def _normalize_name(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        return str(name).strip().lower()

    def _resolve_name(self, name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        if name in self._deployments:
            return name
        return self._name_index.get(self._normalize_name(name) or "")

    def get_deployment(self, model_name: Optional[str]) -> Optional[ModelDeployment]:
        """获取指定模型名称的部署"""
        if model_name is None:
            model_name = self._default_name
        resolved = self._resolve_name(model_name)
        if resolved is None:
            return None
        return self._deployments.get(resolved)

    def get_default_name(self) -> Optional[str]:
        return self._default_name

    def list_names(self) -> List[str]:
        return list(self._deployments.keys())

    def count(self) -> int:
        return len(self._deployments)

    def register_deployment(self, deployment: ModelDeployment) -> None:
        """注册一个已构建的部署"""
        name = deployment.model_name
        self._deployments[name] = deployment
        normalized = self._normalize_name(name)
        if normalized:
            self._name_index[normalized] = name
        if self._default_name is None:
            self._default_name = name
        logger.info(
            "部署已注册: {} (replicas={}, capacity={})",
            name, len(deployment.replicas), deployment.total_capacity,
        )

    def build_and_register(
        self,
        llm_cfg: LLMModelConfig,
        engine_builder,
        engine_defaults,
    ) -> bool:
        """从配置构建部署并注册

        根据 llm_cfg.replicas 构建多个引擎实例（每个绑定到不同设备），
        组装成 ModelDeployment 后注册到管理器。

        Args:
            llm_cfg: LLM 模型配置
            engine_builder: 引擎构建函数 (cfg, engine_type, device, dtype) -> LLMEngine
            engine_defaults: 全局引擎默认配置

        Returns:
            True 表示构建成功
        """
        engine_type = llm_cfg.engine or engine_defaults.llm_engine
        base_device = llm_cfg.device or engine_defaults.device
        dtype = llm_cfg.dtype or engine_defaults.dtype
        num_replicas = max(1, llm_cfg.replicas)

        # 验证模型路径
        if not Path(llm_cfg.path).exists():
            logger.error("模型路径不存在: {} -> {}", llm_cfg.name, llm_cfg.path)
            return False

        # 解析每个副本的设备
        if llm_cfg.devices and len(llm_cfg.devices) >= num_replicas:
            replica_devices = llm_cfg.devices[:num_replicas]
        else:
            replica_devices = [base_device] * num_replicas

        # 计算每副本并发数
        max_concurrent = llm_cfg.max_concurrent

        # 构建副本
        replicas: List[ModelReplica] = []
        for i in range(num_replicas):
            device = replica_devices[i]
            replica_id = f"{llm_cfg.name}_r{i}"

            try:
                engine = engine_builder(llm_cfg, engine_type, device, dtype)

                # 预热：加载模型到 GPU
                if hasattr(engine, "_ensure_loaded"):
                    engine._ensure_loaded()

                # 根据引擎能力调整并发数
                # 优先使用实例级能力（能反映 vLLM fallback 等运行时状态）
                if hasattr(engine, "instance_capabilities"):
                    caps = engine.instance_capabilities()
                else:
                    caps = engine.capabilities()
                effective_concurrent = max_concurrent
                if caps.supports_concurrent_requests and caps.preferred_max_concurrency > 0:
                    effective_concurrent = max(
                        max_concurrent, caps.preferred_max_concurrency
                    )
                elif not caps.supports_concurrent_requests:
                    # 不支持并发的引擎（如 transformers），强制单并发
                    effective_concurrent = 1

                # 检测 vLLM fallback 场景并输出显眼警告
                actual_engine_type = engine_type
                if engine_type == "vllm" and not caps.supports_concurrent_requests:
                    actual_engine_type = "transformers(fallback)"
                    logger.warning(
                        "⚠ 副本 {} 配置为 vLLM 但实际回退到 transformers 引擎"
                        "（单并发串行推理），并发请求将排队执行！"
                        "请检查 vLLM 启动日志排查失败原因。",
                        replica_id,
                    )

                replica = ModelReplica(
                    engine=engine,
                    device=device,
                    max_concurrent=effective_concurrent,
                    replica_id=replica_id,
                )
                replicas.append(replica)
                logger.info(
                    "副本构建成功: {} via {} (device={}, concurrent={})",
                    replica_id, actual_engine_type, device, effective_concurrent,
                )

            except Exception as e:
                logger.error("副本构建失败: {} -> {}", replica_id, e)
                continue

        if not replicas:
            logger.error("模型 {} 所有副本构建失败", llm_cfg.name)
            return False

        deployment = ModelDeployment(
            model_name=llm_cfg.name,
            strategy_key=llm_cfg.chat_strategy,
            replicas=replicas,
            max_queue_size=llm_cfg.max_queue_size,
            request_timeout=llm_cfg.request_timeout,
            config=llm_cfg,
        )
        self.register_deployment(deployment)
        return True

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有部署的指标"""
        return {
            name: dep.get_metrics()
            for name, dep in self._deployments.items()
        }

    def shutdown(self) -> None:
        """关闭所有部署"""
        for name, dep in self._deployments.items():
            dep.shutdown()
        self._deployments.clear()
        self._name_index.clear()
        self._default_name = None
        logger.info("所有部署已关闭")

    def clear(self) -> None:
        """清空所有部署（别名）"""
        self.shutdown()
