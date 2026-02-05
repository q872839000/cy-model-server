"""
轻量级记忆管理器

管理想定生成的中间状态，支持TTL自动清理。
"""
import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class ScenarioMemory:
    """想定记忆（中间状态）"""
    memory_id: str
    phase: str = "processing"
    equipment_library: List[Dict[str, Any]] = field(default_factory=list)
    equipment_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    equipment_plan: List[Dict[str, Any]] = field(default_factory=list)
    allocated_positions: List[Dict[str, float]] = field(default_factory=list)
    current_instances: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    validation_attempts: int = 0
    current_proposal: Optional[Dict[str, Any]] = None  # 当前待确认的部署方案
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def update(self):
        """更新时间戳"""
        self.updated_at = time.time()
    
    def to_context_dict(self) -> Dict[str, Any]:
        """转换为上下文字典（用于System Prompt）"""
        return {
            "phase": self.phase,
            "equipment_library_count": len(self.equipment_library),
            "equipment_plan": self.equipment_plan,
            "allocated_positions": self.allocated_positions,
            "current_instances_count": len(self.current_instances),
            "validation_errors": self.validation_errors,
            "validation_attempts": self.validation_attempts,
        }
    
    def add_equipment_plan(self, item: Dict[str, Any]):
        """添加装备规划"""
        self.equipment_plan.append(item)
        self.update()
        logger.debug("【Memory】 添加装备规划: {} (共{}项)", item.get("equipId", "?"), len(self.equipment_plan))
    
    def add_position(self, position: Dict[str, float]):
        """添加位置"""
        self.allocated_positions.append(position)
        self.update()
        logger.debug("【Memory】 添加位置: lon={}, lat={} (共{}个)", 
                    position.get("lon"), position.get("lat"), len(self.allocated_positions))
    
    def add_instance(self, instance: Dict[str, Any]):
        """添加装备实例"""
        self.current_instances.append(instance)
        self.update()
        logger.debug("【Memory】 添加实例: {} [{}] (共{}个)", 
                    instance.get("otherName"), instance.get("classify"), len(self.current_instances))
    
    def set_validation_errors(self, errors: List[Dict[str, Any]]):
        """设置校验错误"""
        self.validation_errors = errors
        self.validation_attempts += 1
        self.update()
        logger.debug("【Memory】 设置校验错误: {}个 (第{}次尝试)", len(errors), self.validation_attempts)
    
    def clear_validation_errors(self):
        """清除校验错误"""
        self.validation_errors = []
        self.update()
        logger.debug("【Memory】 清除校验错误")
    
    def reset_for_readjustment(self):
        """重置记忆以进行二次调整（保留装备库，清除实例相关数据）"""
        old_instance_count = len(self.current_instances)
        old_plan_count = len(self.equipment_plan)
        
        # 清除实例和计划，保留装备库和Schema
        self.current_instances = []
        self.equipment_plan = []
        self.allocated_positions = []
        self.validation_errors = []
        self.validation_attempts = 0
        self.is_completed = False
        
        self.update()
        logger.info("【Memory】 重置二次调整: 清除{}个实例, {}项计划", old_instance_count, old_plan_count)


class MemoryManager:
    """
    记忆管理器
    
    功能：
    - 创建/获取/删除记忆
    - TTL自动清理
    - 内存存储（可扩展为Redis）
    """
    
    def __init__(self, ttl_seconds: int = 1800, cleanup_interval: int = 300):
        """
        Args:
            ttl_seconds: 记忆过期时间（默认30分钟）
            cleanup_interval: 清理间隔（默认5分钟）
        """
        self._memories: Dict[str, ScenarioMemory] = {}
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    def create(self, memory_id: Optional[str] = None) -> ScenarioMemory:
        """创建新记忆"""
        if memory_id is None:
            memory_id = f"mem_{int(time.time() * 1000) % 1000000000000}"
        
        memory = ScenarioMemory(memory_id=memory_id)
        self._memories[memory_id] = memory
        logger.debug("创建记忆: {}", memory_id)
        return memory
    
    def get(self, memory_id: str) -> Optional[ScenarioMemory]:
        """获取记忆"""
        memory = self._memories.get(memory_id)
        if memory:
            # 检查是否过期
            if time.time() - memory.updated_at > self._ttl_seconds:
                self.delete(memory_id)
                return None
            return memory
        return None
    
    def get_or_create(self, memory_id: Optional[str] = None) -> ScenarioMemory:
        """获取或创建记忆"""
        if memory_id:
            memory = self.get(memory_id)
            if memory:
                logger.debug("【MemoryManager】 获取已有记忆: {} (phase={})", memory_id, memory.phase)
                return memory
        memory = self.create(memory_id)
        logger.info("【MemoryManager】 创建新记忆: {}", memory.memory_id)
        return memory
    
    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        if memory_id in self._memories:
            del self._memories[memory_id]
            logger.debug("删除记忆: {}", memory_id)
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """清理过期记忆"""
        now = time.time()
        expired = [
            mid for mid, mem in self._memories.items()
            if now - mem.updated_at > self._ttl_seconds
        ]
        for mid in expired:
            del self._memories[mid]
        
        if expired:
            logger.info("清理过期记忆: {} 个", len(expired))
        return len(expired)
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("记忆清理异常: {}", e)
    
    async def start(self):
        """启动管理器"""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("MemoryManager 启动, TTL={}s", self._ttl_seconds)
    
    async def stop(self):
        """停止管理器"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("MemoryManager 停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_memories": len(self._memories),
            "ttl_seconds": self._ttl_seconds,
            "running": self._running,
        }


# 全局记忆管理器实例
MEMORY_MANAGER = MemoryManager()
