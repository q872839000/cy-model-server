"""
Agent基类定义

定义Agent的核心数据结构和抽象接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class Intent(str, Enum):
    """用户意图类型"""
    CREATE = "create"           # 创建新想定
    MODIFY = "modify"           # 修改现有想定
    PLAN_ONLY = "plan_only"     # 只制定计划，不执行
    QUERY = "query"             # 查询信息
    CONFIRM = "confirm"         # 确认执行
    UNCLEAR = "unclear"         # 信息不完整，需要澄清


class StepStatus(str, Enum):
    """计划步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """执行计划步骤"""
    step_id: int
    description: str
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentState:
    """Agent状态"""
    intent: Optional[Intent] = None
    intent_details: Dict[str, Any] = field(default_factory=dict)
    plan: List[PlanStep] = field(default_factory=list)
    current_step: int = 0
    thoughts: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    clarification_options: List[str] = field(default_factory=list)
    iteration: int = 0
    
    def add_thought(self, thought: str):
        """添加思考记录"""
        self.thoughts.append(thought)
    
    def add_observation(self, observation: str):
        """添加观察记录"""
        self.observations.append(observation)
    
    def get_current_plan_step(self) -> Optional[PlanStep]:
        """获取当前待执行的步骤"""
        for step in self.plan:
            if step.status == StepStatus.PENDING:
                return step
        return None
    
    def mark_step_completed(self, step_id: int, result: Dict[str, Any]):
        """标记步骤完成"""
        for step in self.plan:
            if step.step_id == step_id:
                step.status = StepStatus.COMPLETED
                step.result = result
                break
    
    def mark_step_failed(self, step_id: int, error: str):
        """标记步骤失败"""
        for step in self.plan:
            if step.step_id == step_id:
                step.status = StepStatus.FAILED
                step.error = error
                break
    
    def is_plan_completed(self) -> bool:
        """检查计划是否全部完成"""
        if not self.plan:
            return False
        return all(
            step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) 
            for step in self.plan
        )
    
    def has_failed_steps(self) -> bool:
        """检查是否有失败的步骤"""
        return any(step.status == StepStatus.FAILED for step in self.plan)


@dataclass
class AgentResponse:
    """Agent响应"""
    message: str
    memory_id: str
    success: bool = True
    needs_input: bool = False
    needs_confirmation: bool = False
    plan: List[Dict[str, Any]] = field(default_factory=list)
    instances: List[Dict[str, Any]] = field(default_factory=list)
    thought_process: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "message": self.message,
            "memory_id": self.memory_id,
            "success": self.success,
            "needs_input": self.needs_input,
            "needs_confirmation": self.needs_confirmation,
            "plan": self.plan,
            "instances": self.instances,
            "thought_process": self.thought_process,
        }


class BaseAgent(ABC):
    """Agent抽象基类"""
    
    @abstractmethod
    async def run(
        self, 
        user_input: str, 
        memory: Any,
        history: Optional[List[Dict[str, str]]] = None
    ) -> AgentResponse:
        """
        执行Agent循环
        
        Args:
            user_input: 用户输入
            memory: 记忆对象
            history: 对话历史
            
        Returns:
            AgentResponse
        """
        pass
    
    @abstractmethod
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考：分析当前状态，决定下一步
        
        Args:
            context: 当前上下文
            
        Returns:
            思考结果，包含action或completed标记
        """
        pass
    
    @abstractmethod
    async def act(self, action: Dict[str, Any], memory: Any) -> Dict[str, Any]:
        """
        行动：执行工具调用
        
        Args:
            action: 要执行的动作
            memory: 记忆对象
            
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    async def observe(self, result: Dict[str, Any]) -> str:
        """
        观察：分析执行结果
        
        Args:
            result: 执行结果
            
        Returns:
            观察总结
        """
        pass
