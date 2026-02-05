"""
想定生成Agent模块

基于ReAct模式的智能体框架，支持：
- 意图解析：理解用户真正想要什么
- 动态规划：根据意图制定执行计划
- 工具执行：调用各种工具完成任务
- 反思调整：根据执行结果调整计划
"""

from scenario.agent.base import (
    AgentState, 
    AgentResponse, 
    BaseAgent,
    Intent,
    PlanStep,
    StepStatus,
)
from scenario.agent.scenario_agent import ScenarioAgent
from scenario.agent.intent_parser import IntentParser
from scenario.agent.planner import Planner
from scenario.agent.executor import AgentExecutor
from scenario.agent.reflector import Reflector

__all__ = [
    # 基础类
    "AgentState",
    "AgentResponse", 
    "BaseAgent",
    "Intent",
    "PlanStep",
    "StepStatus",
    # 主Agent
    "ScenarioAgent",
    # 组件
    "IntentParser",
    "Planner",
    "AgentExecutor",
    "Reflector",
]
