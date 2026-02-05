"""
执行器

执行计划中的工具调用。
"""

from typing import Dict, Any, Optional
from loguru import logger

from scenario.agent.base import PlanStep, StepStatus
from scenario.memory import ScenarioMemory
from scenario.tools import ToolExecutor
from scenario.generators import InstanceNameGenerator


class AgentExecutor:
    """Agent执行器：执行工具调用"""
    
    def __init__(self, tool_executor: ToolExecutor):
        self.tool_executor = tool_executor
        self.name_generator = InstanceNameGenerator()
    
    async def execute_step(
        self, 
        step: PlanStep, 
        memory: ScenarioMemory
    ) -> Dict[str, Any]:
        """
        执行单个计划步骤
        
        Args:
            step: 计划步骤
            memory: 记忆对象
            
        Returns:
            执行结果
        """
        logger.info("【执行器】 执行步骤{}: {} ({})", 
                   step.step_id, step.description, step.tool)
        
        step.status = StepStatus.RUNNING
        
        try:
            if step.tool == "ask_user":
                return await self._handle_ask_user(step)
            
            result = await self.tool_executor.execute(
                tool_name=step.tool,
                arguments=step.arguments,
                memory=memory,
                name_generator=self.name_generator
            )
            
            if result.success:
                step.status = StepStatus.COMPLETED
                step.result = result.data
                logger.info("【执行器】 步骤{}成功: {}", step.step_id, result.message)
                return {
                    "success": True,
                    "data": result.data,
                    "message": result.message
                }
            else:
                step.status = StepStatus.FAILED
                step.error = result.error
                logger.warning("【执行器】 步骤{}失败: {}", step.step_id, result.error)
                return {
                    "success": False,
                    "error": result.error
                }
                
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            logger.error("【执行器】 步骤{}异常: {}", step.step_id, str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_ask_user(self, step: PlanStep) -> Dict[str, Any]:
        """处理询问用户的特殊工具"""
        question = step.arguments.get("question", "请问您有什么具体需求？")
        options = step.arguments.get("options", [])
        
        step.status = StepStatus.COMPLETED
        
        return {
            "success": True,
            "needs_input": True,
            "question": question,
            "options": options,
            "data": {
                "type": "ask_user",
                "question": question,
                "options": options
            }
        }
    
    async def execute_tool_directly(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        memory: ScenarioMemory
    ) -> Dict[str, Any]:
        """
        直接执行工具（不通过计划步骤）
        
        用于动态决策时的工具调用
        """
        logger.debug("【执行器】 直接执行工具: {}", tool_name)
        
        try:
            result = await self.tool_executor.execute(
                tool_name=tool_name,
                arguments=arguments,
                memory=memory,
                name_generator=self.name_generator
            )
            
            return {
                "success": result.success,
                "data": result.data if result.success else None,
                "error": result.error if not result.success else None,
                "message": result.message
            }
            
        except Exception as e:
            logger.error("【执行器】 工具执行异常: {}", str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_step_arguments(self, step: PlanStep) -> tuple[bool, str]:
        """
        验证步骤参数是否完整
        
        Returns:
            (是否有效, 错误信息)
        """
        tool = step.tool
        args = step.arguments
        
        required_args = {
            "get_equipment_library": [],
            "get_equipment_schema": ["equip_ids"],
            "resolve_location": ["location_description"],
            "create_scenario_instance": ["equip_id", "classify"],
            "modify_instances": ["filter", "updates"],
            "delete_instances": ["filter"],
            "validate_scenario": [],
            "ask_user": ["question"],
        }
        
        if tool not in required_args:
            return False, f"未知工具: {tool}"
        
        missing = [arg for arg in required_args[tool] if arg not in args]
        if missing:
            return False, f"缺少必要参数: {missing}"
        
        return True, ""
