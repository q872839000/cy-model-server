"""想定生成服务

主服务入口，采用Agent架构。
支持意图理解、动态规划、工具执行、反思调整。
"""
from typing import Dict, Any, Optional, List
from loguru import logger

from core.services.chat_service import ChatService
from models.scenario_schemas import ScenarioResponse
from scenario.constants import DialogPhase
from scenario.tools import ToolExecutor
from scenario.tools.external_api import EXTERNAL_API
from scenario.memory import ScenarioMemory, MEMORY_MANAGER
from scenario.agent import ScenarioAgent


class ScenarioService:
    """
    想定生成服务（Agent架构）
    
    主要职责：
    1. 接收用户输入和历史消息
    2. 通过Agent理解用户意图
    3. 动态规划执行计划
    4. 执行Tool调用
    5. 反思结果，必要时调整
    """
    
    def __init__(self, chat_service: ChatService):
        """
        Args:
            chat_service: LLM对话服务
        """
        self.chat_service = chat_service
        self.tool_executor = ToolExecutor(api_client=EXTERNAL_API)
        self.memory_manager = MEMORY_MANAGER
        self.agent = ScenarioAgent(
            chat_service=chat_service,
            tool_executor=self.tool_executor
        )
        logger.info("ScenarioService 初始化完成 (Agent模式)")
    
    async def chat(
        self,
        user_message: str,
        messages: Optional[List[Dict[str, str]]] = None,
        memory_id: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: int = 15,
    ) -> ScenarioResponse:
        """
        处理用户对话
        
        Args:
            user_message: 当前用户消息
            messages: 历史消息列表（调用方传入）
            memory_id: 记忆ID（None则创建新记忆）
            model: LLM模型名称（暂未使用，Agent内部管理）
            max_iterations: 最大迭代次数（传递给Agent）
            
        Returns:
            ScenarioResponse
        """
        messages = messages or []
        
        # 获取或创建记忆
        memory = self.memory_manager.get_or_create(memory_id)
        
        logger.info("=" * 50)
        logger.info("【想定对话】 memory_id={}", memory.memory_id)
        logger.info("  用户输入: {}", user_message[:100] if len(user_message) > 100 else user_message)
        logger.info("  历史消息: {}条", len(messages))
        
        try:
            # 调用Agent处理
            result = await self.agent.run(
                user_input=user_message,
                memory=memory,
                history=messages
            )
            
            logger.info("【Agent完成】 success={}, needs_input={}", 
                       result.success, result.needs_input)
            
            # 根据Agent结果确定phase
            if result.success and not result.needs_input and not result.needs_confirmation:
                if memory.current_instances:
                    memory.phase = DialogPhase.COMPLETED.value
            
            return ScenarioResponse(
                memory_id=result.memory_id,
                phase=memory.phase,
                message=result.message,
                scenario=result.instances if result.success else None,
                is_completed=memory.phase == DialogPhase.COMPLETED.value,
                validation_attempts=memory.validation_attempts,
                needs_input=result.needs_input,
                needs_confirmation=result.needs_confirmation,
                plan=result.plan,
                thought_process=result.thought_process,
            )
            
        except Exception as e:
            logger.exception("Agent执行失败")
            return ScenarioResponse(
                memory_id=memory.memory_id,
                phase=memory.phase,
                message=f"处理失败: {str(e)}",
                is_completed=False,
            )
    
    async def start(self):
        """启动服务"""
        await self.memory_manager.start()
        logger.info("ScenarioService 启动")
    
    async def stop(self):
        """停止服务"""
        await self.memory_manager.stop()
        logger.info("ScenarioService 停止")
