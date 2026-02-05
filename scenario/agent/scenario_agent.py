"""
想定生成Agent

基于ReAct模式的智能体，负责理解用户意图、规划执行、反思调整。
"""

import json
from typing import Dict, Any, Optional, List
from loguru import logger

from core.services.chat_service import ChatService
from scenario.agent.base import (
    BaseAgent, AgentState, AgentResponse, 
    Intent, PlanStep, StepStatus
)
from scenario.agent.intent_parser import IntentParser
from scenario.agent.planner import Planner
from scenario.agent.executor import AgentExecutor
from scenario.agent.reflector import Reflector
from scenario.memory import ScenarioMemory
from scenario.tools import ToolExecutor


class ScenarioAgent(BaseAgent):
    """
    想定生成Agent
    
    采用ReAct模式：Think → Act → Observe → Think → ...
    
    主要流程：
    1. 解析用户意图
    2. 制定执行计划（或询问澄清）
    3. 逐步执行计划
    4. 反思结果，必要时调整
    5. 返回最终结果
    """
    
    MAX_ITERATIONS = 15
    MAX_TOOL_CALLS_PER_ITERATION = 5
    
    def __init__(
        self, 
        chat_service: ChatService, 
        tool_executor: ToolExecutor
    ):
        self.chat_service = chat_service
        self.intent_parser = IntentParser(chat_service)
        self.planner = Planner(chat_service)
        self.executor = AgentExecutor(tool_executor)
        self.reflector = Reflector(chat_service)
    
    async def run(
        self, 
        user_input: str, 
        memory: ScenarioMemory,
        history: Optional[List[Dict[str, str]]] = None
    ) -> AgentResponse:
        """
        执行Agent主循环
        
        Args:
            user_input: 用户输入
            memory: 记忆对象
            history: 对话历史
            
        Returns:
            AgentResponse
        """
        logger.info("=" * 50)
        logger.info("【Agent启动】 用户输入: {}", user_input[:100])
        
        state = AgentState()
        
        # Step 1: 意图解析
        logger.info("【Agent】 Step 1: 解析用户意图")
        intent_result = await self.intent_parser.parse(user_input, memory, history)
        
        state.intent = Intent(intent_result.get("intent", "unclear"))
        state.intent_details = intent_result.get("details", {})
        state.add_thought(f"意图识别: {state.intent.value}, 置信度: {intent_result.get('confidence', 0):.2f}")
        
        # 只有当意图为unclear时才需要澄清
        if state.intent == Intent.UNCLEAR:
            logger.info("【Agent】 意图不明确，需要澄清")
            return AgentResponse(
                message=intent_result.get("clarification_question", "请问您具体想要做什么？"),
                memory_id=memory.memory_id,
                needs_input=True,
                thought_process=state.thoughts
            )
        
        # 如果是确认意图，检查是否有待确认的计划
        if state.intent == Intent.CONFIRM:
            return await self._handle_confirm(state, memory)
        
        # 如果有待确认的方案，且用户想要修改（不是确认）
        if memory.current_proposal and state.intent == Intent.MODIFY:
            return await self._handle_proposal_adjustment(state, user_input, memory)
        
        # 如果是查询意图
        if state.intent == Intent.QUERY:
            return await self._handle_query(state, user_input, memory)
        
        # Step 1.5: 预加载装备库（确保Planner能使用真实equipId）
        if not memory.equipment_library:
            logger.info("【Agent】 预加载装备库")
            lib_result = await self.executor.execute_tool_directly("get_equipment_library", {}, memory)
            if lib_result.get("success"):
                logger.debug("【Agent】 装备库预加载完成: {}个装备", len(memory.equipment_library))
        
        # Step 2: 制定计划
        logger.info("【Agent】 Step 2: 制定执行计划")
        state.plan = await self.planner.create_plan(
            intent=state.intent,
            intent_details=state.intent_details,
            user_input=user_input,
            memory=memory
        )
        
        if not state.plan:
            return AgentResponse(
                message="抱歉，我无法为您的请求制定执行计划。请提供更多细节。",
                memory_id=memory.memory_id,
                needs_input=True,
                thought_process=state.thoughts
            )
        
        state.add_thought(f"计划制定完成，共{len(state.plan)}个步骤")
        logger.info("【Agent】 计划: {}个步骤", len(state.plan))
        
        # 如果用户只想看方案（不立即执行）
        if state.intent == Intent.PLAN_ONLY:
            logger.info("【Agent】 生成部署方案供用户确认")
            # 生成面向用户的部署方案
            proposal = await self.planner.generate_proposal(
                intent_details=state.intent_details,
                user_input=user_input,
                memory=memory
            )
            # 保存方案到记忆中，用于后续确认执行
            memory.current_proposal = proposal
            
            proposal_display = self.planner.format_proposal_for_display(proposal)
            return AgentResponse(
                message=f"{proposal_display}\n\n---\n如需调整请直接说明（如「红方再加2架歼-16」），确认执行请回复「确认」或「执行」。",
                memory_id=memory.memory_id,
                needs_confirmation=True,
                plan=[self._step_to_dict(s) for s in state.plan],
                thought_process=state.thoughts
            )
        
        # Step 3: 执行循环
        logger.info("【Agent】 Step 3: 开始执行循环")
        return await self._execute_loop(state, user_input, memory)
    
    async def _execute_loop(
        self, 
        state: AgentState, 
        user_input: str,
        memory: ScenarioMemory
    ) -> AgentResponse:
        """执行循环"""
        
        for iteration in range(self.MAX_ITERATIONS):
            state.iteration = iteration
            logger.info("【Agent】 迭代 {}/{}", iteration + 1, self.MAX_ITERATIONS)
            
            # Think: 决定下一步
            thought = await self.think({
                "state": state,
                "memory": memory,
                "user_input": user_input
            })
            
            state.add_thought(thought.get("thought", ""))
            
            # 检查是否完成
            if thought.get("completed"):
                logger.info("【Agent】 执行完成")
                return self._build_success_response(state, memory, thought.get("message"))
            
            # 检查是否需要用户输入
            if thought.get("needs_input"):
                return AgentResponse(
                    message=thought.get("question", "请提供更多信息"),
                    memory_id=memory.memory_id,
                    needs_input=True,
                    thought_process=state.thoughts
                )
            
            # 检查是否需要重新规划（参数不完整等情况）
            if thought.get("needs_replan"):
                failed_step = thought.get("failed_step")
                logger.info("【Agent】 步骤参数不完整，调用反思器处理")
                
                # 调用反思器检查用户要求是否满足
                reflection = await self.reflector.reflect(state, {"success": False, "error": thought.get("error")}, memory, user_input)
                
                if reflection.get("status") == "replan":
                    unmet = reflection.get("unmet_requirements", [])
                    if unmet:
                        logger.info("【Agent】 用户要求未满足: {}", unmet)
                        fix_plan = await self._create_supplement_plan(unmet, user_input, memory)
                        if fix_plan:
                            for step in fix_plan:
                                state.plan.append(step)
                            logger.info("【Agent】 添加{}个补充步骤", len(fix_plan))
                elif reflection.get("status") == "completed":
                    return self._build_success_response(state, memory, reflection.get("message_to_user"))
                elif reflection.get("status") == "failed":
                    return AgentResponse(
                        message=reflection.get("message_to_user", "执行失败"),
                        memory_id=memory.memory_id,
                        success=False,
                        thought_process=state.thoughts
                    )
                continue
            
            # Act: 执行动作
            action = thought.get("action")
            if not action:
                current_step = state.get_current_plan_step()
                if current_step:
                    action = {
                        "tool": current_step.tool,
                        "arguments": current_step.arguments,
                        "step_id": current_step.step_id
                    }
                else:
                    logger.info("【Agent】 无更多步骤，执行完成")
                    return self._build_success_response(state, memory)
            
            result = await self.act(action, memory)
            
            # 处理需要用户输入的情况
            if result.get("needs_input"):
                return AgentResponse(
                    message=result.get("question", "请回答以下问题"),
                    memory_id=memory.memory_id,
                    needs_input=True,
                    thought_process=state.thoughts
                )
            
            # Observe: 分析结果
            observation = await self.observe(result)
            state.add_observation(observation)
            
            # 更新步骤状态
            if action.get("step_id"):
                if result.get("success"):
                    state.mark_step_completed(action["step_id"], result.get("data"))
                else:
                    state.mark_step_failed(action["step_id"], result.get("error", ""))
            
            state.current_step += 1
            
            # Reflect: 反思结果
            reflection = await self.reflector.reflect(state, result, memory, user_input)
            
            if reflection.get("status") == "completed":
                return self._build_success_response(state, memory, reflection.get("message_to_user"))
            
            if reflection.get("status") == "failed":
                return AgentResponse(
                    message=reflection.get("message_to_user", "执行失败，请重试"),
                    memory_id=memory.memory_id,
                    success=False,
                    thought_process=state.thoughts
                )
            
            if reflection.get("status") == "needs_input":
                return AgentResponse(
                    message=reflection.get("question", "请提供更多信息"),
                    memory_id=memory.memory_id,
                    needs_input=True,
                    thought_process=state.thoughts
                )
            
            # 处理replan：校验失败或用户要求未满足
            if reflection.get("status") == "replan":
                errors = reflection.get("errors", [])
                unmet = reflection.get("unmet_requirements", [])
                
                if errors:
                    # 校验错误修复
                    logger.info("【Agent】 需要修复校验错误: {}个", len(errors))
                    state.add_thought(f"校验失败，需要修复{len(errors)}个错误")
                    fix_plan = await self._create_fix_plan(errors, memory)
                elif unmet:
                    # 用户要求未满足，重新规划
                    logger.info("【Agent】 用户要求未满足: {}", unmet)
                    state.add_thought(f"部分要求未满足: {', '.join(unmet)}")
                    fix_plan = await self._create_supplement_plan(unmet, reflection.get("user_input", ""), memory)
                else:
                    fix_plan = None
                
                if fix_plan:
                    for step in fix_plan:
                        state.plan.append(step)
                    logger.info("【Agent】 添加{}个补充步骤", len(fix_plan))
                else:
                    logger.warning("【Agent】 无法生成补充计划")
        
        # 达到最大迭代次数
        logger.warning("【Agent】 达到最大迭代次数")
        return AgentResponse(
            message="执行步骤过多，已中止。当前已创建{}个实例。".format(len(memory.current_instances)),
            memory_id=memory.memory_id,
            success=False,
            instances=memory.current_instances,
            thought_process=state.thoughts
        )
    
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考：分析当前状态，决定下一步
        """
        state: AgentState = context.get("state")
        memory: ScenarioMemory = context.get("memory")
        
        # 检查计划是否完成
        if state.is_plan_completed():
            # 检查是否需要校验
            if self.reflector.should_validate(state, memory):
                return {
                    "thought": "计划执行完成，需要进行校验",
                    "action": {
                        "tool": "validate_scenario",
                        "arguments": {}
                    }
                }
            return {
                "completed": True,
                "thought": "所有步骤已完成",
                "message": self._generate_completion_message(state, memory)
            }
        
        # 检查是否有失败的步骤
        if state.has_failed_steps():
            failed = [s for s in state.plan if s.status == StepStatus.FAILED]
            return {
                "thought": f"有{len(failed)}个步骤失败，需要处理",
                "needs_replan": True
            }
        
        # 获取下一个待执行的步骤
        next_step = state.get_current_plan_step()
        if next_step:
            # 验证参数
            valid, error = self.executor.validate_step_arguments(next_step)
            if not valid:
                # 参数不完整时标记为失败，触发反思器的replan机制
                logger.warning("【Agent】 步骤{}参数不完整: {}", next_step.step_id, error)
                next_step.status = StepStatus.FAILED
                next_step.error = f"参数不完整: {error}"
                state.add_thought(f"步骤{next_step.step_id}失败: {error}")
                # 返回需要重新规划的信号
                return {
                    "thought": f"步骤{next_step.step_id}参数不完整，需要调整计划",
                    "needs_replan": True,
                    "failed_step": next_step,
                    "error": error
                }
            
            return {
                "thought": f"执行步骤{next_step.step_id}: {next_step.description}",
                "action": {
                    "tool": next_step.tool,
                    "arguments": next_step.arguments,
                    "step_id": next_step.step_id
                }
            }
        
        return {
            "completed": True,
            "thought": "无更多待执行步骤"
        }
    
    async def act(self, action: Dict[str, Any], memory: ScenarioMemory) -> Dict[str, Any]:
        """
        行动：执行工具调用
        """
        tool_name = action.get("tool")
        arguments = action.get("arguments", {})
        
        logger.debug("【Agent Act】 工具: {}, 参数: {}", tool_name, arguments)
        
        result = await self.executor.execute_tool_directly(
            tool_name=tool_name,
            arguments=arguments,
            memory=memory
        )
        
        return result
    
    async def observe(self, result: Dict[str, Any]) -> str:
        """
        观察：分析执行结果
        """
        if result.get("success"):
            data = result.get("data", {})
            message = result.get("message", "执行成功")
            
            if isinstance(data, list):
                return f"成功: {message} (返回{len(data)}条数据)"
            elif isinstance(data, dict):
                return f"成功: {message}"
            else:
                return f"成功: {message}"
        else:
            error = result.get("error", "未知错误")
            return f"失败: {error}"
    
    async def _handle_confirm(
        self, 
        state: AgentState, 
        memory: ScenarioMemory
    ) -> AgentResponse:
        """处理确认意图 - 执行之前保存的部署方案"""
        # 检查是否有待确认的部署方案
        if not memory.current_proposal:
            logger.warning("【Agent】 没有待确认的方案")
            return AgentResponse(
                message="当前没有待确认的部署方案。请先描述您想要部署的场景。",
                memory_id=memory.memory_id,
                needs_input=True,
                thought_process=state.thoughts
            )
        
        proposal = memory.current_proposal
        logger.info("【Agent】 确认执行方案: {}", proposal.get("title", "未命名"))
        state.add_thought(f"用户确认执行方案: {proposal.get('title', '未命名')}")
        
        # 确保装备库已加载
        if not memory.equipment_library:
            logger.info("【Agent】 预加载装备库")
            await self.executor.execute_tool_directly("get_equipment_library", {}, memory)
        
        # 根据方案生成执行计划
        state.plan = self.planner.create_plan_from_proposal(proposal, memory)
        
        if not state.plan:
            return AgentResponse(
                message="无法从方案生成执行计划，请重新描述您的需求。",
                memory_id=memory.memory_id,
                needs_input=True,
                thought_process=state.thoughts
            )
        
        logger.info("【Agent】 从方案生成{}个执行步骤，开始执行", len(state.plan))
        state.add_thought(f"从方案生成{len(state.plan)}个执行步骤")
        
        # 清除已确认的方案
        memory.current_proposal = None
        
        # 执行计划
        return await self._execute_loop(state, f"执行方案: {proposal.get('title', '')}", memory)
    
    async def _handle_proposal_adjustment(
        self,
        state: AgentState,
        user_input: str,
        memory: ScenarioMemory
    ) -> AgentResponse:
        """处理用户对方案的调整请求"""
        logger.info("【Agent】 用户请求调整方案: {}", user_input[:50])
        state.add_thought(f"用户请求调整方案: {user_input}")
        
        # 确保装备库已加载
        if not memory.equipment_library:
            await self.executor.execute_tool_directly("get_equipment_library", {}, memory)
        
        # 调用Planner调整方案
        adjusted_proposal = await self.planner.adjust_proposal(
            current_proposal=memory.current_proposal,
            adjustment_request=user_input,
            memory=memory
        )
        
        # 保存调整后的方案
        memory.current_proposal = adjusted_proposal
        
        proposal_display = self.planner.format_proposal_for_display(adjusted_proposal)
        return AgentResponse(
            message=f"已根据您的要求调整方案：\n\n{proposal_display}\n\n---\n如需继续调整请直接说明，确认执行请回复「确认」或「执行」。",
            memory_id=memory.memory_id,
            needs_confirmation=True,
            thought_process=state.thoughts
        )
    
    async def _handle_query(
        self, 
        state: AgentState, 
        user_input: str,
        memory: ScenarioMemory
    ) -> AgentResponse:
        """处理查询意图"""
        # 检查是否需要获取装备库
        if not memory.equipment_library:
            result = await self.executor.execute_tool_directly(
                tool_name="get_equipment_library",
                arguments={},
                memory=memory
            )
        
        # 构建查询响应
        if memory.equipment_library:
            equip_summary = self._summarize_equipment(memory.equipment_library)
            message = f"当前装备库共有{len(memory.equipment_library)}种装备：\n\n{equip_summary}"
        else:
            message = "装备库为空，请稍后重试。"
        
        if memory.current_instances:
            message += f"\n\n当前已部署{len(memory.current_instances)}个实例。"
        
        return AgentResponse(
            message=message,
            memory_id=memory.memory_id,
            thought_process=state.thoughts
        )
    
    def _build_success_response(
        self, 
        state: AgentState, 
        memory: ScenarioMemory,
        message: Optional[str] = None
    ) -> AgentResponse:
        """构建成功响应"""
        if not message:
            message = self._generate_completion_message(state, memory)
        
        return AgentResponse(
            message=message,
            memory_id=memory.memory_id,
            success=True,
            instances=memory.current_instances,
            plan=[self._step_to_dict(s) for s in state.plan],
            thought_process=state.thoughts
        )
    
    def _generate_completion_message(
        self, 
        state: AgentState, 
        memory: ScenarioMemory
    ) -> str:
        """生成完成消息"""
        parts = []
        
        if memory.current_instances:
            red = sum(1 for i in memory.current_instances if i.get("classify") == "red")
            blue = sum(1 for i in memory.current_instances if i.get("classify") == "blue")
            green = sum(1 for i in memory.current_instances if i.get("classify") == "green")
            
            parts.append(f"想定生成完成！共创建{len(memory.current_instances)}个装备实例：")
            if red:
                parts.append(f"- 红方: {red}个")
            if blue:
                parts.append(f"- 蓝方: {blue}个")
            if green:
                parts.append(f"- 绿方: {green}个")
        else:
            parts.append("操作已完成。")
        
        if memory.validation_errors:
            parts.append(f"\n⚠️ 校验发现{len(memory.validation_errors)}个问题，可能需要调整。")
        
        return "\n".join(parts)
    
    def _summarize_equipment(self, equipment_list: List[Dict]) -> str:
        """总结装备列表"""
        if not equipment_list:
            return "无"
        
        by_type = {}
        for eq in equipment_list[:20]:
            eq_type = eq.get("type", "其他")
            if eq_type not in by_type:
                by_type[eq_type] = []
            by_type[eq_type].append(eq.get("name", eq.get("equipId", "未知")))
        
        lines = []
        for eq_type, names in by_type.items():
            names_str = "、".join(names[:5])
            if len(names) > 5:
                names_str += f"等{len(names)}种"
            lines.append(f"- {eq_type}: {names_str}")
        
        return "\n".join(lines)
    
    async def _create_fix_plan(
        self, 
        errors: List[Dict[str, Any]], 
        memory: ScenarioMemory
    ) -> List[PlanStep]:
        """
        根据校验错误创建修复计划
        """
        fix_steps = []
        next_step_id = 100
        
        # 分析错误类型（去重）
        has_position_error = False
        for error in errors:
            error_msg = error.get("message", str(error))
            if "位置" in error_msg or "lon" in error_msg.lower() or "lat" in error_msg.lower():
                has_position_error = True
                break
        
        # 位置信息缺失 - 需要解析位置或使用已有位置
        if has_position_error:
            if memory.allocated_positions:
                # 有已分配位置，直接修复
                pos = memory.allocated_positions[-1]
                fix_steps.append(PlanStep(
                    step_id=next_step_id,
                    description="批量修复所有实例的位置信息",
                    tool="modify_instances",
                    arguments={
                        "filter": {"all": True},
                        "updates": {
                            "port_id": "position",
                            "values": {"lon": pos.get("lon"), "lat": pos.get("lat"), "alt": pos.get("alt", 10000)}
                        }
                    }
                ))
                next_step_id += 1
            else:
                # 没有已分配位置，需要先解析位置（使用默认位置描述）
                logger.warning("【修复计划】 缺少位置信息且无已分配位置，使用默认位置")
                fix_steps.append(PlanStep(
                    step_id=next_step_id,
                    description="解析默认位置",
                    tool="resolve_location",
                    arguments={"location_description": "中国东部海域"}
                ))
                next_step_id += 1
                # 解析后再修复位置
                fix_steps.append(PlanStep(
                    step_id=next_step_id,
                    description="批量修复所有实例的位置信息",
                    tool="modify_instances",
                    arguments={
                        "filter": {"all": True},
                        "updates": {
                            "port_id": "position",
                            "values": {}  # 将在执行时从memory获取
                        }
                    },
                    depends_on=[next_step_id - 1]
                ))
                next_step_id += 1
        
        # 添加重新校验步骤
        if fix_steps:
            fix_steps.append(PlanStep(
                step_id=next_step_id,
                description="重新校验想定",
                tool="validate_scenario",
                arguments={}
            ))
        
        return fix_steps
    
    async def _create_supplement_plan(
        self,
        unmet_requirements: List[str],
        user_input: str,
        memory: ScenarioMemory
    ) -> List[PlanStep]:
        """
        根据未满足的用户要求创建补充计划
        """
        # 使用Planner重新规划未满足的需求
        supplement_input = f"补充完成以下要求: {', '.join(unmet_requirements)}"
        
        # 调用planner生成补充计划
        try:
            from scenario.agent.base import Intent
            supplement_plan = await self.planner.create_plan(
                intent=Intent.MODIFY,
                intent_details={"unmet_requirements": unmet_requirements},
                user_input=supplement_input,
                memory=memory
            )
            
            # 调整step_id避免冲突
            max_id = max((s.step_id for s in memory.current_instances), default=100) if hasattr(memory, 'current_instances') else 100
            for i, step in enumerate(supplement_plan):
                step.step_id = max_id + 100 + i + 1
            
            logger.info("【Agent】 生成{}个补充步骤", len(supplement_plan))
            return supplement_plan
            
        except Exception as e:
            logger.error("【Agent】 生成补充计划失败: {}", str(e))
            return []
    
    def _step_to_dict(self, step: PlanStep) -> Dict[str, Any]:
        """将步骤转换为字典"""
        return {
            "step_id": step.step_id,
            "description": step.description,
            "tool": step.tool,
            "arguments": step.arguments,
            "status": step.status.value,
            "error": step.error
        }
