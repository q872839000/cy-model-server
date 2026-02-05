"""
反思器

分析执行结果，决定是否需要调整计划。
"""

import json
import re
from typing import Dict, Any, Optional, List
from loguru import logger

from core.services.chat_service import ChatService
from scenario.agent.base import AgentState, PlanStep, StepStatus
from scenario.memory import ScenarioMemory
from prompts import get_prompt_loader
from utils.message_filter import filter_thinking_content


class Reflector:
    """反思器：分析执行结果，决定是否调整计划"""

    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service
        self.prompt_loader = get_prompt_loader()
    
    def _get_prompt_template(self) -> str:
        """获取提示词模板"""
        try:
            return self.prompt_loader.render("scenario/agent_reflector")
        except Exception:
            logger.warning("加载agent_reflector提示词失败，使用默认提示词")
            return self._default_prompt()
    
    def _default_prompt(self) -> str:
        """默认提示词"""
        return '''分析执行结果，返回JSON：
{
    "status": "continue/completed/failed/replan",
    "reason": "原因",
    "message_to_user": ""
}'''
    
    async def reflect(
        self,
        state: AgentState,
        last_result: Dict[str, Any],
        memory: ScenarioMemory,
        user_input: str
    ) -> Dict[str, Any]:
        """
        反思执行结果
        
        Args:
            state: Agent状态
            last_result: 最近执行结果
            memory: 记忆对象
            user_input: 用户原始输入
            
        Returns:
            反思结果
        """
        if last_result.get("needs_input"):
            return {
                "status": "needs_input",
                "question": last_result.get("question"),
                "options": last_result.get("options", [])
            }
        
        if not last_result.get("success"):
            return await self._handle_failure(state, last_result, memory)
        
        completed = [s for s in state.plan if s.status == StepStatus.COMPLETED]
        pending = [s for s in state.plan if s.status == StepStatus.PENDING]
        failed = [s for s in state.plan if s.status == StepStatus.FAILED]
        
        # 如果还有待执行的步骤，直接返回continue（不调用LLM）
        if pending:
            next_step = pending[0]
            logger.debug("【反思器】 还有{}个待执行步骤，继续执行: {}", 
                        len(pending), next_step.description)
            return {
                "status": "continue",
                "reason": f"还有{len(pending)}个步骤待执行"
            }
        
        # 【重要】校验错误检查必须在计划完成检查之前
        # 即使所有步骤都完成了，如果校验失败也需要修复
        validation_status = "未校验"
        if memory.validation_errors:
            validation_status = f"有{len(memory.validation_errors)}个错误"
            # 校验失败，需要修复
            if memory.validation_attempts < 3:  # 最多尝试3次修复
                logger.info("【反思器】 校验失败，需要修复 (第{}次)", memory.validation_attempts)
                return {
                    "status": "replan",
                    "reason": f"校验失败，有{len(memory.validation_errors)}个错误需要修复",
                    "errors": memory.validation_errors
                }
            else:
                logger.warning("【反思器】 修复尝试次数已达上限")
                return {
                    "status": "failed",
                    "reason": "校验失败且修复尝试次数已达上限",
                    "message_to_user": f"想定校验失败，存在以下问题：\n" + "\n".join(
                        [f"- {e.get('message', str(e))}" for e in memory.validation_errors[:5]]
                    )
                }
        elif any(s.tool == "validate_scenario" and s.status == StepStatus.COMPLETED 
                for s in state.plan):
            validation_status = "校验通过"
        
        # 【新增】检查是否有跳过/失败的步骤，如有则检测用户要求是否满足
        skipped = [s for s in state.plan if s.status == StepStatus.SKIPPED]
        if (failed or skipped) and not pending:
            # 有失败/跳过的步骤且无待执行步骤，需要检查用户要求是否满足
            unmet = await self._check_user_requirements(state, memory, user_input)
            if unmet:
                logger.info("【反思器】 检测到用户要求未满足: {}", unmet)
                return {
                    "status": "replan",
                    "reason": f"部分步骤失败/跳过，用户要求未完全满足",
                    "unmet_requirements": unmet,
                    "user_input": user_input
                }
        
        # 计划完成检查（在校验通过之后）
        if state.is_plan_completed():
            return {
                "status": "completed",
                "reason": "所有计划步骤已完成且校验通过"
            }
        
        # 只有在计划状态不明确时才调用LLM判断
        prompt_template = self._get_prompt_template()
        last_tool = state.plan[state.current_step - 1].tool if state.current_step > 0 and state.plan else "无"
        prompt = prompt_template.replace(
            "{{completed_steps}}", str(len(completed))
        ).replace(
            "{{failed_steps}}", str(len(failed))
        ).replace(
            "{{instance_count}}", str(len(memory.current_instances))
        ).replace(
            "{{validation_status}}", validation_status
        ).replace(
            "{{last_tool}}", last_tool
        ).replace(
            "{{last_result}}", json.dumps(last_result, ensure_ascii=False, default=str)[:500]
        ).replace(
            "{{user_input}}", user_input
        )
        
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请分析当前执行状态"}
            ]
            
            response = await self.chat_service.generate(
                model_name=None,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
                enable_thinking=True
            )
            
            # 过滤think内容
            response = self._filter_thinking(response)
            result = self._parse_response(response)
            logger.debug("【反思器】 LLM判断状态: {}, 原因: {}", 
                        result.get("status"), result.get("reason"))
            return result
            
        except Exception as e:
            logger.error("【反思器失败】 {}", str(e))
            return {
                "status": "continue",
                "reason": "反思失败，继续执行"
            }
    
    async def _handle_failure(
        self,
        state: AgentState,
        result: Dict[str, Any],
        memory: ScenarioMemory
    ) -> Dict[str, Any]:
        """处理执行失败"""
        error = result.get("error", "未知错误")
        
        if "out of memory" in error.lower():
            return {
                "status": "failed",
                "reason": "GPU显存不足",
                "message_to_user": "抱歉，系统资源不足，请稍后重试或减少请求复杂度。"
            }
        
        if "not found" in error.lower() or "不存在" in error:
            return {
                "status": "replan",
                "reason": f"资源不存在: {error}",
                "suggestion": "检查参数是否正确"
            }
        
        if state.iteration < 3:
            return {
                "status": "retry",
                "reason": f"执行失败，尝试重试: {error}"
            }
        
        return {
            "status": "failed",
            "reason": error,
            "message_to_user": f"执行失败: {error}"
        }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析反思响应"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()
            if json_str.startswith('```'):
                json_str = re.sub(r'^```\w*\n?', '', json_str)
                json_str = re.sub(r'\n?```$', '', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "status": "continue",
                "reason": "解析失败，默认继续"
            }
    
    def should_validate(self, state: AgentState, memory: ScenarioMemory) -> bool:
        """判断是否应该进行校验"""
        if not memory.current_instances:
            return False
        
        has_validate_step = any(
            s.tool == "validate_scenario" 
            for s in state.plan
        )
        
        if has_validate_step:
            return False
        
        create_steps = [
            s for s in state.plan 
            if s.tool == "create_scenario_instance" and s.status == StepStatus.COMPLETED
        ]
        
        return len(create_steps) > 0
    
    def summarize_progress(self, state: AgentState, memory: ScenarioMemory) -> str:
        """总结当前进度"""
        completed = sum(1 for s in state.plan if s.status == StepStatus.COMPLETED)
        total = len(state.plan)
        
        summary_parts = [f"进度: {completed}/{total} 步骤已完成"]
        
        if memory.current_instances:
            red = sum(1 for i in memory.current_instances if i.get("classify") == "red")
            blue = sum(1 for i in memory.current_instances if i.get("classify") == "blue")
            summary_parts.append(f"实例: 红方{red}个, 蓝方{blue}个")
        
        if memory.validation_errors:
            summary_parts.append(f"校验错误: {len(memory.validation_errors)}个")
        
        return " | ".join(summary_parts)
    
    def _filter_thinking(self, content: str) -> str:
        """过滤响应中的think标签内容"""
        if not content or not isinstance(content, str):
            return content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL)
        return content.strip()
    
    async def _check_user_requirements(
        self,
        state: AgentState,
        memory: ScenarioMemory,
        user_input: str
    ) -> Optional[List[str]]:
        """
        检查用户要求是否满足
        
        Returns:
            未满足的要求列表，如果全部满足则返回None
        """
        # 构建当前状态摘要
        red_count = sum(1 for i in memory.current_instances if i.get("classify") == "red")
        blue_count = sum(1 for i in memory.current_instances if i.get("classify") == "blue")
        
        # 获取失败/跳过的步骤描述
        failed_steps = [s.description for s in state.plan if s.status == StepStatus.FAILED]
        skipped_steps = [s.description for s in state.plan if s.status == StepStatus.SKIPPED]
        
        # 获取已创建的装备类型
        created_types = set()
        for inst in memory.current_instances:
            created_types.add(inst.get("equipName", inst.get("equipId", "未知")))
        
        prompt = f"""分析用户要求是否已满足。

用户原始要求: {user_input}

当前状态:
- 红方实例: {red_count}个
- 蓝方实例: {blue_count}个
- 已创建装备类型: {', '.join(created_types) if created_types else '无'}

执行情况:
- 失败的步骤: {failed_steps if failed_steps else '无'}
- 跳过的步骤: {skipped_steps if skipped_steps else '无'}

请判断用户要求是否已全部满足，返回JSON:
```json
{{
    "satisfied": true/false,
    "unmet_requirements": ["未满足的要求1", "未满足的要求2"]
}}
```

注意: 只关注用户明确提出的要求，不要过度推断。"""

        try:
            messages = [
                {"role": "system", "content": "你是一个需求分析助手，判断用户要求是否被满足。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.chat_service.generate(
                model_name=None,
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
                enable_thinking=True
            )
            
            response = self._filter_thinking(response)
            result = self._parse_response(response)
            
            if not result.get("satisfied", True):
                unmet = result.get("unmet_requirements", [])
                if unmet:
                    logger.debug("【反思器】 用户要求检查: 未满足 {}", unmet)
                    return unmet
            
            logger.debug("【反思器】 用户要求检查: 已满足")
            return None
            
        except Exception as e:
            logger.error("【反思器】 检查用户要求失败: {}", str(e))
            return None
