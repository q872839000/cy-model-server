"""
规划器

根据用户意图制定执行计划。
"""

import json
import re
from typing import Dict, Any, Optional, List
from loguru import logger

from core.services.chat_service import ChatService
from scenario.agent.base import Intent, PlanStep, StepStatus
from scenario.memory import ScenarioMemory
from prompts import get_prompt_loader
from utils.message_filter import filter_thinking_content


class Planner:
    """规划器：根据意图制定执行计划"""

    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service
        self.prompt_loader = get_prompt_loader()
    
    def _get_prompt_template(self) -> str:
        """获取提示词模板"""
        try:
            return self.prompt_loader.render("scenario/agent_planner")
        except Exception:
            logger.warning("加载agent_planner提示词失败，使用默认提示词")
            return self._default_prompt()
    
    def _default_prompt(self) -> str:
        """默认提示词"""
        return '''根据用户需求制定执行计划，返回JSON：
{
    "summary": "计划概述",
    "steps": [{"step_id": 1, "description": "", "tool": "", "arguments": {}}]
}'''
    
    async def create_plan(
        self,
        intent: Intent,
        intent_details: Dict[str, Any],
        user_input: str,
        memory: ScenarioMemory
    ) -> List[PlanStep]:
        """
        制定执行计划
        
        Args:
            intent: 用户意图
            intent_details: 意图详情
            user_input: 用户原始输入
            memory: 当前记忆状态
            
        Returns:
            计划步骤列表
        """
        if intent == Intent.MODIFY:
            return await self._create_modify_plan(intent_details, user_input, memory)
        elif intent in (Intent.CREATE, Intent.PLAN_ONLY):
            # PLAN_ONLY 和 CREATE 使用相同的计划生成逻辑，区别在于是否立即执行
            return await self._create_new_plan(intent_details, user_input, memory)
        elif intent == Intent.QUERY:
            return await self._create_query_plan(intent_details, user_input, memory)
        else:
            return []
    
    async def _create_new_plan(
        self,
        intent_details: Dict[str, Any],
        user_input: str,
        memory: ScenarioMemory
    ) -> List[PlanStep]:
        """创建新想定的计划"""
        # 构建装备库摘要（供LLM选择真实装备）
        equipment_summary = []
        for eq in memory.equipment_library[:20]:  # 限制数量
            equipment_summary.append({
                "equipId": eq.get("equipId"),
                "name": eq.get("equipName"),
                "classifies": eq.get("classifies", [])
            })
        
        current_state = {
            "has_equipment_library": len(memory.equipment_library) > 0,
            "equipment_count": len(memory.equipment_library),
            "equipment_list": equipment_summary,
            "has_location": len(memory.allocated_positions) > 0,
            "allocated_positions": memory.allocated_positions[:3] if memory.allocated_positions else [],
            "instance_count": len(memory.current_instances)
        }
        
        # 加载并填充提示词模板
        prompt_template = self._get_prompt_template()
        prompt = prompt_template.replace(
            "{{current_state}}", json.dumps(current_state, ensure_ascii=False)
        ).replace(
            "{{intent}}", Intent.CREATE.value
        ).replace(
            "{{intent_details}}", json.dumps(intent_details, ensure_ascii=False)
        ).replace(
            "{{user_input}}", user_input
        )
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"请为以下需求制定计划：{user_input}"}
        ]
        
        try:
            # 使用支持自动续接的生成方法
            response = await self._generate_with_continuation(messages, max_tokens=4000)
            logger.debug("【规划器】 最终响应: {}", response[:500] if response else "空")
            
            plan_data = self._parse_plan_response(response)
            steps = self._build_plan_steps(plan_data, memory)
            
            logger.info("【规划器】 生成{}个步骤: {}", len(steps), plan_data.get("summary", ""))
            return steps
            
        except Exception as e:
            logger.error("【规划器失败】 {}", str(e))
            return self._create_default_plan(memory)
    
    async def _create_modify_plan(
        self,
        intent_details: Dict[str, Any],
        user_input: str,
        memory: ScenarioMemory
    ) -> List[PlanStep]:
        """创建修改计划"""
        red_count = sum(1 for i in memory.current_instances if i.get("classify") == "red")
        blue_count = sum(1 for i in memory.current_instances if i.get("classify") == "blue")
        
        type_dist = {}
        for inst in memory.current_instances:
            etype = inst.get("equipName", "unknown")
            type_dist[etype] = type_dist.get(etype, 0) + 1
        
        # 构建装备库摘要（供LLM选择真实装备）
        equipment_summary = []
        for eq in memory.equipment_library[:20]:
            equipment_summary.append({
                "equipId": eq.get("equipId"),
                "name": eq.get("equipName"),
                "classifies": eq.get("classifies", [])
            })
        
        current_state = {
            "instance_count": len(memory.current_instances),
            "red_count": red_count,
            "blue_count": blue_count,
            "type_distribution": type_dist,
            "equipment_list": equipment_summary,
            "allocated_positions": memory.allocated_positions[:3] if memory.allocated_positions else []
        }
        
        # 加载并填充提示词模板
        prompt_template = self._get_prompt_template()
        prompt = prompt_template.replace(
            "{{current_state}}", json.dumps(current_state, ensure_ascii=False)
        ).replace(
            "{{intent}}", Intent.MODIFY.value
        ).replace(
            "{{intent_details}}", json.dumps(intent_details, ensure_ascii=False)
        ).replace(
            "{{user_input}}", user_input
        )
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
        
        try:
            # 使用支持自动续接的生成方法
            response = await self._generate_with_continuation(messages, max_tokens=4000)
            
            plan_data = self._parse_plan_response(response)
            steps = self._build_plan_steps(plan_data, memory)
            
            logger.info("【规划器】 修改计划{}个步骤", len(steps))
            return steps
            
        except Exception as e:
            logger.error("【规划器失败】 {}", str(e))
            return []
    
    async def _create_query_plan(
        self,
        intent_details: Dict[str, Any],
        user_input: str,
        memory: ScenarioMemory
    ) -> List[PlanStep]:
        """创建查询计划"""
        steps = []
        
        if not memory.equipment_library:
            steps.append(PlanStep(
                step_id=1,
                description="获取装备库",
                tool="get_equipment_library",
                arguments={}
            ))
        
        return steps
    
    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """解析计划响应"""
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
            logger.warning("【规划器】 JSON解析失败")
            return {"summary": "解析失败", "steps": []}
    
    def _build_plan_steps(
        self, 
        plan_data: Dict[str, Any],
        memory: ScenarioMemory
    ) -> List[PlanStep]:
        """构建计划步骤对象"""
        steps = []
        raw_steps = plan_data.get("steps", [])
        
        for raw in raw_steps:
            step = PlanStep(
                step_id=raw.get("step_id", len(steps) + 1),
                description=raw.get("description", ""),
                tool=raw.get("tool", ""),
                arguments=raw.get("arguments", {}),
                depends_on=raw.get("depends_on", []),
                status=StepStatus.PENDING
            )
            steps.append(step)
        
        return steps
    
    def _create_default_plan(self, memory: ScenarioMemory) -> List[PlanStep]:
        """创建默认计划"""
        steps = []
        step_id = 1
        
        if not memory.equipment_library:
            steps.append(PlanStep(
                step_id=step_id,
                description="获取装备库列表",
                tool="get_equipment_library",
                arguments={}
            ))
            step_id += 1
        
        steps.append(PlanStep(
            step_id=step_id,
            description="询问用户具体需求",
            tool="ask_user",
            arguments={
                "question": "请问您想要创建什么类型的想定？需要哪些装备？",
                "options": ["红蓝对抗", "海上作战", "空中打击", "其他"]
            }
        ))
        
        return steps
    
    async def replan(
        self,
        current_state: Dict[str, Any],
        failed_step: PlanStep,
        error: str,
        memory: ScenarioMemory
    ) -> List[PlanStep]:
        """重新规划（当步骤失败时）"""
        logger.info("【规划器】 重新规划，失败步骤: {}", failed_step.description)
        
        return []
    
    def format_plan_for_display(self, steps: List[PlanStep]) -> str:
        """格式化计划用于展示"""
        if not steps:
            return "暂无计划"
        
        lines = ["## 执行计划\n"]
        for step in steps:
            status_icon = {
                StepStatus.PENDING: "⏳",
                StepStatus.RUNNING: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }.get(step.status, "❓")
            
            lines.append(f"{status_icon} **步骤{step.step_id}**: {step.description}")
            lines.append(f"   工具: `{step.tool}`")
            if step.arguments:
                lines.append(f"   参数: `{json.dumps(step.arguments, ensure_ascii=False)}`")
            lines.append("")
        
        return "\n".join(lines)
    
    def _filter_thinking(self, content: str) -> str:
        """过滤响应中的think标签内容"""
        if not content or not isinstance(content, str):
            return content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL)
        return content.strip()
    
    def _is_json_complete(self, text: str) -> bool:
        """检测JSON是否完整"""
        if not text:
            return False
        try:
            # 提取JSON块
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json.loads(json_match.group(1))
                return True
            # 尝试直接解析
            json_str = text.strip()
            if json_str.startswith('{'):
                json.loads(json_str)
                return True
            return False
        except json.JSONDecodeError:
            return False
    
    async def _generate_with_continuation(
        self, 
        messages: list, 
        max_tokens: int = 4000,
        max_retries: int = 3
    ) -> str:
        """
        支持自动续接的生成方法
        当JSON输出不完整时，自动让LLM继续生成
        """
        full_response = ""
        current_messages = messages.copy()
        
        for attempt in range(max_retries):
            response = await self.chat_service.generate(
                model_name=None,
                messages=current_messages,
                temperature=0.2,
                max_tokens=max_tokens,
                enable_thinking=True
            )
            
            # 过滤thinking
            filtered = self._filter_thinking(response)
            
            if attempt == 0:
                full_response = filtered
            else:
                # 续接时拼接内容
                full_response += filtered
            
            # 检测JSON是否完整
            if self._is_json_complete(full_response):
                logger.debug("【规划器】 JSON完整，生成成功")
                return full_response
            
            if attempt < max_retries - 1:
                # JSON不完整，构造续接请求
                logger.debug("【规划器】 JSON不完整，尝试续接 ({}/{})", attempt + 1, max_retries)
                current_messages = [
                    {"role": "assistant", "content": full_response},
                    {"role": "user", "content": "JSON输出不完整，请从断点处继续完成，只输出剩余的JSON内容"}
                ]
        
        logger.warning("【规划器】 达到最大重试次数，返回可能不完整的响应")
        return full_response
    
    async def generate_proposal(
        self,
        intent_details: Dict[str, Any],
        user_input: str,
        memory: ScenarioMemory
    ) -> Dict[str, Any]:
        """
        生成部署方案（面向用户展示）
        
        Args:
            intent_details: 意图详情
            user_input: 用户原始输入
            memory: 当前记忆状态
            
        Returns:
            部署方案字典
        """
        # 构建装备库摘要
        equipment_summary = []
        for eq in memory.equipment_library:
            equipment_summary.append({
                "equipId": eq.get("equipId"),
                "name": eq.get("equipName"),
                "classifies": eq.get("classifies", [])
            })
        
        # 提取用户需求信息
        location = intent_details.get("location", "未指定")
        factions = intent_details.get("factions", ["red", "blue"])
        special_requirements = intent_details.get("special_requirements", [])
        
        # 加载提示词模板
        try:
            prompt_template = self.prompt_loader.render("scenario/agent_proposal")
        except Exception:
            logger.warning("加载agent_proposal提示词失败")
            return self._default_proposal(location, factions)
        
        # 填充模板
        prompt = prompt_template.replace(
            "{{equipment_library}}", json.dumps(equipment_summary, ensure_ascii=False, indent=2)
        ).replace(
            "{{location}}", location
        ).replace(
            "{{factions}}", json.dumps(factions, ensure_ascii=False)
        ).replace(
            "{{special_requirements}}", json.dumps(special_requirements, ensure_ascii=False)
        ).replace(
            "{{user_input}}", user_input
        )
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"请为以下需求生成部署方案：{user_input}"}
        ]
        
        try:
            response = await self._generate_with_continuation(messages, max_tokens=3000)
            proposal = self._parse_proposal_response(response)
            logger.info("【规划器】 生成部署方案: {}", proposal.get("title", "未命名"))
            return proposal
        except Exception as e:
            logger.error("【规划器】 生成部署方案失败: {}", str(e))
            return self._default_proposal(location, factions)
    
    async def adjust_proposal(
        self,
        current_proposal: Dict[str, Any],
        adjustment_request: str,
        memory: ScenarioMemory
    ) -> Dict[str, Any]:
        """
        根据用户请求调整部署方案
        
        Args:
            current_proposal: 当前方案
            adjustment_request: 用户的调整请求
            memory: 当前记忆状态
            
        Returns:
            调整后的方案
        """
        # 构建装备库摘要
        equipment_summary = []
        for eq in memory.equipment_library:
            equipment_summary.append({
                "equipId": eq.get("equipId"),
                "name": eq.get("equipName"),
                "classifies": eq.get("classifies", [])
            })
        
        prompt = f"""你是一个军事想定专家。请根据用户的调整请求，修改当前的部署方案。

## 当前方案
```json
{json.dumps(current_proposal, ensure_ascii=False, indent=2)}
```

## 可用装备库
```json
{json.dumps(equipment_summary, ensure_ascii=False, indent=2)}
```

## 用户的调整请求
{adjustment_request}

## 要求
1. 理解用户的调整意图（增加、减少、删除、替换装备等）
2. 在当前方案基础上进行修改
3. 保持方案结构不变
4. 只能使用装备库中存在的装备

## 输出格式
直接输出修改后的完整方案JSON，格式与当前方案相同。只输出JSON，不要其他内容。
"""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"请根据以下要求调整方案：{adjustment_request}"}
        ]
        
        try:
            response = await self._generate_with_continuation(messages, max_tokens=3000)
            adjusted = self._parse_proposal_response(response)
            logger.info("【规划器】 方案已调整: {}", adjusted.get("title", "未命名"))
            return adjusted
        except Exception as e:
            logger.error("【规划器】 调整方案失败: {}", str(e))
            return current_proposal
    
    def _parse_proposal_response(self, response: str) -> Dict[str, Any]:
        """解析部署方案响应"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("【规划器】 部署方案JSON解析失败")
            return {"title": "方案解析失败", "summary": response[:200]}
    
    def _default_proposal(self, location: str, factions: List[str]) -> Dict[str, Any]:
        """默认部署方案"""
        return {
            "title": f"{location}部署方案",
            "summary": "基础红蓝对抗配置",
            "location": {"name": location, "description": location},
            "red_forces": [],
            "blue_forces": [],
            "tactical_notes": "请补充具体装备配置"
        }
    
    def create_plan_from_proposal(
        self,
        proposal: Dict[str, Any],
        memory: ScenarioMemory
    ) -> List[PlanStep]:
        """
        根据部署方案生成执行计划
        
        Args:
            proposal: 部署方案
            memory: 当前记忆状态
            
        Returns:
            执行计划步骤列表
        """
        steps = []
        step_id = 1
        
        # Step 1: 解析位置
        location = proposal.get("location", {})
        location_name = location.get("name", "未指定位置")
        if location_name and location_name != "未指定位置":
            steps.append(PlanStep(
                step_id=step_id,
                description=f"解析{location_name}的地理坐标",
                tool="resolve_location",
                arguments={"location_description": location_name},
                depends_on=[]
            ))
            step_id += 1
        
        # Step 2: 创建红方装备
        red_forces = proposal.get("red_forces", [])
        for force in red_forces:
            equip_id = force.get("equip_id")
            equip_name = force.get("equip_name", equip_id)
            count = force.get("count", 1)
            if equip_id:
                steps.append(PlanStep(
                    step_id=step_id,
                    description=f"部署红方{equip_name} × {count}",
                    tool="create_scenario_instance",
                    arguments={
                        "equip_id": equip_id,
                        "classify": "red",
                        "count": count,
                        "params": {}
                    },
                    depends_on=[1] if location_name != "未指定位置" else []
                ))
                step_id += 1
        
        # Step 3: 创建蓝方装备
        blue_forces = proposal.get("blue_forces", [])
        for force in blue_forces:
            equip_id = force.get("equip_id")
            equip_name = force.get("equip_name", equip_id)
            count = force.get("count", 1)
            if equip_id:
                steps.append(PlanStep(
                    step_id=step_id,
                    description=f"部署蓝方{equip_name} × {count}",
                    tool="create_scenario_instance",
                    arguments={
                        "equip_id": equip_id,
                        "classify": "blue",
                        "count": count,
                        "params": {}
                    },
                    depends_on=[1] if location_name != "未指定位置" else []
                ))
                step_id += 1
        
        # Step 4: 校验场景
        if steps:
            steps.append(PlanStep(
                step_id=step_id,
                description="校验想定合法性",
                tool="validate_scenario",
                arguments={},
                depends_on=[s.step_id for s in steps[:-1] if s.step_id > 1] if len(steps) > 1 else []
            ))
        
        logger.info("【规划器】 从方案生成{}个执行步骤", len(steps))
        return steps
    
    def format_proposal_for_display(self, proposal: Dict[str, Any]) -> str:
        """格式化部署方案用于展示"""
        lines = []
        
        # 标题
        title = proposal.get("title", "部署方案")
        lines.append(f"## {title}\n")
        
        # 概述
        summary = proposal.get("summary", "")
        if summary:
            lines.append(f"{summary}\n")
        
        # 位置信息
        location = proposal.get("location", {})
        if location:
            loc_name = location.get("name", "未指定")
            lines.append(f"**部署位置**：{loc_name}\n")
        
        # 红方兵力
        red_forces = proposal.get("red_forces", [])
        if red_forces:
            lines.append("### 🔴 红方兵力\n")
            for force in red_forces:
                name = force.get("equip_name", force.get("equip_id", "未知"))
                count = force.get("count", 1)
                role = force.get("role", "")
                role_str = f"（{role}）" if role else ""
                lines.append(f"- **{name}** × {count} {role_str}")
            lines.append("")
        
        # 蓝方兵力
        blue_forces = proposal.get("blue_forces", [])
        if blue_forces:
            lines.append("### 🔵 蓝方兵力\n")
            for force in blue_forces:
                name = force.get("equip_name", force.get("equip_id", "未知"))
                count = force.get("count", 1)
                role = force.get("role", "")
                role_str = f"（{role}）" if role else ""
                lines.append(f"- **{name}** × {count} {role_str}")
            lines.append("")
        
        # 战术说明
        tactical_notes = proposal.get("tactical_notes", "")
        if tactical_notes:
            lines.append(f"### 战术说明\n{tactical_notes}\n")
        
        return "\n".join(lines)
