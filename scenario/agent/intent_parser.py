"""
意图解析器

解析用户输入，识别用户真正想要什么。
"""

import json
import re
from typing import Dict, Any, Optional, List
from loguru import logger

from core.services.chat_service import ChatService
from scenario.agent.base import Intent
from scenario.memory import ScenarioMemory
from prompts import get_prompt_loader
from utils.message_filter import filter_thinking_content


class IntentParser:
    """意图解析器"""

    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service
        self.prompt_loader = get_prompt_loader()
    
    def _get_prompt_template(self) -> str:
        """获取提示词模板"""
        try:
            return self.prompt_loader.render("scenario/agent_intent")
        except Exception:
            logger.warning("加载agent_intent提示词失败，使用默认提示词")
            return self._default_prompt()
    
    def _default_prompt(self) -> str:
        """默认提示词"""
        return '''分析用户意图，返回JSON：
{
    "intent": "create/modify/plan_only/query/confirm/unclear",
    "confidence": 0.0-1.0,
    "details": {},
    "clarification_question": ""
}'''
    
    async def parse(
        self, 
        user_input: str, 
        memory: ScenarioMemory,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        解析用户意图
        
        Args:
            user_input: 用户输入
            memory: 当前记忆状态
            history: 对话历史
            
        Returns:
            解析结果字典
        """
        # 加载并填充提示词模板
        prompt_template = self._get_prompt_template()
        prompt = prompt_template.replace(
            "{{instance_count}}", str(len(memory.current_instances))
        ).replace(
            "{{has_equipment}}", str(len(memory.equipment_library) > 0)
        ).replace(
            "{{user_input}}", user_input
        )
        
        # 构建messages：系统提示 + 过滤后的历史记录 + 当前用户输入
        messages = [{"role": "system", "content": prompt}]
        
        # 添加历史记录（过滤think内容）
        if history:
            filtered_history = filter_thinking_content(history)
            # 只保留最近几轮对话，避免上下文过长
            recent_history = filtered_history[-6:] if len(filtered_history) > 6 else filtered_history
            messages.extend(recent_history)
            logger.debug("【意图解析】 加载{}条历史记录", len(recent_history))
        
        messages.append({"role": "user", "content": user_input})
        
        try:
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
            logger.info("【意图解析】 intent={}, confidence={:.2f}", 
                       result.get("intent"), result.get("confidence", 0))
            
            if result.get("missing_info"):
                logger.debug("【意图解析】 缺失信息: {}", result["missing_info"])
            
            return result
            
        except Exception as e:
            logger.error("【意图解析失败】 {}", str(e))
            return {
                "intent": Intent.UNCLEAR.value,
                "confidence": 0.0,
                "error": str(e),
                "clarification_question": "抱歉，我没有理解您的请求，能否再说明一下？"
            }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        logger.debug("【意图解析】 过滤后响应: {}", response[:500])
        """解析LLM响应"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()
            if json_str.startswith('```'):
                json_str = re.sub(r'^```\w*\n?', '', json_str)
                json_str = re.sub(r'\n?```$', '', json_str)
        
        try:
            result = json.loads(json_str)
            if "intent" not in result:
                result["intent"] = Intent.UNCLEAR.value
            return result
        except json.JSONDecodeError:
            logger.warning("【意图解析】 JSON解析失败，尝试提取关键信息")
            return self._extract_from_text(response)
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取意图信息"""
        text_lower = text.lower()
        
        if "create" in text_lower or "创建" in text:
            intent = Intent.CREATE.value
        elif "modify" in text_lower or "修改" in text:
            intent = Intent.MODIFY.value
        elif "plan_only" in text_lower or "计划" in text or "方案" in text:
            intent = Intent.PLAN_ONLY.value
        elif "query" in text_lower or "查询" in text:
            intent = Intent.QUERY.value
        elif "confirm" in text_lower or "确认" in text:
            intent = Intent.CONFIRM.value
        else:
            intent = Intent.UNCLEAR.value
        
        return {
            "intent": intent,
            "confidence": 0.5,
            "details": {},
            "clarification_question": "请问您具体想要做什么？" if intent == Intent.UNCLEAR.value else ""
        }
    
    def _filter_thinking(self, content: str) -> str:
        """过滤响应中的think标签内容"""
        if not content or not isinstance(content, str):
            return content
        # 移除<think>...</think>标签及其内容
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # 移除未闭合的<think>标签
        content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL)
        return content.strip()
