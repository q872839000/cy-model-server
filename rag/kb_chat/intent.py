"""
意图识别模块

本模块负责识别用户输入的意图，决定检索策略：
- 规则匹配：基于关键词和模式，零延迟
- LLM 识别：更准确，但有延迟（可选）

意图分类：
- KNOWLEDGE_QUERY: 知识查询，直接检索
- FOLLOW_UP: 追问细节，改写后检索
- TOPIC_SWITCH: 换话题，检索新话题
- CLARIFICATION: 要求澄清，可能检索
- CHITCHAT: 闲聊确认，不检索
"""

from typing import List, Dict, Optional, Callable, Awaitable
from loguru import logger

from rag.kb_chat.types import UserIntent, IntentResult, KBChatConfig


class IntentRouter:
    """
    意图路由器
    
    识别用户输入的意图，决定是否需要检索和改写。
    支持规则匹配和 LLM 两种模式。
    
    Usage:
        >>> router = IntentRouter(config)
        >>> result = await router.analyze("它有什么优点？", history)
        >>> if result.should_search:
        ...     # 执行检索
    """
    
    # 追问指示词（包含指代或追问意图）
    FOLLOW_UP_INDICATORS = [
        "它", "这个", "那个", "这", "那", "他", "她", "他们", "它们",
        "上面", "前面", "刚才", "之前",
        "详细", "具体", "展开", "解释一下", "说说",
        "为什么", "怎么", "如何", "什么意思",
    ]
    
    # 换话题指示词
    TOPIC_SWITCH_INDICATORS = [
        "另外", "还有", "顺便", "对了",
        "换个话题", "说说", "聊聊",
        "那", "那么",  # "那XXX呢" 模式
    ]
    
    # 澄清指示词
    CLARIFICATION_INDICATORS = [
        "第一点", "第二点", "第三点", "第1点", "第2点", "第3点",
        "刚才说的", "你说的", "上面说的",
        "没太懂", "不太明白", "再说一遍",
    ]
    
    def __init__(
        self,
        config: KBChatConfig,
        llm_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        """
        初始化意图路由器
        
        Args:
            config: 对话配置
            llm_fn: LLM 调用函数，签名: async (prompt: str) -> str
        """
        self._config = config
        self._llm_fn = llm_fn
        self._chitchat_keywords = set(config.chitchat_keywords)
    
    async def analyze(
        self,
        current_query: str,
        history: List[Dict],
    ) -> IntentResult:
        """
        分析用户意图
        
        Args:
            current_query: 当前用户输入
            history: 对话历史（上层传入的完整历史）
            
        Returns:
            IntentResult: 意图识别结果
        """
        if not self._config.intent_enabled:
            # 禁用意图识别，默认检索
            return IntentResult(
                intent=UserIntent.KNOWLEDGE_QUERY,
                confidence=1.0,
                should_search=True,
                rewrite_needed=bool(history),
                reasoning="意图识别已禁用，默认检索"
            )
        
        # 优先使用规则匹配
        result = self._analyze_with_rules(current_query, history)
        
        # 如果配置了 LLM 且规则置信度不高，使用 LLM 辅助
        if (
            self._config.intent_use_llm 
            and self._llm_fn 
            and result.confidence < 0.7
        ):
            try:
                llm_result = await self._analyze_with_llm(current_query, history)
                if llm_result.confidence > result.confidence:
                    result = llm_result
            except Exception as e:
                logger.warning("LLM 意图识别失败，使用规则结果: {}", str(e))
        
        logger.debug(
            "意图识别: query='{}' intent={} confidence={:.2f} search={} rewrite={}",
            current_query[:50], result.intent.value, result.confidence,
            result.should_search, result.rewrite_needed
        )
        
        return result
    
    def _analyze_with_rules(
        self,
        current_query: str,
        history: List[Dict],
    ) -> IntentResult:
        """
        基于规则的意图识别
        
        Args:
            current_query: 当前用户输入
            history: 对话历史
            
        Returns:
            IntentResult: 意图识别结果
        """
        query_stripped = current_query.strip()
        
        # 1. 检测闲聊/确认（短句 + 关键词）
        if len(query_stripped) <= 15:
            for keyword in self._chitchat_keywords:
                if keyword in query_stripped:
                    return IntentResult(
                        intent=UserIntent.CHITCHAT,
                        confidence=0.9,
                        should_search=False,
                        rewrite_needed=False,
                        reasoning=f"匹配闲聊关键词: {keyword}"
                    )
        
        # 2. 无历史 → 首次知识查询
        if not history:
            return IntentResult(
                intent=UserIntent.KNOWLEDGE_QUERY,
                confidence=0.95,
                should_search=True,
                rewrite_needed=False,
                reasoning="无对话历史，首次查询"
            )
        
        # 3. 检测澄清请求
        for indicator in self.CLARIFICATION_INDICATORS:
            if indicator in query_stripped:
                return IntentResult(
                    intent=UserIntent.CLARIFICATION,
                    confidence=0.8,
                    should_search=True,
                    rewrite_needed=True,
                    reasoning=f"匹配澄清指示词: {indicator}"
                )
        
        # 4. 检测换话题（"那XXX呢"模式）
        if self._is_topic_switch(query_stripped):
            return IntentResult(
                intent=UserIntent.TOPIC_SWITCH,
                confidence=0.8,
                should_search=True,
                rewrite_needed=False,  # 新话题已明确，无需改写
                reasoning="检测到换话题模式"
            )
        
        # 5. 检测追问（有历史 + 指代词/追问词）
        has_follow_up = any(
            ind in query_stripped for ind in self.FOLLOW_UP_INDICATORS
        )
        
        if has_follow_up:
            return IntentResult(
                intent=UserIntent.FOLLOW_UP,
                confidence=0.8,
                should_search=True,
                rewrite_needed=True,
                reasoning="检测到追问指示词，需要Query改写"
            )
        
        # 6. 有历史但无明显指代，可能是新问题或隐式追问
        # 通过问题长度和是否包含疑问词判断
        if len(query_stripped) > 20 and any(
            q in query_stripped for q in ["什么", "如何", "怎么", "为什么", "哪些", "是否"]
        ):
            # 较长的完整问句，可能是新问题
            return IntentResult(
                intent=UserIntent.KNOWLEDGE_QUERY,
                confidence=0.6,
                should_search=True,
                rewrite_needed=False,
                reasoning="完整问句，视为新查询"
            )
        
        # 7. 默认：视为追问（保守策略，进行改写）
        return IntentResult(
            intent=UserIntent.FOLLOW_UP,
            confidence=0.5,
            should_search=True,
            rewrite_needed=True,
            reasoning="有历史上下文，默认视为追问"
        )
    
    def _is_topic_switch(self, query: str) -> bool:
        """
        判断是否为换话题
        
        检测模式：
        - "那XXX呢？"
        - "另外，XXX"
        - "说说XXX"
        """
        # "那...呢" 模式
        if query.startswith("那") and "呢" in query and len(query) > 5:
            return True
        
        # 明确的换话题指示词
        for indicator in ["另外", "还有", "顺便", "换个话题"]:
            if query.startswith(indicator):
                return True
        
        return False
    
    async def _analyze_with_llm(
        self,
        current_query: str,
        history: List[Dict],
    ) -> IntentResult:
        """
        使用 LLM 进行意图识别
        
        Args:
            current_query: 当前用户输入
            history: 对话历史
            
        Returns:
            IntentResult: 意图识别结果
        """
        # 构建历史摘要（最近 2 轮）
        history_text = self._format_history(history[-4:]) if history else "无"
        
        prompt = f"""分析用户意图，判断是否需要检索知识库。

## 对话历史
{history_text}

## 当前用户输入
{current_query}

## 请从以下意图中选择一个：
1. knowledge_query - 新的知识查询，需要检索
2. follow_up - 追问上文细节（如"它的优点是什么"），需要改写后检索
3. topic_switch - 明确换到新话题（如"那NLP呢"），需要检索新话题
4. clarification - 要求澄清上文回答，可能需要检索补充
5. chitchat - 闲聊、确认、感谢，不需要检索

## 输出格式（仅输出JSON，不要其他内容）
{{"intent": "意图名称", "should_search": true/false, "rewrite_needed": true/false, "reasoning": "简短理由"}}
"""

        response = await self._llm_fn(prompt)
        
        # 解析 LLM 返回的 JSON
        import json
        
        # 尝试提取 JSON
        response = response.strip()
        if response.startswith("```"):
            # 去除代码块标记
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        
        result = json.loads(response)
        
        intent_str = result.get("intent", "knowledge_query").lower()
        intent_map = {
            "knowledge_query": UserIntent.KNOWLEDGE_QUERY,
            "follow_up": UserIntent.FOLLOW_UP,
            "topic_switch": UserIntent.TOPIC_SWITCH,
            "clarification": UserIntent.CLARIFICATION,
            "chitchat": UserIntent.CHITCHAT,
        }
        
        return IntentResult(
            intent=intent_map.get(intent_str, UserIntent.KNOWLEDGE_QUERY),
            confidence=0.85,
            should_search=result.get("should_search", True),
            rewrite_needed=result.get("rewrite_needed", False),
            reasoning=result.get("reasoning", "LLM判断")
        )
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化历史对话"""
        lines = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # 截断
            role_name = {"user": "用户", "assistant": "助手"}.get(role, role)
            lines.append(f"{role_name}: {content}")
        return "\n".join(lines) if lines else "无"
