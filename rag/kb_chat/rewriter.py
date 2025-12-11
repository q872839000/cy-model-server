"""
Query 改写模块

本模块负责将依赖上下文的问题改写为独立的检索 Query。

典型场景：
- 指代消解："它有什么优点？" → "RAG有什么优点？"
- 省略补全："和传统数据库有什么区别？" → "向量数据库和传统数据库有什么区别？"
- 追问细节："第二点能详细说说吗？" → "[具体内容] 详细解释"
"""

import asyncio
from typing import List, Dict, Optional, Callable, Awaitable
from loguru import logger

from rag.kb_chat.types import RewriteResult, KBChatConfig


# Query 改写 Prompt 模板
REWRITE_PROMPT_TEMPLATE = """请根据对话历史，将用户最新问题改写为一个独立、完整、适合知识库检索的问题。

## 改写规则
1. 消解指代词（它、这个、那个、他们等），替换为具体实体
2. 补充省略的上下文信息，使问题独立完整
3. 保持问题简洁，适合检索（不超过50字）
4. 如果问题已经完整独立，保持原样
5. 只输出改写后的问题，不要其他任何内容

## 对话历史
{history}

## 用户最新问题
{query}

## 改写后的问题："""


class QueryRewriter:
    """
    Query 改写器
    
    将依赖上下文的问题改写为独立的检索 Query。
    
    Usage:
        >>> rewriter = QueryRewriter(config, llm_fn)
        >>> result = await rewriter.rewrite("它有什么优点？", history)
        >>> search_query = result.rewritten_query
    """
    
    def __init__(
        self,
        config: KBChatConfig,
        llm_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        """
        初始化 Query 改写器
        
        Args:
            config: 对话配置
            llm_fn: LLM 调用函数，签名: async (prompt: str) -> str
        """
        self._config = config
        self._llm_fn = llm_fn
    
    async def rewrite(
        self,
        query: str,
        history: List[Dict],
    ) -> RewriteResult:
        """
        改写 Query
        
        Args:
            query: 当前用户问题
            history: 对话历史
            
        Returns:
            RewriteResult: 改写结果
        """
        # 检查改写模式
        mode = self._config.query_rewrite_mode
        
        if mode == "never":
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_applied=False,
                reasoning="改写模式为never，跳过改写"
            )
        
        # 无 LLM 可用
        if not self._llm_fn:
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_applied=False,
                reasoning="无LLM可用，跳过改写"
            )
        
        # 无历史，无需改写
        if not history:
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_applied=False,
                reasoning="无对话历史，无需改写"
            )
        
        # mode == "auto" 时，检查是否真的需要改写
        if mode == "auto" and not self._needs_rewrite(query):
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_applied=False,
                reasoning="问题已完整，无需改写"
            )
        
        # 执行 LLM 改写
        try:
            result = await self._rewrite_with_llm(query, history)
            logger.debug(
                "Query改写: '{}' -> '{}' (applied={})",
                query[:30], result.rewritten_query[:30], result.rewrite_applied
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Query改写超时，使用原始query")
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_applied=False,
                reasoning="改写超时，使用原始query"
            )
        except Exception as e:
            logger.warning("Query改写失败: {}，使用原始query", str(e))
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_applied=False,
                reasoning=f"改写失败: {str(e)}"
            )
    
    def _needs_rewrite(self, query: str) -> bool:
        """
        判断是否需要改写
        
        检查是否包含需要消解的指代词或省略。
        
        Args:
            query: 用户问题
            
        Returns:
            bool: 是否需要改写
        """
        # 指代词列表
        pronouns = [
            "它", "这个", "那个", "这", "那", "他", "她", "他们", "它们",
            "上面", "前面", "刚才", "之前", "上文",
        ]
        
        # 检查是否包含指代词
        for pronoun in pronouns:
            if pronoun in query:
                return True
        
        # 检查是否是省略主语的问句
        # 例如："有什么优点？"、"怎么使用？"
        short_patterns = ["有什么", "是什么", "怎么", "如何", "为什么"]
        if len(query) < 15:
            for pattern in short_patterns:
                if query.startswith(pattern):
                    return True
        
        return False
    
    async def _rewrite_with_llm(
        self,
        query: str,
        history: List[Dict],
    ) -> RewriteResult:
        """
        使用 LLM 进行改写
        
        Args:
            query: 用户问题
            history: 对话历史
            
        Returns:
            RewriteResult: 改写结果
        """
        # 构建历史文本（最近 N 轮）
        max_turns = self._config.rewrite_max_history_turns
        recent_history = history[-(max_turns * 2):]
        history_text = self._format_history(recent_history)
        
        # 构建 prompt
        prompt = REWRITE_PROMPT_TEMPLATE.format(
            history=history_text,
            query=query
        )
        
        # 调用 LLM（带超时）
        timeout_sec = self._config.rewrite_timeout_ms / 1000.0
        rewritten = await asyncio.wait_for(
            self._llm_fn(prompt),
            timeout=timeout_sec
        )
        
        # 清理结果
        rewritten = rewritten.strip()
        
        # 去除可能的引号
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        if rewritten.startswith("'") and rewritten.endswith("'"):
            rewritten = rewritten[1:-1]
        
        # 检查是否实际改写
        rewrite_applied = rewritten.lower() != query.lower() and len(rewritten) > 0
        
        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten if rewritten else query,
            rewrite_applied=rewrite_applied,
            reasoning="LLM改写完成" if rewrite_applied else "LLM判断无需改写"
        )
    
    def _format_history(self, history: List[Dict]) -> str:
        """
        格式化历史对话
        
        Args:
            history: 对话历史
            
        Returns:
            str: 格式化后的文本
        """
        lines = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # 跳过 observation（检索结果）
            if role == "observation":
                continue
            
            # 截断过长内容
            if len(content) > 300:
                content = content[:300] + "..."
            
            role_name = {"user": "用户", "assistant": "助手"}.get(role, role)
            lines.append(f"{role_name}: {content}")
        
        return "\n".join(lines) if lines else "无"
