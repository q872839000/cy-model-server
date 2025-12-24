"""
Query 改写模块

本模块负责将依赖上下文的问题改写为独立的检索 Query。

"""

import asyncio
from typing import List, Dict, Optional, Callable, Awaitable
from loguru import logger

from rag.kb_chat.types import RewriteResult
from core.config import KBChatSettings


# Query 改写 Prompt 模板
REWRITE_PROMPT_TEMPLATE = """你的任务是将用户的问题改写为适合知识库检索的完整问题。

## 改写目标
将用户输入转换为能够在知识库中有效检索到相关文档的查询语句。

## 核心原则
1. **检索导向**：改写后的问题必须能够匹配知识库中的具体内容
2. **上下文融合**：分析对话历史，识别当前讨论的主题和实体
3. **语义完整**：消解指代词和省略，形成独立完整的问题
4. **相关性保持**：改写内容必须与对话上下文直接相关

## 处理逻辑
- 如果用户使用指代词，替换为对话中的具体实体
- 如果用户问题省略主语，补充上下文中的主题
- 如果用户表达质疑或困惑，将其转换为对当前讨论主题的具体重新查询
- 如果问题已经完整独立，保持原样
- 确保改写后的问题能够检索到实际的知识内容，而不是过程性或元认知问题

## 输出要求
只输出改写后的问题，不要解释或其他内容。

## 对话历史
{history}

## 用户最新问题
{query}

## 改写后的问题："""


class QueryRewriter:
    """
    Query 改写器
    
    将依赖上下文的问题改写为独立的检索 Query。
    """
    
    def __init__(
        self,
        config: KBChatSettings,
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
            # if len(content) > 300:
            #     content = content[:300] + "..."
            
            role_name = {"user": "用户", "assistant": "助手"}.get(role, role)
            lines.append(f"{role_name}: {content}")
        
        return "\n".join(lines) if lines else "无"
