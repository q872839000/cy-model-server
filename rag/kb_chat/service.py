"""
知识库对话服务

本模块提供知识库对话的核心服务，整合：
- 意图识别
- Query 改写
- RAG 检索
- Prompt 组装
- LLM 生成

作为知识库对话的统一入口，对外提供完整的对话能力。
"""

import time
from typing import List, Dict, Optional, Any, Callable, Awaitable, AsyncIterator
from loguru import logger

from rag.kb_chat.types import (
    KBChatResult,
)
from models import KBSourceReference, KBChatInfo
from core.config import KBChatConfigManager, KBChatSettings
from rag.kb_chat.intent import IntentRouter
from rag.kb_chat.rewriter import QueryRewriter
from rag.kb_chat.prompt import PromptBuilder
from rag.service import RAGService
from models import KBSearchRequest, SearchMode


class KBChatService:
    """
    知识库对话服务
    
    整合意图识别、Query改写、RAG检索、LLM生成的完整流程。
    
    """

    def __init__(
        self,
        rag_service: RAGService,
        llm_fn: Callable[..., Awaitable[str]],
        llm_stream_fn: Optional[Callable[..., AsyncIterator[str]]] = None,
        config: Optional[KBChatSettings] = None,
    ):
        """
        初始化知识库对话服务
        
        Args:
            rag_service: RAG 检索服务
            llm_fn: LLM 调用函数，签名: (messages, **kwargs) -> str
            llm_stream_fn: LLM 流式调用函数 (可选)
            config: 对话配置 (可选，为 None 时使用全局配置管理器)
        """
        self._rag_service = rag_service
        self._llm_fn = llm_fn
        self._llm_stream_fn = llm_stream_fn
        # 优先使用传入的配置，否则使用全局配置管理器
        self._config = config if config is not None else KBChatConfigManager.get_config()
        
        # 创建简单的 LLM 调用函数（用于意图识别和改写）
        simple_llm_fn = self._create_simple_llm_fn()
        
        # 初始化子组件
        self._intent_router = IntentRouter(
            config=self._config,
            llm_fn=simple_llm_fn if self._config.intent_use_llm else None,
        )
        self._query_rewriter = QueryRewriter(
            config=self._config,
            llm_fn=simple_llm_fn,
        )
        self._prompt_builder = PromptBuilder(config=self._config)
    
    def _create_simple_llm_fn(self) -> Callable[[str], Awaitable[str]]:
        """创建简单的 LLM 调用函数（用于意图识别和 Query 改写）"""
        async def simple_llm(prompt: str, enable_thinking: bool = False) -> str:
            messages = [{"role": "user", "content": prompt}]
            return await self._llm_fn(
                model=None,  # 使用默认模型
                messages=messages,
                max_tokens=200,
                temperature=0.0,
                enable_thinking=enable_thinking,
            )
        return simple_llm
    
    async def chat(
        self,
        model: str,
        collection_name: str,
        messages: List[Dict],
        search_enabled: bool = True,
        search_mode: Optional[str] = None,
        search_top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        query_rewrite: str = "auto",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        session_id: Optional[str] = None,
        return_sources: bool = True,
        enable_thinking: bool = True,
    ) -> KBChatResult:
        """
        执行知识库对话（非流式）
        
        Args:
            model: LLM 模型名称
            collection_name: 知识库名称（Milvus Collection）
            messages: 对话消息列表（上层传入的完整历史）
            search_enabled: 是否启用检索（默认 True）
            search_mode: 检索模式（dense/sparse/hybrid）
            search_top_k: 检索数量
            rerank: 是否重排序
            score_threshold: 分数阈值
            query_rewrite: Query 改写模式（auto/always/never）
            max_tokens: 最大生成 token
            temperature: 生成温度
            top_p: Top-p 采样
            session_id: 会话 ID（仅用于日志追踪）
            return_sources: 是否返回引用来源
            
        Returns:
            KBChatResult: 对话结果
        """
        # 消息处理
        context = await self._prepare_chat_context(
            messages=messages,
            model=model,
            collection_name=collection_name,
            search_enabled=search_enabled,
            search_mode=search_mode,
            search_top_k=search_top_k,
            rerank=rerank,
            score_threshold=score_threshold,
            query_rewrite=query_rewrite,
            return_sources=return_sources,
            session_id=session_id,
        )
        
        # LLM 生成（非流式）
        response_content = await self._llm_fn(
            model=model,
            messages=context["llm_messages"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        
        # 计算总时间并记录日志
        total_time = (time.time() - context["start_time"]) * 1000
        logger.info(
            "[KBChat] 完成: intent={} search={} rewrite={} total={:.1f}ms",
            context["intent_result"].intent.value, 
            context["intent_result"].should_search,
            context["kb_info"].query_rewritten, 
            total_time
        )
        
        return KBChatResult(
            content=response_content,
            kb_info=context["kb_info"],
        )
    
    async def chat_stream(
        self,
        model: str,
        collection_name: str,
        messages: List[Dict],
        search_enabled: bool = True,
        search_mode: Optional[str] = None,
        search_top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        query_rewrite: str = "auto",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        session_id: Optional[str] = None,
        return_sources: bool = True,
        enable_thinking: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式知识库对话
        
        先同步完成检索，再流式输出LLM生成。
        
        Yields:
            Dict: 流式事件
            - {"type": "search_start"}: 开始检索
            - {"type": "search_done", "data": {...}}: 检索完成
            - {"type": "content", "data": {"text": "..."}}: 内容片段
            - {"type": "done", "data": {...}}: 完成
            - {"type": "error", "data": {"message": "..."}}: 错误
        """
        if not self._llm_stream_fn:
            raise ValueError("流式生成函数未配置")
        
        try:
            # 发送检索开始事件（如果需要检索）
            if search_enabled:
                yield {"type": "search_start", "data": {}}
            
            # 使用公共预处理逻辑
            context = await self._prepare_chat_context(
                messages=messages,
                model=model,
                collection_name=collection_name,
                search_enabled=search_enabled,
                search_mode=search_mode,
                search_top_k=search_top_k,
                rerank=rerank,
                score_threshold=score_threshold,
                query_rewrite=query_rewrite,
                return_sources=return_sources,
                session_id=session_id,
            )
            
            # 发送检索完成事件（如果进行了检索）
            if context["kb_info"].search_performed:
                yield {
                    "type": "search_done",
                    "data": {
                        "sources": [s.model_dump() for s in context["sources"]],
                        "search_query": context["kb_info"].search_query,
                        "query_rewritten": context["kb_info"].query_rewritten,
                        "original_query": context["kb_info"].original_query,
                        "search_took_ms": context["kb_info"].search_took_ms,
                    }
                }
            
            # 流式生成
            async for chunk in self._llm_stream_fn(
                model=model,
                messages=context["llm_messages"],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            ):
                yield {"type": "content", "data": {"text": chunk}}
            
            # 发送完成事件
            total_time = (time.time() - context["start_time"]) * 1000
            yield {
                "type": "done",
                "data": {
                    "intent": context["intent_result"].intent.value,
                    "total_time_ms": total_time,
                }
            }
            
        except Exception as e:
            logger.error("[KBChat] 流式对话错误: {}", str(e))
            yield {
                "type": "error",
                "data": {"message": str(e)}
            }
    
    def _extract_query_and_history(
        self,
        messages: List[Dict],
    ) -> tuple[str, List[Dict]]:
        """
        从消息列表中提取当前问题和历史
        
        Args:
            messages: 完整消息列表
            
        Returns:
            tuple: (当前问题, 历史消息列表)
        """
        if not messages:
            return "", []
        
        # 最后一条 user 消息作为当前问题
        current_query = ""
        history = []
        
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                current_query = messages[i].get("content", "")
                history = messages[:i]
                break
        
        # 如果没找到 user 消息，取最后一条
        if not current_query and messages:
            current_query = messages[-1].get("content", "")
            history = messages[:-1]
        
        return current_query, history

    
    def _search(
        self,
        collection_name: str,
        query: str,
        mode: str,
        top_k: int,
        rerank: bool,
        score_threshold: float,
    ) -> List[Dict]:
        """
        执行 RAG 检索
        
        Args:
            collection_name: 知识库名称
            query: 检索 query
            mode: 检索模式
            top_k: 检索数量
            rerank: 是否重排序
            score_threshold: 分数阈值
            
        Returns:
            List[Dict]: 检索结果列表
        """
        # 转换检索模式
        mode_map = {
            "dense": SearchMode.DENSE,
            "sparse": SearchMode.SPARSE,
            "hybrid": SearchMode.HYBRID,
        }
        search_mode = mode_map.get(mode.lower(), SearchMode.HYBRID)
        
        # 构建检索请求
        request = KBSearchRequest(
            collection_name=collection_name,
            query=query,
            mode=search_mode,
            top_k=top_k,
            rerank=rerank,
            score_threshold=score_threshold,
        )
        
        # 执行检索
        response = self._rag_service.search(request)
        
        # 转换结果格式
        hits = []
        for hit in response.hits:
            hits.append({
                "id": hit.chunk.id,
                "doc_name": hit.chunk.doc_name,
                "chapter": hit.chunk.chapter,
                "content": hit.chunk.content,
                "score": hit.score,
            })
        
        return hits
    
    def _filter_irrelevant_hits(
        self,
        search_hits: List[Dict],
        min_score_ratio: float = 0.85,
        max_hits_for_llm: int = 3,
    ) -> List[Dict]:
        """
        过滤掉可能不相关的检索结果
        
        策略：
        1. 保留分数 >= 最高分 * min_score_ratio 的结果
        2. 如果最高分 < 0.6，认为所有结果都不太相关，只保留最高分那一条
        3. 最多保留 max_hits_for_llm 条结果
        
        Args:
            search_hits: 检索结果列表
            min_score_ratio: 相对于最高分的最低比例阈值（默认 0.85）
            max_hits_for_llm: 传给 LLM 的最大结果数（默认 3）
            
        Returns:
            List[Dict]: 过滤后的检索结果
        """
        if not search_hits:
            return search_hits
        
        # 获取最高分
        max_score = max(hit.get("score", 0) for hit in search_hits)
        
        # 如果最高分较低，只保留第一条（最相关的）
        if max_score < 0.6:
            logger.info("[KBChat] 最高分({:.3f})较低，仅保留 top-1 结果", max_score)
            return search_hits[:1]
        
        # 计算分数阈值（更严格：最高分的 85%）
        score_threshold = max_score * min_score_ratio
        
        # 过滤低分结果
        filtered = [
            hit for hit in search_hits
            if hit.get("score", 0) >= score_threshold
        ]
        
        # 限制最大数量
        if len(filtered) > max_hits_for_llm:
            filtered = filtered[:max_hits_for_llm]
        
        if len(filtered) < len(search_hits):
            logger.info(
                "[KBChat] 过滤低相关性结果: {} -> {} (阈值: {:.3f}, 最高分: {:.3f})",
                len(search_hits), len(filtered), score_threshold, max_score
            )
        
        return filtered
    
    async def _prepare_chat_context(
        self,
        messages: List[Dict],
        model: str,
        collection_name: str,
        search_enabled: bool = True,
        search_mode: Optional[str] = None,
        search_top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        query_rewrite: str = "auto",
        return_sources: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        公共的聊天预处理逻辑
        
        公共步骤：意图识别、检索、Prompt组装
        
        Returns:
            Dict: 包含预处理结果的上下文字典
        """
        start_time = time.time()
        
        # 提取当前问题和历史
        current_query, history = self._extract_query_and_history(messages)
        
        logger.info(
            "[KBChat] session={} query='{}' history_turns={}",
            session_id or "unknown", current_query[:50], len(history) // 2
        )

        # 意图识别
        intent_result = await self._intent_router.analyze(current_query, history)
        
        # 检索流程
        search_hits = []
        search_query = None
        query_rewritten = False
        search_took_ms = None
        
        if intent_result.should_search:
            search_start = time.time()
            
            # Query 改写（如果需要）
            if intent_result.rewrite_needed and query_rewrite != "never":
                rewrite_result = await self._query_rewriter.rewrite(current_query, history)
                search_query = rewrite_result.rewritten_query
                query_rewritten = rewrite_result.rewrite_applied
            else:
                search_query = current_query
            
            # 执行检索
            search_hits = self._search(
                collection_name=collection_name,
                query=search_query,
                mode=search_mode or self._config.search_mode,
                top_k=search_top_k or self._config.search_top_k,
                rerank=rerank if rerank is not None else self._config.search_rerank,
                score_threshold=score_threshold or self._config.search_score_threshold,
            )
            
            search_took_ms = (time.time() - search_start) * 1000
            logger.info(
                "[KBChat] 检索完成: query='{}' hits={} took={:.1f}ms",
                search_query[:30], len(search_hits), search_took_ms
            )
        
        # 过滤低相关性结果（传给 LLM 前）
        filtered_hits = self._filter_irrelevant_hits(search_hits)
        
        # 组装 Prompt
        model_family = PromptBuilder.detect_model_family(model)
        llm_messages = self._prompt_builder.build(
            model_family=model_family,
            user_query=current_query,
            search_hits=filtered_hits,
            history=history,
        )
        
        # 构建来源信息
        sources = self._build_sources(search_hits) if return_sources else []
        
        # 构建 KB信息
        kb_info = KBChatInfo(
            search_performed=intent_result.should_search,
            search_query=search_query,
            query_rewritten=query_rewritten,
            original_query=current_query if query_rewritten else None,
            intent=intent_result.intent.value,
            sources=sources,
            search_took_ms=search_took_ms,
        )
        
        return {
            "current_query": current_query,
            "intent_result": intent_result,
            "llm_messages": llm_messages,
            "kb_info": kb_info,
            "sources": sources,
            "start_time": start_time,
        }

    def _build_sources(self, search_hits: List[Dict]) -> List[KBSourceReference]:
        """
        构建引用来源列表
        
        Args:
            search_hits: 检索结果
            
        Returns:
            List[KBSourceReference]: 引用来源列表
        """
        sources = []
        for i, hit in enumerate(search_hits, 1):
            content = hit.get("content", "")
            preview = content[:100] + "..." if len(content) > 100 else content
            
            sources.append(KBSourceReference(
                index=i,
                doc_name=hit.get("doc_name", ""),
                chapter=hit.get("chapter", ""),
                content_preview=preview,
                score=hit.get("score", 0.0),
            ))
        
        return sources
