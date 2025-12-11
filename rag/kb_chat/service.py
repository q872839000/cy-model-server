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
    UserIntent,
    IntentResult,
    RewriteResult,
    SourceReference,
    KBChatConfig,
    KBInfo,
    KBChatResult,
)
from rag.kb_chat.intent import IntentRouter
from rag.kb_chat.rewriter import QueryRewriter
from rag.kb_chat.prompt import PromptBuilder, ModelFamily
from rag.service import RAGService
from models.kb_schemas import KBSearchRequest, SearchMode


class KBChatService:
    """
    知识库对话服务
    
    整合意图识别、Query改写、RAG检索、LLM生成的完整流程。
    
    Usage:
        >>> service = KBChatService(rag_service, llm_fn)
        >>> result = await service.chat(
        ...     model="glm-4-9b",
        ...     collection_name="product_manual",
        ...     messages=[{"role": "user", "content": "什么是RAG？"}],
        ... )
        >>> print(result.content)
    """
    
    def __init__(
        self,
        rag_service: RAGService,
        llm_fn: Callable[..., Awaitable[str]],
        llm_stream_fn: Optional[Callable[..., AsyncIterator[str]]] = None,
        config: Optional[KBChatConfig] = None,
    ):
        """
        初始化知识库对话服务
        
        Args:
            rag_service: RAG 检索服务
            llm_fn: LLM 非流式生成函数
                签名: async (model, messages, max_tokens, temperature, ...) -> str
            llm_stream_fn: LLM 流式生成函数（可选）
                签名: async (model, messages, ...) -> AsyncIterator[str]
            config: 服务配置
        """
        self._rag_service = rag_service
        self._llm_fn = llm_fn
        self._llm_stream_fn = llm_stream_fn
        self._config = config or KBChatConfig()
        
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
        async def simple_llm(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            return await self._llm_fn(
                model=None,  # 使用默认模型
                messages=messages,
                max_tokens=200,
                temperature=0.0,
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
    ) -> KBChatResult:
        """
        执行知识库对话
        
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
        start_time = time.time()
        
        # 提取当前问题和历史
        current_query, history = self._extract_query_and_history(messages)
        
        logger.info(
            "[KBChat] session={} query='{}' history_turns={}",
            session_id or "unknown", current_query[:50], len(history) // 2
        )
        
        # 1. 意图识别
        intent_result = await self._analyze_intent(
            current_query, history, search_enabled
        )
        
        # 2. 检索流程
        search_hits = []
        search_query = None
        query_rewritten = False
        search_took_ms = None
        
        if intent_result.should_search:
            search_start = time.time()
            
            # 2a. Query 改写（如果需要）
            if intent_result.rewrite_needed and query_rewrite != "never":
                rewrite_result = await self._rewrite_query(current_query, history)
                search_query = rewrite_result.rewritten_query
                query_rewritten = rewrite_result.rewrite_applied
            else:
                search_query = current_query
            
            # 2b. 执行检索
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
        
        # 2c. 过滤低相关性结果（传给 LLM 前）
        filtered_hits = self._filter_irrelevant_hits(search_hits)
        
        # 3. 组装 Prompt
        model_family = PromptBuilder.detect_model_family(model)
        llm_messages = self._prompt_builder.build(
            model_family=model_family,
            user_query=current_query,
            search_hits=filtered_hits,  # 使用过滤后的结果
            history=history,
        )
        
        # 4. LLM 生成
        response_content = await self._llm_fn(
            model=model,
            messages=llm_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # 5. 构建结果
        sources = self._build_sources(search_hits) if return_sources else []
        
        kb_info = KBInfo(
            search_performed=intent_result.should_search,
            search_query=search_query,
            query_rewritten=query_rewritten,
            original_query=current_query if query_rewritten else None,
            intent=intent_result.intent.value,
            sources=sources,
            search_took_ms=search_took_ms,
        )
        
        total_time = (time.time() - start_time) * 1000
        logger.info(
            "[KBChat] 完成: intent={} search={} rewrite={} total={:.1f}ms",
            intent_result.intent.value, intent_result.should_search,
            query_rewritten, total_time
        )
        
        return KBChatResult(
            content=response_content,
            kb_info=kb_info,
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
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式知识库对话
        
        先同步完成检索，再流式输出 LLM 生成。
        
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
        
        start_time = time.time()
        
        # 提取当前问题和历史
        current_query, history = self._extract_query_and_history(messages)
        
        try:
            # 1. 意图识别
            intent_result = await self._analyze_intent(
                current_query, history, search_enabled
            )
            
            # 2. 检索流程
            search_hits = []
            search_query = None
            query_rewritten = False
            search_took_ms = None
            
            if intent_result.should_search:
                yield {"type": "search_start", "data": {}}
                
                search_start = time.time()
                
                # Query 改写
                if intent_result.rewrite_needed and query_rewrite != "never":
                    rewrite_result = await self._rewrite_query(current_query, history)
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
                sources = self._build_sources(search_hits) if return_sources else []
                
                yield {
                    "type": "search_done",
                    "data": {
                        "sources": [s.model_dump() for s in sources],
                        "search_query": search_query,
                        "query_rewritten": query_rewritten,
                        "original_query": current_query if query_rewritten else None,
                        "search_took_ms": search_took_ms,
                    }
                }
            
            # 2c. 过滤低相关性结果（传给 LLM 前）
            filtered_hits = self._filter_irrelevant_hits(search_hits)
            
            # 3. 组装 Prompt
            model_family = PromptBuilder.detect_model_family(model)
            llm_messages = self._prompt_builder.build(
                model_family=model_family,
                user_query=current_query,
                search_hits=filtered_hits,  # 使用过滤后的结果
                history=history,
            )
            
            # 4. 流式生成
            async for chunk in self._llm_stream_fn(
                model=model,
                messages=llm_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                yield {"type": "content", "data": {"text": chunk}}
            
            # 5. 完成
            total_time = (time.time() - start_time) * 1000
            yield {
                "type": "done",
                "data": {
                    "intent": intent_result.intent.value,
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
    
    async def _analyze_intent(
        self,
        current_query: str,
        history: List[Dict],
        search_enabled: bool,
    ) -> IntentResult:
        """
        分析用户意图
        
        Args:
            current_query: 当前问题
            history: 历史消息
            search_enabled: 是否启用检索
            
        Returns:
            IntentResult: 意图识别结果
        """
        # 如果禁用检索，直接返回不检索
        if not search_enabled:
            return IntentResult(
                intent=UserIntent.CHITCHAT,
                confidence=1.0,
                should_search=False,
                rewrite_needed=False,
                reasoning="检索已禁用"
            )
        
        return await self._intent_router.analyze(current_query, history)
    
    async def _rewrite_query(
        self,
        current_query: str,
        history: List[Dict],
    ) -> RewriteResult:
        """
        改写 Query
        
        Args:
            current_query: 当前问题
            history: 历史消息
            
        Returns:
            RewriteResult: 改写结果
        """
        return await self._query_rewriter.rewrite(current_query, history)
    
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
    
    def _build_sources(self, search_hits: List[Dict]) -> List[SourceReference]:
        """
        构建引用来源列表
        
        Args:
            search_hits: 检索结果
            
        Returns:
            List[SourceReference]: 引用来源列表
        """
        sources = []
        for i, hit in enumerate(search_hits, 1):
            content = hit.get("content", "")
            preview = content[:100] + "..." if len(content) > 100 else content
            
            sources.append(SourceReference(
                index=i,
                doc_name=hit.get("doc_name", ""),
                chapter=hit.get("chapter", ""),
                content_preview=preview,
                score=hit.get("score", 0.0),
            ))
        
        return sources
