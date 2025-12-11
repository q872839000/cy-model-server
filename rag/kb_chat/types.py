"""
知识库对话类型定义

本模块定义知识库对话系统的核心类型，包括：
- 用户意图枚举
- 意图识别结果
- Query 改写结果
- 对话配置
- 对话结果
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class UserIntent(str, Enum):
    """
    用户意图枚举
    
    用于分类用户输入，决定检索策略。
    """
    KNOWLEDGE_QUERY = "knowledge_query"   # 知识查询：首次提问或明确主题
    FOLLOW_UP = "follow_up"               # 追问细节：需要结合上下文理解
    TOPIC_SWITCH = "topic_switch"         # 换话题：明确切换到新主题
    CLARIFICATION = "clarification"       # 要求澄清：解释上文回答
    CHITCHAT = "chitchat"                 # 闲聊确认：感谢、确认等


class IntentResult(BaseModel):
    """
    意图识别结果
    
    Attributes:
        intent: 识别出的用户意图
        confidence: 置信度 (0-1)
        should_search: 是否需要检索知识库
        rewrite_needed: 是否需要改写 Query
        reasoning: 识别依据（调试用）
    """
    intent: UserIntent = Field(..., description="用户意图")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    should_search: bool = Field(..., description="是否需要检索")
    rewrite_needed: bool = Field(default=False, description="是否需要改写Query")
    reasoning: Optional[str] = Field(default=None, description="识别依据")


class RewriteResult(BaseModel):
    """
    Query 改写结果
    
    Attributes:
        original_query: 原始问题
        rewritten_query: 改写后的问题
        rewrite_applied: 是否实际进行了改写
        reasoning: 改写理由
    """
    original_query: str = Field(..., description="原始问题")
    rewritten_query: str = Field(..., description="改写后的问题")
    rewrite_applied: bool = Field(default=False, description="是否实际改写")
    reasoning: Optional[str] = Field(default=None, description="改写理由")


class SourceReference(BaseModel):
    """
    引用来源
    
    记录检索结果中被引用的来源信息。
    
    Attributes:
        index: 引用序号（对应回答中的【1†来源】）
        doc_name: 文档名称
        chapter: 章节标题
        content_preview: 内容预览（截断）
        score: 相关性分数
    """
    index: int = Field(..., description="引用序号")
    doc_name: str = Field(..., description="文档名称")
    chapter: str = Field(default="", description="章节标题")
    content_preview: str = Field(..., description="内容预览")
    score: float = Field(..., description="相关性分数")


class KBChatConfig(BaseModel):
    """
    知识库对话配置
    
    Attributes:
        intent_enabled: 是否启用意图识别
        intent_use_llm: 是否使用 LLM 进行意图识别
        query_rewrite_mode: Query 改写模式 (auto/always/never)
        rewrite_max_history_turns: 改写时参考的历史轮数
        rewrite_timeout_ms: 改写超时时间
        search_top_k: 检索数量
        search_rerank: 是否重排序
        search_score_threshold: 分数阈值
        search_mode: 检索模式
        prompt_max_history_turns: 传给 LLM 的历史轮数
        prompt_max_context_chars: 检索内容最大字符数
        chitchat_keywords: 闲聊关键词列表
    """
    # 意图识别
    intent_enabled: bool = Field(default=True, description="是否启用意图识别")
    intent_use_llm: bool = Field(default=False, description="是否使用LLM识别意图")
    
    # Query 改写
    query_rewrite_mode: str = Field(default="auto", description="改写模式: auto/always/never")
    rewrite_max_history_turns: int = Field(default=2, description="改写时参考的历史轮数")
    rewrite_timeout_ms: int = Field(default=3000, description="改写超时(毫秒)")
    
    # 检索参数
    search_top_k: int = Field(default=5, description="检索数量")
    search_rerank: bool = Field(default=True, description="是否重排序")
    search_score_threshold: float = Field(default=0.3, description="分数阈值")
    search_mode: str = Field(default="hybrid", description="检索模式")
    
    # Prompt 组装
    prompt_max_history_turns: int = Field(default=3, description="传给LLM的历史轮数")
    prompt_max_context_chars: int = Field(default=6000, description="检索内容最大字符")
    
    # 闲聊关键词
    chitchat_keywords: List[str] = Field(
        default_factory=lambda: [
            "好的", "谢谢", "明白了", "知道了", "了解", "OK", "ok",
            "嗯", "哦", "行", "可以", "没问题", "感谢", "辛苦",
            "太棒了", "很好", "不错", "懂了", "收到",
        ],
        description="闲聊关键词"
    )


class KBInfo(BaseModel):
    """
    知识库对话附加信息
    
    包含检索和处理的详细信息，用于调试和溯源。
    """
    search_performed: bool = Field(..., description="是否执行了检索")
    search_query: Optional[str] = Field(default=None, description="实际检索的query")
    query_rewritten: bool = Field(default=False, description="是否进行了改写")
    original_query: Optional[str] = Field(default=None, description="原始问题")
    intent: str = Field(..., description="识别的意图")
    sources: List[SourceReference] = Field(default_factory=list, description="引用来源")
    search_took_ms: Optional[float] = Field(default=None, description="检索耗时(毫秒)")


class KBChatResult(BaseModel):
    """
    知识库对话结果
    
    对话服务的完整输出。
    """
    content: str = Field(..., description="回复内容")
    kb_info: KBInfo = Field(..., description="知识库相关信息")
