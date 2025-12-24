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
from typing import Optional, List
from pydantic import BaseModel, Field
from models import KBChatInfo


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


class KBChatResult(BaseModel):
    """
    知识库对话结果
    
    对话服务的完整输出。
    """
    content: str = Field(..., description="回复内容")
    kb_info: KBChatInfo = Field(..., description="知识库相关信息")
