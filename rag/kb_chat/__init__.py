"""
知识库对话模块

本模块提供知识库对话的完整能力，包括：
- 意图识别：判断用户输入类型，决定检索策略
- Query 改写：多轮场景下优化检索 query
- Prompt 组装：根据模型类型组装消息
- 对话服务：整合以上能力的统一入口
"""

from rag.kb_chat.types import (
    UserIntent,
    IntentResult,
    RewriteResult,
    KBChatResult,
)
from core.config import KBChatSettings
from rag.kb_chat.intent import IntentRouter
from rag.kb_chat.rewriter import QueryRewriter
from rag.kb_chat.prompt import PromptBuilder, ModelFamily
from rag.kb_chat.service import KBChatService


__all__ = [
    # 类型
    "UserIntent",
    "IntentResult",
    "RewriteResult",
    "KBChatResult",
    # 配置管理
    "KBChatSettings",
    # 组件
    "IntentRouter",
    "QueryRewriter",
    "PromptBuilder",
    "ModelFamily",
    # 服务
    "KBChatService",
]
