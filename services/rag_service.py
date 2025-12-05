"""
RAG服务：检索增强生成（Retrieval-Augmented Generation）

功能：
- 结合向量检索和LLM生成，基于知识库回答问题
- 支持多种RAG策略和提示词模板
- 支持流式和非流式生成
- 支持引用来源
"""

from typing import List, Dict, Optional, Any, Iterator
from loguru import logger
from dataclasses import dataclass
from enum import Enum

from models.schemas import RetrievalItem, ChatMessage
from services.advanced_retrieval_service import (
    ADVANCED_RETRIEVAL_SERVICE,
    RecallConfig,
    RerankConfig,
    RecallStrategy,
    RerankStrategy
)
from core.registry import REGISTRY
from services.container import CONTAINER


class RAGStrategy(str, Enum):
    """RAG策略枚举"""
    SIMPLE = "simple"  # 简单拼接：直接将检索结果拼接到prompt
    CITATION = "citation"  # 引用式：生成时标注来源
    CONVERSATIONAL = "conversational"  # 对话式：保持上下文，支持多轮对话
    REFINE = "refine"  # 精炼式：逐步精炼答案


class PromptTemplate(str, Enum):
    """提示词模板枚举"""
    DEFAULT = "default"
    PROFESSIONAL = "professional"  # 专业的、正式的
    FRIENDLY = "friendly"  # 友好的、随和的
    CONCISE = "concise"  # 简洁的
    DETAILED = "detailed"  # 详细的


@dataclass
class RAGConfig:
    """RAG配置"""
    strategy: RAGStrategy = RAGStrategy.SIMPLE
    prompt_template: PromptTemplate = PromptTemplate.DEFAULT
    max_context_length: int = 3000  # 最大上下文长度（字符数）
    include_citations: bool = True  # 是否包含引用
    system_prompt: Optional[str] = None  # 自定义系统提示


class PromptBuilder:
    """提示词构建器"""
    
    TEMPLATES = {
        PromptTemplate.DEFAULT: {
            "system": "你是一个有帮助的AI助手。请根据以下提供的参考资料回答用户的问题。如果参考资料中没有相关信息，请诚实地告知用户。",
            "context_prefix": "参考资料：\n",
            "context_format": "[{idx}] {content}\n",
            "question_prefix": "\n用户问题：{query}\n\n请基于上述参考资料回答："
        },
        PromptTemplate.PROFESSIONAL: {
            "system": "您是一位专业的知识顾问。请基于提供的权威参考资料，为用户提供准确、专业的答案。",
            "context_prefix": "权威参考资料：\n",
            "context_format": "【资料{idx}】{content}\n",
            "question_prefix": "\n咨询问题：{query}\n\n专业解答："
        },
        PromptTemplate.FRIENDLY: {
            "system": "你是一个友好、乐于助人的AI伙伴。用轻松、易懂的方式回答问题。",
            "context_prefix": "我找到了一些相关信息：\n",
            "context_format": "{idx}. {content}\n",
            "question_prefix": "\n你的问题是：{query}\n\n让我来帮你："
        },
        PromptTemplate.CONCISE: {
            "system": "请用最简洁的方式回答问题，直接给出答案要点。",
            "context_prefix": "参考信息：\n",
            "context_format": "• {content}\n",
            "question_prefix": "\n问题：{query}\n简答："
        },
        PromptTemplate.DETAILED: {
            "system": "请提供详细、全面的答案，充分利用参考资料中的信息。",
            "context_prefix": "详细参考资料：\n",
            "context_format": "参考资料{idx}：\n{content}\n\n",
            "question_prefix": "问题：{query}\n\n详细回答："
        }
    }
    
    @classmethod
    def build_prompt(
        cls,
        query: str,
        retrieved_items: List[RetrievalItem],
        config: RAGConfig
    ) -> tuple[str, List[ChatMessage]]:
        """
        构建提示词
        
        Returns:
            (system_prompt, messages)
        """
        template = cls.TEMPLATES.get(config.prompt_template, cls.TEMPLATES[PromptTemplate.DEFAULT])
        
        # 系统提示
        system_prompt = config.system_prompt or template["system"]
        
        # 构建上下文
        context_parts = [template["context_prefix"]]
        total_length = 0
        used_items = []
        
        for idx, item in enumerate(retrieved_items, 1):
            content = item.document.content
            # 截断过长的内容
            if len(content) > 500:
                content = content[:500] + "..."
            
            item_text = template["context_format"].format(idx=idx, content=content)
            
            # 检查是否超过最大长度
            if total_length + len(item_text) > config.max_context_length:
                break
            
            context_parts.append(item_text)
            used_items.append((idx, item))
            total_length += len(item_text)
        
        context = "".join(context_parts)
        
        # 构建问题部分
        question_part = template["question_prefix"].format(query=query)
        
        # 组合完整的用户消息
        user_message = context + question_part
        
        # 构建消息列表
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message)
        ]
        
        return system_prompt, messages


class RAGService:
    """RAG服务：检索增强生成"""
    
    def __init__(self):
        self._retrieval_service = ADVANCED_RETRIEVAL_SERVICE
    
    def generate(
        self,
        query: str,
        collection_name: str,
        model_name: str,
        embedding_model: str,
        rag_config: Optional[RAGConfig] = None,
        recall_config: Optional[RecallConfig] = None,
        rerank_config: Optional[RerankConfig] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        RAG生成（非流式）
        
        Args:
            query: 用户问题
            collection_name: 知识库集合名称
            model_name: LLM模型名称
            embedding_model: Embedding模型名称
            rag_config: RAG配置
            recall_config: 召回配置
            rerank_config: 重排配置
            generation_params: 生成参数（max_tokens, temperature等）
        
        Returns:
            包含答案、来源等信息的字典
        """
        # 默认配置
        if rag_config is None:
            rag_config = RAGConfig()
        if recall_config is None:
            recall_config = RecallConfig(top_k=5)
        if rerank_config is None:
            rerank_config = RerankConfig(final_top_k=3)
        if generation_params is None:
            generation_params = {}
        
        logger.info(f"RAG生成开始: query='{query[:50]}...', collection={collection_name}, model={model_name}")
        
        # 第一步：检索相关文档
        retrieved_items = self._retrieval_service.retrieve(
            query=query,
            collection_name=collection_name,
            embedding_model=embedding_model,
            recall_config=recall_config,
            rerank_config=rerank_config
        )
        
        if not retrieved_items:
            logger.warning("未检索到相关文档")
            return {
                "query": query,
                "answer": "抱歉，我在知识库中没有找到相关信息来回答您的问题。",
                "sources": [],
                "retrieved_count": 0
            }
        
        logger.info(f"检索到 {len(retrieved_items)} 条相关文档")
        
        # 第二步：构建提示词
        system_prompt, messages = PromptBuilder.build_prompt(
            query=query,
            retrieved_items=retrieved_items,
            config=rag_config
        )
        
        # 第三步：LLM生成
        engine, strategy = CONTAINER.get_llm_and_strategy(model_name)
        if not engine:
            raise ValueError(f"未找到LLM模型: {model_name}")
        
        # 转换消息格式
        messages_dict = [{"role": m.role, "content": m.content} for m in messages]
        
        # 生成答案
        answer = strategy.generate(
            engine,
            messages_dict,
            max_tokens=generation_params.get("max_tokens", 512),
            temperature=generation_params.get("temperature", 0.7),
            top_p=generation_params.get("top_p", 0.95)
        )
        
        logger.info(f"RAG生成完成: answer_length={len(answer)}")
        
        # 第四步：构建返回结果
        result = {
            "query": query,
            "answer": answer,
            "sources": [],
            "retrieved_count": len(retrieved_items)
        }
        
        # 添加来源信息
        if rag_config.include_citations:
            for idx, item in enumerate(retrieved_items, 1):
                result["sources"].append({
                    "index": idx,
                    "content": item.document.content[:200] + "..." if len(item.document.content) > 200 else item.document.content,
                    "score": item.score,
                    "metadata": item.document.metadata,
                    "document_id": item.document.id
                })
        
        return result
    
    def generate_stream(
        self,
        query: str,
        collection_name: str,
        model_name: str,
        embedding_model: str,
        rag_config: Optional[RAGConfig] = None,
        recall_config: Optional[RecallConfig] = None,
        rerank_config: Optional[RerankConfig] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """
        RAG流式生成
        
        Yields:
            生成的文本片段
        """
        # 默认配置
        if rag_config is None:
            rag_config = RAGConfig()
        if recall_config is None:
            recall_config = RecallConfig(top_k=5)
        if rerank_config is None:
            rerank_config = RerankConfig(final_top_k=3)
        if generation_params is None:
            generation_params = {}
        
        logger.info(f"RAG流式生成开始: query='{query[:50]}...', collection={collection_name}")
        
        # 第一步：检索相关文档
        retrieved_items = self._retrieval_service.retrieve(
            query=query,
            collection_name=collection_name,
            embedding_model=embedding_model,
            recall_config=recall_config,
            rerank_config=rerank_config
        )
        
        if not retrieved_items:
            yield "抱歉，我在知识库中没有找到相关信息来回答您的问题。"
            return
        
        logger.info(f"检索到 {len(retrieved_items)} 条相关文档")
        
        # 第二步：构建提示词
        system_prompt, messages = PromptBuilder.build_prompt(
            query=query,
            retrieved_items=retrieved_items,
            config=rag_config
        )
        
        # 第三步：LLM流式生成
        engine, strategy = CONTAINER.get_llm_and_strategy(model_name)
        if not engine:
            raise ValueError(f"未找到LLM模型: {model_name}")
        
        # 转换消息格式
        messages_dict = [{"role": m.role, "content": m.content} for m in messages]
        
        # 流式生成答案
        for chunk in strategy.generate_stream(
            engine,
            messages_dict,
            max_tokens=generation_params.get("max_tokens", 512),
            temperature=generation_params.get("temperature", 0.7),
            top_p=generation_params.get("top_p", 0.95)
        ):
            yield chunk
        
        logger.info("RAG流式生成完成")


# 全局实例
RAG_SERVICE = RAGService()

