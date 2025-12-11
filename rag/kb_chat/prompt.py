"""
Prompt 组装模块

本模块负责根据不同模型类型组装对话消息：
- GLM 系列：使用 observation 角色注入检索结果
- 通用模型：将检索结果嵌入 user 消息

支持的消息格式：
- GLM observation: {"role": "observation", "content": "【1†章节†文档】\n内容"}
- 通用格式: 检索结果嵌入 user 消息中
"""

from typing import List, Dict, Optional
from enum import Enum
from loguru import logger

from rag.kb_chat.types import KBChatConfig


class ModelFamily(str, Enum):
    """模型家族"""
    GLM = "glm"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    GENERIC = "generic"


# 默认 System Prompt
DEFAULT_KB_SYSTEM_PROMPT = """你是一个专业的知识库问答助手，严格根据检索到的知识库内容回答用户问题。

## 核心原则（必须严格遵守）
1. **只使用检索结果中明确存在的信息**：绝对禁止编造、推测或添加任何检索结果中没有的内容
2. **相关性判断**：如果检索结果与用户问题无关，必须明确告知用户"检索到的内容与您的问题不相关"，而不是强行从中提取信息
3. **忠实引用**：回答中的每个事实必须能在检索结果中找到原文依据

## 回答规范
1. 信息整合：归纳总结相关的检索内容，而非简单复制粘贴
2. 格式美观：善用列表、分段提高可读性
3. 引用标注：引用知识库内容时，在句末标注来源序号，格式为【1】【2】等
4. 语言一致：与用户问题语言保持一致
5. 坦诚告知：信息不足时请说明"知识库中未找到相关信息"

## 禁止行为
- 禁止从"员工招聘"相关内容回答"请假"问题
- 禁止编造具体数字、百分比、流程步骤等知识库中不存在的内容
- 禁止将检索结果中的概念错误关联

注意：检索结果的编号从1开始，请使用对应的数字序号进行引用。"""


# 无检索结果时的回复提示
NO_RESULT_HINT = """## 提示
知识库中未找到与问题相关的信息。请根据你的知识尽可能回答，如果无法回答请告知用户。"""


class PromptBuilder:
    """
    Prompt 组装器
    
    根据不同模型类型组装对话消息。
    
    Usage:
        >>> builder = PromptBuilder(config)
        >>> messages = builder.build(
        ...     model_family=ModelFamily.GLM,
        ...     user_query="什么是RAG？",
        ...     search_hits=[...],
        ...     history=[...]
        ... )
    """
    
    # 模型家族映射（模型名称关键词 -> 家族）
    MODEL_FAMILY_MAP = {
        "glm": ModelFamily.GLM,
        "chatglm": ModelFamily.GLM,
        "qwen": ModelFamily.QWEN,
        "deepseek": ModelFamily.DEEPSEEK,
    }
    
    def __init__(
        self,
        config: KBChatConfig,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化 Prompt 组装器
        
        Args:
            config: 对话配置
            system_prompt: 自定义 system prompt
        """
        self._config = config
        self._system_prompt = system_prompt or DEFAULT_KB_SYSTEM_PROMPT
    
    @classmethod
    def detect_model_family(cls, model_name: str) -> ModelFamily:
        """
        根据模型名称检测模型家族
        
        Args:
            model_name: 模型名称
            
        Returns:
            ModelFamily: 模型家族
        """
        if not model_name:
            return ModelFamily.GENERIC
        
        model_lower = model_name.lower()
        for keyword, family in cls.MODEL_FAMILY_MAP.items():
            if keyword in model_lower:
                return family
        
        return ModelFamily.GENERIC
    
    def build(
        self,
        model_family: ModelFamily,
        user_query: str,
        search_hits: List[Dict],
        history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        组装对话消息
        
        Args:
            model_family: 模型家族
            user_query: 用户问题
            search_hits: 检索结果列表，每个元素需包含:
                - doc_name: 文档名称
                - chapter: 章节标题
                - content: 内容
                - score: 分数
            history: 对话历史
            
        Returns:
            List[Dict]: 消息列表
        """
        if model_family == ModelFamily.GLM:
            return self._build_glm_messages(user_query, search_hits, history)
        else:
            return self._build_generic_messages(user_query, search_hits, history)
    
    def _build_glm_messages(
        self,
        user_query: str,
        search_hits: List[Dict],
        history: Optional[List[Dict]],
    ) -> List[Dict]:
        """
        构建 GLM 格式消息
        
        GLM 使用 observation 角色注入检索结果。
        
        消息结构：
        1. system prompt
        2. 历史对话（最近N轮，不含 observation）
        3. 当前 user 问题
        4. observation（检索结果）
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        
        # 添加历史对话
        if history:
            processed_history = self._process_history(history)
            messages.extend(processed_history)
        
        # 添加当前用户问题
        messages.append({"role": "user", "content": user_query})
        
        # 添加检索结果作为 observation
        if search_hits:
            observation_content = self._format_search_results_glm(search_hits)
            messages.append({
                "role": "observation",
                "content": observation_content
            })
        else:
            # 无检索结果时，添加提示
            messages.append({
                "role": "observation",
                "content": "未找到相关知识库内容。"
            })
        
        return messages
    
    def _build_generic_messages(
        self,
        user_query: str,
        search_hits: List[Dict],
        history: Optional[List[Dict]],
    ) -> List[Dict]:
        """
        构建通用格式消息
        
        将检索结果嵌入 user 消息中。
        
        消息结构：
        1. system prompt
        2. 历史对话（最近N轮）
        3. 当前 user 问题（包含检索结果）
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        
        # 添加历史对话
        if history:
            processed_history = self._process_history(history)
            messages.extend(processed_history)
        
        # 构建包含检索结果的 user 消息
        if search_hits:
            context = self._format_search_results_generic(search_hits)
            combined_content = f"""## 知识库检索结果
{context}

## 用户问题
{user_query}

请根据以上检索结果回答用户问题。如果检索结果与问题无关，请说明。"""
        else:
            combined_content = f"""{NO_RESULT_HINT}

## 用户问题
{user_query}"""
        
        messages.append({"role": "user", "content": combined_content})
        
        return messages
    
    def _process_history(self, history: List[Dict]) -> List[Dict]:
        """
        处理历史对话
        
        - 限制轮次
        - 过滤 observation
        - 验证角色
        
        Args:
            history: 原始历史
            
        Returns:
            List[Dict]: 处理后的历史
        """
        max_turns = self._config.prompt_max_history_turns
        
        # 过滤 observation 和无效角色
        valid_history = [
            msg for msg in history
            if msg.get("role") in ("user", "assistant")
        ]
        
        # 限制轮次（每轮 2 条消息）
        if len(valid_history) > max_turns * 2:
            valid_history = valid_history[-(max_turns * 2):]
        
        return valid_history
    
    def _format_search_results_glm(self, hits: List[Dict]) -> str:
        """
        格式化检索结果（GLM observation 格式）
        
        格式: 【{序号}†{章节标题}†{文档名}】\n{内容}
        
        Args:
            hits: 检索结果
            
        Returns:
            str: 格式化后的文本
        """
        max_chars = self._config.prompt_max_context_chars
        parts = []
        total_chars = 0
        
        for i, hit in enumerate(hits, 1):
            doc_name = hit.get("doc_name", "未知文档")
            chapter = hit.get("chapter", "")
            content = hit.get("content", "")
            
            # 构建标题
            if chapter:
                header = f"【{i}†{chapter}†{doc_name}】"
            else:
                header = f"【{i}†{doc_name}】"
            
            entry = f"{header}\n{content}"
            
            # 检查长度限制
            if total_chars + len(entry) > max_chars:
                # 截断当前条目或跳过
                remaining = max_chars - total_chars - len(header) - 10
                if remaining > 100:
                    content = content[:remaining] + "..."
                    entry = f"{header}\n{content}"
                    parts.append(entry)
                break
            
            parts.append(entry)
            total_chars += len(entry) + 2  # +2 for \n\n
        
        return "\n\n".join(parts)
    
    def _format_search_results_generic(self, hits: List[Dict]) -> str:
        """
        格式化检索结果（通用格式）
        
        格式: [来源{序号}: {文档名} - {章节}]\n{内容}
        
        Args:
            hits: 检索结果
            
        Returns:
            str: 格式化后的文本
        """
        max_chars = self._config.prompt_max_context_chars
        parts = []
        total_chars = 0
        
        for i, hit in enumerate(hits, 1):
            doc_name = hit.get("doc_name", "未知文档")
            chapter = hit.get("chapter", "")
            content = hit.get("content", "")
            
            # 构建来源标签
            if chapter:
                source = f"[来源{i}: {doc_name} - {chapter}]"
            else:
                source = f"[来源{i}: {doc_name}]"
            
            entry = f"{source}\n{content}"
            
            # 检查长度限制
            if total_chars + len(entry) > max_chars:
                remaining = max_chars - total_chars - len(source) - 10
                if remaining > 100:
                    content = content[:remaining] + "..."
                    entry = f"{source}\n{content}"
                    parts.append(entry)
                break
            
            parts.append(entry)
            total_chars += len(entry) + 2
        
        return "\n\n".join(parts)
