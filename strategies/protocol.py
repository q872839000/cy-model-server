"""
策略协议定义

本模块定义 Strategy 层的统一输入输出协议，确保所有策略实现行为一致。

设计原则：
1. StrategyInput 封装所有输入参数，类型安全
2. StrategyOutput 封装输出结果，统一格式
3. 所有 strategy 禁止直接 print / yield 特殊前缀
4. Engine 只处理 StrategyOutput
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Union


@dataclass
class StrategyInput:
    """
    策略输入协议
    
    封装对话生成的所有输入参数，提供类型安全和默认值。
    
    Attributes:
        messages: 对话消息列表，格式 [{"role": "user", "content": "..."}]
        stream: 是否流式输出
        max_tokens: 最大生成 token 数
        temperature: 生成温度，越高越随机
        top_p: 核采样参数
        stop: 停止词列表
        enable_thinking: 是否启用深度思考模式（部分模型支持）
        tools: OpenAI 格式的工具定义列表（function calling 支持）
        tool_choice: 工具选择策略（"auto"/"none"/"required" 或指定工具）
        extra: 扩展参数，用于模型特定配置
    """
    messages: List[Dict[str, Any]]
    stream: bool = False
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    enable_thinking: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_kwargs(
        cls,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> "StrategyInput":
        """
        从 kwargs 构造 StrategyInput
        
        便捷方法，将松散的 kwargs 转换为类型安全的 StrategyInput。
        已知参数会被提取，其余放入 extra。
        
        Args:
            messages: 对话消息列表
            stream: 是否流式输出
            **kwargs: 其他生成参数
            
        Returns:
            StrategyInput 实例
        """
        return cls(
            messages=messages,
            stream=stream,
            max_tokens=kwargs.pop("max_tokens", 256),
            temperature=kwargs.pop("temperature", 0.7),
            top_p=kwargs.pop("top_p", 0.95),
            top_k=kwargs.pop("top_k", None),
            stop=kwargs.pop("stop", None),
            enable_thinking=kwargs.pop("enable_thinking", False),
            tools=kwargs.pop("tools", None),
            tool_choice=kwargs.pop("tool_choice", None),
            extra=kwargs,
        )


@dataclass
class StrategyOutput:
    """
    策略输出协议
    
    封装生成结果，统一所有策略的输出格式。
    
    Attributes:
        text: 生成的完整文本（非流式时使用）
        stream_iterator: 流式文本迭代器（流式时使用）
        thinking_content: 思考内容（如果启用了 enable_thinking）
        tool_calls: 模型请求的工具调用列表（function calling 时返回）
        finish_reason: 生成结束原因 ("stop"/"length"/"tool_calls")
        metadata: 输出元数据，如 token 统计等
    """
    text: Optional[str] = None
    stream_iterator: Optional[Iterator[str]] = None
    thinking_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_stream(self) -> bool:
        """是否为流式输出"""
        return self.stream_iterator is not None
    
    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return bool(self.tool_calls)

    def get_text(self) -> str:
        """
        获取完整文本
        
        非流式时直接返回 text，流式时消费迭代器拼接。
        注意：流式模式下此方法会消费迭代器。
        
        Returns:
            完整生成文本
        """
        if self.text is not None:
            return self.text
        if self.stream_iterator is not None:
            chunks = list(self.stream_iterator)
            self.text = "".join(chunks)
            self.stream_iterator = None
            return self.text
        return ""
    
    def iter_chunks(self) -> Iterator[str]:
        """
        迭代文本块
        
        流式时返回迭代器，非流式时将 text 包装为单元素迭代器。
        
        Yields:
            文本块
        """
        if self.stream_iterator is not None:
            yield from self.stream_iterator
        elif self.text:
            yield self.text


@dataclass
class PromptOutput:
    """
    Prompt 构建输出
    
    封装 apply_chat_template 的结果，支持扩展元数据。
    
    Attributes:
        prompt: 构建后的 prompt 字符串
        thinking_prefix: 思考前缀（如 "<think>"），由调用方决定如何使用
        stop_words: 建议的停止词列表
        metadata: 额外元数据
    """
    prompt: str
    thinking_prefix: Optional[str] = None
    stop_words: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
