from abc import ABC, abstractmethod
from typing import List, Dict, Iterator, Union, TYPE_CHECKING

from strategies.protocol import StrategyInput, StrategyOutput, PromptOutput

if TYPE_CHECKING:
    from engines.base import LLMEngine


class LLMStrategy(ABC):
    """
    LLM 策略接口，负责对话模板与生成调用。
    
    设计原则：
    1. build_prompt() 负责构建 prompt，返回 PromptOutput
    2. execute() 负责调用引擎生成，返回 StrategyOutput
    3. 所有子类禁止在 execute() 中直接 yield 特殊前缀
    4. thinking_prefix 由 PromptOutput 声明，调用方统一处理
    """

    @abstractmethod
    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """将消息列表转换为模型输入的prompt（兼容旧接口）"""
        ...

    def get_default_stop_words(self) -> List[str]:
        """获取默认停止词列表，子类可覆盖"""
        return []

    def build_prompt(self, input: StrategyInput) -> PromptOutput:
        """
        构建 Prompt（新协议接口）
        
        子类可覆盖此方法以实现特定的 prompt 构建逻辑。
        默认实现调用 apply_chat_template。
        
        Args:
            input: 策略输入
            
        Returns:
            PromptOutput: 包含 prompt、thinking_prefix、stop_words 等
        """
        prompt = self.apply_chat_template(
            input.messages,
            enable_thinking=input.enable_thinking,
            **input.extra
        )
        return PromptOutput(
            prompt=prompt,
            thinking_prefix=None,
            stop_words=input.stop or self.get_default_stop_words() or None,
        )

    def execute(self, engine: "LLMEngine", input: StrategyInput) -> StrategyOutput:
        """
        执行生成（新协议接口）
        
        统一的生成入口，返回 StrategyOutput。
        子类一般不需要覆盖此方法，只需覆盖 build_prompt()。
        
        Args:
            engine: LLM 引擎实例
            input: 策略输入
            
        Returns:
            StrategyOutput: 统一的输出结构
        """
        prompt_output = self.build_prompt(input)
        
        gen_kwargs = {
            "max_tokens": input.max_tokens,
            "temperature": input.temperature,
            "top_p": input.top_p,
        }
        if prompt_output.stop_words:
            gen_kwargs["stop"] = prompt_output.stop_words
        
        if input.stream:
            iterator = self._wrap_stream(
                engine.generate(prompt_output.prompt, stream=True, **gen_kwargs),
                prompt_output.thinking_prefix,
            )
            return StrategyOutput(
                stream_iterator=iterator,
                thinking_content=None,
                metadata={"thinking_prefix": prompt_output.thinking_prefix},
            )
        else:
            text = engine.generate(prompt_output.prompt, stream=False, **gen_kwargs)
            final_text = text
            if prompt_output.thinking_prefix:
                final_text = prompt_output.thinking_prefix + text
            return StrategyOutput(
                text=final_text,
                thinking_content=None,
                metadata={"thinking_prefix": prompt_output.thinking_prefix},
            )

    def _wrap_stream(
        self, iterator: Iterator[str], thinking_prefix: str | None
    ) -> Iterator[str]:
        """包装流式输出，统一处理 thinking_prefix"""
        if thinking_prefix:
            yield thinking_prefix
        yield from iterator

    def generate(self, engine, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """
        生成响应（兼容旧接口，内部委托给 execute）
        
        Args:
            engine: LLM引擎实例
            messages: 消息列表
            stream: 是否流式输出
            **kwargs: 其他生成参数
        
        Returns:
            stream=False: 返回完整文本
            stream=True: 返回文本片段迭代器
        """
        input = StrategyInput.from_kwargs(messages, stream=stream, **kwargs)
        output = self.execute(engine, input)
        
        if stream:
            return output.iter_chunks()
        return output.get_text()


class GenericChatStrategy(LLMStrategy):
    """通用聊天策略：简单拼接user与system上下文。适用于大多数兼容模型。"""

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def get_default_stop_words(self) -> List[str]:
        """通用策略的停止词：仅包含通用角色标记，不包含模型特定标记"""
        return [
            "\nsystem:",
            "\nuser:",
            "\nassistant:",
            "\nsystem：",
            "\nuser：",
            "\nassistant：",
        ]
