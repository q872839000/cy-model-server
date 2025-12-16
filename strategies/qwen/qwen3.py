"""Qwen3 系列模型策略（支持深度思考模式）"""

from typing import List, Dict, Iterator, Union
from strategies.qwen.base import QwenBaseStrategy


class Qwen3Strategy(QwenBaseStrategy):
    """
    Qwen3 系列策略。
    
    Qwen3 支持 enable_thinking 参数控制深度思考模式：
    - enable_thinking=True: 开启深度思考，模型自动生成 <think>...</think> 包裹的思考内容
    - enable_thinking=False: 关闭深度思考，通过添加空思考标签跳过思考
    
    参考: https://www.modelscope.cn/models/Qwen/Qwen3-4B
    """

    def apply_chat_template(self, messages: List[Dict], enable_thinking: bool = True) -> str:
        """
        将消息列表转换为 Qwen3 模型输入的 prompt。
        
        参数:
            messages: 消息列表
            enable_thinking: 是否启用深度思考模式，默认 True
        """
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # Qwen3 深度思考模式控制
        # 根据官方文档：enable_thinking=False 时添加空思考标签让模型跳过思考
        if enable_thinking:
            parts.append("<|im_start|>assistant\n<think>")
        else:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n")
        
        return "\n".join(parts)

    def generate(self, engine, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """
        生成响应（支持 enable_thinking 参数）
        
        参数:
            engine: LLM引擎实例
            messages: 消息列表
            stream: 是否流式输出
            enable_thinking: 是否启用深度思考模式（从 kwargs 中获取）
            **kwargs: 其他生成参数
        
        返回:
            stream=False: 返回完整文本
            stream=True: 返回文本片段迭代器
        """
        enable_thinking = kwargs.pop("enable_thinking", True)
        prompt = self.apply_chat_template(messages, enable_thinking=enable_thinking)
        
        # stop tokens 对 Qwen3 很关键：用于防止模型进入自问自答/自我续写
        stop_words = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
        kwargs.setdefault("stop", stop_words)
        
        if stream:
            return self._generate_stream(engine, prompt, enable_thinking, **kwargs)
        
        output = engine.generate(prompt, stream=False, **kwargs)
        if enable_thinking:
            return "<think>" + output
        return output

    def _generate_stream(self, engine, prompt, enable_thinking, **kwargs):
        if enable_thinking:
            yield "<think>"
        for chunk in engine.generate(prompt, stream=True, **kwargs):
            yield chunk
