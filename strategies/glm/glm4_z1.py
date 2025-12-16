"""GLM4-Z1 系列模型策略"""

from typing import List, Dict, Union, Iterator
from strategies.glm.base import GLMBaseStrategy


class GLM4Z1Strategy(GLMBaseStrategy):
    """GLM4-Z1 系列策略（强制深度思考）。"""

    def apply_chat_template(self, messages: List[Dict], enable_thinking: bool = True) -> str:
        prompt = super().apply_chat_template(messages)
        if enable_thinking:
            return prompt.rstrip("\n") + "\n<think>"
        return prompt

    def generate(self, engine, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """
        生成回复（包含 Z1 特定逻辑）。
        
        Args:
            engine: LLM 引擎实例
            messages: 消息列表
            stream: 是否流式输出
            **kwargs: 生成参数，enable_thinking 默认为 True
        """
        enable_thinking = kwargs.pop("enable_thinking", True)
        
        prompt = self.apply_chat_template(messages, enable_thinking=enable_thinking)

        stop_words = ["<|user|>", "<|assistant|>", "<|system|>", "<|observation|>"]
        kwargs.setdefault("stop", stop_words)
        
        if stream:
            return self._generate_stream(engine, prompt, enable_thinking, **kwargs)
            
        output = engine.generate(prompt, stream=stream, **kwargs)
        if enable_thinking:
            if output.lstrip().startswith("<think>"):
                return output
            return "<think>" + output
        return output

    def _generate_stream(self, engine, prompt, enable_thinking, **kwargs):
        first = True
        for chunk in engine.generate(prompt, stream=True, **kwargs):
            if first:
                first = False
                if enable_thinking:
                    if chunk.lstrip().startswith("<think>"):
                        yield chunk
                        continue
                    yield "<think>"
            yield chunk
