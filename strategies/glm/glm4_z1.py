"""GLM4-Z1 series model strategy"""

from typing import List, Dict, Union, Iterator
from strategies.glm.base import GLMBaseStrategy


class GLM4Z1Strategy(GLMBaseStrategy):
    """GLM4-Z1 series strategy with enforced thinking."""

    def apply_chat_template(self, messages: List[Dict]) -> str:
        prompt = super().apply_chat_template(messages)
        # GLM-Z1 enforced thinking: add think tag after assistant
        return prompt + "\n<think>\n"

    def generate(self, engine, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Generate response with Z1 specific parameters."""
        # Z1 model benefits from sampling for reasoning
        kwargs.setdefault("do_sample", True)
        kwargs.setdefault("temperature", 0.6)
        kwargs.setdefault("top_p", 0.9)
        
        prompt = self.apply_chat_template(messages)
        return engine.generate(prompt, stream=stream, **kwargs)
