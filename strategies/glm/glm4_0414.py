"""GLM4-0414 series model strategy (supports deep thinking)"""

from typing import List, Dict, Iterator, Union
from strategies.glm.base import GLMBaseStrategy


class GLM4_0414Strategy(GLMBaseStrategy):
    """
    GLM4-0414 series strategy.
    
    Supports deep thinking mode via do_sample parameter.
    """

    def generate(self, engine, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Generate response with thinking mode support."""
        enable_thinking = kwargs.pop("enable_thinking", True)
        if enable_thinking:
            kwargs.setdefault("do_sample", True)
            kwargs.setdefault("temperature", 0.7)
        else:
            kwargs.setdefault("do_sample", False)
        
        prompt = self.apply_chat_template(messages)
        return engine.generate(prompt, stream=stream, **kwargs)
