"""Qwen3 系列模型策略（支持深度思考模式）"""

from typing import List, Dict
from strategies.qwen.base import QwenBaseStrategy
from strategies.protocol import StrategyInput, PromptOutput


class Qwen3Strategy(QwenBaseStrategy):
    """
    Qwen3 系列策略。
    
    Qwen3 支持 enable_thinking 参数控制深度思考模式：
    - enable_thinking=True: 开启深度思考，模型自动生成思考内容
    - enable_thinking=False: 关闭深度思考，通过空思考标签通知模型跳过思考
    
    参考: https://www.modelscope.cn/models/Qwen/Qwen3-4B
    """

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """
        将消息列表转换为 Qwen3 模型输入的 prompt。
        
        参数:
            messages: 消息列表
            enable_thinking: 是否启用深度思考模式，默认 True
        """
        enable_thinking = kwargs.get("enable_thinking", True)
        IM_START = "<" + "|im_start|" + ">"
        IM_END = "<" + "|im_end|" + ">"
        THINK_START = "<" + "think" + ">"
        THINK_END = "<" + "/think" + ">"
        
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            parts.append(f"{IM_START}{role}\n{content}{IM_END}")
        
        # Qwen3 深度思考模式控制
        if enable_thinking:
            parts.append(f"{IM_START}assistant\n{THINK_START}")
        else:
            # Qwen3 官方要求：关闭思考需添加空的 <think></think> 标签
            parts.append(f"{IM_START}assistant\n{THINK_START}\n\n{THINK_END}\n")
        
        return "\n".join(parts)

    def build_prompt(self, input: StrategyInput) -> PromptOutput:
        """
        构建 Prompt，支持深度思考模式。
        
        当 enable_thinking=True 时，设置 thinking_prefix 为 "<think>"，
        由基类 execute() 统一处理前缀输出。
        """
        prompt = self.apply_chat_template(
            input.messages,
            enable_thinking=input.enable_thinking,
            **input.extra
        )
        
        THINK_START = "<" + "think" + ">"
        thinking_prefix = THINK_START if input.enable_thinking else None
        
        return PromptOutput(
            prompt=prompt,
            thinking_prefix=thinking_prefix,
            stop_words=input.stop or self.get_default_stop_words(),
        )
