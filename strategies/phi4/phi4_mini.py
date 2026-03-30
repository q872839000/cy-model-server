"""Phi-4-mini 系列模型策略（3.8B）

Phi-4-mini-instruct 使用 Phi-3 风格的对话模板，
采用 system / user / assistant / end 角色标签。

对话格式：
    system_tag + 系统提示 + end_tag + user_tag + 问题 + end_tag + assistant_tag

工具调用格式：
    工具定义通过 tool_open / tool_close 标签注入到 system 消息中。

适用模型：
- microsoft/Phi-4-mini-instruct (3.8B)

参考：https://huggingface.co/microsoft/Phi-4-mini-instruct
"""

from typing import List, Dict, Any
import json
import re
import uuid
from strategies.base import (
    LLMStrategy,
    _extract_allowed_tool_names,
    _normalize_tool_call,
    _parse_tool_calls_from_text,
    _tool_choice_instruction,
)


class Phi4MiniStrategy(LLMStrategy):
    """
    Phi-4-mini (3.8B) 策略。

    工具调用：工具定义通过 tool_open/tool_close 标签注入到 system 消息中，
    模型的工具调用输出以 tool_call 标签包裹。
    """

    # 特殊标记定义
    _SYSTEM = "<" + "|system|" + ">"
    _USER = "<" + "|user|" + ">"
    _ASSISTANT = "<" + "|assistant|" + ">"
    _END = "<" + "|end|" + ">"
    _TOOL_OPEN = "<" + "|tool|" + ">"
    _TOOL_CLOSE = "<" + "|/tool|" + ">"
    _TOOL_CALL = "<" + "|tool_call|" + ">"
    _ENDOFTEXT = "<" + "|endoftext|" + ">"

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """将消息列表转换为 Phi-4-mini 模型输入的 prompt"""
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            if role == "system":
                parts.append(f"{self._SYSTEM}{content}{self._END}")
            elif role == "user":
                parts.append(f"{self._USER}{content}{self._END}")
            elif role == "assistant":
                parts.append(f"{self._ASSISTANT}{content}{self._END}")
            elif role in ("tool", "observation"):
                parts.append(f"{self._USER}{content}{self._END}")
            else:
                parts.append(f"{self._USER}{content}{self._END}")
        parts.append(self._ASSISTANT)
        return "".join(parts)

    def get_default_stop_words(self) -> List[str]:
        return [self._END, self._ENDOFTEXT]

    def supports_native_tools(self) -> bool:
        """Phi-4-mini 的 chat_template 使用 message 级别的 tools 字段注入工具定义，
        与 GLM-4 类似检查 item['tools']，而非 HuggingFace 标准的全局 tools 参数。
        返回 False 强制走 inject_tools_into_messages 回退路径。
        """
        return False

    def parse_tool_calls(
        self,
        text: str,
        tools: List[Dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """解析 Phi-4-mini 模型的工具调用输出

        Phi-4-mini 训练时使用 tool/tool_close 标签注入工具定义，
        模型输出工具调用为 JSON 数组格式：[{"name": "...", "arguments": {...}}]

        解析策略（按优先级）：
        1. JSON 数组格式（Phi-4-mini 原生输出）
        2. 回退到基类默认解析（<tool_call> 标签 + 独立 JSON 对象）
        """
        allowed_tool_names = _extract_allowed_tool_names(tools)

        # 策略 1：JSON 数组格式 [{"name": ..., "arguments": ...}]
        # 查找文本中的 JSON 数组
        bracket_start = text.find('[')
        if bracket_start != -1:
            # 从 '[' 开始尝试找到配对的 ']'
            depth = 0
            for i in range(bracket_start, len(text)):
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        candidate = text[bracket_start:i + 1]
                        try:
                            arr = json.loads(candidate)
                            if isinstance(arr, list) and arr:
                                tool_calls = []
                                for obj in arr:
                                    if isinstance(obj, dict) and "name" in obj:
                                        normalized = _normalize_tool_call(obj)
                                        name = normalized.get("function", {}).get("name", "")
                                        if allowed_tool_names and name not in allowed_tool_names:
                                            continue
                                        tool_calls.append(normalized)
                                if tool_calls:
                                    return tool_calls
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break

        # 策略 2：回退到基类默认解析
        return _parse_tool_calls_from_text(
            text,
            allowed_tool_names=allowed_tool_names,
        )

    def inject_tools_into_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = None,
    ) -> List[Dict[str, Any]]:
        """Phi-4-mini 使用原生 tool 标签注入工具定义

        将工具定义以 Phi-4-mini 原生格式注入到 system 消息中，
        而非使用基类的通用文本注入方式。
        """
        tool_defs = []
        for tool in tools:
            fn = tool.get("function", tool)
            tool_defs.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        tool_json = json.dumps(tool_defs, ensure_ascii=False)
        injection = f"{self._TOOL_OPEN} {tool_json} {self._TOOL_CLOSE}"
        choice_instruction = _tool_choice_instruction(tool_choice)
        if choice_instruction:
            injection = choice_instruction + "\n\n" + injection

        messages = [dict(m) for m in messages]
        if messages and messages[0].get("role") == "system":
            messages[0] = dict(messages[0])
            base_content = messages[0].get("content") or ""
            if base_content:
                messages[0]["content"] = base_content + "\n\n" + injection
            else:
                messages[0]["content"] = injection
        else:
            messages.insert(0, {
                "role": "system",
                "content": "You are a helpful assistant with some tools.\n\n" + injection,
            })
        return messages
