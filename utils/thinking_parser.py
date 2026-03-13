"""
<think> 标签解析工具

提供非流式和流式两种解析方式，将模型输出中的 <think>...</think> 标签
拆分为 reasoning_content 和 content 两个独立字段。

适用于 DeepSeek R1 / Qwen3 / GLM4-Z1 等支持深度思考的模型。
"""

import re
from typing import Optional, Tuple, List


# ==================== 非流式解析 ====================


def parse_thinking_content(text: str) -> Tuple[Optional[str], str]:
    """从完整文本中提取 <think>...</think> 内容

    将包含 <think> 标签的文本拆分为推理内容和正文内容。

    参数:
        text: 模型生成的完整文本，可能包含 <think>...</think> 标签

    返回:
        (reasoning_content, content) 元组:
        - 有 think 标签: ("推理内容", "正文内容")
        - 无 think 标签: (None, 原始文本)

    示例:
        >>> parse_thinking_content("<think>分析问题...</think>\\n\\n最终回答")
        ("分析问题...", "最终回答")
        >>> parse_thinking_content("普通回答")
        (None, "普通回答")
    """
    if not text:
        return None, ""

    # 匹配 <think>...</think> 标签对（DOTALL 使 . 匹配换行符）
    match = re.match(r"<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        content = match.group(2).strip()
        # 思考内容为空时视为未启用思考模式
        return (reasoning if reasoning else None), content

    # 未闭合的 <think> 标签（模型生成被截断时可能出现）
    if text.startswith("<think>"):
        reasoning = text[len("<think>"):].strip()
        return (reasoning if reasoning else None), ""

    # 不包含 <think> 标签，全部为正文
    return None, text


# ==================== 流式解析 ====================


# </think> 标签长度，用于缓冲区尾部保留
_THINK_END_TAG = "</think>"
_THINK_END_LEN = len(_THINK_END_TAG)
_THINK_START_TAG = "<think>"
_THINK_START_LEN = len(_THINK_START_TAG)


class ThinkingStreamSplitter:
    """流式 <think> 标签拆分器

    逐 chunk 输入模型生成的文本流，实时将 <think>...</think> 内的内容
    与正文内容分离到不同的输出字段。

    状态转换:
        INIT ──检测到<think>──▶ THINKING ──检测到</think>──▶ CONTENT
          │                                                     │
          │ 无<think>标签                                        │
          ▼                                                     ▼
        CONTENT (全部→content)                         (所有后续→content)

    使用示例:
        splitter = ThinkingStreamSplitter()
        for chunk in model_stream:
            for field, text in splitter.feed(chunk):
                # field: "reasoning_content" 或 "content"
                delta = {field: text}
        # 流结束时刷新缓冲区
        for field, text in splitter.flush():
            delta = {field: text}
    """

    def __init__(self) -> None:
        # 状态: "init" → "thinking" → "content"
        self._state: str = "init"
        # 缓冲区，用于处理跨 chunk 的标签边界
        self._buffer: str = ""
        # 从 thinking 切换到 content 后，首次输出需跳过前导空白行
        self._content_started: bool = False

    @property
    def state(self) -> str:
        """当前解析状态"""
        return self._state

    def feed(self, chunk: str) -> List[Tuple[str, str]]:
        """输入一个文本 chunk，返回解析后的 (字段名, 文本) 列表

        参数:
            chunk: 模型流式输出的一个文本片段

        返回:
            列表，每个元素为 (field_name, text) 元组:
            - field_name: "reasoning_content" 或 "content"
            - text: 对应的文本片段
        """
        if not chunk:
            return []

        results: List[Tuple[str, str]] = []
        self._buffer += chunk

        if self._state == "init":
            results.extend(self._handle_init())

        if self._state == "thinking":
            results.extend(self._handle_thinking())

        if self._state == "content":
            results.extend(self._handle_content())

        return results

    def flush(self) -> List[Tuple[str, str]]:
        """流结束时刷新缓冲区中的剩余内容

        返回:
            剩余内容的 (field_name, text) 列表
        """
        results: List[Tuple[str, str]] = []
        if self._buffer:
            if self._state == "thinking":
                # 未闭合的 <think>，剩余内容作为 reasoning_content
                results.append(("reasoning_content", self._buffer))
            else:
                results.append(("content", self._buffer))
            self._buffer = ""
        return results

    def _handle_init(self) -> List[Tuple[str, str]]:
        """处理 INIT 状态：检测是否以 <think> 开头"""
        results: List[Tuple[str, str]] = []

        # 缓冲区不够长，可能是 <think> 的前缀，继续等待
        if len(self._buffer) < _THINK_START_LEN:
            if _THINK_START_TAG.startswith(self._buffer):
                return results
            # 不是 <think> 前缀，切换到 content 状态
            self._state = "content"
            return results

        # 判断是否以 <think> 开头
        if self._buffer.startswith(_THINK_START_TAG):
            # 去掉 <think> 标签本身，切换到 thinking 状态
            self._buffer = self._buffer[_THINK_START_LEN:]
            self._state = "thinking"
        else:
            # 不含 <think>，全部作为 content（无需跳过前导空白）
            self._state = "content"
            self._content_started = True

        return results

    def _handle_thinking(self) -> List[Tuple[str, str]]:
        """处理 THINKING 状态：检测 </think> 并分离内容"""
        results: List[Tuple[str, str]] = []

        end_idx = self._buffer.find(_THINK_END_TAG)
        if end_idx != -1:
            # 找到 </think>，标签前的内容为 reasoning_content
            thinking_text = self._buffer[:end_idx]
            if thinking_text:
                results.append(("reasoning_content", thinking_text))

            # 标签后的内容为 content（去除前导空白行）
            after = self._buffer[end_idx + _THINK_END_LEN:]
            self._buffer = after.lstrip("\n")
            self._state = "content"
        else:
            # 未检测到完整的 </think>
            # 保留尾部缓冲区，防止 </think> 被跨 chunk 切割
            safe_len = len(self._buffer) - _THINK_END_LEN
            if safe_len > 0:
                results.append(("reasoning_content", self._buffer[:safe_len]))
                self._buffer = self._buffer[safe_len:]

        return results

    def _handle_content(self) -> List[Tuple[str, str]]:
        """处理 CONTENT 状态：直接输出为 content"""
        results: List[Tuple[str, str]] = []

        if self._buffer:
            # 从 thinking 切换到 content 后，跳过前导空白行（如 </think> 与正文之间的 \n\n）
            if not self._content_started:
                self._buffer = self._buffer.lstrip("\n")
                if not self._buffer:
                    return results
                self._content_started = True
            results.append(("content", self._buffer))
            self._buffer = ""

        return results
