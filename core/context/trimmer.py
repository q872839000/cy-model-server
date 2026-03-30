"""消息截断器：滑动窗口策略

将超长的 messages 列表裁剪到 token 预算内，遵循以下规则：
1. system 消息永远保留（定义模型角色/行为）
2. 最后一条 user 消息永远保留（当前用户输入）
3. tool_call 消息成对保留或丢弃（assistant 的 tool_calls + 后续 tool 响应）
4. 从最早的非保护消息开始丢弃，直到总 token 数在预算内

设计原则：
- 纯函数实现，不依赖 Service / Engine / Registry
- TokenCounter 通过协议注入，支持精确 tokenizer 或粗略估算
- 可独立测试
"""

import json
from typing import List, Dict, Any, Optional, Protocol


class TokenCounter(Protocol):
    """Token 计数协议

    实现此协议即可作为 trim_messages 的 token 计数器。
    """

    def count(self, text: str) -> int:
        """返回文本的 token 数"""
        ...


class TokenizerCounter:
    """基于 HuggingFace tokenizer 的精确计数器"""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        if not text:
            return 0
        try:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return len(self._tokenizer.encode(text))
        except Exception:
            return _estimate_tokens(text)


class EstimateCounter:
    """粗略估算计数器（无 tokenizer 时的兜底）

    中英文混合场景下，1 token ≈ 2-3 个字符是合理的近似。
    """

    def count(self, text: str) -> int:
        return _estimate_tokens(text)


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中英文混合取 1 token ≈ 2.5 字符"""
    if not text:
        return 0
    return max(1, len(text) * 2 // 5)


def _message_text(msg: Dict[str, Any]) -> str:
    """提取消息的文本内容（用于 token 计数）

    处理 content 为 string 或 content blocks 列表两种格式。
    同时计入 role、tool_calls JSON 等结构化开销。
    """
    parts: list[str] = []

    # role 本身消耗少量 token
    parts.append(msg.get("role", ""))

    content = msg.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)

    # tool_calls / tool_call_id 等结构化字段也会消耗 token
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        try:
            parts.append(json.dumps(tool_calls, ensure_ascii=False))
        except (TypeError, ValueError):
            pass

    name = msg.get("name")
    if name:
        parts.append(name)

    tool_call_id = msg.get("tool_call_id")
    if tool_call_id:
        parts.append(tool_call_id)

    return "\n".join(parts)


def _count_message_tokens(msg: Dict[str, Any], counter: TokenCounter) -> int:
    """计算单条消息的 token 数（含格式开销）"""
    # 每条消息有固定的格式开销（<|im_start|>role\n ... <|im_end|> 等），约 4 token
    return counter.count(_message_text(msg)) + 4


def _build_tool_groups(messages: List[Dict[str, Any]]) -> List[List[int]]:
    """识别 tool_call 消息组

    一个 tool 组 = 一条 assistant（含 tool_calls）+ 紧随其后的所有 tool role 响应。
    这些消息必须一起保留或一起丢弃，否则会导致消息序列非法。

    Returns:
        每组包含的消息索引列表
    """
    groups: list[list[int]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            group = [i]
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                group.append(j)
                j += 1
            groups.append(group)
            i = j
        else:
            i += 1
    return groups


def trim_messages(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    counter: TokenCounter,
    *,
    reserve_output_tokens: int = 0,
) -> List[Dict[str, Any]]:
    """将消息列表裁剪到 token 预算内

    算法：
    1. 分离 system 消息（始终保留）
    2. 识别 tool_call 消息组（必须整组保留/丢弃）
    3. 从最早的非保护消息开始丢弃
    4. 最后一条 user 消息始终保留

    Args:
        messages: OpenAI 格式的消息列表
        max_tokens: 上下文窗口总 token 数（context_window）
        counter: token 计数器
        reserve_output_tokens: 为输出预留的 token 数

    Returns:
        裁剪后的消息列表（不修改原列表）
    """
    if not messages:
        return []

    token_budget = max_tokens - reserve_output_tokens

    # ---- 1. 分离 system 消息 ----
    system_msgs: list[tuple[int, Dict[str, Any]]] = []
    non_system: list[tuple[int, Dict[str, Any]]] = []

    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            system_msgs.append((i, msg))
        else:
            non_system.append((i, msg))

    # system 消息固定开销
    system_tokens = sum(_count_message_tokens(m, counter) for _, m in system_msgs)

    # ---- 2. 计算每条非 system 消息的 token 数 ----
    msg_tokens: dict[int, int] = {}
    for idx, msg in non_system:
        msg_tokens[idx] = _count_message_tokens(msg, counter)

    # ---- 3. 保护最后一条 user 消息（当前请求，无条件保留） ----
    last_user_idx: Optional[int] = None
    last_user_tokens = 0
    for idx, msg in reversed(non_system):
        if msg.get("role") == "user":
            last_user_idx = idx
            last_user_tokens = msg_tokens.get(idx, 0)
            break

    # 固定开销 = system + 最后一条 user
    fixed_tokens = system_tokens + last_user_tokens

    # 如果固定消息已超预算，返回 [system..., last_user] 让下游兜底报错
    if fixed_tokens >= token_budget:
        result = [m for _, m in system_msgs]
        if last_user_idx is not None:
            result.append(messages[last_user_idx])
        return result

    if not non_system:
        return [m for _, m in system_msgs]

    remaining_budget = token_budget - fixed_tokens

    # ---- 4. 识别 tool_call 组（用原始索引） ----
    tool_groups = _build_tool_groups(messages)
    idx_to_group: dict[int, int] = {}
    for gid, group in enumerate(tool_groups):
        for idx in group:
            idx_to_group[idx] = gid

    # ---- 5. 从后往前累加，确定保留哪些历史消息 ----
    kept_indices: set[int] = set()
    used_tokens = 0

    # last_user 已在固定开销中，标记为已保留
    if last_user_idx is not None:
        kept_indices.add(last_user_idx)

    # 从后往前扫描非 system 消息
    for idx, msg in reversed(non_system):
        if idx in kept_indices:
            continue

        # 如果这条消息属于某个 tool 组，计算整组成本
        gid = idx_to_group.get(idx)
        if gid is not None:
            group_indices = tool_groups[gid]
            # 如果组内任何成员已经被保留了，跳过（整组处理）
            if any(gi in kept_indices for gi in group_indices):
                continue
            group_cost = sum(msg_tokens.get(gi, 0) for gi in group_indices)
            if used_tokens + group_cost <= remaining_budget:
                kept_indices.update(group_indices)
                used_tokens += group_cost
            # 整组放不下就整组丢弃
        else:
            cost = msg_tokens.get(idx, 0)
            if used_tokens + cost <= remaining_budget:
                kept_indices.add(idx)
                used_tokens += cost

    # ---- 6. 按原始顺序组装结果 ----
    result: list[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            result.append(msg)
        elif i in kept_indices:
            result.append(msg)

    return result
