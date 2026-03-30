from abc import ABC, abstractmethod
from dataclasses import replace as dataclass_replace
from enum import Enum
from typing import List, Dict, Any, Iterator, Optional, Union, TYPE_CHECKING
import json
import re
import uuid

from loguru import logger

from strategies.protocol import StrategyInput, StrategyOutput, PromptOutput

if TYPE_CHECKING:
    from engines.base import LLMEngine


# ==================== 通用工具调用解析 ====================

# 匹配 <tool_call> 标签的起止位置（不依赖内部 JSON 格式）
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*")
_TOOL_CALL_CLOSE = "</tool_call>"

# 匹配 function tag 格式：<function=Name><parameter=key>value</parameter>...</function>
# 用于 Qwen3-Coder 等模型的 chat_template 输出
_FUNCTION_TAG_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL,
)
_PARAMETER_TAG_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)


def _find_json_span(text: str, start: int) -> int | None:
    """从 text[start] 的 '{' 开始，用括号计数找到配对的 '}'。

    正确处理字符串内的转义引号和嵌套大括号。

    Args:
        text: 源文本
        start: '{' 所在的索引位置

    Returns:
        配对 '}' 的索引位置（含），找不到则返回 None
    """
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_str = False
    escape = False
    for j in range(start, len(text)):
        ch = text[j]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return j
    return None


def _extract_json_objects(text: str) -> list[dict]:
    """从文本中提取所有顶层 JSON 对象（支持嵌套括号）

    使用括号计数而非正则，正确处理任意深度的嵌套 JSON。
    仅返回包含 "name" 和 "arguments" 键的对象（工具调用特征）。
    """
    results: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        end = _find_json_span(text, i)
        if end is None:
            i += 1
            continue
        candidate = text[i : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                results.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
        i = end + 1
    return results


def _parse_function_tag_calls(
    text: str,
    allowed_tool_names: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """解析 function tag 格式的工具调用

    部分模型（如 Qwen3-Coder）的 chat_template 指示模型输出如下格式：
        <tool_call>
        <function=Bash>
        <parameter=command>ls -la</parameter>
        <parameter=description>List files</parameter>
        </function>
        </tool_call>

    也兼容不带外层 <tool_call> 的情况。

    Returns:
        OpenAI 格式的 tool_calls 列表，空列表表示无工具调用
    """
    tool_calls: list[dict[str, Any]] = []
    allowed_tool_names = allowed_tool_names or set()
    for fn_match in _FUNCTION_TAG_RE.finditer(text):
        func_name = fn_match.group(1)
        if allowed_tool_names and func_name not in allowed_tool_names:
            continue
        inner = fn_match.group(2)
        arguments: dict[str, Any] = {}
        for param_match in _PARAMETER_TAG_RE.finditer(inner):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()
            # 尝试将值解析为 JSON（支持 dict/list/number/bool）
            try:
                arguments[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                arguments[param_name] = param_value
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
    return tool_calls


def _extract_allowed_tool_names(tools: Optional[List[Dict[str, Any]]]) -> set[str]:
    allowed: set[str] = set()
    if not tools:
        return allowed
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            allowed.add(name)
    return allowed


def _parse_named_json_tool_calls(
    text: str,
    allowed_tool_names: set[str],
) -> list[dict[str, Any]]:
    if not allowed_tool_names:
        return []
    tool_calls: list[dict[str, Any]] = []
    lines = text.splitlines()
    for i, raw_line in enumerate(lines):
        func_name = raw_line.strip()
        if func_name not in allowed_tool_names:
            continue
        remainder = "\n".join(lines[i + 1:]).lstrip()
        if not remainder:
            continue
        if remainder.startswith("```json"):
            remainder = remainder[len("```json"):].lstrip()
        elif remainder.startswith("```"):
            remainder = remainder[len("```"):].lstrip()
        json_start = remainder.find("{")
        if json_start == -1:
            continue
        json_end = _find_json_span(remainder, json_start)
        if json_end is None:
            continue
        try:
            arguments = json.loads(remainder[json_start : json_end + 1])
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(arguments, dict):
            continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
    return tool_calls


def _parse_tool_calls_from_text(
    text: str,
    allowed_tool_names: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """从模型输出文本中提取工具调用

    解析策略（按优先级）：
    1. <tool_call>{...}</tool_call> 标签 + JSON 格式（Qwen3/通用回退模板的标准输出）
    2. <function=Name><parameter=key>value</parameter></function> 格式（Qwen3-Coder 等）
    3. 独立 JSON 对象 {"name": "...", "arguments": {...}}（兜底，支持嵌套参数）

    Returns:
        OpenAI 格式的 tool_calls 列表，空列表表示无工具调用
    """
    tool_calls: list[dict[str, Any]] = []
    allowed_tool_names = allowed_tool_names or set()

    # 策略 1：<tool_call> + JSON 标签（优先，因为边界明确不易误判）
    # 使用 _find_json_span 提取标签内的 JSON，正确处理嵌套参数
    for match in _TOOL_CALL_OPEN_RE.finditer(text):
        json_start = match.end()
        if json_start >= len(text) or text[json_start] != '{':
            continue
        json_end = _find_json_span(text, json_start)
        if json_end is None:
            continue
        # 验证后面紧跟 </tool_call>（允许中间有空白）
        after_json = text[json_end + 1:].lstrip()
        if after_json and not after_json.startswith(_TOOL_CALL_CLOSE):
            continue
        try:
            obj = json.loads(text[json_start : json_end + 1])
            normalized = _normalize_tool_call(obj)
            name = normalized.get("function", {}).get("name", "")
            if allowed_tool_names and name not in allowed_tool_names:
                continue
            tool_calls.append(normalized)
        except (json.JSONDecodeError, KeyError):
            continue

    if tool_calls:
        return tool_calls

    # 策略 2：function tag 格式（Qwen3-Coder 等模型的原生输出）
    tool_calls = _parse_function_tag_calls(text, allowed_tool_names)
    if tool_calls:
        return tool_calls

    tool_calls = _parse_named_json_tool_calls(text, allowed_tool_names)
    if tool_calls:
        return tool_calls

    # 策略 3：从全文提取 JSON 对象（支持任意深度嵌套）
    for obj in _extract_json_objects(text):
        normalized = _normalize_tool_call(obj)
        name = normalized.get("function", {}).get("name", "")
        if allowed_tool_names and name not in allowed_tool_names:
            continue
        tool_calls.append(normalized)

    # 诊断日志：文本中包含工具调用特征但解析为空
    if not tool_calls:
        _log_tool_parse_miss(text, allowed_tool_names)

    return tool_calls


def _log_tool_parse_miss(
    text: str,
    allowed_tool_names: Optional[set[str]] = None,
) -> None:
    """当文本包含工具调用特征但解析结果为空时记录警告日志

    帮助排查工具调用解析失败的情况（如 special token 被误删、
    格式不匹配等）。仅在检测到多个潜在工具调用模式时才记录，
    避免普通文本中偶然包含 JSON 关键词时产生噪音。
    """
    indicators = ['tool_call', '<function=', '<parameter=', '"name"', '"arguments"']
    text_lower = text.lower()
    found = [ind for ind in indicators if ind.lower() in text_lower]
    allowed_tool_names = allowed_tool_names or set()
    if allowed_tool_names and not any(name in text for name in allowed_tool_names):
        return
    if len(found) >= 2:
        preview = text[:500].replace('\n', '\\n')
        logger.warning(
            "工具调用解析为空但文本包含工具特征 {}: text[:500]='{}' len={}",
            found, preview, len(text),
        )


def _normalize_tool_call(raw: dict) -> dict[str, Any]:
    """将模型原始工具调用格式标准化为 OpenAI tool_calls 元素格式

    OpenAI 规范: {"id": "call_xxx", "type": "function",
                  "function": {"name": "...", "arguments": "<json-string>"}}
    """
    name = raw.get("name", "")
    arguments = raw.get("arguments", {})
    # arguments 可能是 dict 或已序列化的 str
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    elif not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _strip_tool_call_text(text: str, tool_calls: list[dict[str, Any]]) -> str:
    """从模型输出中移除已解析的工具调用文本，返回干净的 content

    同时处理 <tool_call> 标签格式、function tag 格式和裸 JSON 格式。
    使用与 _extract_json_objects 一致的括号计数法，正确处理嵌套参数。
    """
    spans_to_remove: list[tuple[int, int]] = []

    # 移除 <tool_call>...</tool_call> 标签（包含 JSON 格式）
    for match in _TOOL_CALL_OPEN_RE.finditer(text):
        tag_start = match.start()
        json_start = match.end()
        if json_start >= len(text) or text[json_start] != '{':
            continue
        json_end = _find_json_span(text, json_start)
        if json_end is None:
            continue
        after_json = text[json_end + 1:]
        close_offset = after_json.lstrip()
        whitespace_len = len(after_json) - len(close_offset)
        if close_offset.startswith(_TOOL_CALL_CLOSE):
            tag_end = json_end + 1 + whitespace_len + len(_TOOL_CALL_CLOSE)
            spans_to_remove.append((tag_start, tag_end))
        elif not close_offset.strip():
            spans_to_remove.append((tag_start, json_end + 1))

    # 移除包含 function tag 格式的 <tool_call>...</tool_call> 块
    # 匹配 <tool_call>...<function=...>...</function>...</tool_call>
    for match in re.finditer(
        r"<tool_call>\s*<function=\w+>.*?</function>\s*</tool_call>",
        text, re.DOTALL,
    ):
        spans_to_remove.append((match.start(), match.end()))

    # 移除不带 <tool_call> 外层的 <function=...>...</function> 块
    for match in _FUNCTION_TAG_RE.finditer(text):
        # 仅移除未被上面的 <tool_call> 区间覆盖的
        already_covered = any(
            s <= match.start() and match.end() <= e
            for s, e in spans_to_remove
        )
        if not already_covered:
            spans_to_remove.append((match.start(), match.end()))

    # 按起始位置排序后从后向前移除
    spans_to_remove.sort(key=lambda x: x[0])
    cleaned = text
    for start, end in reversed(spans_to_remove):
        cleaned = cleaned[:start] + cleaned[end:]

    # 收集需要移除的函数名集合（每个名称只移除一次）
    fn_names_to_strip: list[str] = []
    for tc in tool_calls:
        fn_name = tc.get("function", {}).get("name", "")
        if fn_name:
            fn_names_to_strip.append(fn_name)

    if not fn_names_to_strip:
        return cleaned.strip()

    # 用括号计数法扫描文本，定位并移除匹配的 JSON 工具调用
    # 从后向前移除，避免偏移量变化
    spans_to_remove: list[tuple[int, int]] = []
    for fn_name in list(fn_names_to_strip):
        pattern = re.compile(
            rf"(^|\n)\s*{re.escape(fn_name)}\s*\n",
            re.MULTILINE,
        )
        for match in pattern.finditer(cleaned):
            search_start = match.end()
            probe = cleaned[search_start : search_start + 1200]
            brace_rel = probe.find('{')
            if brace_rel == -1 or probe[:brace_rel].count('\n') > 4:
                continue
            json_start = search_start + brace_rel
            json_end = _find_json_span(cleaned, json_start)
            if json_end is None:
                continue
            spans_to_remove.append((match.start(), json_end + 1))
            fn_names_to_strip.remove(fn_name)
            break
    i = 0
    while i < len(cleaned):
        if cleaned[i] != '{':
            i += 1
            continue
        end = _find_json_span(cleaned, i)
        if end is None:
            i += 1
            continue
        candidate = cleaned[i : end + 1]
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            i = end + 1
            continue
        if isinstance(obj, dict) and "name" in obj:
            name = obj["name"]
            if name in fn_names_to_strip:
                spans_to_remove.append((i, end + 1))
                fn_names_to_strip.remove(name)  # 每个名称只移除一次
        i = end + 1

    # 排序后从后向前移除，保持前面的索引不受影响
    spans_to_remove.sort(key=lambda x: x[0])
    for start, end in reversed(spans_to_remove):
        cleaned = cleaned[:start] + cleaned[end:]

    return cleaned.strip()


def _build_tools_system_injection(
    tools: list[dict],
    tool_choice: Any = None,
    max_desc_length: int | None = None,
) -> str:
    """将 OpenAI 格式的 tools 定义注入到 system prompt 中（回退方案）

    当 tokenizer 不支持原生 tools 模板时，将工具定义以文本形式追加到 system message。
    同时根据 tool_choice 添加对应的行为指令。

    Args:
        tools: OpenAI 格式的工具定义列表
        tool_choice: 工具选择策略 ("auto"/"none"/"required"/具体工具对象)
        max_desc_length: 工具描述的最大字符长度，超出时截断。None 不截断。
    """
    lines = [
        "\n\n# Available Tools",
        "You have access to the following tools. To call a tool, respond with a JSON object "
        'wrapped in <tool_call></tool_call> tags, like: <tool_call>{"name": "tool_name", '
        '"arguments": {"arg1": "value1"}}</tool_call>',
        "You may call multiple tools in a single response if needed.",
        "",
    ]
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})

        # 工具描述压缩：减少小模型的 system prompt 负担
        if max_desc_length and desc and len(desc) > max_desc_length:
            desc = desc[:max_desc_length] + "..."

        # 参数 schema 压缩：移除 properties 内嵌的 description 字段
        if max_desc_length and params:
            params = _compact_params_schema(params)

        lines.append(f"## {name}")
        if desc:
            lines.append(f"Description: {desc}")
        if params:
            lines.append(f"Parameters: {json.dumps(params, ensure_ascii=False)}")
        lines.append("")

    # 根据 tool_choice 添加行为指令
    choice_instruction = _tool_choice_instruction(tool_choice)
    if choice_instruction:
        lines.append(choice_instruction)

    return "\n".join(lines)


def _compact_params_schema(schema: dict) -> dict:
    """递归压缩 JSON Schema 参数定义：移除所有层级的 description 字段

    保留类型信息和结构约束，仅移除冗长的文本描述。
    用于小模型的工具注入场景，减少 system prompt 长度。
    """
    if not isinstance(schema, dict):
        return schema
    result = {}
    for k, v in schema.items():
        if k == "description":
            continue
        elif k == "properties" and isinstance(v, dict):
            compact_props = {}
            for prop_name, prop_def in v.items():
                if isinstance(prop_def, dict):
                    compact_props[prop_name] = _compact_params_schema(prop_def)
                else:
                    compact_props[prop_name] = prop_def
            result[k] = compact_props
        elif isinstance(v, dict):
            result[k] = _compact_params_schema(v)
        else:
            result[k] = v
    return result


def _compact_tool_definitions(
    tools: list[dict],
    max_desc_length: int | None,
) -> list[dict]:
    if not max_desc_length:
        return tools
    compacted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            compacted.append(tool)
            continue
        compact_tool = dict(tool)
        fn_source = tool.get("function", tool)
        if not isinstance(fn_source, dict):
            compacted.append(compact_tool)
            continue
        fn = dict(fn_source)
        desc = fn.get("description")
        if isinstance(desc, str) and len(desc) > max_desc_length:
            fn["description"] = desc[:max_desc_length] + "..."
        params = fn.get("parameters")
        if isinstance(params, dict) and params:
            fn["parameters"] = _compact_params_schema(params)
        if "function" in compact_tool:
            compact_tool["function"] = fn
        else:
            compact_tool = fn
        compacted.append(compact_tool)
    return compacted


def _tool_choice_instruction(tool_choice: Any) -> str:
    """根据 OpenAI tool_choice 参数生成行为指令文本

    OpenAI tool_choice 规范:
    - "auto" (默认): 模型自行决定是否调用工具
    - "none": 禁止调用工具，必须直接回复文本
    - "required": 必须调用至少一个工具
    - {"type": "function", "function": {"name": "xxx"}}: 必须调用指定工具
    """
    if tool_choice is None or tool_choice == "auto":
        return ""
    if tool_choice == "none":
        return "Important: Do NOT call any tools. Respond with text only."
    if tool_choice == "required":
        return "Important: You MUST call at least one tool in your response."
    # 指定具体工具: {"type": "function", "function": {"name": "xxx"}}
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        name = fn.get("name", "")
        if name:
            return f'Important: You MUST call the tool "{name}" in your response.'
    return ""


class ToolPromptMode(str, Enum):
    ENGINE_NATIVE = "engine_native"
    INJECTED = "injected"


class LLMStrategy(ABC):
    """
    LLM 策略接口，负责对话模板与生成调用。
    
    设计原则：
    1. build_prompt() 负责构建 prompt，返回 PromptOutput
    2. execute() 负责调用引擎生成，返回 StrategyOutput
    3. 所有子类禁止在 execute() 中直接 yield 特殊前缀
    4. thinking_prefix 由 PromptOutput 声明，调用方统一处理
    5. tools 处理路径: 引擎原生模板 → system prompt 注入（回退）
    6. tool_choice 同时传递给原生模板和回退路径，确保行为一致
    7. parse_tool_calls() 从生成文本中提取工具调用，子类可覆盖
    """

    @abstractmethod
    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """将消息列表转换为模型输入的prompt（兼容旧接口）"""
        ...

    def get_default_stop_words(self) -> List[str]:
        """获取默认停止词列表，子类可覆盖"""
        return []

    def prefer_native_template(self) -> bool:
        """是否优先使用引擎原生 tokenizer.apply_chat_template()

        返回 True 时，execute() 在无 tools 场景也会优先尝试引擎原生模板
        （tokenize=False，返回字符串），仅在原生模板不可用时回退到策略手动模板。

        适用于自定义 tokenizer（如 GLM-4 系列），引擎模板格式比策略手动
        拼接更可靠，且 tokenizer.__call__ 能正确编码自身的特殊标记。
        """
        return False

    def supports_native_tools(self) -> bool:
        """是否支持引擎原生 tools 模板（tokenizer.apply_chat_template(tools=...)）

        返回 False 时，execute() 跳过引擎原生 tools 模板路径，
        强制走 fallback 路径（将工具定义注入到 system prompt）。

        部分模型（如 GLM-4 系列）的 chat_template 使用非标准的 tools 注入方式
        （在消息级别检查 item['tools'] 而非使用全局 tools 参数），
        导致引擎原生 tools 模板无法正确注入工具定义。
        此类模型应返回 False，使用 fallback 路径确保工具定义可见。
        """
        return True

    def tool_prompt_mode(self) -> ToolPromptMode:
        if self.supports_native_tools():
            return ToolPromptMode.ENGINE_NATIVE
        return ToolPromptMode.INJECTED

    def max_tool_description_length(self) -> int | None:
        """工具描述的最大字符长度，超出时在注入 system prompt 时截断。

        返回 None 表示不截断（默认）。
        小参数模型（如 GLM-4-9B）应覆盖此方法返回较小值（如 500），
        以减少 system prompt 中的工具定义长度，让模型更好地理解工具格式。
        """
        return None

    def parse_tool_calls(
        self,
        text: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """从模型输出中解析工具调用

        子类可覆盖以实现模型特定的解析逻辑。
        默认实现支持 <tool_call>JSON</tool_call> 和独立 JSON 两种格式。

        Args:
            text: 模型生成的完整文本

        Returns:
            OpenAI 格式的 tool_calls 列表，空列表表示无工具调用
        """
        return _parse_tool_calls_from_text(
            text,
            allowed_tool_names=_extract_allowed_tool_names(tools),
        )

    def prepare_tools(
        self,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return tools
        return _compact_tool_definitions(
            tools,
            self.max_tool_description_length(),
        )

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

    def inject_tools_into_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = None,
    ) -> List[Dict[str, Any]]:
        """将工具定义注入到消息列表的 system prompt 中（回退方案）

        当引擎不支持原生 tools 模板时调用。
        如果已有 system 消息，追加工具说明；否则插入新 system 消息。
        同时根据 tool_choice 注入行为指令。
        """
        injection = _build_tools_system_injection(
            tools, tool_choice,
            max_desc_length=self.max_tool_description_length(),
        )
        messages = [dict(m) for m in messages]  # shallow copy
        if messages and messages[0].get("role") == "system":
            messages[0] = dict(messages[0])
            messages[0]["content"] = (messages[0].get("content") or "") + injection
        else:
            messages.insert(0, {"role": "system", "content": injection.strip()})
        return messages

    def resolve_prompt(self, engine: "LLMEngine", input: StrategyInput) -> PromptOutput:
        prompt_output = self.build_prompt(input)
        prompt = None
        effective_input = input
        prepared_tools = self.prepare_tools(input.tools)

        _tc = input.tool_choice
        tool_choice_is_default = (_tc is None or _tc == "auto")
        if prepared_tools:
            if (
                tool_choice_is_default
                and self.tool_prompt_mode() == ToolPromptMode.ENGINE_NATIVE
                and engine.supports_native_tools_template()
            ):
                prompt = engine.apply_chat_template(
                    input.messages,
                    tools=prepared_tools,
                    enable_thinking=input.enable_thinking,
                )
            if prompt is None:
                injected = self.inject_tools_into_messages(
                    input.messages, prepared_tools, input.tool_choice,
                )
                effective_input = dataclass_replace(
                    input, messages=injected, tools=prepared_tools,
                )

        if prompt is None and self.prefer_native_template():
            prompt = engine.apply_chat_template(
                effective_input.messages,
                enable_thinking=effective_input.enable_thinking,
            )

        if prompt is None:
            if effective_input is not input:
                prompt_output = self.build_prompt(effective_input)
            prompt = prompt_output.prompt
        else:
            if (prompt_output.thinking_prefix
                    and isinstance(prompt, str)
                    and not prompt.rstrip().endswith(prompt_output.thinking_prefix)):
                prompt = prompt.rstrip("\n") + "\n" + prompt_output.thinking_prefix

        return PromptOutput(
            prompt=prompt,
            thinking_prefix=prompt_output.thinking_prefix,
            stop_words=prompt_output.stop_words,
            metadata=prompt_output.metadata,
        )

    def _should_parse_tool_calls(self, input: StrategyInput) -> bool:
        """判断是否需要解析工具调用

        根据 tools 和 tool_choice 决定：
        - 无 tools 定义 → 不解析
        - tool_choice="none" → 不解析（模型被指示不调用工具）
        - 其他情况 → 解析
        """
        if not input.tools:
            return False
        if input.tool_choice == "none":
            return False
        return True

    def execute(self, engine: "LLMEngine", input: StrategyInput) -> StrategyOutput:
        """
        执行生成（新协议接口）

        统一的生成入口，返回 StrategyOutput。
        所有路径最终产出 **字符串 prompt**，由引擎统一编码和生成。

        Prompt 选择优先级：
        1. 引擎原生 tools 模板（tokenize=False，返回字符串）
        2. prefer_native_template 引擎原生模板（tokenize=False，返回字符串）
        3. 策略 build_prompt() 手动模板（返回字符串）
        回退时如有 tools，通过 system prompt 注入工具定义和 tool_choice 行为指令。

        thinking 处理：
        - 策略通过 build_prompt() 声明 thinking_prefix（如 "<think>"）
        - 使用原生模板时，若原生字符串未包含 thinking 标记，自动追加
        - 输出流中统一前置 thinking_prefix（供 ThinkingStreamSplitter 解析）

        Args:
            engine: LLM 引擎实例
            input: 策略输入（含 tools、tool_choice）

        Returns:
            StrategyOutput: 统一的输出结构
        """
        # ---- 1. 构建 prompt（统一路径：原生 tools 模板 / prefer_native / 手动模板） ----
        # 路径选择优先级（在 resolve_prompt 内部决策）：
        #   a) 引擎原生 tools 模板（tool_choice 为默认值且引擎支持）
        #   b) prefer_native_template 引擎模板（GLM-4 等自定义 tokenizer）
        #   c) 策略手动模板（最终回退）
        # tools 不可用时通过 inject_tools_into_messages 注入 system prompt
        prompt_output = self.resolve_prompt(engine, input)
        prompt = prompt_output.prompt

        # ---- 2. 构建生成参数 ----
        gen_kwargs = {
            "max_tokens": input.max_tokens,
            "temperature": input.temperature,
            "top_p": input.top_p,
            "enable_thinking": input.enable_thinking,
        }
        if input.top_k is not None:
            gen_kwargs["top_k"] = input.top_k
        if prompt_output.stop_words:
            gen_kwargs["stop"] = prompt_output.stop_words

        # ---- 6. 调用引擎生成 ----
        if input.stream:
            iterator = self._wrap_stream(
                engine.generate(prompt, stream=True, **gen_kwargs),
                prompt_output.thinking_prefix,
            )
            return StrategyOutput(
                stream_iterator=iterator,
                thinking_content=None,
                finish_reason="stop",
                metadata={
                    "thinking_prefix": prompt_output.thinking_prefix,
                    "has_tools": self._should_parse_tool_calls(input),
                },
            )
        else:
            text = engine.generate(prompt, stream=False, **gen_kwargs)
            final_text = text
            if prompt_output.thinking_prefix:
                final_text = prompt_output.thinking_prefix + text

            # 解析工具调用（仅当 tools 存在且 tool_choice != "none"）
            tool_calls = None
            finish_reason = "stop"
            if self._should_parse_tool_calls(input):
                tool_calls = self.parse_tool_calls(final_text, input.tools)
                if tool_calls:
                    finish_reason = "tool_calls"
                    final_text = _strip_tool_call_text(final_text, tool_calls)

            return StrategyOutput(
                text=final_text,
                thinking_content=None,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=finish_reason,
                metadata={
                    "thinking_prefix": prompt_output.thinking_prefix,
                    "has_tools": self._should_parse_tool_calls(input),
                },
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
