"""GLM 系列模型基础策略"""

from typing import List, Dict, Any
import json
import re
import uuid
from strategies.base import (
    LLMStrategy,
    _extract_allowed_tool_names,
    _find_json_span,
    _tool_choice_instruction,
)

# GLM 嵌入式工具调用匹配：行首 "单词\n{json}" 模式
_GLM_EMBEDDED_PATTERN = re.compile(
    r"(?:^|\n)(\w+)\s*\n(\{[^\n].*?\})\s*(?:\n|$)",
    re.DOTALL,
)

# 匹配行尾的 PascalCase 标识符（允许前面有中文或标点）
# 用于策略 1c：从文本中提取工具函数名
_TRAILING_PASCALCASE_RE = re.compile(
    r'[：:]\s*([A-Z]\w+)\s*$|(?:^|\s)([A-Z]\w+)\s*$'
)


class GLMBaseStrategy(LLMStrategy):
    """GLM 系列基础策略，支持 observation 角色。

    GLM-4 系列模型的 chat_template 使用非标准的 tools 注入方式：
    检查 messages 中每个 item 的 'tools' 字段（item['tools']），
    而非 HuggingFace 标准的全局 tools 参数。

    因此 supports_native_tools() 返回 False，通过重写
    inject_tools_into_messages() 将 tools 作为结构化字段附加到消息上，
    让引擎原生模板（Jinja2）正确渲染工具定义。
    """

    # 定义特殊标记，避免直接在字符串中使用导致解析问题
    _SYSTEM_TAG = "<" + "|system|" + ">"
    _USER_TAG = "<" + "|user|" + ">"
    _ASSISTANT_TAG = "<" + "|assistant|" + ">"
    _OBSERVATION_TAG = "<" + "|observation|" + ">"

    def apply_chat_template(self, messages: List[Dict], **kwargs) -> str:
        """将消息列表转换为 GLM 模型输入的 prompt"""
        parts = []
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content", "")
            if role == "system":
                parts.append(f"{self._SYSTEM_TAG}\n{content}")
            elif role == "user":
                parts.append(f"{self._USER_TAG}\n{content}")
            elif role == "assistant":
                parts.append(f"{self._ASSISTANT_TAG}\n{content}")
            elif role in ("observation", "tool"):
                parts.append(f"{self._OBSERVATION_TAG}\n{content}")
            else:
                parts.append(f"{self._USER_TAG}\n{content}")
        parts.append(f"{self._ASSISTANT_TAG}\n")
        return "[gMASK]<sop>" + "\n".join(parts)

    def get_default_stop_words(self) -> List[str]:
        return [self._USER_TAG, self._ASSISTANT_TAG, self._SYSTEM_TAG, self._OBSERVATION_TAG]

    def supports_native_tools(self) -> bool:
        """GLM-4 的 chat_template 不支持 HuggingFace 标准的 tools 全局参数。

        GLM-4 的 Jinja2 模板检查 item['tools']（消息级别），而非全局 tools 变量。
        使用引擎原生 tools 模板会导致工具定义被静默忽略，模型看不到任何工具。
        返回 False 强制走 inject_tools_into_messages() 路径。
        """
        return False

    def inject_tools_into_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = None,
    ) -> List[Dict[str, Any]]:
        """将工具定义作为结构化字段附加到消息上，并适配 GLM 消息格式

        GLM-4 的 Jinja2 chat_template 通过 item['tools'] 检测工具定义，
        并自动生成格式化的系统提示（包含工具名、参数 schema、调用指令）。
        因此不能用纯文本注入，必须将 tools 作为结构化字段附加到消息上。

        同时处理 GLM 模板与 OpenAI 格式之间的差异：
        1. tools 字段：附加到第一条消息，让 Jinja2 模板发现并渲染
        2. role 映射：OpenAI "tool" → GLM "observation"
        3. metadata 补全：assistant 工具调用消息需要 metadata=函数名
           GLM 模板渲染为 <|assistant|>函数名，模型靠此识别历史工具调用

        tool_choice 非默认值时，额外注入文本行为指令（GLM 模板不支持 tool_choice）。
        """
        messages = [dict(m) for m in messages]  # shallow copy

        if not messages:
            choice_instruction = _tool_choice_instruction(tool_choice)
            messages.append({
                "role": "system",
                "content": choice_instruction or "",
                "tools": tools,
            })
            return messages

        # 将 tools 作为结构化字段附加到第一条消息
        messages[0] = dict(messages[0])
        messages[0]["tools"] = tools

        # 适配 GLM 消息格式：role 映射 + metadata 补全
        for i, msg in enumerate(messages):
            role = msg.get("role")

            # OpenAI "tool" → GLM "observation"
            # GLM 模板直接输出 <|{{ item['role'] }}|>，不做角色转换
            if role == "tool":
                messages[i] = dict(msg)
                messages[i]["role"] = "observation"

            # assistant 消息中的 tool_calls → 补全 metadata 字段
            # GLM 模板：<|assistant|>{{ item['metadata'] }}\n{{ item['content'] }}
            # metadata 应为函数名，模型靠此识别这是一条工具调用
            elif role == "assistant" and msg.get("tool_calls"):
                messages[i] = dict(msg)
                tc_list = msg["tool_calls"]
                if tc_list:
                    first_fn = tc_list[0].get("function", {}).get("name", "")
                    messages[i]["metadata"] = first_fn
                    # content 应为函数参数 JSON（GLM 原生格式）
                    if not msg.get("content"):
                        args = tc_list[0].get("function", {}).get("arguments", "{}")
                        messages[i]["content"] = args

        # tool_choice 非默认值时，注入文本行为指令到 system prompt
        choice_instruction = _tool_choice_instruction(tool_choice)
        if choice_instruction:
            if messages[0].get("role") == "system":
                messages[0]["content"] = (
                    (messages[0].get("content") or "") + "\n\n" + choice_instruction
                )
            else:
                messages.insert(0, {"role": "system", "content": choice_instruction})

        return messages

    def max_tool_description_length(self) -> int | None:
        """GLM 系列模型限制工具描述长度，减少小模型的理解负担"""
        return 500

    def parse_tool_calls(
        self,
        text: str,
        tools: List[Dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """解析 GLM 模型的工具调用输出

        GLM-4 模型使用多种格式输出工具调用：
        1. 直接输出函数名和 JSON 参数（GLM-4 原生 tools 格式）
           支持两种情况：
           a) 整段文本只有 tool call: function_name\n{json}
           b) tool call 嵌入在文本中（允许前面有自然语言描述）
           c) 函数名后跟多行内容再接 JSON（如 function_name\npath\n{json}）
        2. <tool_call>...</tool_call> 标签格式（fallback 模板引导）
        3. <function=Name>...</function> 格式（function tag）
        4. 独立 JSON 对象 {"name": "...", "arguments": {...}}

        Args:
            text: 模型生成的完整文本

        Returns:
            OpenAI 格式的 tool_calls 列表
        """
        tool_calls: list[dict[str, Any]] = []
        allowed_tool_names = _extract_allowed_tool_names(tools)

        # 策略 1a：GLM-4 原生格式（整段）— 函数名\n{json_arguments}
        glm_native_pattern = r"^(\w+)\s*\n(\{.*\})\s*$"
        match = re.match(glm_native_pattern, text.strip(), re.DOTALL)
        if match:
            func_name = match.group(1)
            try:
                arguments = json.loads(match.group(2))
                if allowed_tool_names and func_name not in allowed_tool_names:
                    return []
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                })
                return tool_calls
            except json.JSONDecodeError:
                pass

        # 策略 1b：GLM-4 原生格式（嵌入在文本中）
        # 某些情况下模型会在自然语言描述后跟上 tool call，
        # 匹配行首的 "单词\n{" 模式
        glm_embedded_pattern = _GLM_EMBEDDED_PATTERN
        for m in glm_embedded_pattern.finditer(text):
            func_name = m.group(1)
            if allowed_tool_names and func_name not in allowed_tool_names:
                continue
            json_text = m.group(2)
            try:
                arguments = json.loads(json_text)
                # 确保解析出的是 dict（避免将普通 JSON 字符串/数字误判）
                if isinstance(arguments, dict):
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    })
            except json.JSONDecodeError:
                continue

        if tool_calls:
            return tool_calls

        # 策略 1c：GLM-4 原生格式（函数名与 JSON 之间有额外行）
        # 某些小模型在输出工具调用时，函数名和 JSON 参数之间插入了额外内容
        # 例如：Write\npath/to/file\n{"content": "..."}
        # 或者：让我创建项目：Write\npath\n{"content": "..."}
        # 使用括号计数法提取最后一个 JSON 对象，与前面最近的英文函数名配对
        # 防误判策略：函数名必须是 PascalCase（首字母大写），
        # 这与 Claude Code / OpenAI 的工具命名规范一致（Write, Bash, Read, Agent 等）。
        # 纯小写的英文单词（如 dictionaries, example）不会被匹配。
        lines = text.split('\n')
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if not line.startswith('{'):
                continue
            remaining = '\n'.join(lines[i:])
            end = _find_json_span(remaining, 0)
            if end is None:
                continue
            try:
                obj = json.loads(remaining[:end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            # 额外防误判：JSON 对象必须只包含字符串值的键
            # （工具参数通常是 {key: string_value} 形式，而不是嵌套复杂结构）
            # 跳过此检查可能导致模型输出的普通 JSON 被误判为工具调用
            if not obj:
                continue
            # 向上查找函数名：最近行中的 PascalCase 标识符
            for j in range(i - 1, max(i - 5, -1), -1):
                candidate_line = lines[j].strip()
                # 优先匹配纯 PascalCase 单词行
                if re.match(r'^[A-Z]\w*$', candidate_line):
                    if allowed_tool_names and candidate_line not in allowed_tool_names:
                        continue
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": candidate_line,
                            "arguments": json.dumps(obj, ensure_ascii=False),
                        },
                    })
                    break
                # 匹配行尾的 PascalCase 标识符（如 "让我创建项目：Write"）
                m = _TRAILING_PASCALCASE_RE.search(candidate_line)
                if m:
                    func_name = m.group(1) or m.group(2)
                    if allowed_tool_names and func_name not in allowed_tool_names:
                        continue
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(obj, ensure_ascii=False),
                        },
                    })
                    break
            if tool_calls:
                break

        if tool_calls:
            return tool_calls

        # 回退到父类的默认实现（处理 <tool_call>、function tag 和独立 JSON）
        return super().parse_tool_calls(text, tools)
