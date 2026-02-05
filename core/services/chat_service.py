"""
对话服务：封装 LLM 对话的业务逻辑

职责：
1. 获取 Engine 和 Strategy
2. 构建 StrategyInput
3. 调用 Strategy.execute()
4. 返回 StrategyOutput

设计原则：
- 业务逻辑集中在此层
- Worker 层只负责异步执行
- API 层只负责请求/响应转换
"""

from typing import List, Dict, Any, Optional, Iterator, AsyncIterator

from core.registry import REGISTRY
from core.container import CONTAINER
from core.exceptions import ModelNotFoundError, InferenceError
from strategies.protocol import StrategyInput, StrategyOutput
from workers.async_worker import ASYNC_WORKER
from loguru import logger


class ChatService:
    """
    对话服务
    
    封装 LLM 对话的完整业务流程，提供同步和异步接口。
    
    使用示例:
        service = ChatService()
        
        # 非流式
        text = await service.generate(
            model_name="qwen3",
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=256,
        )
        
        # 流式
        async for chunk in service.generate_stream(
            model_name="qwen3",
            messages=[{"role": "user", "content": "你好"}],
        ):
            print(chunk, end="")
    """

    def _get_engine_and_strategy(self, model_name: Optional[str]):
        """
        获取 Engine 和 Strategy
        
        Args:
            model_name: 模型名称，None 使用默认模型
            
        Returns:
            Tuple[LLMEngine, LLMStrategy]
            
        Raises:
            ModelNotFoundError: 模型不存在
        """
        engine = REGISTRY.get_llm(model_name)
        if not engine:
            raise ModelNotFoundError(model_name or "default")
        
        strategy_key = REGISTRY.get_llm_strategy_key(model_name) or "generic"
        strategy = CONTAINER.get_strategy(strategy_key)
        
        return engine, strategy

    def build_prompt(
        self,
        model_name: Optional[str],
        messages: List[Dict[str, Any]],
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """
        构建 Prompt（用于 token 统计等场景）
        
        Args:
            model_name: 模型名称
            messages: 消息列表
            enable_thinking: 是否启用深度思考
            
        Returns:
            构建好的 prompt 字符串
        """
        try:
            _, strategy = self._get_engine_and_strategy(model_name)
        except ModelNotFoundError:
            return ""
        
        # 从模型配置获取 enable_thinking 默认值
        if enable_thinking is None:
            llm_config = REGISTRY.get_llm_config(model_name)
            if llm_config and llm_config.enable_thinking:
                enable_thinking = llm_config.enable_thinking
            else:
                enable_thinking = False
        
        return strategy.apply_chat_template(messages, enable_thinking=enable_thinking)

    def get_tokenizer(self, model_name: Optional[str]):
        """
        获取指定模型的 tokenizer
        
        Args:
            model_name: 模型名称
            
        Returns:
            tokenizer 实例或 None
        """
        try:
            engine, _ = self._get_engine_and_strategy(model_name)
        except ModelNotFoundError:
            return None
        
        # 优先从引擎获取
        tok = getattr(engine, "_tokenizer", None)
        if tok is not None:
            return tok
        
        # 尝试 fallback
        fallback = getattr(engine, "_fallback", None)
        if fallback is not None:
            tok2 = getattr(fallback, "_tokenizer", None)
            if tok2 is not None:
                return tok2
        
        # 尝试从路径加载
        model_path = getattr(engine, "model_path", None)
        if model_path:
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            except Exception:
                pass
        
        return None

    def _build_input(
        self,
        model_name: Optional[str],
        messages: List[Dict[str, Any]],
        stream: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ) -> StrategyInput:
        """
        构建 StrategyInput
        
        会自动从模型配置中读取 enable_thinking 默认值。
        """
        # 从模型配置获取 enable_thinking 默认值
        if enable_thinking is None:
            llm_config = REGISTRY.get_llm_config(model_name)
            if llm_config and llm_config.enable_thinking:
                enable_thinking = llm_config.enable_thinking
            else:
                enable_thinking = False
        
        return StrategyInput(
            messages=messages,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            enable_thinking=enable_thinking,
            extra=kwargs,
        )

    def _execute_sync(
        self,
        model_name: Optional[str],
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> str:
        """同步执行对话生成（供 Worker 调用）"""
        engine, strategy = self._get_engine_and_strategy(model_name)
        input = self._build_input(model_name, messages, stream=False, **kwargs)
        output = strategy.execute(engine, input)
        return output.get_text()

    def _execute_stream_sync(
        self,
        model_name: Optional[str],
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Iterator[str]:
        """同步执行流式对话生成（供 Worker 调用）"""
        engine, strategy = self._get_engine_and_strategy(model_name)
        input = self._build_input(model_name, messages, stream=True, **kwargs)
        output = strategy.execute(engine, input)
        yield from output.iter_chunks()

    async def generate(
        self,
        model_name: Optional[str],
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        异步生成对话回复（非流式）
        
        Args:
            model_name: 模型名称
            messages: 消息列表
            max_tokens: 最大生成 token 数
            temperature: 生成温度
            top_p: 核采样参数
            stop: 停止词列表
            enable_thinking: 是否启用深度思考
            timeout: 超时时间（秒）
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        try:
            result = await ASYNC_WORKER.run(
                self._execute_sync,
                model_name,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                enable_thinking=enable_thinking,
                timeout=timeout,
                **kwargs,
            )
            return result
        except Exception as e:
            logger.error(f"对话生成失败: {e}")
            raise InferenceError(model_name or "default", str(e))

    async def generate_stream(
        self,
        model_name: Optional[str],
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        异步生成对话回复（流式）
        
        Args:
            model_name: 模型名称
            messages: 消息列表
            max_tokens: 最大生成 token 数
            temperature: 生成温度
            top_p: 核采样参数
            stop: 停止词列表
            enable_thinking: 是否启用深度思考
            timeout: 单个 chunk 超时时间（秒）
            **kwargs: 其他参数
            
        Yields:
            文本块
        """
        try:
            async for chunk in ASYNC_WORKER.run_stream(
                self._execute_stream_sync,
                model_name,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                enable_thinking=enable_thinking,
                timeout=timeout,
                **kwargs,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"流式对话生成失败: {e}")
            raise InferenceError(model_name or "default", str(e))


    async def generate_with_tools(
        self,
        model_name: Optional[str],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        异步生成对话回复，支持Function Calling

        Args:
            model_name: 模型名称
            messages: 消息列表
            tools: Tool定义列表（OpenAI格式）
            tool_choice: Tool选择策略 ("auto" | "none" | "required")
            max_tokens: 最大生成token数
            temperature: 生成温度
            top_p: 核采样参数
            stop: 停止词列表
            enable_thinking: 是否启用深度思考
            timeout: 超时时间（秒）
            **kwargs: 其他参数

        Returns:
            {
                "content": str,           # 文本回复
                "tool_calls": [           # Tool调用列表（可能为空）
                    {
                        "id": str,
                        "function": {
                            "name": str,
                            "arguments": str  # JSON字符串
                        }
                    }
                ]
            }
        """
        import json
        import re

        # 如果没有tools，直接调用普通generate
        if not tools:
            text = await self.generate(
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                enable_thinking=enable_thinking,
                timeout=timeout,
                **kwargs,
            )
            return {"content": text, "tool_calls": []}

        # 构建带Tool的System Prompt
        tools_prompt = self._build_tools_prompt(tools)

        # 在消息开头注入Tool说明
        enhanced_messages = [
            {
                "role": "system",
                "content": tools_prompt,
            }
        ] + messages

        # 调用LLM
        text = await self.generate(
            model_name=model_name,
            messages=enhanced_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            enable_thinking=enable_thinking,
            timeout=timeout,
            **kwargs,
        )
        print(type(text))
        print(repr(text)[:300])
        print(str(text))
        # 解析Tool调用
        tool_calls = self._parse_tool_calls(text)

        # 如果有tool调用，content为空；否则返回原文
        if tool_calls:
            return {"content": "", "tool_calls": tool_calls}
        else:
            return {"content": text, "tool_calls": []}

    def _build_tools_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """构建Tool说明Prompt"""
        import json

        lines = [
            "你可以使用以下工具来完成任务。当需要使用工具时，请严格按照以下JSON格式输出：",
            "",
            '{"tool_call": {"name": "工具名称", "arguments": {参数对象}}}',
            "",
            "可用工具列表：",
            "",
        ]

        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})

            lines.append(f"### {name}")
            lines.append(f"描述: {desc}")

            # 提取参数说明
            props = params.get("properties", {})
            required = params.get("required", [])

            if props:
                lines.append("参数:")
                for pname, pinfo in props.items():
                    ptype = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    req_mark = "(必填)" if pname in required else "(可选)"
                    lines.append(f"  - {pname}: {ptype} {req_mark} - {pdesc}")

            lines.append("")

        lines.append("注意：")
        lines.append("1. 每次只能调用一个工具")
        lines.append("2. 工具调用必须使用上述JSON格式")
        lines.append("3. 如果不需要使用工具，直接用自然语言回复")

        return "\n".join(lines)

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """从LLM输出中解析Tool调用"""
        import json
        import re
        import uuid
        from loguru import logger

        tool_calls = []
        
        # 预处理：移除代码块标记、控制字符和多余空白
        clean_text = text.strip()
        # 移除不可见控制字符（保留换行和空格）
        clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean_text)
        # 移除开头的 ```json 或 ```
        clean_text = re.sub(r'^```(?:json)?\s*', '', clean_text)
        # 移除结尾的 ``` (可能有换行)
        clean_text = re.sub(r'\s*```\s*$', '', clean_text)
        clean_text = clean_text.strip()
        
        logger.debug("【Tool解析】 原始长度={}, 清理后长度={}", len(text), len(clean_text))

        # 方法1: 直接尝试解析整个文本作为JSON
        try:
            data = json.loads(clean_text)
            if "tool_call" in data:
                tc = data["tool_call"]
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                    }
                })
                logger.info("【Tool解析】 方法1成功: {}", tc.get("name"))
                return tool_calls
        except json.JSONDecodeError as e:
            logger.debug("【Tool解析】 方法1失败: {}", str(e))

        # 方法2: 使用栈匹配找到完整的 JSON 对象
        def find_json_object(s: str, start: int = 0) -> str:
            """使用栈匹配找到从 start 开始的完整 JSON 对象"""
            idx = s.find('{', start)
            if idx == -1:
                return ""
            
            stack = []
            in_string = False
            escape = False
            
            for i in range(idx, len(s)):
                c = s[i]
                if escape:
                    escape = False
                    continue
                if c == '\\' and in_string:
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    stack.append(c)
                elif c == '}':
                    stack.pop()
                    if not stack:
                        return s[idx:i+1]
            return ""
        
        # 查找所有包含 "tool_call" 的 JSON 对象（支持多个）
        search_pos = 0
        while True:
            tool_call_pos = clean_text.find('"tool_call"', search_pos)
            if tool_call_pos == -1:
                break
            
            # 向前找到 { 的位置
            start_pos = clean_text.rfind('{', search_pos, tool_call_pos)
            if start_pos != -1:
                json_str = find_json_object(clean_text, start_pos)
                if json_str:
                    try:
                        data = json.loads(json_str)
                        if "tool_call" in data:
                            tc = data["tool_call"]
                            tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                                }
                            })
                            logger.info("【Tool解析】 方法2(栈匹配)成功: {}", tc.get("name"))
                            # 移动搜索位置到当前 JSON 之后
                            search_pos = start_pos + len(json_str)
                            continue
                    except json.JSONDecodeError as e:
                        logger.debug("【Tool解析】 方法2失败: {} | JSON片段: {}...", str(e), json_str[:100])
            
            # 移动搜索位置
            search_pos = tool_call_pos + 1
        
        if tool_calls:
            return tool_calls

        # 方法3: 正则提取 tool_call 的 name 和 arguments
        # 支持嵌套的 arguments
        pattern = r'"tool_call"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*'
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            # 从 arguments 开始提取 JSON
            args_start = match.end()
            args_json = find_json_object(text, args_start)
            if args_json:
                try:
                    args = json.loads(args_json)
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        }
                    })
                    logger.info("【Tool解析】 方法3(正则+栈)成功: {}", name)
                    return tool_calls
                except json.JSONDecodeError as e:
                    logger.debug("【Tool解析】 方法3失败: {}", str(e))

        logger.warning("【Tool解析】 所有方法均失败，原文: {}", text[:300])
        return tool_calls


# 全局服务实例
CHAT_SERVICE = ChatService()
