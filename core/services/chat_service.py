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
from core.exceptions import ModelServerException, ModelNotFoundError, InferenceError
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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> str:
        """
        构建 Prompt（用于 token 统计等场景）

        当 tools 存在时，会尝试引擎原生模板或回退注入，以使返回的 prompt
        与真实生成路径一致，从而获得更准确的 prompt_tokens 统计。
        
        Args:
            model_name: 模型名称
            messages: 消息列表
            enable_thinking: 是否启用深度思考
            tools: OpenAI 格式的工具定义列表（影响 prompt 长度统计）
            
        Returns:
            构建好的 prompt 字符串
        """
        try:
            engine, strategy = self._get_engine_and_strategy(model_name)
        except ModelNotFoundError:
            return ""
        
        # 从模型配置获取 enable_thinking 默认值
        if enable_thinking is None:
            llm_config = REGISTRY.get_llm_config(model_name)
            if llm_config and llm_config.enable_thinking:
                enable_thinking = llm_config.enable_thinking
            else:
                enable_thinking = False

        # 当 tools 存在时，镜像 Strategy.execute() 的路径选择逻辑：
        # - tool_choice 为默认值(None/"auto")时，尝试引擎原生模板
        # - tool_choice 为非默认值时，强制走 fallback（注入行为指令）
        if tools:
            _tc = tool_choice
            tool_choice_is_default = (_tc is None or _tc == "auto")
            if tool_choice_is_default:
                prompt = engine.apply_chat_template(
                    messages, tools=tools, enable_thinking=enable_thinking,
                )
                if prompt is not None:
                    return prompt
            # 原生模板不支持或 tool_choice 非默认，回退：注入 tools + tool_choice 指令
            injected = strategy.inject_tools_into_messages(messages, tools, tool_choice)
            return strategy.apply_chat_template(
                injected, enable_thinking=enable_thinking,
            )

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

    def parse_tool_calls(
        self,
        model_name: Optional[str],
        text: str,
    ) -> list[dict]:
        """从模型输出文本中解析工具调用（委托给对应 Strategy）

        Router 层不应直接依赖 Strategy 层的内部函数，
        通过 Service 层暴露此能力，保持分层架构。

        Args:
            model_name: 模型名称
            text: 模型生成的完整文本

        Returns:
            OpenAI 格式的 tool_calls 列表，空列表表示无工具调用
        """
        try:
            _, strategy = self._get_engine_and_strategy(model_name)
        except ModelNotFoundError:
            return []
        return strategy.parse_tool_calls(text)

    # 动态预算安全余量（预留给模板标记、特殊 token 等）
    _SAFETY_MARGIN_TOKENS = 256
    # 当所有探测方式都失败时的全局兜底值
    _FALLBACK_MAX_TOKENS = 4096

    def get_context_window(self, model_name: Optional[str]) -> Optional[int]:
        """获取指定模型的上下文窗口大小

        优先级：
        1. 模型配置 context_window（手动指定，最高优先级）
        2. 引擎自动探测（从模型 config.json 读取 max_position_embeddings 等）

        Args:
            model_name: 模型名称

        Returns:
            上下文窗口 token 数，无法获取时返回 None
        """
        # 1. 先查模型配置的显式声明
        llm_config = REGISTRY.get_llm_config(model_name)
        if llm_config and llm_config.context_window:
            return llm_config.context_window

        # 2. 引擎自动探测
        engine = REGISTRY.get_llm(model_name)
        if engine is not None:
            detected = engine.get_context_window()
            if detected is not None:
                return detected

        return None

    def _get_hardware_max_tokens(self, model_name: Optional[str]) -> Optional[int]:
        """获取硬件级硬上限（gen_params.max_tokens）

        gen_params.max_tokens 的语义是：你的硬件实际能承受的最大单次输出 token 数。
        动态预算的结果不应超过此值，否则可能因 KV cache 分配导致 OOM。

        Args:
            model_name: 模型名称

        Returns:
            硬件上限值，未配置时返回 None
        """
        llm_config = REGISTRY.get_llm_config(model_name)
        if llm_config and llm_config.gen_params:
            val = llm_config.gen_params.get("max_tokens")
            if val is not None:
                return int(val)
        return None

    def resolve_max_tokens(
        self,
        model_name: Optional[str],
        prompt_tokens: int,
        client_max_tokens: Optional[int] = None,
    ) -> int:
        """动态计算本次请求的 max_tokens

        三层保护机制：
        1. context_window 动态预算 → 不超出模型上下文限制
        2. gen_params.max_tokens 硬上限 → 不超出硬件承受能力（防 OOM）
        3. 客户端传入值 → 尊重客户端意愿

        最终结果 = min(动态预算, 硬件上限, 客户端值)

        配置指南：
        - context_window: 模型理论上下文窗口（可自动探测，一般不用配）
        - gen_params.max_tokens: 你的显卡实际能承受的最大输出长度（建议配置）
          例如 8GB 显存跑 1.7B 模型，建议配 2048~4096

        Args:
            model_name: 模型名称
            prompt_tokens: 本次请求的输入 token 数
            client_max_tokens: 客户端显式传入的 max_tokens（可选）

        Returns:
            本次请求应使用的 max_tokens 值（保证 >= 1）
        """
        # 收集所有候选上限
        candidates: list[int] = []

        # 1. 动态预算：context_window - prompt_tokens - safety_margin
        context_window = self.get_context_window(model_name)
        if context_window is not None and prompt_tokens > 0:
            available = context_window - prompt_tokens - self._SAFETY_MARGIN_TOKENS
            if available > 0:
                candidates.append(available)

        # 2. 硬件上限：gen_params.max_tokens
        hw_limit = self._get_hardware_max_tokens(model_name)
        if hw_limit is not None:
            candidates.append(hw_limit)

        # 3. 客户端显式传入值
        if client_max_tokens is not None:
            candidates.append(client_max_tokens)

        # 取所有候选值中的最小值
        if candidates:
            resolved = max(min(candidates), 1)
        else:
            resolved = self._FALLBACK_MAX_TOKENS

        logger.debug(
            "resolve_max_tokens: model={} prompt_tokens={} context_window={} "
            "hw_limit={} client={} → resolved={}",
            model_name, prompt_tokens, context_window, hw_limit,
            client_max_tokens, resolved,
        )
        return resolved

    def strip_tool_call_text(self, text: str, tool_calls: list[dict]) -> str:
        """从模型输出中移除已解析的工具调用文本，返回干净的 content

        Router 层不应直接依赖 Strategy 层的内部函数，
        通过 Service 层暴露此能力，保持分层架构。

        Args:
            text: 模型生成的完整文本
            tool_calls: 已解析的 OpenAI 格式 tool_calls 列表

        Returns:
            移除工具调用文本后的干净字符串
        """
        from strategies.base import _strip_tool_call_text
        return _strip_tool_call_text(text, tool_calls)

    def _build_input(
        self,
        model_name: Optional[str],
        messages: List[Dict[str, Any]],
        stream: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
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
            top_k=top_k,
            stop=stop,
            enable_thinking=enable_thinking,
            tools=tools,
            tool_choice=tool_choice,
            extra=kwargs,
        )

    def _execute_sync(
        self,
        model_name: Optional[str],
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> StrategyOutput:
        """同步执行对话生成（供 Worker 调用）

        返回 StrategyOutput，包含 text、tool_calls、finish_reason 等。
        """
        engine, strategy = self._get_engine_and_strategy(model_name)
        input = self._build_input(model_name, messages, stream=False, **kwargs)
        output = strategy.execute(engine, input)
        # 确保 text 已求值（消费 stream_iterator，如有）
        output.get_text()
        return output

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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> StrategyOutput:
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
            tools: OpenAI 格式的工具定义列表
            tool_choice: 工具选择策略
            timeout: 超时时间（秒）
            **kwargs: 其他参数
            
        Returns:
            StrategyOutput: 统一输出结构（text + tool_calls + finish_reason）
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
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                **kwargs,
            )
            return result
        except ModelServerException:
            # 保留原始异常语义（ModelNotFoundError → 404, UnsupportedParameterError → 400 等）
            raise
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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
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
            tools: OpenAI 格式的工具定义列表
            tool_choice: 工具选择策略
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
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                **kwargs,
            ):
                yield chunk
        except ModelServerException:
            # 保留原始异常语义
            raise
        except Exception as e:
            logger.error(f"流式对话生成失败: {e}")
            raise InferenceError(model_name or "default", str(e))


# 全局服务实例
CHAT_SERVICE = ChatService()
