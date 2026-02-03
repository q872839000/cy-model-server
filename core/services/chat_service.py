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


# 全局服务实例
CHAT_SERVICE = ChatService()
