"""vLLM 引擎实现（尝试优先使用 vLLM，若不可用则回退到 transformers 实现）。
注意：vLLM 对 CUDA 驱动、显存、环境有较高要求，Windows 不支持 vLLM。
"""

import uuid
from typing import Optional, Dict, Any, Iterator
from loguru import logger

from engines.base import LLMEngine

# vLLM 是否可用的标志
_VLLM_AVAILABLE = False
_LLM = None
_LLMEngine = None
_SamplingParams = None
_EngineArgs = None

try:
    from vllm import LLM as _LLM, SamplingParams as _SamplingParams
    from vllm.engine.llm_engine import LLMEngine as _LLMEngine
    from vllm.engine.arg_utils import EngineArgs as _EngineArgs
    _VLLM_AVAILABLE = True
except ImportError:
    pass


class VLLMLLMEngine(LLMEngine):
    """vLLM 引擎封装，支持高性能推理和流式输出。
    
    使用底层 LLMEngine 实现真正的流式输出。
    """
    
    def __init__(self, model_path: str, dtype: Optional[str] = None, 
                 device: Optional[str] = None, gen_params: Optional[Dict[str, Any]] = None) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.gen_params = gen_params or {}
        self._llm = None           # 高层 LLM 类（用于非流式）
        self._llm_engine = None    # 底层 LLMEngine（用于流式）
        self._fallback = None      # 若 vllm 不可用，回退到 transformers 实现

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> str | None:
        """委托给底层引擎或 fallback 的 apply_chat_template"""
        self._ensure_loaded()
        if self._fallback is not None:
            return self._fallback.apply_chat_template(messages, tools=tools, **kwargs)
        # vLLM 底层 LLMEngine 不直接暴露 tokenizer.apply_chat_template，
        # 尝试从引擎获取 tokenizer
        if self._llm_engine is not None:
            try:
                tokenizer = self._llm_engine.get_tokenizer()
                if tokenizer is None:
                    return None
                template_kwargs = self._build_template_kwargs(tools=tools, **kwargs)
                return tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as e:
                logger.warning("vLLM tokenizer.apply_chat_template 失败: {}", e)
                return None
        return None

    def _ensure_loaded(self) -> None:
        if self._llm_engine is not None or self._fallback is not None:
            return
        
        if _VLLM_AVAILABLE:
            try:
                logger.info("Loading vLLM model {}", self.model_path)
                # 构建引擎参数
                engine_kwargs = {
                    "model": self.model_path, 
                    "trust_remote_code": True,
                }
                if self.dtype:
                    engine_kwargs["dtype"] = self.dtype
                
                # 使用底层 LLMEngine 以支持流式输出
                engine_args = _EngineArgs(**engine_kwargs)
                self._llm_engine = _LLMEngine.from_engine_args(engine_args)
                logger.info("vLLM LLMEngine loaded successfully")
                return
            except Exception as e:
                # 若 float16 不支持，自动尝试 bfloat16
                if "float16" in str(e) and self.dtype == "float16":
                    logger.warning("vLLM 不支持 float16，尝试 bfloat16...")
                    try:
                        engine_kwargs["dtype"] = "bfloat16"
                        engine_args = _EngineArgs(**engine_kwargs)
                        self._llm_engine = _LLMEngine.from_engine_args(engine_args)
                        return
                    except Exception as e2:
                        logger.warning("vLLM bfloat16 也失败，回退到 transformers：{}", e2)
                else:
                    logger.warning("vLLM 加载失败（会回退到 transformers）：{}", e)
        else:
            logger.warning("vLLM 未安装，回退到 transformers")
        
        # 回退：使用 transformers
        try:
            from engines.transformers_engine import TransformersLLMEngine
            self._fallback = TransformersLLMEngine(
                self.model_path, dtype=self.dtype, device=self.device, gen_params=self.gen_params
            )
        except Exception as e2:
            raise RuntimeError("vLLM 与 transformers 均不可用，请检查依赖") from e2

    def _get_sampling_params(self, **kwargs):
        """构建 vLLM SamplingParams"""
        max_tokens = kwargs.get('max_tokens', self.gen_params.get('max_tokens', 128))
        temperature = kwargs.get('temperature', self.gen_params.get('temperature', 0.0))
        top_p = kwargs.get('top_p', self.gen_params.get('top_p', 1.0))
        
        return _SamplingParams(
            temperature=temperature if temperature > 0.0 else 0.0,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def _generate(self, prompt: str, **kwargs) -> str:
        """非流式生成"""
        self._ensure_loaded()
        if self._llm_engine is not None:
            sampling_params = self._get_sampling_params(**kwargs)
            request_id = str(uuid.uuid4())
            
            # 添加请求
            self._llm_engine.add_request(request_id, prompt, sampling_params)
            
            # 执行推理直到完成
            final_output = None
            while self._llm_engine.has_unfinished_requests():
                step_outputs = self._llm_engine.step()
                for output in step_outputs:
                    if output.request_id == request_id and output.finished:
                        final_output = output
                        break
                if final_output:
                    break
            
            if final_output and final_output.outputs:
                return final_output.outputs[0].text
            return ""
        elif self._fallback is not None:
            return self._fallback.generate(prompt, **kwargs)
        else:
            raise RuntimeError("No underlying engine available")
    
    def _generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成（使用 LLMEngine 实现真正的流式输出）"""
        self._ensure_loaded()
        
        if self._llm_engine is not None:
            sampling_params = self._get_sampling_params(**kwargs)
            request_id = str(uuid.uuid4())
            
            # 添加请求
            self._llm_engine.add_request(request_id, prompt, sampling_params)
            
            # 流式执行推理
            prev_text = ""
            while self._llm_engine.has_unfinished_requests():
                step_outputs = self._llm_engine.step()
                for output in step_outputs:
                    if output.request_id == request_id:
                        if output.outputs:
                            current_text = output.outputs[0].text
                            if len(current_text) > len(prev_text):
                                delta = current_text[len(prev_text):]
                                yield delta
                                prev_text = current_text
                        if output.finished:
                            return
        elif self._fallback is not None:
            yield from self._fallback.generate(prompt, stream=True, **kwargs)
        else:
            raise RuntimeError("No underlying engine available")