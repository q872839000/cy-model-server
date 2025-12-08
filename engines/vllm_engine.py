"""vLLM 引擎实现（尝试优先使用 vLLM，若不可用则回退到 transformers 实现）。
注意：vLLM 对 CUDA 驱动、显存、环境有较高要求，Windows 不支持 vLLM。
"""

from typing import Optional, Dict, Any, Iterator
from loguru import logger

from engines.base import LLMEngine

# vLLM 是否可用的标志
_VLLM_AVAILABLE = False
_LLM = None
_SamplingParams = None

try:
    from vllm import LLM as _LLM, SamplingParams as _SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    pass


class VLLMLLMEngine(LLMEngine):
    """vLLM 引擎封装，支持高性能推理和流式输出。"""
    
    def __init__(self, model_path: str, dtype: Optional[str] = None, 
                 device: Optional[str] = None, gen_params: Optional[Dict[str, Any]] = None) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.gen_params = gen_params or {}
        self._engine = None
        self._fallback = None  # 若 vllm 不可用，回退到 transformers 实现

    def _ensure_loaded(self) -> None:
        if self._engine is not None or self._fallback is not None:
            return
        
        if _VLLM_AVAILABLE:
            try:
                logger.info("Loading vLLM model {}", self.model_path)
                # vLLM 配置
                engine_kwargs = {"model": self.model_path, "trust_remote_code": True}
                if self.dtype:
                    engine_kwargs["dtype"] = self.dtype
                self._engine = _LLM(**engine_kwargs)
                return
            except Exception as e:
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
        if self._engine is not None:
            sampling_params = self._get_sampling_params(**kwargs)
            outputs = self._engine.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text
        elif self._fallback is not None:
            return self._fallback.generate(prompt, **kwargs)
        else:
            raise RuntimeError("No underlying engine available")
    
    def _generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成（vLLM 原生支持）"""
        self._ensure_loaded()
        
        if self._engine is not None:
            sampling_params = self._get_sampling_params(**kwargs)
            # vLLM 流式生成：使用 generate 的迭代模式，输出增量内容
            prev_text = ""
            for output in self._engine.generate([prompt], sampling_params, use_tqdm=False):
                current_text = output.outputs[0].text
                if len(current_text) > len(prev_text):
                    delta = current_text[len(prev_text):]
                    yield delta
                    prev_text = current_text
        elif self._fallback is not None:
            yield from self._fallback.generate(prompt, stream=True, **kwargs)
        else:
            raise RuntimeError("No underlying engine available")