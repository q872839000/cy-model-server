"""vLLM 引擎实现（尝试优先使用 vLLM，若不可用则回退到 transformers 实现）。
注意：vLLM 对 CUDA 驱动、显存、环境有较高要求，通常生产环境以独立服务形式部署并通过 RPC 调用。
"""

from typing import Optional, Dict, Any, Iterator
from loguru import logger

from engines.base import LLMEngine

class VLLMLLMEngine(LLMEngine):
    def __init__(self, model_path: str, dtype: Optional[str] = None, device: Optional[str] = None, gen_params: Optional[Dict[str, Any]] = None) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.gen_params = gen_params or {}
        self._engine = None
        self._fallback = None  # 若 vllm 不可用，回退到 transformers 实现

    def _ensure_loaded(self) -> None:
        if self._engine is not None or self._fallback is not None:
            return
        try:
            # 尝试导入 vllm
            from vllm import LLM
            logger.info("Loading vLLM model {}", self.model_path)
            self._engine = LLM(model=self.model_path)
        except Exception as e:
            logger.warning("vLLM 不可用或加载失败（会回退到 transformers）：{}", e)
            # 回退：使用 transformers 的简单生成器（惰性导入）
            try:
                from engines.transformers_engine import TransformersLLMEngine
                self._fallback = TransformersLLMEngine(self.model_path, dtype=self.dtype, device=self.device, gen_params=self.gen_params)
                # 不马上加载 fallback，保持延迟加载特性
            except Exception as e2:
                raise RuntimeError("vLLM 与 transformers 均不可用，请检查依赖") from e2

    def generate(self, prompt: str, **kwargs) -> str:
        self._ensure_loaded()
        if self._engine is not None:
            # 使用 vLLM API
            from vllm import SamplingParams
            
            # 参数映射
            max_tokens = kwargs.get('max_tokens', self.gen_params.get('max_tokens', 128))
            temperature = kwargs.get('temperature', self.gen_params.get('temperature', 0.0))
            top_p = kwargs.get('top_p', self.gen_params.get('top_p', 1.0))
            
            sampling_params = SamplingParams(
                temperature=temperature if temperature > 0.0 else 0.0,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            
            outputs = self._engine.generate(prompt, sampling_params)
            # vLLM返回RequestOutput对象列表
            return outputs[0].outputs[0].text
        elif self._fallback is not None:
            return self._fallback.generate(prompt, **kwargs)
        else:
            raise RuntimeError("No underlying engine available for VLLM wrapper")
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成（vLLM原生支持）"""
        self._ensure_loaded()
        if self._engine is not None:
            from vllm import SamplingParams
            
            # 参数映射
            max_tokens = kwargs.get('max_tokens', self.gen_params.get('max_tokens', 128))
            temperature = kwargs.get('temperature', self.gen_params.get('temperature', 0.0))
            top_p = kwargs.get('top_p', self.gen_params.get('top_p', 1.0))
            
            sampling_params = SamplingParams(
                temperature=temperature if temperature > 0.0 else 0.0,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            
            # vLLM流式生成
            for output in self._engine.generate(prompt, sampling_params, use_tqdm=False):
                # 增量输出新生成的文本
                if output.outputs:
                    yield output.outputs[0].text
        elif self._fallback is not None:
            # 回退到transformers的流式
            yield from self._fallback.generate_stream(prompt, **kwargs)
        else:
            raise RuntimeError("No underlying engine available for VLLM wrapper")