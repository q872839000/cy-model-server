"""Transformers 引擎实现（包含 LLM、Embedding、Reranker）。
实现要点：
- 延迟加载（首次调用时加载模型），避免在没有 GPU / 资源时程序启动失败；
- 提供同步接口，适配现有项目的调用约定；
- 使用 transformers / sentence-transformers（若未安装会抛出清晰提示）。
- 支持真正的流式输出（token级别）
注意：真实生产环境下，模型加载和推理建议放到独立进程/容器中，避免阻塞主 API 进程。
"""

from typing import List, Optional, Dict, Any, Iterator
import torch
from loguru import logger

from engines.base import LLMEngine, EmbeddingEngine, RerankerEngine


class TransformersLLMEngine(LLMEngine):
    """基于 Hugging Face transformers 的简单 LLM 引擎封装（通用/因子车）。
    - model_path: 本地路径或 HF 仓库标识符
    - dtype: 可选，'float16' 等
    - device: 可选，'cpu' 或 'cuda:0'
    - gen_params: 生成默认参数
    """

    def __init__(self, model_path: str, dtype: Optional[str] = None, device: Optional[str] = None, gen_params: Optional[Dict[str, Any]] = None) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.gen_params = gen_params or {}
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        """在首次调用时加载模型，安全捕获导入错误并给出友好提示。"""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("transformers 未安装，请安装相关依赖") from e

        logger.info("Loading Transformers LLM from {} (device={}, dtype={})", self.model_path, self.device, self.dtype)
        
        # 解析dtype
        torch_dtype = None
        if self.dtype == 'float16':
            torch_dtype = torch.float16
        elif self.dtype == 'bfloat16':
            torch_dtype = torch.bfloat16
        
        # 解析device
        use_cuda = self.device and 'cuda' in self.device.lower()
        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map='auto' if use_cuda else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _generate(self, prompt: str, **kwargs) -> str:
        """非流式生成文本"""
        self._ensure_loaded()
        max_new_tokens = kwargs.get('max_tokens', self.gen_params.get('max_tokens', 128))
        temperature = kwargs.get('temperature', self.gen_params.get('temperature', 0.0))
        top_p = kwargs.get('top_p', self.gen_params.get('top_p', 1.0))

        inputs = self._tokenizer(prompt, return_tensors='pt').to(self._device)
        input_length = int(inputs['input_ids'].shape[-1])
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=top_p,
            )
        # 仅解码新增部分（去掉提示词）
        generated_ids = out[0][input_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()
    
    def _generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成文本（token级别的真实流式输出）"""
        self._ensure_loaded()
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        max_new_tokens = kwargs.get('max_tokens', self.gen_params.get('max_tokens', 128))
        temperature = kwargs.get('temperature', self.gen_params.get('temperature', 0.0))
        top_p = kwargs.get('top_p', self.gen_params.get('top_p', 1.0))

        inputs = self._tokenizer(prompt, return_tensors='pt').to(self._device)
        
        # 创建文本流迭代器
        streamer = TextIteratorStreamer(
            self._tokenizer, 
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # 生成参数
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=top_p,
        )
        
        # 在单独的线程中运行生成
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 逐个yield生成的文本片段
        for text_chunk in streamer:
            if text_chunk:
                yield text_chunk
        
        thread.join()


class TransformersEmbeddingEngine(EmbeddingEngine):
    """基于 sentence-transformers 的 Embedding 引擎包装。"""

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.model_path = model_path
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("sentence-transformers 未安装，请安装 sentence-transformers") from e
        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Loading embedding model {} on device {}", self.model_path, device)
        self._model = SentenceTransformer(self.model_path, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        emb = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb.tolist()


class TransformersRerankerEngine(RerankerEngine):
    """基于 sentence-transformers CrossEncoder 的 reranker 封装。
    给定 query 与 document 列表，返回得分列表（score 越大越相关）。"""

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.model_path = model_path
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except Exception as e:
            raise RuntimeError("sentence-transformers 的 CrossEncoder 不可用，请检查依赖") from e
        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Loading reranker CrossEncoder {} on device {}", self.model_path, device)
        self._model = CrossEncoder(self.model_path, device=device)

    def rerank(self, query: str, documents, top_k: Optional[int] = None) -> List[float]:
        self._ensure_loaded()
        pairs = [(query, d) for d in documents]
        # CrossEncoder.predict 返回原始 logits，需要 sigmoid 转换为 0-1 概率分数
        raw_scores = self._model.predict(pairs)
        # 应用 sigmoid 归一化
        scores = torch.sigmoid(torch.tensor(raw_scores)).tolist()
        # 若需要 top_k，可以在调用端裁剪
        return scores