"""Transformers 引擎实现（包含 LLM、Embedding、Reranker）。
- 延迟加载（首次调用时加载模型），避免在没有 GPU / 资源时程序启动失败；
- 使用 transformers / sentence-transformers（若未安装会抛出清晰提示）。
注意：模型加载和推理建议放到独立进程/容器中，避免阻塞主 API 进程。
"""

from typing import List, Optional, Dict, Any, Iterator, Set, Union
import torch
from loguru import logger

from engines.base import LLMEngine, EmbeddingEngine, RerankerEngine

# 思考模式标记 — 不应被 special token 剥离逻辑移除
_THINK_TAGS = {'<think>', '</think>'}


class TransformersLLMEngine(LLMEngine):
    """基于 Hugging Face transformers 的 LLM 引擎封装。"""

    def __init__(self, model_path: str, dtype: Optional[str] = None,
                 device: Optional[str] = None,
                 gen_params: Optional[Dict[str, Any]] = None) -> None:
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.gen_params = gen_params or {}
        self._tokenizer = None
        self._model = None
        self._device = None
        self._supports_tools: Optional[bool] = None
        self._special_tokens_to_strip: Optional[Set[str]] = None

    # ==================== 模型加载 ====================

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("transformers 未安装，请安装相关依赖") from e

        logger.info("Loading Transformers LLM from {} (device={}, dtype={})",
                     self.model_path, self.device, self.dtype)

        torch_dtype = None
        if self.dtype == 'float16':
            torch_dtype = torch.float16
        elif self.dtype == 'bfloat16':
            torch_dtype = torch.bfloat16

        use_cuda = self.device and 'cuda' in self.device.lower()
        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # 部分模型（如 GLM-4-0414）的 from_pretrained 使用 dtype 参数，
        # 标准 transformers 使用 torch_dtype。
        # 优先尝试 dtype（避免 GLM 模型的 deprecation warning），
        # 不支持时回退到 torch_dtype（标准 transformers 参数）。
        base_kwargs = dict(device_map='auto' if use_cuda else None, trust_remote_code=True)
        if torch_dtype is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, dtype=torch_dtype, **base_kwargs)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, torch_dtype=torch_dtype, **base_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **base_kwargs)

        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    # ==================== Special Token 管理 ====================

    def _get_special_tokens_to_strip(self) -> Set[str]:
        """获取需要从解码输出中剥离的 special token 集合（惰性缓存）"""
        if self._special_tokens_to_strip is None:
            tokens = set()
            if self._tokenizer is not None:
                for tok in getattr(self._tokenizer, 'all_special_tokens', []):
                    if tok not in _THINK_TAGS:
                        tokens.add(tok)
            self._special_tokens_to_strip = tokens
            logger.debug("Special tokens to strip ({}): {}", len(tokens), list(tokens)[:10])
        return self._special_tokens_to_strip

    def _clean_decoded_text(self, text: str, strip_think_tags: bool = False) -> str:
        """清理解码后的文本：移除结构性 special token

        Args:
            text: skip_special_tokens=False 解码的原始文本
            strip_think_tags: True 时同时移除 <think></think>（enable_thinking=False 场景）
        """
        for tok in self._get_special_tokens_to_strip():
            text = text.replace(tok, '')
        if strip_think_tags:
            for tok in _THINK_TAGS:
                text = text.replace(tok, '')
        return text

    # ==================== 输入构建 ====================

    def _build_inputs(self, prompt: str) -> dict:
        """将 prompt 字符串转为模型输入 tensor

        所有 prompt 均为字符串，由 tokenizer 统一编码。
        自定义 tokenizer（如 ChatGLM4）能正确处理自身的特殊标记
        （如 [gMASK]<sop>），无需手动构建 token ID。
        """
        inputs = self._tokenizer(prompt, return_tensors='pt')
        return {k: v.to(self._device) for k, v in inputs.items()}

    # ==================== Tools 探测 ====================

    def _check_tools_support(self) -> bool:
        if self._supports_tools is not None:
            return self._supports_tools
        self._ensure_loaded()
        try:
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tools=[], tokenize=False, add_generation_prompt=True)
            self._supports_tools = True
        except (TypeError, Exception):
            self._supports_tools = False
        logger.info("Tokenizer tools 支持: {} (model={})", self._supports_tools, self.model_path)
        return self._supports_tools

    # ==================== Chat Template ====================

    def apply_chat_template(
        self, messages: list[dict], tools: list[dict] | None = None,
        **kwargs,
    ) -> Optional[str]:
        """使用 HuggingFace tokenizer 原生 apply_chat_template 构建 prompt

        始终使用 tokenize=False 返回字符串。自定义 tokenizer（如 ChatGLM4）
        在 tokenize=True 时可能返回 dict/BatchEncoding 等非标准类型，
        而 tokenize=False 返回字符串是所有 tokenizer 的通用行为。

        Returns:
            prompt 字符串，或 None（不支持时）
        """
        self._ensure_loaded()
        if tools and not self._check_tools_support():
            return None
        try:
            template_kwargs = self._build_template_kwargs(
                tools=tools, tokenize=False, **kwargs)
            result = self._tokenizer.apply_chat_template(messages, **template_kwargs)
            return str(result) if result is not None and not isinstance(result, str) else result
        except TypeError:
            # enable_thinking 等参数可能不被 tokenizer 接受，降级重试（不带可选参数）
            template_kwargs_min = {"tokenize": False, "add_generation_prompt": True}
            if tools:
                template_kwargs_min["tools"] = tools
            try:
                result = self._tokenizer.apply_chat_template(messages, **template_kwargs_min)
                return str(result) if result is not None and not isinstance(result, str) else result
            except Exception as e:
                logger.warning("tokenizer.apply_chat_template 降级重试失败: {}", e)
                return None
        except Exception as e:
            logger.warning("tokenizer.apply_chat_template 失败: {}", e)
            return None

    # ==================== 生成 ====================

    def generate(
        self, prompt: str, stream: bool = False, **kwargs,
    ) -> Union[str, Iterator[str]]:
        """Override: 确保 stop words 在 raw 文本上匹配后再清理 special tokens。

        操作顺序:
        1. _generate/_generate_stream 返回含 special tokens 的原始文本
        2. 在原始文本上匹配 stop words（GLM 的 <|user|> 等 special token stop words 才能正确命中）
        3. 清理 special tokens 后返回干净文本
        """
        stop = kwargs.pop("stop", None)
        enable_thinking = kwargs.get('enable_thinking', True)
        strip_think = not enable_thinking

        if stream:
            raw_iter = self._generate_stream(prompt, **kwargs)
            if stop:
                raw_iter = self._apply_stop_stream(raw_iter, stop)
            return self._iter_clean(raw_iter, strip_think)

        raw_text = self._generate(prompt, **kwargs)
        if stop:
            raw_text = self._apply_stop_text(raw_text, stop)
        return self._clean_decoded_text(raw_text, strip_think_tags=strip_think).strip()

    def _iter_clean(self, it: Iterator[str], strip_think: bool) -> Iterator[str]:
        """对流式 chunk 逐个清理 special tokens"""
        for chunk in it:
            if chunk:
                cleaned = self._clean_decoded_text(chunk, strip_think_tags=strip_think)
                if cleaned:
                    yield cleaned

    def _generate(self, prompt: str, **kwargs) -> str:
        """返回含 special tokens 的原始解码文本（清理由 generate() 统一处理）"""
        self._ensure_loaded()
        max_new_tokens = kwargs.get('max_tokens', self.gen_params.get('max_tokens', 128))
        temperature = kwargs.get('temperature', self.gen_params.get('temperature', 0.0))
        top_p = kwargs.get('top_p', self.gen_params.get('top_p', 1.0))
        top_k = kwargs.get('top_k', self.gen_params.get('top_k', None))

        inputs = self._build_inputs(prompt)
        input_length = int(inputs['input_ids'].shape[-1])
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=top_p,
        )
        if top_k is not None:
            gen_kwargs['top_k'] = top_k
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                **gen_kwargs,
            )
        generated_ids = out[0][input_length:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=False)

    def _generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """yield 含 special tokens 的原始 chunk（清理由 generate() 统一处理）"""
        self._ensure_loaded()
        from transformers import TextIteratorStreamer
        from threading import Thread

        max_new_tokens = kwargs.get('max_tokens', self.gen_params.get('max_tokens', 128))
        temperature = kwargs.get('temperature', self.gen_params.get('temperature', 0.0))
        top_p = kwargs.get('top_p', self.gen_params.get('top_p', 1.0))
        top_k = kwargs.get('top_k', self.gen_params.get('top_k', None))

        inputs = self._build_inputs(prompt)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=False)

        generation_kwargs = dict(
            inputs, streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=top_p,
        )
        if top_k is not None:
            generation_kwargs['top_k'] = top_k

        thread_exception = None

        def _generate_target():
            nonlocal thread_exception
            try:
                self._model.generate(**generation_kwargs)
            except Exception as e:
                thread_exception = e
                streamer.end()

        thread = Thread(target=_generate_target, daemon=True)
        thread.start()

        try:
            for text_chunk in streamer:
                if text_chunk:
                    yield text_chunk
        finally:
            thread.join(timeout=30)
            if thread.is_alive():
                logger.warning("流式生成线程超时未结束 (30s)")

        if thread_exception is not None:
            raise RuntimeError(f"流式生成失败: {thread_exception}") from thread_exception


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
            raise RuntimeError("sentence-transformers 未安装") from e
        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Loading embedding model {} on device {}", self.model_path, device)
        self._model = SentenceTransformer(self.model_path, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        emb = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb.tolist()


class TransformersRerankerEngine(RerankerEngine):
    """基于 sentence-transformers CrossEncoder 的 reranker 封装。"""

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
            raise RuntimeError("sentence-transformers CrossEncoder 不可用") from e
        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Loading reranker CrossEncoder {} on device {}", self.model_path, device)
        self._model = CrossEncoder(self.model_path, device=device)

    def rerank(self, query: str, documents, top_k: Optional[int] = None) -> List[float]:
        self._ensure_loaded()
        pairs = [(query, d) for d in documents]
        raw_scores = self._model.predict(pairs)
        scores = torch.sigmoid(torch.tensor(raw_scores)).tolist()
        return scores
