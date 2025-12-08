from typing import Dict

from core.registry import REGISTRY
from strategies.base import GenericChatStrategy, LLMStrategy
from strategies.glm import GLMChatStrategy
from strategies.qwen import QwenChatStrategy
from strategies.deepseek import DeepseekChatStrategy


class ServiceContainer:
	"""服务容器：负责提供面向业务的引用（引擎与策略）。"""

	def __init__(self) -> None:
		self._strategies: Dict[str, LLMStrategy] = {
			"generic": GenericChatStrategy(),
			"glm": GLMChatStrategy(),
			"qwen": QwenChatStrategy(),
			"deepseek": DeepseekChatStrategy(),
		}

	def get_llm_and_strategy(self, model_name: str | None):
		engine = REGISTRY.get_llm(model_name)
		strategy_key = REGISTRY.get_llm_strategy_key(model_name) or "generic"
		strategy = self._strategies.get(strategy_key, self._strategies["generic"])
		return engine, strategy

	def get_embedding(self, model_name: str | None):
		return REGISTRY.get_embedding(model_name)

	def get_reranker(self, model_name: str | None):
		return REGISTRY.get_reranker(model_name)



CONTAINER = ServiceContainer()
