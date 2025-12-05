import uuid
from typing import Optional

from models.schemas import ChatRequest, ChatResponse, Choice, ChatMessage
from services.container import CONTAINER
from core.registry import REGISTRY


class ChatService:
	def __init__(self) -> None:
		pass

	def has_any_model(self) -> bool:
		return REGISTRY.has_any_llm()

	async def generate(self, req: ChatRequest) -> ChatResponse:
		engine, strategy = CONTAINER.get_llm_and_strategy(req.model)
		if engine is None:
			raise ValueError("未找到可用的LLM模型，请检查配置文件")
		text = strategy.generate(engine, [m.model_dump() for m in req.messages], max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p)
		choice = Choice(index=0, message=ChatMessage(role="assistant", content=text), finish_reason="stop")
		return ChatResponse(id=str(uuid.uuid4()), model=req.model, choices=[choice])
