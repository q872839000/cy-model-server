from typing import List

from models.schemas import EmbeddingInput, EmbeddingResponse, EmbeddingVector
from core.registry import REGISTRY


class EmbeddingService:·
	def __init__(self) -> None:
		pass

	def has_any_model(self) -> bool:
		return REGISTRY.has_any_embedding()

	async def embed(self, req: EmbeddingInput) -> EmbeddingResponse:
		engine = REGISTRY.get_embedding(req.model)
		if engine is None:
			raise ValueError("未找到可用的Embedding模型，请检查配置文件")
		vectors = engine.embed(req.input)
		data: List[EmbeddingVector] = []
		for idx, vec in enumerate(vectors):
			data.append(EmbeddingVector(index=idx, embedding=vec))
		return EmbeddingResponse(data=data, model=req.model, dimension=len(data[0].embedding) if data else None)
