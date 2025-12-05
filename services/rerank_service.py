from typing import List

from models.schemas import RerankQuery, RerankResponse, RerankItem
from core.registry import REGISTRY


class RerankService:
	def __init__(self) -> None:
		pass

	def has_any_model(self) -> bool:
		return REGISTRY.has_any_reranker()

	async def rerank(self, req: RerankQuery) -> RerankResponse:
		engine = REGISTRY.get_reranker(req.model)
		if engine is None:
			raise ValueError("未找到可用的Reranker模型，请检查配置文件")
		scores = engine.rerank(req.query, req.documents, req.top_k)
		items: List[RerankItem] = []
		for idx, (doc, score) in enumerate(zip(req.documents, scores)):
			items.append(RerankItem(index=idx, document=doc, score=float(score)))
		# 若提供top_k，截断
		if req.top_k and req.top_k > 0:
			items = sorted(items, key=lambda x: x.score, reverse=True)[: req.top_k]
		return RerankResponse(items=items, model=req.model)
