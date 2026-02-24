"""
想定生成API路由

提供无状态的想定生成接口，历史消息由调用方传入。
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from models.scenario_schemas import (
    ScenarioRequest,
    ScenarioResponse,
)
from scenario.memory import MEMORY_MANAGER


router = APIRouter(prefix="/v1/scenario", tags=["scenario"])


def _get_scenario_service():
    """获取ScenarioService实例"""
    from core.container import CONTAINER
    return CONTAINER.get_scenario_service()


@router.post("/chat", response_model=ScenarioResponse)
async def chat(
    request: ScenarioRequest,
    service=Depends(_get_scenario_service),
) -> ScenarioResponse:
    """
    想定生成对话接口（无状态设计）
    
    调用方需要传入历史消息，服务端仅管理轻量级记忆（中间状态）。
    
    Args:
        request: 包含messages历史消息、当前message、可选的memory_id
        
    Returns:
        ScenarioResponse: 包含memory_id、phase、message等
    """
    logger.info("收到想定对话请求: memory_id={}", request.memory_id)
    
    try:
        response = await service.chat(
            user_message=request.message,
            messages=request.messages or [],
            memory_id=request.memory_id,
            model=request.model,
        )
        return response
    except Exception as e:
        logger.exception("想定对话异常")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """获取服务统计信息"""
    return {
        "memory_stats": MEMORY_MANAGER.get_stats(),
    }
