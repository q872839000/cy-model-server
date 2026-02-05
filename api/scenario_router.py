"""
想定生成API路由

提供无状态的想定生成接口，历史消息由调用方传入。
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from models.scenario_schemas import (
    ScenarioRequest,
    ScenarioResponse,
)
from scenario.memory import MEMORY_MANAGER
from scenario.tools.external_api import EXTERNAL_API


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


@router.get("/memory/{memory_id}")
async def get_memory(memory_id: str) -> Dict[str, Any]:
    """
    获取记忆状态
    
    Args:
        memory_id: 记忆ID
        
    Returns:
        记忆状态信息
    """
    memory = MEMORY_MANAGER.get(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {
        "memory_id": memory.memory_id,
        "phase": memory.phase,
        "equipment_plan": memory.equipment_plan,
        "allocated_positions": memory.allocated_positions,
        "current_instances_count": len(memory.current_instances),
        "validation_errors": memory.validation_errors,
        "validation_attempts": memory.validation_attempts,
    }


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str) -> Dict[str, Any]:
    """
    删除记忆
    
    Args:
        memory_id: 记忆ID
        
    Returns:
        删除结果
    """
    success = MEMORY_MANAGER.delete(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {"success": True, "message": f"Memory {memory_id} deleted"}


@router.get("/equipment")
async def get_equipment_library() -> Dict[str, Any]:
    """
    获取装备库列表（代理外部API）
    
    返回所有可用装备，每个装备包含其支持的阵营列表(classifies)。
        
    Returns:
        装备列表
    """
    logger.info("【API】获取装备库列表")
    result = await EXTERNAL_API.get_equipment_library()
    
    if not result["success"]:
        logger.error("获取装备库失败: {}", result["error"])
        raise HTTPException(status_code=502, detail=result["error"])
    
    logger.info("获取装备库成功: {} 个装备", result["data"].get("total", 0))
    return result["data"]


@router.post("/equipment/schema")
async def get_equipment_schema(equip_ids: List[str]) -> Dict[str, Any]:
    """
    批量获取装备Schema（代理外部API）
    
    Args:
        equip_ids: 装备ID列表
        
    Returns:
        装备Schema字典 {equipId: schema}
    """
    logger.info("【API】获取装备Schema: {}", equip_ids)
    result = await EXTERNAL_API.get_equipment_schema(equip_ids)
    
    if not result["success"]:
        logger.error("获取Schema失败: {}", result["error"])
        raise HTTPException(status_code=502, detail=result["error"])
    
    logger.info("获取Schema成功: {} 个装备", len(result["data"].get("schemas", {})))
    return result["data"]


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """获取服务统计信息"""
    return {
        "memory_stats": MEMORY_MANAGER.get_stats(),
    }
