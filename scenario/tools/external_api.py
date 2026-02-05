"""
外部API客户端

封装对外部仿真系统的API调用，包括：
- 装备库获取
- 装备Schema批量获取
- 想定校验
- 地理位置解析
"""
import httpx
from typing import Dict, Any, Optional, List
from loguru import logger

from core.config import Config


class ExternalAPIClient:
    """外部API客户端，所有装备相关操作都通过外部仿真系统API实现。"""
    
    def __init__(self):
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        scenario_cfg = Config.get_yaml_section("scenario") or {}
        api_cfg = scenario_cfg.get("external_api", {})
        
        self.equipment_base_url = api_cfg.get("equipment_base_url", "http://localhost:8080")
        self.validation_base_url = api_cfg.get("validation_base_url", "http://localhost:8080")
        self.geo_base_url = api_cfg.get("geo_base_url", "http://localhost:8080")
        
        self.equipment_list_path = api_cfg.get("equipment_list_path", "/api/equipment/list")
        self.equipment_schema_path = api_cfg.get("equipment_schema_path", "/api/equipment/schema/batch")
        self.validation_path = api_cfg.get("validation_path", "/api/scenario/validate")
        self.geo_resolve_path = api_cfg.get("geo_resolve_path", "/api/geo/resolve")
        
        self.timeout = api_cfg.get("timeout", 30)
        
        logger.debug("ExternalAPIClient 配置加载完成")
    
    async def get_equipment_library(self) -> Dict[str, Any]:
        """
        获取装备库列表
        
        Returns:
            {"success": bool, "data": {"total": int, "equipments": [...]}, "error": str | None}
            每个装备包含 classifies 字段，表示该装备支持的阵营列表
        """
        url = f"{self.equipment_base_url}{self.equipment_list_path}"
        
        logger.info("【外部API】获取装备库")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                logger.info("  返回装备数量: {}", data.get("total", 0))
                return {"success": True, "data": data, "error": None}
        except httpx.HTTPStatusError as e:
            logger.error("装备库获取HTTP错误: {}", e)
            return {"success": False, "data": None, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error("装备库获取异常: {}", e)
            return {"success": False, "data": None, "error": str(e)}
    
    async def get_equipment_schema(self, equip_ids: List[str]) -> Dict[str, Any]:
        """
        批量获取装备Schema
        
        Args:
            equip_ids: 装备ID列表
            
        Returns:
            {"success": bool, "data": {"schemas": {equipId: {...}, ...}}, "error": str | None}
        """
        url = f"{self.equipment_base_url}{self.equipment_schema_path}"
        
        logger.info("【外部API】批量获取装备Schema: {}", equip_ids)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json={"equipIds": equip_ids})
                response.raise_for_status()
                data = response.json()
                logger.info("  获取到 {} 个装备Schema", len(data.get("schemas", {})))
                return {"success": True, "data": data, "error": None}
        except httpx.HTTPStatusError as e:
            logger.error("获取Schema HTTP错误: {}", e)
            return {"success": False, "data": None, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error("获取Schema异常: {}", e)
            return {"success": False, "data": None, "error": str(e)}
    
    async def resolve_location(self, location_description: str) -> Dict[str, Any]:
        """
        解析地理位置
        
        Args:
            location_description: 位置描述（如"台湾海峡"）
            
        Returns:
            {"success": bool, "data": {"center": {...}, "bounds": {...}, "location_name": str}, "error": str | None}
        """
        url = f"{self.geo_base_url}{self.geo_resolve_path}"
        
        logger.info("【外部API】解析位置: {}", location_description)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json={"location": location_description})
                response.raise_for_status()
                data = response.json()
                center = data.get("center", {})
                logger.info("  解析结果: lon={}, lat={}", center.get("lon"), center.get("lat"))
                return {"success": True, "data": data, "error": None}
        except httpx.HTTPStatusError as e:
            logger.error("地理解析HTTP错误: {}", e)
            return {"success": False, "data": None, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error("地理解析异常: {}", e)
            return {"success": False, "data": None, "error": str(e)}
    
    async def validate_scenario(self, scenario_json: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        校验想定JSON
        
        Args:
            scenario_json: 想定JSON数据
            
        Returns:
            {"success": bool, "data": {"valid": bool, "errors": [...], "message": str}, "error": str | None}
        """
        url = f"{self.validation_base_url}{self.validation_path}"
        
        logger.info("【外部API】校验想定: {} 个实例", len(scenario_json))
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json={"scenario": scenario_json})
                response.raise_for_status()
                data = response.json()
                logger.info("  校验结果: valid={}", data.get("valid", False))
                return {"success": True, "data": data, "error": None}
        except httpx.HTTPStatusError as e:
            logger.error("校验API HTTP错误: {}", e)
            return {"success": False, "data": None, "error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error("校验API异常: {}", e)
            return {"success": False, "data": None, "error": str(e)}


EXTERNAL_API = ExternalAPIClient()
