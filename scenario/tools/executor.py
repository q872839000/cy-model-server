"""
Tool执行器

负责执行Tool调用，通过外部API实现装备相关操作。
"""
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from scenario.memory import ScenarioMemory
from scenario.generators import InstanceNameGenerator
from .external_api import ExternalAPIClient


@dataclass
class ToolCallResult:
    """Tool调用结果"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None


class ToolExecutor:
    """Tool执行器，通过外部API执行装备相关操作。"""
    
    def __init__(self, api_client: ExternalAPIClient):
        self.api_client = api_client
        self._handlers = {
            "get_equipment_library": self._handle_get_equipment_library,
            "get_equipment_schema": self._handle_get_equipment_schema,
            "set_equipment_plan": self._handle_set_equipment_plan,
            "resolve_location": self._handle_resolve_location,
            "create_scenario_instance": self._handle_create_instance,
            "modify_instances": self._handle_modify_instances,
            "delete_instances": self._handle_delete_instances,
            "fix_instance_field": self._handle_fix_field,
            "validate_scenario": self._handle_validate,
        }
    
    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator] = None,
    ) -> ToolCallResult:
        """执行Tool调用"""
        handler = self._handlers.get(tool_name)
        if not handler:
            logger.warning("未知的Tool: {}", tool_name)
            return ToolCallResult(success=False, error=f"未知的Tool: {tool_name}")
        
        try:
            return await handler(arguments, memory, name_generator)
        except Exception as e:
            logger.exception("Tool {} 执行异常", tool_name)
            return ToolCallResult(success=False, error=str(e))
    
    async def _handle_get_equipment_library(
        self,
        args: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator],
    ) -> ToolCallResult:
        """获取装备库列表（装备包含支持的阵营列表）"""
        logger.info("【Tool执行】 get_equipment_library")
        result = await self.api_client.get_equipment_library()
        
        if result["success"]:
            data = result["data"]
            equipments = data.get("equipments", [])
            memory.equipment_library = equipments
            memory.update()
            logger.info("【Tool结果】 装备库获取成功: {}个装备", len(equipments))
            for eq in equipments[:5]:  # 只记录前5个
                logger.debug("  - {} ({}), 阵营: {}", eq.get("equipName"), eq.get("equipId"), eq.get("classifies", []))
            return ToolCallResult(
                success=True,
                data=data,
                message=f"获取到 {len(equipments)} 个装备"
            )
        logger.error("【Tool失败】 get_equipment_library: {}", result["error"])
        return ToolCallResult(success=False, error=result["error"])
    
    async def _handle_get_equipment_schema(
        self,
        args: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator],
    ) -> ToolCallResult:
        """批量获取装备Schema"""
        equip_ids = args.get("equip_ids", [])
        logger.info("【Tool执行】 get_equipment_schema: {}", equip_ids)
        
        if not equip_ids:
            logger.warning("【Tool失败】 缺少equip_ids参数")
            return ToolCallResult(success=False, error="缺少equip_ids参数")
        
        result = await self.api_client.get_equipment_schema(equip_ids)
        
        if result["success"]:
            data = result["data"]
            schemas = data.get("schemas", {})
            memory.equipment_schemas = schemas
            memory.update()
            logger.info("【Tool结果】 Schema获取成功: {}个", len(schemas))
            for eid, schema in schemas.items():
                ports = schema.get("ports", [])
                logger.debug("  - {}: {}个端口", eid, len(ports))
            return ToolCallResult(
                success=True,
                data=data,
                message=f"获取到 {len(schemas)} 个装备的Schema"
            )
        logger.error("【Tool失败】 get_equipment_schema: {}", result["error"])
        return ToolCallResult(success=False, error=result["error"])
    
    async def _handle_set_equipment_plan(
        self,
        args: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator],
    ) -> ToolCallResult:
        """设置装备部署计划"""
        plan = args.get("plan", [])
        logger.info("【Tool执行】 set_equipment_plan: {}项计划", len(plan))
        
        if not plan:
            logger.warning("【Tool失败】 缺少plan参数或计划为空")
            return ToolCallResult(success=False, error="缺少plan参数或计划为空")
        
        # 验证计划中的装备ID是否存在于装备库
        valid_equip_ids = {eq.get("equipId") for eq in memory.equipment_library}
        invalid_items = []
        
        for item in plan:
            equip_id = item.get("equipId")
            if equip_id not in valid_equip_ids:
                invalid_items.append(equip_id)
        
        if invalid_items:
            logger.warning("【Tool失败】 计划中包含无效装备ID: {}", invalid_items)
            return ToolCallResult(
                success=False, 
                error=f"计划中包含无效装备ID: {invalid_items}"
            )
        
        # 设置装备计划
        memory.equipment_plan = plan
        memory.update()
        
        # 日志记录计划详情
        logger.info("【Tool结果】 装备计划设置成功: {}项", len(plan))
        for item in plan:
            logger.debug("  - {} x{} [{}] @ {}", 
                        item.get("equipName", item.get("equipId")),
                        item.get("count", 1),
                        item.get("classify"),
                        item.get("location", "未指定"))
        
        return ToolCallResult(
            success=True,
            data={"plan": plan},
            message=f"装备计划已设置: {len(plan)}项"
        )
    
    async def _handle_resolve_location(
        self,
        args: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator],
    ) -> ToolCallResult:
        """解析地理位置"""
        location_desc = args.get("location_description")
        logger.info("【Tool执行】 resolve_location: {}", location_desc)
        
        if not location_desc:
            logger.warning("【Tool失败】 缺少location_description参数")
            return ToolCallResult(success=False, error="缺少location_description参数")
        
        result = await self.api_client.resolve_location(location_desc)
        
        if result["success"]:
            data = result["data"]
            center = data.get("center", {})
            if center:
                position = {
                    "lon": center.get("lon"),
                    "lat": center.get("lat"),
                    "alt": center.get("alt", 0),
                }
                memory.add_position(position)
                logger.info("【Tool结果】 位置解析成功: {} -> lon={}, lat={}", 
                           data.get("location_name", location_desc), position["lon"], position["lat"])
            return ToolCallResult(
                success=True,
                data=data,
                message=f"解析位置成功: {data.get('location_name', location_desc)}"
            )
        logger.error("【Tool失败】 resolve_location: {}", result["error"])
        return ToolCallResult(success=False, error=result["error"])
    
    async def _handle_create_instance(
        self,
        args: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator],
    ) -> ToolCallResult:
        """创建装备实例（支持单个或批量创建）"""
        equip_id = args.get("equip_id")
        classify = args.get("classify")
        count = args.get("count", 1)
        
        # 处理批量创建格式：equip_id为列表时递归调用
        if isinstance(equip_id, list):
            logger.info("【Tool执行】 create_scenario_instance (批量): {} 种装备", len(equip_id))
            all_created = []
            for eid in equip_id:
                # 获取该装备的数量
                if isinstance(count, dict):
                    eid_count = count.get(eid, 1)
                else:
                    eid_count = count
                # 获取该装备的阵营（如果classify也是列表，取第一个或让系统自动判断）
                eid_classify = classify[0] if isinstance(classify, list) else classify
                # 递归调用单个创建
                single_args = {
                    "equip_id": eid,
                    "classify": eid_classify,
                    "count": eid_count,
                    "base_position": args.get("base_position"),
                    "port_values": args.get("port_values", {})
                }
                result = await self._handle_create_instance(single_args, memory, name_generator)
                if result.success:
                    all_created.extend(result.data.get("instances", []))
            
            if all_created:
                return ToolCallResult(
                    success=True,
                    data={"instances": all_created, "count": len(all_created)},
                    message=f"已批量创建 {len(all_created)} 个实例"
                )
            return ToolCallResult(success=False, error="批量创建失败")
        
        logger.info("【Tool执行】 create_scenario_instance: equip_id={}, classify={}, count={}", 
                   equip_id, classify, count)
        
        if not equip_id or not classify:
            logger.warning("【Tool失败】 缺少equip_id或classify参数")
            return ToolCallResult(success=False, error="缺少equip_id或classify参数")
        
        base_position = args.get("base_position")
        port_values = args.get("port_values", {})
        
        # 从记忆中获取装备信息（支持按名称匹配）
        equip_info = self._find_equipment(memory, equip_id)
        if equip_info:
            # 使用装备库中的真实equipId
            real_equip_id = equip_info.get("equipId", equip_id)
            equip_name = equip_info.get("equipName", equip_id)
            # 验证并修正阵营（必须是装备支持的阵营）
            valid_classifies = equip_info.get("classifies", [])
            if valid_classifies and classify not in valid_classifies:
                old_classify = classify
                classify = valid_classifies[0]  # 使用装备支持的第一个阵营
                logger.warning("【阵营修正】 {} 不支持{}, 改为{}", equip_name, old_classify, classify)
            logger.debug("【装备匹配】 {} -> {} ({}, 阵营:{})", equip_id, real_equip_id, equip_name, classify)
        else:
            real_equip_id = equip_id
            equip_name = equip_id
            logger.warning("【装备匹配】 未找到装备: {}", equip_id)
        
        # 获取Schema（尝试用真实ID和原始ID）
        schema = memory.equipment_schemas.get(real_equip_id, {}) or memory.equipment_schemas.get(equip_id, {})
        
        # 如果没有Schema，自动获取
        if not schema and real_equip_id:
            logger.debug("【自动获取Schema】 {}", real_equip_id)
            schema_result = await self.api_client.get_equipment_schema([real_equip_id])
            if schema_result.get("success"):
                schemas = schema_result.get("data", {}).get("schemas", {})
                if real_equip_id in schemas:
                    schema = schemas[real_equip_id]
                    memory.equipment_schemas[real_equip_id] = schema
                    logger.debug("【Schema获取成功】 {} 有{}个端口", real_equip_id, len(schema.get("ports", [])))
        
        ports = schema.get("ports", [])
        
        # 如果没有base_position，尝试使用最新分配的位置
        if not base_position and memory.allocated_positions:
            base_position = memory.allocated_positions[-1]  # 使用最后一个（最新解析的）位置
        
        created_instances = []
        for i in range(count):
            # 生成实例名称（基于equipName而非equipId）
            if name_generator:
                instance_name = name_generator.generate(equip_id, classify, equip_name)
            else:
                instance_name = f"{equip_name}_{classify}_{i + 1}"
            
            # 构建inits
            inits = self._build_inits(ports, port_values, base_position, i, count)
            
            instance = {
                "equipId": real_equip_id,  # 使用装备库中的真实ID
                "otherName": instance_name,
                "classify": classify,
                "inits": inits,
            }
            
            memory.add_instance(instance)
            created_instances.append(instance_name)
            logger.debug("【实例创建】 {}: {} -> {}", i+1, equip_id, instance_name)
        
        logger.info("【Tool结果】 已创建 {}个实例: {}", count, created_instances)
        return ToolCallResult(
            success=True,
            data={"instances": created_instances},
            message=f"已创建 {count} 个装备实例: {', '.join(created_instances)}"
        )
    
    async def _handle_modify_instances(
        self,
        args: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator],
    ) -> ToolCallResult:
        """批量修改已有装备实例的参数"""
        filter_cond = args.get("filter", {})
        updates = args.get("updates", {})
        
        port_id = updates.get("port_id")
        values = updates.get("values", {})
        
        # 如果是位置端口且values为空，从memory获取最新位置
        if port_id and "position" in port_id.lower() and not values:
            if memory.allocated_positions:
                pos = memory.allocated_positions[-1]
                values = {"lon": pos.get("lon"), "lat": pos.get("lat"), "alt": pos.get("alt", 10000)}
                logger.debug("【Tool】 动态获取位置: {}", values)
        
        logger.info("【Tool执行】 modify_instances: filter={}, port_id={}, values={}", 
                   filter_cond, port_id, values)
        
        if not port_id or not values:
            logger.warning("【Tool失败】 缺少port_id或values参数")
            return ToolCallResult(success=False, error="缺少port_id或values参数")
        
        if not memory.current_instances:
            logger.warning("【Tool失败】 没有可修改的实例")
            return ToolCallResult(success=False, error="没有可修改的实例")
        
        # 筛选要修改的实例
        modified_count = 0
        modified_names = []
        
        for instance in memory.current_instances:
            # 检查是否匹配筛选条件
            if not self._match_filter(instance, filter_cond):
                continue
            
            # 修改实例的inits
            inits = instance.get("inits", [])
            port_found = False
            
            for init in inits:
                if init.get("portId") == port_id:
                    # 更新现有端口值
                    try:
                        init_data = json.loads(init.get("initJson", "{}"))
                    except:
                        init_data = {}
                    
                    # 合并新值
                    init_data.update(values)
                    init["initJson"] = json.dumps(init_data, ensure_ascii=False)
                    port_found = True
                    break
            
            # 如果端口不存在，添加新端口
            if not port_found:
                inits.append({
                    "portId": port_id,
                    "initJson": json.dumps(values, ensure_ascii=False)
                })
            
            modified_count += 1
            modified_names.append(instance.get("otherName", "?"))
            logger.debug("【实例修改】 {}: {}={}", instance.get("otherName"), port_id, values)
        
        if modified_count == 0:
            logger.warning("【Tool失败】 没有匹配筛选条件的实例")
            return ToolCallResult(success=False, error="没有匹配筛选条件的实例")
        
        memory.update()
        logger.info("【Tool结果】 已修改 {}个实例: {}", modified_count, modified_names)
        
        return ToolCallResult(
            success=True,
            data={"modified_count": modified_count, "instances": modified_names},
            message=f"已修改 {modified_count} 个实例的 {port_id}: {values}"
        )
    
    async def _handle_delete_instances(
        self,
        args: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator],
    ) -> ToolCallResult:
        """删除装备实例"""
        filter_cond = args.get("filter", {})
        
        logger.info("【Tool执行】 delete_instances: filter={}", filter_cond)
        
        if not memory.current_instances:
            logger.warning("【Tool失败】 没有可删除的实例")
            return ToolCallResult(success=False, error="没有可删除的实例")
        
        # 筛选要保留的实例（不匹配filter的）
        remaining = []
        deleted_names = []
        
        for instance in memory.current_instances:
            if self._match_filter(instance, filter_cond):
                deleted_names.append(instance.get("otherName", "?"))
            else:
                remaining.append(instance)
        
        if not deleted_names:
            logger.warning("【Tool失败】 没有匹配筛选条件的实例")
            return ToolCallResult(success=False, error="没有匹配筛选条件的实例")
        
        memory.current_instances = remaining
        memory.update()
        
        logger.info("【Tool结果】 已删除 {}个实例: {}", len(deleted_names), deleted_names)
        
        return ToolCallResult(
            success=True,
            data={"deleted_count": len(deleted_names), "instances": deleted_names},
            message=f"已删除 {len(deleted_names)} 个实例: {', '.join(deleted_names)}"
        )
    
    def _match_filter(self, instance: Dict, filter_cond: Dict) -> bool:
        """检查实例是否匹配筛选条件"""
        # 如果指定了all=True，匹配所有
        if filter_cond.get("all"):
            return True
        
        # 按equipId筛选
        if "equipId" in filter_cond:
            if instance.get("equipId") != filter_cond["equipId"]:
                return False
        
        # 按classify筛选
        if "classify" in filter_cond:
            if instance.get("classify") != filter_cond["classify"]:
                return False
        
        # 按实例名称筛选
        if "otherName" in filter_cond:
            if instance.get("otherName") != filter_cond["otherName"]:
                return False
        
        # 如果没有指定任何条件，默认不匹配（安全起见）
        if not filter_cond:
            return False
        
        return True
    
    def _find_equipment(self, memory: ScenarioMemory, equip_id: str) -> Optional[Dict]:
        """从记忆中查找装备信息（支持按equipId或equipName匹配）"""
        equip_id_lower = equip_id.lower().replace("-", "").replace("_", "")
        for equip in memory.equipment_library:
            # 精确匹配equipId
            if equip.get("equipId") == equip_id:
                return equip
            # 模糊匹配equipName（忽略大小写和符号）
            equip_name = equip.get("equipName", "")
            equip_name_lower = equip_name.lower().replace("-", "").replace("_", "")
            if equip_id_lower == equip_name_lower or equip_id_lower in equip_name_lower:
                return equip
        return None
    
    def _build_inits(
        self,
        ports: list,
        port_values: Dict[str, Any],
        base_position: Optional[Dict],
        index: int,
        total: int,
    ) -> list:
        """构建初始化参数列表"""
        inits = []
        
        for port in ports:
            port_id = port.get("portId")
            if not port_id:
                continue
            
            # 检查是否是位置端口（包含lon/lat参数）
            is_position_port = self._is_position_port(port)
            
            if is_position_port and base_position:
                # 位置端口：自动计算位置
                init_data = self._calculate_position(base_position, index, total)
            elif port_id in port_values:
                # 使用用户指定的值
                init_data = port_values[port_id]
            else:
                # 使用默认值构建
                init_data = self._build_default_port_value(port)
            
            inits.append({
                "portId": port_id,
                "initJson": json.dumps(init_data, ensure_ascii=False) if isinstance(init_data, dict) else str(init_data)
            })
        
        return inits
    
    def _is_position_port(self, port: Dict) -> bool:
        """检查是否是位置端口"""
        port_id = port.get("portId", "").lower()
        port_name = port.get("portName", "").lower()
        
        # 通过端口ID或名称判断
        if "position" in port_id or "位置" in port_name:
            return True
        
        # 通过params判断（检查是否有lon/lat参数）
        params = port.get("params", [])
        param_names = {p.get("name", "").lower() for p in params}
        if "lon" in param_names and "lat" in param_names:
            return True
        
        return False
    
    def _build_default_port_value(self, port: Dict) -> Dict:
        """根据端口Schema构建默认值"""
        result = {}
        params = port.get("params", [])
        
        for param in params:
            name = param.get("name")
            if not name:
                continue
            
            # 使用default值或根据类型生成默认值
            if "default" in param:
                result[name] = param["default"]
            elif param.get("type") == "number":
                result[name] = 0
            elif param.get("type") == "integer":
                result[name] = 0
            elif param.get("type") == "string":
                result[name] = ""
            elif param.get("type") == "boolean":
                result[name] = False
        
        return result
    
    def _calculate_position(self, base: Dict, index: int, total: int) -> Dict:
        """计算分散部署的位置"""
        import math
        
        lon = base.get("lon", 0)
        lat = base.get("lat", 0)
        alt = base.get("alt", 0)
        
        if total > 1:
            # 在基准位置周围分散部署，间隔约1km
            angle = 2 * math.pi * index / total
            offset = 0.01  # 约1km
            lon += offset * math.cos(angle)
            lat += offset * math.sin(angle)
        
        return {"lon": lon, "lat": lat, "alt": alt}
    
    async def _handle_fix_field(
        self,
        args: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator],
    ) -> ToolCallResult:
        """修复字段值"""
        idx = args.get("instance_index")
        port_id = args.get("port_id")
        field_name = args.get("field_name")
        new_value = args.get("new_value")
        
        if idx is None or not port_id or not field_name:
            return ToolCallResult(success=False, error="缺少必要参数")
        
        if idx < 0 or idx >= len(memory.current_instances):
            return ToolCallResult(success=False, error=f"无效的实例索引: {idx}")
        
        instance = memory.current_instances[idx]
        
        for init in instance.get("inits", []):
            if init.get("portId") == port_id:
                try:
                    init_data = json.loads(init.get("initJson", "{}"))
                    init_data[field_name] = new_value
                    init["initJson"] = json.dumps(init_data, ensure_ascii=False)
                    memory.update()
                    return ToolCallResult(
                        success=True,
                        message=f"已更新 {instance['otherName']} 的 {port_id}.{field_name}"
                    )
                except json.JSONDecodeError:
                    return ToolCallResult(success=False, error="JSON解析失败")
        
        return ToolCallResult(success=False, error=f"未找到port: {port_id}")
    
    async def _handle_validate(
        self,
        args: Dict[str, Any],
        memory: ScenarioMemory,
        name_generator: Optional[InstanceNameGenerator],
    ) -> ToolCallResult:
        """校验想定"""
        logger.info("【Tool执行】 validate_scenario: {}个实例", len(memory.current_instances))
        
        if not memory.current_instances:
            logger.warning("【Tool失败】 没有可校验的装备实例")
            return ToolCallResult(success=False, error="没有可校验的装备实例")
        
        result = await self.api_client.validate_scenario(memory.current_instances)
        
        if result["success"]:
            data = result["data"]
            valid = data.get("valid", False)
            errors = data.get("errors", [])
            
            if not valid:
                memory.set_validation_errors(errors)
                logger.warning("【Tool结果】 校验失败: {}个错误", len(errors))
                for err in errors[:3]:  # 只记录前3个错误
                    logger.warning("  - {}", err.get("message", err))
            else:
                memory.clear_validation_errors()
                logger.info("【Tool结果】 校验通过")
            
            return ToolCallResult(
                success=True,
                data=data,
                message="校验通过" if valid else f"校验失败: {len(errors)} 个错误"
            )
        logger.error("【Tool失败】 validate_scenario: {}", result["error"])
        return ToolCallResult(success=False, error=result["error"])
