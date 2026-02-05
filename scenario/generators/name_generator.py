"""
实例名称生成器

为装备实例生成唯一的otherName。
"""
from typing import Dict


class InstanceNameGenerator:
    """
    装备实例名称生成器
    
    生成格式: {装备名}_{阵营}_{序号}
    例如: 无人机_blue_1, 驱逐舰_red_2
    """
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
    
    def generate(self, equip_id: str, classify: str, equip_name: str = None) -> str:
        """
        生成实例名称
        
        Args:
            equip_id: 装备ID
            classify: 阵营 (red/blue)
            equip_name: 装备名称（可选）
            
        Returns:
            生成的名称
        """
        # 使用装备ID和阵营作为key
        key = f"{equip_id}_{classify}"
        
        # 增加计数
        if key not in self._counters:
            self._counters[key] = 0
        self._counters[key] += 1
        
        # 生成名称
        base_name = equip_name or equip_id
        return f"{base_name}_{classify}_{self._counters[key]}"
    
    def reset(self):
        """重置计数器"""
        self._counters.clear()
