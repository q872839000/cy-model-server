"""
Tool定义

定义想定生成Agent可用的所有Tools及其JSON Schema。
"""
from typing import List, Dict, Any


TOOL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "get_equipment_library": {
        "type": "function",
        "function": {
            "name": "get_equipment_library",
            "description": "获取装备库列表。返回所有可用装备，每个装备包含其支持的阵营列表（如红、蓝、绿、白）。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    "get_equipment_schema": {
        "type": "function",
        "function": {
            "name": "get_equipment_schema",
            "description": "获取装备的参数Schema，了解装备有哪些可配置参数。支持单个或批量获取。",
            "parameters": {
                "type": "object",
                "properties": {
                    "equip_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "装备ID列表"
                    }
                },
                "required": ["equip_ids"]
            }
        }
    },

    "resolve_location": {
        "type": "function",
        "function": {
            "name": "resolve_location",
            "description": "解析地理位置描述，获取坐标信息和边界范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_description": {
                        "type": "string",
                        "description": "位置描述，如'台湾海峡'、'南海'、'东经120度北纬25度附近'"
                    }
                },
                "required": ["location_description"]
            }
        }
    },

    "create_scenario_instance": {
        "type": "function",
        "function": {
            "name": "create_scenario_instance",
            "description": "创建装备实例。必须先调用get_equipment_schema获取参数定义。",
            "parameters": {
                "type": "object",
                "properties": {
                    "equip_id": {
                        "type": "string",
                        "description": "装备ID（来自装备库）"
                    },
                    "classify": {
                        "type": "string",
                        "description": "部署阵营，必须是该装备支持的阵营之一（装备可能支持red/blue/green/white中的一个或多个）"
                    },
                    "count": {
                        "type": "integer",
                        "description": "创建数量，默认1"
                    },
                    "base_position": {
                        "type": "object",
                        "properties": {
                            "lon": {"type": "number"},
                            "lat": {"type": "number"},
                            "alt": {"type": "number"}
                        },
                        "description": "基准位置，多个实例会自动分散部署"
                    },
                    "port_values": {
                        "type": "object",
                        "description": "各端口的参数值，key为portId，value为参数对象"
                    }
                },
                "required": ["equip_id", "classify"]
            }
        }
    },

    "fix_instance_field": {
        "type": "function",
        "function": {
            "name": "fix_instance_field",
            "description": "修复装备实例中的字段值，用于校验失败后的修正。",
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_index": {
                        "type": "integer",
                        "description": "实例索引（从0开始）"
                    },
                    "port_id": {
                        "type": "string",
                        "description": "端口ID"
                    },
                    "field_name": {
                        "type": "string",
                        "description": "字段名"
                    },
                    "new_value": {
                        "description": "新值"
                    }
                },
                "required": ["instance_index", "port_id", "field_name", "new_value"]
            }
        }
    },

    "set_equipment_plan": {
        "type": "function",
        "function": {
            "name": "set_equipment_plan",
            "description": "设置装备部署计划。在PLANNING阶段使用，根据用户需求和装备库信息制定部署方案。",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "equipId": {"type": "string", "description": "装备ID"},
                                "equipName": {"type": "string", "description": "装备名称"},
                                "classify": {"type": "string", "description": "部署阵营(red/blue/green/white)"},
                                "count": {"type": "integer", "description": "部署数量"},
                                "location": {"type": "string", "description": "部署位置描述"}
                            },
                            "required": ["equipId", "classify", "count"]
                        },
                        "description": "装备部署计划列表"
                    }
                },
                "required": ["plan"]
            }
        }
    },

    "validate_scenario": {
        "type": "function",
        "function": {
            "name": "validate_scenario",
            "description": "校验当前生成的想定，检查是否符合规范。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    "modify_instances": {
        "type": "function",
        "function": {
            "name": "modify_instances",
            "description": "批量修改已有装备实例的参数。用于调整速度、高度、航向等属性，无需重新创建实例。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "properties": {
                            "equipId": {"type": "string", "description": "按装备ID筛选"},
                            "classify": {"type": "string", "description": "按阵营筛选(red/blue)"},
                            "all": {"type": "boolean", "description": "修改所有实例"}
                        },
                        "description": "筛选条件，指定要修改哪些实例"
                    },
                    "updates": {
                        "type": "object",
                        "properties": {
                            "port_id": {"type": "string", "description": "要修改的端口ID，如port_movement"},
                            "values": {"type": "object", "description": "要修改的参数值，如{speed: 5, heading: 90}"}
                        },
                        "required": ["port_id", "values"],
                        "description": "修改内容"
                    }
                },
                "required": ["filter", "updates"]
            }
        }
    },

    "delete_instances": {
        "type": "function",
        "function": {
            "name": "delete_instances",
            "description": "删除装备实例。可按条件筛选或删除全部。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "properties": {
                            "equipId": {"type": "string", "description": "按装备ID筛选"},
                            "classify": {"type": "string", "description": "按阵营筛选(red/blue)"},
                            "otherName": {"type": "string", "description": "按实例名称筛选"},
                            "all": {"type": "boolean", "description": "删除所有实例"}
                        },
                        "description": "筛选条件，指定要删除哪些实例"
                    }
                },
                "required": ["filter"]
            }
        }
    },

    "ask_user": {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "当信息不完整或有歧义时，向用户询问澄清问题。用于获取更多细节以便正确执行任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "要问用户的问题"
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选项列表（如果适用），方便用户快速选择"
                    },
                    "context": {
                        "type": "string",
                        "description": "问题的上下文说明"
                    }
                },
                "required": ["question"]
            }
        }
    }
}



def get_all_tools() -> List[Dict[str, Any]]:
    """
    获取所有可用的Tools
    
    Returns:
        Tool定义列表
    """
    return list(TOOL_DEFINITIONS.values())


def get_tool_by_name(name: str) -> Dict[str, Any]:
    """
    根据名称获取Tool定义
    
    Args:
        name: Tool名称
        
    Returns:
        Tool定义，不存在则返回None
    """
    return TOOL_DEFINITIONS.get(name)
