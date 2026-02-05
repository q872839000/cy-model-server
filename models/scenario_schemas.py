"""
想定生成相关数据模型

定义想定生成API的请求/响应模型。
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ScenarioRequest(BaseModel):
    """
    想定生成请求（无状态设计）
    
    调用方需要传入历史消息，服务端仅管理轻量级记忆。
    """
    message: str = Field(..., description="当前用户消息")
    messages: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="历史消息列表，格式: [{role: 'user'|'assistant', content: str}]"
    )
    memory_id: Optional[str] = Field(
        default=None,
        description="记忆ID，用于关联中间状态。首次请求可不传，后续请求传入返回的memory_id"
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM模型名称"
    )


class ScenarioResponse(BaseModel):
    """
    想定生成响应
    """
    memory_id: str = Field(..., description="记忆ID，后续请求需要传回")
    phase: str = Field(..., description="当前对话阶段")
    message: str = Field(..., description="AI回复消息")
    scenario: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="生成的想定JSON（仅在完成时返回）"
    )
    errors: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="校验错误列表"
    )
    is_completed: bool = Field(default=False, description="是否完成")
    validation_attempts: int = Field(default=0, description="校验尝试次数")
    
    # Agent模式扩展字段
    needs_input: bool = Field(default=False, description="是否需要用户输入更多信息")
    needs_confirmation: bool = Field(default=False, description="是否需要用户确认执行计划")
    plan: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="执行计划（Agent模式）"
    )
    thought_process: Optional[List[str]] = Field(
        default=None,
        description="思考过程（Agent模式，用于调试）"
    )
    
    debug_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="调试信息（开发模式）"
    )


class EquipmentInfo(BaseModel):
    """装备信息"""
    equipId: str = Field(..., description="装备ID")
    equipName: str = Field(..., description="装备名称")
    classifies: List[str] = Field(default_factory=list, description="支持的阵营列表(red/blue/green/white)")
    description: Optional[str] = Field(default=None, description="描述")


class EquipmentListResponse(BaseModel):
    """装备列表响应"""
    total: int = Field(..., description="总数")
    equipments: List[EquipmentInfo] = Field(default_factory=list, description="装备列表")


class EquipmentPort(BaseModel):
    """装备端口定义"""
    portId: str = Field(..., description="端口ID")
    portName: str = Field(..., description="端口名称")
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema", description="参数Schema")


class EquipmentSchema(BaseModel):
    """装备Schema"""
    equipId: str = Field(..., description="装备ID")
    equipName: str = Field(..., description="装备名称")
    ports: List[EquipmentPort] = Field(default_factory=list, description="端口列表")


class EquipmentInstance(BaseModel):
    """装备实例"""
    equipId: str = Field(..., description="装备ID")
    otherName: str = Field(..., description="实例名称")
    classify: str = Field(..., description="阵营")
    inits: List[Dict[str, Any]] = Field(default_factory=list, description="初始化参数")


class ValidationError(BaseModel):
    """校验错误"""
    instanceIndex: int = Field(..., description="实例索引")
    portId: Optional[str] = Field(default=None, description="端口ID")
    fieldName: Optional[str] = Field(default=None, description="字段名")
    message: str = Field(..., description="错误信息")
    suggestion: Optional[str] = Field(default=None, description="修复建议")


class ValidationResult(BaseModel):
    """校验结果"""
    valid: bool = Field(..., description="是否通过校验")
    errors: List[ValidationError] = Field(default_factory=list, description="错误列表")
    message: Optional[str] = Field(default=None, description="校验消息")
