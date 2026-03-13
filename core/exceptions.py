"""
自定义异常类：定义业务相关的异常类型
"""

from typing import Optional, Dict, Any


class ModelServerException(Exception):
    """模型服务器基础异常类"""
    
    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotFoundError(ModelServerException):
    """模型未找到异常"""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"模型 '{model_name}' 未找到",
            error_code="MODEL_NOT_FOUND",
            details={"model_name": model_name}
        )


class ModelLoadError(ModelServerException):
    """模型加载失败异常"""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"模型 '{model_name}' 加载失败: {reason}",
            error_code="MODEL_LOAD_ERROR",
            details={"model_name": model_name, "reason": reason}
        )


class InferenceError(ModelServerException):
    """推理失败异常"""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"模型 '{model_name}' 推理失败: {reason}",
            error_code="INFERENCE_ERROR",
            details={"model_name": model_name, "reason": reason}
        )


class ConfigurationError(ModelServerException):
    """配置错误异常"""
    
    def __init__(self, config_path: str, reason: str):
        super().__init__(
            message=f"配置文件 '{config_path}' 错误: {reason}",
            error_code="CONFIGURATION_ERROR",
            details={"config_path": config_path, "reason": reason}
        )


class ResourceLimitError(ModelServerException):
    """资源限制异常"""
    
    def __init__(self, resource: str, limit: str):
        super().__init__(
            message=f"资源 '{resource}' 超出限制: {limit}",
            error_code="RESOURCE_LIMIT_ERROR",
            details={"resource": resource, "limit": limit}
        )


class UnsupportedParameterError(ModelServerException):
    """请求中包含当前不支持的参数

    当客户端传入了本服务尚未实现的 OpenAI 参数（如 n>1、response_format 等）时抛出，
    避免"静默忽略"导致客户端行为不符预期。
    """

    def __init__(self, param_name: str, reason: str = ""):
        detail = reason or f"参数 '{param_name}' 当前不支持"
        super().__init__(
            message=detail,
            error_code="UNSUPPORTED_PARAMETER",
            details={"param": param_name, "reason": reason},
        )
