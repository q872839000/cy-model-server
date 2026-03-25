"""GLM4 系列模型策略"""

from strategies.glm.base import GLMBaseStrategy


class GLM4Strategy(GLMBaseStrategy):
    """
    GLM4 系列策略（不支持深度思考）。

    适用于 GLM-4 基础模型（如 GLM-4-9B-0414）。
    优先使用 tokenizer 原生模板，确保 [gMASK]<sop> 等特殊标记正确处理。
    GLM-4 基础模型不支持 <think> 协议，enable_thinking 参数会被忽略。
    深度思考仅由 GLM-Z1 系列支持，请使用 glm4-z1 策略。
    """

    def prefer_native_template(self) -> bool:
        return True
