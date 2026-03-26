"""推理取消机制

提供协作式取消令牌，支持：
- Transformers: 通过 StoppingCriteria 在每个 decode step 检查取消状态
- vLLM: 通过 abort(request_id) 取消请求
- 上层: 客户端断开 / 超时 / 手动取消均可触发
"""

import threading
from typing import Optional

from loguru import logger


class CancelToken:
    """协作式取消令牌

    线程安全，可在任意线程中设置取消状态，
    推理线程通过 is_cancelled 轮询检查。

    Attributes:
        request_id: 关联的请求标识，用于日志追踪
    """

    __slots__ = ("_event", "request_id")

    def __init__(self, request_id: Optional[str] = None) -> None:
        self._event = threading.Event()
        self.request_id = request_id

    @property
    def is_cancelled(self) -> bool:
        """检查是否已取消（无阻塞）"""
        return self._event.is_set()

    def cancel(self) -> None:
        """发出取消信号"""
        if not self._event.is_set():
            self._event.set()
            if self.request_id:
                logger.debug("推理请求已取消: {}", self.request_id)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待取消信号，返回 True 表示已取消"""
        return self._event.wait(timeout=timeout)


def make_stopping_criteria(cancel_token: CancelToken):
    """为 Transformers model.generate() 创建取消感知的 StoppingCriteria

    每个 decode step 检查 cancel_token，若已取消则中止生成。
    延迟导入 transformers 以避免未安装时的 ImportError。

    Args:
        cancel_token: 取消令牌

    Returns:
        StoppingCriteriaList 实例
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _CancelCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return cancel_token.is_cancelled

    return StoppingCriteriaList([_CancelCriteria()])
