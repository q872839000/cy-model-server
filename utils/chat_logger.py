"""
对话日志记录工具

将每次完整对话记录保存到文件，用于调试和监控。
可通过环境变量 CHAT_LOG_PATH 修改存储路径。

注意：这是生产级辅助工具，非测试代码，属于 utils 层。
"""

import json
import asyncio
import os
import time
from typing import Any, Dict

# 可通过环境变量修改路径
CHAT_LOG_PATH = os.environ.get("CHAT_LOG_PATH", "./last_chat.json")


async def save_chat_log(
    *,
    model: str,
    params: Dict[str, Any],
    messages: Any,
    assistant_content: str,
    usage: Dict[str, Any] | None = None,
) -> None:
    """
    保存一次完整对话记录（覆盖旧文件）
    不会阻塞主事件循环
    """

    log_obj = {
        "timestamp": int(time.time()),
        "model": model,
        "params": params,
        "messages": messages,
        "assistant": {
            "content": assistant_content,
        },
        "usage": usage,
    }

    def _write():
        with open(CHAT_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(log_obj, ensure_ascii=False, indent=2))

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        print(f"[chat_logger] write failed: {e}")
