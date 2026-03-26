"""
对话日志记录工具

将每次完整对话记录保存到文件，用于调试和监控。
可通过环境变量 CHAT_LOG_PATH 修改存储路径。

注意：这是生产级辅助工具，非测试代码，属于 utils 层。
"""

import json
import asyncio
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

# 可通过环境变量修改路径
CHAT_LOG_PATH = os.environ.get("CHAT_LOG_PATH", "./last_chat.json")
CHAT_LOG_DIR = os.environ.get("CHAT_LOG_DIR", "./chat_logs")


def _sanitize_model_name(model: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", model or "default")
    sanitized = sanitized.strip(" .")
    return sanitized or "default"


async def save_chat_log(
    *,
    model: str,
    params: Dict[str, Any],
    messages: Any,
    assistant_content: str,
    usage: Dict[str, Any] | None = None,
    request_payload: Dict[str, Any] | None = None,
    response_payload: Dict[str, Any] | None = None,
    error: Dict[str, Any] | None = None,
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

    if request_payload is not None:
        log_obj["request"] = request_payload
    if response_payload is not None:
        log_obj["response"] = response_payload
    if error is not None:
        log_obj["error"] = error

    def _write():
        serialized = json.dumps(log_obj, ensure_ascii=False, indent=2)

        latest_path = Path(CHAT_LOG_PATH)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(serialized, encoding="utf-8")

        model_path = Path(CHAT_LOG_DIR) / f"{_sanitize_model_name(model)}.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(serialized, encoding="utf-8")

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        print(f"[chat_logger] write failed: {e}")
