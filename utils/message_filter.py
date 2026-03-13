"""
消息过滤工具

用于过滤对话历史中的思考内容，确保历史消息干净简洁。
"""

import re
from typing import List, Dict, Any


def filter_thinking_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    过滤消息列表中的思考内容
    
    支持的思考标签格式：
    - <think>...</think>
    - <think>...（未闭合）
    
    Args:
        messages: 原始消息列表
        
    Returns:
        过滤后的消息列表
    """
    filtered_messages = []
    
    for msg in messages:
        if not isinstance(msg, dict):
            filtered_messages.append(msg)
            continue
            
        # 复制消息，避免修改原对象
        filtered_msg = msg.copy()
        content = msg.get("content", "")
        
        if not isinstance(content, str):
            filtered_messages.append(filtered_msg)
            continue
            
        # 过滤思考内容
        cleaned_content = _remove_thinking_tags(content)
        filtered_msg["content"] = cleaned_content
        
        # 如果过滤后内容为空且无 tool_calls，跳过该消息
        # 保留带 tool_calls 的 assistant 消息（content 可为空）
        if cleaned_content.strip() or filtered_msg.get("tool_calls") or filtered_msg.get("tool_call_id"):
            filtered_messages.append(filtered_msg)
    
    return filtered_messages


def _remove_thinking_tags(content: str) -> str:
    """
    移除文本中的思考标签及其内容
    
    Args:
        content: 原始文本内容
        
    Returns:
        移除思考内容后的文本
    """
    # 模式1: 完整的思考标签对 <think>...</think>
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # 模式2: 未闭合的思考标签 <think>...（到文本末尾）
    content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL)
    
    # 清理多余的空白字符
    content = re.sub(r'\n\s*\n', '\n', content)  # 多个换行合并为一个
    content = content.strip()
    
    return content


def should_filter_message(msg: Dict[str, Any]) -> bool:
    """
    判断消息是否需要过滤
    
    Args:
        msg: 消息对象
        
    Returns:
        是否需要过滤思考内容
    """
    # 只过滤 assistant 角色的消息
    if msg.get("role") != "assistant":
        return False
        
    content = msg.get("content", "")
    if not isinstance(content, str):
        return False
        
    # 检查是否包含思考标签
    return "<think>" in content


# 便捷函数：用于API路由中的快速过滤
def clean_messages_for_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    为历史对话清理消息列表的便捷函数
    
    专用于API路由中过滤历史消息，移除思考内容
    
    Args:
        messages: 原始消息列表
        
    Returns:
        清理后适合作为历史的消息列表
    """
    return filter_thinking_content(messages)
