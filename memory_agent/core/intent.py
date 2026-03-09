"""意图判断 — 判断用户消息是否需要搜索记忆"""
from __future__ import annotations

MEMORY_KEYWORDS = [
    "上次", "之前", "我们讨论", "你记得", "我说过",
    "按我的", "老规矩", "习惯", "我喜欢", "我不喜欢",
    "以前", "一直", "还是用", "照旧", "我的偏好",
    "我们约定", "按照惯例", "我记得", "你还记得",
    "我们定的", "按我的风格", "用我喜欢的",
]


def check_need_memory(message: str) -> bool:
    """
    关键词 + 长度策略：
    - 包含记忆关键词 → True
    - < 10 字 → False（闲聊）
    - > 30 字 → True（保守策略）
    """
    for kw in MEMORY_KEYWORDS:
        if kw in message:
            return True

    if len(message) < 10:
        return False

    if len(message) > 30:
        return True

    return False
