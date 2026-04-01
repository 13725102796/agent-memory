"""记忆新鲜度感知 — 参考 Claude Code memoryAge.ts"""
from __future__ import annotations

import math
from datetime import datetime

from memory_agent.config import settings


def memory_age_days(updated_at: str | None) -> int:
    """计算记忆距今天数（向下取整），无日期时返回 0"""
    if not updated_at:
        return 0
    try:
        dt = datetime.fromisoformat(updated_at)
        delta = datetime.now() - dt
        return max(0, int(delta.total_seconds() / 86400))
    except (ValueError, TypeError):
        return 0


def memory_age_text(updated_at: str | None) -> str:
    """人类可读的时间距离：'今天' / '昨天' / '47天前'"""
    days = memory_age_days(updated_at)
    if days == 0:
        return "今天"
    if days == 1:
        return "昨天"
    return f"{days}天前"


def freshness_warning(updated_at: str | None) -> str | None:
    """超过阈值天数返回警告文本，否则 None。
    参考 Claude Code: 'This memory is N days old... Verify against current code.'
    """
    days = memory_age_days(updated_at)
    if days <= settings.freshness_warning_days:
        return None
    return (
        f"⚠️ 此记忆已 {days} 天未更新，内容可能已过时。"
        "记忆是时间点快照而非实时状态，请结合当前信息验证后再使用。"
    )


def time_decay_factor(updated_at: str | None) -> float:
    """计算时间衰减因子，用于 hybrid score 乘权。
    衰减函数：max(floor, 1.0 - days / (halflife * 2))
    默认半衰期 90 天，下限 0.5。
    """
    days = memory_age_days(updated_at)
    halflife = settings.time_decay_halflife_days
    floor = settings.time_decay_floor
    return max(floor, 1.0 - days / (halflife * 2))
