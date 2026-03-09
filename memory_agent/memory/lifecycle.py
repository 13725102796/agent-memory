"""记忆生命周期管理 — 升降级 + 清理"""
from __future__ import annotations

from memory_agent.log import get_logger
from memory_agent.store.base import MemoryStore

log = get_logger("memory.lifecycle")


class MemoryLifecycle:
    """管理记忆的降级和清理"""

    def __init__(self, store: MemoryStore):
        self._store = store

    def run(self, user_id: str):
        downgraded = self._store.downgrade_stale(user_id)
        cleaned = self._store.cleanup_old(user_id)
        if downgraded > 0:
            log.info("降级 %d 条记忆", downgraded)
        if cleaned > 0:
            log.info("清理 %d 条过期记忆", cleaned)
