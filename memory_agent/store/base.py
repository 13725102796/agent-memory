"""存储抽象接口 — 参考 OpenClaw MemorySearchManager 接口"""
from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Optional

from memory_agent.types import MemoryPack, MemoryRecord, MemoryStats


class MemoryStore(ABC):
    """记忆存储接口（可替换 SQLite / PostgreSQL / ...）"""

    # ── Core Memory ───────────────────────────────────────

    @abstractmethod
    def get_core_memory(self, user_id: str) -> str:
        ...

    @abstractmethod
    def set_core_memory(self, user_id: str, content: str) -> None:
        ...

    # ── Memories CRUD ─────────────────────────────────────

    @abstractmethod
    def insert_memory(self, record: MemoryRecord) -> str:
        ...

    @abstractmethod
    def update_memory(
        self, memory_id: str, content: str, embedding_bytes: bytes, importance: float
    ) -> None:
        ...

    @abstractmethod
    def delete_memory(self, memory_id: str) -> None:
        ...

    # ── 查询 ──────────────────────────────────────────────

    @abstractmethod
    def get_memories_by_tier(self, user_id: str, tier: str) -> list[MemoryRecord]:
        ...

    @abstractmethod
    def get_all_memories(self, user_id: str) -> list[MemoryRecord]:
        ...

    # ── 命中 / 生命周期 ──────────────────────────────────

    @abstractmethod
    def record_hit(self, memory_id: str) -> None:
        ...

    @abstractmethod
    def downgrade_stale(self, user_id: str) -> int:
        ...

    @abstractmethod
    def cleanup_old(self, user_id: str) -> int:
        ...

    # ── 统计 ──────────────────────────────────────────────

    @abstractmethod
    def get_stats(self, user_id: str) -> MemoryStats:
        ...

    # ── Memory Packs ─────────────────────────────────────

    @abstractmethod
    def insert_pack(self, pack: MemoryPack) -> str:
        ...

    @abstractmethod
    def get_pack_by_id(self, pack_id: str) -> Optional[MemoryPack]:
        ...

    @abstractmethod
    def get_packs(self, user_id: str) -> list[MemoryPack]:
        ...

    @abstractmethod
    def get_latest_pack(self, user_id: str) -> Optional[MemoryPack]:
        ...

    @abstractmethod
    def delete_packs(self, user_id: str) -> int:
        ...

    # ── 记忆 ↔ Pack 关联 ─────────────────────────────────

    @abstractmethod
    def link_memories_to_pack(
        self, user_id: str, start_ts: str, end_ts: str, pack_id: str,
    ) -> int:
        ...

    # ── FTS 全文索引 ─────────────────────────────────────

    @abstractmethod
    def fts_sync(self, memory_id: str, content: str) -> None:
        ...

    @abstractmethod
    def fts_delete(self, memory_id: str) -> None:
        ...

    @abstractmethod
    def fts_search(self, query: str, user_id: str, limit: int = 20) -> list[tuple[str, float]]:
        ...

    # ── 对话消息持久化 ─────────────────────────────────────

    @abstractmethod
    def append_message(self, user_id: str, role: str, content: str, ts: float) -> None:
        ...

    @abstractmethod
    def get_recent_messages(self, user_id: str, limit: int = 80) -> list[dict]:
        ...

    @abstractmethod
    def mark_messages_packed(self, user_id: str, before_ts: float) -> int:
        ...

    # ── 初始化 ────────────────────────────────────────────

    @abstractmethod
    def init(self) -> None:
        ...
