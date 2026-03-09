"""SQLite 存储实现"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from memory_agent.config import settings
from memory_agent.log import get_logger
from memory_agent.store.base import MemoryStore
from memory_agent.types import MemoryPack, MemoryRecord, MemoryStats

log = get_logger("store.sqlite")


class SQLiteMemoryStore(MemoryStore):
    """基于 SQLite 的记忆存储"""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or settings.db_path
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    # ── 初始化 ────────────────────────────────────────────

    def init(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS core_memories (
                user_id    TEXT PRIMARY KEY,
                content    TEXT NOT NULL DEFAULT '',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                content     TEXT NOT NULL,
                embedding   BLOB,
                tier        TEXT NOT NULL DEFAULT 'active',
                importance  REAL NOT NULL DEFAULT 0.5,
                hit_count   INTEGER NOT NULL DEFAULT 0,
                last_hit_at TEXT,
                pack_id     TEXT,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_memories_user_tier
                ON memories (user_id, tier);

            CREATE TABLE IF NOT EXISTS memory_packs (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                summary      TEXT NOT NULL,
                keywords     TEXT NOT NULL DEFAULT '[]',
                topic        TEXT NOT NULL DEFAULT '',
                embedding    BLOB,
                prev_pack_id TEXT,
                prev_context TEXT NOT NULL DEFAULT '',
                turn_count   INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_packs_user
                ON memory_packs (user_id, created_at);
        """)

        # FTS5 虚拟表需要单独创建（不能放在 executescript 中）
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, memory_id UNINDEXED, user_id UNINDEXED,
                tokenize='unicode61'
            )
        """)
        conn.commit()

        # 兼容旧数据库：添加 pack_id 列（已存在则忽略）
        try:
            conn.execute("ALTER TABLE memories ADD COLUMN pack_id TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # 列已存在

        # 同步已有记忆到 FTS（仅首次）
        missing = conn.execute(
            "SELECT m.id, m.content, m.user_id FROM memories m "
            "WHERE m.id NOT IN (SELECT memory_id FROM memories_fts)"
        ).fetchall()
        if missing:
            for row in missing:
                conn.execute(
                    "INSERT INTO memories_fts (content, memory_id, user_id) VALUES (?, ?, ?)",
                    (_cjk_segment(row["content"]), row["id"], row["user_id"]),
                )
            conn.commit()
            log.info("FTS 同步已有记忆: %d 条", len(missing))

        log.info("数据库初始化完成: %s", self._db_path)

    # ── Core Memory ───────────────────────────────────────

    def get_core_memory(self, user_id: str) -> str:
        conn = self._connect()
        row = conn.execute(
            "SELECT content FROM core_memories WHERE user_id = ?", (user_id,)
        ).fetchone()

        return row["content"] if row else ""

    def set_core_memory(self, user_id: str, content: str) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT INTO core_memories (user_id, content, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE
            SET content = excluded.content, updated_at = excluded.updated_at
        """, (user_id, content, _now()))
        conn.commit()

    # ── Memories CRUD ─────────────────────────────────────

    def insert_memory(self, record: MemoryRecord) -> str:
        memory_id = record.id or str(uuid.uuid4())
        embedding_bytes = record.embedding.tobytes() if record.embedding is not None else None
        conn = self._connect()
        conn.execute("""
            INSERT INTO memories (id, user_id, content, embedding, tier, importance, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, record.user_id, record.content, embedding_bytes,
            record.tier, record.importance, _now(), _now(),
        ))
        conn.commit()

        return memory_id

    def update_memory(
        self, memory_id: str, content: str, embedding_bytes: bytes, importance: float
    ) -> None:
        conn = self._connect()
        conn.execute("""
            UPDATE memories
            SET content = ?, embedding = ?, importance = ?, tier = 'active', updated_at = ?
            WHERE id = ?
        """, (content, embedding_bytes, importance, _now(), memory_id))
        conn.commit()

    def delete_memory(self, memory_id: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        conn.commit()

    # ── 查询 ──────────────────────────────────────────────

    def get_memories_by_tier(self, user_id: str, tier: str) -> list[MemoryRecord]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, content, embedding, importance, hit_count, last_hit_at, pack_id, created_at "
            "FROM memories WHERE user_id = ? AND tier = ?",
            (user_id, tier),
        ).fetchall()

        return [_row_to_record(row, user_id, tier) for row in rows]

    def get_all_memories(self, user_id: str) -> list[MemoryRecord]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, content, tier, importance, hit_count, last_hit_at, pack_id, created_at "
            "FROM memories WHERE user_id = ? ORDER BY tier, importance DESC",
            (user_id,),
        ).fetchall()

        return [
            MemoryRecord(
                id=row["id"], user_id=user_id, content=row["content"],
                tier=row["tier"], importance=row["importance"],
                hit_count=row["hit_count"], last_hit_at=row["last_hit_at"],
                pack_id=row["pack_id"], created_at=row["created_at"],
            )
            for row in rows
        ]

    # ── 命中 / 生命周期 ──────────────────────────────────

    def record_hit(self, memory_id: str) -> None:
        conn = self._connect()
        conn.execute("""
            UPDATE memories
            SET hit_count = hit_count + 1,
                last_hit_at = ?,
                importance = MIN(importance + 0.1, 1.0),
                tier = CASE
                    WHEN tier = 'inactive' AND importance + 0.1 > ? THEN 'active'
                    ELSE tier
                END,
                updated_at = ?
            WHERE id = ?
        """, (_now(), settings.downgrade_importance, _now(), memory_id))
        conn.commit()

    def downgrade_stale(self, user_id: str) -> int:
        cutoff = _days_ago(settings.downgrade_days)
        conn = self._connect()
        cursor = conn.execute("""
            UPDATE memories
            SET tier = 'inactive', updated_at = ?
            WHERE user_id = ? AND tier = 'active'
              AND importance < ?
              AND (last_hit_at IS NULL OR last_hit_at < ?)
              AND created_at < ?
        """, (_now(), user_id, settings.downgrade_importance, cutoff, cutoff))
        count = cursor.rowcount
        conn.commit()

        return count

    def cleanup_old(self, user_id: str) -> int:
        cutoff = _days_ago(settings.cleanup_days)
        conn = self._connect()
        # 先获取要删除的记忆 ID 用于清理 FTS
        ids = conn.execute("""
            SELECT id FROM memories
            WHERE user_id = ? AND tier = 'inactive'
              AND importance < ?
              AND last_hit_at IS NULL
              AND created_at < ?
        """, (user_id, settings.cleanup_importance, cutoff)).fetchall()

        if ids:
            id_list = [row["id"] for row in ids]
            placeholders = ",".join("?" * len(id_list))
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", id_list)
            conn.execute(f"DELETE FROM memories_fts WHERE memory_id IN ({placeholders})", id_list)
            conn.commit()

        return len(ids)

    # ── 统计 ──────────────────────────────────────────────

    def get_stats(self, user_id: str) -> MemoryStats:
        conn = self._connect()
        rows = conn.execute(
            "SELECT tier, COUNT(*) as cnt FROM memories WHERE user_id = ? GROUP BY tier",
            (user_id,),
        ).fetchall()
        core = self.get_core_memory(user_id)

        pack_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM memory_packs WHERE user_id = ?", (user_id,)
        ).fetchone()["cnt"]

        stats = MemoryStats(core_length=len(core), pack_count=pack_count)
        for row in rows:
            if row["tier"] == "active":
                stats.active_count = row["cnt"]
            elif row["tier"] == "inactive":
                stats.inactive_count = row["cnt"]
        return stats

    # ── Memory Packs ─────────────────────────────────────

    def insert_pack(self, pack: MemoryPack) -> str:
        pack_id = pack.id or str(uuid.uuid4())
        embedding_bytes = pack.embedding.tobytes() if pack.embedding is not None else None
        conn = self._connect()
        conn.execute("""
            INSERT INTO memory_packs
                (id, user_id, summary, keywords, topic, embedding,
                 prev_pack_id, prev_context, turn_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pack_id, pack.user_id, pack.summary,
            json.dumps(pack.keywords, ensure_ascii=False),
            pack.topic, embedding_bytes,
            pack.prev_pack_id, pack.prev_context,
            pack.turn_count, _now(),
        ))
        conn.commit()
        return pack_id

    def get_pack_by_id(self, pack_id: str) -> Optional[MemoryPack]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM memory_packs WHERE id = ?", (pack_id,)
        ).fetchone()
        return _row_to_pack(row) if row else None

    def get_packs(self, user_id: str) -> list[MemoryPack]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM memory_packs WHERE user_id = ? ORDER BY created_at ASC",
            (user_id,),
        ).fetchall()
        return [_row_to_pack(row) for row in rows]

    def get_latest_pack(self, user_id: str) -> Optional[MemoryPack]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM memory_packs WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return _row_to_pack(row) if row else None

    def delete_packs(self, user_id: str) -> int:
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM memory_packs WHERE user_id = ?", (user_id,)
        )
        count = cursor.rowcount
        conn.commit()
        return count

    # ── 记忆 ↔ Pack 关联 ─────────────────────────────────

    def link_memories_to_pack(
        self, user_id: str, start_ts: str, end_ts: str, pack_id: str,
    ) -> int:
        conn = self._connect()
        cursor = conn.execute("""
            UPDATE memories SET pack_id = ?
            WHERE user_id = ? AND created_at >= ? AND created_at <= ?
              AND pack_id IS NULL
        """, (pack_id, user_id, start_ts, end_ts))
        count = cursor.rowcount
        conn.commit()
        return count

    # ── FTS 全文索引 ─────────────────────────────────────

    def fts_sync(self, memory_id: str, content: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        row = conn.execute("SELECT user_id FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row:
            conn.execute(
                "INSERT INTO memories_fts (content, memory_id, user_id) VALUES (?, ?, ?)",
                (_cjk_segment(content), memory_id, row["user_id"]),
            )
        conn.commit()

    def fts_delete(self, memory_id: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        conn.commit()

    def fts_search(self, query: str, user_id: str, limit: int = 20) -> list[tuple[str, float]]:
        conn = self._connect()
        segmented_query = _cjk_segment(query)
        if not segmented_query.strip():
            return []
        try:
            rows = conn.execute("""
                SELECT memory_id, -bm25(memories_fts) as score
                FROM memories_fts
                WHERE memories_fts MATCH ? AND user_id = ?
                ORDER BY score DESC
                LIMIT ?
            """, (segmented_query, user_id, limit)).fetchall()
            return [(row[0], row[1]) for row in rows]
        except sqlite3.OperationalError:
            return []


# ── 内部辅助 ──────────────────────────────────────────────

def _row_to_record(row: sqlite3.Row, user_id: str, tier: str) -> MemoryRecord:
    emb = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
    return MemoryRecord(
        id=row["id"], user_id=user_id, content=row["content"],
        embedding=emb, tier=tier, importance=row["importance"],
        hit_count=row["hit_count"], last_hit_at=row["last_hit_at"],
        pack_id=row["pack_id"], created_at=row["created_at"],
    )


def _row_to_pack(row: sqlite3.Row) -> MemoryPack:
    emb = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
    keywords = json.loads(row["keywords"]) if row["keywords"] else []
    return MemoryPack(
        id=row["id"], user_id=row["user_id"],
        summary=row["summary"], keywords=keywords, topic=row["topic"],
        embedding=emb, prev_pack_id=row["prev_pack_id"],
        prev_context=row["prev_context"], turn_count=row["turn_count"],
        created_at=row["created_at"],
    )


def _cjk_segment(text: str) -> str:
    """在 CJK 字符间插入空格，使 FTS5 unicode61 能逐字分词"""
    return re.sub(r'([\u4e00-\u9fff])', r' \1 ', text).strip()


def _now() -> str:
    return datetime.now().isoformat()


def _days_ago(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat()
