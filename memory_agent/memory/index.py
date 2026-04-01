"""记忆摘要索引 — 类 Claude Code MEMORY.md，提供轻量级全局视图"""
from __future__ import annotations

import time
from collections import defaultdict

from memory_agent.log import get_logger
from memory_agent.store.base import MemoryStore
from memory_agent.types import MemoryType

log = get_logger("memory.index")

# 索引上限
_MAX_ENTRIES = 200
_MAX_CHARS = 5000

# 类型显示顺序（feedback 优先，参考 Claude Code 的行为指导优先原则）
_TYPE_ORDER = [
    MemoryType.FEEDBACK,
    MemoryType.USER,
    MemoryType.PROJECT,
    MemoryType.REFERENCE,
]

_TYPE_LABELS = {
    MemoryType.FEEDBACK: "行为指导",
    MemoryType.USER: "用户画像",
    MemoryType.PROJECT: "项目上下文",
    MemoryType.REFERENCE: "外部资源",
}


class MemoryIndex:
    """带缓存的记忆摘要索引生成器"""

    def __init__(self, store: MemoryStore):
        self._store = store
        self._cache: dict[str, tuple[str, float]] = {}  # user_id -> (index_str, timestamp)
        self._cache_ttl = 60.0  # 缓存有效期（秒）

    def invalidate(self, user_id: str) -> None:
        """记忆变更时调用，使缓存失效"""
        self._cache.pop(user_id, None)

    def build(self, user_id: str) -> str:
        """生成记忆摘要索引字符串。
        按 type 分组，每条记忆一行：[type] name — description
        上限 200 条 / 5000 字符。
        """
        # 检查缓存
        if user_id in self._cache:
            cached_str, cached_at = self._cache[user_id]
            if time.time() - cached_at < self._cache_ttl:
                return cached_str

        all_memories = self._store.get_all_memories(user_id)
        if not all_memories:
            return ""

        # 按 importance 排序，取前 _MAX_ENTRIES 条
        all_memories.sort(key=lambda m: m.importance, reverse=True)
        all_memories = all_memories[:_MAX_ENTRIES]

        # 按 type 分组
        grouped: dict[MemoryType, list] = defaultdict(list)
        for m in all_memories:
            grouped[m.memory_type].append(m)

        lines: list[str] = []
        total_chars = 0

        for mem_type in _TYPE_ORDER:
            memories = grouped.get(mem_type)
            if not memories:
                continue

            label = _TYPE_LABELS.get(mem_type, mem_type.value)
            header = f"\n## {label}"
            lines.append(header)
            total_chars += len(header)

            for m in memories:
                name_part = m.name or m.content[:20]
                desc_part = m.description or m.content[:80]
                line = f"- {name_part} — {desc_part}"

                if total_chars + len(line) > _MAX_CHARS:
                    lines.append(f"  ... (已截断，共 {len(all_memories)} 条记忆)")
                    break
                lines.append(line)
                total_chars += len(line)

        result = "\n".join(lines)

        # 更新缓存
        self._cache[user_id] = (result, time.time())

        return result
