"""类型定义 — 参考 OpenClaw types.ts 接口先行模式"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MemoryRecord:
    """一条记忆"""
    id: str
    user_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    tier: str = "active"          # "active" | "inactive"
    importance: float = 0.5
    hit_count: int = 0
    last_hit_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pack_id: Optional[str] = None     # 关联的 MemoryPack ID


@dataclass
class SearchResult:
    """搜索命中"""
    id: str
    content: str
    score: float                      # 混合检索最终得分
    tier: str
    pack_id: Optional[str] = None


@dataclass
class ExtractionResult:
    """LLM 记忆提取结果"""
    core: Optional[str] = None
    memories: list[dict] = field(default_factory=list)


@dataclass
class MemoryPack:
    """压缩后的对话记忆包"""
    id: str
    user_id: str
    summary: str
    keywords: list[str]
    topic: str
    embedding: Optional[np.ndarray] = None
    prev_pack_id: Optional[str] = None
    prev_context: str = ""
    turn_count: int = 0
    created_at: Optional[str] = None


@dataclass
class PackSearchResult:
    """通过关联投票检索到的 MemoryPack"""
    id: str
    summary: str
    topic: str
    keywords: list[str]
    weight: float
    hit_count: int
    avg_score: float


@dataclass
class MemoryStats:
    """记忆统计"""
    core_length: int = 0
    active_count: int = 0
    inactive_count: int = 0
    pack_count: int = 0
