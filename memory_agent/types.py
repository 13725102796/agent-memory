"""类型定义 — 参考 OpenClaw types.ts 接口先行模式"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class MemoryType(str, Enum):
    """记忆类型分类（参考 Claude Code memoryTypes.ts 四类型体系）"""
    USER = "user"           # 用户画像：角色、偏好、技术栈、知识背景
    FEEDBACK = "feedback"   # 行为指导：该做/不该做、已验证的好做法
    PROJECT = "project"     # 项目上下文：进度、决策、截止日期、目标
    REFERENCE = "reference" # 外部资源：文档链接、工具地址、看板位置


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
    memory_type: MemoryType = MemoryType.PROJECT
    name: str = ""                    # 短标题（≤50字符）
    description: str = ""             # 一行描述（≤150字符），用于快速相关性判断


@dataclass
class SearchResult:
    """搜索命中"""
    id: str
    content: str
    score: float                      # 混合检索最终得分
    tier: str
    pack_id: Optional[str] = None
    memory_type: MemoryType = MemoryType.PROJECT
    name: str = ""
    updated_at: Optional[str] = None  # 用于新鲜度判断


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


@dataclass
class SubtitleEntry:
    """火山引擎字幕回调条目"""
    text: str
    userId: str
    sequence: int
    definite: bool
    paragraph: bool
    roundId: int
    language: str = "zh"
