"""配置 — Pydantic Settings，支持 .env 和环境变量"""
from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Claude CLI
    claude_cli_path: str = "claude"
    claude_model: str = "claude-sonnet-4-6"
    claude_cheap_model: str = "claude-haiku-4-5-20251001"
    claude_timeout: int = 300

    # 数据库
    db_path: str = os.path.join(os.path.dirname(__file__), "..", "memory.db")

    # Embedding
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # 默认用户
    default_user_id: str = "default-user"

    # 记忆搜索（小模型分数偏低，阈值相应调低）
    search_active_limit: int = 5
    search_inactive_limit: int = 3
    search_min_score: float = 0.25
    inactive_score_discount: float = 0.8
    search_top_k: int = 5

    # 去重阈值
    duplicate_threshold: float = 0.8
    conflict_threshold: float = 0.6

    # 生命周期
    downgrade_days: int = 30
    downgrade_importance: float = 0.3
    cleanup_days: int = 90
    cleanup_importance: float = 0.05

    # Memory Pack 压缩
    pack_trigger_turns: int = 20
    pack_max_turns: int = 35
    pack_overlap_turns: int = 5
    pack_time_gap_minutes: int = 3
    pack_summary_max_chars: int = 300
    pack_prev_context_chars: int = 50
    pack_search_limit: int = 2

    # 混合检索权重
    hybrid_vector_weight: float = 0.7
    hybrid_bm25_weight: float = 0.3


# 全局单例
settings = Settings()
