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
    search_top_k: int = 20

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

    # 新鲜度感知
    freshness_warning_days: int = 7       # 超过此天数的记忆附加过期警告
    time_decay_halflife_days: int = 90    # 时间衰减半衰期（天）
    time_decay_floor: float = 0.5         # 衰减下限（不会低于此值）

    # Prompt Token 预算（字符数，约 2 字符 ≈ 1 token）
    prompt_max_core_chars: int = 2000
    prompt_max_recalled_chars: int = 3000
    prompt_max_pack_chars: int = 1000
    prompt_max_history_turns: int = 20

    # 混合检索权重
    hybrid_vector_weight: float = 0.7
    hybrid_bm25_weight: float = 0.3

    # Reranker 精排
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_min_score: float = 0.01
    reranker_candidates: int = 20

    # 火山引擎字幕回调
    volcano_enabled: bool = False
    volcano_signature: str = ""              # ServerMessageSignature 鉴权
    volcano_flush_timeout_sec: int = 30      # 无新数据时强制刷新缓冲区
    volcano_default_bot_id: str = ""         # 默认 Bot ID（区分 AI vs 用户）


# 全局单例
settings = Settings()
