"""从对话提取记忆 + 去重分流"""
from __future__ import annotations

import json
import re
import uuid

import numpy as np

from memory_agent.config import settings
from memory_agent.log import get_logger
from memory_agent.providers.base import EmbeddingProvider, LLMProvider
from memory_agent.store.base import MemoryStore
from memory_agent.types import MemoryRecord

log = get_logger("memory.extract")


class MemoryExtractor:
    """从对话中提取并存储记忆"""

    def __init__(
        self, store: MemoryStore, llm: LLMProvider, embedder: EmbeddingProvider,
    ):
        self._store = store
        self._llm = llm
        self._embedder = embedder

    def extract_and_save(self, user_id: str, user_msg: str, assistant_reply: str):
        prompt = f"""从这段对话中提取值得长期记住的信息。

分两类输出 JSON（严格 JSON 格式，不要 markdown 代码块）：
{{
  "core": "持久性偏好/身份信息，如有则填写，无则为null",
  "memories": [
    {{"content": "具体事实/决策/方案", "importance": 0.5}}
  ]
}}

分类标准：
- core: 姓名、语言偏好、技术栈、代码风格、长期规则（如"以后注释用英文"）
- memories: 具体方案、决策、讨论结论、项目细节
- 闲聊/问候/临时指令 → 不提取，memories 为空数组，core 为 null

对话：
用户: {user_msg}
助手: {assistant_reply[:1000]}

只输出 JSON，不要其他内容："""

        raw = self._llm.cheap(prompt)
        parsed = _parse_json(raw)
        if not parsed:
            return

        # Core Memory
        core_fact = parsed.get("core")
        if core_fact and isinstance(core_fact, str) and core_fact.strip():
            self._update_core(user_id, core_fact.strip())

        # Active Memory
        memories_list = parsed.get("memories", [])
        if not isinstance(memories_list, list):
            return

        for item in memories_list:
            if not isinstance(item, dict):
                continue
            content = item.get("content", "").strip()
            importance = float(item.get("importance", 0.5))
            if content:
                self._upsert(user_id, content, importance)

    def _update_core(self, user_id: str, new_fact: str):
        current = self._store.get_core_memory(user_id)

        if not current:
            self._store.set_core_memory(user_id, new_fact)
            log.debug("Core Memory 创建: %s", new_fact[:50])
            self._cleanup_conflicting_memories(user_id, new_fact)
            return

        prompt = f"""当前用户核心记忆：
{current}

新发现的信息：
{new_fact}

请把新信息合并到核心记忆中。规则：
- 保持简洁，总字数不超过 500 字
- 用 "key: value" 格式，每行一条
- 新信息与旧信息矛盾时，用新的覆盖旧的
- 只保留持久性信息（姓名、偏好、技术栈、风格等）

只输出合并后的核心记忆，不要其他内容："""

        merged = self._llm.cheap(prompt)
        if merged and len(merged) > 5:
            self._store.set_core_memory(user_id, merged[:2000])
            log.debug("Core Memory 更新")

        # Core 更新后清理 Active Memory 中的矛盾条目
        self._cleanup_conflicting_memories(user_id, new_fact)

    def _cleanup_conflicting_memories(self, user_id: str, core_fact: str):
        """Core Memory 更新后，删除 Active Memory 中与之高度相似但内容矛盾的旧条目"""
        fact_vec = self._embedder.embed(core_fact)
        all_memories = (
            self._store.get_memories_by_tier(user_id, "active")
            + self._store.get_memories_by_tier(user_id, "inactive")
        )
        for rec in all_memories:
            if rec.embedding is None:
                continue
            sim = float(np.dot(fact_vec, rec.embedding))
            if sim > settings.conflict_threshold:
                self._store.delete_memory(rec.id)
                self._store.fts_delete(rec.id)
                log.debug("清理矛盾记忆: %s (sim=%.2f)", rec.content[:50], sim)

    def _upsert(self, user_id: str, content: str, importance: float):
        new_vec = self._embedder.embed(content)

        # 找相似的已有记忆
        all_memories = (
            self._store.get_memories_by_tier(user_id, "active")
            + self._store.get_memories_by_tier(user_id, "inactive")
        )

        if all_memories:
            best_sim = 0.0
            best_rec = None
            for rec in all_memories:
                if rec.embedding is None:
                    continue
                sim = float(np.dot(new_vec, rec.embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_rec = rec

            if best_rec and best_sim > settings.duplicate_threshold:
                return  # 几乎重复，跳过

            if best_rec and best_sim > settings.conflict_threshold:
                self._store.update_memory(
                    best_rec.id, content, new_vec.tobytes(), importance,
                )
                self._store.fts_sync(best_rec.id, content)
                log.debug("更新记忆: %s", content[:50])
                return

        # 全新记忆
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            embedding=new_vec,
            importance=importance,
        )
        memory_id = self._store.insert_memory(record)
        self._store.fts_sync(memory_id, content)
        log.debug("新增记忆: %s", content[:50])


def _parse_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    log.warning("JSON 解析失败: %s", text[:100])
    return None
