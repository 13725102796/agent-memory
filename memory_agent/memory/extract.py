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
from memory_agent.types import MemoryRecord, MemoryType

log = get_logger("memory.extract")


class MemoryExtractor:
    """从对话中提取并存储记忆"""

    def __init__(
        self, store: MemoryStore, llm: LLMProvider, embedder: EmbeddingProvider,
    ):
        self._store = store
        self._llm = llm
        self._embedder = embedder
        self._in_progress = False  # 互斥门控：防止重叠提取

    def extract_and_save(self, user_id: str, user_msg: str, assistant_reply: str, pack_id: str | None = None):
        # 互斥门控：如果上一次提取仍在执行，跳过本次
        if self._in_progress:
            log.info("提取仍在进行中，跳过本次")
            return
        self._in_progress = True
        try:
            self._do_extract(user_id, user_msg, assistant_reply, pack_id)
        finally:
            self._in_progress = False

    def _do_extract(self, user_id: str, user_msg: str, assistant_reply: str, pack_id: str | None = None):
        # 游标检查：只处理 cursor 之后的新消息
        cursor = self._store.get_extraction_cursor(user_id)
        latest_id = self._store.get_latest_message_id(user_id)
        if latest_id <= cursor:
            log.info("无新消息需要提取 (cursor=%d, latest=%d)", cursor, latest_id)
            return

        log.info("开始提取记忆: 用户消息=%s (pack_id=%s, cursor=%d→%d)",
                 user_msg[:60], pack_id, cursor, latest_id)

        # A. 对话原文 → 直接存为 memory（作为检索索引，不依赖 LLM）
        pair_content = f"用户: {user_msg[:200]} | AI: {assistant_reply[:200]}"
        self._upsert(user_id, pair_content, importance=0.4, pack_id=pack_id)
        prompt = f"""从这段对话中提取可能在未来对话中有用的信息。

输出 JSON（严格 JSON，不要 markdown 代码块）：
{{
  "core": "用户身份/持久偏好信息，无则null",
  "memories": [
    {{
      "name": "短标题（不超过20字）",
      "description": "一行描述，用于快速判断相关性（不超过80字）",
      "content": "完整信息（不超过200字）",
      "type": "user|feedback|project|reference",
      "importance": 0.3到0.9
    }}
  ]
}}

记忆类型说明：
- user: 用户画像（角色、偏好、技术栈、知识背景）
- feedback: 行为指导（用户纠正的做法、已验证的好做法）
- project: 项目上下文（进度、决策、计划、行程）
- reference: 外部资源（文档链接、工具地址、看板位置）

core 提取范围：姓名、昵称、年龄、性别、所在城市、语言偏好、职业、技术栈、长期习惯或规则
memories 提取范围（只要包含具体信息就提取）：
- 计划/行程："下周一去珠海长隆"
- 偏好/喜好："喜欢吃火锅"、"喜欢听周杰伦"
- 具体事实："家里有一只猫叫咪咪"
- 人物关系："女朋友叫小美"
- 项目/工作细节："正在开发一个记忆系统"
- 讨论结论/决策："决定用 SQLite 存储"
- 情绪/健康状态："最近压力很大"、"感冒了"
- 用户纠正/指导："不要 mock 数据库"、"用中文回答"

不提取的情况（返回空数组）：
- 纯问候（"你好"、"在吗"）、无实质内容的闲聊（"哈哈"、"嗯嗯"）
- 可从代码或 git 历史推导的信息（文件路径、函数签名等）
- 临时调试步骤和修复方案

示例1 — 有核心信息+具体记忆：
对话：用户: 我叫小王，下周二要去上海出差  助手: 好的小王，上海最近天气不错
输出：{{"core": "姓名: 小王", "memories": [{{"name": "上海出差", "description": "用户下周二计划去上海出差", "content": "用户下周二要去上海出差", "type": "project", "importance": 0.6}}]}}

示例2 — 行为指导：
对话：用户: 以后不要用英文回答我，我看不懂  助手: 好的，以后我都用中文回答
输出：{{"core": null, "memories": [{{"name": "用中文回答", "description": "用户要求所有回答使用中文", "content": "用户要求不要用英文回答，只用中文。原因：用户表示看不懂英文。", "type": "feedback", "importance": 0.8}}]}}

示例3 — 纯闲聊，不提取：
对话：用户: 哈喽  助手: 你好呀！
输出：{{"core": null, "memories": []}}

现在提取以下对话：
用户: {user_msg}
助手: {assistant_reply[:1000]}

只输出 JSON："""

        raw = self._llm.cheap(prompt)
        log.info("LLM cheap 返回 (%d字符): %s", len(raw), raw[:200])
        parsed = _parse_json(raw)
        if not parsed:
            log.warning("JSON 解析失败，记忆提取中止。原始输出: %s", raw[:300])
            return
        log.info("JSON 解析成功: core=%s, memories=%d条",
                 repr(parsed.get("core"))[:60] if parsed.get("core") else "null",
                 len(parsed.get("memories", [])))

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
            # 后置过滤：长度检查 + importance 范围
            if not content or len(content) < 10 or len(content) > 500:
                continue
            importance = max(0.1, min(1.0, importance))
            # 解析类型
            raw_type = item.get("type", "project")
            try:
                mem_type = MemoryType(raw_type)
            except ValueError:
                mem_type = MemoryType.PROJECT
            name = item.get("name", "")[:50]
            description = item.get("description", "")[:150]
            # feedback 类型结构化：content 中融入 why + how_to_apply
            if mem_type == MemoryType.FEEDBACK:
                why = item.get("why", "")
                how_to_apply = item.get("how_to_apply", "")
                if why:
                    content += f"\n原因：{why}"
                if how_to_apply:
                    content += f"\n应用场景：{how_to_apply}"
            self._upsert(
                user_id, content, importance,
                pack_id=pack_id, memory_type=mem_type,
                name=name, description=description,
            )

        # 提取完成，推进游标
        self._store.set_extraction_cursor(user_id, latest_id)

    def _update_core(self, user_id: str, new_fact: str):
        current = self._store.get_core_memory(user_id)

        if not current:
            self._store.set_core_memory(user_id, new_fact)
            log.debug("Core Memory 创建: %s", new_fact[:50])
            self._cleanup_conflicting_memories(user_id, new_fact)
            return

        # 更新前备份当前版本
        self._store.save_core_history(user_id, current, reason=new_fact[:200])

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
            log.debug("Core Memory 更新 (旧版本已备份)")

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

    def _upsert(
        self, user_id: str, content: str, importance: float,
        pack_id: str | None = None,
        memory_type: MemoryType = MemoryType.PROJECT,
        name: str = "", description: str = "",
    ):
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

            # 同类型记忆用更严格的去重阈值（0.7 vs 0.8）
            if (best_rec and best_sim > 0.7
                    and best_rec.memory_type == memory_type):
                return  # 同类型高相似，跳过

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
            pack_id=pack_id,
            memory_type=memory_type,
            name=name,
            description=description,
        )
        memory_id = self._store.insert_memory(record)
        self._store.fts_sync(memory_id, content)
        log.debug("新增记忆: [%s] %s", memory_type.value, content[:50])


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
