"""对话缓冲区压缩 — Memory Pack 生成与软边界检测"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Optional

from memory_agent.config import settings
from memory_agent.log import get_logger
from memory_agent.providers.base import EmbeddingProvider, LLMProvider
from memory_agent.store.base import MemoryStore
from memory_agent.types import MemoryPack

log = get_logger("memory.packer")

# 话题收束 / 切换信号词
_BREAK_SIGNALS: list[str] = [
    "好的", "明白了", "知道了", "懂了", "了解了",
    "换个话题", "另外", "顺便", "对了", "说到这",
    "先这样", "好了", "行了", "没问题", "可以了",
    "那行", "那好", "清楚了", "就这样",
]

# 停用词（关键词漂移检测用）
_STOPWORDS = frozenset(
    "的了是在我你他她它这那有不也就都和与或但用来去到说"
    "会能要把被让给比从对为着过很已还可以"
)


class MemoryPacker:
    """管理对话缓冲区的压缩"""

    def __init__(
        self, store: MemoryStore, llm: LLMProvider, embedder: EmbeddingProvider,
    ):
        self._store = store
        self._llm = llm
        self._embedder = embedder

    def maybe_compress(
        self, user_id: str, history: list[dict],
    ) -> list[dict]:
        """
        检查并执行压缩。由 ChatHandler 后台线程调用。
        返回（可能被裁剪的）history。
        跳过已标记 packed=True 的条目（如导入的历史记录）。
        """
        # 只统计未打包的条目
        unpacked = [e for e in history if not e.get("packed")]
        turn_count = len(unpacked) // 2
        if turn_count < settings.pack_trigger_turns:
            return history

        # 在未打包的条目中寻找断点
        break_idx = _find_break_point(unpacked)
        log.info("触发压缩: turn_count=%d, break_entry=%d", turn_count, break_idx)

        segment = unpacked[:break_idx]
        if not segment:
            return history

        # 记录时间范围用于回填 pack_id
        start_ts = _entry_ts_iso(segment[0])
        end_ts = _entry_ts_iso(segment[-1])

        pack = self._compress_segment(user_id, segment)
        if pack is None:
            log.warning("压缩失败，跳过")
            return history

        self._store.insert_pack(pack)
        log.info("Pack 已存储: id=%s, turns=%d, topic=%s", pack.id, pack.turn_count, pack.topic)

        # 回填关联
        if start_ts and end_ts:
            linked = self._store.link_memories_to_pack(user_id, start_ts, end_ts, pack.id)
            if linked:
                log.info("回填 pack_id 到 %d 条记忆", linked)

        # 将已压缩的条目标记为 packed，而非直接裁剪
        segment_set = set(id(e) for e in segment)
        for entry in history:
            if id(entry) in segment_set:
                entry["packed"] = True
        log.info("已标记 %d 条目为 packed", len(segment))

        return history

    def _compress_segment(
        self, user_id: str, segment: list[dict],
    ) -> Optional[MemoryPack]:
        """调用 LLM cheap() 将对话片段压缩为 MemoryPack"""
        if not segment:
            return None

        # 拼接对话文本
        conv_lines = []
        for entry in segment:
            role = "用户" if entry.get("role") == "user" else "助手"
            content = entry.get("content", "")[:300]
            conv_lines.append(f"{role}: {content}")
        conv_text = "\n".join(conv_lines)

        # 链式上下文
        latest_pack = self._store.get_latest_pack(user_id)
        prev_pack_id = latest_pack.id if latest_pack else None
        prev_context = ""
        if latest_pack and latest_pack.summary:
            prev_context = latest_pack.summary[-settings.pack_prev_context_chars:]

        prev_section = f"\n上一段对话结尾（参考）：{prev_context}" if prev_context else ""

        prompt = f"""将以下对话压缩为结构化记忆包。
{prev_section}

对话内容：
{conv_text}

严格按照以下 JSON 格式输出（不要 markdown 代码块）：
{{"summary": "对话摘要，不超过300字，用第三人称描述", "topic": "一句话描述本段主题", "keywords": ["关键词1", "关键词2", "关键词3"]}}

只输出 JSON，不要其他内容："""

        raw = self._llm.cheap(prompt)
        parsed = _parse_json(raw)
        if not parsed:
            log.warning("Pack JSON 解析失败: %s", raw[:100])
            return None

        summary = parsed.get("summary", "")[:settings.pack_summary_max_chars]
        topic = parsed.get("topic", "")[:100]
        keywords = parsed.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k) for k in keywords[:20]]

        if not summary:
            return None

        embedding = self._embedder.embed(summary)

        return MemoryPack(
            id=str(uuid.uuid4()),
            user_id=user_id,
            summary=summary,
            keywords=keywords,
            topic=topic,
            embedding=embedding,
            prev_pack_id=prev_pack_id,
            prev_context=prev_context,
            turn_count=len(segment) // 2,
        )


# ── 软边界检测 ─────────────────────────────────────────────

def _find_break_point(history: list[dict]) -> int:
    """
    扫描 history[trigger*2 : max*2]，寻找最优断点。
    返回条目索引（entry index），落在 user 消息位置。
    """
    trigger_entry = settings.pack_trigger_turns * 2
    max_entry = min(settings.pack_max_turns * 2, len(history))

    best_idx = max_entry  # 默认强制切
    best_score = -1.0

    for i in range(trigger_entry, max_entry, 2):
        if i >= len(history):
            break
        if history[i].get("role") != "user":
            continue

        score = 0.0

        # 信号 1: 时间间隔
        if i >= 2:
            prev_ts = history[i - 1].get("ts", 0.0)
            curr_ts = history[i].get("ts", 0.0)
            if curr_ts and prev_ts:
                gap_minutes = (curr_ts - prev_ts) / 60.0
                if gap_minutes > settings.pack_time_gap_minutes:
                    score += 3.0

        # 信号 2: 结束语 / 切换信号词
        user_content = history[i].get("content", "")
        for signal in _BREAK_SIGNALS:
            if signal in user_content:
                score += 2.0
                break

        # 信号 3: 关键词漂移
        window_size = 10  # 5 轮 × 2 条目
        window_a = history[max(0, i - window_size):i]
        window_b = history[i:min(len(history), i + window_size)]
        drift = _keyword_drift(window_a, window_b)
        if drift > 0.5:
            score += 1.0

        if score > best_score:
            best_score = score
            best_idx = i

    # 无明显断点信号时强制切
    if best_score <= 0:
        return min(max_entry, len(history))

    return best_idx


def _keyword_drift(window_a: list[dict], window_b: list[dict]) -> float:
    """计算两个对话窗口的关键词漂移率（Jaccard 距离）"""
    words_a = _extract_words(window_a)
    words_b = _extract_words(window_b)
    union = words_a | words_b
    if not union:
        return 0.0
    intersection = words_a & words_b
    return 1.0 - len(intersection) / len(union)


def _extract_words(entries: list[dict]) -> set[str]:
    """从对话条目中提取关键词（去停用词）"""
    text = " ".join(e.get("content", "") for e in entries)
    tokens = re.findall(r'[\w\u4e00-\u9fff]{2,}', text)
    return {t for t in tokens if t not in _STOPWORDS}


def _entry_ts_iso(entry: dict) -> str:
    """把条目中的 ts（Unix float）转为 ISO 字符串"""
    ts = entry.get("ts", 0.0)
    if ts:
        return datetime.fromtimestamp(ts).isoformat()
    return ""


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
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
