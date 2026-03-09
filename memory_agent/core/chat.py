"""聊天编排 — 依赖注入，参考 OpenClaw attempt.ts 编排模式"""
from __future__ import annotations

import threading
import time
from typing import Iterator

from memory_agent.log import get_logger
from memory_agent.memory.extract import MemoryExtractor
from memory_agent.memory.packer import MemoryPacker
from memory_agent.memory.search import MemorySearcher
from memory_agent.providers.base import EmbeddingProvider, LLMProvider
from memory_agent.store.base import MemoryStore
from memory_agent.types import PackSearchResult, SearchResult

log = get_logger("core.chat")


class ChatHandler:
    """聊天主流程编排，通过构造函数注入所有依赖"""

    MAX_HISTORY = 40  # 安全上限，packer 是主要裁剪机制

    def __init__(
        self,
        store: MemoryStore,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
    ):
        self._store = store
        self._llm = llm
        self._searcher = MemorySearcher(store, embedder)
        self._extractor = MemoryExtractor(store, llm, embedder)
        self._packer = MemoryPacker(store, llm, embedder)
        self._history: list[dict] = []

    def handle(self, user_id: str, message: str) -> str:
        """
        处理一条用户消息，返回 AI 回复。

        流程：
        ① 加载 Core Memory
        ② 混合检索（向量 + BM25）+ 关联投票
        ③ 拼 Prompt
        ④ 调 LLM
        ⑤ 记录对话历史（带时间戳）
        ⑥ 可能触发压缩打包
        ⑦ 提取并存储新记忆
        """
        # ① Core Memory
        core = self._store.get_core_memory(user_id)

        # ② 混合检索 + 关联投票
        _t0 = time.time()
        recalled, packs = self._searcher.search(user_id, message)
        log.info("记忆检索耗时: %.2f ms", (time.time() - _t0) * 1000)
        if recalled:
            log.info("召回 %d 条记忆:", len(recalled))
            for r in recalled:
                log.info("  [%s] score=%.2f | %s", r.tier, r.score, r.content[:50])
        else:
            log.info("未找到相关记忆")
        if packs:
            log.info("关联 %d 个 Pack:", len(packs))
            for p in packs:
                log.info("  [%s] weight=%.2f hits=%d", p.topic, p.weight, p.hit_count)

        # ③ 拼 Prompt（含对话历史）
        system_prompt = _build_prompt(core, recalled, packs)
        full_message = _build_message_with_history(self._history, message)

        # ④ 调 LLM
        log.info("调用 Claude ...")
        reply = self._llm.chat(system_prompt, full_message)

        # ⑤ 记录对话历史（带时间戳）
        _ts = time.time()
        self._history.append({"role": "user", "content": message, "ts": _ts})
        self._history.append({"role": "assistant", "content": reply, "ts": _ts})
        if len(self._history) > self.MAX_HISTORY * 2:
            self._history = self._history[-(self.MAX_HISTORY * 2):]

        # ⑥⑦ 后台异步：压缩打包 + 提取记忆（均不阻塞回复）
        threading.Thread(
            target=self._bg_post_process,
            args=(user_id, message, reply),
            daemon=True,
        ).start()

        return reply

    def handle_stream(self, user_id: str, message: str) -> Iterator[str]:
        """流式处理：①②③ 同步完成后，④ 流式返回 LLM 文本片段"""
        # ① Core Memory
        core = self._store.get_core_memory(user_id)

        # ② 混合检索 + 关联投票
        _t0 = time.time()
        recalled, packs = self._searcher.search(user_id, message)
        log.info("记忆检索耗时: %.2f ms", (time.time() - _t0) * 1000)
        if recalled:
            log.info("召回 %d 条记忆:", len(recalled))
            for r in recalled:
                log.info("  [%s] score=%.2f | %s", r.tier, r.score, r.content[:50])
        if packs:
            log.info("关联 %d 个 Pack:", len(packs))
            for p in packs:
                log.info("  [%s] weight=%.2f hits=%d", p.topic, p.weight, p.hit_count)

        # ③ 拼 Prompt
        system_prompt = _build_prompt(core, recalled, packs)
        full_message = _build_message_with_history(self._history, message)

        # ④ 流式调 LLM
        log.info("调用 Claude (streaming) ...")
        chunks = []
        for chunk in self._llm.chat_stream(system_prompt, full_message):
            chunks.append(chunk)
            yield chunk

        reply = "".join(chunks)

        # ⑤ 记录对话历史
        _ts = time.time()
        self._history.append({"role": "user", "content": message, "ts": _ts})
        self._history.append({"role": "assistant", "content": reply, "ts": _ts})
        if len(self._history) > self.MAX_HISTORY * 2:
            self._history = self._history[-(self.MAX_HISTORY * 2):]

        # ⑥⑦ 后台异步
        threading.Thread(
            target=self._bg_post_process,
            args=(user_id, message, reply),
            daemon=True,
        ).start()

    def _bg_post_process(self, user_id: str, message: str, reply: str) -> None:
        """后台线程：压缩打包 + 提取记忆"""
        try:
            self._history = self._packer.maybe_compress(user_id, self._history)
        except Exception:
            log.exception("后台压缩打包异常")
        try:
            self._extractor.extract_and_save(user_id, message, reply)
            log.debug("记忆提取完成")
        except Exception:
            log.exception("后台记忆提取异常")


def _build_message_with_history(
    history: list[dict], current_message: str,
) -> str:
    """把对话历史 + 当前消息拼成完整的用户输入（仅含未打包的近期对话）"""
    # 过滤掉已打包的历史（已保存为 Pack，不需要重复发给 LLM）
    recent = [t for t in history if not t.get("packed")]
    if not recent:
        return current_message

    parts = ["<conversation_history>"]
    for turn in recent:
        role = "用户" if turn["role"] == "user" else "助手"
        parts.append(f"{role}: {turn['content']}")
    parts.append("</conversation_history>")
    parts.append("")
    parts.append(current_message)
    return "\n".join(parts)


def _build_prompt(
    core_memory: str,
    recalled: list[SearchResult],
    packs: list[PackSearchResult],
) -> str:
    parts = ["你是一个有记忆能力的 AI 助手。"]

    if core_memory:
        parts.append("")
        parts.append("<core_memory>")
        parts.append("用户核心信息（始终参考）：")
        parts.append(core_memory)
        parts.append("</core_memory>")

    if packs:
        parts.append("")
        parts.append("<memory_packs>")
        parts.append("相关历史对话摘要（按时间参考）：")
        for p in packs:
            parts.append(f"[{p.topic}] {p.summary}")
        parts.append("</memory_packs>")

    if recalled:
        parts.append("")
        parts.append("<recalled_memory>")
        parts.append("搜索到的相关记忆（按需参考）：")
        for m in recalled:
            parts.append(f"- {m.content}")
        parts.append("</recalled_memory>")

    parts.append("")
    parts.append('请基于以上信息自然地回答用户问题。重要规则：')
    parts.append('- 不要说"根据我的记忆"、"我记得"、"根据记录"等暴露记忆系统的措辞')
    parts.append('- 像朋友一样自然地聊天，把记忆信息融入对话中，而不是刻意强调你在"回忆"')
    parts.append('- 如果记忆信息与用户当前请求矛盾，以用户当前请求为准')

    return "\n".join(parts)
