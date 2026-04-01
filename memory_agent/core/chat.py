"""聊天编排 — 依赖注入，参考 OpenClaw attempt.ts 编排模式"""
from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from memory_agent.log import get_logger
from memory_agent.memory.extract import MemoryExtractor
from memory_agent.memory.index import MemoryIndex
from memory_agent.memory.packer import MemoryPacker
from memory_agent.memory.search import MemorySearcher
from memory_agent.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider
from memory_agent.store.base import MemoryStore
from memory_agent.types import PackSearchResult, SearchResult

log = get_logger("core.chat")


class ChatHandler:
    """聊天主流程编排，通过构造函数注入所有依赖"""

    def __init__(
        self,
        store: MemoryStore,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        reranker: RerankerProvider | None = None,
    ):
        self._store = store
        self._llm = llm
        self._searcher = MemorySearcher(store, embedder, reranker=reranker)
        self._extractor = MemoryExtractor(store, llm, embedder)
        self._packer = MemoryPacker(store, llm, embedder)
        self._index = MemoryIndex(store)
        self._history: list[dict] = []
        # 单线程后台执行器：串行化打包+提取，避免并发竞争
        self._bg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem-bg")

    def load_history(self, user_id: str) -> None:
        """从数据库加载持久化的对话历史（服务启动时调用）"""
        self._history = self._store.get_recent_messages(user_id)
        if self._history:
            log.info("从数据库恢复 %d 条对话历史", len(self._history))

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

        # ②½ 获取记忆摘要索引
        memory_index = self._index.build(user_id)

        # ③ 拼 Prompt（含对话历史）
        system_prompt = _build_prompt(core, recalled, packs, memory_index)
        full_message = _build_message_with_history(self._history, message)

        # ④ 调 LLM
        log.info("调用 Claude ...")
        reply = self._llm.chat(system_prompt, full_message)

        # ⑤ 记录对话历史（带时间戳）+ 持久化到 DB
        _ts = time.time()
        self._history.append({"role": "user", "content": message, "ts": _ts})
        self._history.append({"role": "assistant", "content": reply, "ts": _ts})
        self._store.append_message(user_id, "user", message, _ts)
        self._store.append_message(user_id, "assistant", reply, _ts)

        # ⑥⑦ 后台异步：压缩打包 + 提取记忆（均不阻塞回复）
        # 传递 history 快照，避免后台线程与主线程竞争 self._history
        history_snapshot = copy.deepcopy(self._history)
        self._bg_executor.submit(
            self._bg_post_process, user_id, message, reply, history_snapshot,
        )

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

        # ②½ 获取记忆摘要索引
        memory_index = self._index.build(user_id)

        # ③ 拼 Prompt
        system_prompt = _build_prompt(core, recalled, packs, memory_index)
        full_message = _build_message_with_history(self._history, message)

        # ④ 流式调 LLM
        log.info("调用 Claude (streaming) ...")
        chunks = []
        for chunk in self._llm.chat_stream(system_prompt, full_message):
            chunks.append(chunk)
            yield chunk

        reply = "".join(chunks)

        # ⑤ 记录对话历史 + 持久化到 DB
        _ts = time.time()
        self._history.append({"role": "user", "content": message, "ts": _ts})
        self._history.append({"role": "assistant", "content": reply, "ts": _ts})
        self._store.append_message(user_id, "user", message, _ts)
        self._store.append_message(user_id, "assistant", reply, _ts)

        # ⑥⑦ 后台异步
        history_snapshot = copy.deepcopy(self._history)
        self._bg_executor.submit(
            self._bg_post_process, user_id, message, reply, history_snapshot,
        )

    def _bg_post_process(
        self, user_id: str, message: str, reply: str,
        history_snapshot: list[dict],
    ) -> None:
        """后台线程：先压缩打包（获取 pack_id），再提取记忆（关联 pack_id）。
        操作 history_snapshot（深拷贝），不直接读写 self._history。
        打包完成后仅回写 packed 标记到主线程 history。
        """
        log.info("后台处理开始: %s", message[:40])
        pack_id = None
        try:
            pack_id, history_snapshot = self._packer.maybe_compress(
                user_id, history_snapshot,
            )
            # 将 packed 标记同步回主线程的 history（按 ts 匹配）
            if pack_id:
                packed_ts = {
                    e["ts"] for e in history_snapshot if e.get("packed")
                }
                for entry in self._history:
                    if entry.get("ts") in packed_ts:
                        entry["packed"] = True
        except Exception:
            log.exception("后台压缩打包异常")
        try:
            self._extractor.extract_and_save(user_id, message, reply, pack_id=pack_id)
            log.info("记忆提取完成")
        except Exception:
            log.exception("后台记忆提取异常")


def _build_message_with_history(
    history: list[dict], current_message: str,
) -> str:
    """把对话历史 + 当前消息拼成完整的用户输入（仅含未打包的近期对话）。
    通过滑动窗口限制最多保留 prompt_max_history_turns 轮对话，防止 token 溢出。
    """
    from memory_agent.config import settings

    # 过滤掉已打包的历史（已保存为 Pack，不需要重复发给 LLM）
    recent = [t for t in history if not t.get("packed")]
    if not recent:
        return current_message

    # 滑动窗口：只保留最近 N 轮（每轮 2 条消息）
    max_entries = settings.prompt_max_history_turns * 2
    if len(recent) > max_entries:
        recent = recent[-max_entries:]

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
    memory_index: str = "",
) -> str:
    from memory_agent.config import settings

    parts = ["你是一个有记忆能力的 AI 助手。"]

    # 记忆摘要索引（轻量全局视图，始终注入）
    if memory_index:
        parts.append("")
        parts.append("<memory_index>")
        parts.append("已知记忆概览（轻量索引，详细内容见下方检索结果）：")
        parts.append(memory_index)
        parts.append("</memory_index>")

    # Core Memory（优先级最高，截断到预算上限）
    if core_memory:
        truncated_core = core_memory[:settings.prompt_max_core_chars]
        parts.append("")
        parts.append("<core_memory>")
        parts.append("用户核心信息（始终参考）：")
        parts.append(truncated_core)
        parts.append("</core_memory>")

    # Memory Packs（截断到预算上限）
    if packs:
        parts.append("")
        parts.append("<memory_packs>")
        parts.append("相关历史对话摘要（按时间参考）：")
        pack_chars = 0
        for p in packs:
            line = f"[{p.topic}] {p.summary}"
            if pack_chars + len(line) > settings.prompt_max_pack_chars:
                break
            parts.append(line)
            pack_chars += len(line)
        parts.append("</memory_packs>")

    # Recalled Memories（截断到预算上限 + 新鲜度标记）
    # feedback 类型优先展示（行为指导不被截断）
    if recalled:
        from memory_agent.memory.freshness import freshness_warning, memory_age_text
        from memory_agent.types import MemoryType as _MT

        _type_priority = {_MT.FEEDBACK: 0, _MT.USER: 1, _MT.PROJECT: 2, _MT.REFERENCE: 3}
        recalled = sorted(recalled, key=lambda m: _type_priority.get(m.memory_type, 9))

        parts.append("")
        parts.append("<recalled_memory>")
        parts.append("搜索到的相关记忆（按需参考）：")
        recall_chars = 0
        for m in recalled:
            warning = freshness_warning(m.updated_at)
            age_tag = f" ({memory_age_text(m.updated_at)})" if m.updated_at else ""
            type_tag = f"[{m.memory_type.value}]" if hasattr(m, 'memory_type') else ""
            line = f"- {type_tag}{age_tag} {m.content}"
            if warning:
                line += f"\n  {warning}"
            if recall_chars + len(line) > settings.prompt_max_recalled_chars:
                break
            parts.append(line)
            recall_chars += len(line)
        parts.append("</recalled_memory>")

    parts.append("")
    parts.append('请基于以上信息自然地回答用户问题。重要规则：')
    parts.append('- 不要说"根据我的记忆"、"我记得"、"根据记录"等暴露记忆系统的措辞')
    parts.append('- 像朋友一样自然地聊天，把记忆信息融入对话中，而不是刻意强调你在"回忆"')
    parts.append('- 如果记忆信息与用户当前请求矛盾，以用户当前请求为准')

    return "\n".join(parts)
