"""混合检索（向量 + BM25）+ 关联投票"""
from __future__ import annotations

import time

import numpy as np

from memory_agent.config import settings
from memory_agent.log import get_logger
from memory_agent.providers.base import EmbeddingProvider, RerankerProvider
from memory_agent.store.base import MemoryStore
from memory_agent.types import MemoryRecord, PackSearchResult, SearchResult

log = get_logger("memory.search")


class MemorySearcher:
    """混合检索 + Pack 关联投票"""

    def __init__(self, store: MemoryStore, embedder: EmbeddingProvider, reranker: RerankerProvider | None = None):
        self._store = store
        self._embedder = embedder
        self._reranker = reranker

    def search(
        self, user_id: str, query: str,
    ) -> tuple[list[SearchResult], list[PackSearchResult]]:
        """
        检索流程：
        Stage 1a: 向量搜索 Active + Inactive
        Stage 1b: BM25 全文搜索
        Stage 1c: 归一化 + 加权合并（粗排）
        Stage 1d: Reranker 精排（可选）
        Stage 2:  按 pack_id 聚合投票 → Top-N Pack
        """
        _t = time.time()
        query_vec = self._embedder.embed(query)
        _t_embed = time.time()

        # ── Stage 1a: 向量搜索 ──
        vector_hits = self._vector_search(query_vec, user_id)
        _t_vec = time.time()

        # ── Stage 1b: BM25 搜索 ──
        bm25_hits = self._store.fts_search(query, user_id, limit=20)
        _t_bm25 = time.time()

        # ── Stage 1c: 粗排合并（reranker 模式下多取候选） ──
        merged = self._merge_scores(vector_hits, bm25_hits, user_id)
        _t_merge = time.time()

        # ── Stage 1d: Reranker 精排 ──
        _t_rerank = _t_merge
        if self._reranker and merged:
            merged = self._rerank(query, merged)
            _t_rerank = time.time()

        # 记录命中
        for h in merged:
            self._store.record_hit(h.id)

        if merged:
            log.info("最终命中 %d 条记忆", len(merged))

        # ── Stage 2: 按 pack_id 聚合投票 ──
        packs = self._aggregate_packs(merged)
        _t_pack = time.time()

        rerank_ms = (_t_rerank - _t_merge) * 1000 if self._reranker else 0
        log.info(
            "检索耗时明细: embed=%.1fms | vector=%.1fms | bm25=%.1fms | merge=%.1fms | rerank=%.1fms | pack=%.1fms",
            (_t_embed - _t) * 1000, (_t_vec - _t_embed) * 1000,
            (_t_bm25 - _t_vec) * 1000, (_t_merge - _t_bm25) * 1000,
            rerank_ms, (_t_pack - _t_rerank) * 1000,
        )

        if packs:
            log.info("关联投票命中 %d 个 Pack", len(packs))

        return merged, packs

    def _vector_search(
        self, query_vec: np.ndarray, user_id: str,
    ) -> dict[str, tuple[float, MemoryRecord]]:
        """向量搜索 Active + Inactive，返回 {memory_id: (score, record)}"""
        result: dict[str, tuple[float, MemoryRecord]] = {}

        # Active 层
        for rec in self._store.get_memories_by_tier(user_id, "active"):
            if rec.embedding is None:
                continue
            score = float(np.dot(query_vec, rec.embedding))
            result[rec.id] = (score, rec)

        # Inactive 层（分数打折）
        for rec in self._store.get_memories_by_tier(user_id, "inactive"):
            if rec.embedding is None:
                continue
            score = float(np.dot(query_vec, rec.embedding)) * settings.inactive_score_discount
            result[rec.id] = (score, rec)

        return result

    def _merge_scores(
        self,
        vector_hits: dict[str, tuple[float, MemoryRecord]],
        bm25_hits: list[tuple[str, float]],
        user_id: str,
    ) -> list[SearchResult]:
        """归一化两路分数，加权合并"""
        all_ids = set(vector_hits.keys()) | {mid for mid, _ in bm25_hits}
        if not all_ids:
            return []

        # 归一化
        v_scores = {mid: s for mid, (s, _) in vector_hits.items()}
        v_norm = _min_max_normalize(v_scores)

        b_scores = {mid: s for mid, s in bm25_hits}
        b_norm = _min_max_normalize(b_scores)

        # 加权合并
        results: list[SearchResult] = []
        for mid in all_ids:
            v = v_norm.get(mid, 0.0)
            b = b_norm.get(mid, 0.0)
            final = settings.hybrid_vector_weight * v + settings.hybrid_bm25_weight * b

            if final < settings.search_min_score:
                continue

            # 获取记忆详情
            if mid in vector_hits:
                rec = vector_hits[mid][1]
            else:
                # BM25 独有命中，需要从数据库加载
                rec = self._load_record(mid, user_id)
                if rec is None:
                    continue

            results.append(SearchResult(
                id=mid, content=rec.content, score=final,
                tier=rec.tier, pack_id=rec.pack_id,
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        # reranker 模式：多取候选给精排；否则直接取 top_k
        limit = settings.reranker_candidates if self._reranker else settings.search_top_k
        return results[:limit]

    def _rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Cross-Encoder 精排：用 reranker 重新打分，过滤低分结果"""
        documents = [c.content for c in candidates]
        scores = self._reranker.rerank(query, documents)

        # 记录 rerank 前后分数对比
        for candidate, rerank_score in zip(candidates, scores):
            log.info("  rerank: 粗排=%.2f → 精排=%.4f | %s",
                     candidate.score, rerank_score, candidate.content[:60])

        # 用 reranker 分数替换粗排分数
        reranked: list[SearchResult] = []
        for candidate, rerank_score in zip(candidates, scores):
            if rerank_score >= settings.reranker_min_score:
                reranked.append(SearchResult(
                    id=candidate.id,
                    content=candidate.content,
                    score=rerank_score,
                    tier=candidate.tier,
                    pack_id=candidate.pack_id,
                ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        reranked = reranked[:settings.search_top_k]

        if not reranked:
            log.info("  rerank: 所有候选分数低于阈值 %.2f，无结果", settings.reranker_min_score)

        return reranked

    def _load_record(self, memory_id: str, user_id: str) -> MemoryRecord | None:
        """加载 BM25 独有命中的记忆（向量搜索未返回的）"""
        for tier in ("active", "inactive"):
            for rec in self._store.get_memories_by_tier(user_id, tier):
                if rec.id == memory_id:
                    return rec
        return None

    def _aggregate_packs(
        self, hits: list[SearchResult],
    ) -> list[PackSearchResult]:
        """按 pack_id 聚合投票，返回权重最高的 Pack"""
        pack_votes: dict[str, list[float]] = {}
        for hit in hits:
            if hit.pack_id:
                pack_votes.setdefault(hit.pack_id, []).append(hit.score)

        if not pack_votes:
            return []

        # 计算权重：hit_count × avg_score
        pack_ranking: list[tuple[str, float, int, float]] = []
        for pack_id, scores in pack_votes.items():
            hit_count = len(scores)
            avg_score = sum(scores) / hit_count
            weight = hit_count * avg_score
            pack_ranking.append((pack_id, weight, hit_count, avg_score))

        pack_ranking.sort(key=lambda x: x[1], reverse=True)

        pack_results: list[PackSearchResult] = []
        for pack_id, weight, hit_count, avg_score in pack_ranking[:settings.pack_search_limit]:
            pack = self._store.get_pack_by_id(pack_id)
            if pack:
                pack_results.append(PackSearchResult(
                    id=pack.id, summary=pack.summary, topic=pack.topic,
                    keywords=pack.keywords, weight=weight,
                    hit_count=hit_count, avg_score=avg_score,
                ))

        return pack_results


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-Max 归一化到 [0, 1]"""
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s == min_s:
        return {k: 1.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}
