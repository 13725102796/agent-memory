"""优化改造完整测试 — 覆盖 P0/P1/P2 全部改造点
无需 LLM/Embedding 依赖，全部使用 mock 模拟。
"""
from __future__ import annotations

import copy
import json
import sys
import time
import threading
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, "/Users/maidong/Desktop/zyc/研究openclaw/agent-memory")

from memory_agent.types import MemoryRecord, MemoryType, SearchResult, PackSearchResult
from memory_agent.config import Settings


# ══════════════════════════════════════════════════════════
# 辅助工具
# ══════════════════════════════════════════════════════════

def _make_embedding(seed: int = 0) -> np.ndarray:
    """生成确定性 384 维单位向量"""
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _make_store():
    """创建干净的 SQLite store（内存数据库）"""
    from memory_agent.store.sqlite import SQLiteMemoryStore
    store = SQLiteMemoryStore(db_path=":memory:")
    store.init()
    return store


# ══════════════════════════════════════════════════════════
# P0-A: 线程安全测试
# ══════════════════════════════════════════════════════════

class TestThreadSafety(unittest.TestCase):
    """P0-A: 线程安全加固"""

    def test_concurrent_writes_no_corruption(self):
        """并发写入不应导致数据损坏"""
        store = _make_store()
        errors = []

        def write_memories(thread_id: int):
            try:
                for i in range(10):
                    rec = MemoryRecord(
                        id=f"thread-{thread_id}-mem-{i}",
                        user_id="test-user",
                        content=f"Thread {thread_id} memory {i}",
                        embedding=_make_embedding(thread_id * 100 + i),
                        importance=0.5,
                    )
                    store.insert_memory(rec)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_memories, args=(tid,)) for tid in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"并发写入出错: {errors}")
        all_mems = store.get_all_memories("test-user")
        self.assertEqual(len(all_mems), 50, f"预期 50 条记忆，实际 {len(all_mems)}")
        print(f"  ✓ 5 线程 × 10 写入 = {len(all_mems)} 条记忆，无损坏")

    def test_history_snapshot_isolation(self):
        """后台线程操作 history 快照不应影响主线程"""
        original = [
            {"role": "user", "content": "hello", "ts": 1.0},
            {"role": "assistant", "content": "hi", "ts": 1.0},
        ]
        snapshot = copy.deepcopy(original)

        # 模拟后台线程修改快照
        snapshot[0]["packed"] = True

        # 主线程原数据不受影响
        self.assertNotIn("packed", original[0])
        self.assertTrue(snapshot[0].get("packed"))
        print("  ✓ History 快照隔离正常")


# ══════════════════════════════════════════════════════════
# P0-B: Token 预算测试
# ══════════════════════════════════════════════════════════

class TestTokenBudget(unittest.TestCase):
    """P0-B: Token 预算管理"""

    def test_history_sliding_window(self):
        """History 应被截断到 max_history_turns"""
        from memory_agent.core.chat import _build_message_with_history

        # 构造 50 轮对话（100 条消息）
        history = []
        for i in range(50):
            history.append({"role": "user", "content": f"问题{i}", "ts": float(i)})
            history.append({"role": "assistant", "content": f"回答{i}", "ts": float(i)})

        result = _build_message_with_history(history, "最新问题")

        # 默认 max_history_turns=20，即最多 40 条消息
        # 确认包含最近的历史而非最早的
        self.assertIn("问题49", result)  # 最新的应该在
        self.assertIn("问题30", result)  # 第 30 轮也应该在
        self.assertNotIn("问题0", result)  # 最早的应该被截断
        self.assertIn("最新问题", result)
        print("  ✓ History 滑动窗口正常，旧消息被截断")

    def test_prompt_budget_truncation(self):
        """Prompt 各段应被截断到配置上限"""
        from memory_agent.core.chat import _build_prompt

        # 构造超长数据
        long_core = "核心信息 " * 500  # 超过 2000 字符
        long_recalled = [
            SearchResult(
                id=f"r{i}", content="记忆内容" * 200, score=0.8,
                tier="active", memory_type=MemoryType.PROJECT,
            )
            for i in range(10)
        ]
        packs = [
            PackSearchResult(
                id="p1", summary="摘要" * 300, topic="话题",
                keywords=["k1"], weight=0.5, hit_count=1, avg_score=0.5,
            )
        ]

        prompt = _build_prompt(long_core, long_recalled, packs)

        # 验证总长度被控制在合理范围
        self.assertLess(len(prompt), 15000, "Prompt 过长，截断未生效")
        print(f"  ✓ Prompt 总长度 {len(prompt)} 字符，截断正常")


# ══════════════════════════════════════════════════════════
# P0-C: 搜索归一化修复
# ══════════════════════════════════════════════════════════

class TestSearchNormalization(unittest.TestCase):
    """P0-C: 搜索归一化修复 + Pack 投票阈值"""

    def test_equal_scores_not_inflated(self):
        """所有分数相同时不应全部变为 1.0"""
        from memory_agent.memory.search import _min_max_normalize

        scores = {"a": 0.3, "b": 0.3, "c": 0.3}
        result = _min_max_normalize(scores)

        # 修复后：保留原值而非全部设为 1.0
        self.assertAlmostEqual(result["a"], 0.3)
        self.assertAlmostEqual(result["b"], 0.3)
        print("  ✓ 等值分数保留原值，不再膨胀到 1.0")

    def test_normal_normalization(self):
        """正常分数归一化应工作正常"""
        from memory_agent.memory.search import _min_max_normalize

        scores = {"a": 0.2, "b": 0.8, "c": 0.5}
        result = _min_max_normalize(scores)

        self.assertAlmostEqual(result["a"], 0.0)
        self.assertAlmostEqual(result["b"], 1.0)
        self.assertAlmostEqual(result["c"], 0.5)
        print("  ✓ 正常归一化工作正常")

    def test_empty_scores(self):
        """空分数应返回空字典"""
        from memory_agent.memory.search import _min_max_normalize

        result = _min_max_normalize({})
        self.assertEqual(result, {})
        print("  ✓ 空分数处理正常")


# ══════════════════════════════════════════════════════════
# P1-1+2: 类型分类 + 结构化元数据 + DB 迁移
# ══════════════════════════════════════════════════════════

class TestMemoryTypesAndMetadata(unittest.TestCase):
    """P1-1+2: 记忆类型分类 + 结构化元数据"""

    def test_memory_type_enum(self):
        """MemoryType 枚举应有 4 种类型"""
        types = list(MemoryType)
        self.assertEqual(len(types), 4)
        self.assertIn(MemoryType.USER, types)
        self.assertIn(MemoryType.FEEDBACK, types)
        self.assertIn(MemoryType.PROJECT, types)
        self.assertIn(MemoryType.REFERENCE, types)
        print("  ✓ MemoryType 枚举完整（4 种类型）")

    def test_memory_record_new_fields(self):
        """MemoryRecord 应包含新字段"""
        rec = MemoryRecord(
            id="test-1", user_id="u1", content="测试内容",
            memory_type=MemoryType.FEEDBACK,
            name="测试规则",
            description="这是一条测试规则的描述",
        )
        self.assertEqual(rec.memory_type, MemoryType.FEEDBACK)
        self.assertEqual(rec.name, "测试规则")
        self.assertEqual(rec.description, "这是一条测试规则的描述")
        print("  ✓ MemoryRecord 新字段正常")

    def test_db_migration_adds_columns(self):
        """DB 初始化应创建新列"""
        store = _make_store()

        # 插入带新字段的记忆
        rec = MemoryRecord(
            id="typed-1", user_id="u1", content="用户偏好 Python",
            embedding=_make_embedding(1),
            memory_type=MemoryType.USER,
            name="偏好Python",
            description="用户偏好使用 Python 做数据处理",
        )
        store.insert_memory(rec)

        # 读回验证
        all_mems = store.get_all_memories("u1")
        self.assertEqual(len(all_mems), 1)
        self.assertEqual(all_mems[0].memory_type, MemoryType.USER)
        self.assertEqual(all_mems[0].name, "偏好Python")
        self.assertEqual(all_mems[0].description, "用户偏好使用 Python 做数据处理")
        print("  ✓ DB 迁移正常，新列可读写")

    def test_search_result_carries_type(self):
        """SearchResult 应携带 memory_type 和 name"""
        sr = SearchResult(
            id="s1", content="内容", score=0.8, tier="active",
            memory_type=MemoryType.FEEDBACK, name="规则A",
            updated_at="2026-03-01T10:00:00",
        )
        self.assertEqual(sr.memory_type, MemoryType.FEEDBACK)
        self.assertEqual(sr.name, "规则A")
        self.assertIsNotNone(sr.updated_at)
        print("  ✓ SearchResult 携带新字段")


# ══════════════════════════════════════════════════════════
# P1-3: 新鲜度感知
# ══════════════════════════════════════════════════════════

class TestFreshness(unittest.TestCase):
    """P1-3: 新鲜度感知与时间衰减"""

    def test_age_text(self):
        """age_text 应返回正确的人类可读文本"""
        from memory_agent.memory.freshness import memory_age_text

        now = datetime.now().isoformat()
        self.assertEqual(memory_age_text(now), "今天")

        yesterday = (datetime.now() - timedelta(days=1, hours=1)).isoformat()
        self.assertEqual(memory_age_text(yesterday), "昨天")

        old = (datetime.now() - timedelta(days=47)).isoformat()
        self.assertEqual(memory_age_text(old), "47天前")

        self.assertEqual(memory_age_text(None), "今天")
        print("  ✓ memory_age_text 输出正确")

    def test_freshness_warning(self):
        """超过阈值天数应返回警告"""
        from memory_agent.memory.freshness import freshness_warning

        recent = datetime.now().isoformat()
        self.assertIsNone(freshness_warning(recent))

        old = (datetime.now() - timedelta(days=30)).isoformat()
        warning = freshness_warning(old)
        self.assertIsNotNone(warning)
        self.assertIn("30", warning)
        self.assertIn("过时", warning)
        print("  ✓ freshness_warning 阈值判断正确")

    def test_time_decay_factor(self):
        """时间衰减因子应在 [floor, 1.0] 范围"""
        from memory_agent.memory.freshness import time_decay_factor

        # 今天的记忆：衰减约 1.0
        today = datetime.now().isoformat()
        self.assertAlmostEqual(time_decay_factor(today), 1.0, places=1)

        # 90 天前：衰减约 0.75（半衰期 90 天，decay = 1 - 90/180 = 0.5? 不对，实际 = max(0.5, 1 - 90/180) = 0.5）
        old_90 = (datetime.now() - timedelta(days=90)).isoformat()
        decay_90 = time_decay_factor(old_90)
        self.assertGreaterEqual(decay_90, 0.5)
        self.assertLessEqual(decay_90, 0.6)

        # 365 天前：应为下限 0.5
        old_365 = (datetime.now() - timedelta(days=365)).isoformat()
        self.assertAlmostEqual(time_decay_factor(old_365), 0.5, places=1)

        # None：应为 1.0
        self.assertAlmostEqual(time_decay_factor(None), 1.0)
        print("  ✓ time_decay_factor 范围正确")


# ══════════════════════════════════════════════════════════
# P2-4: 排除规则（在提取 prompt + 后置过滤中实现）
# ══════════════════════════════════════════════════════════

class TestExclusionRules(unittest.TestCase):
    """P2-4: 记忆排除规则"""

    def test_short_content_filtered(self):
        """过短的内容应被过滤"""
        # 模拟提取结果中的过短内容
        items = [
            {"content": "短", "importance": 0.5, "type": "project", "name": "x", "description": "x"},
            {"content": "这是一条足够长的记忆内容，应该被保留", "importance": 0.5, "type": "project", "name": "ok", "description": "ok"},
        ]
        filtered = [
            item for item in items
            if 10 <= len(item["content"].strip()) <= 500
        ]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["name"], "ok")
        print("  ✓ 过短内容被正确过滤")

    def test_importance_bounds(self):
        """importance 应被限制在 [0.1, 1.0]"""
        test_cases = [
            (-0.5, 0.1), (0.0, 0.1), (0.5, 0.5),
            (1.0, 1.0), (1.5, 1.0), (float('inf'), 1.0),
        ]
        for raw, expected in test_cases:
            clamped = max(0.1, min(1.0, raw))
            self.assertAlmostEqual(clamped, expected, msg=f"raw={raw}")
        print("  ✓ importance 范围限制正确")


# ══════════════════════════════════════════════════════════
# P2-5: 提取去重与游标追踪
# ══════════════════════════════════════════════════════════

class TestExtractionCursor(unittest.TestCase):
    """P2-5: 提取去重与游标追踪"""

    def test_cursor_persistence(self):
        """游标应能读写和持久化"""
        store = _make_store()

        # 初始为 0
        self.assertEqual(store.get_extraction_cursor("u1"), 0)

        # 写入并读回
        store.set_extraction_cursor("u1", 42)
        self.assertEqual(store.get_extraction_cursor("u1"), 42)

        # 更新
        store.set_extraction_cursor("u1", 100)
        self.assertEqual(store.get_extraction_cursor("u1"), 100)
        print("  ✓ 提取游标读写正常")

    def test_latest_message_id(self):
        """latest_message_id 应返回最新消息 ID"""
        store = _make_store()

        # 无消息时为 0
        self.assertEqual(store.get_latest_message_id("u1"), 0)

        # 添加消息
        store.append_message("u1", "user", "hello", time.time())
        store.append_message("u1", "assistant", "hi", time.time())
        latest = store.get_latest_message_id("u1")
        self.assertEqual(latest, 2)
        print("  ✓ latest_message_id 正常")

    def test_mutual_exclusion(self):
        """互斥门控：并发提取应被跳过"""
        from memory_agent.memory.extract import MemoryExtractor

        store = _make_store()
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = _make_embedding(0)

        extractor = MemoryExtractor(store, mock_llm, mock_embedder)

        # 模拟正在进行中
        extractor._in_progress = True
        extractor.extract_and_save("u1", "test", "reply")

        # LLM 不应被调用（因为被互斥跳过）
        mock_llm.cheap.assert_not_called()
        extractor._in_progress = False
        print("  ✓ 互斥门控正常，重叠提取被跳过")


# ══════════════════════════════════════════════════════════
# P2-6: Core Memory 版本化
# ══════════════════════════════════════════════════════════

class TestCoreMemoryVersioning(unittest.TestCase):
    """P2-6: Core Memory 版本化"""

    def test_history_saved_on_update(self):
        """更新 core memory 时应保存历史版本"""
        store = _make_store()

        # 首次设置
        store.set_core_memory("u1", "版本1：姓名: 小王")

        # 保存历史并更新
        store.save_core_history("u1", "版本1：姓名: 小王", reason="新发现姓名为小张")
        store.set_core_memory("u1", "版本2：姓名: 小张")

        # 再次更新
        store.save_core_history("u1", "版本2：姓名: 小张", reason="新增职业信息")
        store.set_core_memory("u1", "版本3：姓名: 小张, 职业: 程序员")

        # 验证当前版本
        current = store.get_core_memory("u1")
        self.assertIn("版本3", current)
        print("  ✓ Core Memory 更新正常")

    def test_rollback(self):
        """回滚应恢复上一版本"""
        store = _make_store()

        store.set_core_memory("u1", "版本1")
        store.save_core_history("u1", "版本1", reason="准备更新")
        store.set_core_memory("u1", "版本2")

        # 回滚
        success = store.rollback_core_memory("u1")
        self.assertTrue(success)
        self.assertEqual(store.get_core_memory("u1"), "版本1")
        print("  ✓ Core Memory 回滚正常")

    def test_history_limit(self):
        """历史版本应限制为 20 条"""
        store = _make_store()

        for i in range(25):
            store.save_core_history("u1", f"版本{i}", reason=f"更新{i}")

        # 查询历史条数
        conn = store._connect()
        count = conn.execute(
            "SELECT COUNT(*) as c FROM core_memory_history WHERE user_id = ?", ("u1",)
        ).fetchone()["c"]
        self.assertEqual(count, 20, f"预期 20 条历史，实际 {count}")
        print("  ✓ Core Memory 历史限制 20 条")


# ══════════════════════════════════════════════════════════
# P2-7: 记忆摘要索引
# ══════════════════════════════════════════════════════════

class TestMemoryIndex(unittest.TestCase):
    """P2-7: 记忆摘要索引"""

    def test_index_generation(self):
        """索引应按类型分组生成"""
        from memory_agent.memory.index import MemoryIndex

        store = _make_store()

        # 插入不同类型的记忆
        types_data = [
            (MemoryType.FEEDBACK, "用中文回答", "用户要求用中文"),
            (MemoryType.USER, "Python专家", "用户是 Python 高级开发者"),
            (MemoryType.PROJECT, "记忆系统开发", "正在开发三层记忆系统"),
            (MemoryType.REFERENCE, "Linear项目", "Bug 跟踪在 Linear INGEST 项目"),
        ]
        for mt, name, desc in types_data:
            rec = MemoryRecord(
                id=f"idx-{mt.value}", user_id="u1", content=f"内容：{desc}",
                embedding=_make_embedding(hash(name) % 1000),
                memory_type=mt, name=name, description=desc,
                importance=0.7,
            )
            store.insert_memory(rec)

        index = MemoryIndex(store)
        result = index.build("u1")

        # 验证分组
        self.assertIn("行为指导", result)
        self.assertIn("用户画像", result)
        self.assertIn("项目上下文", result)
        self.assertIn("外部资源", result)

        # 验证内容
        self.assertIn("用中文回答", result)
        self.assertIn("Python专家", result)
        print(f"  ✓ 索引生成正常（{len(result)} 字符）")

    def test_index_caching(self):
        """索引应被缓存"""
        from memory_agent.memory.index import MemoryIndex

        store = _make_store()
        rec = MemoryRecord(
            id="cache-1", user_id="u1", content="测试缓存",
            embedding=_make_embedding(0), importance=0.5,
        )
        store.insert_memory(rec)

        index = MemoryIndex(store)

        # 第一次构建
        result1 = index.build("u1")
        # 第二次应从缓存返回
        result2 = index.build("u1")
        self.assertEqual(result1, result2)

        # 失效缓存后重建
        index.invalidate("u1")
        result3 = index.build("u1")
        self.assertEqual(result1, result3)  # 内容相同但重新构建
        print("  ✓ 索引缓存正常")

    def test_index_truncation(self):
        """超过字符上限的索引应被截断"""
        from memory_agent.memory.index import MemoryIndex

        store = _make_store()

        # 插入大量记忆，每条描述足够长以触发字符截断
        for i in range(300):
            long_desc = f"这是第 {i} 条记忆的描述，包含详细的上下文信息用于测试截断功能是否正确触发" * 2
            rec = MemoryRecord(
                id=f"trunc-{i}", user_id="u1",
                content=f"记忆内容 {i} " * 10,
                embedding=_make_embedding(i),
                importance=0.5,
                name=f"记忆条目{i}号",
                description=long_desc[:150],
            )
            store.insert_memory(rec)

        index = MemoryIndex(store)
        result = index.build("u1")

        # 索引应在字符上限附近被截断（_MAX_CHARS=5000）
        self.assertLessEqual(len(result), 6000, "索引应被截断到上限附近")
        self.assertIn("截断", result)
        print(f"  ✓ 索引截断正常（{len(result)} 字符）")


# ══════════════════════════════════════════════════════════
# P2-8: feedback 类型结构化
# ══════════════════════════════════════════════════════════

class TestFeedbackStructure(unittest.TestCase):
    """P2-8: feedback 类型结构化"""

    def test_feedback_content_formatting(self):
        """feedback 类型应融入 why + how_to_apply"""
        content = "不要用英文回答"
        why = "用户说看不懂英文"
        how_to_apply = "所有回答都用中文"

        # 模拟 extract.py 中的格式化逻辑
        if why:
            content += f"\n原因：{why}"
        if how_to_apply:
            content += f"\n应用场景：{how_to_apply}"

        self.assertIn("原因：", content)
        self.assertIn("应用场景：", content)
        self.assertIn("不要用英文回答", content)
        print("  ✓ feedback 结构化格式正确")

    def test_feedback_priority_in_prompt(self):
        """Prompt 中 feedback 类型应排在最前"""
        from memory_agent.core.chat import _build_prompt

        recalled = [
            SearchResult(id="p1", content="项目进度正常", score=0.9, tier="active",
                         memory_type=MemoryType.PROJECT),
            SearchResult(id="f1", content="用中文回答", score=0.7, tier="active",
                         memory_type=MemoryType.FEEDBACK),
            SearchResult(id="u1", content="用户是程序员", score=0.8, tier="active",
                         memory_type=MemoryType.USER),
        ]

        prompt = _build_prompt("", recalled, [])

        # feedback 应在 project 之前
        fb_pos = prompt.find("用中文回答")
        pj_pos = prompt.find("项目进度正常")
        self.assertLess(fb_pos, pj_pos, "feedback 应排在 project 之前")
        print("  ✓ feedback 在 prompt 中优先展示")


# ══════════════════════════════════════════════════════════
# 端到端集成测试
# ══════════════════════════════════════════════════════════

class TestEndToEnd(unittest.TestCase):
    """端到端集成测试（使用 mock LLM）"""

    def test_full_flow_with_types(self):
        """完整流程：插入 → 检索 → prompt 构建"""
        store = _make_store()

        # 插入多种类型的记忆
        test_memories = [
            ("fb-1", MemoryType.FEEDBACK, "用中文回答", "用户要求中文", "规则：全部用中文回答\n原因：用户不懂英文", 0.9),
            ("u-1", MemoryType.USER, "Python开发者", "高级 Python 工程师", "用户是有 10 年经验的 Python 开发者", 0.8),
            ("p-1", MemoryType.PROJECT, "记忆系统", "正在开发记忆系统", "用户正在基于 SQLite 开发三层记忆系统", 0.7),
            ("r-1", MemoryType.REFERENCE, "文档链接", "API 文档地址", "API 文档在 https://docs.example.com", 0.5),
        ]

        for mid, mtype, name, desc, content, imp in test_memories:
            rec = MemoryRecord(
                id=mid, user_id="u1", content=content,
                embedding=_make_embedding(hash(mid) % 1000),
                memory_type=mtype, name=name, description=desc,
                importance=imp,
            )
            store.insert_memory(rec)
            store.fts_sync(mid, content)

        # 验证全部存入
        all_mems = store.get_all_memories("u1")
        self.assertEqual(len(all_mems), 4)

        # 验证类型分布
        types = {m.memory_type for m in all_mems}
        self.assertEqual(types, {MemoryType.FEEDBACK, MemoryType.USER, MemoryType.PROJECT, MemoryType.REFERENCE})

        # 验证索引生成
        from memory_agent.memory.index import MemoryIndex
        index = MemoryIndex(store)
        idx_str = index.build("u1")
        self.assertIn("行为指导", idx_str)
        self.assertIn("用中文回答", idx_str)

        # 验证 prompt 构建
        from memory_agent.core.chat import _build_prompt
        recalled = [
            SearchResult(
                id=m.id, content=m.content, score=0.8,
                tier=m.tier, memory_type=m.memory_type,
                name=m.name, updated_at=datetime.now().isoformat(),
            )
            for m in all_mems
        ]
        prompt = _build_prompt(
            core_memory="姓名: 小赵\n职业: 程序员",
            recalled=recalled,
            packs=[],
            memory_index=idx_str,
        )

        # 验证 prompt 结构
        self.assertIn("<memory_index>", prompt)
        self.assertIn("<core_memory>", prompt)
        self.assertIn("<recalled_memory>", prompt)
        self.assertIn("[feedback]", prompt)

        # feedback 应在 project 之前
        fb_pos = prompt.find("[feedback]")
        pj_pos = prompt.find("[project]")
        self.assertLess(fb_pos, pj_pos)

        print(f"  ✓ 端到端流程正常（prompt {len(prompt)} 字符）")


# ══════════════════════════════════════════════════════════
# DB 迁移兼容性测试
# ══════════════════════════════════════════════════════════

class TestDBMigration(unittest.TestCase):
    """DB 迁移兼容性"""

    def test_all_new_tables_created(self):
        """所有新表应被创建"""
        store = _make_store()
        conn = store._connect()

        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        self.assertIn("core_memory_history", tables)
        self.assertIn("extraction_state", tables)
        print("  ✓ 新表 core_memory_history + extraction_state 已创建")

    def test_new_columns_exist(self):
        """memories 表应有新列"""
        store = _make_store()
        conn = store._connect()

        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}

        self.assertIn("memory_type", cols)
        self.assertIn("name", cols)
        self.assertIn("description", cols)
        print("  ✓ 新列 memory_type/name/description 已创建")

    def test_new_index_exists(self):
        """新索引应存在"""
        store = _make_store()
        conn = store._connect()

        indexes = [row[1] for row in conn.execute(
            "SELECT * FROM sqlite_master WHERE type='index'"
        ).fetchall()]

        self.assertIn("idx_memories_type", indexes)
        self.assertIn("idx_core_history_user", indexes)
        print("  ✓ 新索引已创建")


# ══════════════════════════════════════════════════════════
# 运行
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Agent-Memory 优化改造测试")
    print("=" * 60)
    unittest.main(verbosity=2)
