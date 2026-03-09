# Memory Pack — 对话压缩打包 + 混合检索 + 关联投票 实现计划

## Context

当前系统每轮对话逐条提取记忆，存入向量库的碎片化记忆噪声多、检索不精准。需要：

1. **会话级压缩**：对话 ≥20 轮时，软边界检测找断点，压缩为 ~300 字记忆包（Memory Pack）
2. **混合检索**：向量搜索 + BM25 全文搜索结合，解决纯向量搜索对专有名词/关键词的盲区
3. **关联投票**：细粒度记忆作为索引，按 pack_id 聚合投票，命中越多的 Pack 优先注入 prompt

核心约束：
- 断点检测**纯启发式**，不调用 LLM
- 用重叠窗口 + 链式摘要防止边界处信息丢失
- 压缩用现有的 `llm.cheap()` 调用
- BM25 使用 SQLite FTS5 内置模块，**零外部依赖**

---

## 改动文件清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `memory_agent/types.py` | 修改 | 新增 `MemoryPack`、`PackSearchResult`；`MemoryRecord` 加 `pack_id`；`SearchResult` 加 `pack_id` |
| `memory_agent/config.py` | 修改 | 新增 `pack_*` 压缩配置 + `hybrid_*` 混合检索配置 |
| `memory_agent/store/base.py` | 修改 | 新增 Pack CRUD + 关联回填 + FTS 同步抽象方法 |
| `memory_agent/store/sqlite.py` | 修改 | 新增 `memory_packs` 表 + FTS5 虚拟表 + memories 加 `pack_id` 列 |
| `memory_agent/memory/packer.py` | **新建** | `MemoryPacker` 类：断点检测 + 压缩 + 回填关联 |
| `memory_agent/memory/search.py` | 修改 | 纯向量搜索 → 混合检索（向量 + BM25）+ 两阶段关联投票 |
| `memory_agent/memory/extract.py` | 修改 | `_upsert` 插入/更新记忆时同步写入 FTS 索引 |
| `memory_agent/core/chat.py` | 修改 | 接入 Packer，历史加时间戳，prompt 加 pack |
| `memory_agent/cli/main.py` | 修改 | 新增 `/packs` 命令，更新 `/clear` `/stats` |

---

## 实现步骤

### Step 1 — types.py：新增 & 修改数据类

**新增：**

```python
@dataclass
class MemoryPack:
    """压缩后的对话记忆包"""
    id: str
    user_id: str
    summary: str                         # ~300字压缩摘要
    keywords: list[str]                  # 检索关键词
    topic: str                           # 一句话主题
    embedding: Optional[np.ndarray] = None
    prev_pack_id: Optional[str] = None   # 链式关联上一个包
    prev_context: str = ""               # 上一包末尾 ~50 字
    turn_count: int = 0
    created_at: Optional[str] = None

@dataclass
class PackSearchResult:
    """通过关联投票检索到的 MemoryPack"""
    id: str
    summary: str
    topic: str
    keywords: list[str]
    weight: float        # hit_count × avg_score
    hit_count: int       # 命中的记忆条数
    avg_score: float     # 命中记忆的平均分
```

**修改 MemoryRecord：**

```python
@dataclass
class MemoryRecord:
    # ... 现有字段不变 ...
    pack_id: Optional[str] = None    # 新增：关联的 MemoryPack ID
```

**修改 SearchResult：**

```python
@dataclass
class SearchResult:
    id: str
    content: str
    score: float          # 混合检索最终得分（向量 + BM25 加权）
    tier: str
    pack_id: Optional[str] = None    # 新增
```

---

### Step 2 — config.py：新增配置项

```python
# ── Memory Pack 压缩配置 ──
pack_trigger_turns: int = 20        # ≥20 轮开始评估
pack_max_turns: int = 35            # 无断点时强制切
pack_overlap_turns: int = 5         # 压缩后保留尾部 5 轮
pack_time_gap_minutes: int = 3      # 时间间隔断点阈值（分钟）
pack_summary_max_chars: int = 300   # 摘要上限
pack_prev_context_chars: int = 50   # 链式上下文长度
pack_search_limit: int = 2          # 检索时最多注入 2 个包

# ── 混合检索配置 ──
hybrid_vector_weight: float = 0.7   # 向量搜索得分权重
hybrid_bm25_weight: float = 0.3     # BM25 得分权重
```

---

### Step 3 — store/base.py：新增抽象方法

```python
# ── Memory Packs ──────────────────────────

@abstractmethod
def insert_pack(self, pack: MemoryPack) -> str: ...

@abstractmethod
def get_pack_by_id(self, pack_id: str) -> Optional[MemoryPack]: ...

@abstractmethod
def get_packs(self, user_id: str) -> list[MemoryPack]: ...

@abstractmethod
def get_latest_pack(self, user_id: str) -> Optional[MemoryPack]: ...

@abstractmethod
def delete_packs(self, user_id: str) -> int: ...

# ── 记忆 ↔ Pack 关联 ─────────────────────

@abstractmethod
def link_memories_to_pack(
    self, user_id: str, start_ts: str, end_ts: str, pack_id: str
) -> int: ...

# ── FTS 全文索引 ──────────────────────────

@abstractmethod
def fts_sync(self, memory_id: str, content: str) -> None: ...

@abstractmethod
def fts_delete(self, memory_id: str) -> None: ...

@abstractmethod
def fts_search(self, query: str, user_id: str, limit: int) -> list[tuple[str, float]]: ...
```

---

### Step 4 — store/sqlite.py：Schema 变更 + 实现

#### 4a. Schema

新增 `memory_packs` 表：

```sql
CREATE TABLE IF NOT EXISTS memory_packs (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    summary      TEXT NOT NULL,
    keywords     TEXT NOT NULL DEFAULT '[]',
    topic        TEXT NOT NULL DEFAULT '',
    embedding    BLOB,
    prev_pack_id TEXT,
    prev_context TEXT NOT NULL DEFAULT '',
    turn_count   INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_packs_user
    ON memory_packs (user_id, created_at);
```

`memories` 表新增列：

```sql
ALTER TABLE memories ADD COLUMN pack_id TEXT;
```

**新增 FTS5 虚拟表：**

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    memory_id UNINDEXED,
    user_id UNINDEXED,
    tokenize='unicode61'
);
```

> **为什么用 `unicode61` tokenizer**：SQLite FTS5 内置，对中文按字符粒度分词，
> 对英文按空格/标点分词。无需 jieba 等外部依赖，足以覆盖关键词匹配需求。
> 查询 "SQLite" 精确命中，查询 "存储" 按字符 "存"+"储" 命中。

#### 4b. 初始化时同步已有数据到 FTS

```python
def init(self):
    # ... 建表 ...

    # 把已有 memories 同步到 FTS（仅首次需要）
    existing = conn.execute(
        "SELECT id FROM memories WHERE id NOT IN (SELECT memory_id FROM memories_fts)"
    ).fetchall()
    if existing:
        for row in conn.execute("SELECT id, content, user_id FROM memories"):
            conn.execute(
                "INSERT INTO memories_fts (content, memory_id, user_id) VALUES (?, ?, ?)",
                (row["content"], row["id"], row["user_id"])
            )
        conn.commit()
```

#### 4c. FTS 同步方法

```python
def fts_sync(self, memory_id: str, content: str) -> None:
    """插入或更新 FTS 索引"""
    conn = self._connect()
    # 先删旧的（如有）
    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
    # 查 user_id
    row = conn.execute("SELECT user_id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if row:
        conn.execute(
            "INSERT INTO memories_fts (content, memory_id, user_id) VALUES (?, ?, ?)",
            (content, memory_id, row["user_id"])
        )
    conn.commit()

def fts_delete(self, memory_id: str) -> None:
    conn = self._connect()
    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
    conn.commit()

def fts_search(self, query: str, user_id: str, limit: int = 20) -> list[tuple[str, float]]:
    """BM25 全文搜索，返回 [(memory_id, bm25_score), ...]"""
    conn = self._connect()
    # FTS5 的 bm25() 返回负数（越小越相关），取反变正数
    rows = conn.execute("""
        SELECT memory_id, -bm25(memories_fts) as score
        FROM memories_fts
        WHERE memories_fts MATCH ? AND user_id = ?
        ORDER BY score DESC
        LIMIT ?
    """, (query, user_id, limit)).fetchall()
    return [(row[0], row[1]) for row in rows]
```

#### 4d. Pack CRUD + 关联回填

```python
def insert_pack(self, pack: MemoryPack) -> str: ...
def get_pack_by_id(self, pack_id: str) -> Optional[MemoryPack]: ...
def get_packs(self, user_id: str) -> list[MemoryPack]: ...
def get_latest_pack(self, user_id: str) -> Optional[MemoryPack]: ...
def delete_packs(self, user_id: str) -> int: ...
def _row_to_pack(row) -> MemoryPack: ...

def link_memories_to_pack(
    self, user_id: str, start_ts: str, end_ts: str, pack_id: str
) -> int:
    """把指定时间范围内的记忆关联到 pack_id"""
    conn = self._connect()
    cursor = conn.execute("""
        UPDATE memories SET pack_id = ?
        WHERE user_id = ? AND created_at >= ? AND created_at <= ?
          AND pack_id IS NULL
    """, (pack_id, user_id, start_ts, end_ts))
    count = cursor.rowcount
    conn.commit()
    return count
```

#### 4e. get_memories_by_tier 返回 pack_id

修改查询 SQL，SELECT 中加入 `pack_id`，`_row_to_record` 也要读取 `pack_id`。

#### 4f. /clear 时清理 FTS

`delete_memory` 和批量清理时同步删除 FTS 记录。

---

### Step 5 — memory/extract.py：写入时同步 FTS

在 `_upsert` 中，插入新记忆或更新已有记忆后，同步写 FTS 索引：

```python
def _upsert(self, user_id: str, content: str, importance: float):
    new_vec = self._embedder.embed(content)
    # ... 已有的相似度判断 ...

    if best_sim > duplicate_threshold:
        return  # 跳过

    if best_sim > conflict_threshold:
        self._store.update_memory(best_rec.id, content, new_vec.tobytes(), importance)
        self._store.fts_sync(best_rec.id, content)     # ← 同步 FTS
        return

    # 新增
    record = MemoryRecord(...)
    memory_id = self._store.insert_memory(record)
    self._store.fts_sync(memory_id, content)            # ← 同步 FTS
```

---

### Step 6 — memory/search.py：混合检索 + 关联投票（核心变更）

#### 整体流程

```
查询 "用SQLite还是PostgreSQL"
        │
        ├── 向量搜索 ──→ 语义相似的记忆 + 分数
        │
        ├── BM25 搜索 ──→ 关键词命中的记忆 + 分数
        │
        ▼
   分数归一化 + 加权合并
   final_score = 0.7 × vector_norm + 0.3 × bm25_norm
        │
        ▼
   按 pack_id 聚合投票 → Top-N Pack
```

#### 混合检索实现

```python
def search(self, user_id: str, query: str) -> tuple[list[SearchResult], list[PackSearchResult]]:
    """
    三阶段检索：
    Stage 1: 向量搜索 + BM25 搜索 → 混合得分
    Stage 2: 按 pack_id 聚合投票 → 提取关联度最高的 Pack
    """
    query_vec = self._embedder.embed(query)

    # ── Stage 1a: 向量搜索（已有逻辑） ──
    vector_hits = self._vector_search(query_vec, user_id)
    # → {memory_id: vector_score}

    # ── Stage 1b: BM25 搜索 ──
    bm25_hits = self._store.fts_search(query, user_id, limit=20)
    # → [(memory_id, bm25_score)]

    # ── Stage 1c: 分数归一化 + 加权合并 ──
    merged = self._merge_scores(vector_hits, bm25_hits)
    # → 按 final_score 排序的 SearchResult 列表（带 pack_id）

    # ── Stage 2: 按 pack_id 聚合投票 ──
    pack_results = self._aggregate_packs(merged)

    return merged, pack_results
```

#### 分数归一化（Min-Max）

```python
def _merge_scores(
    self,
    vector_hits: dict[str, tuple[float, MemoryRecord]],
    bm25_hits: list[tuple[str, float]],
) -> list[SearchResult]:
    """
    归一化两路分数，加权合并。

    归一化方式：Min-Max Normalization
      norm = (score - min) / (max - min)
      范围归到 [0, 1]

    合并公式：
      final = vector_weight × vector_norm + bm25_weight × bm25_norm
      默认  = 0.7 × vector_norm + 0.3 × bm25_norm
    """
    all_ids = set(vector_hits.keys()) | {mid for mid, _ in bm25_hits}

    # 归一化向量分数
    v_scores = {mid: s for mid, (s, _) in vector_hits.items()}
    v_norm = _min_max_normalize(v_scores)

    # 归一化 BM25 分数
    b_scores = {mid: s for mid, s in bm25_hits}
    b_norm = _min_max_normalize(b_scores)

    # 加权合并
    results = []
    for mid in all_ids:
        v = v_norm.get(mid, 0.0)
        b = b_norm.get(mid, 0.0)
        final = settings.hybrid_vector_weight * v + settings.hybrid_bm25_weight * b

        if final < settings.search_min_score:
            continue

        # 从 vector_hits 或数据库获取记忆详情
        record = vector_hits[mid][1] if mid in vector_hits else self._load_record(mid)
        results.append(SearchResult(
            id=mid,
            content=record.content,
            score=final,
            tier=record.tier,
            pack_id=record.pack_id,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:settings.search_top_k]

def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s == min_s:
        return {k: 1.0 for k in scores}  # 全部相同则归为 1.0
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}
```

#### 向量搜索 vs BM25 各自擅长什么

```
                    向量搜索              BM25
                    ────────              ────
查询 "数据库选型"    ✓ 语义命中             ✗ 可能漏掉（内容用的是"SQLite"）
                    "选择SQLite做存储"

查询 "SQLite"       △ 可能漏掉             ✓ 精确关键词命中
                    （embedding距离不确定）   所有含"SQLite"的记忆

查询 "Go 语言"      ✓ 语义关联             ✓ 关键词命中
                    "用户主力语言Go"        精确匹配"Go"

合并后 → 两路优势互补，召回率↑，准确率↑
```

#### 关联投票（同原方案，不变）

```python
def _aggregate_packs(self, hits: list[SearchResult]) -> list[PackSearchResult]:
    pack_votes: dict[str, list[float]] = {}
    for hit in hits:
        if hit.pack_id:
            pack_votes.setdefault(hit.pack_id, []).append(hit.score)

    pack_ranking = []
    for pack_id, scores in pack_votes.items():
        hit_count = len(scores)
        avg_score = sum(scores) / hit_count
        weight = hit_count * avg_score
        pack_ranking.append((pack_id, weight, hit_count, avg_score))

    pack_ranking.sort(key=lambda x: x[1], reverse=True)

    pack_results = []
    for pack_id, weight, hit_count, avg_score in pack_ranking[:settings.pack_search_limit]:
        pack = self._store.get_pack_by_id(pack_id)
        if pack:
            pack_results.append(PackSearchResult(
                id=pack.id, summary=pack.summary, topic=pack.topic,
                keywords=pack.keywords, weight=weight,
                hit_count=hit_count, avg_score=avg_score,
            ))
    return pack_results
```

---

### Step 7 — memory/packer.py：核心新模块

#### 7a. 软边界断点检测

`_find_break_point(history)` — 扫描 `history[trigger*2 : max*2]`，对每个 user 条目打分：

| 信号 | 条件 | 分值 | 原理 |
|---|---|---|---|
| 时间间隔 | 与上一轮间隔 > 3 分钟 | +3 | 人类换话题前通常会停顿 |
| 结束语 | 用户消息含信号词 | +2 | 话题收束的语言模式 |
| 关键词漂移 | 前后 5 轮 Jaccard 距离 > 0.5 | +1 | 前后用词不重叠说明换话题了 |

取最高分位置为断点。全部 0 分则在 `pack_max_turns`（35）处强制切。

**断点信号词列表：**

```python
_BREAK_SIGNALS = [
    "好的", "明白了", "知道了", "懂了", "了解了",
    "换个话题", "另外", "顺便", "对了", "说到这",
    "先这样", "好了", "行了", "没问题", "可以了",
]
```

#### 7b. 关键词漂移检测

`_keyword_drift(window_a, window_b) -> float`

- 用正则 `[\w\u4e00-\u9fff]{2,}` 提取词汇（≥2字符）
- 过滤停用词（的、了、是、在、我、你...）
- 计算 Jaccard 距离：`1 - |A∩B| / |A∪B|`
- 距离 > 0.5 表示话题大概率切换

#### 7c. 压缩 + 回填流程

`MemoryPacker.maybe_compress(user_id, history) -> history`

```
turn_count = len(history) // 2
if turn_count < 20: return history  ← 不够轮次，跳过

break_idx = _find_break_point(history)  ← 纯启发式打分
segment = history[0 : break_idx]        ← 要压缩的部分

# 记录时间范围（用于回填 pack_id）
start_ts = segment[0].get("ts")
end_ts = segment[-1].get("ts")

pack = _compress_segment(user_id, segment)  ← LLM 压缩
store.insert_pack(pack)                     ← 持久化

# 回填：把这段时间内提取的记忆关联到此 pack
store.link_memories_to_pack(user_id, start_ts, end_ts, pack.id)

# 重叠窗口：保留 segment 尾部 5 轮 + break 之后的轮次
overlap_start = max(0, break_idx - overlap_turns * 2)
new_history = history[overlap_start:]
return new_history
```

#### 7d. LLM 压缩

`_compress_segment(user_id, segment) -> MemoryPack`

- 拼接对话文本（每条截断 300 字）
- 获取 `get_latest_pack()` 的 `prev_context` 用于承上启下
- 调用 `llm.cheap()` 压缩为 JSON

压缩 Prompt：

```
将以下对话压缩为结构化记忆包。

上一段对话结尾（参考）：{prev_context}

对话内容：
{conv_text}

严格按照以下 JSON 格式输出（不要 markdown 代码块）：
{"summary": "≤300字摘要", "topic": "一句话主题", "keywords": ["关键词..."]}

只输出 JSON，不要其他内容：
```

- 对 summary 做 `embedder.embed()` 向量化
- 组装 MemoryPack 对象返回

---

### Step 8 — core/chat.py：接入

改动点：

1. **构造函数**：新增 `self._packer = MemoryPacker(store, llm, embedder)`
2. **MAX_HISTORY**：10 → 40（安全上限，packer 是主要裁剪机制）
3. **历史条目加时间戳**：`{"role": "user", "content": msg, "ts": time.time()}`
4. **handle() 流程变更**：

```python
def handle(self, user_id: str, message: str) -> str:
    # ① Core Memory
    core = self._store.get_core_memory(user_id)

    # ② 混合检索 + 关联投票
    recalled, packs = self._searcher.search(user_id, message)

    # ③ 拼 Prompt
    system_prompt = _build_prompt(core, recalled, packs)
    full_message = _build_message_with_history(self._history, message)

    # ④ 调 LLM
    reply = self._llm.chat(system_prompt, full_message)

    # ⑤ 追加历史（带时间戳）
    _ts = time.time()
    self._history.append({"role": "user", "content": message, "ts": _ts})
    self._history.append({"role": "assistant", "content": reply, "ts": _ts})
    if len(self._history) > self.MAX_HISTORY * 2:
        self._history = self._history[-(self.MAX_HISTORY * 2):]

    # ⑥ 可能压缩
    self._history = self._packer.maybe_compress(user_id, self._history)

    # ⑦ 提取记忆（内部会同步 FTS 索引）
    self._extractor.extract_and_save(user_id, message, reply)

    return reply
```

5. **_build_prompt() 新增 `<memory_packs>` 区块**

Prompt 注入顺序（优先级从高到低）：

```
<core_memory>...</core_memory>             ← 始终注入
<memory_packs>                             ← 关联投票 Top-2 包
  [话题] (关联记忆 3 条, 权重 1.60)
  摘要内容...
</memory_packs>
<recalled_memory>...</recalled_memory>     ← 细粒度记忆（混合检索结果）
<conversation_history>...</conversation_history>  ← 滑动窗口
当前消息
```

---

### Step 9 — cli/main.py：命令更新

- `/packs` — 显示所有 Memory Pack（topic + turn_count + summary 前 80 字）
- `/clear` — 增加 `store.delete_packs(user_id)` + 清理 FTS 索引
- `/stats` — 增加 pack 数量显示
- 启动 banner 更新，列出 `/packs` 命令

---

## 混合检索原理

### 向量搜索 vs BM25 互补关系

```
              向量搜索                     BM25
              ────────                     ────
擅长          语义相似（同义词、释义）       精确关键词匹配
弱点          专有名词可能映射不到近处       不理解语义、同义词

示例：
  "数据库选型" → 命中 "选择SQLite做存储"    → 可能漏掉（没有"数据库选型"原文）
  "SQLite"    → 可能漏掉                   → 精确命中所有含"SQLite"的记忆
  "Go 语言"   → 命中 "主力语言Go"           → 命中 "Go"

两路合并 → 召回率 ↑，准确率 ↑
```

### 归一化 + 加权合并

```
向量搜索结果:
  mem_001: 0.82
  mem_002: 0.65
  mem_003: 0.30
  → Min-Max 归一化 → mem_001: 1.0, mem_002: 0.67, mem_003: 0.0

BM25 搜索结果:
  mem_002: 12.5
  mem_004: 8.3
  → Min-Max 归一化 → mem_002: 1.0, mem_004: 0.0

加权合并 (0.7 × 向量 + 0.3 × BM25):
  mem_001: 0.7 × 1.0  + 0.3 × 0.0  = 0.70
  mem_002: 0.7 × 0.67 + 0.3 × 1.0  = 0.77  ← BM25 补充拉高了排名
  mem_003: 0.7 × 0.0  + 0.3 × 0.0  = 0.00
  mem_004: 0.7 × 0.0  + 0.3 × 0.0  = 0.00

最终排序: mem_002 > mem_001 > ...
```

### 为什么 0.7:0.3

- 向量搜索是主力（语义理解能力强），权重高
- BM25 是辅助（补充关键词盲区），权重低
- 这是业界常用的起步值，后续可根据实际效果调

---

## 防丢机制设计

### 1. 重叠窗口（零成本）

压缩 history[0:30] 后，保留 history[25:] 继续累积。

```
Pack A: 轮次 1━━━━━━━━━━━━━━━━━━━━━━30
                              ┃ 重叠 ┃
Pack B:                  轮次 26━━━━━━━━━━━━━━━━━━━━━━55
```

### 2. 链式摘要（多 ~50 字输入）

每个 Pack 携带 `prev_context`（上一包末尾 50 字），压缩时喂给 LLM，保证摘要承上启下。

### 3. 软边界检测（零成本）

不在固定轮次硬切，而是在 20-35 轮范围内找话题自然断点。

---

## 完整检索流程图

```
查询: "我们之前讨论的架构方案是什么？"
        │
        ├────────────────────┬──────────────────────┐
        ▼                    ▼                      │
   向量搜索              BM25 搜索                   │
   embed(query)          FTS5 MATCH                 │
        │                    │                      │
        ▼                    ▼                      │
   vec_hits:             bm25_hits:                 │
   mem_001: 0.82         mem_002: 12.5              │
   mem_002: 0.65         mem_004: 8.3               │
   mem_003: 0.30                                    │
        │                    │                      │
        └────────┬───────────┘                      │
                 ▼                                  │
        Min-Max 归一化                               │
        加权合并 (0.7:0.3)                           │
                 │                                  │
                 ▼                                  │
        merged_hits (按 final_score 排序):           │
          mem_002: 0.77  pack_id=pack_001           │
          mem_001: 0.70  pack_id=pack_001           │
          mem_004: 0.21  pack_id=pack_002           │
                 │                                  │
                 ▼                                  │
        按 pack_id 聚合投票:                         │
          pack_001: 2条, avg=0.74 → weight=1.48 ✓   │
          pack_002: 1条, avg=0.21 → weight=0.21 ✗   │
                 │                                  │
                 ▼                                  │
        加载 pack_001.summary                       │
                 │                                  │
                 ▼                                  │
        注入 Prompt:                                │
        <memory_packs>                              │
          [AI记忆系统架构讨论]                        │
          用户讨论了三层记忆模型...选择SQLite方案...    │
        </memory_packs>                             │
        <recalled_memory>                           │
          - 讨论了SQLite vs PostgreSQL               │
          - 用户选择SQLite做存储                      │
        </recalled_memory>                          │
```

---

## 数据流全景

```
用户输入 Turn N
    │
    ▼
ChatHandler.handle()
    ├── ① 加载 Core Memory
    ├── ② 混合检索 + 关联投票
    │     ├── 向量搜索 memories → vec_hits
    │     ├── BM25 搜索 FTS5   → bm25_hits
    │     ├── 归一化 + 加权合并  → merged_hits
    │     └── 按 pack_id 聚合   → top_packs
    ├── ③ 拼 Prompt（core + packs + recalled + history + message）
    ├── ④ 调 LLM
    ├── ⑤ 追加历史（带 ts 时间戳）
    ├── ⑥ Packer.maybe_compress()
    │     ├── turn_count < 20? → 跳过
    │     ├── _find_break_point() → 纯启发式打分
    │     ├── _compress_segment() → llm.cheap() 压缩
    │     ├── store.insert_pack() → 持久化
    │     └── store.link_memories_to_pack() → 回填关联
    └── ⑦ 提取逐条记忆 + 同步 FTS 索引
```

---

## 记忆层级总览

```
层级               大小         生命周期     注入方式
──────────────────────────────────────────────────────────────
Core Memory        ~200字       永久         始终注入
Memory Pack        ~300字/包    长期         通过关联投票检索注入
Active Memory      ~50字/条    中期         混合检索（向量+BM25）
Inactive Memory    ~50字/条    衰减         混合检索（score × 0.8）
Conversation Hx    原文         会话内       滑动窗口
```

---

## 验证方式

```bash
cd demo
/usr/bin/python3 -m memory_agent.cli.main
```

测试场景：

1. **混合检索**：存入含 "SQLite" 的记忆后，查询 "SQLite" 验证 BM25 命中
2. **向量补充**：查询 "数据库选型" 验证向量搜索命中 "选择SQLite做存储"
3. **对话压缩**：连续对话 20+ 轮，观察日志触发压缩和 pack_id 回填
4. **关联投票**：重启后提问"我们之前聊了什么"，验证 Pack 被关联投票命中
5. `/packs` 查看记忆包内容
6. `/memory` 检查记忆条目 pack_id 关联
7. `/stats` 确认各计数正确
8. `/clear` 确认全部清空（memories + FTS + packs）
