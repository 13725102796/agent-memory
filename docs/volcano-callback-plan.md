# 火山引擎字幕回调接入 agent-memory 方案

## Context

硬件设备通过火山引擎 AI 对话服务进行语音交互，火山引擎提供字幕回调（ServerMessageUrl）推送 AI 对话数据。需要在现有 agent-memory 服务上扩展，接收这些回调，存储对话记录，自动提取记忆，并对外提供查询 API。

**核心设计思路**：`device_id` 直接映射为现有系统的 `user_id`，每个硬件设备视为一个独立"用户"。这样无需修改现有的记忆提取、检索、压缩逻辑，只需新增一个"数据摄入通道"。

## 火山引擎回调格式

```
POST /api/volcano/callback?device_id=xxx
Body: { "message": "Base64编码", "signature": "鉴权签名" }

Base64解码后: 二进制头 (magic "subv" + 4字节长度) + JSON负载
JSON: {
  "type": "subtitle",
  "data": [{
    "text": "字幕文本", "userId": "说话者ID",
    "sequence": 1, "definite": true/false,
    "paragraph": true/false, "roundId": 1
  }]
}
```

- `definite=true` 表示最终文本，`false` 为中间结果（ASR 还在识别）
- `userId == bot_id` 为 AI 回复，否则为用户语音
- `paragraph=true` 标记一句话结束

---

## 实现步骤

### Step 1: 配置扩展
**文件**: `memory_agent/config.py`

新增配置项：
```python
# 火山引擎字幕回调
volcano_enabled: bool = False
volcano_signature: str = ""              # ServerMessageSignature 鉴权
volcano_flush_timeout_sec: int = 30      # 无新数据时强制刷新缓冲区
volcano_default_bot_id: str = ""         # 默认 Bot ID（区分 AI vs 用户）
```

### Step 2: 新增类型
**文件**: `memory_agent/types.py`

```python
@dataclass
class SubtitleEntry:
    """火山引擎字幕回调条目"""
    text: str
    userId: str
    sequence: int
    definite: bool
    paragraph: bool
    roundId: int
    language: str = "zh"
```

### Step 3: 回调解码器
**新建**: `memory_agent/volcano/__init__.py` (空)
**新建**: `memory_agent/volcano/decoder.py`

功能：
1. 签名验证：`body["signature"] == expected_signature`
2. Base64 解码 `body["message"]`
3. 解析二进制头：magic `subv`(4字节) + 长度(4字节大端序) + JSON 负载
4. 返回解析后的字幕数据列表

### Step 4: 字幕组装器
**新建**: `memory_agent/volcano/assembler.py`

核心组件 `SubtitleAssembler`：
- 维护每个 `(device_id, round_id)` 的文本缓冲区（user_buffer / bot_buffer）
- **只处理 `definite=true` 的片段**（中间结果忽略）
- **`paragraph=true` 时刷新缓冲区**，拼接为完整的一句话
- **`roundId` 变化时**，强制刷新上一轮缓冲区，组装为完整对话轮次
- 返回已完成的 `(user_text, bot_text)` 对话对
- 30秒无活动超时强制刷新（处理对话结尾无 paragraph 信号的情况）
- `threading.Lock` 保证线程安全

```
回调片段流 → [只取 definite=true] → 按 speaker 缓冲 → paragraph=true 或 roundId 变化时刷新
              → 输出完整对话对 (user_text, bot_text)
```

### Step 5: SQLite 扩展
**文件**: `memory_agent/store/sqlite.py`

新增表 `subtitle_fragments`（原始片段存储，用于调试/审计）：
```sql
CREATE TABLE IF NOT EXISTS subtitle_fragments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id   TEXT NOT NULL,
    round_id    INTEGER NOT NULL,
    sequence    INTEGER NOT NULL,
    speaker     TEXT NOT NULL,        -- 'user' or 'bot'
    text        TEXT NOT NULL,
    definite    INTEGER DEFAULT 0,
    paragraph   INTEGER DEFAULT 0,
    received_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(device_id, round_id, sequence, speaker)
);
```

新增方法：
- `insert_fragment()` — INSERT OR REPLACE 存储原始片段
- `get_fragments_by_round()` — 按 round 查询片段（调试用）

**线程安全**：给 `SQLiteMemoryStore` 增加 `threading.Lock`，所有数据库操作加锁。

### Step 6: HTTP 路由扩展
**文件**: `memory_agent/web/server.py`

新增路由：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/volcano/callback?device_id=xxx` | 接收火山引擎字幕回调 |
| GET | `/api/device/<device_id>/history` | 查询设备对话记录 |
| GET | `/api/device/<device_id>/memories` | 查询设备记忆 |
| GET | `/api/device/<device_id>/stats` | 查询设备记忆统计 |
| GET | `/api/device/<device_id>/core` | 查询设备核心记忆 |

回调处理流程：
```python
def _handle_volcano_callback(self, device_id):
    body = self._read_body()
    # 1. 验证签名
    # 2. decoder.decode(body) → 字幕条目列表
    # 3. assembler.process(device_id, entries, bot_id) → 完成的对话对
    # 4. 对每个完成的对话对：
    #    store.append_message(device_id, "user", user_text, ts)
    #    store.append_message(device_id, "assistant", bot_text, ts)
    #    后台线程: extractor.extract_and_save(device_id, user_text, bot_text)
    return {"ok": True}
```

设备查询端点复用现有 store 方法（device_id 就是 user_id）：
```python
def _handle_device_memories(self, device_id):
    memories = self._store.get_all_memories(device_id)  # 直接用 device_id 作为 user_id
    ...
```

路由匹配用正则：
```python
_DEVICE_RE = re.compile(r'^/api/device/([^/]+)/(history|memories|stats|core)$')
_VOLCANO_RE = re.compile(r'^/api/volcano/callback')
```

### Step 7: 初始化更新
**文件**: `memory_agent/web/server.py`

`_create_app()` 扩展返回值，增加 `extractor`, `packer`, `assembler`。
`ChatRequestHandler.__init__` 接收新组件。
现有功能完全不受影响（向后兼容）。

---

## 数据流

```
硬件设备 ─────── 火山引擎 AI ──────── 字幕回调
                                        │
                POST /api/volcano/callback?device_id=xxx
                                        │
                                   ┌────▼────┐
                                   │ 签名验证  │
                                   │ Base64解码│
                                   │ 二进制解析│
                                   └────┬────┘
                                        │ SubtitleEntry[]
                                   ┌────▼────┐
                                   │ 组装器    │ 缓冲 → 拼接 → 按轮次刷新
                                   └────┬────┘
                                        │ (user_text, bot_text)
                              ┌─────────┼─────────┐
                              ▼                     ▼
                    conversation_messages     后台线程: extract_and_save()
                    (存储到 SQLite)           (提取记忆 → memories 表)
                              │                     │
                              ▼                     ▼
                    GET /api/device/xxx/     GET /api/device/xxx/
                        history                 memories
```

## 关键文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `memory_agent/config.py` | 修改 | 添加 volcano_* 配置项 |
| `memory_agent/types.py` | 修改 | 添加 SubtitleEntry 类型 |
| `memory_agent/volcano/__init__.py` | 新建 | 包初始化 |
| `memory_agent/volcano/decoder.py` | 新建 | Base64 + 二进制解码 + 签名验证 |
| `memory_agent/volcano/assembler.py` | 新建 | 字幕片段组装为完整对话 |
| `memory_agent/store/sqlite.py` | 修改 | 新增 subtitle_fragments 表 + 线程锁 |
| `memory_agent/web/server.py` | 修改 | 新增回调端点 + 设备查询 API + 初始化扩展 |

## 验证方式

1. **解码器单元测试**：构造 Base64 编码的字幕数据，验证解码正确性
2. **组装器测试**：模拟多个片段输入，验证拼接和轮次刷新逻辑
3. **集成测试**：用 curl 发送模拟回调到 `/api/volcano/callback`，然后查询 `/api/device/xxx/history` 和 `/api/device/xxx/memories` 验证数据链路
4. **现有功能回归**：确认 `/api/chat`、`/api/memory` 等原有端点不受影响
