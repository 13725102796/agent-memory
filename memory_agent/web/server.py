"""Web 聊天服务 — 零依赖，基于标准库 http.server"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler

from memory_agent.config import settings
from memory_agent.core.chat import ChatHandler
from memory_agent.log import get_logger
from memory_agent.memory.extract import MemoryExtractor
from memory_agent.memory.lifecycle import MemoryLifecycle
from memory_agent.providers.embedding_local import LocalEmbeddingProvider
from memory_agent.providers.llm_claude_cli import ClaudeCLIProvider
from memory_agent.providers.reranker_local import LocalRerankerProvider
from memory_agent.store.sqlite import SQLiteMemoryStore

log = get_logger("web.server")

# 导入历史文件的默认路径
_HISTORY_JSON = os.path.realpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "ai_agent_chat_history_20_6e_f1_ba_55_f8.json",
))

_HTML_DIR = os.path.dirname(os.path.abspath(__file__))

# 设备 API 路由正则
_DEVICE_RE = re.compile(r'^/api/device/([^/]+)/(history|memories|stats|core)$')
_VOLCANO_RE = re.compile(r'^/api/volcano/callback')


def _create_app():
    store = SQLiteMemoryStore()
    store.init()
    embedder = LocalEmbeddingProvider()
    embedder._load()  # 启动时预加载模型
    reranker = LocalRerankerProvider()
    reranker._load()  # 预加载 reranker 模型
    llm = ClaudeCLIProvider()
    handler = ChatHandler(store=store, llm=llm, embedder=embedder, reranker=reranker)
    user_id = settings.default_user_id
    handler.load_history(user_id)
    lifecycle = MemoryLifecycle(store=store)
    lifecycle.run(user_id)

    # 火山引擎回调组件（按需初始化）
    assembler = None
    extractor = MemoryExtractor(store, llm, embedder)
    if settings.volcano_enabled:
        from memory_agent.volcano.assembler import SubtitleAssembler
        assembler = SubtitleAssembler(flush_timeout_sec=settings.volcano_flush_timeout_sec)
        log.info("火山引擎字幕回调已启用")

        # 定时刷新超时缓冲区
        def _flush_loop():
            while True:
                time.sleep(settings.volcano_flush_timeout_sec)
                try:
                    flushed = assembler.flush_inactive()
                    for device_id, user_text, bot_text in flushed:
                        _bg_extract_device(store, extractor, device_id, user_text, bot_text)
                    assembler.cleanup_stale_devices()
                except Exception:
                    log.exception("定时刷新异常")
        threading.Thread(target=_flush_loop, daemon=True).start()

    return handler, store, user_id, extractor, assembler


def _bg_extract_device(
    store: SQLiteMemoryStore,
    extractor: MemoryExtractor,
    device_id: str,
    user_text: str,
    bot_text: str,
):
    """后台线程：存储对话消息并提取记忆"""
    def _run():
        try:
            now = time.time()
            if user_text:
                store.append_message(device_id, "user", user_text, now)
            if bot_text:
                store.append_message(device_id, "assistant", bot_text, now)
            if user_text and bot_text:
                extractor.extract_and_save(device_id, user_text, bot_text)
                log.info("设备记忆提取完成: device=%s", device_id)
        except Exception:
            log.exception("设备记忆提取失败: device=%s", device_id)
    threading.Thread(target=_run, daemon=True).start()


class ChatRequestHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理"""

    def __init__(
        self,
        chat: ChatHandler,
        store: SQLiteMemoryStore,
        user_id: str,
        extractor: MemoryExtractor,
        assembler,
        *args, **kwargs,
    ):
        self._chat = chat
        self._store = store
        self._user_id = user_id
        self._extractor = extractor
        self._assembler = assembler
        super().__init__(*args, **kwargs)

    # 不打印每个请求的日志到终端
    def log_message(self, format, *args):
        pass

    # ── 路由 ──────────────────────────────────────────────

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/stats":
            self._handle_stats()
        elif path == "/api/memory":
            self._handle_memory()
        elif path == "/api/packs":
            self._handle_packs()
        elif path == "/api/core":
            self._handle_core()
        elif path == "/api/history":
            self._handle_get_history()
        else:
            # 设备 API 路由
            m = _DEVICE_RE.match(path)
            if m:
                device_id, resource = m.group(1), m.group(2)
                self._handle_device_api(device_id, resource)
            else:
                self._json_response({"error": "Not Found"}, 404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/chat":
            self._handle_chat()
        elif path == "/api/clear":
            self._handle_clear()
        elif path == "/api/import-history":
            self._handle_import_history()
        elif _VOLCANO_RE.match(path):
            self._handle_volcano_callback()
        else:
            self._json_response({"error": "Not Found"}, 404)

    # ── API 处理 ──────────────────────────────────────────

    def _handle_chat(self):
        body = self._read_body()
        if body is None:
            return
        message = body.get("message", "").strip()
        if not message:
            self._json_response({"error": "message is required"}, 400)
            return

        # SSE 流式返回
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.close_connection = True  # 流结束后关闭连接

        try:
            for chunk in self._chat.handle_stream(self._user_id, message):
                event = json.dumps({"chunk": chunk}, ensure_ascii=False)
                self.wfile.write(f"data: {event}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: {\"done\": true}\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            log.warning("客户端断开连接")

    def _handle_stats(self):
        stats = self._store.get_stats(self._user_id)
        self._json_response({
            "core_length": stats.core_length,
            "active_count": stats.active_count,
            "inactive_count": stats.inactive_count,
            "pack_count": stats.pack_count,
        })

    def _handle_memory(self):
        memories = self._store.get_all_memories(self._user_id)
        items = []
        for m in memories:
            items.append({
                "id": m.id,
                "content": m.content,
                "tier": m.tier,
                "importance": m.importance,
                "hit_count": m.hit_count,
                "pack_id": m.pack_id,
            })
        self._json_response({"memories": items})

    def _handle_packs(self):
        packs = self._store.get_packs(self._user_id)
        items = []
        for p in packs:
            items.append({
                "id": p.id,
                "summary": p.summary,
                "topic": p.topic,
                "keywords": p.keywords,
                "turn_count": p.turn_count,
                "created_at": p.created_at,
            })
        self._json_response({"packs": items})

    def _handle_core(self):
        core = self._store.get_core_memory(self._user_id)
        self._json_response({"core": core})

    def _handle_clear(self):
        self._store.set_core_memory(self._user_id, "")
        for m in self._store.get_all_memories(self._user_id):
            self._store.delete_memory(m.id)
        pack_count = self._store.delete_packs(self._user_id)
        # 同时清空内存中的对话历史
        self._chat._history.clear()
        self._json_response({"cleared": True, "packs_deleted": pack_count})

    def _handle_get_history(self):
        """返回当前内存中的对话历史"""
        messages = []
        for turn in self._chat._history:
            messages.append({
                "role": "user" if turn["role"] == "user" else "ai",
                "content": turn["content"],
            })
        self._json_response({"messages": messages})

    def _handle_import_history(self):
        """从 JSON 文件导入聊天记录到对话历史 + 后台提取记忆"""
        body = self._read_body()
        file_path = (body or {}).get("file_path", _HISTORY_JSON)

        if not os.path.exists(file_path):
            self._json_response({"error": f"文件不存在: {file_path}"}, 400)
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            self._json_response({"error": f"读取文件失败: {e}"}, 400)
            return

        # 按时间排序
        records.sort(key=lambda r: r.get("created_at", ""))

        # 转换为对话历史格式并注入
        messages = []
        pairs = []  # (user_msg, ai_msg) 用于后台提取记忆
        pending_user_msg = None

        for rec in records:
            chat_type = rec.get("chat_type")
            content = rec.get("content", "").strip()
            if not content:
                continue

            if chat_type == 1:  # 用户
                role = "user"
                pending_user_msg = content
            elif chat_type == 2:  # 智能体
                role = "assistant"
                if pending_user_msg:
                    pairs.append((pending_user_msg, content))
                    pending_user_msg = None
            else:
                continue

            self._chat._history.append({
                "role": role,
                "content": content,
                "ts": time.time(),
                "packed": True,  # 标记为已处理，packer 跳过
            })
            messages.append({
                "role": "user" if role == "user" else "ai",
                "content": content,
            })

        log.info("导入 %d 条聊天记录，%d 组对话对", len(messages), len(pairs))

        # 后台提取记忆（不阻塞响应）
        if pairs:
            def _bg_extract():
                extractor = self._chat._extractor
                for i, (u, a) in enumerate(pairs):
                    try:
                        extractor.extract_and_save(self._user_id, u, a)
                        log.info("记忆提取进度: %d/%d", i + 1, len(pairs))
                    except Exception:
                        log.exception("记忆提取失败: %s", u[:50])
            threading.Thread(target=_bg_extract, daemon=True).start()

        self._json_response({
            "imported": len(messages),
            "pairs": len(pairs),
            "messages": messages,
        })

    # ── 火山引擎字幕回调 ──────────────────────────────────

    def _handle_volcano_callback(self):
        """接收火山引擎字幕回调"""
        from memory_agent.volcano.decoder import decode_subtitle_message, verify_signature

        if self._assembler is None:
            self._json_response({"error": "volcano not enabled"}, 503)
            return

        # 提取 device_id
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        device_id = params.get("device_id", [None])[0]
        if not device_id:
            self._json_response({"error": "device_id required"}, 400)
            return

        body = self._read_body()
        if body is None:
            return

        # 签名验证
        signature = body.get("signature", "")
        if not verify_signature(signature, settings.volcano_signature):
            log.warning("签名验证失败: device=%s", device_id)
            self._json_response({"ok": True})  # 静默拒绝
            return

        # 解码字幕消息
        entries = decode_subtitle_message(body)
        if entries is None:
            self._json_response({"ok": True})
            return

        # 存储原始片段（调试/审计）
        bot_id = settings.volcano_default_bot_id
        for entry in entries:
            speaker = "bot" if entry.userId == bot_id else "user"
            self._store.insert_fragment(
                device_id=device_id,
                round_id=entry.roundId,
                sequence=entry.sequence,
                speaker=speaker,
                text=entry.text,
                definite=entry.definite,
                paragraph=entry.paragraph,
            )

        # 组装完整对话轮次
        completed = self._assembler.process(device_id, entries, bot_id)
        for user_text, bot_text in completed:
            _bg_extract_device(self._store, self._extractor, device_id, user_text, bot_text)

        self._json_response({"ok": True, "completed_turns": len(completed)})

    # ── 设备 API ──────────────────────────────────────────

    def _handle_device_api(self, device_id: str, resource: str):
        """处理 /api/device/<device_id>/<resource> 请求"""
        if resource == "history":
            messages = self._store.get_recent_messages(device_id, limit=100)
            items = [{"role": m["role"], "content": m["content"], "ts": m["ts"]} for m in messages]
            self._json_response({"device_id": device_id, "messages": items})

        elif resource == "memories":
            memories = self._store.get_all_memories(device_id)
            items = [{
                "id": m.id, "content": m.content, "tier": m.tier,
                "importance": m.importance, "hit_count": m.hit_count,
                "pack_id": m.pack_id,
            } for m in memories]
            self._json_response({"device_id": device_id, "memories": items})

        elif resource == "stats":
            stats = self._store.get_stats(device_id)
            self._json_response({
                "device_id": device_id,
                "core_length": stats.core_length,
                "active_count": stats.active_count,
                "inactive_count": stats.inactive_count,
                "pack_count": stats.pack_count,
            })

        elif resource == "core":
            core = self._store.get_core_memory(device_id)
            self._json_response({"device_id": device_id, "core": core})

        else:
            self._json_response({"error": "Not Found"}, 404)

    # ── 静态文件 ──────────────────────────────────────────

    def _serve_html(self):
        html_path = os.path.join(_HTML_DIR, "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    # ── 工具方法 ──────────────────────────────────────────

    def _read_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._json_response({"error": "Empty body"}, 400)
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON"}, 400)
            return None

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(port: int = 8899):
    print(f"初始化记忆系统 ...")
    chat, store, user_id, extractor, assembler = _create_app()
    handler_class = partial(ChatRequestHandler, chat, store, user_id, extractor, assembler)
    server = HTTPServer(("0.0.0.0", port), handler_class)
    print(f"聊天服务已启动: http://localhost:{port}")
    if settings.volcano_enabled:
        print(f"火山引擎回调端点: POST /api/volcano/callback?device_id=xxx")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
