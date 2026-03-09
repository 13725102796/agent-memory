"""
将聊天记录 JSON 导入 memory_agent 三层记忆系统（无 LLM 版本）。

流程：
1. 读取聊天记录 JSON，按 session_id + 时间排序
2. 将同一轮中连续的智能体回复合并为一条
3. 配对用户消息和 AI 回复，形成对话轮次
4. 对每个有效对话轮次：
   a. 将 "用户: xxx | AI: xxx" 作为记忆内容
   b. 用 embedding 向量化，去重后存入 memories 表
   c. 同步 FTS 全文索引
5. 对每个 session 生成 Memory Pack（文本截取摘要 + embedding）
6. 运行 MemoryLifecycle 做降级/清理

用法：
    python import_chat_history.py <json_file> [--user-id <user_id>] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime
from itertools import groupby

import numpy as np

# 添加项目路径
sys.path.insert(0, ".")

from memory_agent.config import settings
from memory_agent.memory.lifecycle import MemoryLifecycle
from memory_agent.providers.embedding_local import LocalEmbeddingProvider
from memory_agent.store.sqlite import SQLiteMemoryStore
from memory_agent.types import MemoryPack, MemoryRecord


def load_chat_history(json_path: str) -> list[dict]:
    """读取聊天记录 JSON"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    data.sort(key=lambda x: (x["session_id"], x["created_at"]))
    return data


def group_into_turns(records: list[dict]) -> list[dict]:
    """将原始记录按 session 分组，合并连续回复，配对为对话轮次。"""
    all_turns = []
    for session_id, session_group in groupby(records, key=lambda x: x["session_id"]):
        session_records = list(session_group)
        merged = _merge_consecutive(session_records)
        turns = _pair_user_ai(merged, session_id)
        all_turns.extend(turns)
    return all_turns


def _merge_consecutive(records: list[dict]) -> list[dict]:
    """合并连续的同类型消息"""
    if not records:
        return []
    merged = []
    current = {
        "chat_type": records[0]["chat_type"],
        "content": records[0]["content"],
        "created_at": records[0]["created_at"],
        "session_id": records[0]["session_id"],
    }
    for rec in records[1:]:
        if rec["chat_type"] == current["chat_type"]:
            current["content"] += rec["content"]
        else:
            merged.append(current)
            current = {
                "chat_type": rec["chat_type"],
                "content": rec["content"],
                "created_at": rec["created_at"],
                "session_id": rec["session_id"],
            }
    merged.append(current)
    return merged


def _pair_user_ai(merged: list[dict], session_id: str) -> list[dict]:
    """配对为 (user, ai) 轮次"""
    turns = []
    i = 0
    while i < len(merged):
        if merged[i]["chat_type"] == 1:
            user_msg = merged[i]["content"]
            ts_str = merged[i]["created_at"]
            ts = _parse_ts(ts_str)
            ai_reply = ""
            if i + 1 < len(merged) and merged[i + 1]["chat_type"] == 2:
                ai_reply = merged[i + 1]["content"]
                i += 2
            else:
                i += 1
            if user_msg.strip():
                turns.append({
                    "session_id": session_id,
                    "user_msg": user_msg.strip(),
                    "ai_reply": ai_reply.strip(),
                    "timestamp": ts,
                    "created_at": ts_str,
                })
        else:
            i += 1
    return turns


def _parse_ts(ts_str: str) -> float:
    """将 '2025-12-03 17:45:44.000' 转为 Unix timestamp"""
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def estimate_importance(user_msg: str, ai_reply: str) -> float:
    """基于简单规则估算记忆重要性"""
    importance = 0.4

    # 较长的对话通常包含更多信息
    total_len = len(user_msg) + len(ai_reply)
    if total_len > 100:
        importance += 0.1
    if total_len > 200:
        importance += 0.1

    # 包含事实性内容（数字、英文单词等）
    if re.search(r'\d+', user_msg + ai_reply):
        importance += 0.05
    if re.search(r'[a-zA-Z]{3,}', user_msg + ai_reply):
        importance += 0.05

    # 偏好/习惯类关键词
    preference_keywords = ["喜欢", "偏好", "习惯", "总是", "一直", "调到", "设置", "东北", "广东"]
    for kw in preference_keywords:
        if kw in user_msg or kw in ai_reply:
            importance += 0.1
            break

    # 知识性内容
    knowledge_keywords = ["英文", "中文", "意思", "怎么说", "什么是", "天气", "温度"]
    for kw in knowledge_keywords:
        if kw in user_msg or kw in ai_reply:
            importance += 0.1
            break

    return min(importance, 1.0)


def build_memory_content(user_msg: str, ai_reply: str) -> str:
    """构建存入 memories 表的内容"""
    if ai_reply:
        return f"用户: {user_msg[:200]} | AI: {ai_reply[:300]}"
    return f"用户: {user_msg[:200]}"


def build_pack_summary(turns: list[dict]) -> str:
    """不依赖 LLM，从对话轮次中提取摘要"""
    parts = []
    for turn in turns:
        u = turn["user_msg"][:60]
        a = turn["ai_reply"][:60] if turn["ai_reply"] else ""
        if a:
            parts.append(f"用户问「{u}」，AI回答了「{a}」")
        else:
            parts.append(f"用户说「{u}」")
    summary = "；".join(parts)
    return summary[:settings.pack_summary_max_chars]


def extract_keywords(turns: list[dict]) -> list[str]:
    """从对话中提取关键词（简单分词 + 去停用词）"""
    stopwords = frozenset(
        "的了是在我你他她它这那有不也就都和与或但用来去到说"
        "会能要把被让给比从对为着过很已还可以吗呢吧啊呀哦嘛啦耶呐喔嘞"
    )
    text = " ".join(t["user_msg"] + " " + t["ai_reply"] for t in turns)
    tokens = re.findall(r'[\w\u4e00-\u9fff]{2,}', text)
    # 统计词频
    freq = {}
    for t in tokens:
        if t not in stopwords and len(t) >= 2:
            freq[t] = freq.get(t, 0) + 1
    # 按频率排序取 top
    sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_kw[:10]]


def main():
    parser = argparse.ArgumentParser(description="导入聊天记录到记忆系统")
    parser.add_argument("json_file", help="聊天记录 JSON 文件路径")
    parser.add_argument("--user-id", default=None, help="用户 ID（默认使用 mac_address）")
    parser.add_argument("--dry-run", action="store_true", help="仅分析不实际写入")
    args = parser.parse_args()

    # ── 加载数据 ──
    print(f"加载聊天记录: {args.json_file}")
    records = load_chat_history(args.json_file)
    print(f"  原始记录: {len(records)} 条")

    if args.user_id:
        user_id = args.user_id
    else:
        user_id = records[0]["mac_address"] if records else settings.default_user_id
    print(f"  用户 ID: {user_id}")

    # ── 解析对话轮次 ──
    turns = group_into_turns(records)
    print(f"  有效对话轮次: {len(turns)} 轮")

    session_turns = {}
    for turn in turns:
        session_turns.setdefault(turn["session_id"], []).append(turn)
    print(f"  会话数: {len(session_turns)}")
    for sid, st in session_turns.items():
        print(f"    {sid[:16]}... : {len(st)} 轮")

    if args.dry_run:
        print("\n[DRY RUN] 仅展示解析结果，不写入数据库")
        for i, turn in enumerate(turns):
            content = build_memory_content(turn["user_msg"], turn["ai_reply"])
            imp = estimate_importance(turn["user_msg"], turn["ai_reply"])
            print(f"\n  轮次 {i+1} | importance={imp:.2f}")
            print(f"    {content[:100]}")
        return

    # ── 初始化组件 ──
    print("\n初始化记忆系统...")
    store = SQLiteMemoryStore()
    store.init()
    embedder = LocalEmbeddingProvider()
    lifecycle = MemoryLifecycle(store)

    stats_before = store.get_stats(user_id)
    print(f"  导入前: Core={stats_before.core_length}字, "
          f"Active={stats_before.active_count}, Inactive={stats_before.inactive_count}, "
          f"Packs={stats_before.pack_count}")

    # ── 逐轮存储记忆 ──
    print(f"\n存储记忆 ({len(turns)} 轮)...")
    inserted = 0
    skipped_no_reply = 0
    skipped_dup = 0

    # 预加载已有记忆用于去重
    existing = store.get_all_memories(user_id)
    existing_vecs = [
        (rec.id, rec.embedding) for rec in existing if rec.embedding is not None
    ]

    for i, turn in enumerate(turns):
        user_msg = turn["user_msg"]
        ai_reply = turn["ai_reply"]

        if not ai_reply:
            skipped_no_reply += 1
            continue

        content = build_memory_content(user_msg, ai_reply)
        importance = estimate_importance(user_msg, ai_reply)
        vec = embedder.embed(content)

        # 去重：与已有记忆对比余弦相似度
        is_dup = False
        for _, existing_vec in existing_vecs:
            sim = float(np.dot(vec, existing_vec))
            if sim > settings.duplicate_threshold:
                is_dup = True
                break

        if is_dup:
            skipped_dup += 1
            print(f"  [{i+1}/{len(turns)}] 重复，跳过: {user_msg[:40]}")
            continue

        record = MemoryRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            embedding=vec,
            tier="active",
            importance=importance,
        )
        memory_id = store.insert_memory(record)
        store.fts_sync(memory_id, content)

        # 加入去重池
        existing_vecs.append((memory_id, vec))
        inserted += 1
        print(f"  [{i+1}/{len(turns)}] 已存储 (imp={importance:.2f}): {user_msg[:50]}")

    print(f"\n  存储完成: 新增 {inserted} 条, 跳过无回复 {skipped_no_reply}, 跳过重复 {skipped_dup}")

    # ── 按 session 生成 Memory Pack ──
    print(f"\n生成 Memory Pack...")
    pack_count = 0
    for sid, st in session_turns.items():
        if len(st) < 3:
            print(f"  Session {sid[:16]}... 轮次过少({len(st)})，跳过")
            continue

        summary = build_pack_summary(st)
        keywords = extract_keywords(st)
        topic = f"session:{sid[:8]}"

        # 用摘要生成 embedding
        vec = embedder.embed(summary)

        # 链式关联
        latest = store.get_latest_pack(user_id)
        prev_pack_id = latest.id if latest else None
        prev_context = latest.summary[-settings.pack_prev_context_chars:] if latest else ""

        pack = MemoryPack(
            id=str(uuid.uuid4()),
            user_id=user_id,
            summary=summary,
            keywords=keywords,
            topic=topic,
            embedding=vec,
            prev_pack_id=prev_pack_id,
            prev_context=prev_context,
            turn_count=len(st),
        )
        store.insert_pack(pack)

        # 回填 pack_id 到时间范围内的记忆
        start_ts = datetime.strptime(st[0]["created_at"], "%Y-%m-%d %H:%M:%S.%f").isoformat()
        end_ts = datetime.strptime(st[-1]["created_at"], "%Y-%m-%d %H:%M:%S.%f").isoformat()
        linked = store.link_memories_to_pack(user_id, start_ts, end_ts, pack.id)

        print(f"  Session {sid[:16]}... Pack 已存储 ({len(st)} 轮, 关键词: {', '.join(keywords[:5])}, 关联 {linked} 条记忆)")
        pack_count += 1

    # ── 生命周期维护 ──
    print("\n运行生命周期维护...")
    lifecycle.run(user_id)

    # ── 最终统计 ──
    stats_after = store.get_stats(user_id)
    print(f"\n导入完成!")
    print(f"  Core Memory:    {stats_after.core_length} 字")
    print(f"  Active Memory:  {stats_after.active_count} 条")
    print(f"  Inactive Memory:{stats_after.inactive_count} 条")
    print(f"  Memory Packs:   {stats_after.pack_count} 个")
    print(f"  数据库路径:      {settings.db_path}")


if __name__ == "__main__":
    main()
