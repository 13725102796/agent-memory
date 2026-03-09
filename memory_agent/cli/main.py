"""CLI 交互入口 — 组装依赖，启动交互式命令行"""
from __future__ import annotations

from memory_agent.config import settings
from memory_agent.core.chat import ChatHandler
from memory_agent.memory.lifecycle import MemoryLifecycle
from memory_agent.providers.embedding_local import LocalEmbeddingProvider
from memory_agent.providers.llm_claude_cli import ClaudeCLIProvider
from memory_agent.store.sqlite import SQLiteMemoryStore


def _create_handler() -> tuple[ChatHandler, SQLiteMemoryStore, MemoryLifecycle]:
    """工厂函数：组装所有依赖"""
    store = SQLiteMemoryStore()
    store.init()

    embedder = LocalEmbeddingProvider()
    llm = ClaudeCLIProvider()

    handler = ChatHandler(store=store, llm=llm, embedder=embedder)
    lifecycle = MemoryLifecycle(store=store)

    return handler, store, lifecycle


def _print_banner():
    print("=" * 60)
    print("  三层记忆系统 Demo")
    print("  Core Memory / Active Memory / Inactive Memory")
    print("=" * 60)
    print()
    print("特殊命令:")
    print("  /core    — 查看 Core Memory")
    print("  /memory  — 查看所有记忆")
    print("  /packs   — 查看 Memory Pack")
    print("  /stats   — 统计信息")
    print("  /clear   — 清空所有记忆")
    print("  /quit    — 退出")
    print()


def _cmd_show_core(store: SQLiteMemoryStore, user_id: str):
    core = store.get_core_memory(user_id)
    if core:
        print(f"\n{'─' * 40}")
        print("Core Memory:")
        print(core)
        print(f"{'─' * 40}\n")
    else:
        print("\n  Core Memory 为空\n")


def _cmd_show_memories(store: SQLiteMemoryStore, user_id: str):
    memories = store.get_all_memories(user_id)
    if not memories:
        print("\n  暂无记忆\n")
        return

    print(f"\n{'─' * 60}")
    current_tier = None
    for m in memories:
        if m.tier != current_tier:
            current_tier = m.tier
            label = "Active" if current_tier == "active" else "Inactive"
            print(f"\n  [{label} Memory]")
        hits = f"命中{m.hit_count}次" if m.hit_count > 0 else "未命中"
        print(f"    importance={m.importance:.2f} | {hits} | {m.content[:60]}")
    print(f"{'─' * 60}\n")


def _cmd_show_packs(store: SQLiteMemoryStore, user_id: str):
    packs = store.get_packs(user_id)
    if not packs:
        print("\n  暂无 Memory Pack\n")
        return

    print(f"\n{'─' * 60}")
    print(f"  Memory Packs ({len(packs)} 个):")
    for p in packs:
        kw = ", ".join(p.keywords[:5]) if p.keywords else ""
        print(f"\n  [{p.topic}]")
        print(f"    轮次={p.turn_count} | 关键词: {kw}")
        print(f"    {p.summary[:80]}...")
    print(f"{'─' * 60}\n")


def _cmd_show_stats(store: SQLiteMemoryStore, user_id: str):
    stats = store.get_stats(user_id)
    print(f"\n  Core Memory: {'有' if stats.core_length > 0 else '空'} ({stats.core_length} 字)")
    print(f"  Active:      {stats.active_count} 条")
    print(f"  Inactive:    {stats.inactive_count} 条")
    print(f"  Pack:        {stats.pack_count} 个")
    print(f"  总计:        {stats.active_count + stats.inactive_count} 条记忆 + {stats.pack_count} 个包\n")


def _cmd_clear(store: SQLiteMemoryStore, user_id: str):
    confirm = input("  确认清空所有记忆？(y/N) ").strip().lower()
    if confirm == "y":
        store.set_core_memory(user_id, "")
        for m in store.get_all_memories(user_id):
            store.delete_memory(m.id)
        pack_count = store.delete_packs(user_id)
        msg = "  已清空"
        if pack_count:
            msg += f"（含 {pack_count} 个 Pack）"
        print(f"{msg}\n")
    else:
        print("  取消\n")


def main():
    handler, store, lifecycle = _create_handler()
    user_id = settings.default_user_id

    # 启动时执行生命周期维护
    lifecycle.run(user_id)

    _print_banner()

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 特殊命令
        if user_input == "/quit":
            print("再见！")
            break
        elif user_input == "/core":
            _cmd_show_core(store, user_id)
            continue
        elif user_input == "/memory":
            _cmd_show_memories(store, user_id)
            continue
        elif user_input == "/stats":
            _cmd_show_stats(store, user_id)
            continue
        elif user_input == "/packs":
            _cmd_show_packs(store, user_id)
            continue
        elif user_input == "/clear":
            _cmd_clear(store, user_id)
            continue
        elif user_input.startswith("/"):
            print("  未知命令。可用: /core /memory /packs /stats /clear /quit\n")
            continue

        # 正常对话
        reply = handler.handle(user_id, user_input)
        print(f"\nAI: {reply}\n")


if __name__ == "__main__":
    main()
