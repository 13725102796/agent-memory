#!/usr/bin/env python3
"""诊断脚本：逐步测试记忆提取流程，找出断点"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_agent.providers.llm_claude_cli import ClaudeCLIProvider
from memory_agent.memory.extract import MemoryExtractor, _parse_json
from memory_agent.providers.embedding_local import LocalEmbeddingProvider
from memory_agent.store.sqlite import SQLiteMemoryStore


def main():
    print("=" * 50)
    print("记忆提取流程诊断")
    print("=" * 50)

    # ── Step 1: 测试 cheap 模型基础调用 ──
    print("\n[Step 1] 测试 cheap 模型基础调用...")
    llm = ClaudeCLIProvider()
    print(f"  模型: {llm._cheap_model}")
    print(f"  CLI路径: {llm._cli_path}")

    raw = llm.cheap('请直接输出这个 JSON，不要任何其他内容：{"test": true}')
    print(f"  原始输出 ({len(raw)} 字符): {repr(raw[:300])}")

    parsed = _parse_json(raw)
    if parsed:
        print(f"  JSON 解析: 成功 -> {parsed}")
    else:
        print(f"  JSON 解析: 失败!")
        print(f"  这就是记忆无法保存的原因 — cheap 模型返回的内容无法解析为 JSON")
        return

    # ── Step 2: 测试记忆提取 prompt ──
    print("\n[Step 2] 测试记忆提取 prompt...")
    extract_prompt = """从这段对话中提取值得长期记住的信息。

分两类输出 JSON（严格 JSON 格式，不要 markdown 代码块）：
{
  "core": "持久性偏好/身份信息，如有则填写，无则为null",
  "memories": [
    {"content": "具体事实/决策/方案", "importance": 0.5}
  ]
}

分类标准：
- core: 姓名、语言偏好、技术栈、代码风格、长期规则（如"以后注释用英文"）
- memories: 具体方案、决策、讨论结论、项目细节
- 闲聊/问候/临时指令 → 不提取，memories 为空数组，core 为 null

对话：
用户: 我叫小张，以后请叫我小张
助手: 好的小张，我记住了！以后就叫你小张。

只输出 JSON，不要其他内容："""

    raw2 = llm.cheap(extract_prompt)
    print(f"  原始输出 ({len(raw2)} 字符): {repr(raw2[:500])}")

    parsed2 = _parse_json(raw2)
    if parsed2:
        print(f"  JSON 解析: 成功")
        print(f"  core = {parsed2.get('core')}")
        print(f"  memories = {parsed2.get('memories')}")
    else:
        print(f"  JSON 解析: 失败!")
        print(f"  提取 prompt 返回内容无法解析 — 这是问题所在")
        return

    # ── Step 3: 测试 embedding ──
    print("\n[Step 3] 测试 embedding 生成...")
    embedder = LocalEmbeddingProvider()
    vec = embedder.embed("测试文本")
    print(f"  向量维度: {vec.shape}")
    print(f"  向量范数: {float((vec**2).sum()**0.5):.4f}")
    print(f"  Embedding: 正常")

    # ── Step 4: 测试完整提取+存储流程 ──
    print("\n[Step 4] 测试完整 extract_and_save 流程...")
    store = SQLiteMemoryStore()
    store.init()

    before_count = len(store.get_all_memories("default-user"))
    core_before = store.get_core_memory("default-user")
    print(f"  提取前: 记忆数={before_count}, 核心记忆={repr(core_before[:80] if core_before else '')}")

    extractor = MemoryExtractor(store, llm, embedder)
    extractor.extract_and_save(
        "default-user",
        "我叫小张，以后请叫我小张",
        "好的小张，我记住了！以后就叫你小张。",
    )

    after_count = len(store.get_all_memories("default-user"))
    core_after = store.get_core_memory("default-user")
    print(f"  提取后: 记忆数={after_count}, 核心记忆={repr(core_after[:80] if core_after else '')}")

    if after_count > before_count:
        print("  -> 新记忆已写入数据库")
    else:
        print("  -> 没有新记忆写入（可能被去重跳过）")

    if core_after != core_before:
        print("  -> 核心记忆已更新")
    else:
        print("  -> 核心记忆未变化")

    print("\n" + "=" * 50)
    print("诊断完成")


if __name__ == "__main__":
    main()
