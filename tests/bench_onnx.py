"""ONNX Runtime vs PyTorch INT8 Benchmark"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


# ── 测试数据 ──
QUERIES = [
    "你好帮我播放一首歌",
    "我男朋友是程序员",
    "你叫什么名字",
    "讲个笑话给我听",
    "有人在摸摸你的脑袋",
]

MEMORIES = [
    "用户喜欢听财神到这首歌，经常让AI播放",
    "用户的男朋友是程序员，她想送键盘给他",
    "AI的名字叫卡卡，用户经常问AI叫什么",
    "用户喜欢听笑话，AI经常给用户讲笑话",
    "用户喜欢摸AI的脑袋，AI会撒娇回应",
    "用户正在学英语单词，AI会帮忙出题",
    "用户今天心情不好，AI安慰了用户",
]


def bench_pytorch_int8():
    """PyTorch INT8 基线"""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.backends.quantized.engine = "qnnpack"
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    model[0].auto_model = torch.quantization.quantize_dynamic(
        model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8,
    )

    # Warmup
    for _ in range(3):
        model.encode("warmup", normalize_embeddings=True)

    # Benchmark
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        model.encode("你好帮我播放一首歌", normalize_embeddings=True)
        times.append((time.perf_counter() - t0) * 1000)

    # 检索测试
    mem_vecs = model.encode(MEMORIES, normalize_embeddings=True)
    results = {}
    for q in QUERIES:
        qv = model.encode(q, normalize_embeddings=True)
        scores = np.dot(mem_vecs, qv)
        top3_idx = np.argsort(scores)[::-1][:3]
        results[q] = [(MEMORIES[i][:6], float(scores[i])) for i in top3_idx]

    return times, results, mem_vecs


def bench_onnx():
    """ONNX Runtime"""
    from memory_agent.providers.embedding_local import LocalEmbeddingProvider

    provider = LocalEmbeddingProvider()
    provider._load()

    # Warmup
    for _ in range(3):
        provider.embed("warmup")

    # Benchmark
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        provider.embed("你好帮我播放一首歌")
        times.append((time.perf_counter() - t0) * 1000)

    # 检索测试
    mem_vecs = np.array(provider.embed_batch(MEMORIES))
    results = {}
    for q in QUERIES:
        qv = provider.embed(q)
        scores = np.dot(mem_vecs, qv)
        top3_idx = np.argsort(scores)[::-1][:3]
        results[q] = [(MEMORIES[i][:6], float(scores[i])) for i in top3_idx]

    return times, results, mem_vecs


def main():
    print("=" * 60)
    print("ONNX Runtime vs PyTorch INT8 Benchmark")
    print("=" * 60)

    # PyTorch INT8
    print("\n[1/2] PyTorch INT8 ...")
    pt_times, pt_results, pt_vecs = bench_pytorch_int8()

    # ONNX Runtime
    print("\n[2/2] ONNX Runtime ...")
    onnx_times, onnx_results, onnx_vecs = bench_onnx()

    # ── 性能对比 ──
    pt_avg = np.mean(pt_times)
    onnx_avg = np.mean(onnx_times)
    speedup = (pt_avg - onnx_avg) / pt_avg * 100

    print("\n" + "=" * 60)
    print("性能对比 (30 次平均)")
    print("=" * 60)
    print(f"  PyTorch INT8: {pt_avg:.1f}ms (range: {min(pt_times):.1f}-{max(pt_times):.1f}ms)")
    print(f"  ONNX Runtime: {onnx_avg:.1f}ms (range: {min(onnx_times):.1f}-{max(onnx_times):.1f}ms)")
    print(f"  加速: {speedup:+.1f}%")

    # ── Embedding 相似度 ──
    print("\n" + "=" * 60)
    print("Embedding 余弦相似度 (ONNX vs PyTorch INT8)")
    print("=" * 60)
    for i, mem in enumerate(MEMORIES):
        sim = float(np.dot(pt_vecs[i], onnx_vecs[i]))
        print(f"  {mem[:20]:20s}  cos_sim={sim:.6f}")

    # ── 检索对比 ──
    print("\n" + "=" * 60)
    print("检索结果对比")
    print("=" * 60)
    top1_match = 0
    top3_match = 0
    for q in QUERIES:
        pt_top = [x[0] for x in pt_results[q]]
        onnx_top = [x[0] for x in onnx_results[q]]
        t1 = "✓" if pt_top[0] == onnx_top[0] else "✗"
        t3 = "✓" if pt_top == onnx_top else "✗"
        if pt_top[0] == onnx_top[0]:
            top1_match += 1
        if pt_top == onnx_top:
            top3_match += 1
        print(f"\nQ: {q}")
        pt_scores = [f"{x[0]}({x[1]:.3f})" for x in pt_results[q]]
        onnx_scores = [f"{x[0]}({x[1]:.3f})" for x in onnx_results[q]]
        print(f"  PT INT8: {pt_scores}")
        print(f"  ONNX:    {onnx_scores}")
        print(f"  Top1: {t1} | Top3: {t3}")

    print(f"\nTop-1 一致率: {top1_match}/{len(QUERIES)}")
    print(f"Top-3 一致率: {top3_match}/{len(QUERIES)}")


if __name__ == "__main__":
    main()
