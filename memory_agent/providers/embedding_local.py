"""本地 Embedding — ONNX Runtime 推理（比 PyTorch INT8 更快）"""
from __future__ import annotations

import os

import numpy as np

from memory_agent.config import settings
from memory_agent.log import get_logger
from memory_agent.providers.base import EmbeddingProvider

log = get_logger("providers.embedding")

# ONNX 模型缓存目录
_ONNX_CACHE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "onnx")
)


class LocalEmbeddingProvider(EmbeddingProvider):
    """基于 ONNX Runtime 的本地 Embedding"""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []

    def _load(self):
        if self._session is not None:
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        # 解析 HuggingFace 模型全名
        model_id = self._model_name
        if "/" not in model_id:
            model_id = f"sentence-transformers/{model_id}"

        onnx_dir = os.path.join(_ONNX_CACHE, self._model_name.replace("/", "_"))
        onnx_file = os.path.join(onnx_dir, "model.onnx")

        # 首次运行：导出 ONNX 并缓存（含 tokenizer）
        if not os.path.exists(onnx_file):
            self._export_onnx(model_id, onnx_dir)

        # 加载 ONNX Runtime Session（线程数=1，适合多 Pod 部署）
        log.info("加载 ONNX Runtime 模型 ...")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(onnx_file, sess_opts)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        log.info(
            "模型加载完成 (ONNX Runtime, threads=1, inputs=%s)",
            self._input_names,
        )

    @staticmethod
    def _export_onnx(model_id: str, onnx_dir: str) -> None:
        """一次性将 PyTorch 模型导出为 ONNX 格式"""
        log.info("首次导出 ONNX 模型: %s ...", model_id)
        os.makedirs(onnx_dir, exist_ok=True)

        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        model.save_pretrained(onnx_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(onnx_dir)

        del model
        log.info("ONNX 模型已保存到: %s", onnx_dir)

    def _build_feed(self, inputs: dict) -> dict:
        """构建 ONNX 输入，自动补齐缺失的 token_type_ids"""
        feed = {k: v for k, v in inputs.items() if k in self._input_names}
        if "token_type_ids" in self._input_names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(inputs["input_ids"])
        return feed

    def embed(self, text: str) -> np.ndarray:
        self._load()
        inputs = self._tokenizer(
            text, return_tensors="np",
            padding=True, truncation=True, max_length=128,
        )
        feed = self._build_feed(inputs)
        outputs = self._session.run(None, feed)
        emb = _mean_pooling(outputs[0], inputs["attention_mask"])
        emb = emb / np.linalg.norm(emb)
        return emb.flatten().astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self._load()
        inputs = self._tokenizer(
            texts, return_tensors="np",
            padding=True, truncation=True, max_length=128,
        )
        feed = self._build_feed(inputs)
        outputs = self._session.run(None, feed)
        embs = _mean_pooling(outputs[0], inputs["attention_mask"])
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
        embs = embs / norms
        return [e.astype(np.float32) for e in embs]


def _mean_pooling(
    token_embeddings: np.ndarray, attention_mask: np.ndarray,
) -> np.ndarray:
    """Mean pooling — 对有效 token 取平均"""
    mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    return np.sum(token_embeddings * mask, axis=1) / np.sum(mask, axis=1).clip(min=1e-9)
