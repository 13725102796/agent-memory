"""本地 Reranker — ONNX Runtime 推理（Cross-Encoder 精排）"""
from __future__ import annotations

import os

import numpy as np

from memory_agent.config import settings
from memory_agent.log import get_logger
from memory_agent.providers.base import RerankerProvider

log = get_logger("providers.reranker")

# ONNX 模型缓存目录（与 embedding 共用 models/onnx）
_ONNX_CACHE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "onnx")
)


class LocalRerankerProvider(RerankerProvider):
    """基于 ONNX Runtime 的本地 Cross-Encoder Reranker"""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.reranker_model
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []

    def _load(self):
        if self._session is not None:
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_id = self._model_name
        onnx_dir = os.path.join(_ONNX_CACHE, self._model_name.replace("/", "_"))
        onnx_file = os.path.join(onnx_dir, "model.onnx")

        if not os.path.exists(onnx_file):
            self._export_onnx(model_id, onnx_dir)

        log.info("加载 Reranker ONNX 模型 ...")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(onnx_file, sess_opts, providers=["CPUExecutionProvider"])
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        log.info(
            "Reranker 模型加载完成 (ONNX CPU, threads=4, inputs=%s)",
            self._input_names,
        )

    @staticmethod
    def _export_onnx(model_id: str, onnx_dir: str) -> None:
        """首次将 Cross-Encoder 模型导出为 ONNX 格式"""
        log.info("首次导出 Reranker ONNX 模型: %s ...", model_id)
        os.makedirs(onnx_dir, exist_ok=True)

        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
        model.save_pretrained(onnx_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(onnx_dir)

        del model
        log.info("Reranker ONNX 模型已保存到: %s", onnx_dir)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """对 (query, doc) 对打分，返回 sigmoid 归一化后的相关性分数列表"""
        if not documents:
            return []

        self._load()

        inputs = self._tokenizer(
            [query] * len(documents),
            documents,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=256,
        )

        feed = {k: v for k, v in inputs.items() if k in self._input_names}
        if "token_type_ids" in self._input_names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        outputs = self._session.run(None, feed)
        logits = outputs[0]

        # bge-reranker 输出 shape: (batch, 1) — 取第一列作为 logit
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim == 2:
            logits = logits[:, 0]

        scores = _sigmoid(logits)
        return scores.tolist()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
