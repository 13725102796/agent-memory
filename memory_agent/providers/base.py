"""Provider 抽象基类 — 参考 OpenClaw EmbeddingProvider 模式"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Embedding 提供者接口"""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """单条文本 → 向量"""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量文本 → 向量列表"""
        ...


class LLMProvider(ABC):
    """LLM 提供者接口"""

    @abstractmethod
    def chat(self, system_prompt: str, user_message: str) -> str:
        """主对话调用"""
        ...

    def chat_stream(self, system_prompt: str, user_message: str):
        """流式对话，yield 文本片段。默认回退到非流式。"""
        yield self.chat(system_prompt, user_message)

    @abstractmethod
    def cheap(self, prompt: str) -> str:
        """低成本调用（记忆提取/合并等）"""
        ...
