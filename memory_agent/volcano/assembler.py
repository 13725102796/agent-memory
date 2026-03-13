"""字幕片段组装器 — 将流式字幕片段拼接为完整对话轮次"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from memory_agent.log import get_logger
from memory_agent.types import SubtitleEntry

log = get_logger("volcano.assembler")


@dataclass
class _RoundState:
    """单个对话轮次的缓冲状态"""
    round_id: int
    user_texts: list[str] = field(default_factory=list)
    bot_texts: list[str] = field(default_factory=list)
    last_active: float = 0.0

    def flush(self) -> tuple[str, str] | None:
        """刷新缓冲区，返回 (user_text, bot_text)；两边都为空时返回 None"""
        user_text = "".join(self.user_texts).strip()
        bot_text = "".join(self.bot_texts).strip()
        self.user_texts.clear()
        self.bot_texts.clear()
        if user_text or bot_text:
            return (user_text, bot_text)
        return None


@dataclass
class _DeviceState:
    """单个设备的组装状态"""
    current_round: _RoundState | None = None
    last_active: float = 0.0


class SubtitleAssembler:
    """将流式字幕片段组装为完整的 (user_text, bot_text) 对话对"""

    def __init__(self, flush_timeout_sec: int = 30):
        self._devices: dict[str, _DeviceState] = {}
        self._lock = threading.Lock()
        self._flush_timeout = flush_timeout_sec

    def process(
        self,
        device_id: str,
        entries: list[SubtitleEntry],
        bot_id: str,
    ) -> list[tuple[str, str]]:
        """处理一批字幕条目，返回已完成的对话对列表

        Args:
            device_id: 设备 ID
            entries: 本次回调的字幕条目
            bot_id: AI Bot 的 userId（用于区分说话人）

        Returns:
            已完成的 [(user_text, bot_text), ...] 列表
        """
        completed: list[tuple[str, str]] = []

        with self._lock:
            state = self._devices.get(device_id)
            if state is None:
                state = _DeviceState()
                self._devices[device_id] = state

            now = time.time()
            state.last_active = now

            for entry in entries:
                # 只处理最终确认的文本
                if not entry.definite:
                    continue

                if not entry.text.strip():
                    continue

                is_bot = (entry.userId == bot_id)

                # 轮次切换：刷新上一轮
                if state.current_round is not None and state.current_round.round_id != entry.roundId:
                    result = state.current_round.flush()
                    if result:
                        completed.append(result)
                        log.info(
                            "轮次切换刷新 device=%s round=%d → %d",
                            device_id, state.current_round.round_id, entry.roundId,
                        )
                    state.current_round = None

                # 初始化当前轮次
                if state.current_round is None:
                    state.current_round = _RoundState(round_id=entry.roundId, last_active=now)

                rnd = state.current_round
                rnd.last_active = now

                # 追加文本到对应缓冲区
                if is_bot:
                    rnd.bot_texts.append(entry.text)
                else:
                    rnd.user_texts.append(entry.text)

                # paragraph=true 且是 bot 回复结束 → 一轮对话完成
                if entry.paragraph and is_bot:
                    result = rnd.flush()
                    if result:
                        completed.append(result)
                        log.info(
                            "段落结束刷新 device=%s round=%d",
                            device_id, entry.roundId,
                        )

        return completed

    def flush_inactive(self, timeout_sec: int | None = None) -> list[tuple[str, str, str]]:
        """刷新超时的设备缓冲区

        Returns:
            [(device_id, user_text, bot_text), ...] 超时刷新的对话对
        """
        timeout = timeout_sec or self._flush_timeout
        cutoff = time.time() - timeout
        flushed: list[tuple[str, str, str]] = []

        with self._lock:
            for device_id, state in self._devices.items():
                if state.current_round is None:
                    continue
                if state.current_round.last_active < cutoff:
                    result = state.current_round.flush()
                    if result:
                        flushed.append((device_id, result[0], result[1]))
                        log.info("超时刷新 device=%s round=%d", device_id, state.current_round.round_id)
                    state.current_round = None

        return flushed

    def cleanup_stale_devices(self, max_idle_sec: int = 3600) -> int:
        """清理长时间无活动的设备状态"""
        cutoff = time.time() - max_idle_sec
        removed = 0
        with self._lock:
            stale_ids = [
                did for did, s in self._devices.items()
                if s.last_active < cutoff
            ]
            for did in stale_ids:
                del self._devices[did]
                removed += 1
        if removed:
            log.info("清理闲置设备状态: %d 个", removed)
        return removed
