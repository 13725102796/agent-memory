"""火山引擎字幕回调解码 — Base64 + 二进制协议 + JSON"""
from __future__ import annotations

import base64
import json
import struct
from typing import Optional

from memory_agent.log import get_logger
from memory_agent.types import SubtitleEntry

log = get_logger("volcano.decoder")

# 二进制协议 magic number
_MAGIC = b"subv"
_HEADER_SIZE = 8  # 4 bytes magic + 4 bytes length


def verify_signature(signature: str, expected: str) -> bool:
    """验证回调签名（简单字符串比较）"""
    if not expected:
        return True  # 未配置签名则跳过验证
    return signature == expected


def decode_subtitle_message(body: dict) -> Optional[list[SubtitleEntry]]:
    """解码火山引擎字幕回调

    Args:
        body: HTTP POST 请求体 {"message": "base64...", "signature": "..."}

    Returns:
        解码后的字幕条目列表，失败返回 None
    """
    raw_message = body.get("message", "")
    if not raw_message:
        log.warning("回调 message 字段为空")
        return None

    # 1. Base64 解码
    try:
        binary_data = base64.b64decode(raw_message)
    except Exception as e:
        log.warning("Base64 解码失败: %s", e)
        return None

    # 2. 解析二进制头
    if len(binary_data) < _HEADER_SIZE:
        log.warning("二进制数据过短: %d bytes", len(binary_data))
        return None

    magic = binary_data[:4]
    if magic != _MAGIC:
        log.warning("Magic number 不匹配: %s (期望 %s)", magic, _MAGIC)
        return None

    payload_length = struct.unpack(">I", binary_data[4:8])[0]
    payload_bytes = binary_data[8:8 + payload_length]

    if len(payload_bytes) < payload_length:
        log.warning("负载长度不足: 期望 %d, 实际 %d", payload_length, len(payload_bytes))
        return None

    # 3. JSON 解析
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as e:
        log.warning("JSON 解析失败: %s", e)
        return None

    # 4. 提取字幕条目
    msg_type = payload.get("type", "")
    if msg_type != "subtitle":
        log.debug("非字幕类型消息: %s", msg_type)
        return None

    data_list = payload.get("data", [])
    if not isinstance(data_list, list):
        log.warning("data 字段不是数组")
        return None

    entries = []
    for item in data_list:
        try:
            entry = SubtitleEntry(
                text=item.get("text", ""),
                userId=item.get("userId", ""),
                sequence=int(item.get("sequence", 0)),
                definite=bool(item.get("definite", False)),
                paragraph=bool(item.get("paragraph", False)),
                roundId=int(item.get("roundId", 0)),
                language=item.get("language", "zh"),
            )
            entries.append(entry)
        except (TypeError, ValueError) as e:
            log.warning("字幕条目解析异常: %s, 原始数据: %s", e, item)
            continue

    return entries
