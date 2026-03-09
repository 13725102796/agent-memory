"""结构化子系统日志 — 参考 OpenClaw createSubsystemLogger"""
from __future__ import annotations

import logging
import sys

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    root = logging.getLogger("memory_agent")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _configured = True


def get_logger(subsystem: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"memory_agent.{subsystem}")
