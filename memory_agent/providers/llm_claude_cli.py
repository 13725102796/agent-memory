"""Claude CLI 实现 — 通过 subprocess 调用 claude 命令行"""
from __future__ import annotations

import os
import pty
import re
import select
import subprocess
from typing import Iterator

from memory_agent.config import settings
from memory_agent.log import get_logger
from memory_agent.providers.base import LLMProvider

log = get_logger("providers.llm")

# 匹配 ANSI/终端控制序列（颜色、光标、模式切换等）
_ANSI_RE = re.compile(
    r'\x1b\[[0-9;?<>=]*[a-zA-Z~]'  # CSI: ESC[ ... letter (含 ?<>= 参数)
    r'|\x1b\].*?\x07'               # OSC: ESC] ... BEL
    r'|\x1b[()][AB012]'             # 字符集切换
    r'|\x1b[>=<]'                    # 键盘/光标模式
    r'|\x1b'                         # 孤立 ESC（兜底）
    r'|\x0f|\r'                      # SI, CR
    r'|\[[\?<>=][0-9;]*[a-zA-Z~]'   # 跨 chunk 丢失 ESC 的残余序列
)


class ClaudeCLIProvider(LLMProvider):
    """通过本地 Claude CLI 调用 LLM"""

    def __init__(
        self,
        cli_path: str | None = None,
        model: str | None = None,
        cheap_model: str | None = None,
        timeout: int | None = None,
    ):
        self._cli_path = cli_path or settings.claude_cli_path
        self._model = model or settings.claude_model
        self._cheap_model = cheap_model or settings.claude_cheap_model
        self._timeout = timeout or settings.claude_timeout

    def _call(self, prompt: str, model: str) -> str:
        cmd = [
            self._cli_path,
            "-p", prompt,
            "--model", model,
            "--dangerously-skip-permissions",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self._timeout, stdin=subprocess.DEVNULL,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                err = result.stderr.strip()
                log.warning("CLI 错误 (code=%d): %s", result.returncode, err[:200])
                return output or err or "调用失败"
            return output
        except subprocess.TimeoutExpired:
            log.warning("CLI 超时 (%ds)", self._timeout)
            return "调用超时"
        except FileNotFoundError:
            log.error("CLI 未找到: %s", self._cli_path)
            return "Claude CLI 未安装"
        except Exception as e:
            log.error("异常: %s", e)
            return str(e)

    def _call_stream(self, prompt: str, model: str) -> Iterator[str]:
        """流式调用 CLI，用 pty 伪终端获取实时输出"""
        cmd = [
            self._cli_path,
            "-p", prompt,
            "--model", model,
            "--dangerously-skip-permissions",
        ]

        # 用伪终端让 CLI 认为在终端中运行，禁用输出缓冲
        master_fd, slave_fd = pty.openpty()
        try:
            proc = subprocess.Popen(
                cmd, stdout=slave_fd, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, close_fds=True,
            )
        except FileNotFoundError:
            os.close(master_fd)
            os.close(slave_fd)
            log.error("CLI 未找到: %s", self._cli_path)
            yield "Claude CLI 未安装"
            return
        except Exception as e:
            os.close(master_fd)
            os.close(slave_fd)
            log.error("异常: %s", e)
            yield str(e)
            return

        os.close(slave_fd)  # 子进程已继承，父进程关闭

        buf = b""
        pending = ""  # 保留末尾可能是 ESC 开头的不完整序列
        try:
            while True:
                r, _, _ = select.select([master_fd], [], [], 1.0)
                if r:
                    try:
                        data = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    buf += data
                    text = _try_decode(buf)
                    if text is not None:
                        buf = b""
                        text = pending + text
                        pending = ""
                        clean = _strip_control(text)
                        # 如果末尾是 ESC，留到下个 chunk 合并处理
                        if clean.endswith("\x1b"):
                            pending = "\x1b"
                            clean = clean[:-1]
                        if clean:
                            yield clean
                elif proc.poll() is not None:
                    break
        finally:
            # 输出剩余缓冲
            remaining = pending
            if buf:
                remaining += buf.decode("utf-8", errors="replace")
            if remaining:
                clean = _strip_control(remaining)
                if clean:
                    yield clean
            try:
                os.close(master_fd)
            except OSError:
                pass
            proc.wait()
            if proc.returncode != 0:
                err = proc.stderr.read().decode("utf-8", errors="replace").strip()
                if err:
                    log.warning("CLI 流式错误 (code=%d): %s", proc.returncode, err[:200])

    def chat(self, system_prompt: str, user_message: str) -> str:
        prompt = f"{system_prompt}\n\n用户消息：\n{user_message}"
        return self._call(prompt, self._model)

    def chat_stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        prompt = f"{system_prompt}\n\n用户消息：\n{user_message}"
        yield from self._call_stream(prompt, self._model)

    def cheap(self, prompt: str) -> str:
        return self._call(prompt, self._cheap_model)


def _strip_control(text: str) -> str:
    """清除 ANSI 转义序列和残余终端控制符"""
    text = _ANSI_RE.sub("", text)
    # 清除所有 C0 控制字符（保留 \n \t）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text


def _try_decode(buf: bytes) -> str | None:
    """尝试 UTF-8 解码，处理不完整的多字节字符"""
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        # 末尾可能有不完整的多字节字符
        for i in range(1, 4):
            try:
                return buf[:-i].decode("utf-8")
            except UnicodeDecodeError:
                continue
    return None
