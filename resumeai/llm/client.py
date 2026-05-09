"""LLM client abstraction and the production ``claude`` CLI subprocess impl.

Why subprocess?
The user runs Anthropic's Max subscription. The ``claude`` CLI in headless
mode (``--print`` + a prompt) reads that subscription's auth from
``~/.claude/`` and runs the prompt without per-token API billing. We get the
final text response on stdout and exit-code semantics for free.

This module is intentionally tiny — it does one thing (send a prompt, get a
string back). Streaming, tool-use, and MCP wiring are Phase 4 concerns and
will live alongside (or replace) this file when that phase lands.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when the LLM call fails (subprocess error, timeout, missing CLI)."""


class LLMClient(Protocol):
    """The single operation every caller depends on."""

    def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        """Run a single completion. Return the model's text response."""


DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_TIMEOUT_SECONDS = 120


class ClaudeCliClient:
    """Spawns the ``claude`` CLI in headless mode.

    Args:
        claude_bin: name or absolute path of the CLI binary. Defaults to
            ``"claude"`` and is resolved on first use via :func:`shutil.which`.
        timeout: per-call timeout in seconds.
    """

    def __init__(
        self,
        *,
        claude_bin: str = "claude",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._claude_bin = claude_bin
        self._timeout = timeout

    def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        binary = shutil.which(self._claude_bin) or self._claude_bin
        cmd = [
            binary,
            "--print",
            "--system-prompt",
            system,
            "--model",
            model or DEFAULT_MODEL,
            user,
        ]
        try:
            proc = subprocess.run(  # noqa: S603 — args are a list, no shell expansion
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise LLMError(
                f"`claude` CLI not found at {self._claude_bin!r}. "
                "Install Claude Code from https://claude.com/claude-code."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"`claude` CLI timed out after {self._timeout}s") from exc

        if proc.returncode != 0:
            stderr = proc.stderr.strip() or "(no stderr)"
            raise LLMError(f"`claude` CLI exited with {proc.returncode}: {stderr}")

        return proc.stdout


class FakeLLMClient:
    """Test double. Returns scripted responses + records calls.

    If a prompt isn't in ``responses``, ``raise_for_unknown=True`` raises
    :class:`LLMError`; ``False`` returns ``default_response`` instead.
    """

    def __init__(
        self,
        *,
        responses: dict[str, str] | None = None,
        default_response: str = "{}",
        raise_for_unknown: bool = False,
    ) -> None:
        self._responses = responses or {}
        self._default = default_response
        self._raise_for_unknown = raise_for_unknown
        self.calls: list[tuple[str, str, str | None]] = []

    def complete(self, *, system: str, user: str, model: str | None = None) -> str:
        self.calls.append((system, user, model))
        if user in self._responses:
            return self._responses[user]
        if self._raise_for_unknown:
            raise LLMError(f"FakeLLMClient: no scripted response for prompt {user!r}")
        return self._default
