"""LLM access layer.

Single ``LLMClient`` ``Protocol`` that all callers (Phase 2 JD parser, Phase 4
tailoring agent) depend on. The production implementation
(:class:`~resumeai.llm.client.ClaudeCliClient`) spawns the ``claude`` CLI as
a subprocess so we use Jonathan's Anthropic Max subscription instead of the
per-token API. Tests use :class:`~resumeai.llm.client.FakeLLMClient`.
"""

from __future__ import annotations
