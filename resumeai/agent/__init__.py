"""The tailoring agent.

Given a :class:`~resumeai.jd.models.JobRequirements` and a
:class:`~resumeai.context.models.UserContext`, produces a
:class:`~resumeai.agent.models.TailoredResume` — the structured payload the
Phase 5 renderer will paste into a fresh copy of the user's master Google
Doc.

Phase 4 ships with a single-shot LLM call (see ``_private/ARCHITECTURE.md``
ADR-006). MCP-driven tool use is deferred until Phase 7 introduces
interactive iteration that genuinely benefits from it.
"""

from __future__ import annotations
