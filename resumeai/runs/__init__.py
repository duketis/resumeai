"""Tailoring runs — request, persistence, orchestration, event stream.

A "run" is one trip through the pipeline:
    JD → JobRequirements → UserContext → TailoredResume → Google Doc + PDF.

The :class:`~resumeai.runs.orchestrator.TailoringOrchestrator` drives the
pipeline asynchronously, persists the run state via ``RunsStore``, and emits
events to in-memory subscribers consumed by the SSE route.
"""

from __future__ import annotations
