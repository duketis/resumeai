"""Resume-tailoring run orchestration.

The skeleton (Run / RunStatus / RunEvent / TailorRequest models,
``RunsStore`` and ``RunEventBus`` implementations, the ``BaseOrchestrator``
template-method) lives in ``tailor_core.runs``. This package supplies the
resume-flavoured concrete orchestrator and the filename helper used to
build per-JD PDF stems.
"""

from __future__ import annotations
