"""Pydantic model for resumeai runtime settings.

Extends ``tailor_core.settings.BaseRuntimeSettings`` with the one resume-app
specific knob: the LaTeX template filename. ``BaseRuntimeSettings`` already
carries ``model`` (LLM model override) and ``frozen=True`` config.
"""

from __future__ import annotations

from tailor_core.settings.models import BaseRuntimeSettings


class RuntimeSettings(BaseRuntimeSettings):
    """User-tunable resumeai settings persisted in the local SQLite DB."""

    template_name: str = "default.tex.j2"
