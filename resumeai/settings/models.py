"""Pydantic models for runtime settings.

After the LaTeX pivot the only persisted user-tunable knob is which LaTeX
template to use. Agent backend / model preferences would slot in here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RuntimeSettings(BaseModel):
    """User-tunable runtime settings persisted in the local SQLite DB."""

    model_config = ConfigDict(frozen=True)

    template_name: str = "default.tex.j2"
    model: str | None = None
