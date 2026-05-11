"""Output shapes for the renderer."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RenderStatus(StrEnum):
    """One per section we attempted to render."""

    REPLACED = "replaced"
    """Section was found in the template; old content removed, new content inserted."""

    SKIPPED_EMPTY = "skipped_empty"
    """The :class:`TailoredResume` had nothing to render for this section."""

    NOT_FOUND = "not_found"
    """No heading in the template matched any of this section's aliases."""


class RenderDiff(BaseModel):
    """One diff entry per section the renderer attempted."""

    model_config = ConfigDict(frozen=True)

    kind: str = Field(min_length=1)
    heading: str | None = None
    status: RenderStatus
    before_chars: int = 0
    after_chars: int = 0


class RenderResult(BaseModel):
    """The output of a full render.

    The rendered PDF is written to disk by the renderer and addressed via
    ``doc_url``. We don't carry the raw bytes on the model because the Run
    record is JSON-serialised into SQLite and PDF bytes aren't UTF-8.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    doc_url: str = Field(min_length=1)
    pdf_size_bytes: int = Field(default=0, ge=0)
    diffs: tuple[RenderDiff, ...] = ()


class RenderError(RuntimeError):
    """Raised when the render pipeline can't proceed (eg. master missing)."""
