"""Structural model of a resume template.

The reader produces a :class:`TemplateModel` from a raw Google Docs ``documents.get``
response; the renderer (Phase 5) consumes it. Sections are inferred from the
heading paragraph styles (``HEADING_1``, ``HEADING_2``, ...) — Google Docs
exposes these as ``namedStyleType`` on each paragraph.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    """A heading-bounded slice of the document.

    Indices are Google Docs character indices (1-based, end-exclusive). The
    section spans from its heading paragraph (inclusive) to just before the
    next heading of equal-or-higher level (exclusive); the last section runs
    to the end of the body.
    """

    model_config = ConfigDict(frozen=True)

    heading: str
    level: int = Field(ge=1, le=6)
    start_index: int = Field(ge=1)
    end_index: int = Field(ge=1)
    paragraph_count: int = Field(ge=1)


class TemplateModel(BaseModel):
    """The user's resume template as resumeai sees it."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    title: str
    revision_id: str
    sections: tuple[Section, ...]
