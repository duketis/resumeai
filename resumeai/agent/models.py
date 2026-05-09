"""The structural payload the Phase 5 renderer consumes.

Every field here is what gets pasted into the cloned Google Doc, in the
order it should appear. Provenance (``source_slug`` on a bullet) is kept
so the diff view can show *why* the agent chose a specific bullet over
the candidate's other options.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from resumeai.context.models import Contact, Education


class TailoredBullet(BaseModel):
    """One tailored bullet, with optional pointer back to its source entry."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    source_slug: str | None = None


class TailoredWorkEntry(BaseModel):
    """A single work-history block in the tailored resume."""

    model_config = ConfigDict(frozen=True)

    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    period: str | None = None
    location: str | None = None
    bullets: tuple[TailoredBullet, ...] = ()
    source_slug: str | None = None


class TailoredResume(BaseModel):
    """The full tailored resume.

    Names of identity fields (``name``, ``contact``, ``education``,
    ``certifications``) are passed through from the user's
    :class:`~resumeai.context.models.ResumeBase`; the agent never invents
    or alters them.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    headline: str = ""
    contact: Contact
    summary: str = ""
    skills: tuple[str, ...] = ()
    work_history: tuple[TailoredWorkEntry, ...] = ()
    education: tuple[Education, ...] = ()
    certifications: tuple[str, ...] = ()
    rationale: str = ""
