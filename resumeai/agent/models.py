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


class TailoredProject(BaseModel):
    """A personal-projects entry in the tailored resume.

    ``link`` is the full URL when the project should appear as a
    clickable link in the right-column; ``link_label`` is the visible
    label (display text). Private projects keep ``link`` as ``None`` and
    set ``link_label`` to a plain string like ``"Private project"`` so
    the right-column still shows something but isn't a dead URL.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str = ""
    stack: str = ""
    link: str | None = None
    link_label: str | None = None
    bullets: tuple[str, ...] = ()
    source_slug: str | None = None


class TailoredResume(BaseModel):
    """The full tailored resume.

    Names of identity fields (``name``, ``contact``, ``education``,
    ``certifications``) are passed through from the user's
    :class:`~resumeai.context.models.ResumeBase`; the agent never invents
    or alters them.

    ``key_achievements`` is a cross-engagement highlight reel (drawn from
    work history, projects, and the git audit) used between Education
    and Professional Experience. ``personal_projects`` mirrors the
    candidate's ``UserContext/projects/*.md`` entries.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    headline: str = ""
    contact: Contact
    summary: str = ""
    skills: tuple[str, ...] = ()
    key_achievements: tuple[str, ...] = ()
    work_history: tuple[TailoredWorkEntry, ...] = ()
    education: tuple[Education, ...] = ()
    certifications: tuple[str, ...] = ()
    personal_projects: tuple[TailoredProject, ...] = ()
    rationale: str = ""
