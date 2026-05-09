"""Section identification + content-builder functions.

Each "section kind" (summary, skills, work_history, education,
certifications) has:

- A tuple of accepted heading aliases (matched case-insensitively against
  the template's section headings).
- A render function that turns the relevant ``TailoredResume`` field into
  the plain-text payload to drop into the section.

Render functions return ``None`` (rather than an empty string) when the
``TailoredResume`` carries nothing for the section — the orchestrator uses
that to emit a ``SKIPPED_EMPTY`` diff instead of issuing an empty edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resumeai.agent.models import TailoredResume, TailoredWorkEntry
    from resumeai.context.models import Education
    from resumeai.docs.models import Section, TemplateModel


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "about", "profile", "about me", "professional summary"),
    "skills": (
        "skills",
        "technical skills",
        "core skills",
        "technologies",
        "tech stack",
    ),
    "work_history": (
        "experience",
        "work experience",
        "employment",
        "professional experience",
        "work history",
        "career",
    ),
    "education": ("education", "qualifications", "academic"),
    "certifications": ("certifications", "certificates", "credentials"),
}


def find_section(template: TemplateModel, kind: str) -> Section | None:
    """Return the first section whose heading matches an alias for ``kind``."""
    aliases = SECTION_ALIASES.get(kind, ())
    if not aliases:
        return None
    for section in template.sections:
        if section.heading.strip().lower() in aliases:
            return section
    return None


# -- content builders --------------------------------------------------------


def render_summary(tailored: TailoredResume) -> str | None:
    text = tailored.summary.strip()
    return text or None


def render_skills(tailored: TailoredResume) -> str | None:
    if not tailored.skills:
        return None
    return ", ".join(tailored.skills)


def render_work_history(tailored: TailoredResume) -> str | None:
    if not tailored.work_history:
        return None
    return "\n\n".join(_render_work_entry(entry) for entry in tailored.work_history)


def render_education(tailored: TailoredResume) -> str | None:
    if not tailored.education:
        return None
    return "\n".join(_render_education(edu) for edu in tailored.education)


def render_certifications(tailored: TailoredResume) -> str | None:
    if not tailored.certifications:
        return None
    return "\n".join(f"• {cert}" for cert in tailored.certifications)


# -- helpers -----------------------------------------------------------------


def _render_work_entry(entry: TailoredWorkEntry) -> str:
    lines: list[str] = [f"{entry.company} — {entry.title}"]
    meta_bits: list[str] = []
    if entry.location:
        meta_bits.append(entry.location)
    if entry.period:
        meta_bits.append(entry.period)
    if meta_bits:
        lines.append(" · ".join(meta_bits))
    for bullet in entry.bullets:
        lines.append(f"• {bullet.text}")
    return "\n".join(lines)


def _render_education(edu: Education) -> str:
    parts = [f"{edu.institution} — {edu.degree}"]
    if edu.field:
        parts.append(edu.field)
    period = _education_period(edu.year_start, edu.year_end)
    if period:
        parts.append(period)
    return " · ".join(parts)


def _education_period(start: int | None, end: int | None) -> str:
    if start and end:
        return f"{start}–{end}"
    if start:
        return f"{start}–present"
    if end:
        return f"…–{end}"
    return ""
