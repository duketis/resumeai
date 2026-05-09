"""System prompt + user-prompt builder for the tailoring call.

The user prompt is structured markdown rather than raw JSON dumps because
the model handles structured prose better than nested objects, and a human
debugging a failure can read it in one pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resumeai.context.models import (
        CoverLetterEntry,
        GitAuditEntry,
        ResumeBase,
        UserContext,
        WorkHistoryEntry,
    )
    from resumeai.jd.models import JobRequirements


SYSTEM_PROMPT = """\
You are tailoring a candidate's resume for a specific job description.

Two inputs are provided in the user message:
- JOB DESCRIPTION: the structured requirements extracted from the role
  (title, must-haves, required + nice-to-have skills, employer vocabulary).
- CANDIDATE CONTEXT: the candidate's master resume base, full work history
  (with per-role summaries + bullets), git audit, and prior cover letters.

Your task: respond with ONE JSON object (no markdown fences, no commentary,
no preamble) matching this exact schema:

{
  "name": "string — the candidate's name, copied verbatim",
  "headline": "string — one short line tailored to the JD; may rewrite",
  "contact": {
    "email": "string — copied verbatim",
    "phone": "string or null — passthrough",
    "location": "string or null — passthrough",
    "linkedin": "string or null — passthrough",
    "github": "string or null — passthrough",
    "website": "string or null — passthrough"
  },
  "summary": "string — 2-3 sentence summary, freshly written for THIS JD",
  "skills": ["array of strings — most JD-relevant skills FIRST"],
  "work_history": [
    {
      "company": "string — copied verbatim",
      "title": "string — copied verbatim",
      "period": "string or null — eg '2022-03 → 2024-09', passthrough",
      "location": "string or null — passthrough",
      "source_slug": "string — the work_history entry's slug",
      "bullets": [
        {
          "text": "string — tailored bullet, ≤25 words, action verb first",
          "source_slug": "string — the work_history slug this bullet draws from"
        }
      ]
    }
  ],
  "education": [
    {
      "institution": "string — passthrough",
      "degree": "string — passthrough",
      "field": "string or null — passthrough",
      "year_start": "int or null — passthrough",
      "year_end": "int or null — passthrough"
    }
  ],
  "certifications": ["array of strings — drop irrelevant ones"],
  "rationale": "string — one paragraph explaining the major tailoring choices"
}

Hard rules:
- Use ONLY facts present in CANDIDATE CONTEXT. Never invent companies,
  titles, dates, technologies, achievements, or metrics.
- Bullets may be REPHRASED to use the JD's vocabulary, but the underlying
  achievement must come from a source bullet or the role summary.
- Reorder skills so the JD's required skills appear first. Drop skills the
  candidate doesn't have. Don't add skills not in CANDIDATE CONTEXT.
- 4-7 bullets per work_history entry, descending importance.
- Each bullet ≤25 words, action verb first, quantified where the source
  supports it.
- Drop work_history entries the JD makes irrelevant ONLY if the resume is
  too long otherwise; default to keeping every entry.
- Keep candidate name, contact, education, dates, employer names verbatim.
- Output the JSON object and nothing else.
"""


def build_user_prompt(jd: JobRequirements, context: UserContext) -> str:
    """Compose the per-call user prompt from a JD + a candidate context."""
    return "\n\n".join(
        [
            "# JOB DESCRIPTION",
            _format_jd(jd),
            "# CANDIDATE CONTEXT",
            _format_context(context),
            "# OUTPUT",
            "Return the tailored resume JSON per the schema in the system prompt.",
        ]
    )


# -- formatters --------------------------------------------------------------


def _format_jd(jd: JobRequirements) -> str:
    lines: list[str] = [f"**Title:** {jd.title}"]
    if jd.company:
        lines.append(f"**Company:** {jd.company}")
    if jd.location:
        lines.append(f"**Location:** {jd.location}")
    lines.append(f"**Role type:** {jd.role_type.value}")
    lines.append(f"**Seniority:** {jd.seniority.value}")
    lines.append(f"**Employment type:** {jd.employment_type.value}")
    lines.append(f"**Remote type:** {jd.remote_type.value}")
    lines.append("")
    lines.append("**Required skills:**")
    lines.extend(_bulleted(jd.required_skills) or ["- (none extracted)"])
    lines.append("")
    lines.append("**Nice-to-have skills:**")
    lines.extend(_bulleted(jd.nice_to_have_skills) or ["- (none extracted)"])
    lines.append("")
    lines.append("**Must-haves:**")
    lines.extend(_bulleted(jd.must_haves) or ["- (none extracted)"])
    lines.append("")
    lines.append("**Employer vocabulary (echo where natural):**")
    lines.extend(_bulleted(jd.employer_vocabulary) or ["- (none extracted)"])
    if jd.raw_text:
        lines.append("")
        lines.append("**Raw JD text (for nuance):**")
        lines.append("```")
        lines.append(jd.raw_text)
        lines.append("```")
    return "\n".join(lines)


def _format_context(context: UserContext) -> str:
    if context.is_empty():
        return "_(empty context — no resume.yaml or supporting markdown found)_"

    parts: list[str] = []
    if context.resume is not None:
        parts.extend(_format_resume_base(context.resume))
    if context.work_history:
        parts.append("## Work history (most recent first)")
        for entry in context.work_history:
            parts.append("")
            parts.append(_format_work_history(entry))
    if context.git_audit:
        parts.append("")
        parts.append("## Git audit")
        for audit in context.git_audit:
            parts.extend(_format_git_audit(audit))
    if context.cover_letters:
        parts.append("")
        parts.append("## Past cover letters (for tone + vocabulary reference)")
        for letter in context.cover_letters:
            parts.extend(_format_cover_letter(letter))
    return "\n".join(parts)


def _format_resume_base(resume: ResumeBase) -> list[str]:
    lines: list[str] = ["## Resume base", f"**Name:** {resume.name}"]
    if resume.headline:
        lines.append(f"**Headline:** {resume.headline}")
    lines.append(f"**Email:** {resume.contact.email}")
    for label, value in (
        ("Phone", resume.contact.phone),
        ("Location", resume.contact.location),
        ("LinkedIn", resume.contact.linkedin),
        ("GitHub", resume.contact.github),
        ("Website", resume.contact.website),
    ):
        if value:
            lines.append(f"**{label}:** {value}")
    lines.append("")
    lines.append("**Skills (master list):**")
    lines.extend(_bulleted(resume.skills) or ["- (none)"])
    lines.append("")
    lines.append("**Education:**")
    if resume.education:
        for edu in resume.education:
            period = _year_range(edu.year_start, edu.year_end)
            field = f" — {edu.field}" if edu.field else ""
            lines.append(f"- {edu.institution}: {edu.degree}{field}{period}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("**Certifications:**")
    lines.extend(_bulleted(resume.certifications) or ["- (none)"])
    lines.append("")
    return lines


def _format_git_audit(audit: GitAuditEntry) -> list[str]:
    lines: list[str] = ["", f"### {audit.repo} ({audit.slug})"]
    if audit.role:
        lines.append(f"**Role:** {audit.role}")
    if audit.period:
        lines.append(f"**Period:** {audit.period}")
    if audit.summary:
        lines.append("")
        lines.append(audit.summary)
    return lines


def _format_cover_letter(letter: CoverLetterEntry) -> list[str]:
    header = letter.slug
    if letter.role and letter.company:
        header = f"{letter.role} @ {letter.company} ({letter.slug})"
    lines: list[str] = ["", f"### {header}"]
    if letter.body:
        lines.append("")
        lines.append(letter.body)
    return lines


def _format_work_history(entry: WorkHistoryEntry) -> str:
    lines: list[str] = [f"### {entry.title} @ {entry.company} (slug: `{entry.slug}`)"]
    period = _date_range(entry.start, entry.end)
    location_line = f", {entry.location}" if entry.location else ""
    lines.append(f"**Period:** {period}{location_line}")
    if entry.technologies:
        lines.append(f"**Technologies:** {', '.join(entry.technologies)}")
    if entry.summary:
        lines.append("")
        lines.append(entry.summary)
    if entry.bullets:
        lines.append("")
        lines.append("**Source bullets:**")
        for bullet in entry.bullets:
            lines.append(f"- {bullet}")
    return "\n".join(lines)


def _bulleted(items: tuple[str, ...]) -> list[str]:
    return [f"- {item}" for item in items]


def _date_range(start: str | None, end: str | None) -> str:
    if start and end:
        return f"{start} → {end}"
    if start:
        return f"{start} → present"
    if end:
        return f"… → {end}"
    return "(undated)"


def _year_range(start: int | None, end: int | None) -> str:
    if start and end:
        return f" ({start}–{end})"
    if start:
        return f" ({start}–present)"
    if end:
        return f" (…–{end})"
    return ""
