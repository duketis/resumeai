"""System prompt + user-prompt builder for the tailoring call.

The user prompt is structured markdown rather than raw JSON dumps because
the model handles structured prose better than nested objects, and a human
debugging a failure can read it in one pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from resumeai.context.models import (
        CoverLetterEntry,
        GitAuditEntry,
        ProjectEntry,
        ResumeBase,
        UserContext,
        WorkHistoryEntry,
    )
    from resumeai.context_files.models import ContextFile
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
  "key_achievements": [
    "array of 5-7 strings — cross-engagement highlight bullets drawn",
    "from work history, projects, and the git audit; do NOT duplicate a",
    "bullet that already appears verbatim under Professional Experience"
  ],
  "personal_projects": [
    {
      "name": "string — project name (copied verbatim from the projects/ entry)",
      "description": "string — one-line description (≤15 words)",
      "stack": "string — comma-separated tech list",
      "link": "string or null — clickable right-column URL; null for private/never-link projects",
      "link_label": "string or null — right-column text (display label or 'Private project')",
      "bullets": ["array of 2-4 strings — action verb first, ≤25 words each"],
      "source_slug": "string — the projects/<slug> this draws from"
    }
  ],
  "rationale": "string — one paragraph explaining the major tailoring choices"
}

Hard rules:
- Use ONLY facts present in CANDIDATE CONTEXT or UPLOADED CONTEXT FILES.
  Never invent companies, titles, dates, technologies, achievements, or
  metrics.
- Bullets may be REPHRASED to use the JD's vocabulary, but the underlying
  achievement must come from a source bullet, role summary, or uploaded
  file.
- Reorder skills so the JD's required skills appear first. Drop skills the
  candidate doesn't have. Don't add skills not in any context source.
- 4-7 bullets per work_history entry, descending importance.
- Each bullet ≤25 words, action verb first, quantified where the source
  supports it.
- Drop work_history entries the JD makes irrelevant ONLY if the resume is
  too long otherwise; default to keeping every entry.
- Keep candidate name, contact, education, dates, employer names verbatim.

Key Achievements rules:
- Produce 5-7 ``key_achievements`` bullets total, descending importance.
- Each bullet is one cross-engagement highlight that the JD would value.
- Draw from work history, projects, and the git audit. Echo the JD's
  vocabulary where the underlying fact supports it.
- NEVER repeat a bullet verbatim from Professional Experience -- this
  is the summary highlight reel, not a copy.

Personal Projects rules:
- Emit ``personal_projects`` ONLY from the CANDIDATE CONTEXT ``Projects``
  section. Never invent a project.
- Read each project's body carefully and obey its positioning rules.
  If the body says "never link as a URL" or "label it as 'Private project'",
  set ``link`` to ``null`` and ``link_label`` to the literal label
  (e.g. ``"Private project"``).
- Otherwise, set ``link`` to the project's full https URL and
  ``link_label`` to a clean display string (typically the URL with the
  scheme stripped, e.g. ``"github.com/x/y"``).
- 2-4 bullets per project, framed at the architecture / system-design
  level when the project body indicates "ask me about it" (architecture
  walkthrough) and at the implementation/code-review level when the
  project body indicates "click through and audit" (code-review showcase).

Multi-source rules:
- The MASTER TEMPLATE's current content (when present in uploaded files,
  tagged ``source:master_template``) is the existing resume. Treat it as
  authoritative for facts already on the resume — same employer names,
  dates, education entries.
- When two sources disagree on a fact (different dates for the same role,
  different metric values), prefer in this order: (1) the master template,
  (2) the most recent / most specific uploaded file (look at the user's
  note + tags for hints), (3) the structured CANDIDATE CONTEXT.
- When two sources describe the same achievement in different words, pick
  the wording that best echoes the JD's vocabulary, and fold any unique
  details from the other source(s) into one cohesive bullet.

- Output the JSON object and nothing else.
"""


def build_user_prompt(
    jd: JobRequirements,
    context: UserContext,
    *,
    context_files: Sequence[ContextFile] = (),
) -> str:
    """Compose the per-call user prompt from a JD, the structured context,
    and any user-uploaded supplementary context files."""
    parts = [
        "# JOB DESCRIPTION",
        _format_jd(jd),
        "# CANDIDATE CONTEXT",
        _format_context(context),
    ]
    if context_files:
        parts.append("# UPLOADED CONTEXT FILES")
        parts.append(_format_context_files(context_files))
    parts.extend(
        [
            "# OUTPUT",
            "Return the tailored resume JSON per the schema in the system prompt.",
        ]
    )
    return "\n\n".join(parts)


def _format_context_files(files: Sequence[ContextFile]) -> str:
    blocks: list[str] = [
        "These are user-uploaded files (PDFs, CSVs, text, markdown). Treat each as "
        "additional candidate-side context to draw on. The user expects you to use "
        "the file's content where it matches the JD, and to ignore parts that don't.",
    ]
    for f in files:
        header = f"## {f.name} ({f.kind.value})"
        if f.tags:
            header += " — tags: " + ", ".join(f.tags)
        blocks.append(header)
        if f.note:
            blocks.append(f"_Note from user:_ {f.note}")
        blocks.append("```")
        blocks.append(f.extracted_text.strip())
        blocks.append("```")
    return "\n\n".join(blocks)


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
    if context.projects:
        parts.append("")
        parts.append("## Projects")
        parts.append(
            "Each project below ships as a Personal Projects entry on the tailored "
            "resume. Read each project's body for its positioning rules -- in "
            "particular, whether the right-column should be a clickable URL or a "
            "plain label like 'Private project'."
        )
        for project in context.projects:
            parts.extend(_format_project(project))
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


def _format_project(project: ProjectEntry) -> list[str]:
    lines: list[str] = ["", f"### {project.name} (slug: `{project.slug}`)"]
    if project.url:
        lines.append(f"**URL:** {project.url}")
    if project.status:
        lines.append(f"**Status:** {project.status}")
    if project.stack:
        lines.append(f"**Stack:** {project.stack}")
    if project.summary:
        lines.append("")
        lines.append(project.summary)
    if project.bullets:
        lines.append("")
        lines.append("**Source bullets:**")
        for bullet in project.bullets:
            lines.append(f"- {bullet}")
    if project.body:
        lines.append("")
        lines.append("**Full body (read for positioning rules):**")
        lines.append("```")
        lines.append(project.body)
        lines.append("```")
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
