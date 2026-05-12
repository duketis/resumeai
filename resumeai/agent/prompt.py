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
  "headline": "string — short tailored line; may only name techs in the candidate's skills list",
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
      "description": "string — noun phrase ≤8 words; no em-dashes; no mid-sentence punctuation",
      "stack": "string — 4-7 comma-separated technologies (pick the JD-relevant ones; cap at 7)",
      "link": "string or null — clickable right-column URL; null for private/never-link projects",
      "link_label": "string or null — right-column text (display label or 'Private project')",
      "bullets": [
        {
          "text": "string — bullet text, action verb first, ≤25 words",
          "source_slug": "string — the projects/<slug> this bullet draws from"
        }
      ],
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
- The ``skills`` list is for concrete technologies ONLY: programming
  languages, frameworks, libraries, platforms (clouds, OSes), databases,
  protocols, tools (Docker, Git, etc.). FILTER OUT process / hygiene /
  methodology items even if the candidate's source ``resume.yaml`` lists
  them as skills -- e.g. drop "TDD", "structured logging", "conventional
  commits", "GPG-signed commits", "code review", "pair programming",
  "agile development", "documentation". Those are habits, not skills, and
  they read as filler on a tailored resume. If a methodology really wants
  surfacing (because the JD asks for it), put it in the ``summary`` or a
  bullet, never in the ``skills`` array.
- This rule is absolute for ``headline``, ``summary``, and ``skills``: do NOT
  name a JD-required technology that's missing from the candidate's context,
  even if the JD asks for it explicitly. If the candidate doesn't have it,
  it does not appear ANYWHERE in the tailored resume. Do not paraphrase a
  missing tech as a related one (e.g. "Python" is not a substitute for
  "Django"; "AWS Lambda" is not a substitute for "GCP Cloud Functions").
- Same rule for INDUSTRY SECTORS / CLIENT TYPES: do NOT claim experience in
  a sector (government, defence, telco, fintech, education, etc.) unless an
  actual client engagement in the candidate's work_history backs it. The JD
  saying "projects of national significance" or "government clients" is NOT
  permission to claim government experience -- only the candidate's actual
  client list is. If the candidate has worked at zero government clients,
  never use the word "government" in headline / summary / key_achievements.
- 4-7 bullets per work_history entry, descending importance.
- Each bullet ≤25 words, action verb first, quantified where the source
  supports it.
- Drop work_history entries the JD makes irrelevant ONLY if the resume is
  too long otherwise; default to keeping every entry.
- Keep candidate name, contact, education, dates, employer names verbatim.

Consultancy / placement structure:
- When multiple work_history entries share a parent employer (detected via
  ``consulting via X`` text in their location field, or via a dedicated
  consultancy entry like "DiUS Computing"), the parent employer is the
  REAL employer and the client placements are sub-engagements under it.
- DO NOT emit a standalone parent-employer entry. Instead, prefix every
  client placement's ``company`` field with the parent employer name and
  an em-dash separator. Example: ``InTruth (Healthcare/Wellness)``
  becomes ``DiUS — InTruth (Healthcare/Wellness)``. This way every
  ``work_history`` row is self-describing -- humans and ATS parsers both
  see the parent/client relationship at a glance, and the section can't
  break across pages with an orphaned parent heading at the bottom.
- Each client placement gets a short period like ``2021 (6 mo)`` or
  ``Oct 2024 (2 mo)`` instead of bare full dates -- the period is the
  duration AT that client, not the parent consultancy's overall tenure.
- Order client engagements as follows: the LONGEST-running engagement
  first (the candidate's anchor account -- for Jonathan that's
  ``Premium Valet`` at 2020--2025), then every other engagement in
  STRICT DESCENDING order by end date. A run of short placements that
  all ended in the same year keeps strict end-date descending; never
  surface the oldest-ended placement before a more-recent one.
- Never list a client placement without the parent prefix. Recruiters
  reading 7 employers in 5 years will assume job-hopping; the truth is
  one employer (DiUS) placed across 7 clients.
- The ``title`` field should reflect the project's nature (e.g.
  "Customer checkout flow for Australian retailer") rather than just
  repeating "Software Engineer (Consultant)" -- the prefix already
  carries the formal employer relationship.

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
- For projects with a ``Local folder scan``: MINE THE SCAN for resume
  bullets. The scan exposes the project's README files, every CLAUDE.md
  / PLAN_*.md / ARCHITECTURE.md across subfolders, the verbatim
  dependency manifests, and a code-stats breakdown (LOC by language).
  Treat any shipped feature, architectural decision, or quantified
  capability in those docs as legitimate bullet material -- prefer
  these over paraphrasing the hand-written project body. Lift concrete
  numbers (LOC, test counts, sub-system counts) verbatim where they
  appear in the scan.
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
- The rendered heading line ``name --- description $|$ stack`` must
  stay on ONE line. Keep ``description`` to ≤8 words AND keep ``stack``
  to ≤7 entries -- the template can't wrap the project heading. If the
  full stack from the project body has more than 7 entries, drop the
  ones least relevant to the JD.

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
    parts.extend(_format_authoritative_refs(context))
    return "\n".join(parts)


def _format_authoritative_refs(context: UserContext) -> list[str]:
    """Append master resume + reference resume blocks for the agent.

    Split out of ``_format_context`` to keep its branch count manageable
    (ruff PLR0912). Returns an empty list when neither is present.
    """
    parts: list[str] = []
    if context.master_resume:
        parts.append("")
        parts.append("## Master resume (AUTHORITATIVE — verbatim source of truth)")
        parts.append(
            "This is the candidate's plain-text master resume. Treat it as the "
            "AUTHORITATIVE source for every fact: client names, dates, titles, "
            "metrics, technologies, sectors. If a fact isn't here AND isn't in "
            "the structured context above, do not include it. You may rephrase "
            "to echo the JD's vocabulary, but never invent."
        )
        parts.extend(["```", context.master_resume, "```"])
    if context.reference_resumes:
        parts.append("")
        parts.append("## Reference resumes (KNOWN-GOOD shape + tone)")
        parts.append(
            "Below are hand-tailored resumes the candidate has already produced "
            "and considers known-good. Treat them as the model for STRUCTURE "
            "(parent employer + sub-engagements, headline length, bullet style, "
            "section ordering) and TONE (phrasing, level of detail). Your output "
            "should look like a sibling of these, not a different beast. Do NOT "
            "copy a bullet verbatim if it doesn't fit the new JD; do follow the "
            "same shape."
        )
        for idx, ref in enumerate(context.reference_resumes, start=1):
            parts.extend([f"### Reference {idx}", "```", ref, "```"])
    return parts


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
    if project.scanned:
        lines.append("")
        lines.append(
            "**Local folder scan** (README + structure + git log from the "
            "project's actual codebase -- mine this for richer bullet "
            "material than the hand-written body alone):"
        )
        lines.append("```")
        lines.append(project.scanned)
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
