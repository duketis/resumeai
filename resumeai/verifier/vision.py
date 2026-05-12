"""Visual QC of the rendered resume PDF using a vision-capable LLM call.

The text verifier in :mod:`resumeai.verifier.verifier` can't see layout --
widow lines, awkward gaps, section-header orphaning, content overflowing
the right margin, density issues. This module hands off the rendered PDF
to :func:`tailor_core.verifier.vision.run_vision_verification` along with a
resume-flavoured ``SYSTEM_PROMPT``; the lib rasterises and dispatches.

Failures degrade gracefully: a missing token, network error, SDK exception,
or malformed model response all produce ``None`` rather than blocking the
run. The user still gets the text verifier's findings; visual ones are
best-effort.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tailor_core.verifier.vision import (
    DEFAULT_VISION_DPI,
    DEFAULT_VISION_MAX_TOKENS,
    DEFAULT_VISION_MODEL,
    run_vision_verification,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tailor_core.verifier.models import VerificationResult


SYSTEM_PROMPT = """\
You are a senior recruiter reviewing the VISUAL LAYOUT of a rendered
resume PDF. The text content is being QC'd separately -- your job is
PURELY visual: would this resume look professional to a hiring manager
who opens it in a PDF viewer?

Look for:

- **Orphan lines / widow lines**: a paragraph whose first or last line
  sits alone at the top or bottom of a different page from the rest
  of the paragraph. Bullets that wrap with only 1-3 words spilling to
  the next page are the most common case and look unprofessional.
- **Section header orphaning**: a section title at the bottom of one
  page with all its content on the next.
- **Uneven vertical gaps**: inconsistent spacing between equivalent
  list items (e.g. one bullet has noticeably more space below it than
  its siblings).
- **Right-margin overflow**: text or links that visibly press against
  or exceed the right margin (link labels cut off, lines extending
  past where the rest of the body ends).
- **Cramped or sparse density**: bullets so tightly packed they read
  as a wall, or so spread out that the page looks half-empty.
- **Headline / hierarchy issues**: section headings that aren't
  visually distinct from entry-name bold text, or contact-line items
  that wrap awkwardly.
- **Page count**: 3 pages is the target; flag if >= 4.

Respond with ONLY a single JSON object (no markdown fences, no
commentary, no preamble) matching this schema:

{
  "status": "passed" | "concerns" | "failed",
  "summary": "string -- one short sentence headline judgment",
  "issues": [
    {
      "severity": "info" | "warn" | "error",
      "category": "short tag (eg 'orphan_line', 'margin_overflow', 'section_orphan')",
      "message": "the specific issue, plain language. Cite the page number.",
      "suggestion": "string -- what to change (empty if obvious)"
    }
  ],
  "rationale": "string -- one paragraph on the major findings"
}

Status decision:
- ``failed``: any layout problem a recruiter would notice in the first
  5 seconds (page-3 of 3 has a single orphan line at top; section
  header on bottom of page 1 with content on page 2; visible right
  margin overflow on a link).
- ``concerns``: visually OK but tightenable (slightly uneven spacing,
  minor density issues).
- ``passed``: looks like a hand-formatted senior-eng resume.

Output the JSON and nothing else.
"""


_USER_PROMPT = (
    "Above are the rendered pages of a candidate's tailored resume PDF, in "
    "order. Review for VISUAL layout issues per the system prompt schema. "
    "Cite the specific page number where each issue appears."
)


def verify_pdf_visually(
    pdf_path: Path,
    *,
    oauth_token: str | None = None,
    model: str = DEFAULT_VISION_MODEL,
    dpi: int = DEFAULT_VISION_DPI,
    max_tokens: int = DEFAULT_VISION_MAX_TOKENS,
) -> VerificationResult | None:
    """Render ``pdf_path`` to images, send to the Anthropic vision API,
    return a structured verification result.

    Returns ``None`` when the vision pass can't run -- missing OAuth
    token, missing SDK, network error, malformed response. Callers
    should treat ``None`` as "no visual signal" and proceed with the
    text-only verification result. Never raises -- all failure modes
    log a warning and degrade silently.
    """
    return run_vision_verification(
        pdf_path,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_USER_PROMPT,
        oauth_token=oauth_token,
        model=model,
        dpi=dpi,
        max_tokens=max_tokens,
    )


__all__ = [
    "DEFAULT_VISION_DPI",
    "DEFAULT_VISION_MAX_TOKENS",
    "DEFAULT_VISION_MODEL",
    "SYSTEM_PROMPT",
    "verify_pdf_visually",
]
