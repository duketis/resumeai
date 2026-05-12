"""Run a post-render LLM verification pass on a tailored resume.

The verifier is a second LLM call that reviews the agent's output the way
a careful human would:

- Does every JD must-have show up somewhere in the tailored resume?
- Are the JD's required skills represented in the skills list?
- Do the bullets read as the candidate's authentic work, or do they
  feel templated / fabricated?
- Are there obvious quality issues (empty bullets, repeated phrases,
  contradictions, length problems)?

The verifier emits a :class:`VerificationResult` the run detail page
surfaces. ``status=FAILED`` means the user should review before sending;
``CONCERNS`` means usable with caveats; ``PASSED`` means clean.

All the heavy lifting (LLM call, JSON parse, schema validate, page-count
check, fallback synthesis) lives in :mod:`tailor_core.verifier.scaffold`;
this module is the resume-flavoured ``SYSTEM_PROMPT`` + a small wrapper
that wires the inputs together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tailor_core.verifier.scaffold import (
    VerifierError,
    check_pdf_length,
    evaluate_judgement,
    fallback_concerns_result,
    merge_issue,
    parse_verifier_response,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tailor_core.jd.models import JobRequirements
    from tailor_core.llm.client import LLMClient
    from tailor_core.verifier.models import VerificationResult

    from resumeai.agent.models import TailoredResume


# Re-export so resumeai callers don't have to know the scaffolding lives
# in tailor_core. Removing this re-export would force the orchestrator
# and tests to import VerifierError / fallback_concerns_result / etc.
# from the lib directly, which is fine but currently noisy.
__all__ = [
    "SYSTEM_PROMPT",
    "TARGET_MAX_PAGES",
    "VerifierError",
    "build_verifier_prompt",
    "fallback_concerns_result",
    "parse_verifier_response",
    "verify_resume",
]


SYSTEM_PROMPT = """\
You are reviewing a TAILORED RESUME against a JOB DESCRIPTION on behalf
of the candidate. Your job is QC: catch fabrications, missing required
items, empty bullets, weird tone, JD-mismatch problems — anything that
would embarrass the candidate or hurt their chances.

Respond with ONLY a single JSON object (no markdown fences, no commentary,
no preamble) matching this exact schema:

{
  "status": "passed" | "concerns" | "failed",
  "summary": "string — one short sentence explaining the headline judgment",
  "issues": [
    {
      "severity": "info" | "warn" | "error",
      "category": "string — short tag (eg 'missing_must_have', 'fabrication', 'empty_bullet')",
      "message": "string — the specific issue, in plain language",
      "suggestion": "string — what to change (empty if obvious)"
    }
  ],
  "rationale": "string — one paragraph on the major findings"
}

Status decision:
- ``failed``: any ``error``-severity issue. Things like fabricated
  employer / metric, missing JD must-have that the candidate clearly
  has elsewhere, bullets that read as junior when the role is senior.
- ``concerns``: ``warn``-severity issues only. Worth flagging, not
  blocking — eg "skills section could surface Postgres earlier".
- ``passed``: no issues, or only ``info``-severity nice-to-haves.

Hard rules:
- If a JD must-have isn't covered in the tailored output AND the
  candidate's context shows they have it, that's a ``warn`` (the
  agent missed something). If the candidate's context doesn't show
  it at all, that's ``info`` (genuinely a gap, not the agent's
  fault).
- If a tailored bullet contains a number / employer / project that
  isn't in the candidate's context (master template + uploaded files),
  that's a fabrication and an ``error``.
- If a bullet is empty, vague ("did stuff"), or repeats the same idea
  as another bullet, that's a ``warn``.
- Never invent issues. If everything looks fine, return
  ``status=passed`` with an empty issues list.
- Output the JSON object and nothing else.
"""


TARGET_MAX_PAGES = 3

_OVERFLOW_SUGGESTION = (
    "Trim a bullet or two from the longest work-history entries, "
    "or compress Key Achievements (6→5 bullets). Re-run."
)


def verify_resume(
    jd: JobRequirements,
    tailored: TailoredResume,
    llm: LLMClient,
    *,
    model: str | None = None,
    pdf_path: Path | None = None,
) -> VerificationResult:
    """Run the QC pass and return a structured :class:`VerificationResult`.

    When ``pdf_path`` is provided we additionally count pages with
    ``pypdf`` and append a programmatic length-check issue if the
    rendered output exceeds :data:`TARGET_MAX_PAGES`. This catches
    "agent wrote a great resume but the PDF is 4 pages" before the
    user opens Preview.
    """
    user_prompt = build_verifier_prompt(jd, tailored)
    result = evaluate_judgement(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        llm=llm,
        model=model,
    )
    if pdf_path is not None:
        length_issue = check_pdf_length(
            pdf_path,
            target_max_pages=TARGET_MAX_PAGES,
            overflow_suggestion=_OVERFLOW_SUGGESTION,
        )
        if length_issue is not None:
            result = merge_issue(result, length_issue)
    return result


def build_verifier_prompt(jd: JobRequirements, tailored: TailoredResume) -> str:
    """Compose the user-prompt that pairs the JD with the tailored output."""
    return "\n\n".join(
        [
            "# JOB DESCRIPTION",
            jd.model_dump_json(indent=2),
            "# TAILORED RESUME (the agent's output to review)",
            tailored.model_dump_json(indent=2),
            "# OUTPUT",
            "Return the verification JSON per the schema in the system prompt.",
        ]
    )
