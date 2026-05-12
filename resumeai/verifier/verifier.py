"""Run a post-render LLM verification pass.

The verifier is a second LLM call that reviews the agent's output the way
a careful human would:

- Does every JD must-have show up somewhere in the tailored resume?
- Are the JD's required skills represented in the skills list?
- Do the bullets read as the candidate's authentic work, or do they
  feel templated / fabricated?
- Are there obvious quality issues (empty bullets, repeated phrases,
  contradictions, length problems)?

The verifier emits a structured :class:`VerificationResult` the run
detail page surfaces. ``status=FAILED`` means the user should review
before sending; ``CONCERNS`` means usable with caveats; ``PASSED`` means
clean.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from resumeai.verifier.models import (
    IssueSeverity,
    VerificationIssue,
    VerificationResult,
    VerificationStatus,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tailor_core.jd.models import JobRequirements
    from tailor_core.llm.client import LLMClient

    from resumeai.agent.models import TailoredResume


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


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class VerifierError(RuntimeError):
    """Raised when the verifier's response can't be parsed."""


TARGET_MAX_PAGES = 3


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
    raw = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, model=model)
    result = parse_verifier_response(raw)
    if pdf_path is not None:
        length_issue = _check_pdf_length(pdf_path)
        if length_issue is not None:
            result = _merge_issue(result, length_issue)
    return result


def _check_pdf_length(pdf_path: Path) -> VerificationIssue | None:
    """Programmatically check rendered PDF length against the target.

    Returns ``None`` when the PDF is at or under the target page count.
    Returns a ``warn``-severity issue when it overflows. Failures
    reading the PDF degrade silently (return ``None``) -- the LLM
    verifier already pinged on the run, no point fabricating a
    second failure on top.
    """
    from pypdf import PdfReader  # noqa: PLC0415
    from pypdf.errors import PdfReadError  # noqa: PLC0415

    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
    except (OSError, PdfReadError):
        return None
    if page_count <= TARGET_MAX_PAGES:
        return None
    return VerificationIssue(
        severity=IssueSeverity.WARN,
        category="page_overflow",
        message=(
            f"Rendered PDF is {page_count} pages; target is "
            f"≤{TARGET_MAX_PAGES} for a senior-eng resume."
        ),
        suggestion=(
            "Trim a bullet or two from the longest work-history entries, "
            "or compress Key Achievements (6→5 bullets). Re-run."
        ),
    )


def _merge_issue(result: VerificationResult, new_issue: VerificationIssue) -> VerificationResult:
    """Append a programmatic issue to an LLM-verifier result.

    Promotes the status if the new issue is more severe than the
    existing finding (a ``warn`` added to a ``passed`` result flips
    the result to ``concerns``).
    """
    bumped_status = result.status
    is_warn = new_issue.severity is IssueSeverity.WARN
    is_error = new_issue.severity is IssueSeverity.ERROR
    if is_warn and result.status is VerificationStatus.PASSED:
        bumped_status = VerificationStatus.CONCERNS
    elif is_error and result.status is not VerificationStatus.FAILED:
        bumped_status = VerificationStatus.FAILED
    return result.model_copy(
        update={
            "status": bumped_status,
            "issues": (*result.issues, new_issue),
        }
    )


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


def parse_verifier_response(raw: str) -> VerificationResult:
    """Parse the model's text response into a :class:`VerificationResult`."""
    text = raw.strip()
    if not text:
        raise VerifierError("verifier returned an empty response")

    payload = _FENCE_RE.sub(r"\1", text).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VerifierError(
            f"verifier response was not valid JSON: {exc.msg} — got {payload[:200]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise VerifierError(f"verifier response was not a JSON object — got {type(data).__name__}")

    try:
        return VerificationResult.model_validate(data)
    except ValidationError as exc:
        raise VerifierError(f"verifier response failed schema validation: {exc}") from exc


def fallback_concerns_result(reason: str) -> VerificationResult:
    """Synthesise a ``CONCERNS`` result when the verifier itself fails.

    Used by the orchestrator: if the LLM call errors or the response is
    malformed, we don't want to block the whole run on QC infrastructure
    — the user gets a CONCERNS-status run with the failure reason as the
    only issue, and can read the underlying tailored resume themselves.
    """
    return VerificationResult(
        status=VerificationStatus.CONCERNS,
        summary="Verifier itself failed — review the rendered doc manually.",
        issues=(
            VerificationIssue(
                severity=IssueSeverity.WARN,
                category="verifier_failure",
                message=reason,
                suggestion="Re-run; if this keeps happening, raise an issue on the repo.",
            ),
        ),
        rationale="QC pass couldn't complete; surfacing as CONCERNS rather than blocking.",
    )
