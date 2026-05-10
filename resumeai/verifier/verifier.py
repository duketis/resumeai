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
    from resumeai.agent.models import TailoredResume
    from resumeai.jd.models import JobRequirements
    from resumeai.llm.client import LLMClient


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


def verify_resume(
    jd: JobRequirements,
    tailored: TailoredResume,
    llm: LLMClient,
    *,
    model: str | None = None,
) -> VerificationResult:
    """Run the QC pass and return a structured :class:`VerificationResult`."""
    user_prompt = build_verifier_prompt(jd, tailored)
    raw = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, model=model)
    return parse_verifier_response(raw)


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
