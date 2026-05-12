"""Verifier prompt + parser + orchestrator-friendly fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from tailor_core.llm.client import FakeLLMClient

from resumeai.agent.models import TailoredBullet, TailoredResume, TailoredWorkEntry
from resumeai.context.models import Contact
from resumeai.jd.models import JobRequirements, RoleType, Seniority
from resumeai.verifier.models import (
    IssueSeverity,
    VerificationResult,
    VerificationStatus,
)
from resumeai.verifier.verifier import (
    SYSTEM_PROMPT,
    VerifierError,
    build_verifier_prompt,
    fallback_concerns_result,
    parse_verifier_response,
    verify_resume,
)

if TYPE_CHECKING:
    pass


_PASSED_PAYLOAD: dict[str, Any] = {
    "status": "passed",
    "summary": "Output cleanly addresses the JD.",
    "issues": [],
    "rationale": "All must-haves present, skills ordered, bullets quantified.",
}

_CONCERNS_PAYLOAD: dict[str, Any] = {
    "status": "concerns",
    "summary": "Two warns surfaced.",
    "issues": [
        {
            "severity": "warn",
            "category": "missing_skill",
            "message": "Postgres not in skills list.",
            "suggestion": "Add Postgres explicitly.",
        }
    ],
    "rationale": "Skills section trimmed too aggressively.",
}


def _jd() -> JobRequirements:
    return JobRequirements(
        title="Senior Engineer",
        company="Acme",
        role_type=RoleType.ENGINEERING,
        seniority=Seniority.SENIOR,
        required_skills=("Python", "Postgres"),
        must_haves=("AU citizen",),
    )


def _tailored() -> TailoredResume:
    return TailoredResume(
        name="Alex",
        contact=Contact(email="alex@example.com"),
        summary="Backend engineer.",
        skills=("Python",),
        work_history=(
            TailoredWorkEntry(
                company="Acme",
                title="Engineer",
                bullets=(TailoredBullet(text="Built it."),),
            ),
        ),
    )


# -- system prompt & prompt builder --------------------------------------


def test_system_prompt_documents_required_status_values() -> None:
    for value in ('"passed"', '"concerns"', '"failed"'):
        assert value in SYSTEM_PROMPT


def test_system_prompt_documents_required_severity_values() -> None:
    for value in ('"info"', '"warn"', '"error"'):
        assert value in SYSTEM_PROMPT


def test_build_verifier_prompt_pairs_jd_with_tailored() -> None:
    prompt = build_verifier_prompt(_jd(), _tailored())
    assert "# JOB DESCRIPTION" in prompt
    assert "# TAILORED RESUME" in prompt
    assert "# OUTPUT" in prompt
    assert "Senior Engineer" in prompt  # JD title
    assert "alex@example.com" in prompt  # tailored contact


# -- parser --------------------------------------------------------------


def test_parser_accepts_passed_payload() -> None:
    result = parse_verifier_response(json.dumps(_PASSED_PAYLOAD))
    assert result.status is VerificationStatus.PASSED
    assert result.issues == ()


def test_parser_accepts_concerns_payload_with_issues() -> None:
    result = parse_verifier_response(json.dumps(_CONCERNS_PAYLOAD))
    assert result.status is VerificationStatus.CONCERNS
    assert len(result.issues) == 1
    assert result.issues[0].severity is IssueSeverity.WARN
    assert result.issues[0].category == "missing_skill"


def test_parser_strips_markdown_fence() -> None:
    fenced = f"```json\n{json.dumps(_PASSED_PAYLOAD)}\n```"
    result = parse_verifier_response(fenced)
    assert result.status is VerificationStatus.PASSED


def test_parser_strips_unlabelled_fence() -> None:
    fenced = f"```\n{json.dumps(_PASSED_PAYLOAD)}\n```"
    assert parse_verifier_response(fenced).status is VerificationStatus.PASSED


def test_parser_rejects_empty_response() -> None:
    with pytest.raises(VerifierError, match="empty"):
        parse_verifier_response("")


def test_parser_rejects_invalid_json() -> None:
    with pytest.raises(VerifierError, match="not valid JSON"):
        parse_verifier_response("{not json")


def test_parser_rejects_non_object() -> None:
    with pytest.raises(VerifierError, match="not a JSON object"):
        parse_verifier_response("[1, 2]")


def test_parser_rejects_schema_violation() -> None:
    bad = dict(_PASSED_PAYLOAD)
    del bad["status"]
    with pytest.raises(VerifierError, match="schema validation"):
        parse_verifier_response(json.dumps(bad))


# -- verify_resume orchestrator ------------------------------------------


def test_verify_resume_passes_system_prompt_and_user_prompt() -> None:
    llm = FakeLLMClient(default_response=json.dumps(_PASSED_PAYLOAD))
    result = verify_resume(_jd(), _tailored(), llm)

    assert result.status is VerificationStatus.PASSED
    assert llm.calls
    system, user, model = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "TAILORED RESUME" in user
    assert model is None


def test_verify_resume_forwards_model_override() -> None:
    llm = FakeLLMClient(default_response=json.dumps(_PASSED_PAYLOAD))
    verify_resume(_jd(), _tailored(), llm, model="claude-sonnet-4-6")
    assert llm.calls[0][2] == "claude-sonnet-4-6"


def test_verify_resume_propagates_parse_errors() -> None:
    llm = FakeLLMClient(default_response="not valid json")
    with pytest.raises(VerifierError):
        verify_resume(_jd(), _tailored(), llm)


# -- fallback ------------------------------------------------------------


def test_fallback_concerns_result_synthesises_warn_issue() -> None:
    result = fallback_concerns_result("verifier blew up")
    assert result.status is VerificationStatus.CONCERNS
    assert len(result.issues) == 1
    assert result.issues[0].severity is IssueSeverity.WARN
    assert result.issues[0].category == "verifier_failure"
    assert "verifier blew up" in result.issues[0].message


def test_fallback_concerns_result_round_trips() -> None:
    result = fallback_concerns_result("boom")
    assert VerificationResult.model_validate_json(result.model_dump_json()) == result


# -- programmatic page-count check --------------------------------------


def _write_pdf(path: Path, *, pages: int) -> None:
    """Stage a real PDF with the requested page count for the
    pypdf-based length check."""
    from pypdf import PdfWriter  # noqa: PLC0415

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    with path.open("wb") as fh:
        writer.write(fh)


def test_verify_resume_flags_overflowing_pdf_as_concern(tmp_path: Path) -> None:
    """A 4-page rendered PDF triggers a page_overflow warn issue and
    promotes a PASSED result to CONCERNS."""
    pdf = tmp_path / "resume.pdf"
    _write_pdf(pdf, pages=4)

    llm = FakeLLMClient(default_response=json.dumps(_PASSED_PAYLOAD))
    result = verify_resume(_jd(), _tailored(), llm, pdf_path=pdf)

    assert result.status is VerificationStatus.CONCERNS
    overflow_issues = [i for i in result.issues if i.category == "page_overflow"]
    assert len(overflow_issues) == 1
    assert "4 pages" in overflow_issues[0].message
    assert overflow_issues[0].severity is IssueSeverity.WARN


def test_verify_resume_no_overflow_issue_when_pdf_under_target(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "resume.pdf"
    _write_pdf(pdf, pages=3)

    llm = FakeLLMClient(default_response=json.dumps(_PASSED_PAYLOAD))
    result = verify_resume(_jd(), _tailored(), llm, pdf_path=pdf)

    assert result.status is VerificationStatus.PASSED
    assert all(i.category != "page_overflow" for i in result.issues)


def test_verify_resume_silently_skips_when_pdf_unreadable(
    tmp_path: Path,
) -> None:
    """Corrupted PDF -> page check is silently skipped (no spurious warn)."""
    pdf = tmp_path / "resume.pdf"
    pdf.write_bytes(b"not a real pdf")

    llm = FakeLLMClient(default_response=json.dumps(_PASSED_PAYLOAD))
    result = verify_resume(_jd(), _tailored(), llm, pdf_path=pdf)

    # LLM verifier still passed; no page_overflow issue added because we
    # couldn't read the file. PASSED stays PASSED.
    assert result.status is VerificationStatus.PASSED


def test_verify_resume_without_pdf_path_skips_length_check() -> None:
    """Backwards-compat: callers that don't pass ``pdf_path`` get the same
    behaviour as before (LLM verifier only, no programmatic check)."""
    llm = FakeLLMClient(default_response=json.dumps(_PASSED_PAYLOAD))
    result = verify_resume(_jd(), _tailored(), llm)
    assert result.status is VerificationStatus.PASSED
    assert all(i.category != "page_overflow" for i in result.issues)
