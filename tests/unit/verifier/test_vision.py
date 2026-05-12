"""Vision verifier tests — PDF rasterisation + Anthropic vision API call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from resumeai.verifier.models import IssueSeverity, VerificationStatus
from resumeai.verifier.vision import (
    SYSTEM_PROMPT,
    VisionVerifierError,
    _parse_response,
    verify_pdf_visually,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# -- fixtures --------------------------------------------------------------


def _write_pdf(path: Path, *, pages: int = 1) -> None:
    from pypdf import PdfWriter  # noqa: PLC0415

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    with path.open("wb") as fh:
        writer.write(fh)


_PASSED_VISION_RESPONSE = json.dumps(
    {
        "status": "passed",
        "summary": "clean layout",
        "issues": [],
        "rationale": "everything looks fine.",
    }
)

_CONCERNS_VISION_RESPONSE = json.dumps(
    {
        "status": "concerns",
        "summary": "one widow line on page 3",
        "issues": [
            {
                "severity": "warn",
                "category": "orphan_line",
                "message": "Page 3 starts with 3 words finishing a bullet from page 2.",
                "suggestion": "Shorten the bullet on page 2 to fit on one page.",
            }
        ],
        "rationale": "small layout issue.",
    }
)


# -- _parse_response -------------------------------------------------------


def test_parse_response_decodes_passed_json() -> None:
    result = _parse_response(_PASSED_VISION_RESPONSE)
    assert result.status is VerificationStatus.PASSED
    assert result.issues == ()


def test_parse_response_decodes_issue_list() -> None:
    result = _parse_response(_CONCERNS_VISION_RESPONSE)
    assert result.status is VerificationStatus.CONCERNS
    assert len(result.issues) == 1
    assert result.issues[0].category == "orphan_line"
    assert result.issues[0].severity is IssueSeverity.WARN


def test_parse_response_strips_markdown_fences() -> None:
    fenced = f"```json\n{_PASSED_VISION_RESPONSE}\n```"
    result = _parse_response(fenced)
    assert result.status is VerificationStatus.PASSED


def test_parse_response_raises_on_empty() -> None:
    with pytest.raises(VisionVerifierError, match="empty"):
        _parse_response("")


def test_parse_response_raises_on_invalid_json() -> None:
    with pytest.raises(VisionVerifierError, match="not valid JSON"):
        _parse_response("{not json")


def test_parse_response_raises_on_non_object() -> None:
    with pytest.raises(VisionVerifierError, match="not a JSON object"):
        _parse_response("[1, 2, 3]")


def test_parse_response_raises_on_schema_mismatch() -> None:
    bad = json.dumps({"status": "passed"})  # missing required-ish fields are fine,
    # but unknown status flips validation
    bad_bad = json.dumps({"status": "weird"})
    with pytest.raises(VisionVerifierError, match="schema validation"):
        _parse_response(bad_bad)
    # Sanity check that the "passed-only" body still parses.
    _parse_response(bad)


# -- verify_pdf_visually skip paths ----------------------------------------


def test_verify_pdf_visually_returns_none_without_oauth_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    assert verify_pdf_visually(pdf) is None


def test_verify_pdf_visually_returns_none_on_unreadable_pdf(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"not a real pdf")
    assert verify_pdf_visually(pdf, oauth_token="sk-ant-oat01-fake") is None


def test_verify_pdf_visually_returns_none_when_api_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    # Force the API call to blow up; the verifier degrades silently.
    mocker.patch(
        "resumeai.verifier.vision._call_vision_api",
        side_effect=VisionVerifierError("API down"),
    )
    assert verify_pdf_visually(pdf, oauth_token="sk-ant-oat01-fake") is None


def test_verify_pdf_visually_returns_none_when_response_is_malformed(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    mocker.patch(
        "resumeai.verifier.vision._call_vision_api",
        return_value="not valid json garbage",
    )
    assert verify_pdf_visually(pdf, oauth_token="sk-ant-oat01-fake") is None


def test_verify_pdf_visually_returns_none_when_rasterisation_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    mocker.patch(
        "resumeai.verifier.vision._render_pdf_to_pngs",
        side_effect=VisionVerifierError("pdfium broken"),
    )
    assert verify_pdf_visually(pdf, oauth_token="sk-ant-oat01-fake") is None


# -- verify_pdf_visually happy path ----------------------------------------


def test_verify_pdf_visually_returns_parsed_result_when_api_succeeds(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf, pages=2)
    mocker.patch(
        "resumeai.verifier.vision._call_vision_api",
        return_value=_CONCERNS_VISION_RESPONSE,
    )
    result = verify_pdf_visually(pdf, oauth_token="sk-ant-oat01-fake")
    assert result is not None
    assert result.status is VerificationStatus.CONCERNS
    assert result.issues[0].category == "orphan_line"


def test_verify_pdf_visually_uses_env_var_token(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``oauth_token`` arg is absent the env var is read."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-env")
    pdf = tmp_path / "r.pdf"
    _write_pdf(pdf)
    api_mock = mocker.patch(
        "resumeai.verifier.vision._call_vision_api",
        return_value=_PASSED_VISION_RESPONSE,
    )
    verify_pdf_visually(pdf)
    assert api_mock.call_args.kwargs["oauth_token"] == "sk-ant-oat01-env"


# -- system prompt sanity --------------------------------------------------


def test_system_prompt_documents_required_schema_fields() -> None:
    """The system prompt is the contract; if a field name drifts the
    response parser will reject downstream."""
    for field in (
        '"status"',
        '"summary"',
        '"issues"',
        '"severity"',
        '"category"',
        '"message"',
        '"suggestion"',
        '"rationale"',
    ):
        assert field in SYSTEM_PROMPT, f"missing {field}"


def test_system_prompt_calls_out_layout_concerns() -> None:
    """Sanity that the prompt actually asks the model about layout."""
    assert "Orphan lines" in SYSTEM_PROMPT
    assert "Right-margin overflow" in SYSTEM_PROMPT or "margin" in SYSTEM_PROMPT
    assert "Page count" in SYSTEM_PROMPT or "page count" in SYSTEM_PROMPT.lower()
