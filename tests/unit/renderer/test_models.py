"""RenderResult + RenderDiff validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from resumeai.renderer.models import RenderDiff, RenderResult, RenderStatus


def test_render_diff_requires_non_empty_kind() -> None:
    with pytest.raises(ValidationError):
        RenderDiff(kind="", status=RenderStatus.SKIPPED_EMPTY)


def test_render_diff_round_trips() -> None:
    diff = RenderDiff(
        kind="summary",
        heading="Summary",
        status=RenderStatus.REPLACED,
        before_chars=42,
        after_chars=80,
    )
    assert RenderDiff.model_validate_json(diff.model_dump_json()) == diff


def test_render_result_requires_doc_id_and_url() -> None:
    with pytest.raises(ValidationError):
        RenderResult(doc_id="", doc_url="x")
    with pytest.raises(ValidationError):
        RenderResult(doc_id="x", doc_url="")


def test_render_result_round_trips() -> None:
    result = RenderResult(
        doc_id="d",
        doc_url="https://docs.google.com/document/d/d/edit",
        pdf_bytes=b"%PDF",
        diffs=(RenderDiff(kind="summary", status=RenderStatus.SKIPPED_EMPTY),),
    )
    assert RenderResult.model_validate_json(result.model_dump_json()) == result


def test_render_status_values_are_stable() -> None:
    """Pinning the wire-level enum values; the API will serialise these."""
    assert RenderStatus.REPLACED.value == "replaced"
    assert RenderStatus.SKIPPED_EMPTY.value == "skipped_empty"
    assert RenderStatus.NOT_FOUND.value == "not_found"
