"""Run + RunEvent + TailorRequest validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from resumeai.runs.models import Run, RunEvent, RunStatus, TailorRequest


def test_tailor_request_accepts_url_only() -> None:
    req = TailorRequest(jd_url="https://example.com/jd")
    assert req.jd_url == "https://example.com/jd"


def test_tailor_request_accepts_text_only() -> None:
    req = TailorRequest(jd_text="Senior Engineer ...")
    assert req.jd_text == "Senior Engineer ..."


def test_tailor_request_rejects_both_set() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        TailorRequest(jd_url="https://x", jd_text="text")


def test_tailor_request_rejects_neither_set() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        TailorRequest()


def test_tailor_request_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        TailorRequest(jd_text="   \n  ")


def test_run_status_terminal_set() -> None:
    assert RunStatus.SUCCEEDED.is_terminal
    assert RunStatus.FAILED.is_terminal
    assert not RunStatus.PENDING.is_terminal
    assert not RunStatus.TAILORING.is_terminal


def test_run_round_trips_through_json() -> None:
    now = datetime(2026, 5, 9, tzinfo=UTC)
    run = Run(
        id="run_x",
        request=TailorRequest(jd_text="text"),
        status=RunStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    parsed = Run.model_validate_json(run.model_dump_json())
    assert parsed == run


def test_run_event_round_trips() -> None:
    event = RunEvent(
        run_id="run_x",
        status=RunStatus.TAILORING,
        detail="thinking",
        at=datetime(2026, 5, 9, tzinfo=UTC),
    )
    assert RunEvent.model_validate_json(event.model_dump_json()) == event
