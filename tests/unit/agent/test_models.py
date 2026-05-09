"""TailoredResume + nested model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from resumeai.agent.models import TailoredBullet, TailoredResume, TailoredWorkEntry
from resumeai.context.models import Contact


def _contact() -> Contact:
    return Contact(email="alex@example.com")


def test_tailored_bullet_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        TailoredBullet(text="")


def test_tailored_bullet_records_optional_source() -> None:
    bullet = TailoredBullet(text="Built it.", source_slug="01-acme")
    assert bullet.source_slug == "01-acme"


def test_tailored_work_entry_requires_company_and_title() -> None:
    with pytest.raises(ValidationError):
        TailoredWorkEntry(company="", title="Eng")
    with pytest.raises(ValidationError):
        TailoredWorkEntry(company="Acme", title="")


def test_tailored_resume_requires_non_empty_name() -> None:
    with pytest.raises(ValidationError):
        TailoredResume(name="", contact=_contact())


def test_tailored_resume_round_trips_through_json() -> None:
    resume = TailoredResume(
        name="Alex",
        headline="Eng",
        contact=_contact(),
        summary="...",
        skills=("Python",),
        work_history=(
            TailoredWorkEntry(
                company="Acme",
                title="Eng",
                bullets=(TailoredBullet(text="Did stuff.", source_slug="01-acme"),),
                source_slug="01-acme",
            ),
        ),
        rationale="Because.",
    )
    parsed = TailoredResume.model_validate_json(resume.model_dump_json())
    assert parsed == resume


def test_tailored_resume_is_frozen() -> None:
    resume = TailoredResume(name="Alex", contact=_contact())
    with pytest.raises(ValidationError):
        resume.name = "Renamed"
