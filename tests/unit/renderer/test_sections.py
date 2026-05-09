"""Section identification + content builder tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from resumeai.agent.models import TailoredBullet, TailoredResume, TailoredWorkEntry
from resumeai.context.models import Contact, Education
from resumeai.docs.models import Section, TemplateModel
from resumeai.renderer.sections import (
    find_section,
    render_certifications,
    render_education,
    render_skills,
    render_summary,
    render_work_history,
)

if TYPE_CHECKING:
    pass


def _template(*headings: tuple[str, int]) -> TemplateModel:
    sections = tuple(
        Section(
            heading=heading,
            level=level,
            start_index=10 * (i + 1),
            end_index=10 * (i + 2),
            paragraph_count=2,
        )
        for i, (heading, level) in enumerate(headings)
    )
    return TemplateModel(doc_id="d", title="t", revision_id="r", sections=sections)


# -- find_section ------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "kind"),
    [
        ("Summary", "summary"),
        ("About", "summary"),
        ("Profile", "summary"),
        ("Skills", "skills"),
        ("Technical Skills", "skills"),
        ("Tech Stack", "skills"),
        ("Experience", "work_history"),
        ("Work Experience", "work_history"),
        ("Professional Experience", "work_history"),
        ("Career", "work_history"),
        ("Education", "education"),
        ("Qualifications", "education"),
        ("Certifications", "certifications"),
        ("Certificates", "certifications"),
    ],
)
def test_find_section_matches_aliases_case_insensitively(heading: str, kind: str) -> None:
    template = _template((heading.upper(), 1))
    assert find_section(template, kind) is not None


def test_find_section_returns_none_for_no_match() -> None:
    template = _template(("Hobbies", 1))
    assert find_section(template, "summary") is None


def test_find_section_returns_none_for_unknown_kind() -> None:
    template = _template(("Summary", 1))
    assert find_section(template, "unknown_kind") is None


def test_find_section_returns_first_match_when_multiple_qualify() -> None:
    template = _template(("Skills", 1), ("Technical Skills", 1))
    section = find_section(template, "skills")
    assert section is not None
    assert section.heading == "Skills"


# -- content builders --------------------------------------------------------


def _contact() -> Contact:
    return Contact(email="alex@example.com")


def _resume(**overrides: object) -> TailoredResume:
    base = {"name": "Alex", "contact": _contact()}
    base.update(overrides)
    return TailoredResume(**base)  # type: ignore[arg-type]


def test_render_summary_returns_text_when_present() -> None:
    assert render_summary(_resume(summary="A summary.")) == "A summary."


def test_render_summary_returns_none_when_empty_or_whitespace() -> None:
    assert render_summary(_resume(summary="")) is None
    assert render_summary(_resume(summary="   \n  ")) is None


def test_render_skills_joins_with_commas() -> None:
    text = render_skills(_resume(skills=("Python", "Postgres", "Kubernetes")))
    assert text == "Python, Postgres, Kubernetes"


def test_render_skills_returns_none_when_empty() -> None:
    assert render_skills(_resume(skills=())) is None


def test_render_work_history_renders_each_entry_with_meta_and_bullets() -> None:
    entry = TailoredWorkEntry(
        company="Acme",
        title="Engineer",
        period="2022 → 2024",
        location="Melbourne",
        bullets=(TailoredBullet(text="Built it."),),
    )
    text = render_work_history(_resume(work_history=(entry,)))

    assert text is not None
    assert "Acme — Engineer" in text
    assert "Melbourne · 2022 → 2024" in text
    assert "• Built it." in text


def test_render_work_history_omits_meta_line_when_no_period_or_location() -> None:
    entry = TailoredWorkEntry(
        company="Acme",
        title="Engineer",
        bullets=(TailoredBullet(text="Built it."),),
    )
    text = render_work_history(_resume(work_history=(entry,)))
    assert text is not None
    # Heading line then bullet line, with no meta line in between.
    assert text.splitlines() == ["Acme — Engineer", "• Built it."]


def test_render_work_history_includes_only_period_when_no_location() -> None:
    entry = TailoredWorkEntry(
        company="Acme",
        title="Engineer",
        period="2024 → present",
        bullets=(TailoredBullet(text="Built it."),),
    )
    text = render_work_history(_resume(work_history=(entry,)))
    assert text is not None
    assert "2024 → present" in text
    assert " · " not in text


def test_render_work_history_separates_entries_with_blank_lines() -> None:
    entries = (
        TailoredWorkEntry(
            company="Acme",
            title="Engineer",
            bullets=(TailoredBullet(text="A."),),
        ),
        TailoredWorkEntry(
            company="Beta",
            title="Eng",
            bullets=(TailoredBullet(text="B."),),
        ),
    )
    text = render_work_history(_resume(work_history=entries))
    assert text is not None
    assert text == "Acme — Engineer\n• A.\n\nBeta — Eng\n• B."


def test_render_work_history_returns_none_when_empty() -> None:
    assert render_work_history(_resume()) is None


def test_render_education_renders_full_entry() -> None:
    edu = Education(
        institution="Sample U",
        degree="BEng",
        field="Software Engineering",
        year_start=2014,
        year_end=2017,
    )
    text = render_education(_resume(education=(edu,)))
    assert text == "Sample U — BEng · Software Engineering · 2014–2017"


def test_render_education_omits_field_when_missing() -> None:
    edu = Education(institution="U", degree="BEng", year_start=2018, year_end=2021)
    text = render_education(_resume(education=(edu,)))
    assert text == "U — BEng · 2018–2021"


def test_render_education_renders_open_ended_period() -> None:
    edu = Education(institution="U", degree="BEng", year_start=2024)
    text = render_education(_resume(education=(edu,)))
    assert text is not None
    assert "2024–present" in text


def test_render_education_renders_legacy_end_only_period() -> None:
    edu = Education(institution="U", degree="BEng", year_end=2010)
    text = render_education(_resume(education=(edu,)))
    assert text is not None
    assert "…–2010" in text


def test_render_education_omits_period_when_no_years() -> None:
    edu = Education(institution="U", degree="BEng")
    text = render_education(_resume(education=(edu,)))
    assert text == "U — BEng"


def test_render_education_returns_none_when_empty() -> None:
    assert render_education(_resume()) is None


def test_render_certifications_bullets_each_one() -> None:
    text = render_certifications(_resume(certifications=("AWS SAA", "GCP PCA")))
    assert text == "• AWS SAA\n• GCP PCA"


def test_render_certifications_returns_none_when_empty() -> None:
    assert render_certifications(_resume()) is None
