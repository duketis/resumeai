"""TemplateModel + Section validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from resumeai.docs.models import Section, TemplateModel


def test_section_accepts_valid_inputs() -> None:
    section = Section(heading="Summary", level=1, start_index=1, end_index=20, paragraph_count=2)
    assert section.heading == "Summary"
    assert section.level == 1


def test_section_rejects_level_below_one() -> None:
    with pytest.raises(ValidationError):
        Section(heading="x", level=0, start_index=1, end_index=2, paragraph_count=1)


def test_section_rejects_level_above_six() -> None:
    with pytest.raises(ValidationError):
        Section(heading="x", level=7, start_index=1, end_index=2, paragraph_count=1)


def test_section_rejects_zero_index() -> None:
    with pytest.raises(ValidationError):
        Section(heading="x", level=1, start_index=0, end_index=2, paragraph_count=1)


def test_section_is_frozen() -> None:
    section = Section(heading="x", level=1, start_index=1, end_index=2, paragraph_count=1)
    with pytest.raises(ValidationError):
        section.heading = "y"


def test_template_model_rejects_empty_doc_id() -> None:
    with pytest.raises(ValidationError):
        TemplateModel(doc_id="", title="t", revision_id="r", sections=())


def test_template_model_round_trips() -> None:
    template = TemplateModel(
        doc_id="d",
        title="t",
        revision_id="r",
        sections=(Section(heading="h", level=1, start_index=1, end_index=2, paragraph_count=1),),
    )
    parsed = TemplateModel.model_validate_json(template.model_dump_json())
    assert parsed == template
