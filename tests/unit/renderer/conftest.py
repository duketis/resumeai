"""Shared fixtures for the renderer tests."""

from __future__ import annotations

from typing import Any

import pytest

from resumeai.agent.models import (
    TailoredBullet,
    TailoredResume,
    TailoredWorkEntry,
)
from resumeai.context.models import Contact, Education


@pytest.fixture
def sample_tailored() -> TailoredResume:
    return TailoredResume(
        name="Alex Sample",
        headline="Senior backend engineer",
        contact=Contact(
            email="alex@example.com",
            phone="+61 400 000 000",
            location="Melbourne, AU",
        ),
        summary="Backend engineer with 5+ years building data-heavy systems.",
        skills=("Python", "Postgres", "Kubernetes"),
        work_history=(
            TailoredWorkEntry(
                company="Acme Corp",
                title="Senior Software Engineer",
                period="2022-03 → 2024-09",
                location="Melbourne, AU",
                source_slug="01-acme",
                bullets=(
                    TailoredBullet(text="Designed dedup pipeline.", source_slug="01-acme"),
                    TailoredBullet(text="Migrated to Kubernetes.", source_slug="01-acme"),
                ),
            ),
        ),
        education=(
            Education(
                institution="Sample U",
                degree="BEng",
                field="Software Engineering",
                year_start=2014,
                year_end=2017,
            ),
        ),
        certifications=("AWS SAA",),
        rationale="Reordered skills so Python/Postgres lead.",
    )


@pytest.fixture
def template_doc_with_all_sections() -> dict[str, Any]:
    """A ``documents.get`` response shaped like a real resume template.

    Uses the ``make_doc`` helper logic — kept inline here to avoid a
    cross-package conftest import.
    """
    return _make_doc(
        [
            ("Alex Sample", None),
            ("alex@example.com  •  Melbourne", None),
            ("", None),
            ("Summary", "HEADING_1"),
            ("Old summary text that will be replaced.", None),
            ("", None),
            ("Skills", "HEADING_1"),
            ("Old skills list.", None),
            ("", None),
            ("Experience", "HEADING_1"),
            ("Old company role.", None),
            ("- Old bullet 1", None),
            ("- Old bullet 2", None),
            ("", None),
            ("Education", "HEADING_1"),
            ("Old education entry.", None),
            ("", None),
            ("Certifications", "HEADING_1"),
            ("Old cert", None),
        ]
    )


def _make_doc(
    paragraphs: list[tuple[str, str | None]],
) -> dict[str, Any]:
    body_content: list[dict[str, Any]] = []
    cursor = 1
    for text, style in paragraphs:
        content = text + "\n"
        elem: dict[str, Any] = {
            "startIndex": cursor,
            "endIndex": cursor + len(content),
            "paragraph": {
                "elements": [
                    {
                        "startIndex": cursor,
                        "endIndex": cursor + len(content),
                        "textRun": {"content": content, "textStyle": {}},
                    }
                ],
            },
        }
        if style:
            elem["paragraph"]["paragraphStyle"] = {"namedStyleType": style}
        body_content.append(elem)
        cursor += len(content)
    return {
        "documentId": "tmpl-doc-id",
        "title": "Resume Template",
        "revisionId": "rev-1",
        "body": {"content": body_content},
    }
