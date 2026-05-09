"""Shared fixtures for docs tests.

The :func:`make_doc` helper builds a ``documents.get``-shaped dict from a
sequence of ``(text, style)`` paragraphs, auto-incrementing the character
indices so tests don't have to do arithmetic.
"""

from __future__ import annotations

from typing import Any


def make_doc(
    paragraphs: list[tuple[str, str | None]],
    *,
    doc_id: str = "doc-id",
    title: str = "Resume Template",
    revision_id: str = "rev-1",
) -> dict[str, Any]:
    """Build a fake ``documents.get`` response.

    ``paragraphs`` is a list of ``(text, named_style_type | None)``. When
    ``named_style_type`` is ``None`` the paragraph carries no style block
    (so the reader treats it as body text, not a heading).
    """
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
        "documentId": doc_id,
        "title": title,
        "revisionId": revision_id,
        "body": {"content": body_content},
    }


SAMPLE_RESUME_PARAGRAPHS: list[tuple[str, str | None]] = [
    ("Jonathan Duketis", None),
    ("jonathanmarkduketis@example.com  •  Melbourne, AU", None),
    ("", None),
    ("Summary", "HEADING_1"),
    ("Senior software engineer with 5 years of experience...", None),
    ("", None),
    ("Experience", "HEADING_1"),
    ("Acme Corp — Senior Engineer", "HEADING_2"),
    ("Built a thing.", None),
    ("Built another thing.", None),
    ("Beta Inc — Engineer", "HEADING_2"),
    ("Did stuff.", None),
    ("", None),
    ("Skills", "HEADING_1"),
    ("Python, TypeScript, Rust", None),
    ("", None),
    ("Education", "HEADING_1"),
    ("Bachelor of Engineering, RMIT, 2018", None),
]
