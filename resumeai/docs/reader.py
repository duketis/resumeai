"""Turn a raw Google Docs ``documents.get`` response into a TemplateModel.

The reader walks the body's structural elements once, identifies heading
paragraphs from ``paragraphStyle.namedStyleType``, and emits a Section for
each. A section ends when the next heading of *equal-or-higher* level is
encountered (so "Experience > Acme Corp > 2020-2024" produces three nested
sections at levels 1, 2, 3 — the renderer can flatten if it doesn't care).
"""

from __future__ import annotations

from typing import Any

from resumeai.docs.models import Section, TemplateModel

# Map Google's namedStyleType strings to our 1-6 heading levels.
_HEADING_LEVELS: dict[str, int] = {
    "HEADING_1": 1,
    "HEADING_2": 2,
    "HEADING_3": 3,
    "HEADING_4": 4,
    "HEADING_5": 5,
    "HEADING_6": 6,
    "TITLE": 1,
    "SUBTITLE": 2,
}


class ReaderError(ValueError):
    """Raised when the input doesn't look like a documents.get response."""


def read_template(raw: dict[str, Any]) -> TemplateModel:
    """Parse a ``documents.get`` response into a :class:`TemplateModel`."""
    doc_id = _required(raw, "documentId")
    title = _required(raw, "title")
    revision_id = _required(raw, "revisionId")

    body = raw.get("body")
    if not isinstance(body, dict):
        raise ReaderError("response missing 'body' object")
    body_end = _document_end_index(raw)

    headings = list(_iter_headings(body))
    if not headings:
        return TemplateModel(doc_id=doc_id, title=title, revision_id=revision_id, sections=())

    sections: list[Section] = []
    for index, current in enumerate(headings):
        end_index = _section_end(headings, index, body_end)
        paragraph_count = _count_paragraphs_in_range(body, current.start_index, end_index)
        sections.append(
            Section(
                heading=current.text,
                level=current.level,
                start_index=current.start_index,
                end_index=end_index,
                paragraph_count=paragraph_count,
            )
        )

    return TemplateModel(
        doc_id=doc_id,
        title=title,
        revision_id=revision_id,
        sections=tuple(sections),
    )


# -- helpers -----------------------------------------------------------------


class _RawHeading:
    """Tiny mutable record to keep the heading walk readable."""

    __slots__ = ("level", "start_index", "text")

    def __init__(self, *, text: str, level: int, start_index: int) -> None:
        self.text = text
        self.level = level
        self.start_index = start_index


def _required(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ReaderError(f"response missing required string field {key!r}")
    return value


def _document_end_index(raw: dict[str, Any]) -> int:
    """The body's end_index = the largest endIndex we see in body.content."""
    body = raw.get("body", {})
    elements = body.get("content", []) if isinstance(body, dict) else []
    end = 1
    if isinstance(elements, list):
        for el in elements:
            if isinstance(el, dict):
                ei = el.get("endIndex")
                if isinstance(ei, int) and ei > end:
                    end = ei
    return end


def _iter_headings(body: dict[str, Any]) -> list[_RawHeading]:
    """Yield headings in document order."""
    headings: list[_RawHeading] = []
    elements = body.get("content", [])
    if not isinstance(elements, list):
        return headings

    for el in elements:
        if not isinstance(el, dict):
            continue
        paragraph = el.get("paragraph")
        if not isinstance(paragraph, dict):
            continue
        style = paragraph.get("paragraphStyle")
        if not isinstance(style, dict):
            continue
        named = style.get("namedStyleType")
        if not isinstance(named, str) or named not in _HEADING_LEVELS:
            continue

        text = _paragraph_text(paragraph).strip()
        if not text:
            continue

        start_index = el.get("startIndex")
        if not isinstance(start_index, int):
            continue

        headings.append(
            _RawHeading(text=text, level=_HEADING_LEVELS[named], start_index=start_index)
        )

    return headings


def _paragraph_text(paragraph: dict[str, Any]) -> str:
    elements = paragraph.get("elements", [])
    if not isinstance(elements, list):
        return ""
    parts: list[str] = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        text_run = el.get("textRun")
        if isinstance(text_run, dict):
            content = text_run.get("content", "")
            if isinstance(content, str):
                parts.append(content)
    return "".join(parts)


def _section_end(headings: list[_RawHeading], index: int, body_end: int) -> int:
    """End index of the section starting at ``headings[index]``.

    The section ends at the start of the next heading whose level is <= the
    current section's level (i.e. next sibling or any ancestor). If none
    exists, the section runs to the end of the body.
    """
    current_level = headings[index].level
    for next_idx in range(index + 1, len(headings)):
        if headings[next_idx].level <= current_level:
            return headings[next_idx].start_index
    return body_end


def _count_paragraphs_in_range(body: dict[str, Any], start: int, end: int) -> int:
    """Count paragraph elements whose startIndex falls in [start, end).

    By the time this is reached, ``body.content`` is known to be a list
    (otherwise :func:`_iter_headings` would have returned nothing and we'd
    short-circuit before getting here), so the inner type-guards on the
    elements are the only ones we need.
    """
    count = 0
    elements = body["content"]
    for el in elements:
        if not isinstance(el, dict):
            continue
        if "paragraph" not in el:
            continue
        si = el.get("startIndex")
        if not isinstance(si, int):
            continue
        if start <= si < end:
            count += 1
    return count
