"""Tests for the template reader.

Drive the reader from synthetic ``documents.get`` responses built via the
``make_doc`` helper in conftest.py.
"""

from __future__ import annotations

import pytest

from resumeai.docs.reader import ReaderError, read_template
from tests.unit.docs.conftest import SAMPLE_RESUME_PARAGRAPHS, make_doc


def test_returns_top_level_metadata() -> None:
    doc = make_doc(SAMPLE_RESUME_PARAGRAPHS, doc_id="d1", title="Resume", revision_id="r9")

    template = read_template(doc)

    assert template.doc_id == "d1"
    assert template.title == "Resume"
    assert template.revision_id == "r9"


def test_identifies_top_level_sections() -> None:
    doc = make_doc(SAMPLE_RESUME_PARAGRAPHS)

    template = read_template(doc)

    headings = [(s.heading, s.level) for s in template.sections]
    assert ("Summary", 1) in headings
    assert ("Experience", 1) in headings
    assert ("Skills", 1) in headings
    assert ("Education", 1) in headings


def test_includes_nested_subsections() -> None:
    doc = make_doc(SAMPLE_RESUME_PARAGRAPHS)

    template = read_template(doc)

    headings = [(s.heading, s.level) for s in template.sections]
    assert ("Acme Corp — Senior Engineer", 2) in headings
    assert ("Beta Inc — Engineer", 2) in headings


def test_section_end_index_stops_at_next_sibling_or_higher() -> None:
    doc = make_doc(SAMPLE_RESUME_PARAGRAPHS)

    template = read_template(doc)
    by_heading = {s.heading: s for s in template.sections}

    # Acme Corp (H2) should end at Beta Inc (next H2 sibling).
    acme = by_heading["Acme Corp — Senior Engineer"]
    beta = by_heading["Beta Inc — Engineer"]
    assert acme.end_index == beta.start_index

    # Beta Inc (H2) should end at Skills (the next H1 — higher level).
    skills = by_heading["Skills"]
    assert beta.end_index == skills.start_index


def test_last_section_runs_to_body_end() -> None:
    doc = make_doc(SAMPLE_RESUME_PARAGRAPHS)
    body_end = max(el["endIndex"] for el in doc["body"]["content"])

    template = read_template(doc)

    education = next(s for s in template.sections if s.heading == "Education")
    assert education.end_index == body_end


def test_paragraph_count_includes_heading_and_body_paragraphs() -> None:
    doc = make_doc(SAMPLE_RESUME_PARAGRAPHS)

    template = read_template(doc)
    by_heading = {s.heading: s for s in template.sections}

    # Acme Corp section: 1 heading + 2 bullet paragraphs = 3.
    assert by_heading["Acme Corp — Senior Engineer"].paragraph_count == 3
    # Skills section: heading + 1 line + 1 trailing blank line = 3.
    assert by_heading["Skills"].paragraph_count == 3


def test_handles_document_without_headings() -> None:
    doc = make_doc(
        [("Just a name", None), ("Just an email", None)],
        doc_id="d2",
        title="Plain",
        revision_id="r1",
    )

    template = read_template(doc)
    assert template.sections == ()


def test_treats_title_and_subtitle_named_styles_as_headings() -> None:
    doc = make_doc(
        [
            ("Jonathan Duketis", "TITLE"),
            ("Senior Software Engineer", "SUBTITLE"),
            ("Body text", None),
        ]
    )

    template = read_template(doc)
    levels = [s.level for s in template.sections]
    assert levels == [1, 2]


def test_skips_paragraphs_without_text() -> None:
    doc = make_doc(
        [
            ("", "HEADING_1"),  # empty heading — should be skipped
            ("Summary", "HEADING_1"),
            ("body", None),
        ]
    )
    template = read_template(doc)
    assert [s.heading for s in template.sections] == ["Summary"]


def test_skips_unknown_named_styles() -> None:
    doc = make_doc(
        [
            ("Should not appear", "BLOCKQUOTE"),
            ("Real heading", "HEADING_1"),
        ]
    )
    template = read_template(doc)
    assert [s.heading for s in template.sections] == ["Real heading"]


# -- error / edge cases ------------------------------------------------------


def test_raises_when_documentid_missing() -> None:
    doc = make_doc([("h", "HEADING_1")])
    del doc["documentId"]
    with pytest.raises(ReaderError, match="documentId"):
        read_template(doc)


def test_raises_when_title_missing() -> None:
    doc = make_doc([("h", "HEADING_1")])
    del doc["title"]
    with pytest.raises(ReaderError, match="title"):
        read_template(doc)


def test_raises_when_revisionid_missing() -> None:
    doc = make_doc([("h", "HEADING_1")])
    del doc["revisionId"]
    with pytest.raises(ReaderError, match="revisionId"):
        read_template(doc)


def test_raises_when_body_missing() -> None:
    doc = make_doc([("h", "HEADING_1")])
    del doc["body"]
    with pytest.raises(ReaderError, match="body"):
        read_template(doc)


def test_handles_body_with_no_content_list() -> None:
    doc = make_doc([("h", "HEADING_1")])
    doc["body"] = {}
    template = read_template(doc)
    assert template.sections == ()


def test_handles_body_with_non_list_content() -> None:
    doc = make_doc([("h", "HEADING_1")])
    doc["body"]["content"] = "not a list"
    template = read_template(doc)
    assert template.sections == ()


def test_skips_malformed_structural_elements() -> None:
    doc = make_doc([("Real heading", "HEADING_1")])
    # Insert assorted garbage that the reader should walk past.
    doc["body"]["content"].insert(0, "string-not-dict")
    doc["body"]["content"].append({"paragraph": "not-a-dict"})
    doc["body"]["content"].append({"paragraph": {"paragraphStyle": "not-a-dict"}})
    doc["body"]["content"].append({"paragraph": {"paragraphStyle": {"namedStyleType": 99}}})
    doc["body"]["content"].append(
        {"paragraph": {"paragraphStyle": {"namedStyleType": "HEADING_1"}, "elements": "x"}}
    )
    doc["body"]["content"].append(
        {
            "paragraph": {
                "paragraphStyle": {"namedStyleType": "HEADING_1"},
                "elements": ["string-not-dict"],
            }
        }
    )
    doc["body"]["content"].append(
        {
            "paragraph": {
                "paragraphStyle": {"namedStyleType": "HEADING_1"},
                "elements": [{"textRun": "not-a-dict"}],
            }
        }
    )

    template = read_template(doc)
    assert [s.heading for s in template.sections] == ["Real heading"]


def test_skips_heading_paragraph_with_non_int_start_index() -> None:
    doc = make_doc([("Real heading", "HEADING_1"), ("body", None)])
    # Append a heading whose startIndex is the wrong type.
    doc["body"]["content"].append(
        {
            "startIndex": "not-an-int",
            "endIndex": 999,
            "paragraph": {
                "elements": [{"textRun": {"content": "Bad heading\n"}}],
                "paragraphStyle": {"namedStyleType": "HEADING_1"},
            },
        }
    )
    template = read_template(doc)
    assert [s.heading for s in template.sections] == ["Real heading"]


def test_paragraph_text_handles_textrun_content_non_string() -> None:
    doc = {
        "documentId": "d",
        "title": "t",
        "revisionId": "r",
        "body": {
            "content": [
                {
                    "startIndex": 1,
                    "endIndex": 10,
                    "paragraph": {
                        "elements": [{"textRun": {"content": 99}}],
                        "paragraphStyle": {"namedStyleType": "HEADING_1"},
                    },
                }
            ]
        },
    }
    template = read_template(doc)
    # Non-string content yields empty heading text → heading is skipped.
    assert template.sections == ()


def test_document_end_index_handles_missing_endindex() -> None:
    doc = make_doc([("Heading", "HEADING_1")])
    # Insert a malformed element with no endIndex.
    doc["body"]["content"].append({"startIndex": 1000})
    template = read_template(doc)
    # Should still produce the section (last one runs to whatever max endIndex we computed).
    assert len(template.sections) == 1
