"""End-to-end renderer tests + pure helpers (per-paragraph algorithm).

The renderer's per-section algorithm is split into two pure helpers
(``list_section_paragraphs``, ``build_per_paragraph_requests``) that are
unit-tested with crafted inputs, plus the ``render_tailored_resume``
orchestrator that's exercised against ``FakeDocsClient`` with scripted
template responses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from resumeai.docs.client import FakeDocsClient
from resumeai.docs.models import Section
from resumeai.renderer.models import RenderStatus
from resumeai.renderer.render import (
    build_per_paragraph_requests,
    list_section_paragraphs,
    render_tailored_resume,
)

if TYPE_CHECKING:
    from resumeai.agent.models import TailoredResume


_NEW_DOC_ID = "tailored-doc-id"


# -- list_section_paragraphs ------------------------------------------------


def test_list_section_paragraphs_returns_only_in_section_paragraphs() -> None:
    raw = _build_raw_doc(
        [
            ("Header line", None),
            ("Summary", "HEADING_1"),
            ("Old summary text.", None),
            ("Skills", "HEADING_1"),
            ("Old skills text.", None),
        ]
    )
    # Indices: header is (1, 13), Summary heading at (13, 21), body at (21, 39),
    # Skills at (39, 47), body at (47, 64).
    section = Section(heading="Summary", level=1, start_index=13, end_index=39, paragraph_count=2)

    paragraphs = list_section_paragraphs(raw, section)

    assert len(paragraphs) == 1
    start, end = paragraphs[0]
    assert start == 21
    assert end == 39


def test_list_section_paragraphs_skips_heading_paragraph() -> None:
    raw = _build_raw_doc(
        [
            ("Summary", "HEADING_1"),
            ("Body 1", None),
            ("Body 2", None),
        ]
    )
    section = Section(heading="Summary", level=1, start_index=1, end_index=23, paragraph_count=3)
    paragraphs = list_section_paragraphs(raw, section)
    assert len(paragraphs) == 2


def test_list_section_paragraphs_returns_empty_for_section_with_no_body() -> None:
    raw = _build_raw_doc(
        [
            ("Summary", "HEADING_1"),
            ("Skills", "HEADING_1"),
        ]
    )
    section = Section(heading="Summary", level=1, start_index=1, end_index=9, paragraph_count=1)
    assert list_section_paragraphs(raw, section) == []


def test_list_section_paragraphs_handles_missing_body() -> None:
    raw: dict[str, Any] = {"documentId": "d", "title": "t", "revisionId": "r"}
    section = Section(heading="Summary", level=1, start_index=1, end_index=2, paragraph_count=1)
    assert list_section_paragraphs(raw, section) == []


def test_list_section_paragraphs_handles_non_list_content() -> None:
    raw: dict[str, Any] = {
        "documentId": "d",
        "title": "t",
        "revisionId": "r",
        "body": {"content": "garbage"},
    }
    section = Section(heading="Summary", level=1, start_index=1, end_index=2, paragraph_count=1)
    assert list_section_paragraphs(raw, section) == []


def test_list_section_paragraphs_skips_malformed_elements() -> None:
    raw = _build_raw_doc([("Summary", "HEADING_1"), ("Body", None)])
    raw["body"]["content"].append("not-a-dict")
    raw["body"]["content"].append({"sectionBreak": {}})  # not a paragraph
    raw["body"]["content"].append({"paragraph": {}, "startIndex": "not-int"})
    raw["body"]["content"].append({"paragraph": {}, "startIndex": 100})  # missing endIndex
    section = Section(heading="Summary", level=1, start_index=1, end_index=14, paragraph_count=2)
    paragraphs = list_section_paragraphs(raw, section)
    assert len(paragraphs) == 1


# -- build_per_paragraph_requests -------------------------------------------


def test_build_requests_replaces_in_place_when_counts_match() -> None:
    paragraphs = [(10, 16), (16, 22), (22, 28)]
    new_lines = ["a", "b", "c"]

    requests = build_per_paragraph_requests(paragraphs, new_lines, heading_end=10)

    # Reverse order: each replace = delete inner text + insert new text.
    assert len(requests) == 6
    assert requests[0] == {"deleteContentRange": {"range": {"startIndex": 22, "endIndex": 27}}}
    assert requests[1] == {"insertText": {"location": {"index": 22}, "text": "c"}}
    assert requests[2]["deleteContentRange"]["range"] == {"startIndex": 16, "endIndex": 21}
    assert requests[3]["insertText"] == {"location": {"index": 16}, "text": "b"}
    assert requests[4]["deleteContentRange"]["range"] == {"startIndex": 10, "endIndex": 15}
    assert requests[5]["insertText"] == {"location": {"index": 10}, "text": "a"}


def test_build_requests_skips_delete_when_paragraph_was_empty() -> None:
    paragraphs = [(10, 11)]  # empty paragraph (just \n)
    new_lines = ["new"]

    requests = build_per_paragraph_requests(paragraphs, new_lines, heading_end=10)

    assert requests == [{"insertText": {"location": {"index": 10}, "text": "new"}}]


def test_build_requests_skips_insert_when_new_text_empty() -> None:
    paragraphs = [(10, 16)]  # "old\n"
    new_lines = [""]

    requests = build_per_paragraph_requests(paragraphs, new_lines, heading_end=10)

    assert requests == [{"deleteContentRange": {"range": {"startIndex": 10, "endIndex": 15}}}]


def test_build_requests_deletes_excess_paragraphs_from_tail() -> None:
    paragraphs = [(10, 14), (14, 18), (18, 22), (22, 26)]
    new_lines = ["a", "b"]

    requests = build_per_paragraph_requests(paragraphs, new_lines, heading_end=10)

    assert requests[0] == {"deleteContentRange": {"range": {"startIndex": 22, "endIndex": 26}}}
    assert requests[1] == {"deleteContentRange": {"range": {"startIndex": 18, "endIndex": 22}}}
    assert requests[2]["deleteContentRange"]["range"] == {"startIndex": 14, "endIndex": 17}
    assert requests[3]["insertText"] == {"location": {"index": 14}, "text": "b"}
    assert requests[4]["deleteContentRange"]["range"] == {"startIndex": 10, "endIndex": 13}
    assert requests[5]["insertText"] == {"location": {"index": 10}, "text": "a"}


def test_build_requests_appends_extras_after_last_paragraph_mutation() -> None:
    paragraphs = [(10, 14), (14, 18)]
    new_lines = ["a", "b", "c", "d"]

    requests = build_per_paragraph_requests(paragraphs, new_lines, heading_end=10)

    # Order:
    #   [0] delete (14, 17) — inner text of last paragraph
    #   [1] insert "b" at 14 — paragraph at (14, 18) becomes (14, 16): "b\n"
    #   [2] insert "\nc\nd" at 15 (just before the surviving \n)
    #   [3] delete (10, 13) — inner text of first paragraph
    #   [4] insert "a" at 10
    assert requests[0]["deleteContentRange"]["range"] == {"startIndex": 14, "endIndex": 17}
    assert requests[1]["insertText"] == {"location": {"index": 14}, "text": "b"}
    assert requests[2]["insertText"] == {"location": {"index": 15}, "text": "\nc\nd"}
    assert requests[3]["deleteContentRange"]["range"] == {"startIndex": 10, "endIndex": 13}
    assert requests[4]["insertText"] == {"location": {"index": 10}, "text": "a"}


def test_build_requests_handles_extras_when_last_paragraph_was_empty() -> None:
    paragraphs = [(10, 11)]  # empty paragraph
    new_lines = ["", "extra1", "extra2"]

    requests = build_per_paragraph_requests(paragraphs, new_lines, heading_end=10)

    assert requests == [{"insertText": {"location": {"index": 10}, "text": "\nextra1\nextra2"}}]


def test_build_requests_for_empty_section_inserts_lines() -> None:
    requests = build_per_paragraph_requests([], ["a", "b"], heading_end=20)
    assert requests == [{"insertText": {"location": {"index": 20}, "text": "a\nb"}}]


def test_build_requests_for_empty_section_with_no_lines_emits_nothing() -> None:
    assert build_per_paragraph_requests([], [], heading_end=20) == []


def test_build_requests_for_empty_section_with_only_blank_lines_emits_nothing() -> None:
    assert build_per_paragraph_requests([], ["", ""], heading_end=20) == []


def test_build_requests_returns_empty_when_existing_and_new_both_empty() -> None:
    paragraphs = [(10, 11)]
    new_lines = [""]
    assert build_per_paragraph_requests(paragraphs, new_lines, heading_end=10) == []


# -- render_tailored_resume orchestration ----------------------------------


def test_render_copies_master_then_exports_pdf(
    sample_tailored: TailoredResume, template_doc_with_all_sections: dict[str, Any]
) -> None:
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
        pdf_bytes=b"%PDF-rendered",
    )

    result = render_tailored_resume(
        sample_tailored,
        template_doc_id="master-id",
        client=client,
        now=datetime(2026, 5, 9, 12, 30, tzinfo=UTC),
    )

    assert client.copy_calls[0][0] == "master-id"
    assert client.export_calls == [_NEW_DOC_ID]
    assert result.doc_id == _NEW_DOC_ID
    assert result.doc_url == f"https://docs.google.com/document/d/{_NEW_DOC_ID}/edit"
    assert result.pdf_bytes == b"%PDF-rendered"


def test_render_uses_provided_title(
    sample_tailored: TailoredResume, template_doc_with_all_sections: dict[str, Any]
) -> None:
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )
    render_tailored_resume(
        sample_tailored,
        template_doc_id="master-id",
        client=client,
        new_title="Custom title",
    )
    assert client.copy_calls == [("master-id", "Custom title")]


def test_render_default_title_includes_name_and_timestamp(
    sample_tailored: TailoredResume, template_doc_with_all_sections: dict[str, Any]
) -> None:
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )
    render_tailored_resume(
        sample_tailored,
        template_doc_id="master-id",
        client=client,
        now=datetime(2026, 5, 9, 12, 30, tzinfo=UTC),
    )
    title = client.copy_calls[0][1]
    assert title.startswith("Alex Sample — Tailored resume (")
    assert "2026-05-09 12:30" in title


def test_render_default_title_falls_back_when_name_blank(
    template_doc_with_all_sections: dict[str, Any],
) -> None:
    from resumeai.agent.models import TailoredResume  # noqa: PLC0415
    from resumeai.context.models import Contact  # noqa: PLC0415

    bare = TailoredResume(name=" ", contact=Contact(email="a@example.com"))
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )
    render_tailored_resume(
        bare,
        template_doc_id="master-id",
        client=client,
        now=datetime(2026, 5, 9, 12, 30, tzinfo=UTC),
    )
    assert client.copy_calls[0][1].startswith("Resume — Tailored resume (")


def test_render_emits_one_diff_per_pipeline_section(
    sample_tailored: TailoredResume, template_doc_with_all_sections: dict[str, Any]
) -> None:
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )
    result = render_tailored_resume(sample_tailored, template_doc_id="master-id", client=client)
    kinds = [d.kind for d in result.diffs]
    assert kinds == ["summary", "skills", "work_history", "education", "certifications"]


def test_render_replaces_known_sections(
    sample_tailored: TailoredResume, template_doc_with_all_sections: dict[str, Any]
) -> None:
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )
    result = render_tailored_resume(sample_tailored, template_doc_id="master-id", client=client)
    for diff in result.diffs:
        assert diff.status is RenderStatus.REPLACED, f"{diff.kind}: {diff.status}"
        assert diff.heading is not None


def test_render_skips_section_when_tailored_empty(
    template_doc_with_all_sections: dict[str, Any],
) -> None:
    from resumeai.agent.models import TailoredResume  # noqa: PLC0415
    from resumeai.context.models import Contact  # noqa: PLC0415

    bare = TailoredResume(
        name="Alex",
        contact=Contact(email="a@example.com"),
        summary="Has a summary.",
    )
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )
    result = render_tailored_resume(bare, template_doc_id="master-id", client=client)

    statuses = {d.kind: d.status for d in result.diffs}
    assert statuses["summary"] is RenderStatus.REPLACED
    assert statuses["skills"] is RenderStatus.SKIPPED_EMPTY
    assert statuses["work_history"] is RenderStatus.SKIPPED_EMPTY
    assert statuses["education"] is RenderStatus.SKIPPED_EMPTY
    assert statuses["certifications"] is RenderStatus.SKIPPED_EMPTY


def test_render_marks_section_not_found_when_template_missing_heading(
    sample_tailored: TailoredResume,
) -> None:
    template = _build_raw_doc(
        [
            ("Summary", "HEADING_1"),
            ("Old summary.", None),
        ]
    )
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template},
        copy_returns=_NEW_DOC_ID,
    )
    result = render_tailored_resume(sample_tailored, template_doc_id="master-id", client=client)
    statuses = {d.kind: d.status for d in result.diffs}
    assert statuses["summary"] is RenderStatus.REPLACED
    assert statuses["skills"] is RenderStatus.NOT_FOUND
    assert statuses["work_history"] is RenderStatus.NOT_FOUND


def test_render_issues_per_paragraph_ops_against_real_fixture(
    sample_tailored: TailoredResume, template_doc_with_all_sections: dict[str, Any]
) -> None:
    """The orchestrator should emit per-paragraph delete+insert ops (not one
    giant section-wide delete-then-insert)."""
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )
    render_tailored_resume(sample_tailored, template_doc_id="master-id", client=client)

    has_delete = False
    has_insert = False
    for _doc_id, requests in client.batch_update_calls:
        for req in requests:
            if "deleteContentRange" in req:
                has_delete = True
            if "insertText" in req:
                has_insert = True
    assert has_delete
    assert has_insert


def test_render_skips_batch_for_sections_with_no_content(
    template_doc_with_all_sections: dict[str, Any],
) -> None:
    """A bare TailoredResume produces ``None`` for every section's content,
    so each section short-circuits to ``SKIPPED_EMPTY`` without calling
    ``batch_update`` at all."""
    from resumeai.agent.models import TailoredResume  # noqa: PLC0415
    from resumeai.context.models import Contact  # noqa: PLC0415

    bare = TailoredResume(name="Alex", contact=Contact(email="a@example.com"))
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )
    render_tailored_resume(bare, template_doc_id="master-id", client=client)
    assert client.batch_update_calls == []


# -- helpers ---------------------------------------------------------------


def _build_raw_doc(
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
        "documentId": "raw-doc",
        "title": "T",
        "revisionId": "r",
        "body": {"content": body_content},
    }
