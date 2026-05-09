"""End-to-end renderer tests against ``FakeDocsClient``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from resumeai.docs.client import FakeDocsClient
from resumeai.renderer.models import RenderStatus
from resumeai.renderer.render import render_tailored_resume

if TYPE_CHECKING:
    from resumeai.agent.models import TailoredResume


_NEW_DOC_ID = "tailored-doc-id"


def test_render_copies_master_then_exports_pdf(
    sample_tailored: TailoredResume, template_doc_with_all_sections: dict[str, Any]
) -> None:
    """The pipeline runs copy → batch updates → PDF export, in that order."""
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

    # Copy first.
    assert client.copy_calls[0][0] == "master-id"
    # PDF export came last.
    assert client.export_calls == [_NEW_DOC_ID]
    # Result wired correctly.
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


def test_render_default_title_falls_back_when_name_empty_after_strip(
    template_doc_with_all_sections: dict[str, Any],
) -> None:
    """``TailoredResume.name`` is min_length=1 but a single space passes — the
    title should still be sensible."""
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
    title = client.copy_calls[0][1]
    assert title.startswith("Resume — Tailored resume (")


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
        # No skills, no work history, no education, no certs.
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
    """Template without a Skills heading: the skills section is marked
    not_found, not silently dropped."""
    template = {
        "documentId": "tmpl",
        "title": "T",
        "revisionId": "r",
        "body": {
            "content": [
                {
                    "startIndex": 1,
                    "endIndex": 9,
                    "paragraph": {
                        "elements": [
                            {
                                "startIndex": 1,
                                "endIndex": 9,
                                "textRun": {"content": "Summary\n", "textStyle": {}},
                            }
                        ],
                        "paragraphStyle": {"namedStyleType": "HEADING_1"},
                    },
                },
                {
                    "startIndex": 9,
                    "endIndex": 12,
                    "paragraph": {
                        "elements": [
                            {
                                "startIndex": 9,
                                "endIndex": 12,
                                "textRun": {"content": "x.\n", "textStyle": {}},
                            }
                        ],
                    },
                },
            ]
        },
    }
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template},
        copy_returns=_NEW_DOC_ID,
    )

    result = render_tailored_resume(sample_tailored, template_doc_id="master-id", client=client)

    statuses = {d.kind: d.status for d in result.diffs}
    assert statuses["summary"] is RenderStatus.REPLACED
    assert statuses["skills"] is RenderStatus.NOT_FOUND
    assert statuses["work_history"] is RenderStatus.NOT_FOUND


def test_render_issues_delete_then_insert_per_replaced_section(
    sample_tailored: TailoredResume, template_doc_with_all_sections: dict[str, Any]
) -> None:
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )

    render_tailored_resume(sample_tailored, template_doc_id="master-id", client=client)

    assert len(client.batch_update_calls) == 5  # one per pipeline section
    for _doc_id, requests in client.batch_update_calls:
        # Each call's first op should be the deleteContentRange,
        # the second the insertText.
        assert "deleteContentRange" in requests[0]
        assert "insertText" in requests[1]


def test_render_skips_delete_when_section_has_no_existing_content() -> None:
    """When the template has a heading immediately followed by the next
    heading (zero-content section), the renderer should only insert."""
    from resumeai.agent.models import TailoredResume  # noqa: PLC0415
    from resumeai.context.models import Contact  # noqa: PLC0415

    template = {
        "documentId": "tmpl",
        "title": "T",
        "revisionId": "r",
        "body": {
            "content": [
                {
                    "startIndex": 1,
                    "endIndex": 9,
                    "paragraph": {
                        "elements": [
                            {
                                "startIndex": 1,
                                "endIndex": 9,
                                "textRun": {"content": "Summary\n", "textStyle": {}},
                            }
                        ],
                        "paragraphStyle": {"namedStyleType": "HEADING_1"},
                    },
                },
                {
                    "startIndex": 9,
                    "endIndex": 18,
                    "paragraph": {
                        "elements": [
                            {
                                "startIndex": 9,
                                "endIndex": 18,
                                "textRun": {"content": "Education\n", "textStyle": {}},
                            }
                        ],
                        "paragraphStyle": {"namedStyleType": "HEADING_1"},
                    },
                },
            ]
        },
    }
    tailored = TailoredResume(
        name="Alex",
        contact=Contact(email="a@example.com"),
        summary="A new summary.",
    )
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template},
        copy_returns=_NEW_DOC_ID,
    )

    render_tailored_resume(tailored, template_doc_id="master-id", client=client)

    summary_call = client.batch_update_calls[0]
    requests = summary_call[1]
    # No deleteContentRange when there's nothing to delete.
    assert all("deleteContentRange" not in req for req in requests)
    assert "insertText" in requests[0]


def test_render_diff_records_before_and_after_chars(
    sample_tailored: TailoredResume, template_doc_with_all_sections: dict[str, Any]
) -> None:
    client = FakeDocsClient(
        documents={_NEW_DOC_ID: template_doc_with_all_sections},
        copy_returns=_NEW_DOC_ID,
    )
    result = render_tailored_resume(sample_tailored, template_doc_id="master-id", client=client)

    summary = next(d for d in result.diffs if d.kind == "summary")
    assert summary.before_chars > 0
    assert summary.after_chars == len(sample_tailored.summary)
