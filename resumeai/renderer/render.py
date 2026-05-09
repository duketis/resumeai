"""Orchestrator: turn a :class:`TailoredResume` into a Google Doc + PDF."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from resumeai.docs.copy import copy_template
from resumeai.docs.reader import read_template
from resumeai.renderer.models import (
    RenderDiff,
    RenderResult,
    RenderStatus,
)
from resumeai.renderer.sections import (
    find_section,
    render_certifications,
    render_education,
    render_skills,
    render_summary,
    render_work_history,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from resumeai.agent.models import TailoredResume
    from resumeai.docs.client import DocsClient


# Section render order — top of resume first.
_SECTION_PIPELINE: tuple[tuple[str, Callable[[TailoredResume], str | None]], ...] = (
    ("summary", render_summary),
    ("skills", render_skills),
    ("work_history", render_work_history),
    ("education", render_education),
    ("certifications", render_certifications),
)


def render_tailored_resume(
    tailored: TailoredResume,
    template_doc_id: str,
    client: DocsClient,
    *,
    new_title: str | None = None,
    now: datetime | None = None,
) -> RenderResult:
    """Run the full render: copy → per-section batch updates → PDF export."""
    title = new_title or _default_title(tailored, now or datetime.now(UTC))
    new_doc_id = copy_template(client, template_doc_id, title, now=now)

    diffs: list[RenderDiff] = [
        _render_section(client, new_doc_id, kind, builder(tailored))
        for kind, builder in _SECTION_PIPELINE
    ]

    pdf = client.export_pdf(new_doc_id)

    return RenderResult(
        doc_id=new_doc_id,
        doc_url=f"https://docs.google.com/document/d/{new_doc_id}/edit",
        pdf_bytes=pdf,
        diffs=tuple(diffs),
    )


def _render_section(
    client: DocsClient,
    doc_id: str,
    kind: str,
    new_text: str | None,
) -> RenderDiff:
    """Replace a single section's content with ``new_text`` (or skip)."""
    if not new_text:
        return RenderDiff(kind=kind, status=RenderStatus.SKIPPED_EMPTY)

    template = read_template(client.get_document(doc_id))
    section = find_section(template, kind)
    if section is None:
        return RenderDiff(kind=kind, status=RenderStatus.NOT_FOUND)

    heading_end = section.start_index + len(section.heading) + 1
    content_end = section.end_index - 1

    requests = _build_replace_requests(heading_end, content_end, new_text)
    client.batch_update(doc_id, requests)

    return RenderDiff(
        kind=kind,
        heading=section.heading,
        status=RenderStatus.REPLACED,
        before_chars=max(0, content_end - heading_end),
        after_chars=len(new_text),
    )


def _build_replace_requests(
    heading_end: int, content_end: int, new_text: str
) -> list[dict[str, Any]]:
    """Compose the delete + insert pair for a single-section replace.

    Operations within a ``batchUpdate`` run sequentially against the
    current document state, so deleting first then inserting at the same
    starting index is correct.
    """
    requests: list[dict[str, Any]] = []
    if content_end > heading_end:
        requests.append(
            {
                "deleteContentRange": {
                    "range": {
                        "startIndex": heading_end,
                        "endIndex": content_end,
                    }
                }
            }
        )
    requests.append(
        {
            "insertText": {
                "location": {"index": heading_end},
                "text": new_text,
            }
        }
    )
    return requests


def _default_title(tailored: TailoredResume, now: datetime) -> str:
    name = tailored.name.strip() or "Resume"
    stamp = now.strftime("%Y-%m-%d %H:%M")
    return f"{name} — Tailored resume ({stamp})"
