"""Orchestrator: turn a :class:`TailoredResume` into a Google Doc + PDF.

Per-paragraph in-place text-swap algorithm (v0.8.1+):

For each known section:

1. Re-read the doc and locate the heading.
2. List the paragraphs under the heading (excluding the heading itself),
   recording each paragraph's start/end character indices.
3. Build a single ``batchUpdate`` whose requests, applied in order,
   will (a) delete trailing paragraphs we no longer need, (b) replace
   the text inside each kept paragraph in-place, and (c) append any
   extra paragraphs after the last existing one.

Why per-paragraph instead of "delete content + insert plain text":
Google Docs API styles inserted text by inheriting the paragraph style
at the insertion point. Replacing only the *text content* of each
existing paragraph keeps every styling attribute (bullet markers,
indents, list level, font, etc.) the master template already set on
that paragraph.

Why a single batchUpdate per section: operations within a batch run
sequentially against the current state, so requests issued in the
right order (highest-index first) compose correctly without
needing to re-read the doc between every micro-edit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from resumeai.docs.copy import copy_template
from resumeai.docs.reader import read_template
from resumeai.renderer.models import RenderDiff, RenderResult, RenderStatus
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
    from resumeai.docs.models import Section


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
    """Run the full render: copy → per-section paragraph swaps → PDF export."""
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
    """Replace a single section's content via per-paragraph text swaps."""
    if not new_text:
        return RenderDiff(kind=kind, status=RenderStatus.SKIPPED_EMPTY)

    raw_doc = client.get_document(doc_id)
    section = find_section(read_template(raw_doc), kind)
    if section is None:
        return RenderDiff(kind=kind, status=RenderStatus.NOT_FOUND)

    paragraphs = list_section_paragraphs(raw_doc, section)
    new_lines = new_text.split("\n")
    heading_end = section.start_index + len(section.heading) + 1

    requests = build_per_paragraph_requests(paragraphs, new_lines, heading_end)
    client.batch_update(doc_id, requests)

    before_chars = max(0, section.end_index - 1 - heading_end)
    return RenderDiff(
        kind=kind,
        heading=section.heading,
        status=RenderStatus.REPLACED,
        before_chars=before_chars,
        after_chars=len(new_text),
    )


# -- pure helpers (the testable surface) ------------------------------------


def list_section_paragraphs(raw_doc: dict[str, Any], section: Section) -> list[tuple[int, int]]:
    """Return ``(start_index, end_index)`` for each paragraph in ``section``,
    EXCLUDING the heading paragraph itself. Indices are Google Docs character
    indices; ``end_index`` is exclusive and includes the trailing newline.
    """
    body = raw_doc.get("body", {})
    elements = body.get("content", []) if isinstance(body, dict) else []
    if not isinstance(elements, list):
        return []
    heading_end = section.start_index + len(section.heading) + 1
    paragraphs: list[tuple[int, int]] = []
    for element in elements:
        if not isinstance(element, dict) or "paragraph" not in element:
            continue
        start = element.get("startIndex")
        end = element.get("endIndex")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if heading_end <= start < section.end_index:
            paragraphs.append((start, end))
    return paragraphs


def build_per_paragraph_requests(
    paragraphs: list[tuple[int, int]],
    new_lines: list[str],
    heading_end: int,
) -> list[dict[str, Any]]:
    """Compose one batchUpdate request list to swap a section's content
    paragraph-by-paragraph.

    The returned requests must be applied in order; Google Docs API runs
    them sequentially against the doc's current state. Within this list
    every request operates on positions that are still valid after all
    earlier requests in the list, because we either:

    - delete from highest indices first (so lower indices stay valid), or
    - replace text inside a paragraph at its original ``start_index``
      (which is unaffected by mutations at higher indices).
    """
    if not paragraphs:
        return _requests_for_empty_section(new_lines, heading_end)

    requests: list[dict[str, Any]] = []
    n_existing = len(paragraphs)
    n_new = len(new_lines)

    # Step 1: delete excess existing paragraphs from the tail (high to low).
    for i in range(n_existing - 1, n_new - 1, -1):
        s, e = paragraphs[i]
        requests.append({"deleteContentRange": {"range": {"startIndex": s, "endIndex": e}}})

    # Step 2 + Step 3: replace text in kept paragraphs (reverse order), and
    # the moment we mutate the *last* existing paragraph, immediately insert
    # any extras after it — that paragraph's NEW end is computable from the
    # in-place insert we just emitted, and earlier paragraphs (lower indices)
    # haven't been touched yet so the insertion point stays valid through
    # subsequent reverse-iteration mutations.
    n_kept = min(n_existing, n_new)
    for i in range(n_kept - 1, -1, -1):
        s, e = paragraphs[i]
        text_end = e - 1  # don't touch the trailing \n
        new_text = new_lines[i]
        if text_end > s:
            requests.append(
                {"deleteContentRange": {"range": {"startIndex": s, "endIndex": text_end}}}
            )
        if new_text:
            requests.append({"insertText": {"location": {"index": s}, "text": new_text}})
        if i == n_existing - 1 and n_new > n_existing:
            # Append extras right after the last paragraph's new \n position.
            # After this paragraph's mutation, its new end is
            #   start + len(new_text) + 1 (text + the surviving \n).
            insert_at = s + len(new_text) + 1 - 1  # before the \n
            extras = "".join("\n" + line for line in new_lines[n_existing:])
            requests.append({"insertText": {"location": {"index": insert_at}, "text": extras}})

    return requests


def _requests_for_empty_section(new_lines: list[str], heading_end: int) -> list[dict[str, Any]]:
    """Section has heading but zero content paragraphs — insert from scratch."""
    if not new_lines or all(line == "" for line in new_lines):
        return []
    text = "\n".join(new_lines)
    return [{"insertText": {"location": {"index": heading_end}, "text": text}}]


def _default_title(tailored: TailoredResume, now: datetime) -> str:
    name = tailored.name.strip() or "Resume"
    stamp = now.strftime("%Y-%m-%d %H:%M")
    return f"{name} — Tailored resume ({stamp})"
