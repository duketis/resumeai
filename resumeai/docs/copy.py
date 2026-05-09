"""Copy a master Google Doc to a new tailored doc.

Hard rule from CLAUDE.md cross-cutting invariant #1: the master is sacred.
This module is the *only* public entry point that touches a master template
in any way, and the operation it performs (Drive ``files.copy``) creates a
new file rather than mutating the source.

Future write-path code (the renderer in Phase 5) will assert that any
``documents.batchUpdate`` it issues targets a *copy*, not a registered
master — see ``copy_template`` below for the symmetric helper.
"""

from __future__ import annotations

from datetime import UTC, datetime

from resumeai.docs.client import DocsClient


def copy_template(
    client: DocsClient,
    master_doc_id: str,
    new_title: str | None = None,
    *,
    now: datetime | None = None,
) -> str:
    """Copy ``master_doc_id`` to a new doc and return its ID.

    Args:
        client: a :class:`~resumeai.docs.client.DocsClient`.
        master_doc_id: the master template's Google Doc ID. Never written to.
        new_title: title for the copy. Defaults to a timestamped name like
            ``"Tailored resume — 2026-05-09T06:18:55+00:00"``.
        now: injected for tests; defaults to :func:`datetime.now` in UTC.
    """
    if not master_doc_id:
        raise ValueError("master_doc_id is required")

    title = new_title or _default_title(now or datetime.now(UTC))
    return client.copy_document(master_doc_id, title)


def _default_title(now: datetime) -> str:
    return f"Tailored resume — {now.isoformat(timespec='seconds')}"
