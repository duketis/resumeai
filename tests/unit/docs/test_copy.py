"""Tests for the copy helper."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from resumeai.docs.client import FakeDocsClient
from resumeai.docs.copy import copy_template


def test_copy_template_calls_copy_document_with_provided_title() -> None:
    client = FakeDocsClient(copy_returns="new-id")

    new_id = copy_template(client, "master-doc-id", "Tailored Resume — Acme Corp")

    assert new_id == "new-id"
    assert client.copy_calls == [("master-doc-id", "Tailored Resume — Acme Corp")]


def test_copy_template_uses_default_timestamped_title() -> None:
    client = FakeDocsClient()
    fixed_now = datetime(2026, 5, 9, 12, 30, 45, tzinfo=UTC)

    copy_template(client, "master-doc-id", now=fixed_now)

    assert client.copy_calls == [("master-doc-id", "Tailored resume — 2026-05-09T12:30:45+00:00")]


def test_copy_template_default_now_is_utc() -> None:
    """When ``now`` is omitted, the helper uses ``datetime.now(UTC)``."""
    client = FakeDocsClient()

    copy_template(client, "master-doc-id")

    assert len(client.copy_calls) == 1
    title = client.copy_calls[0][1]
    # The default title format includes a UTC offset.
    assert "+00:00" in title


def test_copy_template_rejects_empty_master_id() -> None:
    client = FakeDocsClient()
    with pytest.raises(ValueError, match="master_doc_id"):
        copy_template(client, "")


def test_copy_template_does_not_call_get_or_export() -> None:
    """The copy helper must only invoke copy_document — never any read or
    write API against the master."""
    client = FakeDocsClient()

    copy_template(client, "master-doc-id")

    assert client.get_calls == []
    assert client.export_calls == []
