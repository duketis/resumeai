"""Typed wrapper around ``googleapiclient.discovery`` for Docs + Drive.

The ``DocsClient`` ``Protocol`` is the contract the rest of the codebase sees.
``GoogleDocsClient`` is the only place that imports from ``googleapiclient``
and ``google.oauth2.credentials``; everything above this boundary works with
plain Python types.

Tests use :class:`FakeDocsClient`, which returns scripted responses and
records calls so the reader, copy helper, and routes can be exercised
offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from resumeai.settings.models import GoogleCredentials


class DocsClient(Protocol):
    """The Google operations resumeai needs."""

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """Return the full ``documents.get`` response for ``doc_id``."""

    def copy_document(self, source_doc_id: str, new_title: str) -> str:
        """Copy ``source_doc_id`` via Drive ``files.copy``. Return new doc ID."""

    def export_pdf(self, doc_id: str) -> bytes:
        """Export ``doc_id`` to PDF via Drive ``files.export``."""

    def batch_update(self, doc_id: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
        """Issue a Docs ``documents.batchUpdate`` against ``doc_id``.

        ``requests`` is the list of request objects per the Docs API; the
        Phase 5 renderer is the only caller. Operations within a single
        batch run in order against the current state of the doc, so a
        delete + insert pair shifts indices correctly without further work.
        """


class GoogleDocsClient:
    """Concrete implementation backed by the official Google API discovery clients.

    Constructed once per HTTP request (or per scheduled run) so credentials
    aren't shared across requests. Internally builds two discovery resources:
    Docs v1 for ``documents.get`` / ``documents.batchUpdate``, and Drive v3
    for ``files.copy`` / ``files.export``.
    """

    def __init__(self, credentials: GoogleCredentials) -> None:
        # Imports are deferred so test code that supplies a Fake never has to
        # have google-api-python-client installed in its scope.
        from google.oauth2.credentials import Credentials  # noqa: PLC0415
        from googleapiclient.discovery import build  # noqa: PLC0415

        google_creds = Credentials(
            token=credentials.access_token,
            refresh_token=credentials.refresh_token,
            token_uri=credentials.token_uri,
            client_id=credentials.client_id,
            client_secret=credentials.client_secret,
            scopes=list(credentials.scopes),
        )
        self._docs = build("docs", "v1", credentials=google_creds, cache_discovery=False)
        self._drive = build("drive", "v3", credentials=google_creds, cache_discovery=False)

    def get_document(self, doc_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._docs.documents().get(documentId=doc_id).execute()
        return result

    def copy_document(self, source_doc_id: str, new_title: str) -> str:
        result: dict[str, Any] = (
            self._drive.files().copy(fileId=source_doc_id, body={"name": new_title}).execute()
        )
        new_id = result.get("id")
        if not isinstance(new_id, str) or not new_id:
            raise RuntimeError(f"files.copy response missing 'id' field: {result!r}")
        return new_id

    def export_pdf(self, doc_id: str) -> bytes:
        result: bytes = (
            self._drive.files().export(fileId=doc_id, mimeType="application/pdf").execute()
        )
        return result

    def batch_update(self, doc_id: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = (
            self._docs.documents()
            .batchUpdate(documentId=doc_id, body={"requests": requests})
            .execute()
        )
        return result


class FakeDocsClient:
    """Test double. Records calls; returns scripted responses.

    Construct with the responses you want each method to return:

        client = FakeDocsClient(
            documents={"doc-id": {...raw response...}},
            copy_returns="new-doc-id",
            pdf_bytes=b"%PDF-...",
        )
    """

    def __init__(
        self,
        *,
        documents: dict[str, dict[str, Any]] | None = None,
        copy_returns: str = "fake-new-doc-id",
        pdf_bytes: bytes = b"",
        batch_update_returns: dict[str, Any] | None = None,
    ) -> None:
        self._documents = documents or {}
        self._copy_returns = copy_returns
        self._pdf_bytes = pdf_bytes
        self._batch_update_returns = batch_update_returns or {}
        self.get_calls: list[str] = []
        self.copy_calls: list[tuple[str, str]] = []
        self.export_calls: list[str] = []
        self.batch_update_calls: list[tuple[str, list[dict[str, Any]]]] = []

    def get_document(self, doc_id: str) -> dict[str, Any]:
        self.get_calls.append(doc_id)
        if doc_id not in self._documents:
            raise KeyError(f"FakeDocsClient: no scripted document for {doc_id!r}")
        return self._documents[doc_id]

    def copy_document(self, source_doc_id: str, new_title: str) -> str:
        self.copy_calls.append((source_doc_id, new_title))
        return self._copy_returns

    def export_pdf(self, doc_id: str) -> bytes:
        self.export_calls.append(doc_id)
        return self._pdf_bytes

    def batch_update(self, doc_id: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
        self.batch_update_calls.append((doc_id, requests))
        return self._batch_update_returns
