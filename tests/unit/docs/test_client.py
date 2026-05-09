"""Tests for the DocsClient implementations.

``FakeDocsClient`` is exercised directly. ``GoogleDocsClient`` is tested by
patching ``googleapiclient.discovery.build`` so the test suite never touches
Google.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from resumeai.docs.client import FakeDocsClient, GoogleDocsClient
from resumeai.settings.models import GoogleCredentials

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _creds() -> GoogleCredentials:
    return GoogleCredentials(
        access_token="at",
        refresh_token="rt",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csec",
        scopes=("openid",),
        expiry=datetime(2026, 5, 9, tzinfo=UTC),
        user_email="x@example.com",
        user_id="42",
    )


# -- FakeDocsClient ----------------------------------------------------------


def test_fake_returns_scripted_document_and_records_call() -> None:
    client = FakeDocsClient(documents={"doc-1": {"documentId": "doc-1"}})
    doc = client.get_document("doc-1")

    assert doc == {"documentId": "doc-1"}
    assert client.get_calls == ["doc-1"]


def test_fake_raises_for_unknown_document() -> None:
    client = FakeDocsClient()
    with pytest.raises(KeyError, match="no scripted document"):
        client.get_document("missing")


def test_fake_copy_returns_configured_id_and_records_call() -> None:
    client = FakeDocsClient(copy_returns="new-id")
    new_id = client.copy_document("source", "Tailored")

    assert new_id == "new-id"
    assert client.copy_calls == [("source", "Tailored")]


def test_fake_export_returns_configured_bytes_and_records_call() -> None:
    client = FakeDocsClient(pdf_bytes=b"%PDF-1.4\n...")
    pdf = client.export_pdf("doc-id")

    assert pdf == b"%PDF-1.4\n..."
    assert client.export_calls == ["doc-id"]


def test_fake_batch_update_records_call_and_returns_configured_response() -> None:
    client = FakeDocsClient(batch_update_returns={"replies": [{"insertText": {}}]})
    requests = [{"insertText": {"location": {"index": 1}, "text": "x"}}]

    result = client.batch_update("doc-id", requests)

    assert result == {"replies": [{"insertText": {}}]}
    assert client.batch_update_calls == [("doc-id", requests)]


def test_fake_batch_update_default_returns_empty_dict() -> None:
    client = FakeDocsClient()
    result = client.batch_update("doc-id", [])
    assert result == {}


# -- GoogleDocsClient -------------------------------------------------------


def test_google_client_builds_docs_and_drive_resources(mocker: MockerFixture) -> None:
    build = mocker.patch("googleapiclient.discovery.build")
    creds_cls = mocker.patch("google.oauth2.credentials.Credentials")
    creds_cls.return_value = "google-creds-sentinel"

    GoogleDocsClient(_creds())

    assert build.call_count == 2
    build.assert_any_call("docs", "v1", credentials="google-creds-sentinel", cache_discovery=False)
    build.assert_any_call("drive", "v3", credentials="google-creds-sentinel", cache_discovery=False)


def test_google_client_get_document_calls_discovery_chain(mocker: MockerFixture) -> None:
    docs_mock = MagicMock()
    drive_mock = MagicMock()
    mocker.patch("googleapiclient.discovery.build", side_effect=[docs_mock, drive_mock])
    mocker.patch("google.oauth2.credentials.Credentials")
    docs_mock.documents.return_value.get.return_value.execute.return_value = {"documentId": "abc"}

    client = GoogleDocsClient(_creds())
    result = client.get_document("abc")

    assert result == {"documentId": "abc"}
    docs_mock.documents.return_value.get.assert_called_once_with(documentId="abc")


def test_google_client_copy_document_returns_id(mocker: MockerFixture) -> None:
    docs_mock = MagicMock()
    drive_mock = MagicMock()
    mocker.patch("googleapiclient.discovery.build", side_effect=[docs_mock, drive_mock])
    mocker.patch("google.oauth2.credentials.Credentials")
    drive_mock.files.return_value.copy.return_value.execute.return_value = {"id": "new-id"}

    client = GoogleDocsClient(_creds())
    new_id = client.copy_document("source-id", "Tailored Resume")

    assert new_id == "new-id"
    drive_mock.files.return_value.copy.assert_called_once_with(
        fileId="source-id", body={"name": "Tailored Resume"}
    )


def test_google_client_copy_document_raises_when_id_missing(mocker: MockerFixture) -> None:
    docs_mock = MagicMock()
    drive_mock = MagicMock()
    mocker.patch("googleapiclient.discovery.build", side_effect=[docs_mock, drive_mock])
    mocker.patch("google.oauth2.credentials.Credentials")
    drive_mock.files.return_value.copy.return_value.execute.return_value = {}

    client = GoogleDocsClient(_creds())
    with pytest.raises(RuntimeError, match="missing 'id'"):
        client.copy_document("source-id", "Tailored")


def test_google_client_export_pdf_returns_bytes(mocker: MockerFixture) -> None:
    docs_mock = MagicMock()
    drive_mock = MagicMock()
    mocker.patch("googleapiclient.discovery.build", side_effect=[docs_mock, drive_mock])
    mocker.patch("google.oauth2.credentials.Credentials")
    drive_mock.files.return_value.export.return_value.execute.return_value = b"%PDF"

    client = GoogleDocsClient(_creds())
    pdf = client.export_pdf("doc-id")

    assert pdf == b"%PDF"
    drive_mock.files.return_value.export.assert_called_once_with(
        fileId="doc-id", mimeType="application/pdf"
    )


def test_google_client_batch_update_passes_requests(mocker: MockerFixture) -> None:
    docs_mock = MagicMock()
    drive_mock = MagicMock()
    mocker.patch("googleapiclient.discovery.build", side_effect=[docs_mock, drive_mock])
    mocker.patch("google.oauth2.credentials.Credentials")
    docs_mock.documents.return_value.batchUpdate.return_value.execute.return_value = {"replies": []}

    client = GoogleDocsClient(_creds())
    requests = [{"insertText": {"location": {"index": 1}, "text": "hi"}}]
    result = client.batch_update("doc-id", requests)

    assert result == {"replies": []}
    docs_mock.documents.return_value.batchUpdate.assert_called_once_with(
        documentId="doc-id", body={"requests": requests}
    )
