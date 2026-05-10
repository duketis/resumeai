"""Context-files routes — server-rendered + JSON API."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from pypdf import PdfWriter

from resumeai.context_files.models import ContextFileKind

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from resumeai.context_files.store import InMemoryContextFileStore


def _empty_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# -- GET /context (server-rendered list) -----------------------------------


def test_context_page_renders_empty_state(client: TestClient) -> None:
    response = client.get("/context")
    assert response.status_code == 200
    assert "No context files yet" in response.text


def test_context_page_lists_uploaded_files(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    context_files.add(
        name="git_audit.csv",
        kind=ContextFileKind.CSV,
        extracted_text="repo,commits\nacme/x,42\n",
        byte_size=20,
        tags=("project:acme",),
        note="Q1 2026",
    )
    response = client.get("/context")
    body = response.text
    assert "git_audit.csv" in body
    assert "project:acme" in body
    assert "Q1 2026" in body


def test_context_page_renders_flash_message(client: TestClient) -> None:
    response = client.get("/context?flash=Saved")
    assert "Saved" in response.text


def test_context_page_renders_error_message(client: TestClient) -> None:
    response = client.get("/context?error=Boom")
    assert "Boom" in response.text


def test_context_page_truncates_long_extracted_text_preview(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    long_text = "x" * 5000
    context_files.add(
        name="big.txt",
        kind=ContextFileKind.TEXT,
        extracted_text=long_text,
        byte_size=5000,
    )
    response = client.get("/context")
    assert "more characters truncated" in response.text


# -- POST /context (upload) -------------------------------------------------


def test_upload_text_file_stores_extracted_text(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    response = client.post(
        "/context",
        files={"upload": ("notes.txt", b"hello world", "text/plain")},
        data={"note": "test note", "tags": "project:x, role:eng"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "flash=" in response.headers["location"]

    files = context_files.list_all()
    assert len(files) == 1
    assert files[0].name == "notes.txt"
    assert files[0].extracted_text == "hello world"
    assert files[0].tags == ("project:x", "role:eng")
    assert files[0].note == "test note"


def test_upload_csv_file_keeps_csv_text(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    client.post(
        "/context",
        files={"upload": ("data.csv", b"name,score\nAlex,42\n", "text/csv")},
        follow_redirects=False,
    )
    files = context_files.list_all()
    assert files[0].kind is ContextFileKind.CSV
    assert "Alex,42" in files[0].extracted_text


def test_upload_pdf_file_extracts_pdf_kind(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    client.post(
        "/context",
        files={"upload": ("doc.pdf", _empty_pdf(), "application/pdf")},
        follow_redirects=False,
    )
    files = context_files.list_all()
    assert len(files) == 1
    assert files[0].kind is ContextFileKind.PDF


def test_upload_rejects_unsupported_extension(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    response = client.post(
        "/context",
        files={"upload": ("photo.jpg", b"\x00\x01", "image/jpeg")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert context_files.list_all() == []


def test_upload_rejects_invalid_utf8_text(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    response = client.post(
        "/context",
        files={"upload": ("notes.txt", b"\xff\xfe\x00\x00bad", "text/plain")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert context_files.list_all() == []


def test_upload_without_file_returns_422(client: TestClient) -> None:
    """FastAPI's File(...) requirement rejects requests with no file at all."""
    response = client.post("/context", follow_redirects=False)
    assert response.status_code == 422


def test_upload_with_no_tags_persists_empty_tag_tuple(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    client.post(
        "/context",
        files={"upload": ("x.txt", b"hi", "text/plain")},
        data={"tags": "  ,  "},
        follow_redirects=False,
    )
    assert context_files.list_all()[0].tags == ()


# -- POST /context/{id}/delete (form delete) -------------------------------


def test_form_delete_removes_file(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    file = context_files.add(
        name="x.txt", kind=ContextFileKind.TEXT, extracted_text="t", byte_size=1
    )
    response = client.post(f"/context/{file.id}/delete", follow_redirects=False)

    assert response.status_code == 303
    assert context_files.get(file.id) is None


def test_form_delete_for_unknown_id_still_redirects(client: TestClient) -> None:
    response = client.post("/context/never-saved/delete", follow_redirects=False)
    assert response.status_code == 303


# -- JSON API: GET /api/context, GET /api/context/{id}, DELETE /api/context/{id}


def test_api_list_context_files(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    file = context_files.add(
        name="x.txt", kind=ContextFileKind.TEXT, extracted_text="t", byte_size=1
    )
    response = client.get("/api/context")
    payload = response.json()
    assert payload["files"][0]["id"] == file.id


def test_api_get_context_file(client: TestClient, context_files: InMemoryContextFileStore) -> None:
    file = context_files.add(
        name="x.txt", kind=ContextFileKind.TEXT, extracted_text="hi", byte_size=2
    )
    response = client.get(f"/api/context/{file.id}")
    assert response.status_code == 200
    assert response.json()["extracted_text"] == "hi"


def test_api_get_context_file_returns_404_for_unknown_id(client: TestClient) -> None:
    response = client.get("/api/context/never-saved")
    assert response.status_code == 404


def test_api_delete_context_file(
    client: TestClient, context_files: InMemoryContextFileStore
) -> None:
    file = context_files.add(
        name="x.txt", kind=ContextFileKind.TEXT, extracted_text="t", byte_size=1
    )
    response = client.delete(f"/api/context/{file.id}")
    assert response.status_code == 204
    assert context_files.get(file.id) is None


def test_api_delete_context_file_returns_404_for_unknown_id(
    client: TestClient,
) -> None:
    response = client.delete("/api/context/never-saved")
    assert response.status_code == 404
