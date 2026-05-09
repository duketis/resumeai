"""Tests for ``GET/PUT/DELETE /api/templates``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from resumeai.settings.models import TemplateDoc

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from resumeai.settings.store import InMemorySettingsStore


def test_get_template_returns_null_when_unset(client: TestClient) -> None:
    response = client.get("/api/templates")

    assert response.status_code == 200
    assert response.json() == {"template": None}


def test_get_template_returns_serialised_payload(
    client: TestClient, store: InMemorySettingsStore
) -> None:
    store.set_template(TemplateDoc(doc_id="abc-123"))

    response = client.get("/api/templates")

    payload = response.json()
    assert payload["template"]["doc_id"] == "abc-123"
    assert payload["template"]["url"].endswith("/document/d/abc-123/edit")


def test_put_template_extracts_doc_id_from_url(
    client: TestClient, store: InMemorySettingsStore
) -> None:
    response = client.put(
        "/api/templates",
        json={"template_url": "https://docs.google.com/document/d/abc_XYZ/edit"},
    )

    assert response.status_code == 200
    saved = store.get_template()
    assert saved is not None
    assert saved.doc_id == "abc_XYZ"


def test_put_template_returns_400_on_bad_url(
    client: TestClient, store: InMemorySettingsStore
) -> None:
    response = client.put("/api/templates", json={"template_url": "https://example.com/not-a-doc"})
    assert response.status_code == 400
    assert store.get_template() is None


def test_delete_template_clears_storage(client: TestClient, store: InMemorySettingsStore) -> None:
    store.set_template(TemplateDoc(doc_id="abc"))
    response = client.delete("/api/templates")

    assert response.status_code == 204
    assert store.get_template() is None
