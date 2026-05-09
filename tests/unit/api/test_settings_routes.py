"""Tests for the Settings page + form-submit handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from resumeai.settings.models import GoogleCredentials, TemplateDoc

if TYPE_CHECKING:
    from resumeai.settings.store import InMemorySettingsStore


# -- GET /settings -----------------------------------------------------------


def test_settings_page_warns_when_oauth_client_missing(client: TestClient) -> None:
    response = client.get("/settings")

    assert response.status_code == 200
    body = response.text
    assert "No OAuth client yet" in body
    assert "onboarding wizard" in body


def test_settings_page_shows_oauth_client_when_configured(
    client: TestClient, oauth_client_json: str
) -> None:
    client.post("/settings/oauth-client", data={"client_json": oauth_client_json})

    response = client.get("/settings")

    body = response.text
    assert "An OAuth client is configured" in body
    assert "resumeai-456" in body


def test_settings_page_shows_connected_account(
    store: InMemorySettingsStore, client: TestClient
) -> None:
    from datetime import UTC  # noqa: PLC0415
    from datetime import datetime as _datetime  # noqa: PLC0415

    creds = GoogleCredentials(
        access_token="at",
        refresh_token="rt",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csec",
        scopes=("openid",),
        expiry=_datetime(2026, 5, 9, 12, tzinfo=UTC),
        user_email="user@example.com",
        user_id="42",
    )
    store.set_google_credentials(creds)

    response = client.get("/settings")

    assert "user@example.com" in response.text
    assert "Disconnect" in response.text


def test_settings_page_shows_template(store: InMemorySettingsStore, client: TestClient) -> None:
    store.set_template(TemplateDoc(doc_id="abc-123"))

    response = client.get("/settings")

    body = response.text
    assert "abc-123" in body
    assert "https://docs.google.com/document/d/abc-123/edit" in body


def test_settings_page_renders_flash_message(client: TestClient) -> None:
    response = client.get("/settings?flash=All+done")
    assert "All done" in response.text


def test_settings_page_renders_error_message(client: TestClient) -> None:
    response = client.get("/settings?error=Boom")
    assert "Boom" in response.text


# -- POST /settings/oauth-client --------------------------------------------


def test_post_oauth_client_persists_and_redirects(
    store: InMemorySettingsStore, client: TestClient, oauth_client_json: str
) -> None:
    response = client.post(
        "/settings/oauth-client",
        data={"client_json": oauth_client_json},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?flash=")

    saved = store.get_oauth_client()
    assert saved is not None
    assert saved.client_id == "cid.apps.googleusercontent.com"


def test_post_oauth_client_clears_existing_credentials(
    store: InMemorySettingsStore, client: TestClient, oauth_client_json: str
) -> None:
    """Credentials tied to the *old* client must be invalidated when a new
    client is uploaded — they wouldn't refresh against the new client_secret
    anyway."""
    from datetime import UTC  # noqa: PLC0415
    from datetime import datetime as _datetime  # noqa: PLC0415

    store.set_google_credentials(
        GoogleCredentials(
            access_token="old",
            refresh_token="rt",
            token_uri="t",
            client_id="old-cid",
            client_secret="old-secret",
            scopes=("openid",),
            expiry=_datetime(2026, 5, 9, tzinfo=UTC),
            user_email="x@example.com",
            user_id="42",
        )
    )

    client.post("/settings/oauth-client", data={"client_json": oauth_client_json})

    assert store.get_google_credentials() is None


def test_post_oauth_client_redirects_with_error_on_bad_json(
    store: InMemorySettingsStore, client: TestClient
) -> None:
    response = client.post(
        "/settings/oauth-client",
        data={"client_json": "{not json"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert store.get_oauth_client() is None


# -- POST /settings/template -------------------------------------------------


def test_post_template_extracts_doc_id_from_url(
    store: InMemorySettingsStore, client: TestClient
) -> None:
    response = client.post(
        "/settings/template",
        data={"template_url": "https://docs.google.com/document/d/abc123_XYZ/edit"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?flash=")

    saved = store.get_template()
    assert saved is not None
    assert saved.doc_id == "abc123_XYZ"


def test_post_template_accepts_bare_doc_id(
    store: InMemorySettingsStore, client: TestClient
) -> None:
    client.post("/settings/template", data={"template_url": "bare-id_123"})

    saved = store.get_template()
    assert saved is not None
    assert saved.doc_id == "bare-id_123"


def test_post_template_redirects_with_error_on_bad_url(
    store: InMemorySettingsStore, client: TestClient
) -> None:
    response = client.post(
        "/settings/template",
        data={"template_url": "https://example.com/not-a-doc"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert store.get_template() is None
