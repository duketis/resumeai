"""Tests for /api/auth/google start, callback, and disconnect."""

from __future__ import annotations

import urllib.parse
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from resumeai.auth.google_oauth import OAuthError
from resumeai.settings.models import GoogleCredentials, OAuthClient, OAuthPending

if TYPE_CHECKING:
    from resumeai.settings.store import InMemorySettingsStore
    from tests.unit.api.conftest import FakeOAuthService


def _seed_client(store: InMemorySettingsStore) -> None:
    store.set_oauth_client(OAuthClient(client_id="cid", client_secret="secret"))


def _seed_creds(store: InMemorySettingsStore) -> GoogleCredentials:
    creds = GoogleCredentials(
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
    store.set_google_credentials(creds)
    return creds


# -- /api/auth/google/start --------------------------------------------------


def test_start_redirects_to_consent_url_and_records_pending(
    store: InMemorySettingsStore,
    oauth_service: FakeOAuthService,
    client: TestClient,
) -> None:
    _seed_client(store)

    response = client.get("/api/auth/google/start", follow_redirects=False)

    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("https://fake-google.test/consent?state=")

    state = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)["state"][0]
    pending = store.consume_oauth_pending(state)
    assert pending is not None
    assert pending.redirect_uri.endswith("/api/auth/google/callback")
    assert oauth_service.consent_url_calls[0][2] == state


def test_start_returns_400_when_oauth_client_missing(client: TestClient) -> None:
    response = client.get("/api/auth/google/start", follow_redirects=False)

    assert response.status_code == 400


# -- /api/auth/google/callback ----------------------------------------------


def test_callback_completes_consent_and_persists_credentials(
    store: InMemorySettingsStore,
    oauth_service: FakeOAuthService,
    client: TestClient,
) -> None:
    _seed_client(store)
    state = "fixed-state"
    store.store_oauth_pending(
        OAuthPending(
            state=state,
            redirect_uri="http://testserver/api/auth/google/callback",
            created_at=datetime(2026, 5, 9, tzinfo=UTC),
        )
    )

    response = client.get(
        f"/api/auth/google/callback?code=auth-code&state={state}",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "flash=Connected+as+connected@example.com" in response.headers["location"]

    saved = store.get_google_credentials()
    assert saved is not None
    assert saved.user_email == "connected@example.com"

    assert oauth_service.exchange_calls[0][1] == "auth-code"
    assert oauth_service.exchange_calls[0][2] == "http://testserver/api/auth/google/callback"


def test_callback_with_explicit_error_redirects_to_settings(
    client: TestClient,
) -> None:
    response = client.get("/api/auth/google/callback?error=access_denied", follow_redirects=False)

    assert response.status_code == 303
    assert "error=" in response.headers["location"]


def test_callback_returns_400_when_code_missing(client: TestClient) -> None:
    response = client.get("/api/auth/google/callback?state=abc", follow_redirects=False)
    assert response.status_code == 400


def test_callback_returns_400_for_unknown_state(client: TestClient) -> None:
    response = client.get(
        "/api/auth/google/callback?code=c&state=never-stored",
        follow_redirects=False,
    )
    assert response.status_code == 400


def test_callback_returns_400_when_oauth_client_was_cleared_mid_flow(
    store: InMemorySettingsStore, client: TestClient
) -> None:
    state = "s"
    store.store_oauth_pending(
        OAuthPending(
            state=state,
            redirect_uri="http://testserver/api/auth/google/callback",
            created_at=datetime(2026, 5, 9, tzinfo=UTC),
        )
    )
    # No oauth client stored.
    response = client.get(f"/api/auth/google/callback?code=c&state={state}", follow_redirects=False)
    assert response.status_code == 400


def test_callback_redirects_with_error_when_token_exchange_fails(
    store: InMemorySettingsStore,
    oauth_service: FakeOAuthService,
    client: TestClient,
) -> None:
    _seed_client(store)
    state = "s"
    store.store_oauth_pending(
        OAuthPending(
            state=state,
            redirect_uri="http://testserver/api/auth/google/callback",
            created_at=datetime(2026, 5, 9, tzinfo=UTC),
        )
    )
    oauth_service.exchange_error = OAuthError("token endpoint hates us")

    response = client.get(f"/api/auth/google/callback?code=c&state={state}", follow_redirects=False)

    assert response.status_code == 303
    assert "error=Token+exchange+failed" in response.headers["location"]
    assert store.get_google_credentials() is None


# -- /api/auth/google/disconnect --------------------------------------------


def test_disconnect_revokes_and_clears(
    store: InMemorySettingsStore,
    oauth_service: FakeOAuthService,
    client: TestClient,
) -> None:
    creds = _seed_creds(store)

    response = client.post("/api/auth/google/disconnect", follow_redirects=False)

    assert response.status_code == 303
    assert store.get_google_credentials() is None
    assert oauth_service.revoke_calls == [creds]


def test_disconnect_with_no_credentials_is_a_noop(
    oauth_service: FakeOAuthService, client: TestClient
) -> None:
    response = client.post("/api/auth/google/disconnect", follow_redirects=False)

    assert response.status_code == 303
    assert oauth_service.revoke_calls == []


def test_disconnect_clears_locally_even_if_revoke_fails(
    store: InMemorySettingsStore,
    oauth_service: FakeOAuthService,
    client: TestClient,
) -> None:
    _seed_creds(store)
    oauth_service.revoke_error = OAuthError("Google won't let us revoke")

    response = client.post("/api/auth/google/disconnect", follow_redirects=False)

    assert response.status_code == 303
    assert store.get_google_credentials() is None
