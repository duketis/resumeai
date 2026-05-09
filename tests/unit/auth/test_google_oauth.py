"""Tests for the Google OAuth service.

Uses respx to stub Google's token + userinfo + revoke endpoints so the suite
runs offline.
"""

from __future__ import annotations

import urllib.parse
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from resumeai.auth.google_oauth import (
    REVOKE_ENDPOINT,
    USERINFO_ENDPOINT,
    GoogleOAuthService,
    OAuthError,
)
from resumeai.settings.models import GoogleCredentials, OAuthClient

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def service() -> Iterator[GoogleOAuthService]:
    s = GoogleOAuthService()
    try:
        yield s
    finally:
        s.close()


def _client() -> OAuthClient:
    return OAuthClient(client_id="cid.apps.googleusercontent.com", client_secret="secret")


def _creds(*, refresh: str | None = "rt") -> GoogleCredentials:
    return GoogleCredentials(
        access_token="at",
        refresh_token=refresh,
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csec",
        scopes=("openid", "email"),
        expiry=datetime(2026, 5, 9, tzinfo=UTC),
        user_email="x@example.com",
        user_id="42",
    )


# -- build_consent_url -------------------------------------------------------


def test_consent_url_carries_all_required_params(service: GoogleOAuthService) -> None:
    url = service.build_consent_url(
        _client(),
        redirect_uri="http://localhost:7842/api/auth/google/callback",
        state="abc",
        scopes=("openid", "email"),
    )

    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "accounts.google.com"
    assert qs["client_id"] == ["cid.apps.googleusercontent.com"]
    assert qs["redirect_uri"] == ["http://localhost:7842/api/auth/google/callback"]
    assert qs["response_type"] == ["code"]
    assert qs["scope"] == ["openid email"]
    assert qs["state"] == ["abc"]
    assert qs["access_type"] == ["offline"]
    assert qs["prompt"] == ["consent"]


def test_consent_url_uses_default_scopes_when_omitted(service: GoogleOAuthService) -> None:
    url = service.build_consent_url(_client(), "http://x", "s")

    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert "https://www.googleapis.com/auth/documents" in qs["scope"][0]
    assert "https://www.googleapis.com/auth/drive.file" in qs["scope"][0]


# -- exchange_code -----------------------------------------------------------


def _ok_token(refresh_token: str | None = "rt-from-google") -> dict[str, object]:
    payload: dict[str, object] = {
        "access_token": "at-from-google",
        "expires_in": 3600,
        "scope": "openid email https://www.googleapis.com/auth/documents",
        "token_type": "Bearer",
    }
    if refresh_token is not None:
        payload["refresh_token"] = refresh_token
    return payload


def _ok_userinfo() -> dict[str, object]:
    return {"sub": "user-42", "email": "person@example.com"}


@respx.mock
def test_exchange_code_returns_full_credentials(service: GoogleOAuthService) -> None:
    client = _client()
    respx.post(client.token_uri).mock(return_value=httpx.Response(200, json=_ok_token()))
    respx.get(USERINFO_ENDPOINT).mock(return_value=httpx.Response(200, json=_ok_userinfo()))

    creds = service.exchange_code(client, "auth-code", "http://x/callback")

    assert creds.access_token == "at-from-google"
    assert creds.refresh_token == "rt-from-google"
    assert creds.token_uri == client.token_uri
    assert creds.client_id == client.client_id
    assert creds.client_secret == client.client_secret
    assert creds.scopes == (
        "openid",
        "email",
        "https://www.googleapis.com/auth/documents",
    )
    assert creds.user_email == "person@example.com"
    assert creds.user_id == "user-42"
    assert creds.expiry is not None
    assert creds.expiry > datetime.now(UTC)


@respx.mock
def test_exchange_code_handles_missing_refresh_token(
    service: GoogleOAuthService,
) -> None:
    client = _client()
    respx.post(client.token_uri).mock(
        return_value=httpx.Response(200, json=_ok_token(refresh_token=None))
    )
    respx.get(USERINFO_ENDPOINT).mock(return_value=httpx.Response(200, json=_ok_userinfo()))

    creds = service.exchange_code(client, "code", "http://x/callback")

    assert creds.refresh_token is None


@respx.mock
def test_exchange_code_handles_missing_expires_in(service: GoogleOAuthService) -> None:
    client = _client()
    token = _ok_token()
    del token["expires_in"]
    respx.post(client.token_uri).mock(return_value=httpx.Response(200, json=token))
    respx.get(USERINFO_ENDPOINT).mock(return_value=httpx.Response(200, json=_ok_userinfo()))

    creds = service.exchange_code(client, "code", "http://x/callback")
    assert creds.expiry is None


@respx.mock
def test_exchange_code_raises_on_token_endpoint_error(
    service: GoogleOAuthService,
) -> None:
    client = _client()
    respx.post(client.token_uri).mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"})
    )

    with pytest.raises(OAuthError, match=r"token exchange failed.*400"):
        service.exchange_code(client, "code", "http://x/callback")


@respx.mock
def test_exchange_code_raises_on_userinfo_error(service: GoogleOAuthService) -> None:
    client = _client()
    respx.post(client.token_uri).mock(return_value=httpx.Response(200, json=_ok_token()))
    respx.get(USERINFO_ENDPOINT).mock(return_value=httpx.Response(401))

    with pytest.raises(OAuthError, match=r"userinfo fetch failed.*401"):
        service.exchange_code(client, "code", "http://x/callback")


@respx.mock
def test_exchange_code_raises_when_token_response_missing_access_token(
    service: GoogleOAuthService,
) -> None:
    client = _client()
    respx.post(client.token_uri).mock(return_value=httpx.Response(200, json={"scope": "x"}))

    with pytest.raises(OAuthError, match="missing required string field 'access_token'"):
        service.exchange_code(client, "code", "http://x/callback")


# -- refresh -----------------------------------------------------------------


@respx.mock
def test_refresh_updates_access_token_and_expiry(service: GoogleOAuthService) -> None:
    creds = _creds()
    respx.post(creds.token_uri).mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "new-at", "expires_in": 3600, "token_type": "Bearer"},
        )
    )

    refreshed = service.refresh(creds)

    assert refreshed.access_token == "new-at"
    assert refreshed.refresh_token == creds.refresh_token  # unchanged
    assert refreshed.expiry is not None
    assert refreshed.expiry > datetime.now(UTC)


def test_refresh_raises_when_no_refresh_token(service: GoogleOAuthService) -> None:
    with pytest.raises(OAuthError, match="no refresh token"):
        service.refresh(_creds(refresh=None))


@respx.mock
def test_refresh_raises_on_http_error(service: GoogleOAuthService) -> None:
    creds = _creds()
    respx.post(creds.token_uri).mock(return_value=httpx.Response(401))

    with pytest.raises(OAuthError, match=r"refresh failed.*401"):
        service.refresh(creds)


# -- revoke ------------------------------------------------------------------


@respx.mock
def test_revoke_treats_200_as_success(service: GoogleOAuthService) -> None:
    respx.post(REVOKE_ENDPOINT).mock(return_value=httpx.Response(200))

    service.revoke(_creds())  # must not raise


@respx.mock
def test_revoke_treats_400_as_success(service: GoogleOAuthService) -> None:
    # Google returns 400 if the token was already invalid. Idempotency.
    respx.post(REVOKE_ENDPOINT).mock(return_value=httpx.Response(400))

    service.revoke(_creds())  # must not raise


@respx.mock
def test_revoke_raises_on_other_errors(service: GoogleOAuthService) -> None:
    respx.post(REVOKE_ENDPOINT).mock(return_value=httpx.Response(500))

    with pytest.raises(OAuthError, match=r"revoke failed.*500"):
        service.revoke(_creds())
