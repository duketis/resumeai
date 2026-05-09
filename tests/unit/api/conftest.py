"""Shared fixtures for API route tests.

- ``store``: an :class:`InMemorySettingsStore` so tests don't touch SQLite.
- ``oauth_service``: a :class:`FakeOAuthService` with scripted responses.
- ``client``: a FastAPI ``TestClient`` for the app wired with the above.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from resumeai.api.app import create_app
from resumeai.auth.google_oauth import OAuthError
from resumeai.settings.models import GoogleCredentials, OAuthClient
from resumeai.settings.store import InMemorySettingsStore

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class FakeOAuthService:
    """In-memory ``OAuthService``. Scripts return values + records calls."""

    def __init__(self) -> None:
        self.consent_url_calls: list[tuple[OAuthClient, str, str, Sequence[str]]] = []
        self.exchange_calls: list[tuple[OAuthClient, str, str]] = []
        self.refresh_calls: list[GoogleCredentials] = []
        self.revoke_calls: list[GoogleCredentials] = []
        self.exchange_returns: GoogleCredentials | None = None
        self.exchange_error: Exception | None = None
        self.revoke_error: Exception | None = None

    def build_consent_url(
        self,
        client: OAuthClient,
        redirect_uri: str,
        state: str,
        scopes: Sequence[str] = (),
    ) -> str:
        self.consent_url_calls.append((client, redirect_uri, state, scopes))
        return f"https://fake-google.test/consent?state={state}"

    def exchange_code(self, client: OAuthClient, code: str, redirect_uri: str) -> GoogleCredentials:
        self.exchange_calls.append((client, code, redirect_uri))
        if self.exchange_error is not None:
            raise self.exchange_error
        if self.exchange_returns is None:
            return _default_creds()
        return self.exchange_returns

    def refresh(self, creds: GoogleCredentials) -> GoogleCredentials:
        self.refresh_calls.append(creds)
        return creds

    def revoke(self, creds: GoogleCredentials) -> None:
        self.revoke_calls.append(creds)
        if self.revoke_error is not None:
            raise self.revoke_error


def _default_creds() -> GoogleCredentials:
    return GoogleCredentials(
        access_token="at",
        refresh_token="rt",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csec",
        scopes=("openid",),
        expiry=datetime(2026, 5, 9, 12, tzinfo=UTC),
        user_email="connected@example.com",
        user_id="42",
    )


@pytest.fixture
def store() -> InMemorySettingsStore:
    return InMemorySettingsStore()


@pytest.fixture
def oauth_service() -> FakeOAuthService:
    return FakeOAuthService()


@pytest.fixture
def client(store: InMemorySettingsStore, oauth_service: FakeOAuthService) -> Iterator[TestClient]:
    app = create_app(settings_store=store, oauth_service=oauth_service)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def oauth_client_json() -> str:
    return json.dumps(
        {
            "web": {
                "client_id": "cid.apps.googleusercontent.com",
                "client_secret": "supersecret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "project_id": "resumeai-456",
            }
        }
    )


def make_oauth_error(message: str = "boom") -> OAuthError:
    return OAuthError(message)


__all__ = ["FakeOAuthService", "make_oauth_error"]


def _placeholder_use(_: Any) -> None:
    """Force pytest to recognise this module's helpers as in use."""
