"""Settings models — frozen-ness and basic field acceptance."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from resumeai.settings.models import (
    GoogleCredentials,
    OAuthClient,
    OAuthPending,
    TemplateDoc,
)


def _sample_creds() -> GoogleCredentials:
    return GoogleCredentials(
        access_token="at",
        refresh_token="rt",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csec",
        scopes=("openid", "email"),
        expiry=datetime(2026, 5, 9, 0, 0, tzinfo=UTC),
        user_email="x@example.com",
        user_id="42",
    )


def test_oauth_client_has_sensible_defaults() -> None:
    client = OAuthClient(client_id="x", client_secret="y")

    assert client.auth_uri == "https://accounts.google.com/o/oauth2/auth"
    assert client.token_uri == "https://oauth2.googleapis.com/token"
    assert client.project_id is None
    assert client.redirect_uris == ()


def test_oauth_client_is_frozen() -> None:
    client = OAuthClient(client_id="x", client_secret="y")
    with pytest.raises(ValidationError):
        client.client_id = "z"


def test_google_credentials_round_trips_through_json() -> None:
    creds = _sample_creds()
    parsed = GoogleCredentials.model_validate_json(creds.model_dump_json())
    assert parsed == creds


def test_google_credentials_allows_missing_refresh_token() -> None:
    creds = _sample_creds().model_copy(update={"refresh_token": None})
    assert creds.refresh_token is None


def test_oauth_pending_carries_state_and_redirect() -> None:
    pending = OAuthPending(
        state="abc",
        redirect_uri="http://localhost:7842/callback",
        created_at=datetime(2026, 5, 9, tzinfo=UTC),
    )
    assert pending.state == "abc"
    assert pending.redirect_uri.endswith("/callback")


def test_template_doc_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        TemplateDoc(doc_id="")
