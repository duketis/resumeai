"""Pydantic models for the Settings store.

These shapes are the storage contract. Routes and services consume + produce
these models; the store's only job is to persist + retrieve them.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class OAuthClient(BaseModel):
    """OAuth 2.0 client created by the user in Google Cloud Console.

    The user pastes / uploads ``client_secret.json`` into the Settings page; we
    parse it into this model and persist. These are NOT the access/refresh
    tokens — those land later in :class:`GoogleCredentials` after a successful
    consent flow.
    """

    model_config = ConfigDict(frozen=True)

    client_id: str
    client_secret: str
    auth_uri: str = "https://accounts.google.com/o/oauth2/auth"
    # S105 false positive: token_uri is a public OAuth endpoint, not a secret.
    token_uri: str = "https://oauth2.googleapis.com/token"  # noqa: S105
    project_id: str | None = None
    redirect_uris: tuple[str, ...] = ()


class GoogleCredentials(BaseModel):
    """Access + refresh tokens persisted after a successful OAuth consent.

    ``user_email`` is shown back to the user on the Settings page so they can
    confirm the right Google account is connected.
    """

    model_config = ConfigDict(frozen=True)

    access_token: str
    refresh_token: str | None
    token_uri: str
    client_id: str
    client_secret: str
    scopes: tuple[str, ...]
    expiry: datetime | None
    user_email: str
    user_id: str


class OAuthPending(BaseModel):
    """An in-flight consent flow.

    The state token defends against CSRF on the OAuth callback. We store one
    row per pending flow and consume it on callback receipt; rows older than
    the TTL (10 min) are treated as expired.
    """

    model_config = ConfigDict(frozen=True)

    state: str
    redirect_uri: str
    created_at: datetime


class TemplateDoc(BaseModel):
    """Master Google Doc the user has registered as their resume template.

    The renderer copies this doc on every tailoring run; the master itself is
    never written to.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    nickname: str | None = None
