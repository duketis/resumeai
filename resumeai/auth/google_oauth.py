"""Thin OAuth 2.0 client tailored to the Google identity + Docs/Drive scopes.

We deliberately implement the four operations we need (consent URL, code
exchange, refresh, revoke) on top of httpx instead of pulling in
``google-auth-oauthlib``. The library adds opinionated state and is awkward
to test; the protocol surface is small and the request shapes are well
documented at https://developers.google.com/identity/protocols/oauth2.

Returned credentials are converted to ``google.oauth2.credentials.Credentials``
at the DocsClient boundary, where the Google API client needs them.
"""

from __future__ import annotations

import urllib.parse
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import httpx

from resumeai.settings.models import GoogleCredentials, OAuthClient

if TYPE_CHECKING:
    from collections.abc import Sequence


USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

# Scopes resumeai requests on every consent flow:
# - openid + email + profile  → identify the user (shown back in Settings)
# - documents                 → read + edit Google Docs (incl. tailored copies)
# - drive.file                → create + manage docs the *app* created (the
#                              tailored copies). drive.file does NOT grant
#                              access to the user's other Drive files.
DEFAULT_SCOPES: tuple[str, ...] = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file",
)


class OAuthError(RuntimeError):
    """Raised when an OAuth HTTP exchange fails or returns malformed data."""


class OAuthService(Protocol):
    """The four operations the API + services need from an OAuth provider."""

    def build_consent_url(
        self,
        client: OAuthClient,
        redirect_uri: str,
        state: str,
        scopes: Sequence[str] = DEFAULT_SCOPES,
    ) -> str: ...

    def exchange_code(
        self, client: OAuthClient, code: str, redirect_uri: str
    ) -> GoogleCredentials: ...

    def refresh(self, creds: GoogleCredentials) -> GoogleCredentials: ...

    def revoke(self, creds: GoogleCredentials) -> None: ...


class GoogleOAuthService:
    """Concrete httpx-backed implementation of :class:`OAuthService`."""

    def __init__(self, http_client: httpx.Client | None = None) -> None:
        self._http = http_client or httpx.Client(timeout=10.0)

    def close(self) -> None:
        self._http.close()

    def build_consent_url(
        self,
        client: OAuthClient,
        redirect_uri: str,
        state: str,
        scopes: Sequence[str] = DEFAULT_SCOPES,
    ) -> str:
        """Build the URL the user is redirected to for the consent screen.

        ``access_type=offline`` + ``prompt=consent`` ensures Google returns a
        refresh token on every consent (without ``prompt=consent`` Google only
        returns a refresh token on the *first* grant for that client/user pair,
        which is hostile to dev/test cycles).
        """
        params = {
            "client_id": client.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "include_granted_scopes": "true",
        }
        return f"{client.auth_uri}?{urllib.parse.urlencode(params)}"

    def exchange_code(self, client: OAuthClient, code: str, redirect_uri: str) -> GoogleCredentials:
        """Trade an authorization code for tokens + identify the user."""
        token_resp = self._http.post(
            client.token_uri,
            data={
                "code": code,
                "client_id": client.client_id,
                "client_secret": client.client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != httpx.codes.OK:
            raise OAuthError(f"token exchange failed: {token_resp.status_code} {token_resp.text}")
        token: dict[str, object] = token_resp.json()

        access_token = _required_str(token, "access_token")
        refresh_token = token.get("refresh_token")
        refresh_token_str: str | None = str(refresh_token) if refresh_token is not None else None
        scope_str = _required_str(token, "scope")
        expiry = _expiry_from(token)

        userinfo_resp = self._http.get(
            USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if userinfo_resp.status_code != httpx.codes.OK:
            raise OAuthError(
                f"userinfo fetch failed: {userinfo_resp.status_code} {userinfo_resp.text}"
            )
        userinfo: dict[str, object] = userinfo_resp.json()

        return GoogleCredentials(
            access_token=access_token,
            refresh_token=refresh_token_str,
            token_uri=client.token_uri,
            client_id=client.client_id,
            client_secret=client.client_secret,
            scopes=tuple(scope_str.split()),
            expiry=expiry,
            user_email=_required_str(userinfo, "email"),
            user_id=_required_str(userinfo, "sub"),
        )

    def refresh(self, creds: GoogleCredentials) -> GoogleCredentials:
        """Exchange the refresh token for a new access token + expiry."""
        if creds.refresh_token is None:
            raise OAuthError("cannot refresh — credentials have no refresh token")

        resp = self._http.post(
            creds.token_uri,
            data={
                "refresh_token": creds.refresh_token,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "grant_type": "refresh_token",
            },
        )
        if resp.status_code != httpx.codes.OK:
            raise OAuthError(f"refresh failed: {resp.status_code} {resp.text}")
        token: dict[str, object] = resp.json()

        access_token = _required_str(token, "access_token")
        expiry = _expiry_from(token)

        return creds.model_copy(update={"access_token": access_token, "expiry": expiry})

    def revoke(self, creds: GoogleCredentials) -> None:
        """Best-effort revocation. Google returns 200 even for an unknown token."""
        resp = self._http.post(REVOKE_ENDPOINT, params={"token": creds.access_token})
        # 200 = revoked. 400 = already invalid / unknown — treat as success.
        if resp.status_code not in (httpx.codes.OK, httpx.codes.BAD_REQUEST):
            raise OAuthError(f"revoke failed: {resp.status_code} {resp.text}")


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise OAuthError(f"response missing required string field {key!r}")
    return value


def _expiry_from(payload: dict[str, object]) -> datetime | None:
    """Pull ``expires_in`` (seconds) from a token response and return an absolute
    UTC expiry. Returns None if the field is missing, zero, or non-numeric.
    """
    raw = payload.get("expires_in")
    if not isinstance(raw, int | float) or raw <= 0:
        return None
    return datetime.now(UTC) + timedelta(seconds=int(raw))
