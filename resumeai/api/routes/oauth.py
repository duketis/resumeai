"""Google OAuth start / callback / disconnect."""

from __future__ import annotations

import contextlib
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from resumeai.api.deps import get_oauth_service, get_settings_store
from resumeai.auth.google_oauth import OAuthError
from resumeai.settings.models import OAuthPending

if TYPE_CHECKING:
    from resumeai.auth.google_oauth import OAuthService
    from resumeai.settings.store import SettingsStore


router = APIRouter(prefix="/api/auth/google")


def _redirect_uri_for(request: Request) -> str:
    """The callback URL Google should redirect to. Derived from the request."""
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/auth/google/callback"


@router.get("/start")
def start(
    request: Request,
    store: SettingsStore = Depends(get_settings_store),
    service: OAuthService = Depends(get_oauth_service),
) -> RedirectResponse:
    client = store.get_oauth_client()
    if client is None:
        raise HTTPException(
            status_code=400,
            detail="OAuth client not configured. Save one in /settings first.",
        )

    state = secrets.token_urlsafe(32)
    redirect_uri = _redirect_uri_for(request)
    store.store_oauth_pending(
        OAuthPending(state=state, redirect_uri=redirect_uri, created_at=datetime.now(UTC))
    )
    consent_url = service.build_consent_url(client, redirect_uri, state)
    return RedirectResponse(consent_url, status_code=303)


@router.get("/callback", include_in_schema=False)
def callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    store: SettingsStore = Depends(get_settings_store),
    service: OAuthService = Depends(get_oauth_service),
) -> RedirectResponse:
    if error:
        return RedirectResponse(
            f"/settings?error=Google+denied+consent%3A+{error}",
            status_code=303,
        )
    if not code or not state:
        raise HTTPException(status_code=400, detail="callback missing code or state")

    pending = store.consume_oauth_pending(state)
    if pending is None:
        raise HTTPException(status_code=400, detail="unknown or expired state token")

    client = store.get_oauth_client()
    if client is None:
        raise HTTPException(status_code=400, detail="OAuth client was cleared mid-flow")

    try:
        creds = service.exchange_code(client, code, pending.redirect_uri)
    except OAuthError as exc:
        return RedirectResponse(f"/settings?error=Token+exchange+failed%3A+{exc}", status_code=303)

    store.set_google_credentials(creds)
    return RedirectResponse(f"/settings?flash=Connected+as+{creds.user_email}", status_code=303)


@router.post("/disconnect", include_in_schema=False)
def disconnect(
    store: SettingsStore = Depends(get_settings_store),
    service: OAuthService = Depends(get_oauth_service),
) -> RedirectResponse:
    creds = store.get_google_credentials()
    if creds is not None:
        # Best-effort revoke; we still clear locally even if the network call fails.
        with contextlib.suppress(OAuthError):
            service.revoke(creds)
        store.clear_google_credentials()
    return RedirectResponse("/settings?flash=Disconnected", status_code=303)
