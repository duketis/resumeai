"""Top-level routes: ``GET /`` (auto-route) and ``GET /onboarding``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from resumeai.api.deps import get_settings_store
from resumeai.api.templating import templates

if TYPE_CHECKING:
    from resumeai.settings.store import SettingsStore


router = APIRouter()


@router.get("/", include_in_schema=False)
def index(
    store: SettingsStore = Depends(get_settings_store),
) -> RedirectResponse:
    """Land on /onboarding when nothing's set up yet, /settings otherwise."""
    if store.get_oauth_client() is None:
        return RedirectResponse("/onboarding", status_code=303)
    return RedirectResponse("/settings", status_code=303)


@router.get("/onboarding", response_class=HTMLResponse)
def onboarding(request: Request) -> HTMLResponse:
    """Render the wizard.

    The redirect URI shown to the user is derived from the current
    request's base URL — so whatever hostname they typed in the browser
    (``localhost``, ``127.0.0.1``, an IP, etc.) is what they're told to
    register. The OAuth start route uses the same derivation, so the
    two stay in lockstep and the user can't trip over the
    ``localhost`` / ``127.0.0.1`` foot-gun.
    """
    base = str(request.base_url).rstrip("/")
    callback = f"{base}/api/auth/google/callback"
    return templates.TemplateResponse(
        request,
        "onboarding.html",
        {"callback_url": callback, "base_url": base},
    )


@router.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok"}
