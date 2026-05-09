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
    return templates.TemplateResponse(request, "onboarding.html", {})


@router.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok"}
