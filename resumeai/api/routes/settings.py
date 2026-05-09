"""Settings page + form-submit handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from resumeai.api.deps import get_settings_store
from resumeai.api.templating import templates
from resumeai.settings.models import TemplateDoc
from resumeai.settings.parser import (
    DocUrlParseError,
    OAuthClientParseError,
    parse_doc_id_from_url,
    parse_oauth_client_json,
)

if TYPE_CHECKING:
    from resumeai.settings.store import SettingsStore


router = APIRouter()


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    store: SettingsStore = Depends(get_settings_store),
) -> HTMLResponse:
    oauth_client = store.get_oauth_client()
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "oauth_client": oauth_client,
            "oauth_client_configured": oauth_client is not None,
            "credentials": store.get_google_credentials(),
            "template": store.get_template(),
            "flash": request.query_params.get("flash"),
            "error": request.query_params.get("error"),
        },
    )


@router.post("/settings/oauth-client", include_in_schema=False)
def save_oauth_client(
    client_json: str = Form(...),
    store: SettingsStore = Depends(get_settings_store),
) -> RedirectResponse:
    try:
        client = parse_oauth_client_json(client_json)
    except OAuthClientParseError as exc:
        return RedirectResponse(
            f"/settings?error=Could+not+parse+OAuth+client%3A+{exc}",
            status_code=303,
        )
    store.set_oauth_client(client)
    # New OAuth client invalidates previously stored Google credentials.
    store.clear_google_credentials()
    return RedirectResponse("/settings?flash=OAuth+client+saved", status_code=303)


@router.post("/settings/template", include_in_schema=False)
def save_template(
    template_url: str = Form(...),
    store: SettingsStore = Depends(get_settings_store),
) -> RedirectResponse:
    try:
        doc_id = parse_doc_id_from_url(template_url)
    except DocUrlParseError as exc:
        return RedirectResponse(
            f"/settings?error=Could+not+parse+template+URL%3A+{exc}",
            status_code=303,
        )
    store.set_template(TemplateDoc(doc_id=doc_id))
    return RedirectResponse("/settings?flash=Template+saved", status_code=303)
