"""Template registration API.

Phase 1's settings page already supports paste-in template registration via
``POST /settings/template`` (form). This module exposes the same data as a
JSON API for the React frontend to drive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body, Depends, HTTPException

from resumeai.api.deps import get_settings_store
from resumeai.settings.models import TemplateDoc
from resumeai.settings.parser import DocUrlParseError, parse_doc_id_from_url

if TYPE_CHECKING:
    from resumeai.settings.store import SettingsStore


router = APIRouter(prefix="/api/templates")


@router.get("")
def get_template(
    store: SettingsStore = Depends(get_settings_store),
) -> dict[str, Any]:
    template = store.get_template()
    return {"template": (None if template is None else _serialise(template))}


@router.put("")
def put_template(
    template_url: str = Body(..., embed=True),
    store: SettingsStore = Depends(get_settings_store),
) -> dict[str, Any]:
    try:
        doc_id = parse_doc_id_from_url(template_url)
    except DocUrlParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    template = TemplateDoc(doc_id=doc_id)
    store.set_template(template)
    return {"template": _serialise(template)}


@router.delete("", status_code=204)
def delete_template(
    store: SettingsStore = Depends(get_settings_store),
) -> None:
    store.clear_template()


def _serialise(template: TemplateDoc) -> dict[str, Any]:
    return {
        "doc_id": template.doc_id,
        "nickname": template.nickname,
        "url": f"https://docs.google.com/document/d/{template.doc_id}/edit",
    }
