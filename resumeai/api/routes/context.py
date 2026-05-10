"""Context-files management — server-rendered upload page + JSON API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from resumeai.api.deps import get_context_file_store
from resumeai.api.templating import templates
from resumeai.context_files.extraction import ExtractionError, extract_text

if TYPE_CHECKING:
    from resumeai.context_files.store import ContextFileStore


router = APIRouter()


@router.get("/context", response_class=HTMLResponse)
def context_page(
    request: Request,
    store: ContextFileStore = Depends(get_context_file_store),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "context.html",
        {
            "files": store.list_all(),
            "flash": request.query_params.get("flash"),
            "error": request.query_params.get("error"),
        },
    )


@router.post("/context", include_in_schema=False)
async def upload_context_file(
    upload: UploadFile = File(...),
    note: str = Form(""),
    tags: str = Form(""),
    store: ContextFileStore = Depends(get_context_file_store),
) -> RedirectResponse:
    # FastAPI's ``File(...)`` already rejects requests with no file (422),
    # so ``upload.filename`` is reliably present here. Defensive default
    # only in case a programmatic client posts a file with a stripped
    # Content-Disposition filename.
    filename = upload.filename or "uploaded"
    data = await upload.read()
    try:
        kind, text = extract_text(filename, data)
    except ExtractionError as exc:
        return RedirectResponse(f"/context?error=Could+not+extract+text%3A+{exc}", status_code=303)
    tag_tuple = tuple(t.strip() for t in tags.split(",") if t.strip())
    store.add(
        name=filename,
        kind=kind,
        extracted_text=text,
        byte_size=len(data),
        tags=tag_tuple,
        note=note.strip(),
    )
    return RedirectResponse(f"/context?flash=Uploaded+{filename}", status_code=303)


@router.post("/context/{file_id}/delete", include_in_schema=False)
def delete_context_file_form(
    file_id: str,
    store: ContextFileStore = Depends(get_context_file_store),
) -> RedirectResponse:
    store.remove(file_id)
    return RedirectResponse("/context?flash=Removed", status_code=303)


# -- JSON API ---------------------------------------------------------------


@router.get("/api/context")
def list_context_files(
    store: ContextFileStore = Depends(get_context_file_store),
) -> dict[str, Any]:
    return {"files": [_serialise(file) for file in store.list_all()]}


@router.get("/api/context/{file_id}")
def get_context_file(
    file_id: str,
    store: ContextFileStore = Depends(get_context_file_store),
) -> dict[str, Any]:
    file = store.get(file_id)
    if file is None:
        raise HTTPException(status_code=404, detail=f"unknown file {file_id!r}")
    return _serialise(file)


@router.delete("/api/context/{file_id}", status_code=204)
def delete_context_file(
    file_id: str,
    store: ContextFileStore = Depends(get_context_file_store),
) -> None:
    if not store.remove(file_id):
        raise HTTPException(status_code=404, detail=f"unknown file {file_id!r}")


def _serialise(file: object) -> dict[str, Any]:
    import json  # noqa: PLC0415

    parsed: dict[str, Any] = json.loads(file.model_dump_json())  # type: ignore[attr-defined]
    return parsed
