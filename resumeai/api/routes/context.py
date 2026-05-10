"""Context-files management — server-rendered upload page + JSON API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from resumeai.api.deps import get_context_file_store
from resumeai.api.templating import templates
from resumeai.context_files.extraction import ExtractionError, extract_text
from resumeai.context_files.models import ContextFileKind
from resumeai.local_projects.scanner import ScanError, scan_project

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


@router.post("/context/snippet", include_in_schema=False)
def add_context_snippet(
    name: str = Form(...),
    text: str = Form(...),
    note: str = Form(""),
    tags: str = Form(""),
    store: ContextFileStore = Depends(get_context_file_store),
) -> RedirectResponse:
    """Save an inline text snippet — pasted notes, comments about other
    uploads, anything the user wants to type freely."""
    name_clean = name.strip() or "snippet"
    text_clean = text.strip()
    if not text_clean:
        return RedirectResponse("/context?error=Snippet+text+is+empty", status_code=303)
    tag_tuple = tuple(t.strip() for t in tags.split(",") if t.strip())
    store.add(
        name=name_clean,
        kind=ContextFileKind.TEXT,
        extracted_text=text_clean,
        byte_size=len(text_clean.encode()),
        tags=tag_tuple,
        note=note.strip(),
    )
    return RedirectResponse(f"/context?flash=Added+snippet+{name_clean}", status_code=303)


@router.post("/context/project", include_in_schema=False)
def add_local_project(
    path: str = Form(...),
    name: str = Form(""),
    author_email: str = Form(""),
    note: str = Form(""),
    tags: str = Form(""),
    store: ContextFileStore = Depends(get_context_file_store),
) -> RedirectResponse:
    """Scan a registered local project directory + persist the summary as a
    context file. Re-scanning is a delete + re-add today."""
    path_clean = path.strip()
    if not path_clean:
        return RedirectResponse("/context?error=Project+path+is+required", status_code=303)
    try:
        summary = scan_project(
            path_clean,
            name=name.strip() or None,
            author_email=author_email.strip() or None,
        )
    except ScanError as exc:
        return RedirectResponse(f"/context?error=Could+not+scan+project%3A+{exc}", status_code=303)
    display_name = name.strip() or path_clean.rstrip("/").rsplit("/", 1)[-1] or path_clean
    base_tags = [t.strip() for t in tags.split(",") if t.strip()]
    if "source:local_project" not in base_tags:
        base_tags.append("source:local_project")
    store.add(
        name=f"{display_name} (project scan)",
        kind=ContextFileKind.MARKDOWN,
        extracted_text=summary,
        byte_size=len(summary.encode()),
        tags=tuple(base_tags),
        note=note.strip(),
    )
    return RedirectResponse(f"/context?flash=Scanned+{display_name}", status_code=303)


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
