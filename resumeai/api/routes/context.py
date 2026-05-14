"""Context-files management — server-rendered upload page + JSON API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from tailor_core.context_files.extraction import ExtractionError, extract_text
from tailor_core.context_files.models import ContextFileKind
from tailor_core.local_projects.scanner import ScanError, scan_project

from resumeai.api.deps import get_context_file_store
from resumeai.api.templating import templates

if TYPE_CHECKING:
    from tailor_core.context_files.store import ContextFileStore


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
    # Map host-side paths (the absolute paths the user types into the
    # form / jobai forwards verbatim) to the container-side bind
    # mount. Without this, ``/Users/jonathan/Documents/personal/jobai``
    # 404s inside the container because the only mount is at
    # ``/host/personal``. The translation is configured via env so
    # operators with different mounts can adjust without code change.
    resolved_path = _translate_host_path(path_clean)
    try:
        summary = scan_project(
            resolved_path,
            name=name.strip() or None,
            author_email=author_email.strip() or None,
        )
    except ScanError as exc:
        return RedirectResponse(f"/context?error=Could+not+scan+project%3A+{exc}", status_code=303)
    # The scanner stamps the container-resolved path into the summary's
    # ``PATH:`` header. Rewrite it back to the host path the caller
    # passed in -- that's the path the user (and jobai's refresh
    # lookup) knows about. Without this, jobai's path-based dedupe
    # never finds a match and the entry can't be refreshed.
    if resolved_path != path_clean:
        summary = _rewrite_path_header(summary, resolved_path, path_clean)
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


def _rewrite_path_header(summary: str, resolved: str, original: str) -> str:
    """Swap the container-resolved path in the scanner's ``PATH:`` line
    back to the original host path the caller supplied.

    The scanner stamps the resolved path into the summary header
    because that's what ``Path(...).resolve()`` returns inside the
    container. Callers (jobai's refresh lookup) recognise their own
    host paths, not the bind-mount target -- so we rewrite the
    header to what they sent us.
    """
    needle = f"PATH: {resolved}"
    replacement = f"PATH: {original.rstrip('/')}"
    return summary.replace(needle, replacement, 1)


def _translate_host_path(path: str) -> str:
    """Map a host-absolute path to the in-container bind mount.

    Operators configure two env vars in the deploy:

    * ``RESUMEAI_HOST_ROOT`` -- the host-side prefix (e.g.
      ``/Users/jonathan/Documents/personal``).
    * ``RESUMEAI_PROJECTS_ROOT`` -- the container-side prefix (e.g.
      ``/host/personal``).

    When both are set AND the incoming path starts with the host
    prefix, swap the prefix for the container one. Anything else
    (path already container-rooted, no env vars set, partial match)
    passes through unchanged so backward compatibility holds.
    """
    host_root = os.environ.get("RESUMEAI_HOST_ROOT", "").rstrip("/")
    container_root = os.environ.get("RESUMEAI_PROJECTS_ROOT", "").rstrip("/")
    if not host_root or not container_root:
        return path
    cleaned = path.rstrip("/") if path.endswith("/") and len(path) > 1 else path
    if cleaned == host_root:
        return container_root
    prefix = host_root + "/"
    if cleaned.startswith(prefix):
        return container_root + "/" + cleaned[len(prefix) :]
    return path
