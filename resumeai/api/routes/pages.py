"""Server-rendered tailoring + runs pages.

Mirrors the JSON API surface in :mod:`~resumeai.api.routes.tailor` but
through Jinja2 forms + meta-refresh polling, matching the Phase 1
Settings page pattern.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from resumeai.api.deps import get_orchestrator, get_runs_store
from resumeai.api.templating import templates
from resumeai.runs.models import RunStatus, TailorRequest

if TYPE_CHECKING:
    from resumeai.runs.orchestrator import TailoringOrchestrator
    from resumeai.runs.store import RunsStore


router = APIRouter()

_BACKGROUND_TASKS: set[asyncio.Task[object]] = set()


@router.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    """Bare ``/`` lands on the Tailor page rather than a 404."""
    return RedirectResponse("/tailor", status_code=307)


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    """Static info page about how to change runtime settings (template)."""
    return templates.TemplateResponse(request, "settings.html", {})


@router.get("/tailor", response_class=HTMLResponse)
def tailor_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "tailor.html",
        {
            "flash": request.query_params.get("flash"),
            "error": request.query_params.get("error"),
        },
    )


@router.post("/tailor", include_in_schema=False)
async def submit_tailor(
    jd_url: str = Form(""),
    jd_text: str = Form(""),
    orchestrator: TailoringOrchestrator = Depends(get_orchestrator),
) -> RedirectResponse:
    try:
        body = TailorRequest(
            jd_url=jd_url.strip() or None,
            jd_text=jd_text.strip() or None,
        )
    except ValueError as exc:
        return RedirectResponse(
            f"/tailor?error=Invalid+request%3A+{exc}",
            status_code=303,
        )
    run = orchestrator.create_run(body)
    task: asyncio.Task[object] = asyncio.create_task(orchestrator.execute(run.id))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return RedirectResponse(f"/runs/{run.id}", status_code=303)


@router.get("/runs", response_class=HTMLResponse)
def runs_page(
    request: Request,
    runs: RunsStore = Depends(get_runs_store),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "runs.html",
        {
            "runs": runs.list_recent(),
            "flash": request.query_params.get("flash"),
        },
    )


@router.post("/runs/clear", include_in_schema=False)
def clear_runs(
    runs: RunsStore = Depends(get_runs_store),
) -> RedirectResponse:
    """Wipe every run record from the store. Per-run PDFs on disk stay."""
    deleted = runs.clear()
    return RedirectResponse(f"/runs?flash=Cleared+{deleted}+run(s)", status_code=303)


@router.post("/runs/{run_id}/rerun", include_in_schema=False)
async def rerun(
    run_id: str,
    runs: RunsStore = Depends(get_runs_store),
    orchestrator: TailoringOrchestrator = Depends(get_orchestrator),
) -> RedirectResponse:
    """Re-execute the same run id against the same ``TailorRequest``.

    Mutates the existing record in place rather than cloning -- "re-run
    THIS run", not "make a copy". State (status / detail / error /
    requirements / tailored / result / verification) is reset so the
    UI shows a fresh in-flight pipeline instead of stale fields.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    original = runs.get(run_id)
    if original is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    reset = original.model_copy(
        update={
            "status": RunStatus.PENDING,
            "detail": "",
            "error": None,
            "requirements": None,
            "tailored": None,
            "result": None,
            "verification": None,
            "updated_at": datetime.now(UTC),
        }
    )
    runs.save(reset)
    task: asyncio.Task[object] = asyncio.create_task(orchestrator.execute(run_id))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail_page(
    run_id: str,
    request: Request,
    runs: RunsStore = Depends(get_runs_store),
) -> HTMLResponse:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "run": run,
            "auto_refresh": not run.status.is_terminal,
        },
    )


@router.get("/runs/{run_id}/pdf")
def run_pdf(
    run_id: str,
    runs: RunsStore = Depends(get_runs_store),
) -> FileResponse:
    """Stream the run's rendered PDF over HTTP.

    Browsers won't navigate to ``file://`` URLs from an ``http://`` page,
    so the run-detail template + jobai integration both need an HTTP
    surface for the PDF. ``inline`` Content-Disposition lets the browser
    render the PDF in-tab rather than forcing a download.
    """
    run = runs.get(run_id)
    if run is None or run.result is None:
        raise HTTPException(status_code=404, detail=f"no PDF for run {run_id!r}")
    pdf_path = _pdf_path_for_run(run_id)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail=f"PDF missing on disk for run {run_id!r}")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        content_disposition_type="inline",
    )


def _pdf_path_for_run(run_id: str) -> Path | None:
    """Resolve the on-disk PDF for ``run_id``.

    The render step writes a labelled stem like
    ``Jonathan Duketis - <Company> - <Title>.pdf`` (since commit
    ``47fc61b``), so the filename is dynamic per run. Returning the
    most-recently-modified ``*.pdf`` under ``runs/<run_id>/`` keeps the
    route resilient to filename-stem changes.
    """
    run_dir = Path("runs") / run_id
    if not run_dir.is_dir():
        return None
    pdfs = sorted(run_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pdfs[0] if pdfs else None


# -- helpers exported for tests --------------------------------------------


def _running_statuses() -> tuple[RunStatus, ...]:
    return tuple(s for s in RunStatus if not s.is_terminal)


__all__ = ["_running_statuses", "router"]
