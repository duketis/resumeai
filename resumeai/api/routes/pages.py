"""Server-rendered tailoring + runs pages.

Mirrors the JSON API surface in :mod:`~resumeai.api.routes.tailor` but
through Jinja2 forms + meta-refresh polling, matching the Phase 1
Settings page pattern.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from resumeai.api.deps import get_orchestrator, get_runs_store
from resumeai.api.templating import templates
from resumeai.runs.models import RunStatus, TailorRequest

if TYPE_CHECKING:
    from resumeai.runs.orchestrator import TailoringOrchestrator
    from resumeai.runs.store import RunsStore


router = APIRouter()

_BACKGROUND_TASKS: set[asyncio.Task[object]] = set()


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
    template_doc_id: str = Form(""),
    orchestrator: TailoringOrchestrator = Depends(get_orchestrator),
) -> RedirectResponse:
    try:
        body = TailorRequest(
            jd_url=jd_url.strip() or None,
            jd_text=jd_text.strip() or None,
            template_doc_id=template_doc_id.strip() or None,
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
        {"runs": runs.list_recent()},
    )


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


# -- helpers exported for tests --------------------------------------------


def _running_statuses() -> tuple[RunStatus, ...]:
    return tuple(s for s in RunStatus if not s.is_terminal)


__all__ = ["_running_statuses", "router"]
