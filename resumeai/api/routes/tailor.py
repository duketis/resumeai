"""Tailoring API: kick off runs, fetch their state, stream live events."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse
from tailor_core.runs.models import Run, TailorRequest

from resumeai.api.deps import get_orchestrator, get_runs_store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from tailor_core.runs.store import RunsStore

    from resumeai.agent.models import TailoredResume
    from resumeai.runs.orchestrator import TailoringOrchestrator


router = APIRouter(prefix="/api")

_BACKGROUND_TASKS: set[asyncio.Task[object]] = set()


@router.post("/tailor", status_code=202)
async def kickoff_tailor(
    request: TailorRequest,
    runs: RunsStore[TailoredResume] = Depends(get_runs_store),
    orchestrator: TailoringOrchestrator = Depends(get_orchestrator),
) -> dict[str, str]:
    """Create a new run + kick off the pipeline. Returns the run id."""
    del runs
    run = orchestrator.create_run(request)
    # The orchestrator funnels every error path back into a FAILED run
    # state, so a swallowed exception here can't lose state. We hold a
    # reference to the task so the GC doesn't reap the coroutine mid-run.
    task: asyncio.Task[object] = asyncio.create_task(orchestrator.execute(run.id))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return {"run_id": run.id, "status": run.status.value}


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    runs: RunsStore[TailoredResume] = Depends(get_runs_store),
) -> dict[str, Any]:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return _serialise_run(run)


@router.get("/runs")
def list_runs(
    limit: int = 20,
    runs: RunsStore[TailoredResume] = Depends(get_runs_store),
) -> dict[str, Any]:
    return {"runs": [_serialise_run(run) for run in runs.list_recent(limit=limit)]}


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    request: Request,
    runs: RunsStore[TailoredResume] = Depends(get_runs_store),
    orchestrator: TailoringOrchestrator = Depends(get_orchestrator),
) -> EventSourceResponse:
    """SSE stream for live run events. Emits the current state on connect,
    then every subsequent ``RunEvent`` until the run terminates or the
    client disconnects.
    """
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")

    return EventSourceResponse(
        sse_event_stream(run_id, runs, orchestrator, disconnect_check=request.is_disconnected)
    )


async def sse_event_stream(
    run_id: str,
    runs: RunsStore[TailoredResume],
    orchestrator: TailoringOrchestrator,
    *,
    disconnect_check: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield SSE-shaped event/state frames for a run.

    Extracted from the route so unit tests can drive it without a live
    HTTP transport. ``disconnect_check`` lets the route plumb in
    ``Request.is_disconnected``; tests pass ``None``.
    """
    snapshot = runs.get(run_id)
    if snapshot is not None:
        yield _sse_payload("state", _serialise_run(snapshot))
        if snapshot.status.is_terminal:
            return

    async for event in orchestrator.event_bus.subscribe(run_id):
        if disconnect_check is not None and await disconnect_check():
            return
        yield _sse_payload(
            "event",
            {
                "run_id": event.run_id,
                "status": event.status.value,
                "detail": event.detail,
                "at": event.at.isoformat(),
            },
        )
        if event.status.is_terminal:
            final = runs.get(run_id)
            if final is not None:
                yield _sse_payload("state", _serialise_run(final))
            return


# -- helpers -----------------------------------------------------------------


def _serialise_run(run: Run[TailoredResume]) -> dict[str, Any]:
    """Pydantic model_dump_json round-trip → dict (handles datetime/enum)."""
    parsed: dict[str, Any] = json.loads(run.model_dump_json())
    return parsed


def _sse_payload(event_name: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"event": event_name, "data": json.dumps(data)}
