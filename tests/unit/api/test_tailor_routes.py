"""HTTP-level tests for the JSON tailor + runs API."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from resumeai.agent.models import TailoredResume
from resumeai.context.models import Contact
from resumeai.runs.models import Run, RunEvent, RunStatus, TailorRequest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from resumeai.runs.orchestrator import TailoringOrchestrator
    from resumeai.runs.store import InMemoryRunsStore


def _pending_run(run_id: str = "run_a") -> Run:
    when = datetime(2026, 5, 11, tzinfo=UTC)
    return Run(
        id=run_id,
        request=TailorRequest(jd_text="paste"),
        status=RunStatus.PENDING,
        created_at=when,
        updated_at=when,
    )


def _succeeded_run(run_id: str = "run_a") -> Run:
    when = datetime(2026, 5, 11, tzinfo=UTC)
    return Run(
        id=run_id,
        request=TailorRequest(jd_text="paste"),
        status=RunStatus.SUCCEEDED,
        created_at=when,
        updated_at=when,
        tailored=TailoredResume(name="Jonathan Duketis", contact=Contact(email="x@y.co")),
    )


# -- POST /api/tailor -------------------------------------------------------


def test_post_api_tailor_returns_run_id_and_schedules_execute(
    client: TestClient,
    runs: InMemoryRunsStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POST /api/tailor`` creates a run, schedules ``execute``, returns id."""
    executed: list[str] = []

    async def fake_execute(self: object, run_id: str) -> Run:
        executed.append(run_id)
        return _succeeded_run(run_id)

    monkeypatch.setattr(
        "resumeai.runs.orchestrator.TailoringOrchestrator.execute",
        fake_execute,
    )

    response = client.post("/api/tailor", json={"jd_text": "paste body here"})
    assert response.status_code == 202
    body = response.json()
    assert "run_id" in body
    assert body["status"] == "pending"
    # The persisted run exists.
    assert runs.get(body["run_id"]) is not None
    # Background task scheduled execute(); we don't await it directly, but
    # giving the loop a tick lets it run to completion.
    asyncio.run(asyncio.sleep(0.05))
    assert executed == [body["run_id"]]


def test_post_api_tailor_400_when_request_invalid(client: TestClient) -> None:
    """Empty body fails ``TailorRequest`` validation -> 422 from FastAPI."""
    response = client.post("/api/tailor", json={})
    assert response.status_code == 422


# -- GET /api/runs/<id> -----------------------------------------------------


def test_get_api_run_returns_serialised_record(client: TestClient, runs: InMemoryRunsStore) -> None:
    runs.save(_succeeded_run("run_x"))
    response = client.get("/api/runs/run_x")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "run_x"
    assert body["status"] == "succeeded"
    assert body["tailored"]["name"] == "Jonathan Duketis"


def test_get_api_run_404_when_missing(client: TestClient) -> None:
    response = client.get("/api/runs/run_nope")
    assert response.status_code == 404


# -- GET /api/runs ----------------------------------------------------------


def test_get_api_runs_lists_recent(client: TestClient, runs: InMemoryRunsStore) -> None:
    runs.save(_succeeded_run("run_one"))
    runs.save(_pending_run("run_two"))
    response = client.get("/api/runs?limit=10")
    assert response.status_code == 200
    body = response.json()
    ids = {r["id"] for r in body["runs"]}
    assert ids == {"run_one", "run_two"}


def test_get_api_runs_defaults_to_limit_20(client: TestClient) -> None:
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert "runs" in response.json()


# -- GET /api/runs/<id>/events (SSE) ----------------------------------------


def test_get_api_run_events_404_when_run_missing(client: TestClient) -> None:
    response = client.get("/api/runs/run_nope/events")
    assert response.status_code == 404


def test_sse_event_stream_emits_state_then_returns_for_terminal_run(
    runs: InMemoryRunsStore,
    orchestrator: TailoringOrchestrator,
) -> None:
    """Subscribing to a terminal run sends the state frame and stops."""
    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415

    runs.save(_succeeded_run("run_done"))

    async def collect() -> list[dict[str, object]]:
        frames: list[dict[str, object]] = []
        async for frame in sse_event_stream("run_done", runs, orchestrator):
            frames.append(frame)
        return frames

    frames = asyncio.run(collect())
    assert len(frames) == 1
    assert frames[0]["event"] == "state"
    parsed = json.loads(str(frames[0]["data"]))
    assert parsed["id"] == "run_done"
    assert parsed["status"] == "succeeded"


def test_sse_event_stream_publishes_live_events_for_running_run(
    runs: InMemoryRunsStore,
    orchestrator: TailoringOrchestrator,
) -> None:
    """A non-terminal run keeps the stream open and forwards bus events."""
    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415

    runs.save(_pending_run("run_live"))

    async def drive() -> list[dict[str, object]]:
        frames: list[dict[str, object]] = []

        async def consume() -> None:
            async for frame in sse_event_stream("run_live", runs, orchestrator):
                frames.append(frame)
                # Two frames is enough -- initial state + first event.
                if len(frames) >= 2:
                    # Terminal event flips the run to SUCCEEDED so the
                    # generator returns naturally after this iteration.
                    pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        # Save a succeeded snapshot so the post-event state lookup finds it.
        runs.save(_succeeded_run("run_live"))
        await orchestrator.event_bus.publish(
            RunEvent(
                run_id="run_live",
                status=RunStatus.SUCCEEDED,
                detail="done",
                at=datetime(2026, 5, 11, tzinfo=UTC),
            )
        )
        await task
        return frames

    frames = asyncio.run(drive())
    # Initial state + the live event + the post-terminal state snapshot.
    assert len(frames) >= 2
    assert frames[0]["event"] == "state"
    assert any(f["event"] == "event" for f in frames)


def test_sse_event_stream_respects_disconnect_check(
    runs: InMemoryRunsStore,
    orchestrator: TailoringOrchestrator,
) -> None:
    """When the client disconnects, the generator returns mid-stream."""
    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415

    runs.save(_pending_run("run_disco"))

    async def disconnected() -> bool:
        return True

    async def drive() -> list[dict[str, object]]:
        frames: list[dict[str, object]] = []

        async def consume() -> None:
            async for frame in sse_event_stream(
                "run_disco", runs, orchestrator, disconnect_check=disconnected
            ):
                frames.append(frame)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await orchestrator.event_bus.publish(
            RunEvent(
                run_id="run_disco",
                status=RunStatus.TAILORING,
                detail="midway",
                at=datetime(2026, 5, 11, tzinfo=UTC),
            )
        )
        await task
        return frames

    frames = asyncio.run(drive())
    # Initial state frame only -- disconnected check fires before forwarding.
    assert frames[0]["event"] == "state"
    assert all(f["event"] == "state" for f in frames)


def test_get_api_run_events_endpoint_returns_sse_for_terminal_run(
    client: TestClient,
    runs: InMemoryRunsStore,
) -> None:
    """``GET /api/runs/<id>/events`` returns an EventSourceResponse stream.

    Hitting the actual HTTP endpoint exercises the EventSourceResponse
    construction line that the direct sse_event_stream tests can't cover.
    """
    runs.save(_succeeded_run("run_sse_term"))
    response = client.get("/api/runs/run_sse_term/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_sse_event_stream_when_initial_snapshot_missing(
    runs: InMemoryRunsStore,
    orchestrator: TailoringOrchestrator,
) -> None:
    """Branch 98->103: snapshot is None, generator goes straight to subscribe."""
    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415

    async def drive() -> list[dict[str, object]]:
        frames: list[dict[str, object]] = []

        async def consume() -> None:
            async for frame in sse_event_stream("run_ghost", runs, orchestrator):
                frames.append(frame)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        # Terminal event ends the loop. ``runs.get`` still returns None for
        # the unknown run, so no trailing state frame either.
        await orchestrator.event_bus.publish(
            RunEvent(
                run_id="run_ghost",
                status=RunStatus.FAILED,
                detail="boom",
                at=datetime(2026, 5, 11, tzinfo=UTC),
            )
        )
        await task
        return frames

    frames = asyncio.run(drive())
    # No initial-state frame because the run wasn't in the store. Just the
    # forwarded event frame, no trailing state frame either.
    assert all(f["event"] == "event" for f in frames)


def test_sse_event_stream_forwards_non_terminal_event_then_terminal(
    runs: InMemoryRunsStore,
    orchestrator: TailoringOrchestrator,
) -> None:
    """Branch 115->103: a non-terminal event is forwarded; loop continues."""
    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415

    runs.save(_pending_run("run_two_events"))

    async def drive() -> list[dict[str, object]]:
        frames: list[dict[str, object]] = []

        async def consume() -> None:
            async for frame in sse_event_stream("run_two_events", runs, orchestrator):
                frames.append(frame)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await orchestrator.event_bus.publish(
            RunEvent(
                run_id="run_two_events",
                status=RunStatus.TAILORING,
                detail="midway",
                at=datetime(2026, 5, 11, tzinfo=UTC),
            )
        )
        await asyncio.sleep(0)
        runs.save(_succeeded_run("run_two_events"))
        await orchestrator.event_bus.publish(
            RunEvent(
                run_id="run_two_events",
                status=RunStatus.SUCCEEDED,
                detail="done",
                at=datetime(2026, 5, 11, tzinfo=UTC),
            )
        )
        await task
        return frames

    frames = asyncio.run(drive())
    statuses = [f["event"] for f in frames]
    assert statuses.count("event") == 2  # two forwarded events
    assert statuses[0] == "state"  # initial snapshot
    assert statuses[-1] == "state"  # final snapshot after terminal


def test_sse_event_stream_exits_when_bus_closes_without_terminal_event(
    runs: InMemoryRunsStore,
    orchestrator: TailoringOrchestrator,
) -> None:
    """Branch 103->exit: subscribe loop ends because the bus was closed,
    not because a terminal event was published."""
    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415

    runs.save(_pending_run("run_closed_bus"))

    async def drive() -> list[dict[str, object]]:
        frames: list[dict[str, object]] = []

        async def consume() -> None:
            async for frame in sse_event_stream("run_closed_bus", runs, orchestrator):
                frames.append(frame)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        # Close the bus without publishing a terminal event -- the
        # subscribe async-iterator ends, the for-loop exits cleanly.
        await orchestrator.event_bus.close("run_closed_bus")
        await task
        return frames

    frames = asyncio.run(drive())
    # Only the initial state snapshot frame; no events forwarded.
    assert len(frames) == 1
    assert frames[0]["event"] == "state"


def test_running_statuses_helper_excludes_terminal_states() -> None:
    """``_running_statuses`` helper is exported for tests; covers the helper."""
    from resumeai.api.routes.pages import _running_statuses  # noqa: PLC0415

    statuses = _running_statuses()
    assert RunStatus.SUCCEEDED not in statuses
    assert RunStatus.FAILED not in statuses
    assert RunStatus.PENDING in statuses
    assert RunStatus.TAILORING in statuses
