"""Tests for ``POST /api/tailor``, ``GET /api/runs/{id}``, and the SSE stream."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from resumeai.runs.models import Run, RunStatus, TailorRequest
from resumeai.settings.models import TemplateDoc

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from resumeai.runs.store import InMemoryRunsStore
    from resumeai.settings.store import InMemorySettingsStore


# -- POST /api/tailor -------------------------------------------------------


def test_post_tailor_rejects_request_with_neither_url_nor_text(
    client: TestClient,
) -> None:
    response = client.post("/api/tailor", json={})
    assert response.status_code == 422


def test_post_tailor_rejects_request_with_both_url_and_text(
    client: TestClient,
) -> None:
    response = client.post("/api/tailor", json={"jd_url": "https://x", "jd_text": "y"})
    assert response.status_code == 422


def test_post_tailor_accepts_jd_text_and_returns_run_id(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    response = client.post("/api/tailor", json={"jd_text": "Senior Engineer..."})

    assert response.status_code == 202
    payload = response.json()
    assert payload["run_id"]
    # The run is persisted before the response returns.
    persisted = runs.get(payload["run_id"])
    assert persisted is not None


def test_post_tailor_accepts_template_doc_id_override(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    response = client.post(
        "/api/tailor",
        json={"jd_text": "JD", "template_doc_id": "tmpl-x"},
    )

    payload = response.json()
    persisted = runs.get(payload["run_id"])
    assert persisted is not None
    assert persisted.request.template_doc_id == "tmpl-x"


# -- GET /api/runs/{id} -----------------------------------------------------


def test_get_run_returns_persisted_state(client: TestClient, runs: InMemoryRunsStore) -> None:
    now = datetime(2026, 5, 9, tzinfo=UTC)
    runs.save(
        Run(
            id="run_x",
            request=TailorRequest(jd_text="text"),
            status=RunStatus.TAILORING,
            created_at=now,
            updated_at=now,
            detail="thinking",
        )
    )

    response = client.get("/api/runs/run_x")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "run_x"
    assert payload["status"] == "tailoring"
    assert payload["detail"] == "thinking"


def test_get_run_returns_404_for_unknown_id(client: TestClient) -> None:
    response = client.get("/api/runs/never-saved")
    assert response.status_code == 404


# -- GET /api/runs (list) --------------------------------------------------


def test_list_runs_returns_recent_runs(
    client: TestClient, runs: InMemoryRunsStore, store: InMemorySettingsStore
) -> None:
    del store
    base = datetime(2026, 5, 9, tzinfo=UTC)
    for i in range(3):
        runs.save(
            Run(
                id=f"run_{i}",
                request=TailorRequest(jd_text="t"),
                status=RunStatus.PENDING,
                created_at=base + timedelta(minutes=i),
                updated_at=base + timedelta(minutes=i),
            )
        )

    response = client.get("/api/runs")

    payload = response.json()
    ids = [r["id"] for r in payload["runs"]]
    assert ids == ["run_2", "run_1", "run_0"]


def test_list_runs_respects_limit(client: TestClient, runs: InMemoryRunsStore) -> None:
    base = datetime(2026, 5, 9, tzinfo=UTC)
    for i in range(5):
        runs.save(
            Run(
                id=f"run_{i}",
                request=TailorRequest(jd_text="t"),
                status=RunStatus.PENDING,
                created_at=base + timedelta(minutes=i),
                updated_at=base + timedelta(minutes=i),
            )
        )

    response = client.get("/api/runs", params={"limit": 2})
    payload = response.json()
    assert len(payload["runs"]) == 2


# -- GET /api/runs/{id}/events (SSE) ---------------------------------------


def test_sse_stream_returns_404_for_unknown_run(client: TestClient) -> None:
    response = client.get("/api/runs/never-saved/events")
    assert response.status_code == 404


def test_sse_stream_emits_terminal_state_immediately_for_finished_run(
    client: TestClient,
    runs: InMemoryRunsStore,
    store: InMemorySettingsStore,
) -> None:
    """When the run already terminated, the SSE stream emits the snapshot
    and closes — no need to subscribe to the bus."""
    del store
    now = datetime(2026, 5, 9, tzinfo=UTC)
    runs.save(
        Run(
            id="run_done",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.SUCCEEDED,
            created_at=now,
            updated_at=now,
            detail="render complete",
        )
    )

    with client.stream("GET", "/api/runs/run_done/events") as response:
        assert response.status_code == 200
        body = response.read().decode()
    # Look for the snapshot frame.
    assert "succeeded" in body
    assert "run_done" in body


@pytest.mark.asyncio
async def test_sse_event_stream_emits_live_events_until_terminal(
    runs: object,
) -> None:
    """Drive ``sse_event_stream`` directly: publish events while it's iterating
    and assert they're yielded as SSE frames."""
    import asyncio  # noqa: PLC0415

    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415
    from resumeai.docs.client import FakeDocsClient  # noqa: PLC0415
    from resumeai.llm.client import FakeLLMClient  # noqa: PLC0415
    from resumeai.runs.events import RunEventBus  # noqa: PLC0415
    from resumeai.runs.models import RunEvent  # noqa: PLC0415
    from resumeai.runs.orchestrator import TailoringOrchestrator  # noqa: PLC0415
    from resumeai.settings.store import InMemorySettingsStore  # noqa: PLC0415

    bus = RunEventBus()
    orchestrator = TailoringOrchestrator(
        runs_store=runs,  # type: ignore[arg-type]
        settings_store=InMemorySettingsStore(),
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: FakeDocsClient(),
        event_bus=bus,
    )
    now = datetime(2026, 5, 9, tzinfo=UTC)
    runs.save(  # type: ignore[attr-defined]
        Run(
            id="run_live",
            request=TailorRequest(jd_text="text"),
            status=RunStatus.PARSING_JD,
            created_at=now,
            updated_at=now,
        )
    )

    received: list[dict[str, object]] = []

    async def consume() -> None:
        async for frame in sse_event_stream("run_live", runs, orchestrator):  # type: ignore[arg-type]
            received.append(frame)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    # First frame: the initial state snapshot.
    # Then we publish two events; the second is terminal.
    await bus.publish(
        RunEvent(run_id="run_live", status=RunStatus.TAILORING, detail="working", at=now)
    )
    # Move the persisted state to SUCCEEDED before publishing terminal,
    # so the route's "emit final state" path has something to emit.
    persisted = runs.get("run_live")  # type: ignore[attr-defined]
    assert persisted is not None
    runs.save(persisted.model_copy(update={"status": RunStatus.SUCCEEDED}))  # type: ignore[attr-defined]
    await bus.publish(
        RunEvent(run_id="run_live", status=RunStatus.SUCCEEDED, detail="done", at=now)
    )
    await consumer

    # Frame 1: initial state (parsing_jd snapshot).
    # Frame 2: tailoring event.
    # Frame 3: succeeded event.
    # Frame 4: final state (succeeded snapshot).
    assert len(received) == 4
    assert received[0]["event"] == "state"
    assert "parsing_jd" in received[0]["data"]  # type: ignore[operator]
    assert received[1]["event"] == "event"
    assert "tailoring" in received[1]["data"]  # type: ignore[operator]
    assert received[2]["event"] == "event"
    assert "succeeded" in received[2]["data"]  # type: ignore[operator]
    assert received[3]["event"] == "state"
    assert "succeeded" in received[3]["data"]  # type: ignore[operator]


@pytest.mark.asyncio
async def test_sse_event_stream_handles_missing_run_gracefully() -> None:
    """If the run vanishes between the route's check and the generator's
    first ``runs.get``, the generator must not crash — it just falls
    through to the subscribe path and exits when the bus closes."""
    import asyncio  # noqa: PLC0415

    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415
    from resumeai.docs.client import FakeDocsClient  # noqa: PLC0415
    from resumeai.llm.client import FakeLLMClient  # noqa: PLC0415
    from resumeai.runs.events import RunEventBus  # noqa: PLC0415
    from resumeai.runs.orchestrator import TailoringOrchestrator  # noqa: PLC0415
    from resumeai.runs.store import InMemoryRunsStore  # noqa: PLC0415
    from resumeai.settings.store import InMemorySettingsStore  # noqa: PLC0415

    bus = RunEventBus()
    runs = InMemoryRunsStore()  # deliberately empty
    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=InMemorySettingsStore(),
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: FakeDocsClient(),
        event_bus=bus,
    )

    received: list[dict[str, object]] = []

    async def consume() -> None:
        async for frame in sse_event_stream("missing", runs, orchestrator):
            received.append(frame)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    # Close immediately — no snapshot, no events. The async-for exits cleanly.
    await bus.close("missing")
    await consumer

    assert received == []


@pytest.mark.asyncio
async def test_sse_event_stream_skips_final_state_when_run_disappears(
    runs: object,
) -> None:
    """If the run is deleted between the terminal event and the final
    ``runs.get``, we just skip the final-state frame."""
    import asyncio  # noqa: PLC0415

    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415
    from resumeai.docs.client import FakeDocsClient  # noqa: PLC0415
    from resumeai.llm.client import FakeLLMClient  # noqa: PLC0415
    from resumeai.runs.events import RunEventBus  # noqa: PLC0415
    from resumeai.runs.models import RunEvent  # noqa: PLC0415
    from resumeai.runs.orchestrator import TailoringOrchestrator  # noqa: PLC0415
    from resumeai.settings.store import InMemorySettingsStore  # noqa: PLC0415

    bus = RunEventBus()
    orchestrator = TailoringOrchestrator(
        runs_store=runs,  # type: ignore[arg-type]
        settings_store=InMemorySettingsStore(),
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: FakeDocsClient(),
        event_bus=bus,
    )
    now = datetime(2026, 5, 9, tzinfo=UTC)
    runs.save(  # type: ignore[attr-defined]
        Run(
            id="run_z",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.TAILORING,
            created_at=now,
            updated_at=now,
        )
    )

    received: list[dict[str, object]] = []

    async def consume() -> None:
        async for frame in sse_event_stream("run_z", runs, orchestrator):  # type: ignore[arg-type]
            received.append(frame)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    # Delete the run, then publish a terminal event.
    runs._runs.pop("run_z")  # type: ignore[attr-defined]
    await bus.publish(RunEvent(run_id="run_z", status=RunStatus.SUCCEEDED, detail="done", at=now))
    await consumer

    # Initial state snapshot, then the terminal event — but no final state
    # frame because the run vanished.
    assert len(received) == 2
    assert received[0]["event"] == "state"
    assert received[1]["event"] == "event"


@pytest.mark.asyncio
async def test_sse_event_stream_stops_when_disconnect_check_returns_true(
    runs: object,
) -> None:
    """When the client has disconnected, the stream returns immediately."""
    import asyncio  # noqa: PLC0415

    from resumeai.api.routes.tailor import sse_event_stream  # noqa: PLC0415
    from resumeai.docs.client import FakeDocsClient  # noqa: PLC0415
    from resumeai.llm.client import FakeLLMClient  # noqa: PLC0415
    from resumeai.runs.events import RunEventBus  # noqa: PLC0415
    from resumeai.runs.models import RunEvent  # noqa: PLC0415
    from resumeai.runs.orchestrator import TailoringOrchestrator  # noqa: PLC0415
    from resumeai.settings.store import InMemorySettingsStore  # noqa: PLC0415

    bus = RunEventBus()
    orchestrator = TailoringOrchestrator(
        runs_store=runs,  # type: ignore[arg-type]
        settings_store=InMemorySettingsStore(),
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: FakeDocsClient(),
        event_bus=bus,
    )
    now = datetime(2026, 5, 9, tzinfo=UTC)
    runs.save(  # type: ignore[attr-defined]
        Run(
            id="run_disc",
            request=TailorRequest(jd_text="text"),
            status=RunStatus.TAILORING,
            created_at=now,
            updated_at=now,
        )
    )

    async def disconnected() -> bool:
        return True

    received: list[dict[str, object]] = []

    async def consume() -> None:
        async for frame in sse_event_stream(
            "run_disc",
            runs,  # type: ignore[arg-type]
            orchestrator,
            disconnect_check=disconnected,
        ):
            received.append(frame)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await bus.publish(RunEvent(run_id="run_disc", status=RunStatus.RENDERING, detail="...", at=now))
    await bus.close("run_disc")
    await consumer

    # Initial snapshot, then disconnect-check returns True before yielding
    # the live event.
    assert len(received) == 1
    assert received[0]["event"] == "state"


def test_post_tailor_run_records_failure_when_template_unset(
    client: TestClient, runs: InMemoryRunsStore, store: InMemorySettingsStore
) -> None:
    """End-to-end: POST kicks off a run, the orchestrator pipeline fires,
    fails because there's no template registered + no creds, and the run
    persists as FAILED.

    We poll the run state because the background task is fire-and-forget."""
    del store

    response = client.post("/api/tailor", json={"jd_text": "Senior Engineer..."})
    run_id = response.json()["run_id"]

    import time  # noqa: PLC0415

    # Poll until terminal — the pipeline is fast against fakes.
    for _ in range(40):
        run = runs.get(run_id)
        assert run is not None
        if run.status.is_terminal:
            break
        time.sleep(0.05)

    final = runs.get(run_id)
    assert final is not None
    assert final.status is RunStatus.FAILED
    assert final.error is not None


def test_post_tailor_run_succeeds_when_template_and_creds_present(
    client: TestClient,
    runs: InMemoryRunsStore,
    store: InMemorySettingsStore,
    llm: object,
    docs_client: object,
) -> None:
    """Wire the prerequisites and verify the run terminates as SUCCEEDED."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from resumeai.settings.models import GoogleCredentials  # noqa: PLC0415

    store.set_template(TemplateDoc(doc_id="master"))
    store.set_google_credentials(
        GoogleCredentials(
            access_token="at",
            refresh_token="rt",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="cid",
            client_secret="csec",
            scopes=("openid",),
            expiry=datetime(2026, 5, 9, tzinfo=UTC),
            user_email="alex@example.com",
            user_id="42",
        )
    )

    # The fake LLM returns a payload that satisfies BOTH the JD extractor
    # AND the tailor parser.
    union: dict[str, object] = {
        # JD extractor fields:
        "title": "Senior Engineer",
        "company": "Globex",
        "required_skills": ["Python"],
        "nice_to_have_skills": [],
        "must_haves": [],
        "employer_vocabulary": [],
        # Tailored resume fields:
        "name": "Alex",
        "headline": "Backend",
        "contact": {"email": "alex@example.com"},
        "summary": "Tailored summary.",
        "skills": ["Python"],
        "work_history": [],
        "education": [],
        "certifications": [],
        "rationale": "...",
    }
    llm._default = json.dumps(union)  # type: ignore[attr-defined]

    # Wire the docs client with a minimal template doc.
    docs_client._documents = {  # type: ignore[attr-defined]
        "fake-new-doc-id": {
            "documentId": "fake-new-doc-id",
            "title": "T",
            "revisionId": "r",
            "body": {
                "content": [
                    {
                        "startIndex": 1,
                        "endIndex": 9,
                        "paragraph": {
                            "elements": [
                                {
                                    "startIndex": 1,
                                    "endIndex": 9,
                                    "textRun": {
                                        "content": "Summary\n",
                                        "textStyle": {},
                                    },
                                }
                            ],
                            "paragraphStyle": {"namedStyleType": "HEADING_1"},
                        },
                    },
                    {
                        "startIndex": 9,
                        "endIndex": 12,
                        "paragraph": {
                            "elements": [
                                {
                                    "startIndex": 9,
                                    "endIndex": 12,
                                    "textRun": {"content": "x.\n", "textStyle": {}},
                                }
                            ],
                        },
                    },
                ]
            },
        }
    }

    response = client.post("/api/tailor", json={"jd_text": "Senior Engineer..."})
    run_id = response.json()["run_id"]

    import time  # noqa: PLC0415

    for _ in range(80):
        run = runs.get(run_id)
        assert run is not None
        if run.status.is_terminal:
            break
        time.sleep(0.05)

    final = runs.get(run_id)
    assert final is not None
    assert final.status is RunStatus.SUCCEEDED, f"got status {final.status} / error {final.error}"
    assert final.result is not None
    assert final.result.doc_id == "fake-new-doc-id"
