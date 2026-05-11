"""HTTP-level tests for the server-rendered pages router."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from resumeai.agent.models import TailoredResume
from resumeai.context.models import Contact
from resumeai.renderer.models import RenderResult
from resumeai.runs.models import Run, RunStatus, TailorRequest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from resumeai.runs.store import InMemoryRunsStore


def _make_succeeded_run(*, run_id: str, pdf_path: Path) -> Run:
    when = datetime(2026, 5, 11, tzinfo=UTC)
    return Run(
        id=run_id,
        request=TailorRequest(jd_text="paste"),
        status=RunStatus.SUCCEEDED,
        created_at=when,
        updated_at=when,
        tailored=TailoredResume(name="Jonathan Duketis", contact=Contact(email="x@y.co")),
        result=RenderResult(
            doc_id=run_id,
            doc_url=pdf_path.as_uri(),
            pdf_size_bytes=pdf_path.stat().st_size,
        ),
    )


def test_run_pdf_streams_pdf_inline(
    client: TestClient,
    runs: InMemoryRunsStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GET /runs/<id>/pdf`` returns the rendered PDF with inline disposition."""
    run_id = "run_pdf_ok"
    runs_root = tmp_path / "runs" / run_id
    runs_root.mkdir(parents=True)
    pdf_path = runs_root / "Jonathan Duketis - ACME - Engineer.pdf"
    pdf_path.write_bytes(b"%PDF-fake")

    # The route resolves ``runs/<id>/*.pdf`` relative to the process cwd.
    monkeypatch.chdir(tmp_path)
    runs.save(_make_succeeded_run(run_id=run_id, pdf_path=pdf_path))

    response = client.get(f"/runs/{run_id}/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("inline")
    # Starlette URL-encodes spaces and dashes in the RFC 5987 filename* form;
    # substring match against the encoded filename is the stable assertion.
    assert "ACME" in disposition and "Engineer" in disposition
    assert response.content == b"%PDF-fake"


def test_run_pdf_404_for_unknown_run(client: TestClient) -> None:
    response = client.get("/runs/run_nope/pdf")
    assert response.status_code == 404


def test_run_pdf_404_when_run_succeeded_but_pdf_missing(
    client: TestClient,
    runs: InMemoryRunsStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run record exists with a ``result`` but no on-disk pdf -> 404."""
    monkeypatch.chdir(tmp_path)
    run_id = "run_no_disk"
    # Persist a succeeded run but DON'T create the runs/<id>/*.pdf file.
    fake_pdf = tmp_path / "ghost.pdf"
    fake_pdf.write_bytes(b"%PDF-")
    runs.save(_make_succeeded_run(run_id=run_id, pdf_path=fake_pdf))

    response = client.get(f"/runs/{run_id}/pdf")
    assert response.status_code == 404


def test_run_pdf_404_when_result_field_is_none(
    client: TestClient,
    runs: InMemoryRunsStore,
) -> None:
    """Run record exists but the pipeline never reached the render step."""
    when = datetime(2026, 5, 11, tzinfo=UTC)
    runs.save(
        Run(
            id="run_no_result",
            request=TailorRequest(jd_text="x"),
            status=RunStatus.PENDING,
            created_at=when,
            updated_at=when,
        )
    )
    response = client.get("/runs/run_no_result/pdf")
    assert response.status_code == 404


def test_clear_runs_route_wipes_store_and_redirects(
    client: TestClient,
    runs: InMemoryRunsStore,
) -> None:
    """``POST /runs/clear`` deletes every run and redirects to /runs with flash."""
    when = datetime(2026, 5, 11, tzinfo=UTC)
    runs.save(
        Run(
            id="run_a",
            request=TailorRequest(jd_text="x"),
            status=RunStatus.SUCCEEDED,
            created_at=when,
            updated_at=when,
        )
    )
    runs.save(
        Run(
            id="run_b",
            request=TailorRequest(jd_text="x"),
            status=RunStatus.SUCCEEDED,
            created_at=when,
            updated_at=when,
        )
    )

    response = client.post("/runs/clear", follow_redirects=False)
    assert response.status_code == 303
    assert "/runs?flash=Cleared" in response.headers["location"]
    assert "2" in response.headers["location"]  # count was 2
    assert runs.list_recent() == []


def test_clear_runs_route_works_on_empty_store(
    client: TestClient,
    runs: InMemoryRunsStore,
) -> None:
    response = client.post("/runs/clear", follow_redirects=False)
    assert response.status_code == 303
    assert "Cleared+0+run" in response.headers["location"]


# -- HTML pages: GET routes -------------------------------------------------


def test_tailor_page_renders_with_no_query_params(client: TestClient) -> None:
    response = client.get("/tailor")
    assert response.status_code == 200
    assert "<form" in response.text.lower() or "jd_url" in response.text


def test_root_url_redirects_to_tailor(client: TestClient) -> None:
    """Bare ``/`` lands on the Tailor page rather than a JSON 404."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/tailor"


def test_settings_page_renders(client: TestClient) -> None:
    """``/settings`` returns the static info page about template editing."""
    response = client.get("/settings")
    assert response.status_code == 200
    assert "Settings" in response.text
    assert "default.tex.j2" in response.text


def test_runs_page_renders_with_flash_query_param(client: TestClient) -> None:
    response = client.get("/runs?flash=Hello+world")
    assert response.status_code == 200
    assert "Hello world" in response.text


def test_run_detail_page_renders_for_known_run(client: TestClient, runs: InMemoryRunsStore) -> None:
    when = datetime(2026, 5, 11, tzinfo=UTC)
    runs.save(
        Run(
            id="run_detail_ok",
            request=TailorRequest(jd_text="x"),
            status=RunStatus.SUCCEEDED,
            created_at=when,
            updated_at=when,
        )
    )
    response = client.get("/runs/run_detail_ok")
    assert response.status_code == 200
    assert "run_detail_ok" in response.text


def test_run_detail_page_404_for_unknown_run(client: TestClient) -> None:
    response = client.get("/runs/run_does_not_exist")
    assert response.status_code == 404


# -- HTML form-submission: POST /tailor ------------------------------------


def test_post_tailor_form_creates_run_and_redirects_to_run_detail(
    client: TestClient,
    runs: InMemoryRunsStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitting the Tailor form schedules execute and 303s to /runs/<id>."""

    async def fake_execute(self: object, run_id: str) -> Run:
        when = datetime(2026, 5, 11, tzinfo=UTC)
        return Run(
            id=run_id,
            request=TailorRequest(jd_text="x"),
            status=RunStatus.SUCCEEDED,
            created_at=when,
            updated_at=when,
        )

    monkeypatch.setattr(
        "resumeai.runs.orchestrator.TailoringOrchestrator.execute",
        fake_execute,
    )

    response = client.post(
        "/tailor",
        data={"jd_text": "paste body"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/runs/run_")
    run_id = location.removeprefix("/runs/")
    assert runs.get(run_id) is not None


def test_post_tailor_form_redirects_with_error_when_request_invalid(
    client: TestClient,
) -> None:
    """Empty form (no jd_url + no jd_text) fails validation -> 303 to /tailor."""
    response = client.post("/tailor", data={}, follow_redirects=False)
    assert response.status_code == 303
    assert "/tailor?error=" in response.headers["location"]


# -- get_settings_store dep helper -----------------------------------------


def test_get_settings_store_returns_app_singleton(client: TestClient) -> None:
    """``get_settings_store`` is exercised by reaching into the app directly."""
    from fastapi import Request  # noqa: PLC0415

    from resumeai.api.deps import get_settings_store  # noqa: PLC0415
    from resumeai.settings.store import InMemorySettingsStore  # noqa: PLC0415

    # Pull the wired-up app from the TestClient and synthesise a Request.
    app = client.app
    scope = {"type": "http", "app": app}
    request = Request(scope)
    store = get_settings_store(request)
    assert isinstance(store, InMemorySettingsStore)
