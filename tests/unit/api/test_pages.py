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
