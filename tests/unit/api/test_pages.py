"""Tests for the server-rendered Tailor + Runs pages."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from resumeai.agent.models import TailoredBullet, TailoredResume, TailoredWorkEntry
from resumeai.context.models import Contact, Education
from resumeai.jd.models import (
    EmploymentType,
    JobRequirements,
    RemoteType,
    RoleType,
    Seniority,
)
from resumeai.renderer.models import RenderDiff, RenderResult, RenderStatus
from resumeai.runs.models import Run, RunStatus, TailorRequest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from resumeai.runs.store import InMemoryRunsStore


def _now() -> datetime:
    return datetime(2026, 5, 9, 12, 0, tzinfo=UTC)


# -- GET /tailor ------------------------------------------------------------


def test_tailor_page_renders_form(client: TestClient) -> None:
    response = client.get("/tailor")
    assert response.status_code == 200
    body = response.text
    assert 'name="jd_url"' in body
    assert 'name="jd_text"' in body
    assert 'name="template_doc_id"' in body


def test_tailor_page_renders_flash_message(client: TestClient) -> None:
    response = client.get("/tailor?flash=Hello")
    assert "Hello" in response.text


def test_tailor_page_renders_error_message(client: TestClient) -> None:
    response = client.get("/tailor?error=Boom")
    assert "Boom" in response.text


# -- POST /tailor -----------------------------------------------------------


def test_post_tailor_redirects_to_run_detail_on_success(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    response = client.post(
        "/tailor",
        data={"jd_text": "Senior Engineer JD..."},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/runs/")

    # The run was persisted before the redirect.
    run_id = response.headers["location"].split("/runs/")[1]
    assert runs.get(run_id) is not None


def test_post_tailor_accepts_url_form_field(client: TestClient, runs: InMemoryRunsStore) -> None:
    response = client.post(
        "/tailor",
        data={"jd_url": "https://example.com/jd"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    run_id = response.headers["location"].split("/runs/")[1]
    persisted = runs.get(run_id)
    assert persisted is not None
    assert persisted.request.jd_url == "https://example.com/jd"


def test_post_tailor_redirects_with_error_when_neither_jd_supplied(
    client: TestClient,
) -> None:
    response = client.post("/tailor", data={}, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/tailor?error=")


def test_post_tailor_redirects_with_error_when_both_jd_supplied(
    client: TestClient,
) -> None:
    response = client.post(
        "/tailor",
        data={"jd_url": "https://x", "jd_text": "y"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]


def test_post_tailor_threads_template_doc_id_into_request(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    response = client.post(
        "/tailor",
        data={"jd_text": "JD", "template_doc_id": "tmpl-x"},
        follow_redirects=False,
    )
    run_id = response.headers["location"].split("/runs/")[1]
    persisted = runs.get(run_id)
    assert persisted is not None
    assert persisted.request.template_doc_id == "tmpl-x"


# -- GET /runs --------------------------------------------------------------


def test_runs_page_shows_empty_state(client: TestClient) -> None:
    response = client.get("/runs")
    assert response.status_code == 200
    assert "No runs yet" in response.text


def test_runs_page_lists_recent_runs(client: TestClient, runs: InMemoryRunsStore) -> None:
    runs.save(
        Run(
            id="run_a",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.SUCCEEDED,
            created_at=_now(),
            updated_at=_now(),
            requirements=JobRequirements(
                title="Senior Engineer",
                company="Globex",
                role_type=RoleType.ENGINEERING,
                seniority=Seniority.SENIOR,
                employment_type=EmploymentType.FULL_TIME,
                remote_type=RemoteType.HYBRID,
            ),
        )
    )
    response = client.get("/runs")
    body = response.text

    assert "run_a" in body
    assert "Senior Engineer" in body
    assert "Globex" in body
    assert "succeeded" in body


def test_runs_page_shows_failed_status_styling(client: TestClient, runs: InMemoryRunsStore) -> None:
    runs.save(
        Run(
            id="run_failed",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.FAILED,
            created_at=_now(),
            updated_at=_now(),
        )
    )
    response = client.get("/runs")
    assert "failed" in response.text


def test_runs_page_shows_in_progress_status_neutral(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    runs.save(
        Run(
            id="run_running",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.TAILORING,
            created_at=_now(),
            updated_at=_now(),
        )
    )
    response = client.get("/runs")
    assert "tailoring" in response.text


def test_runs_page_omits_requirements_block_when_unparsed(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    """A pending run with no extracted requirements yet must still render."""
    runs.save(
        Run(
            id="run_pending",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.PENDING,
            created_at=_now(),
            updated_at=_now(),
        )
    )
    response = client.get("/runs")
    body = response.text
    assert "run_pending" in body
    # No "Senior" / "Engineer" leakage from the fixture above.
    assert "Senior Engineer" not in body


# -- GET /runs/{id} ---------------------------------------------------------


def test_run_detail_404_for_unknown_run(client: TestClient) -> None:
    response = client.get("/runs/never-saved")
    assert response.status_code == 404


def test_run_detail_renders_pending_run_with_auto_refresh(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    runs.save(
        Run(
            id="run_running",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.PARSING_JD,
            created_at=_now(),
            updated_at=_now(),
            detail="extracting requirements",
        )
    )
    response = client.get("/runs/run_running")
    body = response.text

    assert "run_running" in body
    assert "parsing_jd" in body
    assert "extracting requirements" in body
    assert 'http-equiv="refresh"' in body  # auto-refresh on for non-terminal


def test_run_detail_omits_auto_refresh_for_terminal_run(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    runs.save(
        Run(
            id="run_done",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.SUCCEEDED,
            created_at=_now(),
            updated_at=_now(),
        )
    )
    response = client.get("/runs/run_done")
    assert 'http-equiv="refresh"' not in response.text


def test_run_detail_renders_jd_url_when_supplied(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    runs.save(
        Run(
            id="run_url",
            request=TailorRequest(jd_url="https://example.com/jd"),
            status=RunStatus.PARSING_JD,
            created_at=_now(),
            updated_at=_now(),
        )
    )
    response = client.get("/runs/run_url")
    assert "https://example.com/jd" in response.text


def test_run_detail_renders_template_doc_id_when_supplied(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    runs.save(
        Run(
            id="run_t",
            request=TailorRequest(jd_text="t", template_doc_id="tmpl-x"),
            status=RunStatus.PENDING,
            created_at=_now(),
            updated_at=_now(),
        )
    )
    response = client.get("/runs/run_t")
    assert "tmpl-x" in response.text


def test_run_detail_renders_extracted_requirements(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    runs.save(
        Run(
            id="run_req",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.TAILORING,
            created_at=_now(),
            updated_at=_now(),
            requirements=JobRequirements(
                title="Senior Engineer",
                company="Globex",
                location="Sydney",
                role_type=RoleType.ENGINEERING,
                seniority=Seniority.SENIOR,
                employment_type=EmploymentType.FULL_TIME,
                remote_type=RemoteType.HYBRID,
                required_skills=("Python", "Postgres"),
                must_haves=("AU citizen",),
            ),
        )
    )
    response = client.get("/runs/run_req")
    body = response.text
    assert "Senior Engineer" in body
    assert "Globex" in body
    assert "Sydney" in body
    assert "Python, Postgres" in body
    assert "AU citizen" in body


def test_run_detail_renders_result_block_with_diffs(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    tailored = TailoredResume(
        name="Alex",
        contact=Contact(email="alex@example.com"),
        education=(Education(institution="U", degree="BSc"),),
        rationale="Reordered skills.",
    )
    result = RenderResult(
        doc_id="new-doc",
        doc_url="https://docs.google.com/document/d/new-doc/edit",
        pdf_bytes=b"%PDF-fake",
        diffs=(
            RenderDiff(
                kind="summary",
                heading="Summary",
                status=RenderStatus.REPLACED,
                before_chars=20,
                after_chars=80,
            ),
            RenderDiff(kind="skills", status=RenderStatus.SKIPPED_EMPTY),
        ),
    )
    runs.save(
        Run(
            id="run_done",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.SUCCEEDED,
            created_at=_now(),
            updated_at=_now(),
            tailored=tailored,
            result=result,
        )
    )
    response = client.get("/runs/run_done")
    body = response.text

    assert "https://docs.google.com/document/d/new-doc/edit" in body
    assert "Summary" in body
    assert "replaced" in body
    assert "skipped_empty" in body
    assert "Reordered skills." in body


def test_run_detail_renders_error_when_failed(client: TestClient, runs: InMemoryRunsStore) -> None:
    runs.save(
        Run(
            id="run_failed",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.FAILED,
            created_at=_now(),
            updated_at=_now(),
            detail="pipeline failed",
            error="OrchestratorError: no template",
        )
    )
    response = client.get("/runs/run_failed")
    body = response.text
    assert "OrchestratorError: no template" in body


def test_run_detail_handles_succeeded_run_with_no_tailored_block(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    """Defensive: a SUCCEEDED run with a result but no tailored payload still
    renders cleanly (shouldn't normally happen)."""
    result = RenderResult(
        doc_id="d",
        doc_url="https://docs.google.com/document/d/d/edit",
        pdf_bytes=b"",
    )
    runs.save(
        Run(
            id="run_no_tailored",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.SUCCEEDED,
            created_at=_now(),
            updated_at=_now(),
            result=result,
        )
    )
    response = client.get("/runs/run_no_tailored")
    assert response.status_code == 200


def test_run_detail_renders_work_entry_on_results(
    client: TestClient, runs: InMemoryRunsStore
) -> None:
    """Coverage: include a work entry in the tailored payload so any
    work-history rendering branch in the template fires."""
    tailored = TailoredResume(
        name="Alex",
        contact=Contact(email="a@example.com"),
        work_history=(
            TailoredWorkEntry(
                company="Acme",
                title="Engineer",
                bullets=(TailoredBullet(text="Built it."),),
            ),
        ),
    )
    result = RenderResult(
        doc_id="d",
        doc_url="https://docs.google.com/document/d/d/edit",
    )
    runs.save(
        Run(
            id="run_we",
            request=TailorRequest(jd_text="t"),
            status=RunStatus.SUCCEEDED,
            created_at=_now(),
            updated_at=_now(),
            tailored=tailored,
            result=result,
        )
    )
    response = client.get("/runs/run_we")
    assert response.status_code == 200


def test_running_statuses_helper_excludes_terminal() -> None:
    from resumeai.api.routes.pages import _running_statuses  # noqa: PLC0415

    statuses = _running_statuses()
    assert RunStatus.SUCCEEDED not in statuses
    assert RunStatus.FAILED not in statuses
    assert RunStatus.PARSING_JD in statuses
