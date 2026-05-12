"""End-to-end pipeline tests for :class:`TailoringOrchestrator`.

The orchestrator drives JD-fetch -> parse -> load-context -> tailor -> render
-> verify, all on a worker thread via ``asyncio.to_thread``. Each external
dependency (fetch_jd, parse_jd_text, load_user_context, tailor_resume,
render_tailored_resume_latex, verify_resume) is monkey-patched at the
orchestrator-module level so tests can drive the pipeline deterministically
without hitting the network, the claude CLI, or tectonic.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from tailor_core.jd.models import (
    EmploymentType,
    FetchedJD,
    JobRequirements,
    RemoteType,
    RoleType,
    Seniority,
)
from tailor_core.llm.client import FakeLLMClient

from resumeai.agent.models import TailoredBullet, TailoredResume, TailoredWorkEntry
from resumeai.context.models import Contact, UserContext
from resumeai.renderer.models import RenderResult
from resumeai.runs import orchestrator as orch_mod
from resumeai.runs.models import RunStatus, TailorRequest
from resumeai.runs.orchestrator import (
    TailoringOrchestrator,
    _generate_run_id,
)
from resumeai.runs.store import InMemoryRunsStore
from resumeai.settings.store import InMemorySettingsStore
from resumeai.verifier.models import VerificationResult, VerificationStatus
from resumeai.verifier.verifier import VerifierError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from resumeai.context_files.models import ContextFile


# -- fixtures + canned outputs ----------------------------------------------


def _requirements() -> JobRequirements:
    return JobRequirements(
        title="Software Developer",
        company="GoSource",
        location="Canberra",
        role_type=RoleType.ENGINEERING,
        seniority=Seniority.MID,
        employment_type=EmploymentType.FULL_TIME,
        remote_type=RemoteType.REMOTE,
    )


def _tailored() -> TailoredResume:
    return TailoredResume(
        name="Jonathan Duketis",
        contact=Contact(email="me@example.com"),
        work_history=(
            TailoredWorkEntry(
                company="DiUS Computing",
                title="Engineer",
                bullets=(TailoredBullet(text="Built things."),),
            ),
        ),
    )


def _render_result(run_id: str = "run") -> RenderResult:
    return RenderResult(
        doc_id=run_id,
        doc_url=f"file:///tmp/{run_id}/resume.pdf",
        pdf_size_bytes=1024,
    )


def _passed_verification() -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.PASSED,
        summary="clean",
        rationale="no issues",
    )


@pytest.fixture
def runs() -> InMemoryRunsStore:
    return InMemoryRunsStore()


@pytest.fixture
def settings() -> InMemorySettingsStore:
    return InMemorySettingsStore()


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient(default_response="{}")


@pytest.fixture
def orchestrator(
    runs: InMemoryRunsStore,
    settings: InMemorySettingsStore,
    llm: FakeLLMClient,
    tmp_path: Path,
) -> TailoringOrchestrator:
    return TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        context_root=tmp_path / "userctx",
        runs_root=tmp_path / "runs",
    )


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Monkeypatch every external dependency with deterministic stubs.

    Returns the captured call args dict so tests can assert on what was
    passed (e.g. that ``stem`` reached the renderer).
    """
    calls: dict[str, Any] = {"render_stem": None, "tailor_files": None}

    def fake_fetch_jd(url: str, **_: object) -> FetchedJD:
        return FetchedJD(
            source_url=url,
            raw_html="<html/>",
            cleaned_text="JD body for " + url,
            fetched_at=datetime(2026, 5, 11, tzinfo=UTC),
            ats="unknown",
        )

    def fake_parse_jd_text(_text: str, **_: object) -> JobRequirements:
        return _requirements()

    def fake_load_user_context(_root: Path) -> UserContext:
        return UserContext()

    def fake_tailor_resume(
        _req: JobRequirements,
        _ctx: UserContext,
        _llm: object,
        *,
        model: str | None = None,
        context_files: Sequence[ContextFile] = (),
    ) -> TailoredResume:
        calls["tailor_files"] = tuple(context_files)
        calls["tailor_model"] = model
        return _tailored()

    def fake_render(
        _tailored: TailoredResume,
        output_dir: Path,
        *,
        stem: str = "resume",
        **_: object,
    ) -> RenderResult:
        calls["render_stem"] = stem
        calls["render_dir"] = output_dir
        return _render_result(output_dir.name)

    def fake_verify(*_: object, **__: object) -> VerificationResult:
        return _passed_verification()

    monkeypatch.setattr(orch_mod, "fetch_jd", fake_fetch_jd)
    monkeypatch.setattr(orch_mod, "parse_jd_text", fake_parse_jd_text)
    monkeypatch.setattr(orch_mod, "load_user_context", fake_load_user_context)
    monkeypatch.setattr(orch_mod, "tailor_resume", fake_tailor_resume)
    monkeypatch.setattr(orch_mod, "render_tailored_resume_latex", fake_render)
    monkeypatch.setattr(orch_mod, "verify_resume", fake_verify)
    return calls


# -- run id generator -------------------------------------------------------


def test_generate_run_id_has_expected_shape() -> None:
    run_id = _generate_run_id()
    assert run_id.startswith("run_")
    # ``run_YYYYMMDDhhmmss_<token>`` -- timestamp piece is 14 digits.
    assert run_id[4:18].isdigit()


# -- create_run -------------------------------------------------------------


def test_create_run_persists_a_pending_record(
    orchestrator: TailoringOrchestrator, runs: InMemoryRunsStore
) -> None:
    run = orchestrator.create_run(TailorRequest(jd_text="paste"))
    assert run.status == RunStatus.PENDING
    assert runs.get(run.id) == run


# -- execute() happy paths --------------------------------------------------


def test_execute_drives_pipeline_to_succeeded_with_jd_url(
    orchestrator: TailoringOrchestrator,
    runs: InMemoryRunsStore,
    patched_pipeline: dict[str, Any],
) -> None:
    run = orchestrator.create_run(TailorRequest(jd_url="https://example.com/job/1"))
    finished = asyncio.run(orchestrator.execute(run.id))

    assert finished.status == RunStatus.SUCCEEDED
    assert finished.detail == "render + verification complete"
    assert finished.requirements is not None
    assert finished.tailored is not None
    assert finished.result is not None
    assert finished.verification is not None
    assert finished.error is None
    # The render step received a JD-flavoured stem and a per-run output dir.
    assert patched_pipeline["render_stem"] == "Jonathan Duketis - GoSource - Software Developer"
    assert patched_pipeline["render_dir"].name == run.id
    # Saved into the same record the test holds.
    assert runs.get(run.id) == finished


def test_execute_with_jd_text_skips_the_fetch_step(
    orchestrator: TailoringOrchestrator,
    runs: InMemoryRunsStore,
    patched_pipeline: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fetch path should never be touched when jd_text is supplied."""
    fetched = {"called": False}

    def boom_fetch(*_: object, **__: object) -> object:
        fetched["called"] = True
        raise AssertionError("fetch_jd should not be called for jd_text runs")

    monkeypatch.setattr(orch_mod, "fetch_jd", boom_fetch)

    run = orchestrator.create_run(TailorRequest(jd_text="paste this body"))
    finished = asyncio.run(orchestrator.execute(run.id))

    assert finished.status == RunStatus.SUCCEEDED
    assert fetched["called"] is False


def test_execute_passes_uploaded_context_files_to_tailor(
    runs: InMemoryRunsStore,
    settings: InMemorySettingsStore,
    llm: FakeLLMClient,
    patched_pipeline: dict[str, Any],
    tmp_path: Path,
) -> None:
    """When a ContextFileStore is wired in, its files reach the tailor call."""
    from resumeai.context_files.models import ContextFileKind  # noqa: PLC0415
    from resumeai.context_files.store import InMemoryContextFileStore  # noqa: PLC0415

    files = InMemoryContextFileStore()
    saved = files.add(
        name="note.md",
        kind=ContextFileKind.MARKDOWN,
        extracted_text="hi",
        byte_size=2,
    )
    orch = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        context_file_store=files,
        context_root=tmp_path / "userctx",
        runs_root=tmp_path / "runs",
    )

    run = orch.create_run(TailorRequest(jd_text="paste"))
    asyncio.run(orch.execute(run.id))

    forwarded = patched_pipeline["tailor_files"]
    assert forwarded is not None
    assert len(forwarded) == 1
    assert forwarded[0].id == saved.id


def test_execute_forwards_model_override_to_tailor(
    orchestrator: TailoringOrchestrator,
    patched_pipeline: dict[str, Any],
) -> None:
    """``TailorRequest.model`` flows through to ``tailor_resume``."""
    run = orchestrator.create_run(TailorRequest(jd_text="x", model="claude-sonnet-4-6"))
    asyncio.run(orchestrator.execute(run.id))
    assert patched_pipeline["tailor_model"] == "claude-sonnet-4-6"


# -- execute() error paths --------------------------------------------------


def test_execute_unknown_run_id_yields_synthetic_failed_run(
    orchestrator: TailoringOrchestrator,
) -> None:
    """When ``execute`` is called on a run that was never created we still get
    a Run model back -- one with id 'unknown' if the input was empty."""
    finished = asyncio.run(orchestrator.execute("not_a_real_run"))
    assert finished.status == RunStatus.FAILED
    assert "OrchestratorError" in (finished.error or "")
    # The synthesised fallback uses the missing id, not "unknown", when given.
    assert finished.id == "not_a_real_run"


def test_execute_empty_run_id_falls_back_to_unknown(
    orchestrator: TailoringOrchestrator,
) -> None:
    finished = asyncio.run(orchestrator.execute(""))
    assert finished.id == "unknown"
    assert finished.status == RunStatus.FAILED


def test_execute_marks_run_failed_when_a_pipeline_step_raises(
    orchestrator: TailoringOrchestrator,
    runs: InMemoryRunsStore,
    monkeypatch: pytest.MonkeyPatch,
    patched_pipeline: dict[str, Any],
) -> None:
    """A raise inside tailor_resume should funnel to a persisted FAILED state."""

    def boom_tailor(*_: object, **__: object) -> TailoredResume:
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(orch_mod, "tailor_resume", boom_tailor)

    run = orchestrator.create_run(TailorRequest(jd_text="x"))
    finished = asyncio.run(orchestrator.execute(run.id))

    assert finished.status == RunStatus.FAILED
    assert "agent exploded" in (finished.error or "")
    assert runs.get(run.id) == finished


# -- _verify_safely fallback ------------------------------------------------


def test_verify_safely_swallows_verifier_error_with_concerns_result(
    orchestrator: TailoringOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
    patched_pipeline: dict[str, Any],
) -> None:
    """A VerifierError shouldn't fail the run -- the pipeline records a
    synthetic CONCERNS verification and proceeds to SUCCEEDED."""

    def boom_verify(*_: object, **__: object) -> VerificationResult:
        raise VerifierError("verifier broke")

    monkeypatch.setattr(orch_mod, "verify_resume", boom_verify)

    run = orchestrator.create_run(TailorRequest(jd_text="x"))
    finished = asyncio.run(orchestrator.execute(run.id))

    assert finished.status == RunStatus.SUCCEEDED
    assert finished.verification is not None
    assert finished.verification.status is VerificationStatus.CONCERNS
    assert "VerifierError" in finished.verification.summary or any(
        "verifier broke" in issue.message for issue in finished.verification.issues
    )


def test_verify_safely_swallows_os_error_too(
    orchestrator: TailoringOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
    patched_pipeline: dict[str, Any],
) -> None:
    """``OSError`` (e.g. subprocess startup failure) also degrades gracefully."""

    def os_err_verify(*_: object, **__: object) -> VerificationResult:
        raise OSError("no such file")

    monkeypatch.setattr(orch_mod, "verify_resume", os_err_verify)

    run = orchestrator.create_run(TailorRequest(jd_text="x"))
    finished = asyncio.run(orchestrator.execute(run.id))
    assert finished.status == RunStatus.SUCCEEDED
    assert finished.verification is not None
    assert finished.verification.status is VerificationStatus.CONCERNS


# -- event_bus property -----------------------------------------------------


def test_event_bus_property_exposes_the_internal_bus(
    orchestrator: TailoringOrchestrator,
) -> None:
    from resumeai.runs.events import RunEventBus  # noqa: PLC0415

    assert isinstance(orchestrator.event_bus, RunEventBus)
