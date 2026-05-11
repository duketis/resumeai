"""TailoringOrchestrator — drives the JD → resume pipeline asynchronously.

A single orchestrator instance is held on ``app.state.app_state`` and serves
every concurrent run. The orchestrator:

- Persists run records via ``RunsStore`` so reconnecting clients see history.
- Publishes ``RunEvent`` to the in-memory ``RunEventBus`` so SSE subscribers
  see live progress.
- Runs the pipeline's blocking steps (HTTP fetch, ``claude`` subprocess,
  Google API calls) on a worker thread via ``asyncio.to_thread`` so the
  event loop stays responsive.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx

from resumeai.agent.tailor import tailor_resume
from resumeai.context.loader import load_user_context
from resumeai.jd.fetcher import fetch_jd
from resumeai.jd.parser import parse_jd_text
from resumeai.renderer.latex_renderer import render_tailored_resume_latex
from resumeai.runs.events import RunEventBus
from resumeai.runs.models import Run, RunEvent, RunStatus, TailorRequest
from resumeai.runs.store import update_run
from resumeai.verifier.verifier import (
    VerifierError,
    fallback_concerns_result,
    verify_resume,
)

if TYPE_CHECKING:
    from resumeai.agent.models import TailoredResume
    from resumeai.context_files.store import ContextFileStore
    from resumeai.jd.models import JobRequirements
    from resumeai.llm.client import LLMClient
    from resumeai.runs.store import RunsStore
    from resumeai.settings.store import SettingsStore
    from resumeai.verifier.models import VerificationResult


DEFAULT_CONTEXT_ROOT = Path("UserContext")
DEFAULT_RUNS_ROOT = Path("runs")


class OrchestratorError(RuntimeError):
    """Raised when the orchestrator can't even start the pipeline."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TailoringOrchestrator:
    """Pipeline driver: kicks off runs + streams their events."""

    def __init__(
        self,
        *,
        runs_store: RunsStore,
        settings_store: SettingsStore,
        llm_client: LLMClient,
        event_bus: RunEventBus | None = None,
        context_root: Path = DEFAULT_CONTEXT_ROOT,
        http_client: httpx.Client | None = None,
        context_file_store: ContextFileStore | None = None,
        runs_root: Path = DEFAULT_RUNS_ROOT,
    ) -> None:
        self._runs = runs_store
        self._settings = settings_store
        self._llm = llm_client
        self._event_bus = event_bus or RunEventBus()
        self._context_root = context_root
        self._http = http_client
        self._context_files = context_file_store
        self._runs_root = runs_root

    @property
    def event_bus(self) -> RunEventBus:
        return self._event_bus

    def create_run(self, request: TailorRequest) -> Run:
        """Persist a new pending run and return it."""
        now = _utcnow()
        run = Run(
            id=_generate_run_id(),
            request=request,
            status=RunStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        self._runs.save(run)
        return run

    async def execute(self, run_id: str) -> Run:
        """Run the pipeline for ``run_id``. Idempotent end-state on failure.

        This is awaited by the API's BackgroundTasks shim so tests can await
        the same coroutine and assert the resulting state.
        """
        try:
            return await self._execute_inner(run_id)
        except Exception as exc:  # noqa: BLE001 — funnel everything to FAILED
            return await self._mark_failed(run_id, exc)

    async def _mark_failed(self, run_id: str, exc: Exception) -> Run:
        """Persist FAILED state if the run exists, then close the event bus.

        If the run was never created (eg. ``execute`` called with a bad id),
        synthesise a one-off FAILED Run so callers always get a model back.
        """
        message = f"{type(exc).__name__}: {exc}"
        try:
            failed = update_run(
                self._runs,
                run_id,
                status=RunStatus.FAILED,
                detail="pipeline failed",
                error=message,
            )
        except KeyError:
            now = _utcnow()
            failed = Run(
                id=run_id or "unknown",
                request=TailorRequest(jd_text="(unknown — run was never created)"),
                status=RunStatus.FAILED,
                created_at=now,
                updated_at=now,
                detail="pipeline failed",
                error=message,
            )
        await self._publish(failed.id, RunStatus.FAILED, str(exc))
        await self._event_bus.close(run_id)
        return failed

    async def _execute_inner(self, run_id: str) -> Run:
        import asyncio  # noqa: PLC0415

        current = self._runs.get(run_id)
        if current is None:
            raise OrchestratorError(f"unknown run {run_id!r}")

        request = current.request

        # Step 1: JD text (fetch URL or use supplied text).
        if request.jd_url:
            await self._step(run_id, RunStatus.FETCHING_JD, f"fetching {request.jd_url}")
            fetched = await asyncio.to_thread(fetch_jd, request.jd_url, http_client=self._http)
            jd_text = fetched.cleaned_text
            source_url = fetched.source_url
        else:
            # TailorRequest's validator guarantees jd_text is set when
            # jd_url is not, so the cast is safe.
            jd_text = cast("str", request.jd_text)
            source_url = None

        # Step 2: parse JD into JobRequirements.
        await self._step(run_id, RunStatus.PARSING_JD, "extracting structured requirements")
        requirements = await asyncio.to_thread(
            parse_jd_text, jd_text, llm=self._llm, source_url=source_url
        )
        update_run(self._runs, run_id, requirements=requirements)

        # Step 3: load the user's context tree + uploaded files. The LaTeX
        # template is local and rebuilt every run, so no master-template
        # snapshot is needed -- the agent gets its content from the user
        # context tree (resume.yaml, work_history/*.md, git_audit/*.md, etc.).
        await self._step(run_id, RunStatus.LOADING_CONTEXT, "loading user context")
        context = await asyncio.to_thread(load_user_context, self._context_root)
        context_files_list = list(self._context_files.list_all()) if self._context_files else []

        # Step 4: tailor.
        await self._step(run_id, RunStatus.TAILORING, "running tailoring agent")
        tailored = await asyncio.to_thread(
            tailor_resume,
            requirements,
            context,
            self._llm,
            model=request.model,
            context_files=tuple(context_files_list),
        )
        update_run(self._runs, run_id, tailored=tailored)

        # Step 5: render via LaTeX -> tectonic -> PDF.
        await self._step(run_id, RunStatus.RENDERING, "rendering tailored PDF")
        output_dir = self._runs_root / run_id
        result = await asyncio.to_thread(
            render_tailored_resume_latex, tailored, output_dir
        )
        update_run(self._runs, run_id, result=result)

        # Step 6: verify (QC pass — never blocks a SUCCEEDED run, surfaces
        # findings on the run detail page so the user can review).
        await self._step(run_id, RunStatus.VERIFYING, "running QC verification")
        verification = await asyncio.to_thread(self._verify_safely, requirements, tailored)
        update_run(self._runs, run_id, verification=verification)

        # Done.
        finished = update_run(
            self._runs,
            run_id,
            status=RunStatus.SUCCEEDED,
            detail="render + verification complete",
        )
        await self._publish(run_id, RunStatus.SUCCEEDED, "render + verification complete")
        await self._event_bus.close(run_id)
        return finished

    def _verify_safely(
        self, requirements: JobRequirements, tailored: TailoredResume
    ) -> VerificationResult:
        """Run verification but never let it block the run — fall back to a
        synthetic ``CONCERNS`` result on any failure."""
        try:
            return verify_resume(requirements, tailored, self._llm)
        except (VerifierError, OSError, RuntimeError) as exc:
            return fallback_concerns_result(f"{type(exc).__name__}: {exc}")

    async def _step(self, run_id: str, status: RunStatus, detail: str) -> None:
        update_run(self._runs, run_id, status=status, detail=detail)
        await self._publish(run_id, status, detail)

    async def _publish(self, run_id: str, status: RunStatus, detail: str) -> None:
        await self._event_bus.publish(
            RunEvent(run_id=run_id, status=status, detail=detail, at=_utcnow())
        )



def _generate_run_id() -> str:
    """Short, URL-safe, sortable-ish run identifier."""
    stamp = _utcnow().strftime("%Y%m%d%H%M%S")
    return f"run_{stamp}_{secrets.token_urlsafe(6)}"


__all__ = [
    "DEFAULT_CONTEXT_ROOT",
    "OrchestratorError",
    "TailoringOrchestrator",
]
