"""FastAPI application factory.

Tests construct the app with in-memory implementations of every singleton;
production builds the SQLite stores and the ``claude`` CLI subprocess LLM
client by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

from resumeai import __version__
from resumeai.api.deps import AppState
from resumeai.api.routes import context as context_routes
from resumeai.api.routes import pages as pages_routes
from resumeai.api.routes import tailor as tailor_routes
from resumeai.context_files.store import SqliteContextFileStore
from resumeai.llm.client import ClaudeCliClient
from resumeai.runs.orchestrator import TailoringOrchestrator
from resumeai.runs.store import SqliteRunsStore
from resumeai.settings.store import SqliteSettingsStore

if TYPE_CHECKING:
    from resumeai.context_files.store import ContextFileStore
    from resumeai.llm.client import LLMClient
    from resumeai.runs.store import RunsStore
    from resumeai.settings.store import SettingsStore


def create_app(
    *,
    settings_store: SettingsStore | None = None,
    runs_store: RunsStore | None = None,
    llm_client: LLMClient | None = None,
    orchestrator: TailoringOrchestrator | None = None,
    context_file_store: ContextFileStore | None = None,
) -> FastAPI:
    """Build a configured FastAPI app.

    Every dependency can be injected from tests; defaults are the production
    implementations. ``orchestrator`` is built from the other singletons
    when not supplied -- pass it directly for tests that want full control.
    """
    settings = settings_store or SqliteSettingsStore()
    runs = runs_store or SqliteRunsStore()
    context_files = context_file_store or SqliteContextFileStore()
    llm = llm_client or ClaudeCliClient()
    orch = orchestrator or TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        context_file_store=context_files,
    )

    app = FastAPI(
        title="resumeai",
        version=__version__,
        description="AI-driven resume tailoring -- LaTeX/Tectonic rendered.",
    )
    app.state.app_state = AppState(
        settings_store=settings,
        runs_store=runs,
        orchestrator=orch,
        context_file_store=context_files,
    )

    app.include_router(tailor_routes.router)
    app.include_router(pages_routes.router)
    app.include_router(context_routes.router)

    return app
