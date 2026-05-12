"""FastAPI application factory.

Tests construct the app with in-memory implementations of every singleton;
production builds the SQLite stores and the ``claude`` CLI subprocess LLM
client by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from tailor_core.context_files.store import SqliteContextFileStore
from tailor_core.llm.client import ClaudeCliClient
from tailor_core.runs.store import SqliteRunsStore
from tailor_core.settings.store import SqliteSettingsStore

from resumeai import __version__
from resumeai.agent.models import TailoredResume
from resumeai.api.deps import AppState
from resumeai.api.routes import context as context_routes
from resumeai.api.routes import pages as pages_routes
from resumeai.api.routes import tailor as tailor_routes
from resumeai.runs.orchestrator import TailoringOrchestrator
from resumeai.settings.models import RuntimeSettings

# Single SQLite file shared by every store the app uses. Lives outside the
# repo so it survives ``docker compose down``. The lib's stores accept a
# ``db_path`` arg so consumers can colocate everything in one file even
# though tailor_core itself defaults to ``~/.tailor_core/tailor_core.db``.
_RESUMEAI_DB_PATH = Path("~/.resumeai/resumeai.db").expanduser()

if TYPE_CHECKING:
    from tailor_core.context_files.store import ContextFileStore
    from tailor_core.llm.client import LLMClient
    from tailor_core.runs.store import RunsStore
    from tailor_core.settings.store import SettingsStore


def create_app(
    *,
    settings_store: SettingsStore[RuntimeSettings] | None = None,
    runs_store: RunsStore[TailoredResume] | None = None,
    llm_client: LLMClient | None = None,
    orchestrator: TailoringOrchestrator | None = None,
    context_file_store: ContextFileStore | None = None,
) -> FastAPI:
    """Build a configured FastAPI app.

    Every dependency can be injected from tests; defaults are the production
    implementations. ``orchestrator`` is built from the other singletons
    when not supplied -- pass it directly for tests that want full control.
    """
    settings = settings_store or SqliteSettingsStore(
        settings_cls=RuntimeSettings, db_path=_RESUMEAI_DB_PATH
    )
    runs = runs_store or SqliteRunsStore(tailored_cls=TailoredResume, db_path=_RESUMEAI_DB_PATH)
    context_files = context_file_store or SqliteContextFileStore(db_path=_RESUMEAI_DB_PATH)
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
