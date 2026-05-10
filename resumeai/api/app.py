"""FastAPI application factory.

Tests construct the app with in-memory implementations of every singleton;
production builds the SQLite stores, the HTTP-backed OAuth service, and the
``claude`` CLI subprocess LLM client by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

from resumeai import __version__
from resumeai.api.deps import AppState
from resumeai.api.routes import context as context_routes
from resumeai.api.routes import oauth as oauth_routes
from resumeai.api.routes import onboarding as onboarding_routes
from resumeai.api.routes import pages as pages_routes
from resumeai.api.routes import settings as settings_routes
from resumeai.api.routes import tailor as tailor_routes
from resumeai.api.routes import templates as templates_routes
from resumeai.auth.google_oauth import GoogleOAuthService
from resumeai.context_files.store import SqliteContextFileStore
from resumeai.docs.client import GoogleDocsClient
from resumeai.llm.client import ClaudeCliClient
from resumeai.runs.orchestrator import TailoringOrchestrator
from resumeai.runs.store import SqliteRunsStore
from resumeai.settings.store import SqliteSettingsStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from resumeai.auth.google_oauth import OAuthService
    from resumeai.context_files.store import ContextFileStore
    from resumeai.docs.client import DocsClient
    from resumeai.llm.client import LLMClient
    from resumeai.runs.store import RunsStore
    from resumeai.settings.models import GoogleCredentials
    from resumeai.settings.store import SettingsStore


def create_app(
    *,
    settings_store: SettingsStore | None = None,
    oauth_service: OAuthService | None = None,
    runs_store: RunsStore | None = None,
    llm_client: LLMClient | None = None,
    docs_client_factory: Callable[[GoogleCredentials], DocsClient] | None = None,
    orchestrator: TailoringOrchestrator | None = None,
    context_file_store: ContextFileStore | None = None,
) -> FastAPI:
    """Build a configured FastAPI app.

    Every dependency can be injected from tests; defaults are the production
    implementations. ``orchestrator`` is built from the other singletons
    when not supplied — pass it directly for tests that want full control.
    """
    settings = settings_store or SqliteSettingsStore()
    oauth = oauth_service or GoogleOAuthService()
    runs = runs_store or SqliteRunsStore()
    context_files = context_file_store or SqliteContextFileStore()
    llm = llm_client or ClaudeCliClient()
    docs_factory = docs_client_factory or GoogleDocsClient
    orch = orchestrator or TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        docs_client_factory=docs_factory,
        context_file_store=context_files,
    )

    app = FastAPI(
        title="resumeai",
        version=__version__,
        description="AI-driven resume tailoring driven by your master Google Doc.",
    )
    app.state.app_state = AppState(
        settings_store=settings,
        oauth_service=oauth,
        runs_store=runs,
        orchestrator=orch,
        context_file_store=context_files,
    )

    app.include_router(onboarding_routes.router)
    app.include_router(settings_routes.router)
    app.include_router(oauth_routes.router)
    app.include_router(tailor_routes.router)
    app.include_router(templates_routes.router)
    app.include_router(pages_routes.router)
    app.include_router(context_routes.router)

    return app
