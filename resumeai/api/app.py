"""FastAPI application factory.

Tests construct the app with in-memory implementations of the store and
oauth service; production constructs it with the SQLite store and the live
Google OAuth client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

from resumeai import __version__
from resumeai.api.deps import AppState
from resumeai.api.routes import oauth as oauth_routes
from resumeai.api.routes import onboarding as onboarding_routes
from resumeai.api.routes import settings as settings_routes
from resumeai.auth.google_oauth import GoogleOAuthService
from resumeai.settings.store import SqliteSettingsStore

if TYPE_CHECKING:
    from resumeai.auth.google_oauth import OAuthService
    from resumeai.settings.store import SettingsStore


def create_app(
    *,
    settings_store: SettingsStore | None = None,
    oauth_service: OAuthService | None = None,
) -> FastAPI:
    """Build a configured FastAPI app.

    Pass concrete ``settings_store`` / ``oauth_service`` from tests; defaults
    are the production SQLite store and the live HTTP-backed OAuth service.
    """
    app = FastAPI(
        title="resumeai",
        version=__version__,
        description="AI-driven resume tailoring driven by your master Google Doc.",
    )
    app.state.app_state = AppState(
        settings_store=settings_store or SqliteSettingsStore(),
        oauth_service=oauth_service or GoogleOAuthService(),
    )

    app.include_router(onboarding_routes.router)
    app.include_router(settings_routes.router)
    app.include_router(oauth_routes.router)

    return app
