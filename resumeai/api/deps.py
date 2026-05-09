"""FastAPI dependency providers.

Routes ask for app singletons via ``Depends(...)``; this file is the single
place that pulls them off ``request.app.state``. The app factory
(``app.py``) is the only thing that puts them there, which keeps the
wiring pinpointable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from resumeai.auth.google_oauth import OAuthService
    from resumeai.runs.orchestrator import TailoringOrchestrator
    from resumeai.runs.store import RunsStore
    from resumeai.settings.store import SettingsStore


@dataclass(frozen=True, slots=True)
class AppState:
    """Container for the app-wide singletons. Lives on ``app.state.app_state``."""

    settings_store: SettingsStore
    oauth_service: OAuthService
    runs_store: RunsStore
    orchestrator: TailoringOrchestrator


def get_app_state(request: Request) -> AppState:
    state: AppState = request.app.state.app_state
    return state


def get_settings_store(request: Request) -> SettingsStore:
    return get_app_state(request).settings_store


def get_oauth_service(request: Request) -> OAuthService:
    return get_app_state(request).oauth_service


def get_runs_store(request: Request) -> RunsStore:
    return get_app_state(request).runs_store


def get_orchestrator(request: Request) -> TailoringOrchestrator:
    return get_app_state(request).orchestrator
