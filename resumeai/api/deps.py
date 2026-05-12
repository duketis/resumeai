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
    from tailor_core.context_files.store import ContextFileStore
    from tailor_core.runs.store import RunsStore
    from tailor_core.settings.store import SettingsStore

    from resumeai.agent.models import TailoredResume
    from resumeai.runs.orchestrator import TailoringOrchestrator
    from resumeai.settings.models import RuntimeSettings


@dataclass(frozen=True, slots=True)
class AppState:
    """Container for the app-wide singletons. Lives on ``app.state.app_state``."""

    settings_store: SettingsStore[RuntimeSettings]
    runs_store: RunsStore[TailoredResume]
    orchestrator: TailoringOrchestrator
    context_file_store: ContextFileStore


def get_app_state(request: Request) -> AppState:
    state: AppState = request.app.state.app_state
    return state


def get_settings_store(request: Request) -> SettingsStore[RuntimeSettings]:
    return get_app_state(request).settings_store


def get_runs_store(request: Request) -> RunsStore[TailoredResume]:
    return get_app_state(request).runs_store


def get_orchestrator(request: Request) -> TailoringOrchestrator:
    return get_app_state(request).orchestrator


def get_context_file_store(request: Request) -> ContextFileStore:
    return get_app_state(request).context_file_store
