"""App-level smoke: factory wiring, healthz, and root redirects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from resumeai.api.app import create_app

if TYPE_CHECKING:
    from resumeai.settings.store import InMemorySettingsStore
    from tests.unit.api.conftest import FakeOAuthService


def test_root_redirects_to_onboarding_when_oauth_client_unset(
    client: TestClient,
) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/onboarding"


def test_root_redirects_to_settings_when_oauth_client_present(
    store: InMemorySettingsStore, client: TestClient, oauth_client_json: str
) -> None:
    # Save the client first via the API.
    client.post(
        "/settings/oauth-client",
        data={"client_json": oauth_client_json},
        follow_redirects=False,
    )

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/settings"


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_factory_uses_supplied_singletons(
    store: InMemorySettingsStore, oauth_service: FakeOAuthService
) -> None:
    app = create_app(settings_store=store, oauth_service=oauth_service)

    state = app.state.app_state
    assert state.settings_store is store
    assert state.oauth_service is oauth_service


def test_factory_falls_back_to_real_implementations_when_omitted(
    tmp_path: object,
    monkeypatch: object,
) -> None:
    """Production defaults are wired: settings + oauth + runs + LLM + orchestrator."""
    import resumeai.runs.store as runs_store_module  # noqa: PLC0415
    import resumeai.settings.store as store_module  # noqa: PLC0415
    from resumeai.auth.google_oauth import GoogleOAuthService  # noqa: PLC0415
    from resumeai.llm.client import ClaudeCliClient  # noqa: PLC0415
    from resumeai.runs.orchestrator import TailoringOrchestrator  # noqa: PLC0415
    from resumeai.runs.store import SqliteRunsStore  # noqa: PLC0415
    from resumeai.settings.store import SqliteSettingsStore  # noqa: PLC0415

    # Point both SQLite stores at tmp files so the user's real DB is untouched.
    monkeypatch.setattr(  # type: ignore[attr-defined]
        store_module,
        "DEFAULT_DB_PATH",
        tmp_path / "settings.db",  # type: ignore[operator]
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        runs_store_module,
        "DEFAULT_DB_PATH",
        tmp_path / "runs.db",  # type: ignore[operator]
    )

    app = create_app()

    state = app.state.app_state
    assert isinstance(state.settings_store, SqliteSettingsStore)
    assert isinstance(state.oauth_service, GoogleOAuthService)
    assert isinstance(state.runs_store, SqliteRunsStore)
    assert isinstance(state.orchestrator, TailoringOrchestrator)
    # The orchestrator should default to the production LLM client too.
    assert isinstance(state.orchestrator._llm, ClaudeCliClient)
