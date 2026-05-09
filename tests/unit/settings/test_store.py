"""SettingsStore implementations — SQLite on disk + in-memory."""

from __future__ import annotations

import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from resumeai.settings.models import (
    GoogleCredentials,
    OAuthClient,
    OAuthPending,
    TemplateDoc,
)
from resumeai.settings.store import (
    InMemorySettingsStore,
    SettingsStore,
    SqliteSettingsStore,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def sqlite_store(tmp_path: Path) -> Iterator[SqliteSettingsStore]:
    store = SqliteSettingsStore(db_path=tmp_path / "test.db")
    try:
        yield store
    finally:
        store.close()


@pytest.fixture(params=["sqlite", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SettingsStore]:
    """Run every behavioural test against both implementations."""
    if request.param == "sqlite":
        s = SqliteSettingsStore(db_path=tmp_path / "test.db")
        try:
            yield s
        finally:
            s.close()
    else:
        yield InMemorySettingsStore()


def _sample_oauth_client() -> OAuthClient:
    return OAuthClient(
        client_id="cid",
        client_secret="csec",
        project_id="proj",
        redirect_uris=("http://localhost:7842/api/auth/google/callback",),
    )


def _sample_google_creds() -> GoogleCredentials:
    return GoogleCredentials(
        access_token="at",
        refresh_token="rt",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csec",
        scopes=("openid", "email"),
        expiry=datetime(2026, 5, 9, 12, tzinfo=UTC),
        user_email="x@example.com",
        user_id="42",
    )


# -- behavioural tests against both backends ----------------------------------


def test_oauth_client_round_trips(store: SettingsStore) -> None:
    assert store.get_oauth_client() is None

    client = _sample_oauth_client()
    store.set_oauth_client(client)

    assert store.get_oauth_client() == client


def test_oauth_client_overwrites_on_second_set(store: SettingsStore) -> None:
    store.set_oauth_client(_sample_oauth_client())
    new_client = _sample_oauth_client().model_copy(update={"client_secret": "newsec"})
    store.set_oauth_client(new_client)

    loaded = store.get_oauth_client()
    assert loaded is not None
    assert loaded.client_secret == "newsec"


def test_google_credentials_round_trips(store: SettingsStore) -> None:
    assert store.get_google_credentials() is None

    creds = _sample_google_creds()
    store.set_google_credentials(creds)

    assert store.get_google_credentials() == creds


def test_clear_google_credentials_removes_them(store: SettingsStore) -> None:
    store.set_google_credentials(_sample_google_creds())
    store.clear_google_credentials()

    assert store.get_google_credentials() is None


def test_clear_google_credentials_is_idempotent(store: SettingsStore) -> None:
    # Calling clear when nothing is stored must not raise.
    store.clear_google_credentials()
    assert store.get_google_credentials() is None


def test_oauth_pending_consumes_once(store: SettingsStore) -> None:
    pending = OAuthPending(
        state="abc",
        redirect_uri="http://localhost:7842/callback",
        created_at=datetime(2026, 5, 9, tzinfo=UTC),
    )
    store.store_oauth_pending(pending)

    consumed = store.consume_oauth_pending("abc")
    assert consumed == pending
    assert store.consume_oauth_pending("abc") is None


def test_oauth_pending_unknown_state_returns_none(store: SettingsStore) -> None:
    assert store.consume_oauth_pending("never-stored") is None


def test_template_round_trips(store: SettingsStore) -> None:
    assert store.get_template() is None

    template = TemplateDoc(doc_id="abc", nickname="primary")
    store.set_template(template)

    assert store.get_template() == template


def test_template_clear_removes_it(store: SettingsStore) -> None:
    store.set_template(TemplateDoc(doc_id="abc"))
    store.clear_template()
    assert store.get_template() is None


def test_template_clear_is_idempotent(store: SettingsStore) -> None:
    store.clear_template()
    assert store.get_template() is None


# -- SQLite-specific ---------------------------------------------------------


def test_sqlite_store_creates_parent_dir(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "deeper" / "resumeai.db"
    store = SqliteSettingsStore(db_path=db_path)
    try:
        assert db_path.exists()
        assert db_path.parent.is_dir()
    finally:
        store.close()


def test_sqlite_store_applies_owner_only_perms(tmp_path: Path) -> None:
    db_path = tmp_path / "resumeai.db"
    store = SqliteSettingsStore(db_path=db_path)
    try:
        mode = db_path.stat().st_mode
        # Owner-rw only; no group/other bits.
        assert mode & 0o077 == 0
        assert mode & stat.S_IRUSR
        assert mode & stat.S_IWUSR
    finally:
        store.close()


def test_sqlite_store_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "resumeai.db"
    store_a = SqliteSettingsStore(db_path=db_path)
    store_a.set_oauth_client(_sample_oauth_client())
    store_a.close()

    store_b = SqliteSettingsStore(db_path=db_path)
    try:
        loaded = store_b.get_oauth_client()
        assert loaded is not None
        assert loaded.client_id == "cid"
    finally:
        store_b.close()
