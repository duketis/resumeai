"""SqliteSettingsStore + InMemorySettingsStore behavioural tests."""

from __future__ import annotations

import stat
from pathlib import Path

from resumeai.settings.models import RuntimeSettings
from resumeai.settings.store import (
    InMemorySettingsStore,
    SqliteSettingsStore,
)


def test_sqlite_default_get_returns_default_settings(tmp_path: Path) -> None:
    """First access with no prior write returns a default ``RuntimeSettings``."""
    store = SqliteSettingsStore(db_path=tmp_path / "rt.db")
    try:
        assert store.get_runtime_settings() == RuntimeSettings()
    finally:
        store.close()


def test_sqlite_set_then_get_round_trips(tmp_path: Path) -> None:
    store = SqliteSettingsStore(db_path=tmp_path / "rt.db")
    try:
        custom = RuntimeSettings(template_name="alt.tex.j2", model="claude-sonnet-4-6")
        store.set_runtime_settings(custom)
        assert store.get_runtime_settings() == custom
    finally:
        store.close()


def test_sqlite_set_is_idempotent_via_upsert(tmp_path: Path) -> None:
    """Calling ``set`` twice updates the single row rather than failing."""
    store = SqliteSettingsStore(db_path=tmp_path / "rt.db")
    try:
        store.set_runtime_settings(RuntimeSettings(template_name="a.tex.j2"))
        store.set_runtime_settings(RuntimeSettings(template_name="b.tex.j2"))
        assert store.get_runtime_settings().template_name == "b.tex.j2"
    finally:
        store.close()


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "rt.db"
    a = SqliteSettingsStore(db_path=db_path)
    a.set_runtime_settings(RuntimeSettings(model="alt-model"))
    a.close()

    b = SqliteSettingsStore(db_path=db_path)
    try:
        assert b.get_runtime_settings().model == "alt-model"
    finally:
        b.close()


def test_sqlite_db_file_is_owner_only_perms(tmp_path: Path) -> None:
    """``0600`` permissions so other users on a shared box can't read settings."""
    db_path = tmp_path / "rt.db"
    store = SqliteSettingsStore(db_path=db_path)
    try:
        mode = db_path.stat().st_mode & 0o777
        assert mode == stat.S_IRUSR | stat.S_IWUSR
    finally:
        store.close()


def test_in_memory_starts_with_defaults() -> None:
    assert InMemorySettingsStore().get_runtime_settings() == RuntimeSettings()


def test_in_memory_set_then_get_round_trips() -> None:
    store = InMemorySettingsStore()
    custom = RuntimeSettings(template_name="x.tex.j2", model="m")
    store.set_runtime_settings(custom)
    assert store.get_runtime_settings() == custom
