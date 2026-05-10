"""Persistent + in-memory storage for :class:`Run` records."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from resumeai.runs.models import Run, RunStatus

DEFAULT_DB_PATH = Path("~/.resumeai/resumeai.db").expanduser()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs (updated_at DESC);
"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RunsStore(Protocol):
    """Storage contract for run records."""

    def get(self, run_id: str) -> Run | None: ...
    def save(self, run: Run) -> None: ...
    def list_recent(self, limit: int = 20) -> list[Run]: ...


class SqliteRunsStore:
    """SQLite-backed implementation. Shares the Settings DB by default."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the connection survives FastAPI's
        # worker-threadpool dispatch. SQLite serialises writes at the C
        # level; per-call usage is safe at our concurrency profile.
        self._conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def get(self, run_id: str) -> Run | None:
        row = self._conn.execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return Run.model_validate_json(row[0])

    def save(self, run: Run) -> None:
        payload = run.model_dump_json()
        self._conn.execute(
            """INSERT INTO runs (id, payload, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE
                 SET payload = excluded.payload,
                     updated_at = excluded.updated_at""",
            (
                run.id,
                payload,
                run.created_at.isoformat(),
                run.updated_at.isoformat(),
            ),
        )

    def list_recent(self, limit: int = 20) -> list[Run]:
        rows = self._conn.execute(
            "SELECT payload FROM runs ORDER BY updated_at DESC LIMIT ?",
            (max(0, limit),),
        ).fetchall()
        return [Run.model_validate_json(row[0]) for row in rows]


class InMemoryRunsStore:
    """In-memory store with identical semantics. For tests."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def save(self, run: Run) -> None:
        self._runs[run.id] = run

    def list_recent(self, limit: int = 20) -> list[Run]:
        ordered = sorted(self._runs.values(), key=lambda r: r.updated_at, reverse=True)
        return ordered[: max(0, limit)]


def update_run(
    store: RunsStore,
    run_id: str,
    *,
    status: RunStatus | None = None,
    detail: str | None = None,
    error: str | None = None,
    requirements: object | None = None,
    tailored: object | None = None,
    result: object | None = None,
) -> Run:
    """Mutate a stored run by replacing fields the caller supplied.

    Returns the updated, persisted run. Raises ``KeyError`` if the run is
    missing — callers should always have created the run via ``save`` first.
    """
    current = store.get(run_id)
    if current is None:
        raise KeyError(f"unknown run {run_id!r}")

    updates: dict[str, object] = {"updated_at": _utcnow()}
    if status is not None:
        updates["status"] = status
    if detail is not None:
        updates["detail"] = detail
    if error is not None:
        updates["error"] = error
    if requirements is not None:
        updates["requirements"] = requirements
    if tailored is not None:
        updates["tailored"] = tailored
    if result is not None:
        updates["result"] = result

    new_run = current.model_copy(update=updates)
    store.save(new_run)
    return new_run
