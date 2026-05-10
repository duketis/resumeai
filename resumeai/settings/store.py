"""Persistent storage for Settings.

Two implementations:

- :class:`SqliteSettingsStore` — production. SQLite at ``~/.resumeai/resumeai.db``
  with owner-only file permissions (``0600``). All credentials persisted as JSON
  blobs in dedicated single-row tables.
- :class:`InMemorySettingsStore` — used by tests. Identical semantics, no I/O.

Both implement the :class:`SettingsStore` ``Protocol`` so the API and services
hold a reference to the protocol, never to the concrete type.
"""

from __future__ import annotations

import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from resumeai.settings.models import (
    GoogleCredentials,
    OAuthClient,
    OAuthPending,
    TemplateDoc,
)

DEFAULT_DB_PATH = Path("~/.resumeai/resumeai.db").expanduser()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS oauth_client (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    client_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS google_credentials (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    creds_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS oauth_pending (
    state TEXT PRIMARY KEY,
    redirect_uri TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS template_doc (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    template_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class SettingsStore(Protocol):
    """The storage contract every Settings backend must satisfy."""

    def get_oauth_client(self) -> OAuthClient | None: ...
    def set_oauth_client(self, client: OAuthClient) -> None: ...

    def get_google_credentials(self) -> GoogleCredentials | None: ...
    def set_google_credentials(self, creds: GoogleCredentials) -> None: ...
    def clear_google_credentials(self) -> None: ...

    def store_oauth_pending(self, pending: OAuthPending) -> None: ...
    def consume_oauth_pending(self, state: str) -> OAuthPending | None: ...

    def get_template(self) -> TemplateDoc | None: ...
    def set_template(self, template: TemplateDoc) -> None: ...
    def clear_template(self) -> None: ...


class SqliteSettingsStore:
    """SQLite-backed store. Default DB path: ``~/.resumeai/resumeai.db``."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the connection survives FastAPI's
        # worker-threadpool dispatch. SQLite serialises writes at the C
        # level; per-call usage is safe at our concurrency profile.
        self._conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)
        # Owner-only perms on the credentials DB.
        self._db_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def close(self) -> None:
        self._conn.close()

    # -- OAuth client ------------------------------------------------------

    def get_oauth_client(self) -> OAuthClient | None:
        row = self._conn.execute("SELECT client_json FROM oauth_client WHERE id = 1").fetchone()
        if row is None:
            return None
        return OAuthClient.model_validate_json(row[0])

    def set_oauth_client(self, client: OAuthClient) -> None:
        self._conn.execute(
            """INSERT INTO oauth_client (id, client_json, updated_at)
               VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE
                 SET client_json = excluded.client_json,
                     updated_at = excluded.updated_at""",
            (client.model_dump_json(), _utcnow_iso()),
        )

    # -- Google credentials ------------------------------------------------

    def get_google_credentials(self) -> GoogleCredentials | None:
        row = self._conn.execute(
            "SELECT creds_json FROM google_credentials WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return GoogleCredentials.model_validate_json(row[0])

    def set_google_credentials(self, creds: GoogleCredentials) -> None:
        self._conn.execute(
            """INSERT INTO google_credentials (id, creds_json, updated_at)
               VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE
                 SET creds_json = excluded.creds_json,
                     updated_at = excluded.updated_at""",
            (creds.model_dump_json(), _utcnow_iso()),
        )

    def clear_google_credentials(self) -> None:
        self._conn.execute("DELETE FROM google_credentials WHERE id = 1")

    # -- OAuth pending state ----------------------------------------------

    def store_oauth_pending(self, pending: OAuthPending) -> None:
        self._conn.execute(
            "INSERT INTO oauth_pending (state, redirect_uri, created_at) VALUES (?, ?, ?)",
            (pending.state, pending.redirect_uri, pending.created_at.isoformat()),
        )

    def consume_oauth_pending(self, state: str) -> OAuthPending | None:
        row = self._conn.execute(
            "SELECT state, redirect_uri, created_at FROM oauth_pending WHERE state = ?",
            (state,),
        ).fetchone()
        if row is None:
            return None
        self._conn.execute("DELETE FROM oauth_pending WHERE state = ?", (state,))
        return OAuthPending(
            state=row[0], redirect_uri=row[1], created_at=datetime.fromisoformat(row[2])
        )

    # -- Template doc ------------------------------------------------------

    def get_template(self) -> TemplateDoc | None:
        row = self._conn.execute("SELECT template_json FROM template_doc WHERE id = 1").fetchone()
        if row is None:
            return None
        return TemplateDoc.model_validate_json(row[0])

    def set_template(self, template: TemplateDoc) -> None:
        self._conn.execute(
            """INSERT INTO template_doc (id, template_json, updated_at)
               VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE
                 SET template_json = excluded.template_json,
                     updated_at = excluded.updated_at""",
            (template.model_dump_json(), _utcnow_iso()),
        )

    def clear_template(self) -> None:
        self._conn.execute("DELETE FROM template_doc WHERE id = 1")


class InMemorySettingsStore:
    """In-memory store with identical semantics. For tests."""

    def __init__(self) -> None:
        self._oauth_client: OAuthClient | None = None
        self._google_creds: GoogleCredentials | None = None
        self._pending: dict[str, OAuthPending] = {}
        self._template: TemplateDoc | None = None

    def get_oauth_client(self) -> OAuthClient | None:
        return self._oauth_client

    def set_oauth_client(self, client: OAuthClient) -> None:
        self._oauth_client = client

    def get_google_credentials(self) -> GoogleCredentials | None:
        return self._google_creds

    def set_google_credentials(self, creds: GoogleCredentials) -> None:
        self._google_creds = creds

    def clear_google_credentials(self) -> None:
        self._google_creds = None

    def store_oauth_pending(self, pending: OAuthPending) -> None:
        self._pending[pending.state] = pending

    def consume_oauth_pending(self, state: str) -> OAuthPending | None:
        return self._pending.pop(state, None)

    def get_template(self) -> TemplateDoc | None:
        return self._template

    def set_template(self, template: TemplateDoc) -> None:
        self._template = template

    def clear_template(self) -> None:
        self._template = None
