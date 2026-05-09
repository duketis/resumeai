"""Settings module: persistent storage for OAuth credentials and user prefs.

Everything that the user configures via the in-app Settings page lives here.
Storage backend is SQLite by default (``SqliteSettingsStore``); tests use the
``InMemorySettingsStore`` to avoid filesystem touches.
"""

from __future__ import annotations
