"""Parsers for user-supplied configuration blobs.

Two surfaces:

- ``parse_oauth_client_json``: turns the JSON Google Cloud Console emits
  (either ``{"web": {...}}`` or ``{"installed": {...}}`` shape) into an
  :class:`~resumeai.settings.models.OAuthClient`.
- ``parse_doc_id_from_url``: extracts the ``DOC_ID`` from a Google Docs URL of
  the form ``https://docs.google.com/document/d/DOC_ID/edit``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from resumeai.settings.models import OAuthClient

_DOC_ID_RE = re.compile(r"/document/d/([A-Za-z0-9_-]+)")


class OAuthClientParseError(ValueError):
    """Raised when the supplied JSON is not a recognised OAuth client blob."""


class DocUrlParseError(ValueError):
    """Raised when the supplied string is not a recognisable Google Doc URL."""


def parse_oauth_client_json(raw: str | bytes) -> OAuthClient:
    """Parse the JSON Google Cloud Console gives the user.

    Accepts both the ``web`` (web-app) and ``installed`` (installed-app)
    shapes; raises :class:`OAuthClientParseError` on anything else.
    """
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OAuthClientParseError(f"not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise OAuthClientParseError("expected a JSON object at the top level")

    inner: Any
    if "web" in data:
        inner = data["web"]
    elif "installed" in data:
        inner = data["installed"]
    else:
        raise OAuthClientParseError(
            "expected top-level 'web' or 'installed' key (Google Cloud Console JSON)"
        )

    if not isinstance(inner, dict):
        raise OAuthClientParseError("'web' / 'installed' value must be an object")

    try:
        client_id = str(inner["client_id"])
        client_secret = str(inner["client_secret"])
    except KeyError as exc:
        raise OAuthClientParseError(f"missing required field: {exc.args[0]}") from exc

    redirect_uris_raw = inner.get("redirect_uris", [])
    redirect_uris: tuple[str, ...]
    if isinstance(redirect_uris_raw, list):
        redirect_uris = tuple(str(u) for u in redirect_uris_raw)
    else:
        redirect_uris = ()

    return OAuthClient(
        client_id=client_id,
        client_secret=client_secret,
        auth_uri=str(inner.get("auth_uri", "https://accounts.google.com/o/oauth2/auth")),
        token_uri=str(inner.get("token_uri", "https://oauth2.googleapis.com/token")),
        project_id=str(inner["project_id"]) if "project_id" in inner else None,
        redirect_uris=redirect_uris,
    )


def parse_doc_id_from_url(url: str) -> str:
    """Extract a Google Doc ID from a docs.google.com URL or a bare ID.

    A bare ID (matching the ID character class) is returned unchanged so users
    can paste either form.
    """
    url = url.strip()
    if not url:
        raise DocUrlParseError("empty input")

    match = _DOC_ID_RE.search(url)
    if match:
        return match.group(1)

    # Bare ID — accept if it matches the Google Docs ID character class.
    if re.fullmatch(r"[A-Za-z0-9_-]+", url):
        return url

    raise DocUrlParseError(f"could not extract a Google Doc ID from {url!r}")
