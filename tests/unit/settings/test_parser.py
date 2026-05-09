"""Parsers for OAuth client JSON and Google Doc URLs."""

from __future__ import annotations

import json

import pytest

from resumeai.settings.parser import (
    DocUrlParseError,
    OAuthClientParseError,
    parse_doc_id_from_url,
    parse_oauth_client_json,
)

_WEB_CLIENT = json.dumps(
    {
        "web": {
            "client_id": "cid.apps.googleusercontent.com",
            "project_id": "resumeai-456",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_secret": "supersecret",
            "redirect_uris": ["http://localhost:7842/api/auth/google/callback"],
        }
    }
)


_INSTALLED_CLIENT = json.dumps(
    {
        "installed": {
            "client_id": "cid.apps.googleusercontent.com",
            "client_secret": "supersecret",
        }
    }
)


def test_parses_web_client_json() -> None:
    client = parse_oauth_client_json(_WEB_CLIENT)

    assert client.client_id == "cid.apps.googleusercontent.com"
    assert client.client_secret == "supersecret"
    assert client.project_id == "resumeai-456"
    assert client.redirect_uris == ("http://localhost:7842/api/auth/google/callback",)


def test_parses_installed_client_json_with_defaults() -> None:
    client = parse_oauth_client_json(_INSTALLED_CLIENT)

    assert client.client_id == "cid.apps.googleusercontent.com"
    assert client.auth_uri == "https://accounts.google.com/o/oauth2/auth"
    assert client.redirect_uris == ()
    assert client.project_id is None


def test_parses_bytes_input() -> None:
    client = parse_oauth_client_json(_WEB_CLIENT.encode())
    assert client.client_id == "cid.apps.googleusercontent.com"


def test_rejects_invalid_json() -> None:
    with pytest.raises(OAuthClientParseError, match="not valid JSON"):
        parse_oauth_client_json("{not json")


def test_rejects_non_object_top_level() -> None:
    with pytest.raises(OAuthClientParseError, match="JSON object"):
        parse_oauth_client_json("[]")


def test_rejects_unrecognised_shape() -> None:
    with pytest.raises(OAuthClientParseError, match="'web' or 'installed'"):
        parse_oauth_client_json('{"foo": {}}')


def test_rejects_non_object_inner() -> None:
    with pytest.raises(OAuthClientParseError, match="must be an object"):
        parse_oauth_client_json('{"web": "not-an-object"}')


def test_rejects_missing_required_field() -> None:
    with pytest.raises(OAuthClientParseError, match="client_secret"):
        parse_oauth_client_json('{"web": {"client_id": "x"}}')


def test_redirect_uris_default_when_not_a_list() -> None:
    client = parse_oauth_client_json(
        '{"web": {"client_id": "c", "client_secret": "s", "redirect_uris": "wrong"}}'
    )
    assert client.redirect_uris == ()


def test_parses_doc_id_from_full_url() -> None:
    url = "https://docs.google.com/document/d/1WwvjMB02GDD1GEKXVqBHfHurM0bD5rpG/edit"
    assert parse_doc_id_from_url(url) == "1WwvjMB02GDD1GEKXVqBHfHurM0bD5rpG"


def test_parses_doc_id_from_url_with_query_params() -> None:
    url = "https://docs.google.com/document/d/abc-123_XYZ/edit?usp=sharing&tab=t.0"
    assert parse_doc_id_from_url(url) == "abc-123_XYZ"


def test_passes_through_bare_doc_id() -> None:
    assert parse_doc_id_from_url("1AbC-def_GHI") == "1AbC-def_GHI"


def test_strips_whitespace() -> None:
    assert parse_doc_id_from_url("  1AbC-def_GHI  ") == "1AbC-def_GHI"


def test_rejects_empty_input() -> None:
    with pytest.raises(DocUrlParseError, match="empty"):
        parse_doc_id_from_url("   ")


def test_rejects_non_doc_url() -> None:
    with pytest.raises(DocUrlParseError, match="extract"):
        parse_doc_id_from_url("https://example.com/not-a-doc")
