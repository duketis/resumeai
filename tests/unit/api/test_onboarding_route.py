"""Onboarding wizard renders + has the load-bearing redirect URL."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_onboarding_page_renders(client: TestClient) -> None:
    response = client.get("/onboarding")

    assert response.status_code == 200
    body = response.text
    # The wizard tells the user to register both the canonical hostnames
    # so they don't trip over Google's localhost / 127.0.0.1 distinction.
    assert "http://localhost:7842/api/auth/google/callback" in body
    assert "http://127.0.0.1:7842/api/auth/google/callback" in body
    assert "Google Cloud Console" in body
    assert "client_secret" in body


def test_onboarding_page_shows_session_specific_callback(
    client: TestClient,
) -> None:
    """The wizard surfaces the redirect URI the OAuth start route will
    actually use for *this* session, so the user can't pick a hostname
    they didn't register."""
    response = client.get("/onboarding")
    body = response.text
    # TestClient defaults to http://testserver, so that's the base.
    assert "http://testserver/api/auth/google/callback" in body
