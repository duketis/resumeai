"""Onboarding wizard renders + has the load-bearing redirect URL."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_onboarding_page_renders(client: TestClient) -> None:
    response = client.get("/onboarding")

    assert response.status_code == 200
    body = response.text
    # If the wizard's recommended redirect URI ever drifts from the route,
    # nothing else will catch it — this is the canonical assertion.
    assert "http://localhost:7842/api/auth/google/callback" in body
    assert "Google Cloud Console" in body
    assert "client_secret" in body
