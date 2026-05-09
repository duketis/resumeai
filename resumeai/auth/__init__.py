"""Google OAuth 2.0 + identity layer.

Owns the consent-flow URLs, the code-for-token exchange, the userinfo lookup
that tells us *which* Google account just connected, and the refresh +
revocation endpoints. Everything goes through the :class:`OAuthService`
``Protocol`` so routes don't import the concrete Google implementation.
"""

from __future__ import annotations
