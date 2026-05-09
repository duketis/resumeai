"""HTTP layer.

The :func:`~resumeai.api.app.create_app` factory wires the routers together
with the dependency-injected :class:`~resumeai.settings.store.SettingsStore`
and :class:`~resumeai.auth.google_oauth.OAuthService`. Routes are split by
concern (settings, oauth, onboarding); Jinja templates live in
``resumeai/api/templates/``.
"""

from __future__ import annotations
