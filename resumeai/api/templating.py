"""Shared Jinja2 ``Templates`` instance.

Single export so every router uses the same template directory + autoescaping
config. Lives at module level so the FastAPI dependency system can rely on
its identity for testing.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
