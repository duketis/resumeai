"""resumeai — AI-driven resume tailoring service.

Reads a job description and the user's full career context, then drives a
master Google Docs template into a tailored resume via the Google Docs API.
The user's master document is never written to: every tailoring run produces
a fresh copy.

Public surface is intentionally minimal at the package level; consumers should
import from submodules (``resumeai.docs``, ``resumeai.jd``, ``resumeai.api``)
rather than re-exporting through here.
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["__version__"]
