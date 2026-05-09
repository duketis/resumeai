"""Phase 0 smoke test: the package imports and exposes its version string.

This test exists to guarantee the CI pipeline has something to run before
any feature code lands. It will be deleted (or absorbed into a richer
package-level test) once Phase 1 introduces the first real module.
"""

from __future__ import annotations

import resumeai


def test_package_exposes_version_string() -> None:
    assert isinstance(resumeai.__version__, str)
    assert resumeai.__version__.count(".") == 2
