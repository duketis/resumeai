"""Focused tests for orchestrator-level helpers that are easier to unit-test
directly than via the full pipeline.

The orchestrator's pipeline coverage lives in API + integration tests; this
file pins behaviour for module-level utilities (filename slugging, currently).
"""

from __future__ import annotations

import pytest

from resumeai.jd.models import (
    EmploymentType,
    JobRequirements,
    RemoteType,
    RoleType,
    Seniority,
)
from resumeai.runs.orchestrator import _resume_filename_stem, _sanitize_for_filename


def _jd(*, title: str = "", company: str | None = None) -> JobRequirements:
    return JobRequirements(
        title=title,
        company=company,
        role_type=RoleType.ENGINEERING,
        seniority=Seniority.MID,
        employment_type=EmploymentType.FULL_TIME,
        remote_type=RemoteType.REMOTE,
    )


class TestResumeFilenameStem:
    def test_combines_name_company_and_title(self) -> None:
        stem = _resume_filename_stem(
            "Jonathan Duketis",
            _jd(title="Software Developer", company="GoSource"),
        )
        assert stem == "Jonathan Duketis - GoSource - Software Developer"

    def test_drops_missing_company(self) -> None:
        stem = _resume_filename_stem(
            "Jonathan Duketis",
            _jd(title="Software Developer", company=None),
        )
        assert stem == "Jonathan Duketis - Software Developer"

    def test_falls_back_to_resume_when_everything_sanitises_to_empty(self) -> None:
        # JobRequirements.title is min_length=1 so we can't pass "", but
        # ``///`` validates fine and sanitises to empty. With name + company
        # also stripped, no parts survive -> the "resume" fallback fires.
        stem = _resume_filename_stem("///", _jd(title="///", company="///"))
        assert stem == "resume"

    def test_strips_filesystem_unsafe_characters(self) -> None:
        # Colons, slashes, and pipe characters can't appear in macOS filenames
        # cleanly -- the slug strips them so Preview can open the file.
        stem = _resume_filename_stem(
            "Name: First/Last",
            _jd(title="DevOps/SRE | Backend", company="ACME: Inc"),
        )
        assert ":" not in stem
        assert "/" not in stem
        assert "|" not in stem
        # Recognisable substrings survive (whitespace collapsed where needed).
        assert "First" in stem and "Last" in stem
        assert "DevOps" in stem and "Backend" in stem
        assert "ACME" in stem


class TestSanitizeForFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Jonathan Duketis", "Jonathan Duketis"),
            ("GoSource", "GoSource"),
            ("Software Developer", "Software Developer"),
            # Trailing dot is stripped along with other punctuation noise.
            ("ACME (Pty) Ltd.", "ACME (Pty) Ltd"),
            ("R&D Lead", "R&D Lead"),
            # Collapses run of whitespace + drops trailing/leading dashes.
            ("  spaces   inside  ", "spaces inside"),
            # Strips characters that break filesystem semantics.
            ("path/with/slashes", "pathwithslashes"),
            ("colons:bad", "colonsbad"),
            ("pipe|bad", "pipebad"),
            # Smart quotes etc. get stripped too.
            ("“curly”", "curly"),
            # Empty becomes empty.
            ("", ""),
            ("///", ""),
        ],
    )
    def test_strips_unsafe_chars(self, raw: str, expected: str) -> None:
        assert _sanitize_for_filename(raw) == expected
