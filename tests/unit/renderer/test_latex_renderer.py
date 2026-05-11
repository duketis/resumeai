"""Unit tests for the LaTeX renderer.

Covers the pure-function entry points:

- :func:`tex_escape` -- LaTeX special-char escaping
- :func:`render_tex` -- TailoredResume + template -> .tex string
- :func:`compile_pdf` -- .tex content + output dir -> PDF bytes
- :func:`render_tailored_resume_latex` -- end-to-end orchestrator entry

``compile_pdf`` is skipped when ``tectonic`` is not on PATH; CI installs
it so coverage is preserved there.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from resumeai.agent.models import TailoredBullet, TailoredResume, TailoredWorkEntry
from resumeai.context.models import Contact, Education
from resumeai.renderer.latex_renderer import (
    compile_pdf,
    render_tailored_resume_latex,
    render_tex,
    tex_escape,
)
from resumeai.renderer.models import RenderError, RenderStatus

if TYPE_CHECKING:
    from collections.abc import Iterator


_TECTONIC_AVAILABLE = shutil.which("tectonic") is not None
_skip_no_tectonic = pytest.mark.skipif(
    not _TECTONIC_AVAILABLE,
    reason="tectonic is not installed on PATH",
)


# ---- fixtures --------------------------------------------------------------


@pytest.fixture
def minimal_tailored() -> TailoredResume:
    return TailoredResume(
        name="Jonathan Duketis",
        contact=Contact(email="me@example.com"),
    )


@pytest.fixture
def full_tailored() -> TailoredResume:
    return TailoredResume(
        name="Jonathan Duketis",
        headline="Full-stack engineer",
        contact=Contact(
            email="me@example.com",
            location="Melbourne",
            github="github.com/duketis",
        ),
        summary="Five years AWS + Rails consulting; ships open-source code.",
        skills=("Python", "TypeScript", "AWS (Lambda, ECS/Fargate)"),
        work_history=(
            TailoredWorkEntry(
                company="DiUS Computing",
                title="Software Engineer",
                period="2020 -- 2025",
                location="Melbourne",
                bullets=(
                    TailoredBullet(text="Built CDK infrastructure."),
                    TailoredBullet(text="Optimistic locking under concurrency."),
                ),
            ),
        ),
        education=(
            Education(
                institution="Swinburne University",
                degree="Bachelor of Computer Science",
                year_end=2020,
            ),
        ),
        certifications=("Certificate IV in Property Services",),
    )


# ---- tex_escape ------------------------------------------------------------


class TestTexEscape:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", ""),
            ("plain text", "plain text"),
            ("100%", r"100\%"),
            ("$10k", r"\$10k"),
            ("R&D", r"R\&D"),
            ("file_name.py", r"file\_name.py"),
            ("hash#tag", r"hash\#tag"),
            ("a{b}c", r"a\{b\}c"),
            ("~tilde~", r"\textasciitilde{}tilde\textasciitilde{}"),
            ("^caret^", r"\textasciicircum{}caret\textasciicircum{}"),
            (r"\already", r"\textbackslash{}already"),
        ],
    )
    def test_escapes_special_chars(self, raw: str, expected: str) -> None:
        assert tex_escape(raw) == expected

    def test_none_returns_empty(self) -> None:
        assert tex_escape(None) == ""

    def test_backslash_does_not_double_escape(self) -> None:
        # The order matters: backslash must be escaped first or it will
        # eat all the other escapes' backslashes.
        out = tex_escape("$ & %")
        assert out == r"\$ \& \%"
        assert "textbackslash" not in out


# ---- render_tex ------------------------------------------------------------


class TestRenderTex:
    def test_minimal_resume_renders(self, minimal_tailored: TailoredResume) -> None:
        out = render_tex(minimal_tailored)
        assert r"\documentclass" in out
        assert "Jonathan Duketis" in out
        assert "me@example.com" in out
        assert r"\begin{document}" in out
        assert r"\end{document}" in out

    def test_full_resume_includes_all_sections(
        self, full_tailored: TailoredResume
    ) -> None:
        out = render_tex(full_tailored)
        assert r"\section{Profile}" in out
        assert r"\section{Technical Skills}" in out
        assert r"\section{Education}" in out
        assert r"\section{Professional Experience}" in out
        # Bullet text
        assert "Built CDK infrastructure." in out
        # Skills joined as a single comma list
        assert "Python, TypeScript, AWS" in out
        # Education year shown
        assert "2020" in out
        # Certification rendered as its own subheading
        assert "Certificate IV in Property Services" in out

    def test_empty_section_is_omitted(self, minimal_tailored: TailoredResume) -> None:
        out = render_tex(minimal_tailored)
        assert r"\section{Profile}" not in out
        assert r"\section{Technical Skills}" not in out
        assert r"\section{Education}" not in out
        assert r"\section{Professional Experience}" not in out

    def test_user_input_is_escaped(self) -> None:
        tailored = TailoredResume(
            name="A & B",
            contact=Contact(email="x@y.com"),
            summary="100% sure file_name.py works for $5k R&D budget.",
        )
        out = render_tex(tailored)
        # Each special char from user input shows up in escaped form. We
        # can't do a blanket ``not in`` on the raw chars because the
        # template itself contains LaTeX special chars (e.g. ``$|$`` math
        # separators in the contact line) -- so we focus on the user data.
        assert r"A \& B" in out
        assert r"100\%" in out
        assert r"\$5k" in out
        assert r"R\&D" in out
        assert r"file\_name.py" in out
        # And the unescaped user strings never appear verbatim.
        assert "A & B" not in out
        assert "R&D" not in out
        assert "file_name.py" not in out

    def test_missing_template_raises(
        self, minimal_tailored: TailoredResume, tmp_path: Path
    ) -> None:
        with pytest.raises(RenderError, match="template not found"):
            render_tex(
                minimal_tailored,
                template_name="nonexistent.tex.j2",
                templates_dir=tmp_path,
            )

    def test_missing_templates_dir_raises(
        self, minimal_tailored: TailoredResume, tmp_path: Path
    ) -> None:
        bogus = tmp_path / "no-such-dir"
        with pytest.raises(RenderError, match="templates directory not found"):
            render_tex(minimal_tailored, templates_dir=bogus)


# ---- compile_pdf -----------------------------------------------------------


class TestCompilePdf:
    @_skip_no_tectonic
    def test_compiles_minimal_doc(self, tmp_path: Path) -> None:
        tex = (
            r"\documentclass{article}"
            "\n"
            r"\begin{document}"
            "\n"
            r"Hello LaTeX."
            "\n"
            r"\end{document}"
            "\n"
        )
        pdf = compile_pdf(tex, tmp_path)
        assert pdf.startswith(b"%PDF-")
        assert (tmp_path / "resume.pdf").exists()

    def test_missing_tectonic_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "resumeai.renderer.latex_renderer.shutil.which",
            lambda _cmd: None,
        )
        with pytest.raises(RenderError, match="tectonic not found"):
            compile_pdf(r"\documentclass{article}", tmp_path)

    @_skip_no_tectonic
    def test_invalid_latex_raises_with_message(self, tmp_path: Path) -> None:
        # Missing \end{document} -> tectonic errors out.
        with pytest.raises(RenderError, match="tectonic compile failed"):
            compile_pdf(r"\documentclass{article}\begin{document}OOPS", tmp_path)


# ---- end-to-end ------------------------------------------------------------


class TestRenderTailoredResumeLatex:
    @_skip_no_tectonic
    def test_end_to_end(
        self, full_tailored: TailoredResume, tmp_path: Path
    ) -> None:
        result = render_tailored_resume_latex(full_tailored, tmp_path / "run-001")
        assert result.doc_id == "run-001"
        assert result.doc_url.startswith("file://")
        assert result.pdf_bytes.startswith(b"%PDF-")
        # Five known sections always produce a diff entry, REPLACED or SKIPPED.
        assert len(result.diffs) == 5
        kinds = {d.kind for d in result.diffs}
        assert kinds == {
            "summary",
            "skills",
            "work_history",
            "education",
            "certifications",
        }

    @_skip_no_tectonic
    def test_diff_status_reflects_section_content(
        self, minimal_tailored: TailoredResume, tmp_path: Path
    ) -> None:
        result = render_tailored_resume_latex(minimal_tailored, tmp_path / "run-002")
        statuses = {d.kind: d.status for d in result.diffs}
        # Minimal resume has none of the optional sections filled.
        assert statuses["summary"] is RenderStatus.SKIPPED_EMPTY
        assert statuses["skills"] is RenderStatus.SKIPPED_EMPTY
        assert statuses["work_history"] is RenderStatus.SKIPPED_EMPTY
        assert statuses["education"] is RenderStatus.SKIPPED_EMPTY
        assert statuses["certifications"] is RenderStatus.SKIPPED_EMPTY
