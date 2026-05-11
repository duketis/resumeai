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

import pytest

from resumeai.agent.models import (
    TailoredBullet,
    TailoredProject,
    TailoredResume,
    TailoredWorkEntry,
)
from resumeai.context.models import Contact, Education
from resumeai.renderer.latex_renderer import (
    compile_pdf,
    render_tailored_resume_latex,
    render_tex,
    tex_escape,
)
from resumeai.renderer.models import RenderError, RenderStatus

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
        key_achievements=(
            "Built and maintain jobai with 705+ tests at 89% coverage.",
            "Shipped CDK infrastructure + observability for InTruth's platform.",
        ),
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
        personal_projects=(
            TailoredProject(
                name="jobai",
                description="local-first AI job-hunting platform",
                stack="Python, FastAPI, React, Docker",
                link="https://github.com/duketis/jobai",
                link_label="github.com/duketis/jobai",
                bullets=(
                    TailoredBullet(
                        text="Ingests 9,000+ jobs per cycle from 50+ ATS APIs.",
                        source_slug="jobai",
                    ),
                    TailoredBullet(
                        text="705+ tests at 89% coverage with mypy strict.",
                        source_slug="jobai",
                    ),
                ),
            ),
            TailoredProject(
                name="Strategy Miner",
                stack="Python, FastAPI, React, Backtrader",
                link=None,
                link_label="Private project",
                bullets=(
                    TailoredBullet(
                        text="AI-driven backtesting + strategy-optimisation platform.",
                        source_slug="strategy-miner",
                    ),
                ),
            ),
        ),
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

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Arrows -- master-resume.txt contains "native ↔ web messaging"
            # and the agent emitted "2021 → 2021" for a work-history period;
            # the default Latin Modern font can't render either, so we
            # substitute the LaTeX equivalent instead of leaving a missing
            # glyph box in the rendered PDF.
            ("native ↔ web", r"native $\leftrightarrow$ web"),
            ("2021 → 2025", r"2021 $\to$ 2025"),
            ("← back", r"$\leftarrow$ back"),
            # Dashes
            ("2020 – 2025", "2020 -- 2025"),
            ("a—b", "a---b"),
            # Smart quotes
            ("“hello”", "``hello''"),
            ("it’s", "it's"),
            # Misc symbols
            ("…end", r"\ldots{}end"),
            ("• bullet", r"\textbullet{} bullet"),
            ("≥90%", r"$\geq$90\%"),
            ("± 0.5", r"$\pm$ 0.5"),
            ("100°C", r"100\textdegree{}C"),
        ],
    )
    def test_unicode_typographic_substitutions(self, raw: str, expected: str) -> None:
        """Common Unicode the default font can't render is rewritten to its
        LaTeX equivalent. Applied AFTER the ASCII escape pass so the
        inserted LaTeX commands aren't themselves re-escaped."""
        assert tex_escape(raw) == expected

    def test_unicode_substitutions_compose_with_ascii_escapes(self) -> None:
        # Special chars AND unicode in the same string -- both pass cleanly.
        out = tex_escape("R&D → 100%")
        assert out == r"R\&D $\to$ 100\%"


# ---- render_tex ------------------------------------------------------------


class TestRenderTex:
    def test_minimal_resume_renders(self, minimal_tailored: TailoredResume) -> None:
        out = render_tex(minimal_tailored)
        assert r"\documentclass" in out
        assert "Jonathan Duketis" in out
        assert "me@example.com" in out
        assert r"\begin{document}" in out
        assert r"\end{document}" in out

    def test_full_resume_includes_all_sections(self, full_tailored: TailoredResume) -> None:
        out = render_tex(full_tailored)
        assert r"\section{Profile}" in out
        assert r"\section{Technical Skills}" in out
        assert r"\section{Education}" in out
        assert r"\section{Key Achievements}" in out
        assert r"\section{Professional Experience}" in out
        assert r"\section{Personal Projects}" in out
        # Bullet text
        assert "Built CDK infrastructure." in out
        # Skills joined as a single comma list
        assert "Python, TypeScript, AWS" in out
        # Education year shown
        assert "2020" in out
        # Certification rendered as its own subheading
        assert "Certificate IV in Property Services" in out
        # Key achievements rendered as bullets
        assert "Built and maintain jobai" in out
        # Personal projects: public one has a clickable href, private one
        # has a plain "Private project" right column. ``\resumeSubheading``
        # is a 4-arg macro -- the names appear inside the first {arg} and
        # the macro adds the \textbf{} itself. The right column is wrapped
        # in ``{\small ...}`` so a long URL doesn't overflow the column.
        assert r"\href{https://github.com/duketis/jobai}{github.com/duketis/jobai}" in out
        assert r"{\small Private project}" in out  # right column literal
        assert r"\resumeSubheading" in out
        assert "{jobai}" in out
        assert "{Strategy Miner}" in out

    def test_empty_section_is_omitted(self, minimal_tailored: TailoredResume) -> None:
        out = render_tex(minimal_tailored)
        assert r"\section{Profile}" not in out
        assert r"\section{Technical Skills}" not in out
        assert r"\section{Education}" not in out
        assert r"\section{Key Achievements}" not in out
        assert r"\section{Professional Experience}" not in out
        assert r"\section{Personal Projects}" not in out

    def test_personal_projects_omits_description_when_empty(self) -> None:
        """When description is blank the subtitle line collapses to just the
        stack -- no leading ``" $|$ "`` separator from an empty join entry."""
        tailored = TailoredResume(
            name="A",
            contact=Contact(email="a@b.co"),
            personal_projects=(
                TailoredProject(
                    name="Solo",
                    stack="Python",
                    link_label="Private project",
                    bullets=(TailoredBullet(text="Did the thing."),),
                ),
            ),
        )
        out = render_tex(tailored)
        # Name appears (bolded by the \resumeSubheading macro).
        assert r"\resumeSubheading" in out
        assert "Solo" in out
        # Subtitle line is just the stack -- no orphan separator.
        assert "{Python}{}" in out
        assert "$|$ Python" not in out

    def test_personal_project_without_link_or_label_has_empty_right_column(self) -> None:
        """``link`` AND ``link_label`` both None -> right column is literally
        empty (no ``{\\small ...}`` wrapper either)."""
        tailored = TailoredResume(
            name="A",
            contact=Contact(email="a@b.co"),
            personal_projects=(TailoredProject(name="X"),),
        )
        out = render_tex(tailored)
        # No project-specific \href (mailto:contact one is fine).
        assert "https://" not in out
        # No ``{\small \href...}`` or ``{\small <label>}`` wrapper since
        # the project has neither a link nor a label to wrap. (Plain
        # ``{\small`` does appear inside the ``\resumeSubheading`` macro
        # definition in the preamble; we check the specific link-wrap forms.)
        assert r"{\small \href" not in out
        # \resumeSubheading{X}{} -- the right column is literally empty.
        assert r"\resumeSubheading" in out
        assert "{X}{}" in out

    def test_personal_project_with_link_but_no_label_uses_link_as_label(self) -> None:
        tailored = TailoredResume(
            name="A",
            contact=Contact(email="a@b.co"),
            personal_projects=(
                TailoredProject(name="X", link="https://example.com/x", link_label=None),
            ),
        )
        out = render_tex(tailored)
        assert r"\href{https://example.com/x}{https://example.com/x}" in out

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

    def test_missing_tectonic_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    @_skip_no_tectonic
    def test_compiles_with_relative_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: orchestrator passes a relative ``output_dir`` like
        ``runs/<run_id>``. With ``cwd=output_dir``, passing the full path
        to tectonic doubled the prefix and the file was not found. The
        fix is to pass the basename, not ``str(tex_path)``."""
        monkeypatch.chdir(tmp_path)
        rel_dir = Path("runs/r1")
        tex = (
            r"\documentclass{article}"
            "\n"
            r"\begin{document}Hi\end{document}"
            "\n"
        )
        pdf = compile_pdf(tex, rel_dir)
        assert pdf.startswith(b"%PDF-")
        assert (tmp_path / rel_dir / "resume.pdf").exists()


# ---- end-to-end ------------------------------------------------------------


class TestRenderTailoredResumeLatex:
    @_skip_no_tectonic
    def test_end_to_end(self, full_tailored: TailoredResume, tmp_path: Path) -> None:
        out_dir = tmp_path / "run-001"
        result = render_tailored_resume_latex(full_tailored, out_dir)
        assert result.doc_id == "run-001"
        assert result.doc_url.startswith("file://")
        pdf_on_disk = out_dir / "resume.pdf"
        assert pdf_on_disk.read_bytes().startswith(b"%PDF-")
        assert result.pdf_size_bytes == pdf_on_disk.stat().st_size
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
    def test_end_to_end_with_relative_output_dir(
        self,
        full_tailored: TailoredResume,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Regression: production passes ``runs/<run_id>`` (relative). The
        # ``doc_url`` needs to be a valid ``file://`` URI, which requires
        # the resolved absolute path.
        monkeypatch.chdir(tmp_path)
        result = render_tailored_resume_latex(full_tailored, Path("runs") / "run-relative")
        assert result.doc_id == "run-relative"
        assert result.doc_url.startswith("file://")
        pdf_on_disk = tmp_path / "runs" / "run-relative" / "resume.pdf"
        assert pdf_on_disk.read_bytes().startswith(b"%PDF-")
        assert result.pdf_size_bytes == pdf_on_disk.stat().st_size

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
