"""LaTeX renderer: TailoredResume -> .tex -> tectonic -> PDF.

Replaces the v0.x Google Docs renderer. Renders the full resume from
scratch on every run -- LaTeX decouples layout from content length, so
adding or removing bullets cannot break formatting.

The renderer is split into three pure-function entry points:

- :func:`render_tex`     -- TailoredResume + template -> .tex string
- :func:`compile_pdf`    -- .tex string + output dir -> PDF bytes on disk
- :func:`render_tailored_resume_latex` -- end-to-end orchestrator entry

The Google Docs ``render_tailored_resume`` shape (returns a
:class:`RenderResult`) is preserved so the orchestrator wiring change is
a single import + call-site swap.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2

from resumeai.renderer.models import (
    RenderDiff,
    RenderError,
    RenderResult,
    RenderStatus,
)

if TYPE_CHECKING:
    from tailor_core.context.models import Education

    from resumeai.agent.models import TailoredProject, TailoredResume, TailoredWorkEntry


DEFAULT_TEMPLATES_DIR: Path = Path(__file__).parent / "templates"
DEFAULT_TEMPLATE_NAME: str = "default.tex.j2"

# Escapes that don't introduce LaTeX control sequences -- safe to apply
# in order. Backslash gets a sentinel pass first so the ``{`` / ``}`` rules
# don't re-escape the braces of ``\textbackslash{}``.
_LATEX_SIMPLE_ESCAPES: tuple[tuple[str, str], ...] = (
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)
# Sentinel must contain no characters that appear in _LATEX_SIMPLE_ESCAPES,
# or the loop will re-escape it. Plain ASCII letters wrapped in NULs is safe.
_BACKSLASH_SENTINEL = "\x00RESUMEAIBSLASH\x00"

# Typographic Unicode that the default Latin Modern font can't render
# (or renders as a missing-glyph box). The agent and the user's source
# material both emit these; we substitute the LaTeX equivalent so the
# PDF stays clean. Applied AFTER the ASCII escape pass so the backslash
# sentinel doesn't eat the LaTeX commands we introduce here.
_LATEX_UNICODE_MAP: tuple[tuple[str, str], ...] = (
    # Arrows
    ("↔", r"$\leftrightarrow$"),
    ("→", r"$\to$"),
    ("←", r"$\leftarrow$"),
    ("↑", r"$\uparrow$"),
    ("↓", r"$\downarrow$"),
    ("⇒", r"$\Rightarrow$"),
    ("⇐", r"$\Leftarrow$"),
    # Dashes
    ("–", "--"),
    ("—", "---"),
    # Ellipsis / bullet
    ("…", r"\ldots{}"),
    ("•", r"\textbullet{}"),
    # Math relations
    ("≤", r"$\leq$"),
    ("≥", r"$\geq$"),
    ("≠", r"$\neq$"),
    ("±", r"$\pm$"),
    ("×", r"$\times$"),
    ("÷", r"$\div$"),
    # Symbols
    ("°", r"\textdegree{}"),
    ("©", r"\textcopyright{}"),
    ("®", r"\textregistered{}"),
    ("™", r"\texttrademark{}"),
    # Smart quotes
    ("“", "``"),
    ("”", "''"),
    ("‘", "`"),
    ("’", "'"),
)


def tex_escape(text: str | None) -> str:
    """Escape LaTeX special characters in plain user text. ``None`` -> ``""``.

    The escape is single-pass safe: backslash is rewritten to a sentinel
    first so the subsequent ``{`` and ``}`` rules can't re-escape the
    braces of the ``\\textbackslash{}`` replacement. The typographic
    Unicode pass runs last so its inserted LaTeX commands aren't
    re-escaped.
    """
    if not text:
        return ""
    out = text.replace("\\", _BACKSLASH_SENTINEL)
    for char, replacement in _LATEX_SIMPLE_ESCAPES:
        out = out.replace(char, replacement)
    out = out.replace(_BACKSLASH_SENTINEL, r"\textbackslash{}")
    for char, replacement in _LATEX_UNICODE_MAP:
        out = out.replace(char, replacement)
    return out


def render_tex(
    tailored: TailoredResume,
    *,
    template_name: str = DEFAULT_TEMPLATE_NAME,
    templates_dir: Path | None = None,
) -> str:
    """Render a :class:`TailoredResume` into a ``.tex`` string."""
    directory = templates_dir or DEFAULT_TEMPLATES_DIR
    if not directory.exists():
        raise RenderError(f"templates directory not found: {directory}")
    env = _build_env(directory)
    try:
        template = env.get_template(template_name)
    except jinja2.TemplateNotFound as exc:
        raise RenderError(f"template not found: {template_name}") from exc
    return template.render(resume=_escape_resume(tailored))


def compile_pdf(
    tex_content: str,
    output_dir: Path,
    *,
    stem: str = "resume",
) -> bytes:
    """Write ``tex_content`` and compile it via tectonic. Returns PDF bytes."""
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RenderError("tectonic not found on PATH. Install with `brew install tectonic`.")
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / f"{stem}.tex"
    pdf_path = output_dir / f"{stem}.pdf"
    tex_path.write_text(tex_content, encoding="utf-8")
    # ``cwd=output_dir`` means tectonic must see the input as just the filename;
    # passing the full path here would compose with cwd and look for the file
    # at ``<output_dir>/<output_dir>/<stem>.tex`` when output_dir is relative.
    proc = subprocess.run(  # noqa: S603 -- args are constants, paths controlled
        [tectonic, "--chatter=minimal", tex_path.name],
        capture_output=True,
        text=True,
        check=False,
        cwd=output_dir,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RenderError(f"tectonic compile failed (exit {proc.returncode}):\n{tail}")
    if not pdf_path.exists():
        raise RenderError(f"tectonic returned 0 but {pdf_path} is missing")
    return pdf_path.read_bytes()


def render_tailored_resume_latex(
    tailored: TailoredResume,
    output_dir: Path,
    *,
    template_name: str = DEFAULT_TEMPLATE_NAME,
    templates_dir: Path | None = None,
    stem: str = "resume",
) -> RenderResult:
    """End-to-end: TailoredResume -> .tex on disk -> PDF on disk -> RenderResult.

    ``stem`` controls the on-disk filename for both the ``.tex`` and ``.pdf``;
    the orchestrator builds a JD-flavoured stem so the user can tell several
    open PDFs apart in Preview (default ``"resume"`` keeps tests stable).
    """
    tex_content = render_tex(tailored, template_name=template_name, templates_dir=templates_dir)
    pdf_bytes = compile_pdf(tex_content, output_dir, stem=stem)
    # Resolve so callers passing a relative ``runs/<run_id>`` get a valid
    # ``file://`` URL -- Path.as_uri() rejects relative paths.
    pdf_path = (output_dir / f"{stem}.pdf").resolve()
    diffs = tuple(
        RenderDiff(
            kind=kind,
            heading=kind,
            status=(
                RenderStatus.REPLACED
                if _section_has_content(tailored, kind)
                else RenderStatus.SKIPPED_EMPTY
            ),
        )
        for kind in ("summary", "skills", "work_history", "education", "certifications")
    )
    return RenderResult(
        doc_id=output_dir.name or "resume",
        doc_url=pdf_path.as_uri(),
        pdf_size_bytes=len(pdf_bytes),
        diffs=diffs,
    )


# --- internals --------------------------------------------------------------


def _build_env(templates_dir: Path) -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_dir),
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,  # noqa: S701 -- output is LaTeX; we tex_escape() user data ourselves
        keep_trailing_newline=True,
    )
    env.filters["format_period"] = _format_education_period
    return env


def _format_education_period(edu: dict[str, Any]) -> str:
    """Jinja filter: render an Education year range. Empty string if absent."""
    start = edu.get("year_start")
    end = edu.get("year_end")
    if start and end:
        return str(end) if start == end else f"{start}--{end}"
    if end:
        return str(end)
    if start:
        return f"{start}--present"
    return ""


def _section_has_content(tailored: TailoredResume, kind: str) -> bool:
    if kind == "summary":
        return bool(tailored.summary.strip())
    if kind == "skills":
        return bool(tailored.skills)
    if kind == "work_history":
        return bool(tailored.work_history)
    if kind == "education":
        return bool(tailored.education)
    if kind == "certifications":
        return bool(tailored.certifications)
    return False


def _escape_resume(tailored: TailoredResume) -> dict[str, Any]:
    """Walk the TailoredResume and produce a dict with every str tex-escaped.

    Integer fields (year_start, year_end) are passed through unchanged.
    Nested objects (Contact, Education, TailoredWorkEntry, TailoredBullet)
    become nested dicts so Jinja can access them by attribute or by key
    interchangeably.
    """
    return {
        "name": tex_escape(tailored.name),
        "headline": tex_escape(tailored.headline),
        "summary": tex_escape(tailored.summary),
        "skills": tuple(tex_escape(s) for s in tailored.skills),
        "key_achievements": tuple(tex_escape(a) for a in tailored.key_achievements),
        "certifications": tuple(tex_escape(c) for c in tailored.certifications),
        "rationale": tex_escape(tailored.rationale),
        "contact": {
            "email": tex_escape(tailored.contact.email),
            "phone": tex_escape(tailored.contact.phone),
            "location": tex_escape(tailored.contact.location),
            "github": tex_escape(tailored.contact.github),
            "linkedin": tex_escape(tailored.contact.linkedin),
            "website": tex_escape(tailored.contact.website),
        },
        "work_history": tuple(_escape_work(w) for w in tailored.work_history),
        "education": tuple(_escape_edu(e) for e in tailored.education),
        "personal_projects": tuple(_escape_project(p) for p in tailored.personal_projects),
    }


def _escape_work(entry: TailoredWorkEntry) -> dict[str, Any]:
    return {
        "company": tex_escape(entry.company),
        "title": tex_escape(entry.title),
        "period": tex_escape(entry.period),
        "location": tex_escape(entry.location),
        "bullets": tuple({"text": tex_escape(b.text)} for b in entry.bullets),
    }


def _escape_edu(edu: Education) -> dict[str, Any]:
    return {
        "institution": tex_escape(edu.institution),
        "degree": tex_escape(edu.degree),
        "field": tex_escape(edu.field),
        "year_start": edu.year_start,
        "year_end": edu.year_end,
    }


def _escape_project(project: TailoredProject) -> dict[str, Any]:
    """Escape a project for the Jinja render.

    ``link`` is NOT tex-escaped because it's substituted into
    ``\\href{<link>}{...}`` where it's treated as a URL argument; LaTeX
    handles URL-specific quoting itself. Everything else flows through
    ``tex_escape`` because it lands in the typeset body.
    """
    return {
        "name": tex_escape(project.name),
        "description": tex_escape(project.description),
        "stack": tex_escape(project.stack),
        "link": project.link,
        "link_label": tex_escape(project.link_label),
        "bullets": tuple({"text": tex_escape(b.text)} for b in project.bullets),
    }
