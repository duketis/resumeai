"""Read a user-context tree from disk into :class:`UserContext`.

Layout the loader expects (rooted at any directory the caller picks):

- ``resume.yaml``               — :class:`ResumeBase` (optional)
- ``work_history/*.md``         — one :class:`WorkHistoryEntry` per file
- ``git_audit/*.md``            — one :class:`GitAuditEntry` per file
- ``cover_letters/*.md``        — one :class:`CoverLetterEntry` per file
- ``projects/*.md``             — one :class:`ProjectEntry` per file

Markdown files use YAML frontmatter for structured metadata + a markdown
body for narrative content. Bullets are extracted from the body where
relevant.

Missing files / directories are tolerated — a fresh user starts with an
empty context and fills it in over time. Malformed YAML / frontmatter
raises :class:`ContextLoadError` with a path so the user knows what to fix.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from resumeai.context.models import (
    CoverLetterEntry,
    GitAuditEntry,
    ProjectEntry,
    ResumeBase,
    UserContext,
    WorkHistoryEntry,
)


class ContextLoadError(RuntimeError):
    """Raised when a context file is present but malformed."""


_BULLET_RE = re.compile(r"^[-*+]\s+(.+?)\s*$", re.MULTILINE)


def load_user_context(root: Path) -> UserContext:
    """Materialise a :class:`UserContext` from a directory tree.

    Returns an empty :class:`UserContext` when ``root`` doesn't exist —
    callers (the store) handle the empty case.
    """
    if not root.is_dir():
        return UserContext()

    return UserContext(
        resume=_load_resume(root / "resume.yaml"),
        work_history=tuple(_load_work_history(root / "work_history")),
        git_audit=tuple(_load_git_audit(root / "git_audit")),
        cover_letters=tuple(_load_cover_letters(root / "cover_letters")),
        projects=tuple(_load_projects(root / "projects")),
        master_resume=_load_text_file(root / "master-resume.txt"),
        reference_resumes=tuple(_load_reference_resumes(root)),
    )


# -- resume.yaml -------------------------------------------------------------


def _load_resume(path: Path) -> ResumeBase | None:
    if not path.is_file():
        return None
    raw = _safe_yaml(path)
    if not isinstance(raw, dict):
        raise ContextLoadError(f"{path}: expected a YAML mapping at top level")
    try:
        return ResumeBase.model_validate(raw)
    except ValidationError as exc:
        raise ContextLoadError(f"{path}: {exc}") from exc


# -- work history ------------------------------------------------------------


def _load_work_history(directory: Path) -> list[WorkHistoryEntry]:
    entries: list[WorkHistoryEntry] = []
    for path in _markdown_files(directory):
        fm, body = _parse_frontmatter(path)
        # All required fields are validated by ``_required_str`` (raising
        # ContextLoadError with the path) before the model sees them, so a
        # ValidationError from this constructor is unreachable today.
        entries.append(
            WorkHistoryEntry(
                slug=path.stem,
                title=_required_str(fm, "title", path),
                company=_required_str(fm, "company", path),
                start=_optional_str(fm, "start"),
                end=_optional_str(fm, "end"),
                location=_optional_str(fm, "location"),
                technologies=_optional_tuple(fm, "technologies"),
                summary=_extract_summary(body),
                bullets=tuple(_BULLET_RE.findall(body)),
                raw_markdown=path.read_text(encoding="utf-8"),
            )
        )
    entries.sort(key=lambda e: (e.end or "9999", e.start or ""), reverse=True)
    return entries


# -- git audit ---------------------------------------------------------------


def _load_git_audit(directory: Path) -> list[GitAuditEntry]:
    entries: list[GitAuditEntry] = []
    for path in _markdown_files(directory):
        fm, body = _parse_frontmatter(path)
        entries.append(
            GitAuditEntry(
                slug=path.stem,
                repo=_required_str(fm, "repo", path),
                role=_optional_str(fm, "role"),
                period=_optional_str(fm, "period"),
                summary=_extract_summary(body),
                raw_markdown=path.read_text(encoding="utf-8"),
            )
        )
    entries.sort(key=lambda e: e.slug)
    return entries


# -- cover letters -----------------------------------------------------------


def _load_cover_letters(directory: Path) -> list[CoverLetterEntry]:
    entries: list[CoverLetterEntry] = []
    for path in _markdown_files(directory):
        fm, body = _parse_frontmatter(path)
        entries.append(
            CoverLetterEntry(
                slug=path.stem,
                role=_optional_str(fm, "role"),
                company=_optional_str(fm, "company"),
                body=body.strip(),
                raw_markdown=path.read_text(encoding="utf-8"),
            )
        )
    entries.sort(key=lambda e: e.slug)
    return entries


# -- master resume + reference resumes ---------------------------------------


def _load_text_file(path: Path) -> str:
    """Return the contents of ``path`` if it exists, else empty string."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _load_reference_resumes(root: Path) -> list[str]:
    """Return any ``Resume*.tex`` files in the root as verbatim strings.

    These are hand-tailored references the candidate has already produced
    and considers known-good. The agent reads them to anchor its own
    output in real phrasing rather than guessing.
    """
    return [
        path.read_text(encoding="utf-8").strip()
        for path in sorted(root.glob("Resume*.tex"))
        if path.is_file()
    ]


# -- projects ----------------------------------------------------------------


def _load_projects(directory: Path) -> list[ProjectEntry]:
    entries: list[ProjectEntry] = []
    for path in _markdown_files(directory):
        fm, body = _parse_frontmatter(path)
        entries.append(
            ProjectEntry(
                slug=path.stem,
                name=_required_str(fm, "project", path),
                url=_optional_str(fm, "url"),
                status=_optional_str(fm, "status"),
                stack=_optional_str(fm, "stack"),
                summary=_extract_summary(body),
                bullets=tuple(_BULLET_RE.findall(body)),
                body=body.strip(),
            )
        )
    entries.sort(key=lambda e: e.slug)
    return entries


# -- helpers -----------------------------------------------------------------


def _markdown_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".md")


def _safe_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContextLoadError(f"{path}: invalid YAML: {exc}") from exc


_FRONTMATTER_OPEN = "---\n"
_FRONTMATTER_CLOSE = "\n---\n"


def _parse_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_FRONTMATTER_OPEN):
        return {}, text

    end = text.find(_FRONTMATTER_CLOSE, len(_FRONTMATTER_OPEN))
    if end == -1:
        raise ContextLoadError(f"{path}: frontmatter opening `---` has no matching close")

    yaml_block = text[len(_FRONTMATTER_OPEN) : end]
    body = text[end + len(_FRONTMATTER_CLOSE) :]
    try:
        loaded = yaml.safe_load(yaml_block)
    except yaml.YAMLError as exc:
        raise ContextLoadError(f"{path}: invalid frontmatter YAML: {exc}") from exc

    if loaded is None:
        return {}, body
    if not isinstance(loaded, dict):
        raise ContextLoadError(f"{path}: frontmatter must be a YAML mapping")
    return loaded, body


def _required_str(fm: dict[str, Any], key: str, path: Path) -> str:
    value = fm.get(key)
    if not isinstance(value, str) or not value:
        raise ContextLoadError(f"{path}: missing required frontmatter string field {key!r}")
    return value


def _optional_str(fm: dict[str, Any], key: str) -> str | None:
    value = fm.get(key)
    if value is None:
        return None
    return str(value)


def _optional_tuple(fm: dict[str, Any], key: str) -> tuple[str, ...]:
    value = fm.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _extract_summary(body: str) -> str:
    """First non-bullet, non-empty paragraph after the frontmatter."""
    paragraphs = [p.strip() for p in body.split("\n\n")]
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if paragraph.lstrip().startswith(("-", "*", "+", "#")):
            continue
        return paragraph
    return ""
