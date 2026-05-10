"""Scan a local project directory into a plain-text summary.

What the scanner returns (in order):

1. **PROJECT** — name + absolute path.
2. **README** — first ``README.md`` / ``README.rst`` / ``README.txt`` it
   finds at the top level, truncated to ``readme_max_chars``.
3. **STRUCTURE** — non-secret top-level files and immediate subdirs
   (skipping the privacy-sensitive list).
4. **GIT LOG** — when ``.git/`` is present and ``git`` is on PATH, runs
   ``git log`` filtered to ``author_email`` (when supplied) and
   summarises: total commits, date range, recent commit subjects, and
   per-month commit counts. Capped at ``git_max_chars``.

Each section is best-effort and returns its placeholder text on failure
so a single broken scan never blocks the whole context payload.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_README_MAX_CHARS = 8000
DEFAULT_GIT_MAX_CHARS = 12000
DEFAULT_GIT_MAX_COMMITS = 1000

_README_NAMES: tuple[str, ...] = (
    "README.md",
    "README.rst",
    "README.txt",
    "Readme.md",
    "readme.md",
    "README",
)

# Top-level entries we *don't* show in the structure listing OR descend into.
_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        ".cache",
        ".idea",
        ".vscode",
        ".DS_Store",
    }
)

_SKIP_FILE_PREFIXES: tuple[str, ...] = (".env",)
_SKIP_FILE_SUBSTRINGS: tuple[str, ...] = ("credentials", "secret", "client_secret")


class ScanError(RuntimeError):
    """Raised when the path can't be scanned (missing dir, permission, etc.)."""


def scan_project(
    path: str | Path,
    *,
    name: str | None = None,
    author_email: str | None = None,
    readme_max_chars: int = DEFAULT_README_MAX_CHARS,
    git_max_chars: int = DEFAULT_GIT_MAX_CHARS,
    git_max_commits: int = DEFAULT_GIT_MAX_COMMITS,
) -> str:
    """Scan ``path`` and return the assembled plain-text summary."""
    project_path = Path(path).expanduser().resolve()
    if not project_path.is_dir():
        raise ScanError(f"{project_path} is not a directory")

    project_name = name or project_path.name
    parts: list[str] = [
        f"PROJECT: {project_name}",
        f"PATH: {project_path}",
        "",
        _readme_section(project_path, readme_max_chars),
        "",
        _structure_section(project_path),
        "",
        _git_section(project_path, author_email, git_max_chars, git_max_commits),
    ]
    return "\n".join(parts).strip() + "\n"


# -- README -----------------------------------------------------------------


def _readme_section(project_path: Path, max_chars: int) -> str:
    for name in _README_NAMES:
        candidate = project_path / name
        if candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return f"## README ({name})\n_(could not read: {exc})_"
            if len(content) > max_chars:
                content = (
                    content[:max_chars] + f"\n\n…({len(content) - max_chars} more chars truncated)"
                )
            return f"## README ({name})\n{content.strip()}"
    return "## README\n_(no README found at top level)_"


# -- Structure --------------------------------------------------------------


def _structure_section(project_path: Path) -> str:
    """Top-level + one level of subdirs. Skips privacy-sensitive entries."""
    try:
        entries = sorted(project_path.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        return f"## STRUCTURE\n_(could not list: {exc})_"

    lines: list[str] = ["## STRUCTURE"]
    for entry in entries:
        if _should_skip(entry):
            continue
        if entry.is_dir():
            lines.append(f"- {entry.name}/")
            try:
                children = sorted(entry.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                continue
            shown = 0
            for child in children:
                if _should_skip(child):
                    continue
                suffix = "/" if child.is_dir() else ""
                lines.append(f"  - {child.name}{suffix}")
                shown += 1
                if shown >= 20:
                    lines.append(f"  - …({len(children) - 20} more)")
                    break
        else:
            lines.append(f"- {entry.name}")
    if len(lines) == 1:
        lines.append("_(no non-skipped entries at top level)_")
    return "\n".join(lines)


def _should_skip(entry: Path) -> bool:
    name = entry.name
    if name in _SKIP_DIR_NAMES:
        return True
    lowered = name.lower()
    if any(lowered.startswith(prefix) for prefix in _SKIP_FILE_PREFIXES):
        return True
    return any(token in lowered for token in _SKIP_FILE_SUBSTRINGS)


# -- Git log ---------------------------------------------------------------


def _git_section(
    project_path: Path,
    author_email: str | None,
    max_chars: int,
    max_commits: int,
) -> str:
    if not (project_path / ".git").exists():
        return "## GIT LOG\n_(not a git repository)_"
    git_bin = shutil.which("git")
    if not git_bin:
        return "## GIT LOG\n_(git binary not found on PATH)_"

    raw_log = _run_git_log(git_bin, project_path, author_email, max_commits)
    if raw_log.startswith("## GIT LOG\n_("):
        return raw_log  # error-text placeholder
    lines = [line for line in raw_log.splitlines() if line.strip()]
    if not lines:
        scope = f" by {author_email}" if author_email else " in this repo"
        return f"## GIT LOG\n_(no commits{scope})_"

    timestamps, months, subjects = _aggregate_git_lines(lines)
    if not timestamps:
        scope = f" by {author_email}" if author_email else " in this repo"
        return f"## GIT LOG\n_(no commits{scope})_"

    section = _format_git_summary(timestamps, months, subjects, author_email)
    if len(section) > max_chars:
        section = section[:max_chars] + f"\n\n…({len(section) - max_chars} more chars truncated)"
    return section


def _run_git_log(
    git_bin: str, project_path: Path, author_email: str | None, max_commits: int
) -> str:
    cmd = [
        git_bin,
        "-C",
        str(project_path),
        "log",
        f"--max-count={max_commits}",
        "--pretty=format:%ai|%H|%s",
    ]
    if author_email:
        cmd.append(f"--author={author_email}")
    try:
        proc = subprocess.run(  # noqa: S603 — args are a list, no shell expansion
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"## GIT LOG\n_(git log failed: {exc})_"
    if proc.returncode != 0:
        return f"## GIT LOG\n_(git log returned {proc.returncode}: {proc.stderr.strip()})_"
    return proc.stdout


def _aggregate_git_lines(
    lines: list[str],
) -> tuple[list[datetime], Counter[str], list[str]]:
    timestamps: list[datetime] = []
    months: Counter[str] = Counter()
    subjects: list[str] = []
    for raw in lines:
        try:
            iso, _sha, subject = raw.split("|", 2)
        except ValueError:
            continue
        try:
            ts = datetime.fromisoformat(iso.replace(" ", "T", 1))
        except ValueError:
            continue
        timestamps.append(ts)
        months[ts.strftime("%Y-%m")] += 1
        subjects.append(subject)
    return timestamps, months, subjects


def _format_git_summary(
    timestamps: list[datetime],
    months: Counter[str],
    subjects: list[str],
    author_email: str | None,
) -> str:
    section_lines: list[str] = ["## GIT LOG"]
    scope_label = f" (filter: --author={author_email})" if author_email else ""
    earliest = min(timestamps).date().isoformat()
    latest = max(timestamps).date().isoformat()
    section_lines.append(
        f"Total commits scanned: {len(timestamps)}{scope_label}. Date range: {earliest} → {latest}."
    )
    # ``months`` is always non-empty here — every parsed timestamp adds an
    # entry, and the caller short-circuits on no timestamps before we get
    # here.
    section_lines.append("")
    section_lines.append("**Per-month commit counts (most active months first):**")
    for month, count in months.most_common(12):
        section_lines.append(f"- {month}: {count}")
    section_lines.append("")
    section_lines.append("**Recent commit subjects (newest first):**")
    for subject in subjects[:80]:
        section_lines.append(f"- {subject}")
    return "\n".join(section_lines)
