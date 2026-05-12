"""Resume-specific orchestrator -- subclasses ``BaseOrchestrator``.

The pipeline skeleton (JD fetch + parse, context load, run lifecycle,
event publishing, error handling, vision verification dispatch) lives in
``tailor_core.runs.orchestrator.BaseOrchestrator``. This module supplies
the four hooks the base calls into:

- :meth:`_tailor` -- run the resume tailoring agent.
- :meth:`_render` -- compose a JD-flavoured filename stem and render via
  LaTeX + tectonic.
- :meth:`_verify` -- run the text-mode resume verifier.
- :meth:`_verify_visually` -- run the vision verifier (best-effort).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from tailor_core.runs.orchestrator import BaseOrchestrator, OrchestratorError

from resumeai.agent.models import TailoredResume
from resumeai.agent.tailor import tailor_resume
from resumeai.renderer.latex_renderer import render_tailored_resume_latex
from resumeai.settings.models import RuntimeSettings
from resumeai.verifier.verifier import verify_resume
from resumeai.verifier.vision import verify_pdf_visually

if TYPE_CHECKING:
    from tailor_core.context.models import UserContext
    from tailor_core.context_files.models import ContextFile
    from tailor_core.jd.models import JobRequirements
    from tailor_core.runs.models import RenderResult, TailorRequest
    from tailor_core.verifier.models import VerificationResult


__all__ = [
    "OrchestratorError",
    "TailoringOrchestrator",
]


class TailoringOrchestrator(BaseOrchestrator[TailoredResume, RuntimeSettings]):
    """Resume-tailoring concrete orchestrator."""

    def _tailor(
        self,
        requirements: JobRequirements,
        context: UserContext,
        request: TailorRequest,
        context_files: tuple[ContextFile, ...],
    ) -> TailoredResume:
        return tailor_resume(
            requirements,
            context,
            self._llm,
            model=request.model,
            context_files=context_files,
        )

    def _render(
        self,
        tailored: TailoredResume,
        requirements: JobRequirements,
        output_dir: Path,
    ) -> RenderResult:
        stem = _resume_filename_stem(tailored.name, requirements)
        return render_tailored_resume_latex(tailored, output_dir, stem=stem)

    def _verify(
        self,
        requirements: JobRequirements,
        tailored: TailoredResume,
        pdf_path: Path,
    ) -> VerificationResult:
        return verify_resume(requirements, tailored, self._llm, pdf_path=pdf_path)

    def _verify_visually(self, pdf_path: Path) -> VerificationResult | None:
        return verify_pdf_visually(pdf_path)


# Filesystem-safe characters: letters, digits, space, dash, underscore, parens,
# ampersand, comma, dot. Anything else (slashes, colons, smart quotes, control
# chars, etc.) gets stripped so Preview/Finder don't choke on the filename.
_FILENAME_SAFE_RE = re.compile(r"[^\w\s\-(),.&]+", re.UNICODE)
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def _resume_filename_stem(candidate_name: str, requirements: JobRequirements) -> str:
    """Build a per-JD filename stem like ``Jonathan Duketis - GoSource - Software Developer``.

    The user opens many PDFs in Preview side-by-side; identical ``resume.pdf``
    names from successive runs are impossible to tell apart. The stem
    incorporates whatever JD context survived parsing (company + title) so the
    Preview titlebar identifies the run at a glance. Falls back to
    ``"resume"`` when sanitisation strips every part to nothing.
    """
    parts = [
        _sanitize_for_filename(candidate_name),
        _sanitize_for_filename(requirements.company or ""),
        _sanitize_for_filename(requirements.title),
    ]
    parts = [p for p in parts if p]
    return " - ".join(parts) if parts else "resume"


def _sanitize_for_filename(raw: str) -> str:
    cleaned = _FILENAME_SAFE_RE.sub("", raw)
    return _WHITESPACE_RUN_RE.sub(" ", cleaned).strip(" -._")
