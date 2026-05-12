"""LaTeX renderer: render a :class:`~resumeai.agent.models.TailoredResume` to PDF.

Pipeline:

1. Compose the .tex string by feeding the tailored resume into a Jinja2
   template. Every user-derived value passes through
   :func:`~resumeai.renderer.latex_renderer.tex_escape` first so LaTeX
   control characters can't break the document.
2. Invoke ``tectonic`` as a subprocess to compile the .tex to a PDF.
   Tectonic is a single-binary LaTeX engine; no external state needed.
3. Return a :class:`~tailor_core.runs.models.RenderResult` whose
   ``doc_url`` points at the rendered PDF on disk.
"""

from __future__ import annotations
