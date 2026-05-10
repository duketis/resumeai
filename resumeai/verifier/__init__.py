"""Post-render verification — content + (eventually) vision review.

After the renderer writes the tailored resume to a fresh Google Doc + PDF,
the verifier runs a second LLM pass that checks the output against the JD
requirements and the user's collective context: did the agent actually
address the must-haves? did it fabricate? are bullets coherent? does the
content fit the JD's seniority? It returns a structured
:class:`VerificationResult` the run page surfaces.

v1 ships textual verification (LLM reviews the ``TailoredResume`` JSON
plus the rendered PDF's *text*). Vision-based verification (LLM reads
the rendered PDF's pixels for layout / formatting issues) lands in a
follow-up — see ``_private/BUILD_PLAN.md``.
"""

from __future__ import annotations
