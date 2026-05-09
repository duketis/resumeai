"""Render a :class:`~resumeai.agent.models.TailoredResume` into Google Docs.

Pipeline:

1. Copy the master template via Drive ``files.copy`` (master is sacred —
   never written to). The copy gets a sensible default title.
2. For each section we know how to render (summary, skills, work history,
   education, certifications):
   - Re-read the doc to get fresh structural indices (each per-section
     batch shifts indices for everything after it).
   - Find the heading by name (case-insensitive alias match).
   - Issue a ``documents.batchUpdate`` that deletes the existing content
     under the heading and inserts the tailored content. The batch is
     atomic — a botched section can't trash the whole doc.
3. Export the final doc to PDF and return a
   :class:`~resumeai.renderer.models.RenderResult`.

Phase 5 v1 inserts content as plain text; the inserted paragraphs inherit
the style at the insertion point. Promoting to per-paragraph in-place text
swaps (which preserves bullet styling) is a v0.6.x follow-up.
"""

from __future__ import annotations
