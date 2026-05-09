"""Job-description ingestion + structured extraction.

Public surface:

- :func:`~resumeai.jd.fetcher.fetch_jd` — pull HTML from a URL and return a
  :class:`~resumeai.jd.models.FetchedJD` (raw HTML + cleaned plain text).
- :func:`~resumeai.jd.parser.parse_jd_text` — turn cleaned plain text into a
  :class:`~resumeai.jd.models.JobRequirements` via deterministic regex passes
  + a single LLM call for the open-ended fields.
- :func:`~resumeai.jd.parser.parse_jd_url` — convenience that does fetch +
  parse in one go.
"""

from __future__ import annotations
