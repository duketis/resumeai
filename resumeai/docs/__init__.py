"""Google Docs / Drive integration.

Three pieces:

- :mod:`~resumeai.docs.client` — the typed wrapper around
  ``googleapiclient.discovery``. Hidden behind a ``DocsClient`` ``Protocol``
  so the reader, copy helper, and (later) renderer never see Google types.
- :mod:`~resumeai.docs.reader` — turns a raw ``documents.get`` response into
  a :class:`~resumeai.docs.models.TemplateModel` (sections inferred from
  ``namedStyleType``).
- :mod:`~resumeai.docs.copy` — copies a master template to a new doc via
  Drive's ``files.copy``. Refuses to call any in-place mutation API
  against a registered master.
"""

from __future__ import annotations
