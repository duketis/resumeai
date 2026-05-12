"""Visual QC of the rendered resume PDF using a vision-capable LLM call.

The text verifier in :mod:`resumeai.verifier.verifier` can't see layout --
widow lines, awkward gaps, section-header orphaning, content
overflowing the right margin, density issues. This module rasterises
each page of the rendered PDF to PNG and hands the images to the
Anthropic messages API for a vision review.

Why a separate auth path?
The text-mode pipeline shells out to the ``claude`` CLI. The CLI
doesn't accept inline image bytes (``--file`` takes a pre-uploaded
Files-API ID). The Anthropic Python SDK does -- and it accepts the
same Max-plan OAuth token via ``CLAUDE_CODE_OAUTH_TOKEN``, so vision
works without an API key.

Failures degrade gracefully: a missing token, network error, SDK
exception, or malformed model response all produce an empty list of
issues rather than blocking the run. The user still gets the text
verifier's findings; visual ones are best-effort.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import TYPE_CHECKING

from anthropic.types import (
    ImageBlockParam,
    TextBlock,
    TextBlockParam,
)
from pydantic import ValidationError

from resumeai.verifier.models import (
    IssueSeverity,
    VerificationIssue,
    VerificationResult,
    VerificationStatus,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_log = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "claude-opus-4-7"
DEFAULT_VISION_MAX_TOKENS = 2000
# PDF pages render to PNG at this DPI; 150 is the sweet spot between
# small payloads and "Claude can actually read the text on the page".
DEFAULT_VISION_DPI = 150

SYSTEM_PROMPT = """\
You are a senior recruiter reviewing the VISUAL LAYOUT of a rendered
resume PDF. The text content is being QC'd separately -- your job is
PURELY visual: would this resume look professional to a hiring manager
who opens it in a PDF viewer?

Look for:

- **Orphan lines / widow lines**: a paragraph whose first or last line
  sits alone at the top or bottom of a different page from the rest
  of the paragraph. Bullets that wrap with only 1-3 words spilling to
  the next page are the most common case and look unprofessional.
- **Section header orphaning**: a section title at the bottom of one
  page with all its content on the next.
- **Uneven vertical gaps**: inconsistent spacing between equivalent
  list items (e.g. one bullet has noticeably more space below it than
  its siblings).
- **Right-margin overflow**: text or links that visibly press against
  or exceed the right margin (link labels cut off, lines extending
  past where the rest of the body ends).
- **Cramped or sparse density**: bullets so tightly packed they read
  as a wall, or so spread out that the page looks half-empty.
- **Headline / hierarchy issues**: section headings that aren't
  visually distinct from entry-name bold text, or contact-line items
  that wrap awkwardly.
- **Page count**: 3 pages is the target; flag if >= 4.

Respond with ONLY a single JSON object (no markdown fences, no
commentary, no preamble) matching this schema:

{
  "status": "passed" | "concerns" | "failed",
  "summary": "string -- one short sentence headline judgment",
  "issues": [
    {
      "severity": "info" | "warn" | "error",
      "category": "short tag (eg 'orphan_line', 'margin_overflow', 'section_orphan')",
      "message": "the specific issue, plain language. Cite the page number.",
      "suggestion": "string -- what to change (empty if obvious)"
    }
  ],
  "rationale": "string -- one paragraph on the major findings"
}

Status decision:
- ``failed``: any layout problem a recruiter would notice in the first
  5 seconds (page-3 of 3 has a single orphan line at top; section
  header on bottom of page 1 with content on page 2; visible right
  margin overflow on a link).
- ``concerns``: visually OK but tightenable (slightly uneven spacing,
  minor density issues).
- ``passed``: looks like a hand-formatted senior-eng resume.

Output the JSON and nothing else.
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class VisionVerifierError(RuntimeError):
    """Raised when the vision verifier can't complete its check."""


def verify_pdf_visually(
    pdf_path: Path,
    *,
    oauth_token: str | None = None,
    model: str = DEFAULT_VISION_MODEL,
    dpi: int = DEFAULT_VISION_DPI,
    max_tokens: int = DEFAULT_VISION_MAX_TOKENS,
) -> VerificationResult | None:
    """Render ``pdf_path`` to images, send to the Anthropic vision API,
    return a structured verification result.

    Returns ``None`` when the vision pass can't run -- missing OAuth
    token, missing SDK, network error, malformed response. Callers
    should treat ``None`` as "no visual signal" and proceed with the
    text-only verification result. Never raises -- all failure modes
    log a warning and degrade silently.
    """
    token = oauth_token or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        _log.info("vision verifier skipped: CLAUDE_CODE_OAUTH_TOKEN not set")
        return None

    try:
        pages = _render_pdf_to_pngs(pdf_path, dpi=dpi)
    except (OSError, VisionVerifierError) as exc:
        _log.warning("vision verifier skipped: PDF rasterisation failed: %s", exc)
        return None

    if not pages:
        _log.warning("vision verifier skipped: PDF has zero pages")
        return None

    try:
        raw = _call_vision_api(pages=pages, oauth_token=token, model=model, max_tokens=max_tokens)
    except VisionVerifierError as exc:
        _log.warning("vision verifier skipped: API call failed: %s", exc)
        return None

    try:
        return _parse_response(raw)
    except VisionVerifierError as exc:
        _log.warning("vision verifier skipped: response parse failed: %s", exc)
        return None


def _render_pdf_to_pngs(pdf_path: Path, *, dpi: int) -> list[bytes]:
    """Use pypdfium2 to rasterise every page of ``pdf_path`` to PNG bytes."""
    import pypdfium2 as pdfium  # noqa: PLC0415

    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:
        raise VisionVerifierError(f"could not open PDF: {exc}") from exc

    pages: list[bytes] = []
    # pypdfium2 uses 72 DPI as native; render scale = dpi / 72.
    scale = dpi / 72.0
    try:
        for page_idx in range(len(document)):
            page = document[page_idx]
            try:
                image = page.render(scale=scale).to_pil()
            except Exception as exc:
                raise VisionVerifierError(f"rasterising page {page_idx + 1} failed: {exc}") from exc
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            pages.append(buffer.getvalue())
    finally:
        document.close()
    return pages


def _call_vision_api(
    *,
    pages: Iterable[bytes],
    oauth_token: str,
    model: str,
    max_tokens: int,
) -> str:
    """Send the rendered pages to the Anthropic messages API. Returns raw text."""
    try:
        from anthropic import Anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise VisionVerifierError("anthropic SDK not installed") from exc

    # The SDK accepts an OAuth token via ``auth_token``. The Max-plan
    # OAuth token from ``claude setup-token`` works with this path.
    client = Anthropic(auth_token=oauth_token)

    content: list[ImageBlockParam | TextBlockParam] = []
    for png_bytes in pages:
        content.append(
            ImageBlockParam(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(png_bytes).decode("ascii"),
                },
            )
        )
    content.append(
        TextBlockParam(
            type="text",
            text=(
                "Above are the rendered pages of a candidate's tailored resume "
                "PDF, in order. Review for VISUAL layout issues per the system "
                "prompt schema. Cite the specific page number where each issue "
                "appears."
            ),
        )
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        raise VisionVerifierError(f"messages API call failed: {exc}") from exc

    text_chunks: list[str] = []
    for block in response.content:
        # Narrow with isinstance — the anthropic SDK's response content is a
        # union of many block types (text, thinking, tool-use, etc.) and a
        # ``getattr(...) == "text"`` check doesn't narrow the type for mypy.
        if isinstance(block, TextBlock):
            text_chunks.append(block.text)
    if not text_chunks:
        raise VisionVerifierError("response had no text blocks")
    return "\n".join(text_chunks)


def _parse_response(raw: str) -> VerificationResult:
    """Parse the model's text response into a :class:`VerificationResult`."""
    text = raw.strip()
    if not text:
        raise VisionVerifierError("response body was empty")

    payload = _FENCE_RE.sub(r"\1", text).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VisionVerifierError(
            f"response was not valid JSON: {exc.msg} — got {payload[:200]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise VisionVerifierError(f"response was not a JSON object — got {type(data).__name__}")

    try:
        return VerificationResult.model_validate(data)
    except ValidationError as exc:
        raise VisionVerifierError(f"response failed schema validation: {exc}") from exc


__all__ = [
    "DEFAULT_VISION_DPI",
    "DEFAULT_VISION_MAX_TOKENS",
    "DEFAULT_VISION_MODEL",
    "SYSTEM_PROMPT",
    "IssueSeverity",
    "VerificationIssue",
    "VerificationStatus",
    "VisionVerifierError",
    "verify_pdf_visually",
]
