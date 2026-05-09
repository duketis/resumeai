"""Parse the LLM's response into a validated :class:`TailoredResume`.

Mirrors the JD parser's defensive shape: tolerate ```json fences, raise
:class:`AgentParseError` with the offending payload on anything malformed.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from resumeai.agent.models import TailoredResume


class AgentParseError(RuntimeError):
    """Raised when the agent's text response can't be parsed into TailoredResume."""


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def parse_tailored_resume(raw: str) -> TailoredResume:
    """Parse the model's response into a validated :class:`TailoredResume`."""
    text = raw.strip()
    if not text:
        raise AgentParseError("LLM returned an empty response")

    payload = _FENCE_RE.sub(r"\1", text).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AgentParseError(
            f"LLM response was not valid JSON: {exc.msg} — got {payload[:200]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise AgentParseError(f"LLM response was not a JSON object — got {type(data).__name__}")

    try:
        return TailoredResume.model_validate(data)
    except ValidationError as exc:
        raise AgentParseError(f"LLM response failed schema validation: {exc}") from exc
