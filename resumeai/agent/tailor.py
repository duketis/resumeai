"""Tailoring orchestrator.

One call: build the prompt, hit the LLM, parse the response. The optional
``context_files`` argument carries user-uploaded supplementary context
(PDFs, CSVs, text files) that the agent considers alongside the
structured ``UserContext``.

If the first response fails to parse (LLMs occasionally emit malformed
JSON for long outputs -- truncated string, mid-string newline,
duplicated key), we retry ONCE with the parse error attached as
feedback. Bigger retry counts don't help: if the second attempt also
fails, the bug is upstream (prompt size, model, etc.) and we surface
the parse error rather than burning more tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from resumeai.agent.models import TailoredResume
from resumeai.agent.parser import AgentParseError, parse_tailored_resume
from resumeai.agent.prompt import SYSTEM_PROMPT, build_user_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from resumeai.context.models import UserContext
    from resumeai.context_files.models import ContextFile
    from resumeai.jd.models import JobRequirements
    from resumeai.llm.client import LLMClient


def tailor_resume(
    jd: JobRequirements,
    context: UserContext,
    llm: LLMClient,
    *,
    model: str | None = None,
    context_files: Sequence[ContextFile] = (),
) -> TailoredResume:
    """Run a full tailoring pass and return the validated :class:`TailoredResume`.

    Retries ONCE on parse failure with the error message fed back to the
    LLM. Re-parse failure is surfaced as the original ``AgentParseError``.
    """
    user_prompt = build_user_prompt(jd, context, context_files=context_files)
    response = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, model=model)
    try:
        return parse_tailored_resume(response)
    except AgentParseError as exc:
        retry_prompt = (
            f"{user_prompt}\n\n"
            "# RETRY\n"
            "Your previous response failed to parse with the error below. "
            "Re-emit the JSON object only -- no markdown fences, no commentary, "
            "no preamble. Make sure every string is properly quote-escaped, "
            "every key is double-quoted, and there are no trailing commas.\n\n"
            f"Parse error: {exc}"
        )
        retry_response = llm.complete(system=SYSTEM_PROMPT, user=retry_prompt, model=model)
        return parse_tailored_resume(retry_response)
