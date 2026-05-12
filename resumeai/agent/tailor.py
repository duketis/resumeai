"""Tailoring orchestrator.

One call: build the prompt, hit the LLM, parse the response, then
**post-process the agent's output** to enforce passthrough fields
(headline, contact) the LLM is not allowed to rewrite.

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

    from tailor_core.context.models import UserContext
    from tailor_core.context_files.models import ContextFile
    from tailor_core.jd.models import JobRequirements
    from tailor_core.llm.client import LLMClient


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
    After the LLM-side parse, :func:`_enforce_passthroughs` overwrites
    fields the agent is forbidden from rewriting (headline) with their
    verbatim values from the candidate's resume base.
    """
    user_prompt = build_user_prompt(jd, context, context_files=context_files)
    response = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, model=model)
    try:
        tailored = parse_tailored_resume(response)
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
        tailored = parse_tailored_resume(retry_response)
    return _enforce_passthroughs(tailored, context)


def _enforce_passthroughs(tailored: TailoredResume, context: UserContext) -> TailoredResume:
    """Overwrite LLM-emitted fields that MUST come from the candidate's base.

    The agent's prompt rules tell it to copy fields like ``headline``
    verbatim from the resume base. LLMs drift on this -- they keep
    "improving" the headline into a tech-stack list. Rather than fight
    that with ever-stronger prompt rules, we stomp the field after
    parse. The LLM has no say.

    Today this just covers ``headline``; future passthroughs (e.g. a
    locked contact line variation per template) would slot in here.
    """
    if context.resume is None or context.resume.headline is None:
        # No base headline to copy; respect whatever the agent emitted
        # (typically empty, per the schema rule).
        return tailored
    return tailored.model_copy(update={"headline": context.resume.headline})
