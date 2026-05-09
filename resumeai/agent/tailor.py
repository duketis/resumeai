"""Tailoring orchestrator.

One call: build the prompt, hit the LLM, parse the response. Phase 6's
API will wrap this in an SSE handler that surfaces the in-flight LLM
output to the React UI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from resumeai.agent.models import TailoredResume
from resumeai.agent.parser import parse_tailored_resume
from resumeai.agent.prompt import SYSTEM_PROMPT, build_user_prompt

if TYPE_CHECKING:
    from resumeai.context.models import UserContext
    from resumeai.jd.models import JobRequirements
    from resumeai.llm.client import LLMClient


def tailor_resume(
    jd: JobRequirements,
    context: UserContext,
    llm: LLMClient,
    *,
    model: str | None = None,
) -> TailoredResume:
    """Run a full tailoring pass and return the validated :class:`TailoredResume`."""
    response = llm.complete(
        system=SYSTEM_PROMPT,
        user=build_user_prompt(jd, context),
        model=model,
    )
    return parse_tailored_resume(response)
