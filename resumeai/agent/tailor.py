"""Tailoring orchestrator.

One call: build the prompt, hit the LLM, parse the response. The optional
``context_files`` argument carries user-uploaded supplementary context
(PDFs, CSVs, text files) that the agent considers alongside the
structured ``UserContext``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from resumeai.agent.models import TailoredResume
from resumeai.agent.parser import parse_tailored_resume
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
    """Run a full tailoring pass and return the validated :class:`TailoredResume`."""
    response = llm.complete(
        system=SYSTEM_PROMPT,
        user=build_user_prompt(jd, context, context_files=context_files),
        model=model,
    )
    return parse_tailored_resume(response)
