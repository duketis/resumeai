"""End-to-end tailoring orchestrator tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from resumeai.agent.parser import AgentParseError
from resumeai.agent.prompt import SYSTEM_PROMPT
from resumeai.agent.tailor import tailor_resume
from resumeai.llm.client import FakeLLMClient

if TYPE_CHECKING:
    from resumeai.context.models import UserContext
    from resumeai.jd.models import JobRequirements


def test_tailor_resume_returns_validated_payload(
    sample_jd: JobRequirements,
    sample_context: UserContext,
    well_formed_response: str,
) -> None:
    llm = FakeLLMClient(default_response=well_formed_response)

    resume = tailor_resume(sample_jd, sample_context, llm)

    assert resume.name == "Alex Sample"
    assert resume.skills[0] == "Python"
    assert resume.work_history[0].company == "Acme Corp"


def test_tailor_resume_passes_system_prompt_and_user_prompt(
    sample_jd: JobRequirements,
    sample_context: UserContext,
    well_formed_response: str,
) -> None:
    llm = FakeLLMClient(default_response=well_formed_response)

    tailor_resume(sample_jd, sample_context, llm)

    assert llm.calls
    system, user, model = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "Senior Backend Engineer" in user
    assert "Acme Corp" in user
    assert model is None


def test_tailor_resume_forwards_model_override(
    sample_jd: JobRequirements,
    sample_context: UserContext,
    well_formed_response: str,
) -> None:
    llm = FakeLLMClient(default_response=well_formed_response)

    tailor_resume(sample_jd, sample_context, llm, model="claude-sonnet-4-6")

    assert llm.calls[0][2] == "claude-sonnet-4-6"


def test_tailor_resume_propagates_parse_errors_after_one_retry(
    sample_jd: JobRequirements,
    sample_context: UserContext,
) -> None:
    """First and second attempts both fail -> the original error type bubbles."""
    llm = FakeLLMClient(default_response="not valid json")

    with pytest.raises(AgentParseError):
        tailor_resume(sample_jd, sample_context, llm)
    # Two LLM calls: original + one retry.
    assert len(llm.calls) == 2


def test_tailor_resume_retries_once_when_first_response_is_malformed_json(
    sample_jd: JobRequirements,
    sample_context: UserContext,
    well_formed_response: str,
) -> None:
    """First attempt returns garbage; retry returns clean JSON. Result parses."""
    bad_response = '{"name": "broken'  # truncated mid-string
    llm = FakeLLMClient(
        responses={},  # no exact-match prompt-keyed responses
        default_response=bad_response,
    )

    # We need the retry call (with "RETRY" in the prompt body) to return
    # the well-formed payload; first call (without "RETRY") returns the bad
    # one. Stub that branching by replacing default_response after first call.
    original_complete = llm.complete

    def alternating_complete(*, system: str, user: str, model: str | None = None) -> str:
        result = original_complete(system=system, user=user, model=model)
        # Swap in the good payload so the retry succeeds.
        llm._default = well_formed_response
        return result

    llm.complete = alternating_complete  # type: ignore[method-assign]

    resume = tailor_resume(sample_jd, sample_context, llm)
    assert resume.name == "Alex Sample"
    # Two calls: original + retry. The retry's user prompt must include
    # the "RETRY" header + the parse error feedback.
    assert len(llm.calls) == 2
    retry_user_prompt = llm.calls[1][1]
    assert "# RETRY" in retry_user_prompt
    assert "previous response failed to parse" in retry_user_prompt
