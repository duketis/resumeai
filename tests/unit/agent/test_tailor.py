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


def test_tailor_resume_propagates_parse_errors(
    sample_jd: JobRequirements,
    sample_context: UserContext,
) -> None:
    llm = FakeLLMClient(default_response="not valid json")

    with pytest.raises(AgentParseError):
        tailor_resume(sample_jd, sample_context, llm)
