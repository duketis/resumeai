"""Parser tests for the LLM's tailored-resume response."""

from __future__ import annotations

import json
from typing import Any

import pytest

from resumeai.agent.parser import AgentParseError, parse_tailored_resume


def test_parses_well_formed_response(well_formed_response: str) -> None:
    resume = parse_tailored_resume(well_formed_response)
    assert resume.name == "Alex Sample"
    assert len(resume.work_history) == 1
    assert resume.work_history[0].bullets[0].source_slug == "01-acme"


def test_parses_response_inside_json_fence(
    well_formed_response_dict: dict[str, Any],
) -> None:
    fenced = f"```json\n{json.dumps(well_formed_response_dict)}\n```"
    resume = parse_tailored_resume(fenced)
    assert resume.name == "Alex Sample"


def test_parses_response_inside_unlabelled_fence(
    well_formed_response_dict: dict[str, Any],
) -> None:
    fenced = f"```\n{json.dumps(well_formed_response_dict)}\n```"
    resume = parse_tailored_resume(fenced)
    assert resume.name == "Alex Sample"


def test_raises_on_empty_response() -> None:
    with pytest.raises(AgentParseError, match="empty"):
        parse_tailored_resume("")


def test_raises_on_whitespace_only_response() -> None:
    with pytest.raises(AgentParseError, match="empty"):
        parse_tailored_resume("   \n\n  ")


def test_raises_on_invalid_json() -> None:
    with pytest.raises(AgentParseError, match="not valid JSON"):
        parse_tailored_resume("{not json")


def test_raises_on_non_object_top_level() -> None:
    with pytest.raises(AgentParseError, match="not a JSON object"):
        parse_tailored_resume("[1, 2, 3]")


def test_raises_on_schema_violation(
    well_formed_response_dict: dict[str, Any],
) -> None:
    bad = dict(well_formed_response_dict)
    del bad["name"]
    with pytest.raises(AgentParseError, match="schema validation"):
        parse_tailored_resume(json.dumps(bad))
