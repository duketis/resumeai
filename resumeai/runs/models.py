"""Run + RunEvent + TailorRequest Pydantic models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from resumeai.agent.models import TailoredResume
from resumeai.jd.models import JobRequirements
from resumeai.renderer.models import RenderResult


class RunStatus(StrEnum):
    """Lifecycle states a run moves through. Strictly forward-progressing."""

    PENDING = "pending"
    FETCHING_JD = "fetching_jd"
    PARSING_JD = "parsing_jd"
    LOADING_CONTEXT = "loading_context"
    TAILORING = "tailoring"
    RENDERING = "rendering"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED)


class TailorRequest(BaseModel):
    """The body of ``POST /api/tailor``.

    Either ``jd_url`` (we'll fetch it) or ``jd_text`` (paste-in) must be
    supplied. ``template_doc_id`` is optional — when omitted we use the
    template registered in Settings.
    """

    model_config = ConfigDict(frozen=True)

    jd_url: str | None = None
    jd_text: str | None = None
    template_doc_id: str | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _exactly_one_jd_source(self) -> Self:
        has_url = bool(self.jd_url and self.jd_url.strip())
        has_text = bool(self.jd_text and self.jd_text.strip())
        if has_url == has_text:
            raise ValueError("supply exactly one of jd_url or jd_text")
        return self


class RunEvent(BaseModel):
    """One progress event. Published as the run moves through the pipeline."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    status: RunStatus
    detail: str = ""
    at: datetime


class Run(BaseModel):
    """Persisted run state. The ``GET /api/runs/{id}`` payload."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    request: TailorRequest
    status: RunStatus = RunStatus.PENDING
    created_at: datetime
    updated_at: datetime
    detail: str = ""
    error: str | None = None
    requirements: JobRequirements | None = None
    tailored: TailoredResume | None = None
    result: RenderResult | None = None
