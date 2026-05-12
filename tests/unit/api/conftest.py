"""Shared fixtures for API route tests.

After the LaTeX pivot the Google OAuth + Docs fakes are gone; what remains
is the minimal kit: an in-memory settings store, runs store, context-file
store, fake LLM, an orchestrator wired to them, and a FastAPI TestClient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from tailor_core.llm.client import FakeLLMClient

from resumeai.api.app import create_app
from resumeai.context_files.store import InMemoryContextFileStore
from resumeai.runs.orchestrator import TailoringOrchestrator
from resumeai.runs.store import InMemoryRunsStore
from resumeai.settings.store import InMemorySettingsStore

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def store() -> InMemorySettingsStore:
    return InMemorySettingsStore()


@pytest.fixture
def runs() -> InMemoryRunsStore:
    return InMemoryRunsStore()


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient(default_response="{}")


@pytest.fixture
def context_files() -> InMemoryContextFileStore:
    return InMemoryContextFileStore()


@pytest.fixture
def orchestrator(
    store: InMemorySettingsStore,
    runs: InMemoryRunsStore,
    llm: FakeLLMClient,
    context_files: InMemoryContextFileStore,
    tmp_path: pytest.TempPathFactory,
) -> TailoringOrchestrator:
    return TailoringOrchestrator(
        runs_store=runs,
        settings_store=store,
        llm_client=llm,
        context_file_store=context_files,
        runs_root=tmp_path,  # type: ignore[arg-type]
    )


@pytest.fixture
def client(
    store: InMemorySettingsStore,
    runs: InMemoryRunsStore,
    llm: FakeLLMClient,
    orchestrator: TailoringOrchestrator,
    context_files: InMemoryContextFileStore,
) -> Iterator[TestClient]:
    app = create_app(
        settings_store=store,
        runs_store=runs,
        llm_client=llm,
        orchestrator=orchestrator,
        context_file_store=context_files,
    )
    with TestClient(app) as test_client:
        yield test_client
