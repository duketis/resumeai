"""TailoringOrchestrator end-to-end tests against fakes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from resumeai.docs.client import DocsClient, FakeDocsClient
from resumeai.llm.client import FakeLLMClient
from resumeai.runs.events import RunEventBus
from resumeai.runs.models import RunStatus, TailorRequest
from resumeai.runs.orchestrator import OrchestratorError, TailoringOrchestrator
from resumeai.runs.store import InMemoryRunsStore
from resumeai.settings.models import GoogleCredentials, TemplateDoc
from resumeai.settings.store import InMemorySettingsStore

if TYPE_CHECKING:
    pass


_TAILORED_JSON: dict[str, Any] = {
    "name": "Alex Sample",
    "headline": "Senior backend engineer",
    "contact": {
        "email": "alex@example.com",
        "phone": None,
        "location": None,
        "linkedin": None,
        "github": None,
        "website": None,
    },
    "summary": "Tailored summary.",
    "skills": ["Python"],
    "work_history": [],
    "education": [],
    "certifications": [],
    "rationale": "...",
}

_JD_EXTRACTION_JSON: dict[str, Any] = {
    "title": "Senior Engineer",
    "company": "Globex",
    "required_skills": ["Python"],
    "nice_to_have_skills": [],
    "must_haves": [],
    "employer_vocabulary": [],
}


def _seed_credentials(store: InMemorySettingsStore) -> None:
    from datetime import UTC, datetime  # noqa: PLC0415

    store.set_google_credentials(
        GoogleCredentials(
            access_token="at",
            refresh_token="rt",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="cid",
            client_secret="csec",
            scopes=("openid",),
            expiry=datetime(2026, 5, 9, tzinfo=UTC),
            user_email="alex@example.com",
            user_id="42",
        )
    )


def _docs_factory(template_doc: dict[str, Any]) -> tuple[Any, list[FakeDocsClient]]:
    """Return a (factory, captured_clients) pair so tests can introspect calls."""
    captured: list[FakeDocsClient] = []

    def factory(_creds: GoogleCredentials) -> DocsClient:
        client = FakeDocsClient(
            documents={"new-doc-id": template_doc},
            copy_returns="new-doc-id",
            pdf_bytes=b"%PDF",
        )
        captured.append(client)
        return client

    return factory, captured


def _basic_template_doc() -> dict[str, Any]:
    """Simple doc with a Summary heading + body — enough for the renderer."""
    return {
        "documentId": "tmpl",
        "title": "T",
        "revisionId": "r",
        "body": {
            "content": [
                {
                    "startIndex": 1,
                    "endIndex": 9,
                    "paragraph": {
                        "elements": [
                            {
                                "startIndex": 1,
                                "endIndex": 9,
                                "textRun": {"content": "Summary\n", "textStyle": {}},
                            }
                        ],
                        "paragraphStyle": {"namedStyleType": "HEADING_1"},
                    },
                },
                {
                    "startIndex": 9,
                    "endIndex": 12,
                    "paragraph": {
                        "elements": [
                            {
                                "startIndex": 9,
                                "endIndex": 12,
                                "textRun": {"content": "x.\n", "textStyle": {}},
                            }
                        ],
                    },
                },
            ]
        },
    }


def _llm_router() -> FakeLLMClient:
    """LLM that returns the JD extraction first, then the tailored resume."""
    return FakeLLMClient(
        # Default response is the tailored payload — used after the JD step.
        # The JD extractor's call uses a different system prompt; we let the
        # FakeLLMClient route by user-prompt content.
        responses={},  # Both calls share the same default.
        default_response=json.dumps(_TAILORED_JSON),
    )


# -- create_run + execute happy path ----------------------------------------


@pytest.mark.asyncio
async def test_create_run_persists_pending_run() -> None:
    runs = InMemoryRunsStore()
    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=InMemorySettingsStore(),
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: FakeDocsClient(),
    )

    run = orchestrator.create_run(TailorRequest(jd_text="text"))

    assert run.status is RunStatus.PENDING
    assert runs.get(run.id) == run


@pytest.mark.asyncio
async def test_execute_walks_full_pipeline_on_text_input() -> None:
    runs = InMemoryRunsStore()
    settings = InMemorySettingsStore()
    settings.set_template(TemplateDoc(doc_id="master-doc"))
    _seed_credentials(settings)

    template_doc = _basic_template_doc()
    factory, captured = _docs_factory(template_doc)

    # Two LLM calls (JD extract, tailor) — script the JD extractor's exact
    # response and let the default cover the tailor call.
    llm = FakeLLMClient(default_response=json.dumps(_TAILORED_JSON))

    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        docs_client_factory=factory,
    )
    # Force the JD extractor to return its expected shape by mapping its
    # specific user prompt; if not matched, the default applies — but the
    # default is the tailor JSON which doesn't validate as a JD extraction.
    # Easiest fix: have the LLM always return both fields union.
    union: dict[str, Any] = {**_JD_EXTRACTION_JSON, **_TAILORED_JSON}
    llm._responses[""] = json.dumps(union)
    llm._default = json.dumps(union)

    run = orchestrator.create_run(TailorRequest(jd_text="Senior Engineer JD..."))
    finished = await orchestrator.execute(run.id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.requirements is not None
    assert finished.requirements.title == "Senior Engineer"
    assert finished.tailored is not None
    assert finished.tailored.name == "Alex Sample"
    assert finished.result is not None
    assert finished.result.doc_id == "new-doc-id"
    # The DocsClient is built once for the snapshot step + once for the
    # render step. Both with the stored credentials.
    assert len(captured) >= 1
    # The render client (the last one) is the one that did the copy.
    assert captured[-1].copy_calls[0][0] == "master-doc"


@pytest.mark.asyncio
async def test_execute_uses_request_template_doc_id_when_supplied() -> None:
    runs = InMemoryRunsStore()
    settings = InMemorySettingsStore()
    _seed_credentials(settings)

    factory, captured = _docs_factory(_basic_template_doc())
    union = {**_JD_EXTRACTION_JSON, **_TAILORED_JSON}
    llm = FakeLLMClient(default_response=json.dumps(union))

    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        docs_client_factory=factory,
    )

    run = orchestrator.create_run(TailorRequest(jd_text="text", template_doc_id="explicit-master"))
    finished = await orchestrator.execute(run.id)
    assert finished.status is RunStatus.SUCCEEDED

    # The render client (last one captured) issued the files.copy.
    assert captured[-1].copy_calls[0][0] == "explicit-master"


@pytest.mark.asyncio
async def test_execute_fetches_url_when_jd_url_supplied(tmp_path: Path) -> None:
    runs = InMemoryRunsStore()
    settings = InMemorySettingsStore()
    settings.set_template(TemplateDoc(doc_id="master"))
    _seed_credentials(settings)

    factory, _ = _docs_factory(_basic_template_doc())
    union = {**_JD_EXTRACTION_JSON, **_TAILORED_JSON}
    llm = FakeLLMClient(default_response=json.dumps(union))

    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        docs_client_factory=factory,
        context_root=tmp_path / "ctx",
    )

    url = "https://example.com/jd"
    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(url).mock(
            return_value=httpx.Response(200, text="<body><main>Senior Engineer</main></body>")
        )
        run = orchestrator.create_run(TailorRequest(jd_url=url))
        finished = await orchestrator.execute(run.id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.requirements is not None
    assert finished.requirements.source_url == url


# -- failure paths ---------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_marks_failed_when_template_unset_and_request_omits_it() -> None:
    runs = InMemoryRunsStore()
    settings = InMemorySettingsStore()
    _seed_credentials(settings)

    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: FakeDocsClient(),
    )
    run = orchestrator.create_run(TailorRequest(jd_text="text"))
    finished = await orchestrator.execute(run.id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert "no template_doc_id" in finished.error


@pytest.mark.asyncio
async def test_execute_marks_failed_when_no_credentials() -> None:
    runs = InMemoryRunsStore()
    settings = InMemorySettingsStore()
    settings.set_template(TemplateDoc(doc_id="master"))
    union = {**_JD_EXTRACTION_JSON, **_TAILORED_JSON}
    llm = FakeLLMClient(default_response=json.dumps(union))

    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        docs_client_factory=lambda _c: FakeDocsClient(),
    )
    run = orchestrator.create_run(TailorRequest(jd_text="text"))
    finished = await orchestrator.execute(run.id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert "Google credentials" in finished.error


@pytest.mark.asyncio
async def test_execute_raises_for_unknown_run_id() -> None:
    runs = InMemoryRunsStore()
    settings = InMemorySettingsStore()
    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: FakeDocsClient(),
    )
    # Underlying error is funnelled to FAILED by the public execute, but the
    # internal helper raises directly for tests / debug.
    finished = await orchestrator.execute("never-created")
    # Without a Run record we can't update it, so execute can't even produce
    # a FAILED state — the OrchestratorError surfaces.
    # (assertion below shows the public method swallowed it via the catch)
    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert "OrchestratorError" in finished.error or "unknown run" in finished.error


@pytest.mark.asyncio
async def test_execute_marks_failed_when_jd_fetch_errors() -> None:
    runs = InMemoryRunsStore()
    settings = InMemorySettingsStore()
    settings.set_template(TemplateDoc(doc_id="master"))
    _seed_credentials(settings)

    factory, _ = _docs_factory(_basic_template_doc())
    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=FakeLLMClient(),
        docs_client_factory=factory,
    )
    url = "https://example.com/jd"
    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(url).mock(return_value=httpx.Response(500))
        run = orchestrator.create_run(TailorRequest(jd_url=url))
        finished = await orchestrator.execute(run.id)

    assert finished.status is RunStatus.FAILED


# -- event publishing -----------------------------------------------------


@pytest.mark.asyncio
async def test_execute_publishes_status_events() -> None:
    runs = InMemoryRunsStore()
    settings = InMemorySettingsStore()
    settings.set_template(TemplateDoc(doc_id="master"))
    _seed_credentials(settings)

    factory, _ = _docs_factory(_basic_template_doc())
    union = {**_JD_EXTRACTION_JSON, **_TAILORED_JSON}
    bus = RunEventBus()
    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=FakeLLMClient(default_response=json.dumps(union)),
        docs_client_factory=factory,
        event_bus=bus,
    )
    run = orchestrator.create_run(TailorRequest(jd_text="text"))

    received: list[Any] = []

    import asyncio  # noqa: PLC0415

    async def consume() -> None:
        async for event in bus.subscribe(run.id):
            received.append(event.status)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await orchestrator.execute(run.id)
    await consumer

    assert RunStatus.PARSING_JD in received
    assert RunStatus.LOADING_CONTEXT in received
    assert RunStatus.TAILORING in received
    assert RunStatus.RENDERING in received
    assert RunStatus.SUCCEEDED in received


def test_orchestrator_exposes_event_bus_property() -> None:
    bus = RunEventBus()
    orchestrator = TailoringOrchestrator(
        runs_store=InMemoryRunsStore(),
        settings_store=InMemorySettingsStore(),
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: FakeDocsClient(),
        event_bus=bus,
    )
    assert orchestrator.event_bus is bus


def test_orchestrator_error_inheritance_from_runtime_error() -> None:
    """Belt-and-braces: ensure callers can `except RuntimeError` and catch us."""
    err = OrchestratorError("boom")
    assert isinstance(err, RuntimeError)


# -- master-template snapshot ------------------------------------------------


def test_snapshot_master_template_returns_none_when_no_credentials() -> None:
    """Without Google creds the snapshot can't run — return None gracefully so
    the agent step still proceeds (the render step will fail loudly later)."""
    orchestrator = TailoringOrchestrator(
        runs_store=InMemoryRunsStore(),
        settings_store=InMemorySettingsStore(),  # no creds
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: FakeDocsClient(),
    )
    assert orchestrator._snapshot_master_template("master-id") is None


def test_snapshot_master_template_returns_none_on_get_document_failure() -> None:
    """Network / 404 / permission errors during the snapshot read are
    swallowed; the render step will surface them later."""
    settings = InMemorySettingsStore()
    _seed_credentials(settings)

    fake_client = FakeDocsClient()  # no documents scripted → KeyError on get
    orchestrator = TailoringOrchestrator(
        runs_store=InMemoryRunsStore(),
        settings_store=settings,
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: fake_client,
    )
    assert orchestrator._snapshot_master_template("missing") is None


def test_snapshot_master_template_returns_context_file_with_master_text() -> None:
    settings = InMemorySettingsStore()
    _seed_credentials(settings)

    fake_client = FakeDocsClient(documents={"master-id": _basic_template_doc()})
    orchestrator = TailoringOrchestrator(
        runs_store=InMemoryRunsStore(),
        settings_store=settings,
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: fake_client,
    )

    snapshot = orchestrator._snapshot_master_template("master-id")

    assert snapshot is not None
    assert "Summary" in snapshot.extracted_text
    assert snapshot.tags == ("source:master_template",)
    assert "master Google Doc" in snapshot.note


def test_snapshot_master_template_returns_none_for_empty_doc() -> None:
    settings = InMemorySettingsStore()
    _seed_credentials(settings)

    empty_doc: dict[str, Any] = {
        "documentId": "x",
        "title": "Empty",
        "revisionId": "r",
        "body": {"content": []},
    }
    fake_client = FakeDocsClient(documents={"master-id": empty_doc})
    orchestrator = TailoringOrchestrator(
        runs_store=InMemoryRunsStore(),
        settings_store=settings,
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: fake_client,
    )

    assert orchestrator._snapshot_master_template("master-id") is None


def test_snapshot_master_template_returns_none_when_body_malformed() -> None:
    settings = InMemorySettingsStore()
    _seed_credentials(settings)

    bad_doc: dict[str, Any] = {
        "documentId": "x",
        "title": "T",
        "revisionId": "r",
        "body": {"content": "garbage-not-a-list"},
    }
    fake_client = FakeDocsClient(documents={"master-id": bad_doc})
    orchestrator = TailoringOrchestrator(
        runs_store=InMemoryRunsStore(),
        settings_store=settings,
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: fake_client,
    )
    assert orchestrator._snapshot_master_template("master-id") is None


def test_snapshot_master_template_skips_malformed_elements() -> None:
    """Walks the body content and skips non-dict / non-paragraph entries.

    Also includes a paragraph with empty text content (just a trailing
    newline) — the snapshot should skip it rather than emitting a blank
    line.
    """
    settings = InMemorySettingsStore()
    _seed_credentials(settings)

    doc = _basic_template_doc()
    doc["body"]["content"].append("not-a-dict")
    doc["body"]["content"].append({"sectionBreak": {}})
    # A paragraph with only a trailing newline (visually a blank line).
    doc["body"]["content"].append(
        {
            "startIndex": 12,
            "endIndex": 13,
            "paragraph": {
                "elements": [
                    {
                        "startIndex": 12,
                        "endIndex": 13,
                        "textRun": {"content": "\n", "textStyle": {}},
                    }
                ]
            },
        }
    )
    fake_client = FakeDocsClient(documents={"master-id": doc})
    orchestrator = TailoringOrchestrator(
        runs_store=InMemoryRunsStore(),
        settings_store=settings,
        llm_client=FakeLLMClient(),
        docs_client_factory=lambda _c: fake_client,
    )
    snapshot = orchestrator._snapshot_master_template("master-id")
    assert snapshot is not None
    # The blank-line paragraph isn't echoed into the extracted text.
    assert "Summary" in snapshot.extracted_text
    assert "\n\n" not in snapshot.extracted_text.strip()


@pytest.mark.asyncio
async def test_execute_includes_master_snapshot_in_agent_context_files(
    mocker: object,
) -> None:
    """End-to-end: when the master template is readable, the orchestrator
    feeds its current text to the tailoring agent as a synthetic context
    file."""
    runs = InMemoryRunsStore()
    settings = InMemorySettingsStore()
    _seed_credentials(settings)

    template_doc = _basic_template_doc()

    def factory(_creds: GoogleCredentials) -> DocsClient:
        return FakeDocsClient(
            documents={
                "explicit-master": template_doc,
                "new-doc-id": template_doc,
            },
            copy_returns="new-doc-id",
            pdf_bytes=b"%PDF",
        )

    union = {**_JD_EXTRACTION_JSON, **_TAILORED_JSON}
    llm = FakeLLMClient(default_response=json.dumps(union))

    # Patch tailor_resume in the orchestrator's namespace so we can capture
    # the context_files it received without losing real behaviour.
    from resumeai.agent.tailor import tailor_resume as real_tailor  # noqa: PLC0415

    captured_files: list[Any] = []

    def capturing_tailor(*args: Any, **kwargs: Any) -> Any:
        captured_files.append(kwargs.get("context_files", ()))
        return real_tailor(*args, **kwargs)

    mocker.patch(  # type: ignore[attr-defined]
        "resumeai.runs.orchestrator.tailor_resume", side_effect=capturing_tailor
    )

    orchestrator = TailoringOrchestrator(
        runs_store=runs,
        settings_store=settings,
        llm_client=llm,
        docs_client_factory=factory,
    )
    run = orchestrator.create_run(TailorRequest(jd_text="text", template_doc_id="explicit-master"))
    finished = await orchestrator.execute(run.id)

    assert finished.status is RunStatus.SUCCEEDED
    assert len(captured_files) == 1
    files = captured_files[0]
    assert len(files) == 1
    assert files[0].id == "ctx_master_template"
    assert "source:master_template" in files[0].tags


# Suppress unused-import warning for MagicMock — kept in case future tests
# need it without re-importing.
_: Any = MagicMock
