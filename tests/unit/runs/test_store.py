"""SqliteRunsStore + InMemoryRunsStore behavioural tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from resumeai.runs.models import Run, RunStatus, TailorRequest
from resumeai.runs.store import (
    InMemoryRunsStore,
    RunsStore,
    SqliteRunsStore,
    update_run,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _make_run(*, run_id: str = "run_a", offset_minutes: int = 0) -> Run:
    when = datetime(2026, 5, 9, tzinfo=UTC) + timedelta(minutes=offset_minutes)
    return Run(
        id=run_id,
        request=TailorRequest(jd_text="text"),
        status=RunStatus.PENDING,
        created_at=when,
        updated_at=when,
    )


@pytest.fixture(params=["sqlite", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[RunsStore]:
    if request.param == "sqlite":
        s = SqliteRunsStore(db_path=tmp_path / "runs.db")
        try:
            yield s
        finally:
            s.close()
    else:
        yield InMemoryRunsStore()


def test_save_then_get_round_trips(store: RunsStore) -> None:
    run = _make_run()
    store.save(run)
    assert store.get(run.id) == run


def test_get_returns_none_for_unknown_id(store: RunsStore) -> None:
    assert store.get("never-saved") is None


def test_save_overwrites_on_second_save(store: RunsStore) -> None:
    store.save(_make_run())
    store.save(_make_run().model_copy(update={"status": RunStatus.SUCCEEDED}))
    loaded = store.get("run_a")
    assert loaded is not None
    assert loaded.status is RunStatus.SUCCEEDED


def test_list_recent_orders_by_updated_at_descending(store: RunsStore) -> None:
    older = _make_run(run_id="run_old", offset_minutes=0)
    newer = _make_run(run_id="run_new", offset_minutes=5)
    store.save(older)
    store.save(newer)

    listed = store.list_recent()
    assert [r.id for r in listed] == ["run_new", "run_old"]


def test_list_recent_respects_limit(store: RunsStore) -> None:
    for i in range(3):
        store.save(_make_run(run_id=f"run_{i}", offset_minutes=i))

    listed = store.list_recent(limit=2)
    assert len(listed) == 2


def test_list_recent_with_zero_limit_returns_empty(store: RunsStore) -> None:
    store.save(_make_run())
    assert store.list_recent(limit=0) == []


def test_list_recent_with_negative_limit_returns_empty(store: RunsStore) -> None:
    store.save(_make_run())
    assert store.list_recent(limit=-1) == []


# -- update_run helper ------------------------------------------------------


def test_update_run_replaces_supplied_fields(store: RunsStore) -> None:
    store.save(_make_run())
    updated = update_run(
        store,
        "run_a",
        status=RunStatus.TAILORING,
        detail="thinking",
    )
    assert updated.status is RunStatus.TAILORING
    assert updated.detail == "thinking"
    assert updated.updated_at > updated.created_at


def test_update_run_records_error(store: RunsStore) -> None:
    store.save(_make_run())
    updated = update_run(store, "run_a", error="boom")
    assert updated.error == "boom"


def test_update_run_raises_for_unknown_id(store: RunsStore) -> None:
    with pytest.raises(KeyError):
        update_run(store, "missing", status=RunStatus.SUCCEEDED)


def test_update_run_can_attach_jd_requirements_and_result(store: RunsStore) -> None:
    from resumeai.agent.models import TailoredResume  # noqa: PLC0415
    from resumeai.context.models import Contact  # noqa: PLC0415
    from resumeai.jd.models import JobRequirements  # noqa: PLC0415
    from resumeai.renderer.models import RenderResult  # noqa: PLC0415

    store.save(_make_run())
    requirements = JobRequirements(title="Engineer")
    tailored = TailoredResume(name="Alex", contact=Contact(email="a@example.com"))
    result = RenderResult(doc_id="d", doc_url="https://docs.google.com/document/d/d/edit")

    updated = update_run(
        store, "run_a", requirements=requirements, tailored=tailored, result=result
    )
    assert updated.requirements == requirements
    assert updated.tailored == tailored
    assert updated.result == result


# -- SQLite-specific persistence -------------------------------------------


def test_sqlite_store_is_usable_from_a_worker_thread(tmp_path: Path) -> None:
    """FastAPI dispatches sync route handlers to a threadpool — the
    connection must survive cross-thread access (``check_same_thread=False``).
    Regression test for a 500 on /runs and /runs/{id}."""
    import threading  # noqa: PLC0415

    db_path = tmp_path / "runs.db"
    store = SqliteRunsStore(db_path=db_path)
    try:
        store.save(_make_run())
        result: list[Run | None] = []

        def fetch_in_thread() -> None:
            result.append(store.get("run_a"))

        thread = threading.Thread(target=fetch_in_thread)
        thread.start()
        thread.join(timeout=2)

        assert result and result[0] is not None
    finally:
        store.close()


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    a = SqliteRunsStore(db_path=db_path)
    a.save(_make_run())
    a.close()

    b = SqliteRunsStore(db_path=db_path)
    try:
        loaded = b.get("run_a")
        assert loaded is not None
    finally:
        b.close()
