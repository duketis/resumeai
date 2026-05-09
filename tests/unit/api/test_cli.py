"""CLI surface — version + serve."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from resumeai import __version__
from resumeai.cli import app

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_version_command_prints_package_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_serve_command_invokes_uvicorn(mocker: MockerFixture) -> None:
    run = mocker.patch("uvicorn.run")
    runner = CliRunner()

    # S104: binding 0.0.0.0 in production would be a finding; here we're
    # verifying CLI option plumbing only — the command never actually binds.
    result = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "9000"])  # noqa: S104

    assert result.exit_code == 0
    run.assert_called_once_with(
        "resumeai.api.app:create_app",
        factory=True,
        host="0.0.0.0",  # noqa: S104
        port=9000,
        reload=False,
    )


def test_serve_command_supports_reload_flag(mocker: MockerFixture) -> None:
    run = mocker.patch("uvicorn.run")
    runner = CliRunner()

    runner.invoke(app, ["serve", "--reload"])

    assert run.call_args.kwargs["reload"] is True
