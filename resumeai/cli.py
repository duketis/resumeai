"""``resumeai`` CLI entrypoint.

Right now the only command is ``serve``, which boots the FastAPI app on a
local port. Phase 6 will add ``tailor`` etc. as the agent surfaces ship.
"""

from __future__ import annotations

import typer
import uvicorn

from resumeai import __version__

app = typer.Typer(
    name="resumeai",
    help="AI-driven resume tailoring service.",
    add_completion=False,
)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host. localhost by default."),
    port: int = typer.Option(7842, help="Bind port. Default matches the OAuth redirect."),
    reload: bool = typer.Option(False, help="Enable autoreload (development only)."),
) -> None:
    """Boot the resumeai web app and Settings UI."""
    typer.echo(f"resumeai v{__version__} → http://{host}:{port}")
    uvicorn.run(
        "resumeai.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def version() -> None:
    """Print the resumeai version and exit."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
