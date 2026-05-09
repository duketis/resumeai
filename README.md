# resumeai

> AI-driven resume tailoring. Reads a job description, reads your full career context, and drives a Google Docs template into a tailored, beautifully formatted resume — without ever touching your master.

`resumeai` is the resume-tailoring half of a small ecosystem of personal AI tools by [Jonathan Duketis](https://github.com/duketis). Sibling project: [jobai](https://github.com/duketis/jobai), which finds Australian jobs and (later) calls into resumeai to produce a tailored resume per opportunity.

## What it does

Given a job description (text or URL) and a master Google Docs resume template, resumeai:

1. **Parses the JD** for skills, seniority, must-haves, and the vocabulary the employer uses.
2. **Loads your full career context** — resume facts, work history, git audit, prior cover letters — from a hot-reloadable local directory.
3. **Runs a Claude-powered tailoring agent** that selects, reorders, and rewrites the right experiences/bullets for the role.
4. **Copies your master template** in Google Docs (the original is never touched), then uses the Google Docs API to fill in the tailored content while preserving every bit of your original styling.
5. **Returns the new doc URL plus a PDF export** and a side-by-side diff of what changed and why.

## Status

**v0.8.0 — standalone end-to-end pipeline shipped.** Run `resumeai serve` and open `http://localhost:7842/` to:

1. Walk the **onboarding wizard** for the one-time Google Cloud OAuth client setup.
2. Connect your **Google account** + register your **master resume Google Doc** in Settings.
3. Hit **Tailor**, paste a JD URL or text, and watch the run page auto-refresh through `parsing_jd → loading_context → tailoring → rendering → succeeded`. The result links straight to a freshly-copied Google Doc — your master is never written to — plus a PDF and a per-section diff.

Under the hood: httpx fetches the JD; deterministic + `claude` CLI subprocess extracts a typed `JobRequirements`; a hot-reloadable user-context store loads your `resume.yaml` + `work_history/*.md` + `git_audit/*.md`; a single-shot tailoring agent emits a structured `TailoredResume`; the renderer copies your template via Drive `files.copy` and applies surgical per-section `documents.batchUpdate`s. Async orchestrator persists every step to SQLite + publishes events to an in-memory bus; SSE endpoint and server-rendered `meta refresh` page both consume it.

**452 tests, 100% line + branch coverage on 1672 statements, CI green.** Architecture decisions (ADR-001..007 with rejected alternatives) live in `_private/ARCHITECTURE.md`; phased build plan in `_private/BUILD_PLAN.md`. Designed standalone; the JSON API + SSE stream are the documented integration surface for any future jobai-side plug-in.

**Roadmap:**
- v1.0.0: README polish + walkthrough screenshots, tag.
- v0.8.x: per-paragraph in-place text swaps in the renderer (preserve bullet styling that the v1 plain-text insert path doesn't).
- v0.8.x: React SPA promotion when interactive editing or external-app embedding wants it (see ADR-007).

## Architecture (high level)

```
[Job Description (text or URL)]
        │
        ▼
[JD Parser] ── structured requirements (skills, seniority, must-haves, vocab)
        │
[Context Store] ── resume facts, work history, git audit, cover letters
        │           (hot-reloadable from a local UserContext/ directory)
        ▼
[Tailoring Agent]
   - spawns the `claude` CLI as a subprocess (Anthropic Max subscription)
   - exposes Google Docs and the context store as MCP servers
   - runs a tool loop: read template → produce tailored ResumeModel
        │
        ▼
[Google Docs Renderer]
   1. Copy master template doc → new tailored doc
   2. documents.batchUpdate to fill structured content
   3. Export to PDF
        │
        ▼
[Output] → new doc URL + PDF + diff
```

## Why these choices

Decision rationale, including alternatives considered and rejected, lives in `_private/ARCHITECTURE.md`. Headlines:

- **Google Docs API, not screen-control automation.** Reliable, programmatic, testable, callable as a service. Screen control is a brittle dead end at this scope.
- **Master-copy pattern, not in-place edits.** Your existing resume is the template; every tailoring run produces a fresh copy. The master is sacred.
- **`claude` CLI subprocess + MCP, not Anthropic SDK + API key.** Uses the Anthropic Max subscription instead of per-token API billing. MCP servers give the spawned agent a clean tool loop for Google Docs and the context store.
- **Separate repo from jobai, not a sub-package.** Each repo tells one story; clean HTTP boundary between them is the senior-engineering signal. Cross-referencing in READMEs.

## Engineering bar

Same bar as the sibling repos. Staff/lead-engineer-at-a-top-tier-company, no shortcuts:

- Python 3.12, `mypy --strict`, `ruff` strict, `from __future__ import annotations` everywhere
- TDD-first, 100% coverage on every change
- Conventional commits, granular history, GPG-signed
- CI from day one: ruff + mypy + pytest on every push
- Pre-commit hooks mirror CI locally

## Quality gate (run before every commit)

```bash
./Tools/quality-gate.sh
```

That sets up a scratch venv at `/tmp/resumeai-tools` on first run, then runs `ruff check`, `ruff format --check`, `mypy`, and `pytest` — the same set CI runs.

## Acknowledgements

Built with [Claude Code](https://claude.com/claude-code).
