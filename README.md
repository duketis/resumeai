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

**v0.3.0 — Phase 2 shipped.** Adds the JD ingestion + structured-extraction layer on top of the v0.2.0 Settings/OAuth/Docs surface. Given a JD URL or text, resumeai now produces a typed `JobRequirements` (title, company, location, role type, seniority, employment + remote type, required + nice-to-have skills, must-haves, employer vocabulary) by combining a single httpx fetch + HTML cleaner, deterministic regex extractors for the closed-enum fields, and a single `claude` CLI subprocess call for the open-ended fields. The end-to-end tailoring pipeline (context store → tailoring agent → renderer) lands in Phases 3–5; React frontend in Phase 7; jobai integration in Phase 8. **227 tests, 100% line + branch coverage, CI green.** See `_private/BUILD_PLAN.md` (gitignored — interview material) for the full plan.

Run `resumeai serve` and open `http://localhost:7842/` to follow the in-app onboarding wizard for the one-time Google Cloud OAuth client setup.

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
