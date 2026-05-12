# resumeai

> AI-driven resume tailoring. Reads a job description, reads your full career context, then drives a Jinja2 + LaTeX template through [Tectonic](https://tectonic-typesetting.github.io/) to produce a tailored, beautifully formatted PDF — without ever touching your master.

`resumeai` is the resume-tailoring service in a small family of personal AI tools by [Jonathan Duketis](https://github.com/duketis).

- Sibling: [coverletterai](https://github.com/duketis/coverletterai) — the cover-letter counterpart. Given a JD + a resumeai run id, it produces a 1-page cover letter that lines up with the resume's bullets without contradicting them.
- Shared library: [ai-tailor-core](https://github.com/duketis/ai-tailor-core) — the engine room both apps consume (LLM subprocess client, JD ingest, run orchestration, verifier scaffold, user-context loader).
- Future caller: [jobai](https://github.com/duketis/jobai) — finds Australian jobs and will eventually orchestrate the resumeai + coverletterai pair per opportunity.

## What it does

Given a job description (URL or paste-in):

1. **Parse the JD** via httpx + selectolax + a deterministic regex pass + a single LLM call for the open-ended fields → typed `JobRequirements`.
2. **Load your full career context** from `UserContext/`: `resume.yaml`, `work_history/*.md`, `projects/*.md`, `cover_letters/*.md`, `git_audit/*.md`. Each project entry gets enriched with a recursive scan of the underlying repo (READMEs, manifests, code stats, git log).
3. **Run a Claude-powered tailoring agent** that emits a structured `TailoredResume` pydantic model. Headline + name + contact are verbatim passthroughs from `resume.yaml`; everything else is reshaped for the JD.
4. **Render to PDF** by feeding the structured output through a Jinja2 LaTeX template and compiling with Tectonic. Layout is decoupled from content length — adding or removing a bullet rebuilds the document; it can never shove the layout around.
5. **QC the output** with two passes: a text-mode LLM-judge call (catches fabrications, missing must-haves, weird tone, page-count overflow against the 3-page target) and a vision pass (rasterises every page and asks Claude for layout-only feedback — orphan lines, margin overflow, density issues).

## Status

**Pre-1.0, end-to-end pipeline shipping.** Run `docker compose up -d` and the FastAPI app listens on `:8765`. Hit `POST /api/tailor` with a `jd_url` or `jd_text`, poll `GET /api/runs/{run_id}` until terminal, download the PDF from `GET /api/runs/{run_id}/pdf`. A server-rendered HTML page at `/tailor` is the human-facing surface.

Architecture decisions with rejected alternatives live in `_private/ARCHITECTURE.md`; the phased build plan in `_private/BUILD_PLAN.md`.

**Headline numbers (2026-05-13):**
- 216 unit tests locally (the missing ~300 from the pre-extraction days moved into [ai-tailor-core](https://github.com/duketis/ai-tailor-core), which has 348)
- mypy strict, ruff strict, ruff-format clean
- CI green on every push (ruff + mypy + pytest)
- Docker container running on `:8765`

## Architecture (high level)

```
[Job description URL or text]
          │
          ▼
[ tailor_core.jd ] ── deterministic regex pass + LLM extractor → typed JobRequirements
          │
          ▼
[ tailor_core.context ] ── resume.yaml + work_history + projects + git_audit + cover_letters
          │                + recursive local-repo scan per project (READMEs, manifests, git log)
          │
          ▼
[ resumeai.agent ] ── single-shot Claude tailoring agent → structured TailoredResume
          │           (claude CLI subprocess, Max-subscription OAuth path)
          │
          ▼
[ resumeai.renderer ] ── Jinja2 LaTeX template → Tectonic → 3-page PDF on disk
          │              (master never touched; full rebuild every run)
          │
          ▼
[ tailor_core.verifier ] ── text-mode LLM-judge pass + vision-mode pass (rasterise + Anthropic SDK)
          │                 page-count overflow check (target ≤ 3)
          ▼
[ Run record in SQLite ] ── status, requirements, tailored output, verification result, PDF URL
[ FastAPI surface ] ── POST /api/tailor + GET /api/runs/{id} + GET /api/runs/{id}/pdf + SSE events
```

The `BaseOrchestrator[TailoredResume, RuntimeSettings]` template-method skeleton lives in `tailor_core.runs.orchestrator`; this repo's `TailoringOrchestrator` is a four-method subclass that supplies `_tailor` / `_render` / `_verify` / `_verify_visually`. Same skeleton serves coverletterai with its own four methods.

## Why these choices

Decision rationale (including rejected alternatives) lives in `_private/ARCHITECTURE.md`. Headlines:

- **LaTeX + Tectonic, not Google Docs.** The Google Docs renderer (v0.x) used `documents.batchUpdate` to fill a master template — same root pathology as a WYSIWYG editor: text length controls layout, so a longer bullet shoves the layout around. LaTeX rebuilds the full document on every compile; layout is decoupled from content. Tectonic is a single-binary engine that needs zero external state.
- **`claude` CLI subprocess, not Anthropic API key.** Uses the Anthropic Max subscription. Auth flows through a long-lived `CLAUDE_CODE_OAUTH_TOKEN` generated once via `claude setup-token`; the vision verifier uses the Anthropic Python SDK with the same OAuth token, so no API key anywhere.
- **Shared lib, not copy-paste.** When the cover-letter sibling needed the same JD ingest + orchestration + verifier scaffold, those parts moved to `ai-tailor-core` rather than getting duplicated. The lib carries everything that isn't resume- or cover-letter-specific.
- **Master-copy pattern.** The candidate's `UserContext/` tree (resume.yaml, work_history/*.md, projects/*.md, ...) is the source of truth. Every tailoring run produces a fresh PDF from a fresh LaTeX compile; the source tree is read-only at runtime.
- **Separate repo from jobai + coverletterai, not a monorepo.** Each repo tells one story; the HTTP boundary between them is the senior-engineering signal. Cross-referencing in READMEs.

## Engineering bar

Same bar as the sibling repos. Staff/lead-engineer-at-a-top-tier-company, no shortcuts:

- Python 3.12, `mypy --strict`, `ruff` strict (lint + format), `from __future__ import annotations` everywhere
- TDD-first, 100% line + branch coverage on every change
- Conventional commits, granular history, GPG-signed
- CI from day one: ruff + ruff-format + mypy + pytest on every push and PR
- Pre-commit hooks mirror CI locally

## How to run locally

```bash
# 1. One-time: generate a Max-subscription OAuth token on the host.
claude setup-token   # writes the token; copy it into .env as CLAUDE_CODE_OAUTH_TOKEN

# 2. Bring up the container.
docker compose up -d

# 3. Kick off a tailoring run by JD URL.
curl -X POST http://localhost:8765/api/tailor \
  -H 'Content-Type: application/json' \
  -d '{"jd_url": "https://au.seek.com/job/12345"}'
# => {"run_id":"run_...","status":"pending"}

# 4. Poll until terminal.
curl http://localhost:8765/api/runs/<run_id>

# 5. Download the PDF.
curl -o resume.pdf http://localhost:8765/api/runs/<run_id>/pdf
```

A server-rendered HTML page at `http://localhost:8765/tailor` is the human-facing surface.

## Quality gate (run before every commit)

```bash
./Tools/quality-gate.sh
```

That sets up a scratch venv at `/tmp/resumeai-tools` on first run, installs the sibling `ai-tailor-core` checkout editable so live edits drive the test run, then runs `ruff check` + `ruff format --check` + `mypy` + `pytest` — the same set CI runs.

## Acknowledgements

Built with [Claude Code](https://claude.com/claude-code).
