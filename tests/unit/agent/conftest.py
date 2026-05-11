"""Shared fixtures for the agent tests.

Builds a representative ``JobRequirements`` + ``UserContext`` pair plus a
canonical "well-formed LLM response" payload so each test can focus on
the surface it cares about.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from resumeai.context.models import (
    Contact,
    CoverLetterEntry,
    Education,
    GitAuditEntry,
    ProjectEntry,
    ResumeBase,
    UserContext,
    WorkHistoryEntry,
)
from resumeai.jd.models import (
    EmploymentType,
    JobRequirements,
    RemoteType,
    RoleType,
    Seniority,
)


@pytest.fixture
def sample_jd() -> JobRequirements:
    return JobRequirements(
        title="Senior Backend Engineer",
        company="Globex",
        location="Sydney",
        role_type=RoleType.ENGINEERING,
        seniority=Seniority.SENIOR,
        employment_type=EmploymentType.FULL_TIME,
        remote_type=RemoteType.HYBRID,
        required_skills=("Python", "Postgres", "Kubernetes"),
        nice_to_have_skills=("Rust",),
        must_haves=("AU citizen",),
        employer_vocabulary=("first principles", "high-trust environment"),
        raw_text="Senior Backend Engineer at Globex...",
        source_url="https://example.com/jd",
    )


@pytest.fixture
def sample_context() -> UserContext:
    return UserContext(
        resume=ResumeBase(
            name="Alex Sample",
            headline="Senior backend engineer",
            contact=Contact(
                email="alex@example.com",
                phone="+61 400 000 000",
                location="Melbourne, AU",
                linkedin="https://linkedin.com/in/alex",
                github="https://github.com/alex",
            ),
            skills=("Python", "Postgres", "Kubernetes", "TypeScript"),
            education=(
                Education(
                    institution="Sample U",
                    degree="BEng",
                    field="Software Engineering",
                    year_start=2014,
                    year_end=2017,
                ),
            ),
            certifications=("AWS SAA",),
        ),
        work_history=(
            WorkHistoryEntry(
                slug="01-acme",
                title="Senior Software Engineer",
                company="Acme Corp",
                start="2022-03",
                end="2024-09",
                location="Melbourne, AU",
                technologies=("Python", "FastAPI", "Postgres", "Kubernetes"),
                summary="Built the data ingestion platform.",
                bullets=(
                    "Designed a SHA-based dedup pipeline that cut duplicates 92%.",
                    "Migrated to Kubernetes, saving $14K/month.",
                    "Mentored two mid-level engineers.",
                ),
                raw_markdown="# raw",
            ),
        ),
        git_audit=(
            GitAuditEntry(
                slug="acme-platform",
                repo="acme/platform",
                role="Senior Software Engineer",
                period="2022-03 → 2024-09",
                summary="Owner of services/ingestor for the full tenure.",
                raw_markdown="# raw",
            ),
        ),
        cover_letters=(
            CoverLetterEntry(
                slug="acme-cover",
                role="Senior Engineer",
                company="Acme",
                body="Hi team, ...",
                raw_markdown="---\nrole: Senior Engineer\n---\nHi team, ...",
            ),
        ),
        projects=(
            ProjectEntry(
                slug="sample-tool",
                name="sample-tool",
                url="https://github.com/alex/sample-tool",
                status="v0.4.x (open-source)",
                stack="Python, FastAPI, SQLite",
                summary="A sample local-first ingestion tool used in tests.",
                bullets=(
                    "Ingests sample data into SQLite.",
                    "FastAPI surface for queries.",
                ),
                body="A sample personal project. Always describe as 'open-source'.",
            ),
        ),
    )


def _well_formed_response_dict(*, name: str = "Alex Sample") -> dict[str, Any]:
    return {
        "name": name,
        "headline": "Senior backend engineer with deep Python + Postgres",
        "contact": {
            "email": "alex@example.com",
            "phone": "+61 400 000 000",
            "location": "Melbourne, AU",
            "linkedin": "https://linkedin.com/in/alex",
            "github": "https://github.com/alex",
            "website": None,
        },
        "summary": "Backend engineer with 5+ years building data-heavy systems...",
        "skills": ["Python", "Postgres", "Kubernetes", "TypeScript"],
        "work_history": [
            {
                "company": "Acme Corp",
                "title": "Senior Software Engineer",
                "period": "2022-03 → 2024-09",
                "location": "Melbourne, AU",
                "source_slug": "01-acme",
                "bullets": [
                    {
                        "text": "Designed SHA-based deduplication that cut dupes 92%.",
                        "source_slug": "01-acme",
                    },
                    {
                        "text": "Migrated platform to Kubernetes; saved $14K/month.",
                        "source_slug": "01-acme",
                    },
                ],
            }
        ],
        "education": [
            {
                "institution": "Sample U",
                "degree": "BEng",
                "field": "Software Engineering",
                "year_start": 2014,
                "year_end": 2017,
            }
        ],
        "certifications": ["AWS SAA"],
        "key_achievements": [
            "Designed a SHA-based deduplication pipeline that cut duplicates 92% at Acme.",
            "Built sample-tool (github.com/alex/sample-tool) with 120+ tests and mypy strict.",
        ],
        "personal_projects": [
            {
                "name": "sample-tool",
                "description": "local-first sample ingestion tool",
                "stack": "Python, FastAPI, SQLite",
                "link": "https://github.com/alex/sample-tool",
                "link_label": "github.com/alex/sample-tool",
                "bullets": [
                    {
                        "text": "Ingests sample data from public APIs into SQLite.",
                        "source_slug": "sample-tool",
                    }
                ],
                "source_slug": "sample-tool",
            }
        ],
        "rationale": "Reordered skills so Python + Postgres lead, kept Acme bullets...",
    }


@pytest.fixture
def well_formed_response() -> str:
    return json.dumps(_well_formed_response_dict())


@pytest.fixture
def well_formed_response_dict() -> dict[str, Any]:
    return _well_formed_response_dict()
