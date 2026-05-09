"""User-prompt builder + system-prompt sanity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from resumeai.agent.prompt import SYSTEM_PROMPT, build_user_prompt
from resumeai.context.models import (
    Contact,
    Education,
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

if TYPE_CHECKING:
    pass


def test_system_prompt_documents_required_schema_fields() -> None:
    """The system prompt is the contract; if a field name drifts, the model
    won't emit it."""
    for field in (
        '"name"',
        '"headline"',
        '"contact"',
        '"summary"',
        '"skills"',
        '"work_history"',
        '"education"',
        '"certifications"',
        '"rationale"',
        '"source_slug"',
        '"bullets"',
    ):
        assert field in SYSTEM_PROMPT, f"missing {field}"


def test_system_prompt_states_anti_fabrication_rule() -> None:
    assert "Use ONLY facts present in CANDIDATE CONTEXT" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT


# -- build_user_prompt -------------------------------------------------------


def test_user_prompt_includes_jd_facts(
    sample_jd: JobRequirements, sample_context: UserContext
) -> None:
    prompt = build_user_prompt(sample_jd, sample_context)

    assert "Senior Backend Engineer" in prompt
    assert "Globex" in prompt
    assert "Sydney" in prompt
    assert "Python" in prompt
    assert "Postgres" in prompt
    assert "first principles" in prompt
    assert "high-trust environment" in prompt
    assert "AU citizen" in prompt
    assert "Senior Backend Engineer at Globex" in prompt  # raw text included


def test_user_prompt_includes_every_work_history_slug(
    sample_jd: JobRequirements, sample_context: UserContext
) -> None:
    prompt = build_user_prompt(sample_jd, sample_context)

    for entry in sample_context.work_history:
        assert f"`{entry.slug}`" in prompt
        assert entry.company in prompt
        assert entry.title in prompt


def test_user_prompt_includes_every_master_skill(
    sample_jd: JobRequirements, sample_context: UserContext
) -> None:
    prompt = build_user_prompt(sample_jd, sample_context)
    assert sample_context.resume is not None
    for skill in sample_context.resume.skills:
        assert skill in prompt


def test_user_prompt_includes_git_audit_and_cover_letter(
    sample_jd: JobRequirements, sample_context: UserContext
) -> None:
    prompt = build_user_prompt(sample_jd, sample_context)

    assert "acme/platform" in prompt
    assert "Owner of services/ingestor" in prompt
    assert "Hi team," in prompt


def test_user_prompt_handles_empty_context(sample_jd: JobRequirements) -> None:
    """The agent should still produce a (minimal) tailored resume even with
    no candidate context — the prompt must not crash."""
    prompt = build_user_prompt(sample_jd, UserContext())

    assert "Senior Backend Engineer" in prompt
    assert "empty context" in prompt


def test_user_prompt_handles_jd_with_no_extracted_lists() -> None:
    """When the JD parser couldn't find any skills/must-haves/vocabulary,
    the prompt must still render rather than producing confusing blank lists."""
    bare_jd = JobRequirements(title="Cosmic Wrangler", raw_text="...")

    prompt = build_user_prompt(bare_jd, UserContext())

    assert "Cosmic Wrangler" in prompt
    assert "(none extracted)" in prompt


def test_user_prompt_includes_resume_headline_and_contact_extras() -> None:
    """All optional contact fields + headline must surface for the model."""
    context = UserContext(
        resume=ResumeBase(
            name="Alex",
            headline="Backend engineer",
            contact=Contact(
                email="alex@example.com",
                phone="+61 400 000 000",
                location="Melbourne",
                linkedin="https://linkedin.com/in/alex",
                github="https://github.com/alex",
                website="https://alex.dev",
            ),
            skills=("Python",),
            education=(Education(institution="U", degree="BSc"),),
            certifications=("AWS SAA",),
        )
    )
    jd = JobRequirements(
        title="Eng",
        role_type=RoleType.ENGINEERING,
        seniority=Seniority.SENIOR,
        employment_type=EmploymentType.FULL_TIME,
        remote_type=RemoteType.REMOTE,
    )

    prompt = build_user_prompt(jd, context)

    for fragment in (
        "Backend engineer",
        "+61 400 000 000",
        "Melbourne",
        "linkedin.com/in/alex",
        "github.com/alex",
        "https://alex.dev",
        "AWS SAA",
        "U: BSc",
    ):
        assert fragment in prompt


def test_user_prompt_renders_education_with_no_year_metadata() -> None:
    context = UserContext(
        resume=ResumeBase(
            name="Alex",
            contact=Contact(email="alex@example.com"),
            education=(Education(institution="Sample U", degree="BEng"),),
        )
    )
    jd = JobRequirements(title="Eng")

    prompt = build_user_prompt(jd, context)

    assert "Sample U: BEng" in prompt


def test_user_prompt_renders_education_with_only_start_year() -> None:
    context = UserContext(
        resume=ResumeBase(
            name="Alex",
            contact=Contact(email="alex@example.com"),
            education=(Education(institution="U", degree="BEng", year_start=2018),),
        )
    )
    prompt = build_user_prompt(JobRequirements(title="Eng"), context)
    assert "(2018–present)" in prompt


def test_user_prompt_renders_education_with_only_end_year() -> None:
    context = UserContext(
        resume=ResumeBase(
            name="Alex",
            contact=Contact(email="alex@example.com"),
            education=(Education(institution="U", degree="BEng", year_end=2020),),
        )
    )
    prompt = build_user_prompt(JobRequirements(title="Eng"), context)
    assert "(…–2020)" in prompt


def test_user_prompt_renders_resume_with_no_education_or_certs() -> None:
    context = UserContext(resume=ResumeBase(name="Alex", contact=Contact(email="alex@example.com")))
    prompt = build_user_prompt(JobRequirements(title="Eng"), context)
    assert "**Education:**\n- (none)" in prompt
    assert "**Certifications:**\n- (none)" in prompt


def test_user_prompt_renders_work_history_undated_role() -> None:
    context = UserContext(
        work_history=(
            WorkHistoryEntry(
                slug="x",
                title="Engineer",
                company="Acme",
            ),
        )
    )
    prompt = build_user_prompt(JobRequirements(title="Eng"), context)
    assert "(undated)" in prompt


def test_user_prompt_renders_work_history_open_ended_role() -> None:
    context = UserContext(
        work_history=(
            WorkHistoryEntry(
                slug="x",
                title="Engineer",
                company="Acme",
                start="2024-01",
            ),
        )
    )
    prompt = build_user_prompt(JobRequirements(title="Eng"), context)
    assert "2024-01 → present" in prompt


def test_user_prompt_renders_work_history_legacy_role() -> None:
    """End-only is unusual but valid — pre-fill role with no start date."""
    context = UserContext(
        work_history=(
            WorkHistoryEntry(
                slug="x",
                title="Engineer",
                company="Acme",
                end="2018-05",
            ),
        )
    )
    prompt = build_user_prompt(JobRequirements(title="Eng"), context)
    assert "… → 2018-05" in prompt


def test_user_prompt_renders_cover_letter_without_role_or_company() -> None:
    from resumeai.context.models import CoverLetterEntry  # noqa: PLC0415

    context = UserContext(
        cover_letters=(CoverLetterEntry(slug="generic", body="Dear hiring manager,"),)
    )
    prompt = build_user_prompt(JobRequirements(title="Eng"), context)
    assert "### generic" in prompt
    assert "Dear hiring manager," in prompt


def test_user_prompt_renders_cover_letter_without_body() -> None:
    """A registered letter with empty body still renders a header — used as
    a placeholder while a draft is in flight."""
    from resumeai.context.models import CoverLetterEntry  # noqa: PLC0415

    context = UserContext(cover_letters=(CoverLetterEntry(slug="empty"),))
    prompt = build_user_prompt(JobRequirements(title="Eng"), context)
    assert "### empty" in prompt
    # No body line means no extra paragraph after the header.
    assert "Dear" not in prompt


def test_user_prompt_renders_minimal_git_audit_entry() -> None:
    """Audit entries with only repo (no role/period/summary) still render."""
    from resumeai.context.models import GitAuditEntry  # noqa: PLC0415

    context = UserContext(git_audit=(GitAuditEntry(slug="x", repo="acme/x"),))
    prompt = build_user_prompt(JobRequirements(title="Eng"), context)
    assert "### acme/x (x)" in prompt
    assert "Role:" not in prompt
    assert "Period:" not in prompt


def test_user_prompt_orders_sections_consistently(
    sample_jd: JobRequirements, sample_context: UserContext
) -> None:
    """JD first, then candidate context, then OUTPUT marker — the model
    relies on this ordering."""
    prompt = build_user_prompt(sample_jd, sample_context)
    jd_idx = prompt.index("# JOB DESCRIPTION")
    ctx_idx = prompt.index("# CANDIDATE CONTEXT")
    out_idx = prompt.index("# OUTPUT")
    assert jd_idx < ctx_idx < out_idx
