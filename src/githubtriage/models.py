import operator
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator


class Issue(BaseModel):
    title: str
    body: str
    author: str
    updated_at: datetime

    @field_validator("body", mode="before")
    @classmethod
    def none_to_empty(cls, value):
        return "" if value is None else value


class Classification(BaseModel):
    """Classification of a GitHub issue by what the reporter is asking for."""

    type: Literal["question", "bug", "feature", "docs", "unclear"] = Field(
        description=(
            "What the reporter is doing. "
            "bug: reports that existing behaviour is broken, crashes, or gives wrong results. "
            "feature: asks for new behaviour, or an improvement to something that works as designed. "
            "question: asks how to do something or how something works; nothing is reported as broken. "
            "docs: reports that documentation is wrong, outdated, unclear, or missing "
            "(the fix is a change to the docs, not an answer). "
            "unclear: the issue mixes several types, or has too little information to decide."
        )
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "How certain you are of the chosen type, from 0 to 1. "
            "Use below 0.5 when torn between two types."
        ),
    )
    reasoning: str = Field(
        description=(
            "One sentence (max 25 words) naming or quoting the words in the issue "
            "that led to this type."
        )
    )


FieldStatus = Literal["missing", "vague", "ok", "not_needed"]


class FieldCheck(BaseModel):
    """Assessment of one piece of information a maintainer needs."""

    status: FieldStatus = Field(
        description=(
            "missing: the issue does not give this information at all. "
            "vague: it is mentioned but not usable (see the field's own description). "
            "ok: it is given clearly enough for a maintainer to act on. "
            "not_needed: this information does not apply to this issue."
        )
    )
    note: str = Field(
        description=(
            "If missing or vague: a short, polite phrase (max 15 words) saying exactly "
            "what to add, written so it can be shown to the reporter, e.g. "
            "\"the exact version number, not 'latest'\". "
            "If ok or not_needed: an empty string."
        )
    )


class Completeness(BaseModel):
    """Whether a bug report contains what a maintainer needs to reproduce it."""

    version: FieldCheck = Field(
        description=(
            "The version of the project the reporter is using. "
            "ok: a specific version such as 2.3.0. "
            "vague: 'latest', 'master', 'the new one', or only a date. "
            "Never not_needed for a bug."
        )
    )
    repro_steps: FieldCheck = Field(
        description=(
            "Steps or a code snippet someone else could follow to see the problem. "
            "ok: concrete steps or runnable code. "
            "vague: only a description of the symptom, e.g. 'it crashes when I use it'. "
            "Never not_needed for a bug."
        )
    )
    logs: FieldCheck = Field(
        description=(
            "Error messages, tracebacks or log output. "
            "ok: the error text or traceback is included. "
            "missing: an error, exception or crash is described but its text is not given. "
            "not_needed: no error, exception or crash is described "
            "(e.g. the code runs but returns a wrong value)."
        )
    )


class Decision(BaseModel):
    """A maintainer's decision on a draft (R6). Also an audit record and a label."""

    action: Literal["approve", "edit", "reject"]
    edited_body: str | None = None  # the replacement reply; only for "edit"
    reason: str | None = None  # optional, mostly for "reject"
    approver: str  # who decided (GitHub username once auth exists, slice 7)
    decided_at: AwareDatetime  # timezone-aware, so it compares safely with GitHub times

    @model_validator(mode="after")
    def edited_body_only_for_edit(self) -> "Decision":
        has_body = bool(self.edited_body and self.edited_body.strip())
        if self.action == "edit" and not has_body:
            raise ValueError("an 'edit' decision needs a non-empty edited_body")
        if self.action != "edit" and self.edited_body is not None:
            raise ValueError("edited_body is only allowed for an 'edit' decision")
        return self


class FieldReview(BaseModel):
    """One completeness field as shown to a maintainer (status + the LLM's note)."""

    status: FieldStatus
    note: str


class ApprovalRequest(BaseModel):
    """What a maintainer sees when asked to approve a draft — the interrupt payload.

    An explicit, minimal view of the state for the graph → UI boundary (CLI now,
    approval page in slice 7): stable while internal state changes, and only what's
    needed. The issue body is deliberately left out — maintainers read it on GitHub
    via issue_url, so we don't re-render untrusted text (XSS sink, DESIGN §5.5).
    """

    repo: str
    issue_number: int
    issue_title: str
    issue_url: str
    issue_type: str
    confidence: float
    reasoning: str
    # None = completeness wasn't checked (not a bug, or the check failed).
    completeness: dict[str, FieldReview] | None
    draft: str
    errors: list[str]


class TriageState(BaseModel):
    repo: str
    issue_number: int
    issue: Issue
    classification: Classification | None = None
    completeness: Completeness | None = None  # None until R3 runs, or if skipped
    draft: str | None = None
    # R6 — None until the maintainer decides / the comment is posted.
    decision: Decision | None = None
    posted_comment_id: str | None = None  # set once posted; prevents posting twice
    errors: Annotated[list[str], operator.add] = []
