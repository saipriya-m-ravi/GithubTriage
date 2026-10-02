import operator
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


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


class TriageState(BaseModel):
    repo: str
    issue_number: int
    issue: Issue
    classification: Classification | None = None
    draft: str | None = None
    errors: Annotated[list[str], operator.add] = []
