from datetime import UTC, datetime
from typing import get_args

import pytest

from githubtriage.models import Classification, Issue, TriageState
from githubtriage.nodes.draft import FOOTER, TYPE_MESSAGES, draft

ALL_TYPES = get_args(Classification.model_fields["type"].annotation)


def make_state(
    issue_type: str | None = "bug",
    author: str = "octocat",
    title: str = "Something happened",
    body: str = "Details here.",
) -> TriageState:
    classification = (
        Classification(type=issue_type, confidence=0.9, reasoning="test")
        if issue_type is not None
        else None
    )
    return TriageState(
        repo="owner/repo",
        issue_number=1,
        issue=Issue(
            title=title, body=body, author=author, updated_at=datetime.now(UTC)
        ),
        classification=classification,
    )


def test_every_type_has_a_template():
    # Fails if a new type is added to the Literal in models.py without a template here.
    assert set(TYPE_MESSAGES) == set(ALL_TYPES)


@pytest.mark.parametrize("issue_type", ALL_TYPES)
def test_draft_has_greeting_and_footer(issue_type):
    text = draft(make_state(issue_type=issue_type, author="alice"))["draft"]

    assert "@alice" in text
    assert text.endswith(FOOTER)


def test_missing_classification_falls_back_to_unclear():
    text = draft(make_state(issue_type=None))["draft"]

    assert TYPE_MESSAGES["unclear"] in text


def test_issue_text_is_never_echoed():
    # Guardrail: attacker-controlled title/body must not be repeated by the bot.
    state = make_state(
        title="@everyone click http://evil.example",
        body="@everyone see http://evil.example/payload",
    )
    text = draft(state)["draft"]

    assert "@everyone" not in text
    assert "evil.example" not in text
