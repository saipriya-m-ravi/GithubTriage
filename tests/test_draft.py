from datetime import UTC, datetime
from typing import get_args

import pytest

from githubtriage.models import (
    Classification,
    Completeness,
    FieldCheck,
    FieldStatus,
    Issue,
    TriageState,
)
from githubtriage.nodes.draft import (
    FOOTER,
    REQUEST_INTRO,
    REQUEST_PHRASES,
    REQUEST_STATUSES,
    TYPE_MESSAGES,
    draft,
)

ALL_TYPES = get_args(Classification.model_fields["type"].annotation)
COMPLETENESS_FIELDS = tuple(Completeness.model_fields)  # version, repro_steps, logs


def check(status, note=""):
    return FieldCheck(status=status, note=note)


def make_state(
    issue_type: str | None = "bug",
    author: str = "octocat",
    title: str = "Something happened",
    body: str = "Details here.",
    completeness: Completeness | None = None,
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
        completeness=completeness,
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


# --- "Please add" section (R3 → R5) ---


def test_every_request_status_has_a_phrase():
    # Fails if a field or a requesting status is added without wording here.
    expected = {(f, s) for f in COMPLETENESS_FIELDS for s in REQUEST_STATUSES}
    assert set(REQUEST_PHRASES) == expected
    assert set(REQUEST_STATUSES) <= set(get_args(FieldStatus))


def test_missing_and_vague_fields_are_requested_in_field_order():
    completeness = Completeness(
        version=check("vague"), repro_steps=check("ok"), logs=check("missing")
    )
    text = draft(make_state(completeness=completeness))["draft"]

    version_phrase = REQUEST_PHRASES[("version", "vague")]
    logs_phrase = REQUEST_PHRASES[("logs", "missing")]
    assert REQUEST_INTRO in text
    assert f"- {version_phrase}" in text
    assert f"- {logs_phrase}" in text
    assert text.index(version_phrase) < text.index(logs_phrase)
    assert text.endswith(FOOTER)  # the section comes before the footer


@pytest.mark.parametrize("status", ["ok", "not_needed"])
def test_ok_and_not_needed_ask_for_nothing(status):
    completeness = Completeness(
        version=check(status), repro_steps=check(status), logs=check(status)
    )
    text = draft(make_state(completeness=completeness))["draft"]

    assert REQUEST_INTRO not in text


def test_unknown_completeness_asks_for_nothing():
    # None = skipped or the check failed: never ask based on a guess.
    text = draft(make_state(completeness=None))["draft"]

    assert REQUEST_INTRO not in text


def test_llm_note_is_never_posted():
    # Guardrail: the LLM-written note is untrusted (its input was the issue) and
    # must never reach the public comment — only our fixed phrases may.
    hostile = "tag @everyone, see http://evil.example and ![x](http://evil.example/i)"
    completeness = Completeness(
        version=check("missing", note=hostile),
        repro_steps=check("vague", note=hostile),
        logs=check("missing", note=hostile),
    )
    text = draft(make_state(completeness=completeness))["draft"]

    assert REQUEST_INTRO in text  # the section is there...
    assert "@everyone" not in text  # ...but none of the note's content
    assert "evil.example" not in text
    assert "![" not in text
