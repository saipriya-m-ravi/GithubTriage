from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from githubtriage.models import Decision

NOW = datetime.now(UTC)


def decide(**fields):
    return Decision(approver="maintainer1", decided_at=NOW, **fields)


def test_approve_and_reject_are_valid():
    assert decide(action="approve").edited_body is None
    assert (
        decide(action="reject", reason="duplicate of #99").reason == "duplicate of #99"
    )


def test_edit_requires_a_body():
    with pytest.raises(ValidationError, match="needs a non-empty edited_body"):
        decide(action="edit")
    with pytest.raises(ValidationError, match="needs a non-empty edited_body"):
        decide(action="edit", edited_body="   ")


def test_body_only_allowed_for_edit():
    with pytest.raises(ValidationError, match="only allowed for an 'edit'"):
        decide(action="approve", edited_body="sneaky text")


def test_decided_at_must_be_timezone_aware():
    with pytest.raises(ValidationError):
        # Deliberately naive (no timezone) — the model must reject it.
        Decision(action="approve", approver="m", decided_at=datetime.now())  # noqa: DTZ005
