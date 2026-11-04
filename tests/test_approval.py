import logging
from datetime import UTC, datetime

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from githubtriage.checkpointing import make_serializer
from githubtriage.models import (
    ApprovalRequest,
    Classification,
    Completeness,
    Decision,
    FieldCheck,
    Issue,
    TriageState,
)
from githubtriage.nodes.approval import (
    approval,
    build_approval_request,
    notify,
    route_after_approval,
)

NOW = datetime.now(UTC)


def make_state(**overrides) -> TriageState:
    values = {
        "repo": "owner/repo",
        "issue_number": 7,
        "issue": Issue(
            title="Wrong date", body="SECRET-BODY", author="a", updated_at=NOW
        ),
        "classification": Classification(type="bug", confidence=0.9, reasoning="r"),
        "completeness": Completeness(
            version=FieldCheck(status="vague", note="says latest"),
            repro_steps=FieldCheck(status="ok", note=""),
            logs=FieldCheck(status="not_needed", note=""),
        ),
        "draft": "Thanks!",
        "errors": ["completeness: X"],
    }
    values.update(overrides)
    return TriageState(**values)


def decision(action, **fields) -> Decision:
    return Decision(action=action, approver="maintainer1", decided_at=NOW, **fields)


# --- build_approval_request: what the maintainer sees ---


def test_request_contains_what_a_maintainer_needs():
    req = build_approval_request(make_state())

    assert req.issue_url == "https://github.com/owner/repo/issues/7"
    assert req.issue_type == "bug"
    # The LLM's note is shown to maintainers (it never goes into the public reply).
    assert req.completeness["version"].note == "says latest"
    assert req.draft == "Thanks!"
    assert req.errors == ["completeness: X"]  # degraded runs are visible


def test_request_leaves_out_the_issue_body():
    # Untrusted text isn't re-rendered; maintainers read it on GitHub (issue_url).
    payload = build_approval_request(make_state()).model_dump_json()
    assert "SECRET-BODY" not in payload


def test_request_handles_unchecked_and_unclassified_issues():
    req = build_approval_request(make_state(classification=None, completeness=None))

    assert req.issue_type == "unclear"
    assert req.completeness is None


# --- notify ---


def test_notify_logs_and_changes_nothing(caplog):
    with caplog.at_level(logging.INFO):
        assert notify(make_state()) == {}
    assert "owner/repo#7" in caplog.text


# --- route_after_approval ---


@pytest.mark.parametrize(
    "dec, expected",
    [
        (decision("approve"), "stale_check"),
        (decision("edit", edited_body="Better reply"), "stale_check"),
        (decision("reject", reason="duplicate"), "__end__"),
        (None, "__end__"),
    ],
)
def test_router(dec, expected):
    assert route_after_approval(make_state(decision=dec)) == expected


# --- approval: pause and resume inside a real (tiny) graph ---


def tiny_graph():
    builder = StateGraph(TriageState)
    builder.add_node("approval", approval)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=InMemorySaver(serde=make_serializer()))


def test_approval_pauses_with_the_request_then_resumes_with_a_decision():
    graph = tiny_graph()
    config = {"configurable": {"thread_id": "owner/repo#7"}}

    paused = graph.invoke(make_state(), config=config)

    # Paused: the interrupt carries the request as plain JSON.
    (interrupt_,) = paused["__interrupt__"]
    assert ApprovalRequest.model_validate(interrupt_.value).issue_number == 7
    assert graph.get_state(config).next == ("approval",)

    # Resume with the maintainer's decision (plain JSON in, validated by the node).
    answer = decision("reject", reason="duplicate")
    done = graph.invoke(Command(resume=answer.model_dump(mode="json")), config=config)

    assert TriageState.model_validate(done).decision == answer
    assert graph.get_state(config).next == ()
