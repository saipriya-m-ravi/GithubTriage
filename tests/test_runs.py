"""Run identity helpers: thread ids, parsing, waiting runs, cancellation."""

import logging
from datetime import UTC, datetime

import pytest

from githubtriage.checkpointing import (
    cancel_waiting_runs,
    memory_checkpointer,
    new_thread_id,
    parse_issue_key,
    parse_thread_id,
    thread_config,
    waiting_runs,
)
from githubtriage.graph import build_graph
from githubtriage.models import Classification, Issue, TriageState


def test_new_thread_id_format_and_round_trip():
    thread = new_thread_id(
        "owner/repo", 7, now=datetime(2026, 11, 4, 10, 15, tzinfo=UTC)
    )

    assert thread == "owner/repo#7@20261104T101500.000000Z"
    assert parse_thread_id(thread) == ("owner/repo", 7)


def test_consecutive_runs_get_different_ids():
    assert new_thread_id("owner/repo", 7) != new_thread_id("owner/repo", 7)


@pytest.mark.parametrize("bad", ["owner/repo#7", "x#7@1Z/../..", "owner/repo#x@1Z"])
def test_parse_thread_id_rejects_non_run_ids(bad):
    with pytest.raises(ValueError, match="invalid run id"):
        parse_thread_id(bad)


def test_parse_issue_key():
    assert parse_issue_key("owner/repo#7") == ("owner/repo", 7)
    with pytest.raises(ValueError, match="invalid issue reference"):
        parse_issue_key("owner/repo")


def test_waiting_runs_and_cancellation(use_fake_llm, caplog):
    use_fake_llm(result=Classification(type="feature", confidence=0.9, reasoning="x"))
    saver = memory_checkpointer()
    graph = build_graph(checkpointer=saver)
    state = TriageState(
        repo="owner/repo",
        issue_number=7,
        issue=Issue(title="t", body="b", author="a", updated_at=datetime.now(UTC)),
    )
    run_a = new_thread_id("owner/repo", 7)
    run_other_issue = new_thread_id("owner/repo", 8)
    graph.invoke(state, config=thread_config(run_a))
    graph.invoke(
        state.model_copy(update={"issue_number": 8}),
        config=thread_config(run_other_issue),
    )

    with caplog.at_level(logging.WARNING):
        assert waiting_runs(graph, saver, "owner/repo", 7) == [run_a]
    # Listing runs loads every stored checkpoint, including the invoke() input
    # (a whole TriageState). Regression: it was missing from the allowlist.
    assert not [r for r in caplog.records if "deserializ" in r.getMessage().lower()]

    assert cancel_waiting_runs(graph, saver, "owner/repo", 7) == [run_a]
    assert waiting_runs(graph, saver, "owner/repo", 7) == []
    # Other issues' runs are untouched.
    assert waiting_runs(graph, saver, "owner/repo", 8) == [run_other_issue]
