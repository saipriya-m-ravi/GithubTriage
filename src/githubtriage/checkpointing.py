"""Checkpointers: where paused runs are saved, and which classes may be restored.

Saved state is turned back into Python objects when a paused run resumes.
Recreating arbitrary classes from stored data is a known vulnerability
(insecure deserialization), so LangGraph is moving to an allowlist. We declare
ours now instead of waiting for a future upgrade to block them.

A class missing from this list is NOT an error: in strict mode it silently comes
back as a plain dict, and code fails later somewhere unrelated. A test checks that
every model reachable from TriageState is listed.
"""

import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from githubtriage.models import (
    Classification,
    Completeness,
    Decision,
    FieldCheck,
    Issue,
    TriageState,
)

# TriageState itself is stored too: LangGraph saves the input passed to invoke()
# (a whole TriageState) in the checkpoint, not only the individual field values.
CHECKPOINT_CLASSES = (
    TriageState,
    Issue,
    Classification,
    Completeness,
    FieldCheck,
    Decision,
)


def make_serializer() -> JsonPlusSerializer:
    """A serializer that may only recreate the classes in CHECKPOINT_CLASSES."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(c.__module__, c.__name__) for c in CHECKPOINT_CLASSES]
    )


def memory_checkpointer() -> InMemorySaver:
    """Checkpoints in process memory — lost on exit. For tests and one-off runs."""
    return InMemorySaver(serde=make_serializer())


@contextmanager
def sqlite_checkpointer(path: Path) -> Iterator[SqliteSaver]:
    """Checkpoints in a SQLite file, so a run paused by one process can be resumed
    by another. A context manager: the database connection is always closed.

        with sqlite_checkpointer(settings.checkpoint_path) as saver:
            graph = build_graph(checkpointer=saver)
            ...

    (SqliteSaver.from_conn_string can't take a serializer, hence our own.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph may use the connection from worker threads.
    with closing(sqlite3.connect(path, check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=make_serializer())


# --- Run identity -----------------------------------------------------------
#
# A thread is ONE triage run, not one issue. Reusing a thread for a new triage
# would continue the old state (e.g. posted_comment_id stays set, so the new reply
# is silently skipped). So each triage gets its own thread:
#     owner/repo#7@20261104T101500.123456Z
# and an issue can have several runs over time.

_THREAD_PATTERN = re.compile(
    r"^(?P<repo>[A-Za-z0-9-]+/[A-Za-z0-9._-]+)#(?P<number>\d+)@(?P<run>[0-9T.]+Z)$"
)


_ISSUE_PATTERN = re.compile(
    r"^(?P<repo>[A-Za-z0-9-]+/[A-Za-z0-9._-]+)#(?P<number>\d+)$"
)


def issue_key(repo: str, issue_number: int) -> str:
    """How a person refers to an issue: "owner/repo#7"."""
    return f"{repo}#{issue_number}"


def parse_issue_key(ref: str) -> tuple[str, int]:
    """(repo, issue_number) from "owner/repo#7"; ValueError otherwise."""
    match = _ISSUE_PATTERN.fullmatch(ref)
    if match is None:
        raise ValueError(f"invalid issue reference: {ref!r} (expected owner/repo#N)")
    return match["repo"], int(match["number"])


def new_thread_id(repo: str, issue_number: int, now: datetime | None = None) -> str:
    """A fresh thread for a new triage run of this issue (UTC, microseconds)."""
    now = now or datetime.now(UTC)
    return f"{issue_key(repo, issue_number)}@{now:%Y%m%dT%H%M%S.%f}Z"


def parse_thread_id(thread: str) -> tuple[str, int]:
    """(repo, issue_number) of a thread id; ValueError if it isn't one of ours.

    Validated because thread ids become file names (outbox) and markers.
    """
    match = _THREAD_PATTERN.fullmatch(thread)
    if match is None:
        raise ValueError(f"invalid run id: {thread!r}")
    return match["repo"], int(match["number"])


def thread_config(thread: str) -> dict:
    """The config that tells the checkpointer which run to save or resume."""
    return {"configurable": {"thread_id": thread}}


def all_threads(saver: BaseCheckpointSaver) -> set[str]:
    """Every run the checkpointer knows about."""
    return {c.config["configurable"]["thread_id"] for c in saver.list(None)}


def waiting_runs(
    graph: CompiledStateGraph, saver: BaseCheckpointSaver, repo: str, issue_number: int
) -> list[str]:
    """Runs of this issue that are paused, waiting for a maintainer's decision."""
    prefix = issue_key(repo, issue_number) + "@"
    return sorted(
        t
        for t in all_threads(saver)
        if t.startswith(prefix)
        and graph.get_state(thread_config(t)).next == ("approval",)
    )


def cancel_waiting_runs(
    graph: CompiledStateGraph, saver: BaseCheckpointSaver, repo: str, issue_number: int
) -> list[str]:
    """Cancel (delete) waiting runs of this issue — a newer triage replaces them.

    Guarantees at most one draft per issue is ever waiting for approval.
    Deleting also removes the cancelled draft's record; production should mark
    runs cancelled in a runs table instead (DESIGN.md R6).
    """
    cancelled = waiting_runs(graph, saver, repo, issue_number)
    for thread in cancelled:
        saver.delete_thread(thread)
    return cancelled
