"""The issue index: GitHub JSON -> records, what gets embedded, incremental upserts.

Unit tests use a fake embedder (no model download). The upsert tests need a
real database and are opt-in (GITHUBTRIAGE_DB_TESTS=1); they run inside a
transaction that is rolled back.
"""

import os
from datetime import UTC, datetime

import pytest

from githubtriage.github_issues import NotAnIssueError
from githubtriage.issue_index import (
    EMBED_DIMS,
    EMBED_MAX_CHARS,
    IssueRecord,
    embed_text,
    fetch_issue_records,
    index_issues,
    record_from_github,
    text_hash,
)


def raw_issue(number=7, body="body text", created="2025-03-01T10:00:00Z", **extra):
    return {
        "number": number,
        "title": f"Issue {number}",
        "body": body,
        "state": "closed",
        "state_reason": "duplicate",
        "labels": [{"name": "bug V2"}],
        "html_url": f"https://github.com/o/r/issues/{number}",
        "repository_url": "https://api.github.com/repos/o/r",
        "created_at": created,
        "updated_at": created,
        "closed_at": created,
        **extra,
    }


def test_record_from_github_maps_the_fields_we_store():
    r = record_from_github(raw_issue())
    assert (r.repo, r.number, r.state_reason, r.labels) == (
        "o/r",
        7,
        "duplicate",
        ["bug V2"],
    )
    assert r.created_at.tzinfo is not None


def test_null_body_becomes_empty():
    assert record_from_github(raw_issue(body=None)).body == ""


def test_pull_requests_are_refused():
    with pytest.raises(NotAnIssueError):
        record_from_github(raw_issue(pull_request={}))


def test_embed_text_keeps_the_start_and_is_capped():
    # The model reads ~512 tokens and ignores the rest silently: cut it
    # ourselves, keeping title + the start of the description.
    r = record_from_github(raw_issue(body="x" * 10_000))
    text = embed_text(r)
    assert text.startswith("Issue 7\n\n")
    assert len(text) == EMBED_MAX_CHARS


def test_text_hash_changes_only_with_the_embedded_text():
    a = record_from_github(raw_issue(body="same"))
    b = record_from_github(raw_issue(body="same"))
    c = record_from_github(raw_issue(body="edited"))
    assert text_hash(embed_text(a)) == text_hash(embed_text(b))
    assert text_hash(embed_text(a)) != text_hash(embed_text(c))


def test_model_cache_lives_in_the_data_dir(isolated_data_dir):
    # Not the OS temp folder (clean-up tools may empty it); gitignored.
    from githubtriage.issue_index import model_cache_dir

    assert model_cache_dir() == isolated_data_dir / "models"


class FakeGitHub:
    def __init__(self, issues):
        self.issues = issues

    def list_issues(self, repo, *, state, max_pages):
        yield from self.issues


def test_fetch_stops_at_the_first_issue_older_than_since():
    gh = FakeGitHub(
        [
            raw_issue(3, created="2024-01-01T00:00:00Z"),
            raw_issue(2, created="2023-06-30T23:59:59Z"),  # older: stop here
            raw_issue(1, created="2023-08-01T00:00:00Z"),  # never reached
        ]
    )
    since = datetime(2023, 7, 1, tzinfo=UTC)
    assert [r.number for r in fetch_issue_records(gh, "o/r", created_since=since)] == [
        3
    ]


# --- With a real database (opt-in) -----------------------------------------------


class FakeEmbedder:
    model = "fake-model"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t) % 7)] + [0.0] * (EMBED_DIMS - 1) for t in texts]


needs_db = pytest.mark.skipif(
    os.environ.get("GITHUBTRIAGE_DB_TESTS") != "1",
    reason="needs a running database: docker compose up -d, GITHUBTRIAGE_DB_TESTS=1",
)


@pytest.fixture
def rollback_conn():
    from githubtriage.db import connect, migrate

    migrate()
    # force_rollback: an OUTER transaction that always rolls back. The code
    # under test opens its own `with conn.transaction()` blocks — on an idle
    # connection those COMMIT, so a plain rollback() at the end left test rows
    # behind (found 2026-11-26). Inside this block they are only savepoints.
    with connect() as conn, conn.transaction(force_rollback=True):
        yield conn


def records(*bodies):
    return [
        IssueRecord.model_validate(
            record_from_github(raw_issue(n, body=b)).model_dump() | {"repo": "t/idx"}
        )
        for n, b in enumerate(bodies, start=1)
    ]


@needs_db
def test_index_is_incremental(rollback_conn):
    embedder = FakeEmbedder()

    first = index_issues(rollback_conn, records("a", "b", "c"), embedder, batch_size=2)
    again = index_issues(rollback_conn, records("a", "b", "c"), embedder, batch_size=2)
    edited = index_issues(
        rollback_conn, records("a", "B!", "c"), embedder, batch_size=2
    )

    assert (first.embedded, first.unchanged) == (3, 0)
    assert (again.embedded, again.unchanged) == (0, 3)  # nothing re-embedded
    assert (edited.embedded, edited.unchanged) == (1, 2)  # only the edited one
    body = rollback_conn.execute(
        "SELECT body FROM issues WHERE repo = 't/idx' AND number = 2"
    ).fetchone()
    assert body == ("B!",)  # the issue row was updated too


@needs_db
def test_tests_leave_no_rows_behind():
    # Runs after the tests above (file order): proves the isolation works.
    from githubtriage.db import connect

    with connect() as conn:
        left = conn.execute(
            "SELECT count(*) FROM issues WHERE repo = 't/idx'"
        ).fetchone()
    assert left == (0,)


@needs_db
def test_wrong_vector_size_is_refused(rollback_conn):
    class TooShort(FakeEmbedder):
        def embed(self, texts):
            return [[0.0] * 10 for _ in texts]

    with pytest.raises(ValueError, match="10 dims"):
        index_issues(rollback_conn, records("a"), TooShort())
