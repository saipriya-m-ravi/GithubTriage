"""Collecting the eval set, against a fake GitHub."""

from datetime import UTC, datetime

import httpx
import pytest

from githubtriage.evaluation.dataset import collect, natural_frequencies
from githubtriage.github_client import GitHubClient

REPO = "owner/repo"
SINCE = datetime(2024, 1, 1, tzinfo=UTC)
MAINTAINER = {"login": "maintainer", "type": "User"}


def gh_issue(number, label, created="2025-06-01T00:00:00Z", *extra_labels):
    return {
        "number": number,
        "title": f"title {number}",
        "body": None if number == 2 else f"body {number}",
        "user": {"login": "issue-author-7"},
        "author_association": "NONE",
        "created_at": created,
        "labels": [{"name": n} for n in (label, *extra_labels)],
    }


def maintainer_labelled(label):
    return [{"event": "labeled", "label": {"name": label}, "actor": MAINTAINER}]


class FakeGitHub:
    """Issues per label (newest first) and each issue's events."""

    def __init__(self, issues_by_label, events=None):
        self.issues_by_label = issues_by_label
        self.events = events or {}
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/events"):
            number = int(path.split("/")[-2])
            return httpx.Response(200, json=self.events.get(number, []))
        if path == "/search/issues":
            return httpx.Response(200, json={"total_count": 42, "items": []})
        label = request.url.params["labels"]
        assert request.url.params["state"] == "closed"  # settled labels only
        return httpx.Response(200, json=self.issues_by_label.get(label, []))


def client(fake):
    return GitHubClient(None, transport=httpx.MockTransport(fake), sleep=lambda s: None)


def test_collects_answer_key_and_text_separately():
    fake = FakeGitHub(
        {"question": [gh_issue(1, "question"), gh_issue(2, "question")]},
        events={1: maintainer_labelled("question"), 2: maintainer_labelled("question")},
    )
    with client(fake) as gh:
        result = collect(gh, REPO, created_since=SINCE, per_type=5, labels=["question"])

    assert [r.number for r in result.rows] == [1, 2]
    assert result.rows[0].truth.type == "question"
    assert result.rows[0].truth.trust == "gold"
    assert [t.body for t in result.texts] == ["body 1", ""]  # null body -> ""
    # The manifest carries no text, and neither file carries the author.
    assert "title" not in result.rows[0].model_dump()
    assert "issue-author-7" not in result.model_dump_json()


def test_stops_at_per_type():
    issues = [gh_issue(n, "question") for n in range(1, 10)]
    events = {n: maintainer_labelled("question") for n in range(1, 10)}
    fake = FakeGitHub({"question": issues}, events)

    with client(fake) as gh:
        result = collect(gh, REPO, created_since=SINCE, per_type=3, labels=["question"])

    assert len(result.rows) == 3
    # Only the kept issues' events were fetched: no wasted requests.
    assert sum(r.url.path.endswith("/events") for r in fake.requests) == 3


def test_stops_paging_at_the_first_issue_older_than_since():
    issues = [
        gh_issue(3, "question", "2024-03-01T00:00:00Z"),
        gh_issue(2, "question", "2023-12-31T23:59:59Z"),  # too old: stop here
        gh_issue(1, "question", "2023-06-01T00:00:00Z"),
    ]
    fake = FakeGitHub({"question": issues}, {3: maintainer_labelled("question")})

    with client(fake) as gh:
        result = collect(gh, REPO, created_since=SINCE, per_type=5, labels=["question"])

    assert [r.number for r in result.rows] == [3]


def test_excluded_issues_are_counted_not_kept(caplog):
    fake = FakeGitHub(
        {
            "bug V2": [
                gh_issue(1, "bug V2", "2025-01-01T00:00:00Z", "pending"),
                gh_issue(2, "bug V2", "2025-01-01T00:00:00Z", "documentation"),
            ]
        }
    )
    with client(fake) as gh:
        result = collect(gh, REPO, created_since=SINCE, per_type=5, labels=["bug V2"])

    assert result.rows == []
    assert result.excluded == {
        "bug V2: unreviewed: still pending": 1,
        "bug V2: several of our labels: bug V2, documentation": 1,
    }
    assert "only 0 usable issues" in caplog.text  # a shortfall is reported


def test_reporter_form_is_kept_for_the_baseline():
    events = [
        {
            "event": "labeled",
            "label": {"name": "bug V2"},
            "actor": {"login": "issue-author-7"},
        },
        {"event": "unlabeled", "label": {"name": "bug V2"}, "actor": MAINTAINER},
        *maintainer_labelled("question"),
    ]
    fake = FakeGitHub({"question": [gh_issue(1, "question")]}, {1: events})

    with client(fake) as gh:
        result = collect(gh, REPO, created_since=SINCE, per_type=5, labels=["question"])

    truth = result.rows[0].truth
    assert (truth.type, truth.reporter_label, truth.relabelled) == (
        "question",
        "bug V2",
        True,
    )


def test_a_rerun_uses_cached_events(tmp_path):
    # The first real backfill died on a network drop and lost every request:
    # a re-run must not fetch the same histories again.
    issues = [gh_issue(n, "question") for n in (1, 2)]
    fake = FakeGitHub({"question": issues}, {1: maintainer_labelled("question")})
    args = {
        "created_since": SINCE,
        "per_type": 5,
        "labels": ["question"],
        "cache_dir": tmp_path,
    }

    with client(fake) as gh:
        first = collect(gh, REPO, **args)
    fetched_before = sum(r.url.path.endswith("/events") for r in fake.requests)
    with client(fake) as gh:
        second = collect(gh, REPO, **args)
    fetched_after = sum(r.url.path.endswith("/events") for r in fake.requests)

    assert fetched_before == 2
    assert fetched_after == 2  # nothing fetched again
    assert first == second  # same answer key from the cache
    assert not list(tmp_path.glob("*.tmp"))  # atomic writes leave no temp files


@pytest.mark.parametrize("label", ["bug V2", "documentation"])
def test_natural_frequencies_query(label):
    fake = FakeGitHub({})
    with client(fake) as gh:
        counts = natural_frequencies(gh, REPO, created_since=SINCE)

    assert counts == {"bug": 42, "feature": 42, "question": 42, "docs": 42}
    queries = [r.url.params["q"] for r in fake.requests]
    assert (
        f'repo:owner/repo is:issue is:closed label:"{label}" created:>=2024-01-01'
        in queries
    )
