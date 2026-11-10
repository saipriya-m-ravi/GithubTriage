from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from githubtriage.github_issues import (
    NotAnIssueError,
    parse_issue_ref,
    state_from_github,
)
from githubtriage.models import Issue


def github_issue(**overrides) -> dict:
    """A trimmed version of what GET /repos/{o}/{r}/issues/{n} returns."""
    raw = {
        "number": 7,
        "title": "parse_date returns the wrong day",
        "body": "Steps to reproduce...",
        "user": {"login": "reporter1", "id": 1},
        "updated_at": "2026-11-05T10:00:00Z",
        "repository_url": "https://api.github.com/repos/owner/repo",
        "html_url": "https://github.com/owner/repo/issues/7",
        "labels": [{"name": "bug"}],
        "state": "open",
        "comments": 3,
    }
    raw.update(overrides)
    return raw


def test_maps_the_fields_we_use_and_ignores_the_rest():
    state = state_from_github(github_issue())

    assert state.repo == "owner/repo"
    assert state.issue_number == 7
    assert state.issue.title == "parse_date returns the wrong day"
    assert state.issue.author == "reporter1"
    assert state.issue.updated_at == datetime(2026, 11, 5, 10, 0, tzinfo=UTC)
    assert state.classification is None  # nothing triaged yet


def test_null_body_becomes_empty_string():
    assert state_from_github(github_issue(body=None)).issue.body == ""


def test_pull_requests_are_rejected():
    raw = github_issue(pull_request={"url": "https://api.github.com/..."})
    with pytest.raises(NotAnIssueError, match="#7 is a pull request"):
        state_from_github(raw)


def test_transferred_issue_uses_its_new_location():
    # Asked for owner/repo#7; GitHub redirected to where the issue lives now.
    raw = github_issue(
        number=42, repository_url="https://api.github.com/repos/owner/new-home"
    )
    state = state_from_github(raw)
    assert (state.repo, state.issue_number) == ("owner/new-home", 42)


@pytest.mark.parametrize("missing", ["title", "user", "updated_at", "repository_url"])
def test_missing_field_gives_a_clear_error(missing):
    raw = github_issue()
    del raw[missing]
    with pytest.raises(ValueError, match=f"missing {missing!r}"):
        state_from_github(raw)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("pydantic/pydantic#12580", ("pydantic/pydantic", 12580)),
        (
            "https://github.com/pydantic/pydantic/issues/12580",
            ("pydantic/pydantic", 12580),
        ),
        (
            "https://github.com/pydantic/pydantic/issues/12580#issuecomment-99",
            ("pydantic/pydantic", 12580),
        ),
        ("samples/issues/bug.json", None),  # not a reference: treated as a file
        ("https://evil.example/owner/repo/issues/7", None),  # only github.com
        ("../../etc#7", None),
    ],
)
def test_parse_issue_ref(text, expected):
    assert parse_issue_ref(text) == expected


def test_pull_request_url_is_rejected_by_the_parser():
    with pytest.raises(NotAnIssueError, match="is a pull request"):
        parse_issue_ref("https://github.com/pydantic/pydantic/pull/14233")


def test_issue_rejects_timezone_naive_updated_at():
    # Without a timezone, comparing with GitHub's UTC time later would raise.
    with pytest.raises(ValidationError):
        Issue(title="t", body="b", author="a", updated_at="2026-11-05T10:00:00")
