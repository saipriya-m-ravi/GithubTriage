"""Adapter: GitHub's issue JSON -> our TriageState (the boundary where real data enters).

Picks the fields we use explicitly instead of passing GitHub's JSON straight
into our models: the shapes differ (author is user.login), and a missing field
gets a clear error naming it. Everything else in GitHub's JSON is ignored.
"""

import re

from githubtriage.models import Issue, TriageState
from githubtriage.repo_names import REPO_PATTERN, validate_repo

_REPO = REPO_PATTERN.pattern.strip("^$")
_ISSUE_REF = re.compile(rf"^(?P<repo>{_REPO})#(?P<number>\d+)$")
_GITHUB_URL = re.compile(
    rf"^https://github\.com/(?P<repo>{_REPO})/(?P<kind>issues|pull)/(?P<number>\d+)"
    r"(?:[/?#].*)?$"  # allow trailing parts, e.g. #issuecomment-123
)


class NotAnIssueError(ValueError):
    """The number refers to a pull request (GitHub's issues API returns both)."""


def parse_issue_ref(text: str) -> tuple[str, int] | None:
    """(repo, number) from "owner/repo#7" or a GitHub issue URL; None if neither.

    A pull-request URL raises NotAnIssueError straight away (no API call needed).
    The "owner/repo#7" form can't tell — issues and PRs share one number
    sequence — so the adapter checks the API response as well.
    """
    if match := _ISSUE_REF.fullmatch(text):
        return match["repo"], int(match["number"])
    if match := _GITHUB_URL.fullmatch(text):
        if match["kind"] == "pull":
            raise NotAnIssueError(f"{text} is a pull request, not an issue")
        return match["repo"], int(match["number"])
    return None


def state_from_github(raw: dict) -> TriageState:
    """Build the initial TriageState for one GitHub issue.

    Repo and number come from GitHub's response, not from what was requested:
    an issue transferred to another repository is returned from its new location.
    """
    number = _field(raw, "number")
    if "pull_request" in raw:
        raise NotAnIssueError(f"#{number} is a pull request, not an issue")

    repository_url = _field(raw, "repository_url")  # https://api.github.com/repos/o/r
    repo = validate_repo(repository_url.split("/repos/", 1)[-1])

    return TriageState(
        repo=repo,
        issue_number=number,
        issue=Issue(
            title=_field(raw, "title"),
            body=raw.get("body"),  # may be null; Issue turns None into ""
            author=_field(_field(raw, "user"), "login"),
            updated_at=_field(raw, "updated_at"),  # "...Z" -> timezone-aware UTC
            html_url=raw.get("html_url"),
        ),
    )


def _field(data: dict, name: str):
    try:
        return data[name]
    except (KeyError, TypeError):
        raise ValueError(f"GitHub issue JSON is missing {name!r}") from None
