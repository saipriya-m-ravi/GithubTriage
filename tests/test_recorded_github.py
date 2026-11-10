"""Tests against REAL GitHub responses, recorded and anonymised.

Hand-written fakes encode our assumptions about GitHub; when an assumption is
wrong, the code and its tests are wrong together (the stale-check bug: the fake
gave the bot and the maintainer different logins, real life gave them the same).
These fixtures are what GitHub actually sent. Re-record them with
debug_helper_scripts/record_github_fixtures.py; if these tests then fail, our
picture of GitHub's API is out of date.
"""

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from githubtriage.checkpointing import thread_config
from githubtriage.config import settings
from githubtriage.github_client import GitHubClient, NotFoundError
from githubtriage.github_issues import NotAnIssueError, state_from_github
from githubtriage.models import Decision
from githubtriage.nodes.post import post, stale_check

FIXTURES = Path(__file__).parent / "fixtures" / "github"
SANDBOX = "maintainer1/githubtriage-sandbox"  # anonymised owner


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class RecordedGitHub:
    """Replays recorded responses, matched by method + path.

    Emulates `since` on list endpoints the way GitHub does (updated_at >= since),
    so a test sees what GitHub would have returned for that request.
    """

    def __init__(self, *names: str):
        self.routes = {(fixture(n)["method"], fixture(n)["path"]): n for n in names}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        name = self.routes.get((request.method, request.url.path))
        if name is None:
            raise AssertionError(f"no fixture for {request.method} {request.url.path}")
        recorded = fixture(name)
        body = recorded["json"]
        if since := request.url.params.get("since"):
            body = [item for item in body if item["updated_at"] >= since]
        return httpx.Response(
            recorded["status"], json=body, headers=recorded["headers"]
        )


def client(fake: RecordedGitHub) -> GitHubClient:
    return GitHubClient(
        SecretStr("t"), transport=httpx.MockTransport(fake), sleep=lambda s: None
    )


def bot_reply_run() -> str:
    """The run id inside the recorded bot reply's marker."""
    for comment in fixture("comments")["json"]:
        if match := re.match(r"<!-- githubtriage:(\S+) -->", comment["body"]):
            return match[1]
    raise AssertionError("no bot reply in comments.json")


# --- Fixture hygiene: these files are committed to a public repository ---------


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_fixtures_are_anonymised(path):
    text = path.read_text(encoding="utf-8")
    logins = set(re.findall(r'"login": "([^"]+)"', text))
    assert all(re.fullmatch(r"maintainer1|user\d+", login) for login in logins)
    assert "avatar_url" not in text
    emails = set(re.findall(r"[\w.+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b", text))
    assert emails <= {"user@example.com"}


# --- The adapter on real JSON ------------------------------------------------------


def test_recorded_issue_becomes_a_triage_state():
    state = state_from_github(fixture("issue")["json"])

    assert state.repo == SANDBOX
    assert state.issue_number == 1
    assert state.issue.author == "maintainer1"
    assert state.issue.updated_at.tzinfo is not None  # "Z" parsed as UTC
    assert state.issue.html_url == f"https://github.com/{SANDBOX}/issues/1"
    assert state.issue.body  # a real, non-empty body


def test_recorded_pull_request_is_refused():
    with pytest.raises(NotAnIssueError, match="#14233 is a pull request"):
        state_from_github(fixture("pull_request")["json"])


# --- The client on real responses ---------------------------------------------------


def test_recorded_404_is_not_found():
    with client(RecordedGitHub("not_found")) as gh, pytest.raises(NotFoundError):
        gh.get_issue(SANDBOX, 999_999)


def test_list_issues_follows_githubs_real_next_link():
    fake = RecordedGitHub("issues_page1", "issues_page2")

    with client(fake) as gh:
        numbers = [issue["number"] for issue in gh.list_issues(SANDBOX)]

    assert numbers == [3, 2, 1]
    # GitHub's real next link: a different path and a cursor, not "page=2" on
    # the same URL. Building page URLs ourselves would have broken here.
    second = fake.requests[1].url
    assert second.path.startswith("/repositories/")
    assert "after" in second.params


# --- Stale check and posting on the real comment thread ----------------------------
# comments.json (sandbox #1), all by the token's account "maintainer1":
#   09:12:55 "System Info"   (hand-written, during the failed live test)
#   10:31:20 "my comment"    (hand-written, during the live re-test)
#   10:32:39 the bot's reply for run ...@20261106T103200.253531Z (the control run)


@pytest.fixture
def recorded_github_mode(monkeypatch):
    monkeypatch.setattr(settings, "post_mode", "github")
    monkeypatch.setattr(settings, "github_token", SecretStr("t"))
    monkeypatch.setattr(settings, "post_allowed_repos", [SANDBOX])
    fake = RecordedGitHub("user", "issue", "comments")
    monkeypatch.setattr(
        "githubtriage.nodes.post.GitHubClient",
        lambda token, url: client(fake),
    )
    return fake


def approved_state():
    state = state_from_github(fixture("issue")["json"])
    decision = Decision(action="approve", approver="m", decided_at=datetime.now(UTC))
    return state.model_copy(update={"draft": "Thanks!", "decision": decision})


def test_own_reply_is_not_a_change(recorded_github_mode):
    # The control run: only its own reply came after it started.
    assert stale_check(approved_state(), thread_config(bot_reply_run())) == {}


def test_hand_written_comments_by_the_token_account_are_changes(recorded_github_mode):
    # The live bug, on the real data: a run that started before the first
    # comment. Every later comment has the bot's login; none is THIS run's reply.
    # (The recorder guarantees hand-written comments by that account exist.)
    comments = fixture("comments")["json"]
    first = min(c["updated_at"] for c in comments)
    start = datetime.fromisoformat(first) - timedelta(minutes=1)
    earlier_run = f"{SANDBOX}#1@{start:%Y%m%dT%H%M%S.%fZ}"

    update = stale_check(approved_state(), thread_config(earlier_run))

    assert update["stale_reason"].startswith(
        f"{len(comments)} new comment(s) were added"
    )


def test_post_reuses_the_recorded_reply(recorded_github_mode):
    update = post(approved_state(), thread_config(bot_reply_run()))

    assert update == {"posted_comment_id": "github:6682791719"}
    assert {r.method for r in recorded_github_mode.requests} == {"GET"}
