"""GitHub client tests. httpx.MockTransport answers every request in-process:
no network, no token, no rate limits — the HTTP equivalent of the fake LLM."""

import logging

import httpx
import pytest
from pydantic import SecretStr

from githubtriage.github_client import (
    AuthError,
    GitHubClient,
    GitHubError,
    NotFoundError,
    RateLimitError,
)

TOKEN = SecretStr("ghp_test_token_123")


class FakeGitHub:
    """Serves queued responses in order and records every request."""

    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


def client(fake: FakeGitHub, token=TOKEN, sleeps=None) -> GitHubClient:
    sleeps = [] if sleeps is None else sleeps
    return GitHubClient(
        token,
        transport=httpx.MockTransport(fake),
        sleep=sleeps.append,  # record waits instead of really sleeping
    )


def ok(json, headers=None) -> httpx.Response:
    return httpx.Response(200, json=json, headers=headers or {})


# --- Requests ---------------------------------------------------------------


def test_sends_the_standard_headers_and_token():
    fake = FakeGitHub(ok({"number": 7}))
    with client(fake) as gh:
        gh.get_issue("owner/repo", 7)

    (req,) = fake.requests
    assert req.url.path == "/repos/owner/repo/issues/7"
    assert req.headers["Accept"] == "application/vnd.github+json"
    assert req.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert req.headers["Authorization"] == "Bearer ghp_test_token_123"


def test_no_authorization_header_without_a_token():
    fake = FakeGitHub(ok({"number": 7}))
    with client(fake, token=None) as gh:
        gh.get_issue("owner/repo", 7)

    assert "Authorization" not in fake.requests[0].headers


def test_create_comment_posts_the_body():
    fake = FakeGitHub(httpx.Response(201, json={"id": 99}))
    with client(fake) as gh:
        comment = gh.create_comment("owner/repo", 7, "Hello")

    (req,) = fake.requests
    assert req.method == "POST"
    assert req.url.path == "/repos/owner/repo/issues/7/comments"
    assert req.content == b'{"body":"Hello"}'
    assert comment == {"id": 99}


@pytest.mark.parametrize("repo", ["../../etc", "owner/repo/../../other", "owner"])
def test_invalid_repo_names_never_reach_the_url(repo):
    fake = FakeGitHub()
    with client(fake) as gh, pytest.raises(ValueError, match="invalid repository"):
        gh.get_issue(repo, 7)
    assert fake.requests == []


def test_follows_redirects_for_transferred_issues():
    fake = FakeGitHub(
        httpx.Response(
            301, headers={"location": "https://api.github.com/repos/o/new/issues/42"}
        ),
        ok({"number": 42}),
    )
    with client(fake) as gh:
        assert gh.get_issue("owner/repo", 7) == {"number": 42}
    assert fake.requests[1].url.path == "/repos/o/new/issues/42"


# --- Errors -------------------------------------------------------------------


def test_404_means_not_found_or_no_access():
    fake = FakeGitHub(httpx.Response(404, json={"message": "Not Found"}))
    with client(fake) as gh, pytest.raises(NotFoundError, match="no access"):
        gh.get_issue("owner/repo", 7)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_not_retried(status):
    fake = FakeGitHub(httpx.Response(status, json={"message": "nope"}))
    with client(fake) as gh, pytest.raises(AuthError):
        gh.get_issue("owner/repo", 7)
    assert len(fake.requests) == 1


def test_errors_never_contain_the_token():
    fake = FakeGitHub(httpx.Response(401, json={"message": "Bad credentials"}))
    with client(fake) as gh, pytest.raises(AuthError) as excinfo:
        gh.get_issue("owner/repo", 7)
    assert "ghp_test_token_123" not in str(excinfo.value)


# --- Retries and rate limits -------------------------------------------------


def test_short_rate_limit_is_waited_out_then_retried():
    sleeps = []
    fake = FakeGitHub(
        httpx.Response(429, headers={"retry-after": "3"}),
        ok({"number": 7}),
    )
    with client(fake, sleeps=sleeps) as gh:
        assert gh.get_issue("owner/repo", 7) == {"number": 7}

    assert sleeps == [3.0]
    assert len(fake.requests) == 2


def test_long_rate_limit_raises_instead_of_blocking():
    sleeps = []
    fake = FakeGitHub(
        httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "9999999999"},
        )
    )
    with client(fake, sleeps=sleeps) as gh, pytest.raises(RateLimitError) as excinfo:
        gh.get_issue("owner/repo", 7)

    assert excinfo.value.retry_after > 60
    assert sleeps == []  # didn't sleep for hours


def test_server_error_on_get_is_retried():
    sleeps = []
    fake = FakeGitHub(httpx.Response(502), ok({"number": 7}))
    with client(fake, sleeps=sleeps) as gh:
        assert gh.get_issue("owner/repo", 7) == {"number": 7}
    assert sleeps == [1]


def test_server_error_on_post_is_not_retried():
    # The comment may have been created before the error: retrying could
    # post it twice. Duplicates are prevented by the marker check instead.
    fake = FakeGitHub(httpx.Response(502), httpx.Response(201, json={"id": 1}))
    with client(fake) as gh, pytest.raises(GitHubError):
        gh.create_comment("owner/repo", 7, "Hello")
    assert len(fake.requests) == 1


def test_network_error_on_get_is_retried():
    calls = []

    def flaky(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection reset")
        return ok({"number": 7})

    with GitHubClient(
        TOKEN, transport=httpx.MockTransport(flaky), sleep=lambda s: None
    ) as gh:
        assert gh.get_issue("owner/repo", 7) == {"number": 7}
    assert len(calls) == 2


# --- Pagination ---------------------------------------------------------------


def page(items, next_url=None) -> httpx.Response:
    headers = {"link": f'<{next_url}>; rel="next"'} if next_url else {}
    return ok(items, headers)


def test_list_issues_follows_link_headers_and_skips_pull_requests():
    fake = FakeGitHub(
        page(
            [{"number": 3}, {"number": 2, "pull_request": {}}],
            next_url="https://api.github.com/repositories/1/issues?page=2",
        ),
        page([{"number": 1}]),
    )
    with client(fake) as gh:
        numbers = [i["number"] for i in gh.list_issues("owner/repo")]

    assert numbers == [3, 1]  # the PR (#2) is skipped
    assert fake.requests[0].url.params["per_page"] == "100"
    assert fake.requests[1].url.params["page"] == "2"  # the Link URL, as given


def test_pagination_stops_at_max_pages_and_says_so(caplog):
    fake = FakeGitHub(
        page([{"number": 2}], next_url="https://api.github.com/x?page=2"),
        page([{"number": 1}], next_url="https://api.github.com/x?page=3"),
    )
    with client(fake) as gh, caplog.at_level(logging.WARNING):
        numbers = [i["number"] for i in gh.list_issues("owner/repo", max_pages=2)]

    assert numbers == [2, 1]
    assert len(fake.requests) == 2  # page 3 never requested
    assert "Stopped after 2 pages" in caplog.text


def test_list_comments_since_and_complete():
    from githubtriage.github_client import TruncatedError

    fake = FakeGitHub(
        page([{"id": 1}], next_url="https://api.github.com/x?page=2"),
    )
    with client(fake) as gh, pytest.raises(TruncatedError):
        list(
            gh.list_comments(
                "owner/repo",
                7,
                since="2026-11-05T00:00:00Z",
                max_pages=1,
                complete=True,
            )
        )
    assert fake.requests[0].url.params["since"] == "2026-11-05T00:00:00Z"


def test_get_authenticated_login():
    fake = FakeGitHub(ok({"login": "triage-bot", "id": 1}))
    with client(fake) as gh:
        assert gh.get_authenticated_login() == "triage-bot"
    assert fake.requests[0].url.path == "/user"
