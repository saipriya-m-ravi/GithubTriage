"""A small GitHub REST API client (httpx): only the endpoints GithubTriage uses.

Error handling, rate limits and retries live here, so the rest of the code
deals with Python objects and clear exceptions, not HTTP details.

Retry policy:
- Rate limited (429, or 403 with retry-after / remaining=0): the request was
  REJECTED, not processed, so retrying is safe for every method — if the wait
  is short. Long waits raise RateLimitError instead of blocking for an hour.
- Server errors (5xx) and network errors: retried for GET only. A POST may have
  been processed before the error, so retrying could post a comment twice;
  idempotency for posting is handled with the marker check (slice 4, module 5).
- Everything else (401, 403, 404, 422...): not retried — the same request would
  get the same answer.
"""

import logging
import time
from collections.abc import Callable, Iterator
from typing import Self

import httpx
from pydantic import SecretStr

from githubtriage.repo_names import validate_repo

logger = logging.getLogger(__name__)

API_VERSION = "2022-11-28"
MAX_RATE_LIMIT_WAIT = 60  # seconds; longer waits raise RateLimitError
DEFAULT_MAX_PAGES = 10  # x 100 per page = 1,000 items, unless asked for more


class GitHubError(Exception):
    """A GitHub API request failed."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class NotFoundError(GitHubError):
    """404 — doesn't exist, OR it's private and the token can't see it
    (GitHub doesn't say which, so as not to reveal that private repos exist)."""


class AuthError(GitHubError):
    """401 (bad or missing token) or 403 (token lacks permission)."""


class RateLimitError(GitHubError):
    """Rate limited, and the wait is too long to sleep through."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message, status=429)
        self.retry_after = retry_after


class TruncatedError(GitHubError):
    """Pagination hit max_pages, and the caller needs the COMPLETE list (e.g. a
    duplicate check: an incomplete search could miss our comment)."""


class GitHubClient:
    """Use as a context manager so the connection pool is always closed:

    with GitHubClient(settings.github_token, settings.github_api_url) as gh:
        issue = gh.get_issue("owner/repo", 7)
    """

    def __init__(
        self,
        token: SecretStr | None,
        base_url: str = "https://api.github.com",
        *,
        transport: httpx.BaseTransport | None = None,  # tests: httpx.MockTransport
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,  # tests: record instead of wait
        timeout: float = 30.0,
    ):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "githubtriage",  # GitHub rejects requests without one
        }
        if token is not None and token.get_secret_value():
            headers["Authorization"] = f"Bearer {token.get_secret_value()}"
        self._http = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
            # GitHub answers 301 for issues transferred to another repo; follow it.
            follow_redirects=True,
        )
        self._max_retries = max_retries
        self._sleep = sleep

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self._http.close()

    # --- Endpoints ------------------------------------------------------------

    def get_authenticated_login(self) -> str:
        """The GitHub username the token belongs to (who our comments are by)."""
        return self._request("GET", "/user").json()["login"]

    def get_issue(self, repo: str, number: int) -> dict:
        """One issue (or pull request) as GitHub's JSON."""
        return self._request(
            "GET", f"/repos/{validate_repo(repo)}/issues/{number}"
        ).json()

    def list_issues(
        self,
        repo: str,
        *,
        state: str = "all",
        labels: str | None = None,
        since: str | None = None,
        max_pages: int = DEFAULT_MAX_PAGES,
    ) -> Iterator[dict]:
        """Issues of a repo, newest first, EXCLUDING pull requests.

        GitHub's issues endpoint also returns pull requests (every PR is an issue
        in its data model); they carry a "pull_request" key and are skipped here.
        """
        params = {
            "state": state,
            "per_page": 100,
            "sort": "created",
            "direction": "desc",
        }
        if labels:
            params["labels"] = labels
        if since:
            params["since"] = since
        path = f"/repos/{validate_repo(repo)}/issues"
        for item in self._paginate(path, params, max_pages):
            if "pull_request" not in item:
                yield item

    def list_comments(
        self,
        repo: str,
        number: int,
        *,
        since: str | None = None,
        max_pages: int = DEFAULT_MAX_PAGES,
        complete: bool = False,
    ) -> Iterator[dict]:
        """Comments on one issue, oldest first.

        since: only comments updated at or after this ISO 8601 time.
        complete: raise TruncatedError instead of stopping quietly at max_pages.
        """
        path = f"/repos/{validate_repo(repo)}/issues/{number}/comments"
        params = {"per_page": 100}
        if since:
            params["since"] = since
        yield from self._paginate(path, params, max_pages, complete=complete)

    def create_comment(self, repo: str, number: int, body: str) -> dict:
        """Post a comment. NOT retried on server/network errors (see module docstring)."""
        path = f"/repos/{validate_repo(repo)}/issues/{number}/comments"
        return self._request("POST", path, json={"body": body}).json()

    # --- Internals ------------------------------------------------------------

    def _paginate(
        self, path: str, params: dict, max_pages: int, *, complete: bool = False
    ) -> Iterator[dict]:
        """Follow the Link header's rel="next" — never build page URLs ourselves."""
        url: str | None = path
        for page in range(1, max_pages + 1):
            response = self._request("GET", url, params=params if page == 1 else None)
            yield from response.json()
            url = response.links.get("next", {}).get("url")
            if url is None:
                return
        # Stopped by the limit, not by running out of pages: never hide it.
        if complete:
            raise TruncatedError(f"{path}: more than {max_pages} pages")
        logger.warning("Stopped after %s pages of %s (more available)", max_pages, path)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            last_attempt = attempt == self._max_retries
            try:
                response = self._http.request(method, url, **kwargs)
            except httpx.TransportError as e:  # connection reset, timeout, DNS...
                if method == "GET" and not last_attempt:
                    self._backoff(attempt, f"network error ({type(e).__name__})")
                    continue
                raise GitHubError(f"{method} {url}: network error: {e}") from e

            if response.status_code < 400:
                return response

            wait = _rate_limit_wait(response)
            if wait is not None:
                if wait > MAX_RATE_LIMIT_WAIT or last_attempt:
                    raise RateLimitError(
                        f"GitHub rate limit reached; retry in {wait:.0f}s", wait
                    )
                logger.warning("Rate limited; waiting %.0fs before retrying", wait)
                self._sleep(wait)
                continue

            if response.status_code >= 500 and method == "GET" and not last_attempt:
                self._backoff(attempt, f"server error {response.status_code}")
                continue

            raise _error_for(method, url, response)
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _backoff(self, attempt: int, reason: str) -> None:
        delay = 2**attempt  # 1s, 2s, 4s...
        logger.warning("GitHub %s; retrying in %ss", reason, delay)
        self._sleep(delay)


def _rate_limit_wait(response: httpx.Response) -> float | None:
    """Seconds to wait if this response is a rate limit, else None."""
    if response.status_code not in (403, 429):
        return None
    if "retry-after" in response.headers:  # secondary limits say exactly how long
        return float(response.headers["retry-after"])
    if response.headers.get("x-ratelimit-remaining") == "0":  # primary limit used up
        reset = float(response.headers.get("x-ratelimit-reset", time.time() + 60))
        return max(reset - time.time(), 1.0)
    if response.status_code == 429:
        return 60.0
    return None  # a 403 that isn't a rate limit is a permission problem


def _error_for(method: str, url: str, response: httpx.Response) -> GitHubError:
    """A clear exception for a failed response. Never includes request headers
    (they contain the token)."""
    status = response.status_code
    try:
        detail = response.json().get("message", "")
    except ValueError:
        detail = response.text[:200]
    where = f"{method} {url} -> {status}"
    if status == 404:
        return NotFoundError(
            f"{where}: not found, or no access with this token", status
        )
    if status == 401:
        return AuthError(
            f"{where}: authentication failed (missing or invalid token)", status
        )
    if status == 403:
        return AuthError(
            f"{where}: forbidden — the token lacks permission ({detail})", status
        )
    return GitHubError(f"{where}: {detail}", status)
