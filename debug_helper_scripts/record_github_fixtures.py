"""Record real GitHub API responses as anonymised test fixtures (read-only).

    uv run python debug_helper_scripts/record_github_fixtures.py

Needs GITHUB_TOKEN with read access to the sandbox repo. Only GET requests are
sent. Writes tests/fixtures/github/*.json, which tests replay through
httpx.MockTransport. Re-run it when GitHub's responses may have changed, then
re-run the tests: a failure means our assumptions about GitHub's API are out of date.

Anonymisation (the files are committed to a public repo):
- every GitHub login becomes a placeholder (the token's owner: "maintainer1",
  others "user1", "user2"...), everywhere it appears: user objects, URLs, bodies
- user objects are reduced to {login, type}: no avatars, ids, names or emails
- third-party issue bodies (outside the sandbox) are replaced
- any remaining e-mail address is replaced
The script then refuses to write anything if a real login or the token value is
still present (fail closed).

Scenarios: the recorder copies whatever the sandbox holds right now, so the
tests are only as meaningful as that data. Before writing, it checks that the
situations the tests exist for are still present (scenario_problems) and refuses
otherwise. Sandbox issue #1 is RECORDED EVIDENCE — don't comment on it or clean
it up; experiment on other issues.
"""

import json
import re
import sys
from pathlib import Path

import httpx

from githubtriage.checkpointing import thread_started_at
from githubtriage.config import settings
from githubtriage.github_client import GitHubClient, NotFoundError

SANDBOX = "saipriya-m-ravi/githubtriage-sandbox"
SANDBOX_ISSUE = 1  # has hand-written comments by the token's account + a bot reply
PULL_REQUEST = ("pydantic/pydantic", 14233)  # a PR, fetched via the issues endpoint
OUT_DIR = Path("tests/fixtures/github")
KEPT_HEADERS = ("link", "content-type")
THIRD_PARTY = {"pull_request"}  # fixtures whose text isn't ours: bodies removed
# The domain must end in a letters-only TLD: our run ids ("#1@20261106T103239.441Z")
# also look like "x@y.z", and a looser pattern rewrote the bot's marker.
EMAIL = re.compile(r"[\w.+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")
MARKER = re.compile(r"<!-- githubtriage:(\S+) -->")
OWNER = "maintainer1"  # the token owner's placeholder after anonymisation


class RecordingTransport(httpx.BaseTransport):
    """Passes requests to the real network and keeps a copy of each exchange."""

    def __init__(self) -> None:
        self._inner = httpx.HTTPTransport()
        self.exchanges: list[tuple[httpx.Request, httpx.Response]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET":
            raise RuntimeError(f"recorder is read-only, refused {request.method}")
        response = self._inner.handle_request(request)
        response.read()
        self.exchanges.append((request, response))
        return response

    def close(self) -> None:
        self._inner.close()


def record(transport: RecordingTransport) -> dict[str, httpx.Response]:
    """Run the real client against GitHub; return {fixture name: exchange}."""
    names = []
    with GitHubClient(
        settings.github_token, settings.github_api_url, transport=transport
    ) as gh:
        gh.get_authenticated_login()
        names.append("user")
        gh.get_issue(SANDBOX, SANDBOX_ISSUE)
        names.append("issue")
        list(gh.list_comments(SANDBOX, SANDBOX_ISSUE))
        names.append("comments")
        gh.get_issue(*PULL_REQUEST)
        names.append("pull_request")
        try:
            gh.get_issue(SANDBOX, 999_999)
        except NotFoundError:
            pass
        names.append("not_found")
        # per_page=2 on a 3-issue repo: two pages, with a real Link header.
        list(
            gh._paginate(f"/repos/{SANDBOX}/issues", {"state": "all", "per_page": 2}, 2)
        )
        names += ["issues_page1", "issues_page2"]
    if len(names) != len(transport.exchanges):
        raise RuntimeError(
            f"expected {len(names)} requests, got {len(transport.exchanges)}"
        )
    return dict(zip(names, transport.exchanges, strict=True))


def collect_logins(value, found: list[str]) -> None:
    """Every login in user objects, in order of first appearance."""
    if isinstance(value, dict):
        if isinstance(value.get("login"), str) and value["login"] not in found:
            found.append(value["login"])
        for item in value.values():
            collect_logins(item, found)
    elif isinstance(value, list):
        for item in value:
            collect_logins(item, found)


def anonymise(value, replace, *, third_party: bool):
    if isinstance(value, dict):
        if "login" in value:  # a user object: keep only what our code reads
            return {"login": replace(value["login"]), "type": value.get("type")}
        out = {}
        for key, item in value.items():
            if key == "body" and third_party and isinstance(item, str):
                out[key] = "(third-party issue text removed)"
            else:
                out[key] = anonymise(item, replace, third_party=third_party)
        return out
    if isinstance(value, list):
        return [anonymise(item, replace, third_party=third_party) for item in value]
    if isinstance(value, str):
        return EMAIL.sub("user@example.com", replace(value))
    return value


def scenario_problems(fixtures: dict[str, dict]) -> list[str]:
    """What tests/test_recorded_github.py relies on, checked on the anonymised data.

    Without this, a re-record after someone cleaned up sandbox #1 would quietly
    replace the PAT-bug scenario (bot and maintainer are the same account) with
    data where it can't happen — the tests would still pass, testing nothing.
    """
    problems = []
    comments = fixtures["comments"]["json"]
    by_owner = [c for c in comments if c["user"]["login"] == OWNER]
    replies = [c for c in by_owner if MARKER.match(c["body"] or "")]
    hand_written = [c for c in by_owner if not MARKER.match(c["body"] or "")]
    if not replies:
        problems.append("comments: no bot reply (token account + marker)")
    if not hand_written:
        problems.append("comments: no hand-written comment by the token account")
    if replies:
        # The newest reply's run must have nothing after its start but the reply,
        # or "our own reply is not a change" has no clean case to test.
        reply = replies[-1]
        started = thread_started_at(MARKER.match(reply["body"])[1])
        since = started.strftime("%Y-%m-%dT%H:%M:%SZ")
        if [c for c in comments if c["updated_at"] >= since and c is not reply]:
            problems.append(
                "comments: other comments after the bot reply's run started"
            )
    if "pull_request" not in fixtures["pull_request"]["json"]:
        problems.append("pull_request: not a pull request")
    if fixtures["not_found"]["status"] != 404:
        problems.append("not_found: status is not 404")
    if 'rel="next"' not in fixtures["issues_page1"]["headers"].get("link", ""):
        problems.append('issues_page1: no rel="next" link (sandbox needs ≥ 3 issues)')
    return problems


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    token = settings.github_token.get_secret_value() if settings.github_token else ""
    if not token:
        print("GITHUB_TOKEN is required (sandbox access).", file=sys.stderr)
        return 1

    transport = RecordingTransport()
    exchanges = record(transport)
    transport.close()

    owner = SANDBOX.split("/")[0]
    logins = [owner]
    for _, response in exchanges.values():
        collect_logins(response.json(), logins)
    placeholders = {
        login.lower(): ("maintainer1" if i == 0 else f"user{i}")
        for i, login in enumerate(logins)
    }
    pattern = re.compile(
        r"(?<![A-Za-z0-9-])("
        + "|".join(re.escape(login) for login in sorted(logins, key=len, reverse=True))
        + r")(?![A-Za-z0-9-])",
        re.IGNORECASE,
    )

    def replace(text: str) -> str:
        return pattern.sub(lambda m: placeholders[m[1].lower()], text)

    fixtures = {}
    for name, (request, response) in exchanges.items():
        # By name, not path: GitHub's "next" links use /repositories/<id>/...,
        # so the sandbox's second page doesn't start with /repos/<owner>/<name>.
        third_party = name in THIRD_PARTY
        fixtures[name] = {
            "method": request.method,
            "path": replace(request.url.path),
            "query": replace(request.url.query.decode()),
            "status": response.status_code,
            "headers": {
                h: replace(response.headers[h])
                for h in KEPT_HEADERS
                if h in response.headers
            },
            "json": anonymise(response.json(), replace, third_party=third_party),
        }
    files = {
        name: json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
        for name, fixture in fixtures.items()
    }

    # Fail closed: nothing is written if a scenario the tests need is gone...
    if problems := scenario_problems(fixtures):
        raise RuntimeError(
            "sandbox data no longer covers the tested scenarios; nothing written:\n- "
            + "\n- ".join(problems)
        )
    # ...or if anything identifying slipped through.
    for name, text in files.items():
        if token in text:
            raise RuntimeError(f"{name}: token value present")
        if leaked := [login for login in logins if login.lower() in text.lower()]:
            raise RuntimeError(f"{name}: real login(s) still present: {len(leaked)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT_DIR / f"{name}.json").write_text(text, encoding="utf-8")
        print(f"wrote {OUT_DIR / name}.json")
    print(f"{len(logins)} login(s) anonymised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
