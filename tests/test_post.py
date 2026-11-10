from datetime import UTC, datetime

import pytest

from githubtriage.checkpointing import new_thread_id, thread_config
from githubtriage.config import settings
from githubtriage.models import Decision, Issue, TriageState
from githubtriage.nodes.draft import FOOTER
from githubtriage.nodes.post import (
    POSTED_FOOTER,
    PostNotAllowedError,
    marker,
    outbox_path,
    post,
    stale_check,
    with_disclosure,
)

NOW = datetime.now(UTC)
RUN = "owner/repo#7@20261104T101500.000000Z"
CONFIG = thread_config(RUN)


def make_state(decision=None, **overrides) -> TriageState:
    values = {
        "repo": "owner/repo",
        "issue_number": 7,
        "issue": Issue(title="t", body="b", author="a", updated_at=NOW),
        "draft": "Thanks for the report!",
        "decision": decision,
    }
    values.update(overrides)
    return TriageState(**values)


def decide(action, **fields) -> Decision:
    return Decision(action=action, approver="maintainer1", decided_at=NOW, **fields)


def outbox_files():
    return sorted(p.name for p in settings.outbox_dir.glob("*"))


def test_stale_check_skips_local_issue_files():
    # No html_url: the issue came from a file, there's nothing on GitHub to compare.
    assert stale_check(make_state(), CONFIG) == {}


def test_approve_posts_the_draft_with_a_marker():
    update = post(make_state(decide("approve")), CONFIG)

    assert update == {"posted_comment_id": f"outbox:{RUN}"}
    assert outbox_path(RUN).read_text(encoding="utf-8") == (
        f"{marker(RUN)}\nThanks for the report!\n\n{POSTED_FOOTER}\n"
    )


def test_outbox_file_name_is_per_run():
    assert outbox_path(RUN).name == "owner__repo#7@20261104T101500.000000Z.md"


def test_edit_posts_the_maintainers_text_not_the_draft():
    post(make_state(decide("edit", edited_body="Edited by a human.")), CONFIG)

    text = outbox_path(RUN).read_text(encoding="utf-8")
    assert "Edited by a human." in text
    assert "Thanks for the report!" not in text


def test_already_posted_is_skipped():
    update = post(make_state(decide("approve"), posted_comment_id="outbox:x"), CONFIG)

    assert update == {}
    assert outbox_files() == []  # nothing written


def test_retry_of_the_same_run_does_not_duplicate():
    # Crash after writing but before returning: the checkpoint never saw
    # posted_comment_id, so the retry runs post again with it still unset.
    state = make_state(decide("approve"))

    first = post(state, CONFIG)
    # Simulate a leftover temp file from a crash mid-write, too.
    outbox_path(RUN).with_suffix(".tmp").write_text("half", encoding="utf-8")
    second = post(state, CONFIG)

    assert first == second  # same deterministic id
    assert outbox_files() == [outbox_path(RUN).name]  # one file, no .tmp left


def test_a_new_run_of_the_same_issue_posts_a_new_reply():
    # The idempotency key is the run, not the issue: a re-triage may reply again.
    state = make_state(decide("approve"))
    later_run = new_thread_id("owner/repo", 7)

    post(state, CONFIG)
    post(state, thread_config(later_run))

    assert outbox_files() == sorted(
        [outbox_path(RUN).name, outbox_path(later_run).name]
    )


@pytest.mark.parametrize(
    "dec", [None, decide("reject", reason="duplicate")], ids=["none", "reject"]
)
def test_refuses_to_post_without_approval(dec):
    with pytest.raises(ValueError, match="without approval"):
        post(make_state(dec), CONFIG)
    assert outbox_files() == []


def test_refuses_to_post_an_empty_reply():
    with pytest.raises(ValueError, match="empty reply"):
        post(make_state(decide("approve"), draft="   "), CONFIG)


@pytest.mark.parametrize(
    "thread",
    [
        "../../etc#7@20261104T101500.000000Z",
        "owner/repo/extra#7@20261104T101500.000000Z",
        "owner/repo#7",  # no run part
        "owner/repo#7@../../x",
    ],
)
def test_rejects_run_ids_that_could_escape_the_outbox(thread):
    with pytest.raises(ValueError, match="invalid run id"):
        outbox_path(thread)


# --- Disclosure footer --------------------------------------------------------


def test_draft_footer_becomes_the_posted_disclosure():
    posted = with_disclosure(f"Thanks!\n\n{FOOTER}")
    assert FOOTER not in posted  # "pending review" is no longer true
    assert posted.endswith(POSTED_FOOTER)


def test_edited_reply_without_footer_gets_the_disclosure():
    assert with_disclosure("A human reply.\n") == f"A human reply.\n\n{POSTED_FOOTER}"


def test_disclosure_mentions_nobody():
    # --approver is an OS username until OAuth: "@name" could notify a stranger.
    assert "@" not in POSTED_FOOTER


# --- POST_MODE=github (GitHub answered by httpx.MockTransport) ---------------


BOT = "triage-bot"  # the account the (fake) token belongs to


@pytest.fixture
def github_mode(monkeypatch):
    """Switch post to GitHub mode against a fake GitHub; returns (responses, requests)."""
    import httpx
    from pydantic import SecretStr

    from githubtriage.github_client import GitHubClient

    monkeypatch.setattr(settings, "post_mode", "github")
    # monkeypatch bypasses Pydantic validation: set the type the app really has.
    monkeypatch.setattr(settings, "github_token", SecretStr("t"))
    monkeypatch.setattr(settings, "post_allowed_repos", ["owner/repo"])
    responses, requests = [], []

    def handler(request):
        if request.url.path == "/user":  # "who does this token belong to?"
            return httpx.Response(200, json={"login": BOT})
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(
        "githubtriage.nodes.post.GitHubClient",
        lambda token, url: GitHubClient(
            None, url, transport=httpx.MockTransport(handler), sleep=lambda s: None
        ),
    )
    return responses, requests


def test_github_mode_posts_a_comment_with_the_marker(github_mode):
    import httpx

    responses, requests = github_mode
    responses += [
        httpx.Response(200, json=[]),  # no comments since the run started
        httpx.Response(201, json={"id": 555}),
    ]

    update = post(make_state(decide("approve")), CONFIG)

    assert update == {"posted_comment_id": "github:555"}
    search, create = requests
    # Only comments since the run started (from the run id): small, one request.
    assert search.url.params["since"] == "2026-11-04T10:15:00Z"
    assert create.method == "POST"
    body = create.read().decode()
    assert marker(RUN) in body
    assert "Thanks for the report!" in body


def test_github_mode_reuses_an_existing_comment_after_a_crash(github_mode):
    import httpx

    responses, requests = github_mode
    responses.append(
        httpx.Response(
            200,
            json=[
                {"id": 1, "user": {"login": "alice"}, "body": "a reporter's comment"},
                {
                    "id": 777,
                    "user": {"login": BOT},
                    "body": f"{marker(RUN)}\nThanks for the report!",
                },
            ],
        )
    )

    update = post(make_state(decide("approve")), CONFIG)

    assert update == {"posted_comment_id": "github:777"}
    assert [r.method for r in requests] == ["GET"]  # nothing posted again


def test_github_mode_ignores_a_forged_marker_from_another_user(github_mode):
    # Anyone can comment on a public issue. A copied marker must not make us
    # skip posting and adopt their comment (a suppression attack).
    import httpx

    responses, requests = github_mode
    responses += [
        httpx.Response(
            200,
            json=[
                {
                    "id": 666,
                    "user": {"login": "attacker"},
                    "body": f"{marker(RUN)}\nnothing to see here",
                }
            ],
        ),
        httpx.Response(201, json={"id": 555}),
    ]

    update = post(make_state(decide("approve")), CONFIG)

    assert update == {"posted_comment_id": "github:555"}  # ours, not 666
    assert [r.method for r in requests] == ["GET", "POST"]  # we did post


def test_github_mode_refuses_repos_not_on_the_allowlist(github_mode, monkeypatch):
    monkeypatch.setattr(settings, "post_allowed_repos", ["me/sandbox"])
    _, requests = github_mode

    with pytest.raises(PostNotAllowedError, match="owner/repo is not allowed"):
        post(make_state(decide("approve")), CONFIG)
    assert requests == []  # refused before any API call


def test_github_mode_fails_closed_if_the_search_is_incomplete(github_mode):
    import httpx

    from githubtriage.github_client import TruncatedError

    responses, requests = github_mode
    for page in range(2, 12):  # 10 pages, each pointing to another
        responses.append(
            httpx.Response(
                200,
                json=[{"id": page, "body": "x"}],
                headers={"link": f'<https://api.github.com/x?page={page}>; rel="next"'},
            )
        )

    with pytest.raises(TruncatedError):
        post(make_state(decide("approve")), CONFIG)
    assert all(r.method == "GET" for r in requests)  # never posted


def test_outbox_mode_never_calls_github(github_mode, monkeypatch):
    monkeypatch.setattr(settings, "post_mode", "outbox")
    _, requests = github_mode

    post(make_state(decide("approve")), CONFIG)

    assert requests == []
    assert outbox_path(RUN).exists()


# --- Stale check against a fake GitHub ------------------------------------------


def github_state(**overrides) -> TriageState:
    """A state for an issue that came from GitHub (has html_url)."""
    state = make_state(decide("approve"), **overrides)
    issue = state.issue.model_copy(
        update={"html_url": "https://github.com/owner/repo/issues/7"}
    )
    return state.model_copy(update={"issue": issue})


def live_issue(title="t", body="b") -> dict:
    return {"number": 7, "title": title, "body": body}


def test_stale_check_passes_when_nothing_changed(github_mode):
    import httpx

    responses, requests = github_mode
    responses += [httpx.Response(200, json=live_issue()), httpx.Response(200, json=[])]

    assert stale_check(github_state(), CONFIG) == {}
    # Comments are searched only since the run started.
    assert requests[1].url.params["since"] == "2026-11-04T10:15:00Z"


def test_edited_issue_is_stale(github_mode):
    import httpx

    responses, _ = github_mode
    responses += [
        httpx.Response(200, json=live_issue(body="b — edit: version is 2.3.0")),
        httpx.Response(200, json=[]),
    ]

    update = stale_check(github_state(), CONFIG)
    assert update == {"stale_reason": "the issue was edited since the draft was made"}


def test_new_comment_from_someone_else_is_stale(github_mode):
    # E.g. the reporter added the missing version in a comment after triage.
    import httpx

    responses, _ = github_mode
    responses += [
        httpx.Response(200, json=live_issue()),
        httpx.Response(
            200, json=[{"id": 1, "user": {"login": "alice"}, "body": "v2.3"}]
        ),
    ]

    update = stale_check(github_state(), CONFIG)
    assert "1 new comment(s) were added" in update["stale_reason"]


def test_a_bot_reply_is_not_a_change(github_mode):
    import httpx

    responses, _ = github_mode
    reply = {"id": 2, "user": {"login": BOT}, "body": f"{marker(RUN)}\nThanks!"}
    responses += [
        httpx.Response(200, json=live_issue()),
        httpx.Response(200, json=[reply]),
    ]

    assert stale_check(github_state(), CONFIG) == {}


def test_a_hand_written_comment_from_the_token_account_is_a_change(github_mode):
    # Found in a live test: a personal token posts as its owner, so the
    # maintainer's own comment ("System Info: …") has the bot's login. No marker
    # means a human wrote it — it's new information, the draft is stale.
    import httpx

    responses, _ = github_mode
    responses += [
        httpx.Response(200, json=live_issue()),
        httpx.Response(
            200, json=[{"id": 3, "user": {"login": BOT}, "body": "System Info: …"}]
        ),
    ]

    update = stale_check(github_state(), CONFIG)
    assert "1 new comment(s) were added" in update["stale_reason"]


def test_a_reply_from_another_run_is_a_change(github_mode):
    # Only this run's own reply is ignored; another run's reply is new information.
    import httpx

    responses, _ = github_mode
    other_run = "owner/repo#7@20261104T120000.000000Z"
    reply = {"id": 5, "user": {"login": BOT}, "body": f"{marker(other_run)}\nHi"}
    responses += [
        httpx.Response(200, json=live_issue()),
        httpx.Response(200, json=[reply]),
    ]

    assert "stale_reason" in stale_check(github_state(), CONFIG)


def test_a_forged_marker_from_another_user_is_a_change(github_mode):
    import httpx

    responses, _ = github_mode
    forged = {"id": 4, "user": {"login": "attacker"}, "body": f"{marker(RUN)}\nhi"}
    responses += [
        httpx.Response(200, json=live_issue()),
        httpx.Response(200, json=[forged]),
    ]

    assert "stale_reason" in stale_check(github_state(), CONFIG)


def test_github_failure_during_stale_check_propagates(github_mode):
    # Never post unchecked: the run stays at stale_check, `retry` resumes it.
    import httpx

    from githubtriage.github_client import GitHubError

    responses, _ = github_mode
    responses += [httpx.Response(502)] * 3  # GET is retried twice, then gives up

    with pytest.raises(GitHubError):
        stale_check(github_state(), CONFIG)


def test_null_live_body_matches_empty_snapshot(github_mode):
    # GitHub sends body: null for an empty issue; our snapshot stores "".
    import httpx

    responses, _ = github_mode
    responses += [
        httpx.Response(200, json=live_issue(body=None)),
        httpx.Response(200, json=[]),
    ]
    state = github_state()
    state = state.model_copy(
        update={"issue": state.issue.model_copy(update={"body": ""})}
    )

    assert stale_check(state, CONFIG) == {}
