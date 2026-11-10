"""CLI tests. Each main([...]) call is like a separate process: the only thing
shared between calls is the SQLite checkpoint file in the test's data dir —
exactly how `triage` and a later `approve` work for real.
"""

import json
import re

import httpx
import pytest

from githubtriage.cli import EXIT_BAD_INPUT, EXIT_OK, main
from githubtriage.config import settings
from githubtriage.models import Classification, Completeness, FieldCheck, TriageState
from githubtriage.nodes.draft import TYPE_MESSAGES

ISSUE = {
    "repo": "owner/repo",
    "issue_number": 7,
    "issue": {
        "title": "App crashes on login 🐛",  # non-ASCII on purpose: UTF-8 reading
        "body": "Clicking Sign in shows a blank page.",
        "author": "alice",
        "updated_at": "2026-10-01T10:00:00Z",
    },
}


@pytest.fixture
def issue_file(tmp_path):
    path = tmp_path / "issue.json"
    path.write_text(json.dumps(ISSUE, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def fake_bug(use_fake_llm):
    # A bug goes through classify AND completeness, so both need a fake answer.
    ok = FieldCheck(status="ok", note="")
    use_fake_llm(result=Completeness(version=ok, repro_steps=ok, logs=ok))
    return use_fake_llm(
        result=Classification(type="bug", confidence=0.9, reasoning="blank page")
    )


def run_id(text: str) -> str:
    return re.search(r"Run: (\S+)", text).group(1)


def outbox_files():
    return sorted(p.name for p in settings.outbox_dir.glob("*.md"))


# --- triage -----------------------------------------------------------------


def test_triage_prints_draft_and_waits(issue_file, fake_bug, capsys):
    code = main(["triage", str(issue_file)])

    out, err = capsys.readouterr()
    assert code == EXIT_OK
    assert "type:       bug (0.90)" in out
    assert TYPE_MESSAGES["bug"] in out
    assert run_id(out).startswith("owner/repo#7@")
    assert "githubtriage approve|edit|reject owner/repo#7" in out
    assert "Warning" not in err
    assert outbox_files() == []  # nothing posted before approval


def test_triage_json_keeps_stdout_pure_json(issue_file, fake_bug, capsys):
    code = main(["triage", str(issue_file), "--json"])

    out, err = capsys.readouterr()
    assert code == EXIT_OK
    result = TriageState.model_validate_json(out)
    assert result.issue.title == ISSUE["issue"]["title"]  # emoji survived
    assert "Run: owner/repo#7@" in err  # the run id goes to stderr in JSON mode


def test_degraded_triage_exits_0_with_warning(issue_file, use_fake_llm, capsys):
    use_fake_llm(error=RuntimeError("boom"))

    code = main(["triage", str(issue_file)])

    out, err = capsys.readouterr()
    assert code == EXIT_OK
    assert TYPE_MESSAGES["unclear"] in out
    assert "Warning: classify: RuntimeError" in err


def test_file_with_utf8_bom_is_accepted(tmp_path, fake_bug):
    # PowerShell 5.1 and Notepad often save UTF-8 with a BOM (﻿) at the start.
    path = tmp_path / "bom.json"
    path.write_text(json.dumps(ISSUE), encoding="utf-8-sig")

    assert main(["triage", str(path)]) == EXIT_OK


def test_missing_file_exits_1(tmp_path, capsys):
    code = main(["triage", str(tmp_path / "nope.json")])

    out, err = capsys.readouterr()
    assert code == EXIT_BAD_INPUT
    assert "no such file, and not an issue reference" in err
    assert out == ""


def test_invalid_file_exits_1(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"repo": "owner/repo"}', encoding="utf-8")  # missing fields

    code = main(["triage", str(path)])

    out, err = capsys.readouterr()
    assert code == EXIT_BAD_INPUT
    assert "invalid issue file" in err
    assert "issue_number" in err  # Pydantic names the missing field
    assert out == ""


# --- triage a live issue (GitHub answered by httpx.MockTransport) -----------


def github_issue_json(number=7, **overrides) -> dict:
    raw = {
        "number": number,
        "title": "App crashes on login",
        "body": "Clicking Sign in shows a blank page.",
        "user": {"login": "alice"},
        "updated_at": "2026-11-05T10:00:00Z",
        "repository_url": "https://api.github.com/repos/owner/repo",
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def fake_github(monkeypatch):
    """Point the CLI's GitHubClient at a fake transport; returns the request log."""
    from githubtriage import cli
    from githubtriage.github_client import GitHubClient

    responses, requests = [], []

    def handler(request):
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(
        cli,
        "GitHubClient",
        lambda token, url: GitHubClient(
            token, url, transport=httpx.MockTransport(handler), sleep=lambda s: None
        ),
    )
    return responses, requests


@pytest.mark.parametrize(
    "ref",
    ["owner/repo#7", "https://github.com/owner/repo/issues/7#issuecomment-1"],
    ids=["ref", "url"],
)
def test_triage_a_live_issue(ref, fake_github, fake_bug, capsys):
    responses, requests = fake_github
    responses.append(httpx.Response(200, json=github_issue_json()))

    code = main(["triage", ref])

    out, _ = capsys.readouterr()
    assert code == EXIT_OK
    assert requests[0].url.path == "/repos/owner/repo/issues/7"
    assert run_id(out).startswith("owner/repo#7@")
    assert TYPE_MESSAGES["bug"] in out


def test_pull_request_url_is_refused_without_calling_github(fake_github, capsys):
    _, requests = fake_github

    code = main(["triage", "https://github.com/owner/repo/pull/9"])

    assert code == EXIT_BAD_INPUT
    assert "is a pull request" in capsys.readouterr().err
    assert requests == []


def test_pull_request_number_is_refused_after_fetching(fake_github, capsys):
    responses, _ = fake_github
    responses.append(
        httpx.Response(200, json=github_issue_json(number=9, pull_request={}))
    )

    assert main(["triage", "owner/repo#9"]) == EXIT_BAD_INPUT
    assert "#9 is a pull request" in capsys.readouterr().err


def test_not_found_anonymously_suggests_a_token(fake_github, monkeypatch, capsys):
    monkeypatch.setattr(settings, "github_token", None)
    responses, _ = fake_github
    responses.append(httpx.Response(404, json={"message": "Not Found"}))

    assert main(["triage", "owner/private#7"]) == EXIT_BAD_INPUT
    err = capsys.readouterr().err
    assert "not found, or no access" in err
    assert "Set GITHUB_TOKEN if the repository is private" in err


def test_rate_limit_message_is_clear(fake_github, capsys):
    responses, _ = fake_github
    responses.append(
        httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "9999999999"},
        )
    )

    assert main(["triage", "owner/repo#7"]) == EXIT_BAD_INPUT
    assert "rate limit reached; retry in" in capsys.readouterr().err


# --- show / approve / edit / reject (separate "processes") -------------------


def test_show_displays_the_waiting_draft(issue_file, fake_bug, capsys):
    main(["triage", str(issue_file)])
    capsys.readouterr()

    code = main(["show", "owner/repo#7"])

    out, _ = capsys.readouterr()
    assert code == EXIT_OK
    assert "https://github.com/owner/repo/issues/7" in out
    assert TYPE_MESSAGES["bug"] in out
    assert ISSUE["issue"]["body"] not in out  # the body isn't re-displayed


def test_approve_posts_the_draft(issue_file, fake_bug, capsys):
    main(["triage", str(issue_file)])
    run = run_id(capsys.readouterr().out)

    code = main(["approve", "owner/repo#7", "--approver", "maintainer1"])

    out, _ = capsys.readouterr()
    assert code == EXIT_OK
    assert f"Posted: outbox:{run}" in out
    assert len(outbox_files()) == 1


def test_edit_posts_the_edited_text(issue_file, fake_bug, tmp_path, capsys):
    main(["triage", str(issue_file)])
    body = tmp_path / "reply.md"
    body.write_text("A reply written by a human.", encoding="utf-8")

    code = main(["edit", "owner/repo#7", "--body-file", str(body)])

    assert code == EXIT_OK
    (posted,) = settings.outbox_dir.glob("*.md")
    assert "A reply written by a human." in posted.read_text(encoding="utf-8")


def test_reject_posts_nothing(issue_file, fake_bug, capsys):
    main(["triage", str(issue_file)])
    capsys.readouterr()

    code = main(["reject", "owner/repo#7", "--reason", "duplicate of #3"])

    out, _ = capsys.readouterr()
    assert code == EXIT_OK
    assert "nothing posted" in out
    assert outbox_files() == []


def test_deciding_when_nothing_is_waiting_exits_1(capsys):
    code = main(["approve", "owner/repo#7"])

    _, err = capsys.readouterr()
    assert code == EXIT_BAD_INPUT
    assert "no draft waiting for owner/repo#7" in err


def test_edit_with_empty_file_is_refused(issue_file, fake_bug, tmp_path, capsys):
    main(["triage", str(issue_file)])
    body = tmp_path / "empty.md"
    body.write_text("   ", encoding="utf-8")

    code = main(["edit", "owner/repo#7", "--body-file", str(body)])

    _, err = capsys.readouterr()
    assert code == EXIT_BAD_INPUT
    assert "invalid decision" in err
    assert outbox_files() == []


# --- Re-triage policy ---------------------------------------------------------


def test_retriage_cancels_the_waiting_draft(issue_file, fake_bug, capsys):
    main(["triage", str(issue_file)])
    first = run_id(capsys.readouterr().out)

    main(["triage", str(issue_file)])
    out, err = capsys.readouterr()
    second = run_id(out)

    assert f"Cancelled previous draft: {first}" in err
    # The old draft can no longer be approved...
    assert main(["approve", first]) == EXIT_BAD_INPUT
    assert "no draft waiting for run" in capsys.readouterr().err
    # ...and approving the issue approves the new one.
    assert main(["approve", "owner/repo#7"]) == EXIT_OK
    assert f"Posted: outbox:{second}" in capsys.readouterr().out


def test_retriage_after_posting_can_post_again(issue_file, fake_bug, capsys):
    main(["triage", str(issue_file)])
    main(["approve", "owner/repo#7"])
    main(["triage", str(issue_file)])  # e.g. the reporter added more information
    main(["approve", "owner/repo#7"])

    # Two runs, two replies: the idempotency key is the run, not the issue.
    assert len(outbox_files()) == 2


# --- Posting fails after approval -> retry ------------------------------------


def test_failed_posting_can_be_retried(issue_file, fake_bug, monkeypatch, capsys):
    from githubtriage.github_client import GitHubError

    main(["triage", str(issue_file)])
    capsys.readouterr()

    def github_down(thread, text):
        raise GitHubError("GitHub unavailable (502)")

    with monkeypatch.context() as m:
        m.setattr("githubtriage.nodes.post._post_to_outbox", github_down)
        code = main(["approve", "owner/repo#7"])

    err = capsys.readouterr().err
    assert code == EXIT_BAD_INPUT
    assert "posting failed" in err
    assert "githubtriage retry owner/repo#7" in err  # says how to recover
    assert outbox_files() == []

    # Nothing is waiting for a decision any more — the decision is saved...
    assert main(["approve", "owner/repo#7"]) == EXIT_BAD_INPUT
    capsys.readouterr()
    # ...so `retry` resumes the stuck run and posts, without deciding again.
    assert main(["retry", "owner/repo#7"]) == EXIT_OK
    assert "Posted: outbox:" in capsys.readouterr().out
    assert len(outbox_files()) == 1


def test_retry_with_nothing_stuck_exits_1(capsys):
    assert main(["retry", "owner/repo#7"]) == EXIT_BAD_INPUT
    assert "no approved run waiting to be retried" in capsys.readouterr().err


def test_decision_commands_accept_an_issue_url(issue_file, fake_bug, capsys):
    # Same reference forms as `triage`: owner/repo#7 or the issue's URL.
    main(["triage", str(issue_file)])
    capsys.readouterr()

    assert main(["show", "https://github.com/owner/repo/issues/7"]) == EXIT_OK
    assert main(["approve", "https://github.com/owner/repo/issues/7"]) == EXIT_OK
    assert "Posted: outbox:" in capsys.readouterr().out


def test_decision_command_refuses_a_pull_request_url(capsys):
    assert main(["approve", "https://github.com/owner/repo/pull/9"]) == EXIT_BAD_INPUT
    assert "is a pull request" in capsys.readouterr().err
