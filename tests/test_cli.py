"""CLI tests. Each main([...]) call is like a separate process: the only thing
shared between calls is the SQLite checkpoint file in the test's data dir —
exactly how `triage` and a later `approve` work for real.
"""

import json
import re

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
    assert "file not found" in err
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
