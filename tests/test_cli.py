import json

import pytest

from githubtriage.cli import EXIT_BAD_INPUT, EXIT_OK, main
from githubtriage.models import Classification, TriageState
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
    return use_fake_llm(
        result=Classification(type="bug", confidence=0.9, reasoning="blank page")
    )


def test_valid_file_prints_classification_and_draft(issue_file, fake_bug, capsys):
    code = main([str(issue_file)])

    out, err = capsys.readouterr()
    assert code == EXIT_OK
    assert "type:       bug (0.90)" in out
    assert TYPE_MESSAGES["bug"] in out
    assert "Warning" not in err


def test_json_flag_prints_valid_state(issue_file, fake_bug, capsys):
    code = main([str(issue_file), "--json"])

    out, _ = capsys.readouterr()
    assert code == EXIT_OK
    result = TriageState.model_validate_json(out)
    assert result.classification.type == "bug"
    assert result.issue.title == ISSUE["issue"]["title"]  # emoji survived


def test_degraded_run_exits_0_with_warning(issue_file, use_fake_llm, capsys):
    use_fake_llm(error=RuntimeError("boom"))

    code = main([str(issue_file)])

    out, err = capsys.readouterr()
    assert code == EXIT_OK
    assert TYPE_MESSAGES["unclear"] in out
    assert "Warning: classify: RuntimeError" in err


def test_file_with_utf8_bom_is_accepted(tmp_path, fake_bug, capsys):
    # PowerShell 5.1 and Notepad often save UTF-8 with a BOM (﻿) at the start.
    path = tmp_path / "bom.json"
    path.write_text(json.dumps(ISSUE), encoding="utf-8-sig")

    code = main([str(path)])

    assert code == EXIT_OK


def test_missing_file_exits_1(tmp_path, capsys):
    code = main([str(tmp_path / "nope.json")])

    out, err = capsys.readouterr()
    assert code == EXIT_BAD_INPUT
    assert "file not found" in err
    assert out == ""


def test_invalid_file_exits_1(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"repo": "owner/repo"}', encoding="utf-8")  # missing fields

    code = main([str(path)])

    out, err = capsys.readouterr()
    assert code == EXIT_BAD_INPUT
    assert "invalid issue file" in err
    assert "issue_number" in err  # Pydantic names the missing field
    assert out == ""
