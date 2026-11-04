from datetime import UTC, datetime

import pytest

from githubtriage.checkpointing import new_thread_id, thread_config
from githubtriage.config import settings
from githubtriage.models import Decision, Issue, TriageState
from githubtriage.nodes.post import marker, outbox_path, post, stale_check

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


def test_stale_check_is_a_pass_through_stub_until_slice_4():
    assert stale_check(make_state()) == {}


def test_approve_posts_the_draft_with_a_marker():
    update = post(make_state(decide("approve")), CONFIG)

    assert update == {"posted_comment_id": f"outbox:{RUN}"}
    assert outbox_path(RUN).read_text(encoding="utf-8") == (
        f"{marker(RUN)}\nThanks for the report!\n"
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
